from typing import Tuple, Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from spanet.options import Options
from spanet.dataset.types import Batch, Source, AssignmentTargets
from spanet.dataset.regressions import regression_loss
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
from spanet.network.utilities.divergence_losses import assignment_cross_entropy_loss, jensen_shannon_divergence


def numpy_tensor_array(tensor_list):
    output = np.empty(len(tensor_list), dtype=object)
    output[:] = tensor_list

    return output


class JetReconstructionTraining(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetReconstructionTraining, self).__init__(options, torch_script)

        self.log_clip = torch.log(10 * torch.scalar_tensor(torch.finfo(torch.float32).eps)).item()

        self.event_particle_names = list(self.training_dataset.event_info.product_particles.keys())
        self.product_particle_names = {
            particle: self.training_dataset.event_info.product_particles[particle][0]
            for particle in self.event_particle_names
        }

        # --- Permutation selection groups (t 먼저 만족, 그 안에서 H 최소) ---
        # event_particle_names 는 training_dataset.event_info.product_particles.keys() 기반
        # 이름 규칙이 다르면 여기 매칭 룰만 바꾸면 됨
        def _is_t(name: str) -> bool:
            n = name.lower()
            # 예: "t", "t_bar", "top", "top_bar" 등 케이스 커버
            return (n == "t1") or ("t2" in n) or ("top" in n)

        def _is_h(name: str) -> bool:
            n = name.lower()
            # 예: "h", "higgs" 등
            return (n == "h") or ("higgs" in n)

        self.t_particle_indices = [i for i, n in enumerate(self.event_particle_names) if _is_t(n)]
        self.h_particle_indices = [i for i, n in enumerate(self.event_particle_names) if _is_h(n)]

        # tolerance: t-loss가 최솟값 대비 이만큼 이내면 "충분히 만족"으로 간주
        # (원하면 options로 빼도 됨)
        self.t_loss_tolerance = 1e-6

        

    def particle_symmetric_loss(self, assignment: Tensor, detection: Tensor, target: Tensor, mask: Tensor, weight: Tensor) -> Tensor:
        assignment_loss = assignment_cross_entropy_loss(assignment, target, mask, weight, self.options.focal_gamma)
        detection_loss = F.binary_cross_entropy_with_logits(detection, mask.float(), weight=weight, reduction='none')

        return torch.stack((
            self.options.assignment_loss_scale * assignment_loss,
            self.options.detection_loss_scale * detection_loss
        ))

    def compute_symmetric_losses(self, assignments: List[Tensor], detections: List[Tensor], targets):
        symmetric_losses = []

        # TODO think of a way to avoid this memory transfer but keep permutation indices synced with checkpoint
        # Compute a separate loss term for every possible target permutation.
        for permutation in self.event_permutation_tensor.cpu().numpy():

            # Find the assignment loss for each particle in this permutation.
            current_permutation_loss = tuple(
                self.particle_symmetric_loss(assignment, detection, target, mask, weight)
                for assignment, detection, (target, mask, weight)
                in zip(assignments, detections, targets[permutation])
            )

            # The loss for a single permutation is the sum of particle losses.
            symmetric_losses.append(torch.stack(current_permutation_loss))

        # Shape: (NUM_PERMUTATIONS, NUM_PARTICLES, 2, BATCH_SIZE)
        return torch.stack(symmetric_losses)

    def combine_symmetric_losses(self, symmetric_losses: Tensor) -> Tuple[Tensor, Tensor]:
        # symmetric_losses: (P, N, 2, B)
        # P: num_permutations, N: num_particles(branches), 2: (assign, detect), B: batch
        total_loss = symmetric_losses.sum((1, 2))          # (P, B)
        fallback_index = total_loss.argmin(0)              # (B,)
        
        # 인덱스 없으면 기존 방식
        if (len(self.t_particle_indices) == 0) or (len(self.h_particle_indices) == 0):
            index = fallback_index
            combined_loss = torch.gather(symmetric_losses, 0, index.expand_as(symmetric_losses))[0]
            return combined_loss, index

        device = symmetric_losses.device
        t_idx = torch.tensor(self.t_particle_indices, device=device, dtype=torch.long)
        h_idx = torch.tensor(self.h_particle_indices, device=device, dtype=torch.long)

        # 1) t-loss 계산 (t branches만 합)
        t_loss = symmetric_losses.index_select(1, t_idx).sum((1, 2))  # (P, B)
        min_t = t_loss.min(0).values                                  # (B,)

        # 절대/상대 tolerance 같이 사용(권장)
        abs_tol = getattr(self, "t_loss_tolerance", 1e-6)
        rel_tol = getattr(self, "t_loss_rel_tolerance", 0.0)          # 필요하면 추가
        thresh = torch.minimum(min_t + abs_tol, min_t * (1.0 + rel_tol) + abs_tol)

        candidates = t_loss <= thresh                                 # (P, B)

        # 2) candidates 안에서 H-loss 최소
        h_loss = symmetric_losses.index_select(1, h_idx).sum((1, 2))  # (P, B)
        
        inf = torch.tensor(float("inf"), device=device, dtype=h_loss.dtype)
        h_loss_masked = h_loss.masked_fill(~candidates, inf)
        
        index = h_loss_masked.argmin(0)                               # (B,)
        
        # 후보가 하나도 없으면 fallback
        all_invalid = torch.isinf(h_loss_masked).all(0)
        index = torch.where(all_invalid, fallback_index, index)
        
        combined_loss = torch.gather(symmetric_losses, 0, index.expand_as(symmetric_losses))[0]
        return combined_loss, index
    
    
    def combine_symmetric_losses_v0(self, symmetric_losses: Tensor) -> Tuple[Tensor, Tensor]:
        # symmetric_losses: (P, N, 2, B)
        total_symmetric_loss = symmetric_losses.sum((1, 2))  # (P, B)  전체 기준(기존 방식)

        # 기본 fallback: 전체 loss 최소 permutation
        fallback_index = total_symmetric_loss.argmin(0)  # (B,)

        # t/H 인덱스가 없으면 기존 로직 유지
        if (len(self.t_particle_indices) == 0) or (len(self.h_particle_indices) == 0):
            index = fallback_index
            combined_loss = torch.gather(symmetric_losses, 0, index.expand_as(symmetric_losses))[0]
            # mean/softmin 옵션은 그대로 유지
            if self.options.combine_pair_loss.lower() == "mean":
                combined_loss = symmetric_losses.mean(0)
            if self.options.combine_pair_loss.lower() == "softmin":
                weights = F.softmin(total_symmetric_loss, 0).unsqueeze(1).unsqueeze(1)
                combined_loss = (weights * symmetric_losses).sum(0)
            return combined_loss, index

        #print("DOYEONG: alternative combined loss will be used")
        
        device = symmetric_losses.device
        t_idx = torch.tensor(self.t_particle_indices, device=device, dtype=torch.long)
        h_idx = torch.tensor(self.h_particle_indices, device=device, dtype=torch.long)

        # 1) t-loss가 충분히 작은 permutation subset 선택
        t_loss = symmetric_losses.index_select(1, t_idx).sum((1, 2))  # (P, B)
        min_t = t_loss.min(0).values  # (B,)
        candidates = t_loss <= (min_t + self.t_loss_tolerance)  # (P, B) bool

        # 2) subset 안에서 H-loss 최소 permutation 선택
        h_loss = symmetric_losses.index_select(1, h_idx).sum((1, 2))  # (P, B)

        inf = torch.tensor(float("inf"), device=device, dtype=h_loss.dtype)
        h_loss_masked = h_loss.masked_fill(~candidates, inf)  # 후보가 아니면 inf

        index = h_loss_masked.argmin(0)  # (B,)

        # subset이 완전히 비는(전부 inf) 배치가 있으면 fallback 사용
        all_invalid = torch.isinf(h_loss_masked).all(0)  # (B,)
        index = torch.where(all_invalid, fallback_index, index)

        combined_loss = torch.gather(symmetric_losses, 0, index.expand_as(symmetric_losses))[0]

        # 기존 옵션(mean/softmin)은 "선택 방식"과 충돌하므로
        # 이 2-stage 모드에서는 보통 argmin 고정이 자연스러움.
        # 그래도 유지하고 싶으면 아래처럼 분기 가능.
        if self.options.combine_pair_loss.lower() == "mean":
            combined_loss = symmetric_losses.mean(0)

        if self.options.combine_pair_loss.lower() == "softmin":
            weights = F.softmin(total_symmetric_loss, 0).unsqueeze(1).unsqueeze(1)
            combined_loss = (weights * symmetric_losses).sum(0)

        return combined_loss, index


    def symmetric_losses(
        self,
        assignments: List[Tensor],
        detections: List[Tensor],
        targets: Tuple[Tuple[Tensor, Tensor, Tensor], ...]
    ) -> Tuple[Tensor, Tensor]:
        # We are only going to look at a single prediction points on the distribution for more stable loss calculation
        # We multiply the softmax values by the size of the permutation group to make every target the same
        # regardless of the number of sub-jets in each target particle
        assignments = [prediction + torch.log(torch.scalar_tensor(decoder.num_targets))
                       for prediction, decoder in zip(assignments, self.branch_decoders)]

        # Convert the targets into a numpy array of tensors so we can use fancy indexing from numpy
        targets = numpy_tensor_array(targets)

        # Compute the loss on every valid permutation of the targets
        symmetric_losses = self.compute_symmetric_losses(assignments, detections, targets)

        # Squash the permutation losses into a single value.
        return self.combine_symmetric_losses(symmetric_losses)

    def symmetric_divergence_loss(self, predictions: List[Tensor], masks: Tensor) -> Tensor:
        divergence_loss = []

        for i, j in self.event_info.event_transpositions:
            # Symmetric divergence between these two distributions
            div = jensen_shannon_divergence(predictions[i], predictions[j])

            # ERF term for loss
            loss = torch.exp(-(div ** 2))
            loss = loss.masked_fill(~masks[i], 0.0)
            loss = loss.masked_fill(~masks[j], 0.0)

            divergence_loss.append(loss)

        return torch.stack(divergence_loss).mean(0)
        # return -1 * torch.stack(divergence_loss).sum(0) / len(self.training_dataset.unordered_event_transpositions)

    def add_kl_loss(
            self,
            total_loss: List[Tensor],
            assignments: List[Tensor],
            masks: Tensor,
            weights: Tensor
    ) -> List[Tensor]:
        if len(self.event_info.event_transpositions) == 0:
            return total_loss

        # Compute the symmetric loss between all valid pairs of distributions.
        kl_loss = self.symmetric_divergence_loss(assignments, masks)
        kl_loss = (weights * kl_loss).sum() / masks.sum()

        with torch.no_grad():
            self.log("loss/symmetric_loss", kl_loss, sync_dist=True)
            if torch.isnan(kl_loss):
                raise ValueError("Symmetric KL Loss has diverged.")

        return total_loss + [self.options.kl_loss_scale * kl_loss]

    def add_regression_loss(
            self,
            total_loss: List[Tensor],
            predictions: Dict[str, Tensor],
            targets:  Dict[str, Tensor]
    ) -> List[Tensor]:
        regression_terms = []

        for key in targets:
            current_target_type = self.training_dataset.regression_types[key]
            current_prediction = predictions[key]
            current_target = targets[key]

            current_mean = self.regression_decoder.networks[key].mean
            current_std = self.regression_decoder.networks[key].std

            current_mask = ~torch.isnan(current_target)

            current_loss = regression_loss(current_target_type)(
                current_prediction[current_mask],
                current_target[current_mask],
                current_mean,
                current_std
            )
            current_loss = torch.mean(current_loss)

            with torch.no_grad():
                self.log(f"loss/regression/{key}", current_loss, sync_dist=True)

            regression_terms.append(self.options.regression_loss_scale * current_loss)

        return total_loss + regression_terms

    def add_classification_loss(
            self,
            total_loss: List[Tensor],
            predictions: Dict[str, Tensor],
            targets: Dict[str, Tensor]
    ) -> List[Tensor]:
        classification_terms = []

        for key in targets:
            current_prediction = predictions[key]
            current_target = targets[key]

            weight = None if not self.balance_classifications else self.classification_weights[key]
            current_loss = F.cross_entropy(
                current_prediction,
                current_target,
                ignore_index=-1,
                weight=weight
            )

            classification_terms.append(self.options.classification_loss_scale * current_loss)

            with torch.no_grad():
                self.log(f"loss/classification/{key}", current_loss, sync_dist=True)

        return total_loss + classification_terms

    def training_step(self, batch: Batch, batch_nb: int) -> Dict[str, Tensor]:
        # ===================================================================================================
        # Network Forward Pass
        # ---------------------------------------------------------------------------------------------------
        outputs = self.forward(batch.sources)

        # ===================================================================================================
        # Initial log-likelihood loss for classification task
        # ---------------------------------------------------------------------------------------------------
        symmetric_losses, best_indices = self.symmetric_losses(
            outputs.assignments,
            outputs.detections,
            batch.assignment_targets,
        )

        # Construct the newly permuted masks based on the minimal permutation found during NLL loss.
        permutations = self.event_permutation_tensor[best_indices].T
        masks = torch.stack([target.mask for target in batch.assignment_targets])
        masks = torch.gather(masks, 0, permutations)

        # ===================================================================================================
        # Balance the loss based on the distribution of various classes in the dataset.
        # ---------------------------------------------------------------------------------------------------

        # Default unity weight on correct device.
        weights = torch.ones_like(symmetric_losses)

        # Balance based on the particles present - only used in partial event training
        if self.balance_particles:
            class_indices = (masks * self.particle_index_tensor.unsqueeze(1)).sum(0)
            weights *= self.particle_weights_tensor[class_indices]

        # Balance based on the number of jets in this event
        if self.balance_jets:
            weights *= self.jet_weights_tensor[batch.num_vectors]

        # Take the weighted average of the symmetric loss terms.
        masks = masks.unsqueeze(1)
        symmetric_losses = (weights * symmetric_losses).sum(-1) / torch.clamp(masks.sum(-1), 1, None)
        assignment_loss, detection_loss = torch.unbind(symmetric_losses, 1)


        # ===================================================================================================
        # DY: Add weight only to Higgs losses
        # ---------------------------------------------------------------------------------------------------        
        branch_names = list(self.training_dataset.assignments.keys())
        
        def _find_h_index(names):
            for i, n in enumerate(names):
                nl = n.lower()
                if nl == "h" or "higgs" in nl:
                    return i
            return None

        h_idx = _find_h_index(branch_names)

        H_ASSIGN_MULT = 2.0
        H_DETECT_MULT = 1.0
        
        if h_idx is not None:
            mult = torch.ones_like(assignment_loss)
            mult[h_idx] = H_ASSIGN_MULT
            assignment_loss = assignment_loss * mult
            
            mult_det = torch.ones_like(detection_loss)
            mult_det[h_idx] = H_DETECT_MULT
            detection_loss = detection_loss * mult_det
            
        
        # ===================================================================================================
        # Some basic logging
        # ---------------------------------------------------------------------------------------------------
        with torch.no_grad():
            for name, l in zip(self.training_dataset.assignments, assignment_loss):
                self.log(f"loss/{name}/assignment_loss", l, sync_dist=True)

            for name, l in zip(self.training_dataset.assignments, detection_loss):
                self.log(f"loss/{name}/detection_loss", l, sync_dist=True)

            if torch.isnan(assignment_loss).any():
                raise ValueError("Assignment loss has diverged!")

            if torch.isinf(assignment_loss).any():
                raise ValueError("Assignment targets contain a collision.")

        # ===================================================================================================
        # Start constructing the list of all computed loss terms.
        # ---------------------------------------------------------------------------------------------------
        total_loss = []

        if self.options.assignment_loss_scale > 0:
            total_loss.append(assignment_loss)

        if self.options.detection_loss_scale > 0:
            total_loss.append(detection_loss)

        # ===================================================================================================
        # Auxiliary loss terms which are added to reconstruction loss for alternative targets.
        # ---------------------------------------------------------------------------------------------------
        if self.options.kl_loss_scale > 0:
            total_loss = self.add_kl_loss(total_loss, outputs.assignments, masks, weights)

        if self.options.regression_loss_scale > 0:
            total_loss = self.add_regression_loss(total_loss, outputs.regressions, batch.regression_targets)

        if self.options.classification_loss_scale > 0:
            total_loss = self.add_classification_loss(total_loss, outputs.classifications, batch.classification_targets)

        # ===================================================================================================
        # Combine and return the loss
        # ---------------------------------------------------------------------------------------------------
        total_loss = torch.cat([loss.view(-1) for loss in total_loss])

        self.log("loss/total_loss", total_loss.sum(), sync_dist=True)

        return total_loss.mean()
