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
        self.t_loss_tolerance = 1e-6
        
        # 상대 tolerance (이미 쓰고 있으니 기본값만 세팅)
        self.t_loss_rel_tolerance = 0.0
        
        # ✅ H selection weight: candidates 안에서 H를 고를 때 H-loss에 곱해짐
        #    (1.0이면 기존과 동일, >1이면 H를 더 빡세게 맞추려고 함)
        self.h_select_mult = 2.0
        
        # ✅ logging frequency (너무 자주 찍히면 지저분하니)
        self.selection_log_every = 200  # step 기준

        # =========================
        # Semi-auto tolerance control (recommended)
        # =========================
        # Base tolerances (human-chosen)
        self.t_tol_abs_early = 1e-6
        self.t_tol_rel_early = 0.0

        self.t_tol_abs_late  = 5e-6
        self.t_tol_rel_late  = 0.003

        # Start in early phase
        self._tol_phase = "early"
        self.t_loss_tolerance = self.t_tol_abs_early
        self.t_loss_rel_tolerance = self.t_tol_rel_early

        # Event-driven "switch" condition (no hard-coded step)
        # If candidates_mean stays collapsed near 1 for long enough -> switch to late phase
        self.tol_collapse_threshold = 1.2   # candidates_mean < 1.2 => essentially collapsed
        self.tol_collapse_patience = 10     # require this many checks in a row
        self._collapse_counter = 0

        # Mild feedback (slow adaptation) to keep candidates_mean in a target band
        # This is intentionally gentle to avoid making the objective non-stationary.
        self.tol_target_low  = 1.8   # want at least ~2 candidates on avg
        self.tol_target_high = 3.0   # but not too many

        self.tol_update_every = 200  # steps; align with logging to avoid spam & jitter
        self.tol_adjust_rate  = 0.10 # +/-10% per update max (gentle)

        # Safety bounds
        self.tol_abs_min = 5e-7
        self.tol_abs_max = 2e-5
        self.tol_rel_min = 0.0
        self.tol_rel_max = 0.02

        # --- temperature for t-loss softening ---
        self.t_loss_temp = 1.0        # early: ~1.0
        self.t_loss_temp_min = 0.05   # late: 거의 hard argmin
        self.t_loss_temp_decay = 0.995  # step마다 곱해짐
        # t-loss temperature (sharpness control)
        self.t_loss_temp_early = 1.0
        self.t_loss_temp_late  = 2.0   # ← 첫 시도 추천값
        self.t_loss_temp = self.t_loss_temp_early


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

    def _maybe_update_tolerances(self, candidates: Tensor):
        """
        Semi-auto tolerance control.
        - Switch early -> late when candidates_mean collapses near 1 for long enough
        - Mildly adapt abs/rel tol to keep candidates_mean in a target band
        """
        with torch.no_grad():
            step = getattr(self, "global_step", 0)
            if step % getattr(self, "tol_update_every", 200) != 0:
                return

            cand_count = candidates.sum(0).float()  # (B,)
            cand_mean = cand_count.mean().item()

            # --- 1) Event-driven phase switch (no hard-coded step) ---
            if self._tol_phase == "early":
                if cand_mean < getattr(self, "tol_collapse_threshold", 1.2):
                    self._collapse_counter += 1
                else:
                    self._collapse_counter = 0

                if self._collapse_counter >= getattr(self, "tol_collapse_patience", 10):
                    self._tol_phase = "late"
                    self.t_loss_tolerance = self.t_tol_abs_late
                    self.t_loss_rel_tolerance = self.t_tol_rel_late
                    self.t_loss_temp = self.t_loss_temp_late
                    self._collapse_counter = 0

            # --- 2) Mild feedback within phase (gentle) ---
            # Goal: keep candidates_mean in [target_low, target_high]
            target_low  = getattr(self, "tol_target_low", 1.8)
            target_high = getattr(self, "tol_target_high", 3.0)
            rate        = getattr(self, "tol_adjust_rate", 0.10)

            abs_tol = float(getattr(self, "t_loss_tolerance", 1e-6))
            rel_tol = float(getattr(self, "t_loss_rel_tolerance", 0.0))

            if cand_mean < target_low:
                # too few candidates -> relax tolerance a bit
                abs_tol *= (1.0 + rate)
                rel_tol *= (1.0 + rate) if rel_tol > 0 else rel_tol
                # also allow rel_tol to "turn on" gently in late phase
                if rel_tol == 0.0 and self._tol_phase == "late":
                    rel_tol = max(rel_tol, self.t_tol_rel_late)

            elif cand_mean > target_high:
                # too many candidates -> tighten tolerance a bit
                abs_tol *= (1.0 - rate)
                rel_tol *= (1.0 - rate)

            # clamp
            abs_tol = min(max(abs_tol, self.tol_abs_min), self.tol_abs_max)
            rel_tol = min(max(rel_tol, self.tol_rel_min), self.tol_rel_max)

            self.t_loss_tolerance = abs_tol
            self.t_loss_rel_tolerance = rel_tol

            # log what controller is doing (low frequency)
            self.log("select/tol_abs", torch.tensor(abs_tol, device=candidates.device), sync_dist=True)
            self.log("select/tol_rel", torch.tensor(rel_tol, device=candidates.device), sync_dist=True)
            self.log("select/tol_phase", torch.tensor(0 if self._tol_phase == "early" else 1,
                                                      device=candidates.device), sync_dist=True)
            self.log("select/candidates_mean_ctrl", torch.tensor(cand_mean, device=candidates.device), sync_dist=True)
            self.log("select/t_loss_temp",
                     torch.tensor(self.t_loss_temp, device=candidates.device),
                     sync_dist=True)

    
    def combine_symmetric_losses(self, symmetric_losses: Tensor) -> Tuple[Tensor, Tensor]:
        """
        symmetric_losses: (P, N, 2, B)
        P: num_permutations
        N: num_particles(branches)
        2: (assignment, detection)
        B: batch size
        """

        # --------------------------------------------------------------------------------
        # Fallback: 전체 loss 기준 argmin (안전망)
        # --------------------------------------------------------------------------------
        total_loss = symmetric_losses.sum((1, 2))          # (P, B)
        fallback_index = total_loss.argmin(0)              # (B,)

        # t / H 인덱스 없으면 기존 방식 유지
        if (len(self.t_particle_indices) == 0) or (len(self.h_particle_indices) == 0):
            index = fallback_index
            combined_loss = torch.gather(
                symmetric_losses, 0, index.expand_as(symmetric_losses)
            )[0]
            return combined_loss, index

        device = symmetric_losses.device
        t_idx = torch.tensor(self.t_particle_indices, device=device, dtype=torch.long)
        h_idx = torch.tensor(self.h_particle_indices, device=device, dtype=torch.long)

        # --------------------------------------------------------------------------------
        # 1) t-loss 계산 (t branches만)
        # --------------------------------------------------------------------------------
        #t_loss = symmetric_losses.index_select(1, t_idx).sum((1, 2))  # (P, B)
        t_loss_raw = symmetric_losses.index_select(1, t_idx).sum((1, 2))
        t_loss = t_loss_raw / getattr(self, "t_loss_temp", 1.0)
        min_t  = t_loss.min(0).values                                  # (B,)

        # 🔥 temperature scaling (core)
        temp = getattr(self, "t_loss_temp", 1.0)
        t_loss_eff = (t_loss - min_t) / temp
        
        abs_tol = getattr(self, "t_loss_tolerance", 1e-6)
        rel_tol = getattr(self, "t_loss_rel_tolerance", 0.0)
        
        # threshold: absolute + relative tolerance 혼합
        thresh = torch.minimum(
            min_t + abs_tol,
            min_t * (1.0 + rel_tol) + abs_tol
        )

        candidates = t_loss_eff <= ((thresh - min_t) / temp)
        # ✅ semi-auto tolerance control
        self._maybe_update_tolerances(candidates)
        
        # --------------------------------------------------------------------------------
        # Debug logging: candidates 통계
        # --------------------------------------------------------------------------------
        with torch.no_grad():
            candidates_count = candidates.sum(0).to(torch.float32)    # (B,)
            
            step = getattr(self, "global_step", 0)
            log_every = getattr(self, "selection_log_every", 200)
            
            self.t_loss_temp = max(
                self.t_loss_temp * self.t_loss_temp_decay,
                self.t_loss_temp_min
            )

            if step % log_every == 0:
                self.log("select/candidates_mean", candidates_count.mean(), sync_dist=True)
                self.log("select/candidates_min",  candidates_count.min(),  sync_dist=True)
                self.log("select/candidates_max",  candidates_count.max(),  sync_dist=True)

                no_candidate_frac = (candidates_count == 0).to(torch.float32).mean()
                self.log("select/no_candidate_frac", no_candidate_frac, sync_dist=True)
                
                # 얼마나 타이트한 t 조건인지
                margin = (thresh - min_t).to(torch.float32)
                self.log("select/t_margin_mean", margin.mean(), sync_dist=True)
                self.log("select/t_margin_min",  margin.min(),  sync_dist=True)
                self.log("select/t_margin_max",  margin.max(),  sync_dist=True)

                self.log("select/t_loss_temp",
                         torch.tensor(self.t_loss_temp, device=device),
                         sync_dist=True)
                

        # --------------------------------------------------------------------------------
        # 2) candidates 안에서 H-loss 최소 선택
        #    (✅ H selection weight 적용)
        # --------------------------------------------------------------------------------
        h_loss = symmetric_losses.index_select(1, h_idx).sum((1, 2))  # (P, B)

        h_select_mult = getattr(self, "h_select_mult", 1.0)
        if h_select_mult != 1.0:
            h_loss = h_loss * h_select_mult

        inf = torch.tensor(float("inf"), device=device, dtype=h_loss.dtype)
        h_loss_masked = h_loss.masked_fill(~candidates, inf)

        index = h_loss_masked.argmin(0)                               # (B,)

        # --------------------------------------------------------------------------------
        # fallback: candidates가 전부 비는 경우
        # --------------------------------------------------------------------------------
        all_invalid = torch.isinf(h_loss_masked).all(0)
        index = torch.where(all_invalid, fallback_index, index)
        
        with torch.no_grad():
            step = getattr(self, "global_step", 0)
            log_every = getattr(self, "selection_log_every", 200)
            if step % log_every == 0:
                fallback_frac = all_invalid.to(torch.float32).mean()
                self.log("select/fallback_frac", fallback_frac, sync_dist=True)

        # --------------------------------------------------------------------------------
        # 선택된 permutation으로 loss gather
        # --------------------------------------------------------------------------------
        combined_loss = torch.gather(
            symmetric_losses, 0, index.expand_as(symmetric_losses)
        )[0]

        return combined_loss, index


    def combine_symmetric_losses_v1(self, symmetric_losses: Tensor) -> Tuple[Tensor, Tensor]:
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
        abs_tol = getattr(self, "t_loss_tolerance", 5e-6)
        rel_tol = getattr(self, "t_loss_rel_tolerance", 0.003)          # 필요하면 추가
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
