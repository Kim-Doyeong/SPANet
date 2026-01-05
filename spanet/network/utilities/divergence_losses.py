import torch
from torch import Tensor
from torch.nn import functional as F


@torch.jit.script
def assignment_cross_entropy_loss(prediction: Tensor, target_data: Tensor, target_mask: Tensor, weight: Tensor, gamma: float) -> Tensor:
    batch_size = prediction.shape[0]
    prediction_shape = prediction.shape[1:]

    # Remove missing jets
    target_data = target_data.clamp(0, None)

    # Find the unravelling shape required to flatten the target indices
    ravel_sizes = torch.tensor(prediction_shape).flip(0)
    ravel_sizes = torch.cumprod(ravel_sizes, 0)
    ravel_sizes = torch.div(ravel_sizes, ravel_sizes[0], rounding_mode='floor')
    # ravel_sizes = ravel_sizes // ravel_sizes[0]
    ravel_sizes = ravel_sizes.flip(0).unsqueeze(0)
    ravel_sizes = ravel_sizes.to(target_data.device)

    # Flatten the target and predicted data to be one dimensional
    ravel_target = (target_data * ravel_sizes).sum(1)
    ravel_prediction = prediction.reshape(batch_size, -1).contiguous()

    # ================= DEBUG: masked target sanity check =================
    if not torch.jit.is_scripting():
        with torch.no_grad():
            active = target_mask > 0
            if active.any():
                tgt = ravel_target[active]
                picked = ravel_prediction[active, tgt]
                finite = torch.isfinite(picked)
                
                if finite.sum() == 0:
                    print("\n🚨 assignment_cross_entropy_loss: no finite logits for masked targets")
                    print("  prediction shape:", prediction.shape)
                    print("  ravel_prediction shape:", ravel_prediction.shape)
                    print("  target_mask sum:", target_mask.sum().item())
                    print("  ravel_target min/max:",
                          ravel_target.min().item(),
                          ravel_target.max().item())
                #raise RuntimeError("Masked targets point only to non-finite logits")
    # ====================================================================

    
    #log_probability = ravel_prediction.gather(-1, ravel_target.view(-1, 1)).squeeze()
    #log_probability = log_probability.masked_fill(~target_mask, 0.0)
    log_probability = ravel_prediction.gather(-1, ravel_target.view(-1, 1)).squeeze()

    # ✅ 1. 비정상 값 제거
    finite = torch.isfinite(log_probability)
    
    # ✅ 2. target_mask를 finite와 결합
    target_mask = target_mask & finite
    
    # ✅ 3. 안전하게 0으로 대체
    log_probability = log_probability.masked_fill(~target_mask, 0.0)


    focal_scale = (1 - torch.exp(log_probability)) ** gamma

    return -log_probability * focal_scale * weight


@torch.jit.script
def kl_divergence_old(p: Tensor, log_p: Tensor, log_q: Tensor) -> Tensor:
    sum_dim = [i for i in range(1, p.ndim)]
    return torch.sum(p * log_p - p * log_q, sum_dim)


@torch.jit.script
def kl_divergence(log_prediction: Tensor, log_target: Tensor) -> Tensor:
    sum_dim = [i for i in range(1, log_prediction.ndim)]
    return torch.nansum(F.kl_div(log_prediction, log_target, reduction='none', log_target=True), dim=sum_dim)


@torch.jit.script
def jensen_shannon_divergence(log_p: Tensor, log_q: Tensor) -> Tensor:
    sum_dim = [i for i in range(1, log_p.ndim)]

    # log_m = log( (exp(log_p) + exp(log_q)) / 2 )
    log_m = torch.logsumexp(torch.stack((log_p, log_q)), dim=0) - 0.69314718056

    # TODO play around with gradient
    # log_m = log_m.detach()
    log_p = log_p.detach()
    log_q = log_q.detach()

    kl_p = F.kl_div(log_m, log_p, reduction='none', log_target=True)
    kl_q = F.kl_div(log_m, log_q, reduction='none', log_target=True)

    return torch.nansum(kl_p + kl_q, dim=sum_dim) / 2.0
