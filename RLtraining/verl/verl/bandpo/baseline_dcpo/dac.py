# 文件: verl/verl/bandpo/baseline_dcpo/dac.py

# -*- coding: utf-8 -*-
"""
DCPO baseline: Dynamic Adaptive Clipping (DAC)

公式来自 DCPO:
    L(x) = 0.5 + 0.5 * sqrt(max(1 - 4 * eps_low  / q(x), 0))
    U(x) = 0.5 + 0.5 * sqrt(    1 + 4 * eps_high / q(x))

其中:
    q(x) = old policy 对 sampled token 的概率 = exp(old_log_prob)

实现目标:
- 仅返回 token-wise ratio bounds
- 不混入 BandPO 的 heuristic
- 单独支持 DCPO 的 r_max 上界截断
"""

from __future__ import annotations

import time
import torch

_LAST_DCPO_DAC_TIME_PROFILE = {
    "core_ms": 0.0,
    "numel": 0,
}

def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)

def get_last_dcpo_dac_time_profile() -> dict:
    return dict(_LAST_DCPO_DAC_TIME_PROFILE)

@torch.no_grad()
def compute_dcpo_dac_ratio_bounds(
    old_log_prob: torch.Tensor,
    *,
    eps_clip_low: float = 0.16,
    eps_clip_high: float = 0.20,
    ratio_max: float = 10.0,
    q_min: float = 1e-12,
    MIN_LOGP: float = -700.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    根据 old_log_prob 计算 DCPO-DAC 的 token-wise ratio 下/上界。

    Args:
        old_log_prob:
            old policy 对 sampled token 的 log prob, shape = [B, T]
        eps_clip_low:
            DAC lower hyperparameter, DCPO 默认建议 0.16
        eps_clip_high:
            DAC upper hyperparameter, DCPO 默认建议 0.20
        ratio_max:
            DCPO 中对动态上界施加的 hard ceiling, 默认 10.0
        q_min:
            数值稳定项，避免 q 太小导致除零或上界爆炸
        MIN_LOGP:
            对 old_log_prob 的下界截断，避免 exp 下溢太严重

    Returns:
        bound_low, bound_high:
            与 old_log_prob 同 shape 的 token-wise bounds
    """
    global _LAST_DCPO_DAC_TIME_PROFILE

    if eps_clip_low < 0:
        raise ValueError(f"eps_clip_low must be >= 0, got {eps_clip_low}")
    if eps_clip_high < 0:
        raise ValueError(f"eps_clip_high must be >= 0, got {eps_clip_high}")
    if ratio_max < 1.0:
        raise ValueError(f"ratio_max must be >= 1.0, got {ratio_max}")
    if q_min <= 0:
        raise ValueError(f"q_min must be > 0, got {q_min}")

    in_dtype = old_log_prob.dtype
    device = old_log_prob.device

    # 为了让 sqrt / 除法更稳，内部统一转 float32 计算
    work_dtype = torch.float32 if in_dtype in (torch.float16, torch.bfloat16) else in_dtype

    lp = old_log_prob.to(work_dtype)
    lp = torch.clamp(lp, min=MIN_LOGP, max=0.0)

    q = torch.exp(lp)
    q_safe = torch.clamp(q, min=q_min)

    eps_low_t = torch.tensor(eps_clip_low, dtype=work_dtype, device=device)
    eps_high_t = torch.tensor(eps_clip_high, dtype=work_dtype, device=device)
    ratio_max_t = torch.tensor(ratio_max, dtype=work_dtype, device=device)

    _sync_if_cuda(device)
    t0 = time.perf_counter()
    
    # lower: 0.5 + 0.5 * sqrt(max(1 - 4 eps_low / q, 0))
    lower_inner = 1.0 - 4.0 * eps_low_t / q_safe
    lower_inner = torch.clamp(lower_inner, min=0.0)
    bound_low = 0.5 + 0.5 * torch.sqrt(lower_inner)

    # upper: 0.5 + 0.5 * sqrt(1 + 4 eps_high / q)
    upper_inner = 1.0 + 4.0 * eps_high_t / q_safe
    bound_high = 0.5 + 0.5 * torch.sqrt(upper_inner)

    _sync_if_cuda(device)
    core_ms = (time.perf_counter() - t0) * 1000.0
    
    # DCPO 的 hard ceiling: r_max = 10
    bound_high = torch.clamp(bound_high, max=ratio_max_t)

    # 基本安全截断
    bound_low = torch.clamp(bound_low, min=0.0, max=1.0)
    bound_high = torch.clamp(bound_high, min=1.0, max=ratio_max_t)

    _LAST_DCPO_DAC_TIME_PROFILE = {
        "core_ms": float(core_ms),
        "numel": int(old_log_prob.numel()),
    }
    
    return bound_low.to(in_dtype), bound_high.to(in_dtype)