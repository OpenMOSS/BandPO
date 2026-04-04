# -*- coding: utf-8 -*-
"""
DCPO baseline: Dynamic Adaptive Clipping (DAC)
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

    # 核心闭式公式
    lower_inner = 1.0 - 4.0 * eps_low_t / q_safe
    lower_inner = torch.clamp(lower_inner, min=0.0)
    bound_low = 0.5 + 0.5 * torch.sqrt(lower_inner)

    upper_inner = 1.0 + 4.0 * eps_high_t / q_safe
    bound_high = 0.5 + 0.5 * torch.sqrt(upper_inner)

    _sync_if_cuda(device)
    core_ms = (time.perf_counter() - t0) * 1000.0

    # post-process
    bound_high = torch.clamp(bound_high, max=ratio_max_t)
    bound_low = torch.clamp(bound_low, min=0.0, max=1.0)
    bound_high = torch.clamp(bound_high, min=1.0, max=ratio_max_t)

    _LAST_DCPO_DAC_TIME_PROFILE = {
        "core_ms": float(core_ms),
        "numel": int(old_log_prob.numel()),
    }

    return bound_low.to(in_dtype), bound_high.to(in_dtype)