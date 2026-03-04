# -*- coding: utf-8 -*-
# DCPO(SAC-style) 自适应 ratio 上下界（逐 token）
# 公式： r_high = 0.5 + 0.5*sqrt(1 + 4*eps_high/p)
#      r_low  = 0.5 + 0.5*sqrt(max(1 - 4*eps_low/p, 0))
# 注意：端点与极端概率的最终处理放在 tokenwise_bounds.py 统一做。

from __future__ import annotations
import torch
import time


@torch.no_grad()
def compute_tokenwise_ratio_bounds_by_dcpo(
    old_log_prob: torch.Tensor,
    *,
    eps_clip_high: float,
    eps_clip_low: float,
    upper_bound_max: float,
    MIN_LOGP: float = -700.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        old_log_prob: shape (...), 旧策略的 log p。
        eps_clip_high/eps_clip_low: DCPO 的上/下截断系数（直接使用标准 PPO 的 ε_high / ε_low）。
        upper_bound_max: 允许的 ratio 最大值（仅对上界有效，最终裁剪见上层）。
    Returns:
        (lower, upper): 与 old_log_prob 同形。
    """
    start = time.perf_counter_ns()
    dtype = old_log_prob.dtype
    device = old_log_prob.device

    # 安全 p
    lp = torch.clamp(old_log_prob, MIN_LOGP, 0.0)
    p = torch.exp(lp)                               # (0,1]
    denom = torch.clamp(p, min=torch.finfo(dtype).tiny)

    # 上界
    disc_high = 1.0 + 4.0 * eps_clip_high / denom
    r_high = 0.5 + 0.5 * torch.sqrt(torch.clamp(disc_high, min=0.0))

    # 下界（保证非负）
    disc_low = 1.0 - 4.0 * eps_clip_low / denom
    r_low = 0.5 + 0.5 * torch.sqrt(torch.clamp(disc_low, min=0.0))

    # upper_bound_max 约束先给一个软裁剪（最终由上层统一处理端点与限幅）
    r_high = torch.minimum(r_high, torch.tensor(upper_bound_max, dtype=dtype, device=device))
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000  # float 毫秒
    # print(f"dcpo elapsed_ms: {elapsed_ms}\n")
    return r_low.to(dtype), r_high.to(dtype)
