# -*- coding: utf-8 -*-
# 统一入口：根据 old_log_prob 逐 token 生成 ratio 的 (lower, upper)。
# 支持两种方法：{"bandkl","dcpo"}，并提供“高概率松弛”与 upper_bound_max 限幅，端点极限统一处理。

from __future__ import annotations
import torch

from verl.bandpo.kl2clipbound.dcpo import compute_tokenwise_ratio_bounds_by_dcpo
from verl.bandpo.kl2clipbound.bandkl import compute_tokenwise_ratio_bounds_by_bandkl

@torch.no_grad()
def compute_tokenwise_ratio_bounds_core(
    old_log_prob: torch.Tensor,
    *,
    delta: float = 0.1,
    eps_clip_high: float = 0.28,
    eps_clip_low: float = 0.20,
    does_relax_high_p_bound: bool = True,
    upper_bound_max: float = 100.0,
    method: str = "bandkl",                # {"bandkl","dcpo"}
    band_numerical_solver: str = "bisect",   # {"newton","bisect"}
    band_use_log_domain: bool = True,
    max_iter_bisect: int = 80,
    max_iter_newton: int = 12,
    tol: float = 1e-10,
    TINY_ABS: float = 1e-18,
    REL_MARGIN: float = 1e-12,
    BRACKET_ABS: float = 1e-12,
    MIN_LOGP: float = -700.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    返回 (bound_low, bound_high) = (lower, upper)，用于 torch.clamp(ratio, min=lower, max=upper)。
    """
    dtype = old_log_prob.dtype
    device = old_log_prob.device

    # 安全 p
    lp = torch.clamp(old_log_prob, MIN_LOGP, 0.0)
    p = torch.exp(lp)

    # 先按方法得到“基础（未做端点/松弛/限幅）”的 lower/upper
    if method.lower() == "dcpo":
        lower, upper = compute_tokenwise_ratio_bounds_by_dcpo(
            old_log_prob,
            eps_clip_high=eps_clip_high,
            eps_clip_low=eps_clip_low,
            upper_bound_max=upper_bound_max,
            MIN_LOGP=MIN_LOGP,
        )
    elif method.lower() == "bandkl":
        lower, upper = compute_tokenwise_ratio_bounds_by_bandkl(
            old_log_prob,
            delta=delta,
            solve=band_numerical_solver,
            use_log_domain=band_use_log_domain,
            max_iter_bisect=max_iter_bisect,
            max_iter_newton=max_iter_newton,
            tol=tol,
            TINY_ABS=TINY_ABS,
            REL_MARGIN=REL_MARGIN,
            BRACKET_ABS=BRACKET_ABS,
            MIN_LOGP=MIN_LOGP,
        )
    else:
        raise ValueError(f"Unknown method='{method}', expected 'bandkl' or 'dcpo'.")

    nan_id = torch.where(torch.isnan(lower).ravel())[0]
    nan_id_list = nan_id.tolist()
    if nan_id_list:
        raise ValueError(f"lower: NaN IDs: {nan_id.tolist()}")
        # print("lower: NaN IDs:", nan_id.tolist())
        # lower = torch.nan_to_num(lower, nan=0.9, posinf=1.0, neginf=0.0)
    else:
        pass
    nan_id = torch.where(torch.isnan(upper).ravel())[0]
    nan_id_list = nan_id.tolist()
    if nan_id_list:
        raise ValueError(f"upper: NaN IDs: {nan_id.tolist()}")
        # print("upper: NaN IDs:", nan_id.tolist())
        # upper = torch.nan_to_num(upper, nan=1.0, posinf=upper_bound_max, neginf=1.0)
    else:
        pass

    # 高概率松弛（"不过窄"）：与常数 clip 合并
    if does_relax_high_p_bound:
        upper = torch.maximum(upper, torch.tensor(1.0 + eps_clip_high, dtype=dtype, device=device))
        lower = torch.minimum(lower, torch.tensor(1.0 - eps_clip_low,  dtype=dtype, device=device))

    # 最终限幅与合法区间
    upper = torch.clamp(upper, min=1.0, max=upper_bound_max)
    lower = torch.clamp(lower, min=0.0, max=1.0)

    return lower.to(dtype), upper.to(dtype)
