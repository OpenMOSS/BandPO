# -*- coding: utf-8 -*-
"""
BandPO Main Dispatcher and Analytical Formulas.
提供针对 KL, TV, Pearson Chi^2 等散度的 Trust Region 到 Clip Bounds 的映射计算。
"""

import math
import torch
from verl.bandpo.band.solver import (
    universal_bisection_solver,
    get_last_universal_bisection_time_profile,
)

_LAST_BAND_TIME_PROFILE = {
    "core_ms": 0.0,
    "numel": 0,
    "method": "",
}

def get_last_band_time_profile() -> dict:
    return dict(_LAST_BAND_TIME_PROFILE)

# ---------------------------------------------------------
# Generator Functions f(u) and Limits r* (as p -> 1)
# ---------------------------------------------------------

def f_kl(u: torch.Tensor) -> torch.Tensor:
    return -torch.log(u) + u - 1.0

def f_tv(u: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.abs(u - 1.0)

def f_chi2(u: torch.Tensor) -> torch.Tensor:
    return (u - 1.0)**2

SUPPORTED_DIVERGENCES = {
    "bandkl":   {"f": f_kl,   "r_star": lambda d: math.exp(-d),      "has_analytical": False},
    "bandtv":   {"f": f_tv,   "r_star": lambda d: max(1.0 - d, 0.0), "has_analytical": True},
    "bandchi2": {"f": f_chi2, "r_star": lambda d: 1.0,               "has_analytical": True},
}

def _solve_bandtv_analytical(p: torch.Tensor, delta: float) -> tuple[torch.Tensor, torch.Tensor]:
    upper_raw = 1.0 + delta / p
    lower_raw = 1.0 - delta / p
    upper = torch.clamp(upper_raw, max=1.0 / p)
    lower = torch.clamp(lower_raw, min=0.0)
    return lower, upper

def _solve_bandchi2_analytical(p: torch.Tensor, delta: float) -> tuple[torch.Tensor, torch.Tensor]:
    term = torch.sqrt(delta * (1.0 - p) / p)
    upper_raw = 1.0 + term
    lower_raw = 1.0 - term
    upper = torch.clamp(upper_raw, max=1.0 / p)
    lower = torch.clamp(lower_raw, min=0.0)
    return lower, upper

@torch.no_grad()
def band(
    old_log_prob: torch.Tensor,
    *,
    method: str = "bandkl",
    delta: float = 0.05,
    eps_clip_high: float = 0.28,
    eps_clip_low: float = 0.20,
    does_relax_high_p_bound: bool = True,
    upper_bound_max: float = 100.0,
    MIN_LOGP: float = -700.0,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    global _LAST_BAND_TIME_PROFILE

    method = method.lower()
    if method not in SUPPORTED_DIVERGENCES:
        raise ValueError(
            f"[BandPO] Error: Unsupported divergence '{method}'. "
            f"Supported list: {list(SUPPORTED_DIVERGENCES.keys())}"
        )

    dtype = old_log_prob.dtype
    device = old_log_prob.device

    meta = SUPPORTED_DIVERGENCES[method]
    r_star_val = meta["r_star"](delta)

    lp = torch.clamp(old_log_prob, MIN_LOGP, 0.0)
    p = torch.exp(lp)

    core_ms = 0.0

    # -----------------------------------------------------
    # 1. 核心映射 (Analytical or Numerical)
    # -----------------------------------------------------
    if meta["has_analytical"]:
        if method == "bandtv":
            lower_raw, upper_raw = _solve_bandtv_analytical(p, delta)
        elif method == "bandchi2":
            lower_raw, upper_raw = _solve_bandchi2_analytical(p, delta)
    else:
        lower_raw, upper_raw = universal_bisection_solver(p, delta, f_func=meta["f"])
        solver_prof = get_last_universal_bisection_time_profile()
        core_ms = float(solver_prof.get("core_ms", 0.0))

    # -----------------------------------------------------
    # 2. 极限处理
    # -----------------------------------------------------
    eps_limit = 1e-6
    mask_to_1 = p > (1.0 - eps_limit)
    mask_to_0 = p < eps_limit

    upper = torch.where(mask_to_1, torch.ones_like(p), upper_raw)
    lower = torch.where(mask_to_1, torch.full_like(p, r_star_val), lower_raw)

    upper = torch.where(mask_to_0, 1.0 / torch.clamp(p, min=1e-12), upper)
    lower = torch.where(mask_to_0, torch.zeros_like(p), lower)

    # -----------------------------------------------------
    # 3. 启发式松弛与最大安全限幅
    # -----------------------------------------------------
    if does_relax_high_p_bound:
        lower = torch.minimum(lower, torch.tensor(1.0 - eps_clip_low, dtype=dtype, device=device))
        upper = torch.maximum(upper, torch.tensor(1.0 + eps_clip_high, dtype=dtype, device=device))

    lower = torch.clamp(lower, min=0.0, max=1.0)
    upper = torch.clamp(upper, min=1.0, max=upper_bound_max)

    _LAST_BAND_TIME_PROFILE = {
        "core_ms": float(core_ms),
        "numel": int(p.numel()),
        "method": method,
    }

    return lower.to(dtype), upper.to(dtype)