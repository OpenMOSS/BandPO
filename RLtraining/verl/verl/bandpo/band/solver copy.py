# -*- coding: utf-8 -*-
"""
Universal Numerical Solver for BandPO Constraints.
此模块提供完全通用的二分法求解器，用于寻找满足 f-divergence 约束的 ratio 上下界。
基于论文 Theorem 1 和 Proposition 3 严格实现。
"""

import time
import torch

_LAST_UNIVERSAL_BISECTION_TIME_PROFILE = {
    "core_ms": 0.0,
    "numel": 0,
    "max_iter": 0,
}

def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)

def get_last_universal_bisection_time_profile() -> dict:
    return dict(_LAST_UNIVERSAL_BISECTION_TIME_PROFILE)

def _safe_g_f(p: torch.Tensor, r: torch.Tensor, f_func: callable, eps: float = 1e-12) -> torch.Tensor:
    """
    计算标量化散度约束函数: g_f(p, r) = p * f(r) + (1-p) * f((1-rp)/(1-p))
    由于 GPU 并行计算中存在越界探索，需要用安全截断防止 f_func 内部产生 NaN。
    """
    comp = (1.0 - r * p) / (1.0 - p)
    r_safe = torch.clamp(r, min=eps)
    comp_safe = torch.clamp(comp, min=eps)
    return p * f_func(r_safe) + (1.0 - p) * f_func(comp_safe)

def check_simplex_saturation(p: torch.Tensor, delta: float, f_func: callable) -> tuple[torch.Tensor, torch.Tensor]:
    """
    基于论文 Proposition 3 检查单纯形边界饱和情况。
    返回: (is_upper_saturated, is_lower_saturated)
    """
    r_max = 1.0 / p
    g_max = _safe_g_f(p, r_max, f_func)
    is_upper_sat = g_max <= delta

    r_min = torch.zeros_like(p)
    g_min = _safe_g_f(p, r_min, f_func)
    is_lower_sat = g_min <= delta

    return is_upper_sat, is_lower_sat

def universal_bisection_solver(
    p: torch.Tensor,
    delta: float,
    f_func: callable,
    max_iter: int = 50,
    tol: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    完全通用的纯二分法求解器，仅依赖于生成函数 f_func。
    无需 Newton 法，不依赖于 log_domain，稳定且鲁棒。
    """
    global _LAST_UNIVERSAL_BISECTION_TIME_PROFILE

    device = p.device
    dtype = p.dtype

    _sync_if_cuda(device)
    t0 = time.perf_counter()

    # ---------------- 饱和性检查 ----------------
    is_upper_sat, is_lower_sat = check_simplex_saturation(p, delta, f_func)

    # ---------------- 求解上界 (Upper Bound) ----------------
    L_up = torch.ones_like(p)
    U_up = 1.0 / p

    for _ in range(max_iter):
        M = 0.5 * (L_up + U_up)
        gM = _safe_g_f(p, M, f_func)
        move_right = gM < delta
        L_up = torch.where(move_right, M, L_up)
        U_up = torch.where(move_right, U_up, M)
        if torch.max(U_up - L_up).item() <= tol:
            break

    upper = 0.5 * (L_up + U_up)
    upper = torch.where(is_upper_sat, 1.0 / p, upper)

    # ---------------- 求解下界 (Lower Bound) ----------------
    L_low = torch.zeros_like(p)
    U_low = torch.ones_like(p)

    for _ in range(max_iter):
        M = 0.5 * (L_low + U_low)
        gM = _safe_g_f(p, M, f_func)
        move_left = gM < delta
        U_low = torch.where(move_left, M, U_low)
        L_low = torch.where(move_left, L_low, M)
        if torch.max(U_low - L_low).item() <= tol:
            break

    lower = 0.5 * (L_low + U_low)
    lower = torch.where(is_lower_sat, torch.zeros_like(p), lower)

    _sync_if_cuda(device)
    core_ms = (time.perf_counter() - t0) * 1000.0
    _LAST_UNIVERSAL_BISECTION_TIME_PROFILE = {
        "core_ms": float(core_ms),
        "numel": int(p.numel()),
        "max_iter": int(max_iter),
    }

    return lower.to(dtype), upper.to(dtype)