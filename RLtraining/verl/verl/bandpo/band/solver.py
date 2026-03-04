# -*- coding: utf-8 -*-
"""
Universal Numerical Solver for BandPO Constraints.
此模块提供完全通用的二分法求解器，用于寻找满足 f-divergence 约束的 ratio 上下界。
基于论文 Theorem 1 和 Proposition 3 严格实现。
"""

import torch

def _safe_g_f(p: torch.Tensor, r: torch.Tensor, f_func: callable, eps: float = 1e-12) -> torch.Tensor:
    """
    计算标量化散度约束函数: g_f(p, r) = p * f(r) + (1-p) * f((1-rp)/(1-p))
    由于 GPU 并行计算中存在越界探索，需要用安全截断防止 f_func 内部产生 NaN。
    """
    # 保证 r 和 互补项 (1-rp)/(1-p) 大于0，防止取对数等操作出现 NaN
    comp = (1.0 - r * p) / (1.0 - p)
    
    # 限制在极小正数，防止越界进入负域
    r_safe = torch.clamp(r, min=eps)
    comp_safe = torch.clamp(comp, min=eps)
    
    return p * f_func(r_safe) + (1.0 - p) * f_func(comp_safe)


def check_simplex_saturation(p: torch.Tensor, delta: float, f_func: callable) -> tuple[torch.Tensor, torch.Tensor]:
    """
    基于论文 Proposition 3 检查单纯形边界饱和情况。
    返回: (is_upper_saturated, is_lower_saturated)
    """
    # 1. 检查上界是否饱和：评估 r = 1/p
    r_max = 1.0 / p
    g_max = _safe_g_f(p, r_max, f_func)
    is_upper_sat = g_max <= delta
    
    # 2. 检查下界是否饱和：评估 r = 0
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
    device = p.device
    dtype = p.dtype
    
    # ---------------- 饱和性检查 ----------------
    is_upper_sat, is_lower_sat = check_simplex_saturation(p, delta, f_func)
    
    # ---------------- 求解上界 (Upper Bound) ----------------
    # r ∈[1, 1/p]。在此区间内 g_f(p,r) 严格单调递增。
    L_up = torch.ones_like(p)
    U_up = 1.0 / p
    
    for _ in range(max_iter):
        M = 0.5 * (L_up + U_up)
        gM = _safe_g_f(p, M, f_func)
        # 递增函数：如果 gM < delta，说明根在右边，L = M
        move_right = gM < delta
        L_up = torch.where(move_right, M, L_up)
        U_up = torch.where(move_right, U_up, M)
        if torch.max(U_up - L_up).item() <= tol:
            break
            
    upper = 0.5 * (L_up + U_up)
    # 如果饱和，直接使用单纯形极值 1/p
    upper = torch.where(is_upper_sat, 1.0 / p, upper)
    
    # ---------------- 求解下界 (Lower Bound) ----------------
    # r ∈ [0, 1]。在此区间内 g_f(p,r) 严格单调递减。
    L_low = torch.zeros_like(p)
    U_low = torch.ones_like(p)
    
    for _ in range(max_iter):
        M = 0.5 * (L_low + U_low)
        gM = _safe_g_f(p, M, f_func)
        # 递减函数：如果 gM < delta，说明根在左边，U = M
        move_left = gM < delta
        U_low = torch.where(move_left, M, U_low)
        L_low = torch.where(move_left, L_low, M)
        if torch.max(U_low - L_low).item() <= tol:
            break
            
    lower = 0.5 * (L_low + U_low)
    # 如果饱和，直接使用单纯形极值 0
    lower = torch.where(is_lower_sat, torch.zeros_like(p), lower)
    
    return lower.to(dtype), upper.to(dtype)