# -*- coding: utf-8 -*-
"""
BandPO Main Dispatcher and Analytical Formulas.
提供针对 KL, TV, Pearson Chi^2 等散度的 Trust Region 到 Clip Bounds 的映射计算。
"""

import math
import torch
from verl.bandpo.band.solver import universal_bisection_solver

# ---------------------------------------------------------
# Generator Functions f(u) and Limits r* (as p -> 1)
# ---------------------------------------------------------

def f_kl(u: torch.Tensor) -> torch.Tensor:
    # 论文推导版本: f(u) = -log(u) + u - 1
    return -torch.log(u) + u - 1.0

def f_tv(u: torch.Tensor) -> torch.Tensor:
    # f_TV(u) = 0.5 * |u - 1|
    return 0.5 * torch.abs(u - 1.0)

def f_chi2(u: torch.Tensor) -> torch.Tensor:
    # f_chi2(u) = (u - 1)^2
    return (u - 1.0)**2

# 注册支持的 divergence 极其在 p->1 时的极值 r_star 的计算闭包
SUPPORTED_DIVERGENCES = {
    "bandkl":   {"f": f_kl,   "r_star": lambda d: math.exp(-d),      "has_analytical": False},
    "bandtv":   {"f": f_tv,   "r_star": lambda d: max(1.0 - d, 0.0), "has_analytical": True},
    "bandchi2": {"f": f_chi2, "r_star": lambda d: 1.0,               "has_analytical": True},
}

# ---------------------------------------------------------
# Analytical Handlers (闭式解处理函数)
# ---------------------------------------------------------

def _solve_bandtv_analytical(p: torch.Tensor, delta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """ Total Variation 解析解 (Prop 4) """
    # 公式：r = 1 ± delta / p
    upper_raw = 1.0 + delta / p
    lower_raw = 1.0 - delta / p
    # 应用单纯形限制 (Clamp to simplex implicitly acts as saturation check for analytical)
    upper = torch.clamp(upper_raw, max=1.0 / p)
    lower = torch.clamp(lower_raw, min=0.0)
    return lower, upper

def _solve_bandchi2_analytical(p: torch.Tensor, delta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """ Pearson Chi^2 解析解 (Prop 4) """
    # 公式：r = 1 ± sqrt(delta * (1-p) / p)
    term = torch.sqrt(delta * (1.0 - p) / p)
    upper_raw = 1.0 + term
    lower_raw = 1.0 - term
    upper = torch.clamp(upper_raw, max=1.0 / p)
    lower = torch.clamp(lower_raw, min=0.0)
    return lower, upper


# ---------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------

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
    **kwargs # 吸收老版本的不兼容参数(如 band_use_log_domain)，防止外部调用崩溃
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    BandPO Version 2 通用映射函数。
    计算公式: max(-A * r, -A * clip(r, lower, upper))
    """
    method = method.lower()
    if method not in SUPPORTED_DIVERGENCES:
        raise ValueError(f"[BandPO] Error: Unsupported divergence '{method}'. "
                         f"Supported list: {list(SUPPORTED_DIVERGENCES.keys())}")
    
    dtype = old_log_prob.dtype
    device = old_log_prob.device
    
    # 获取散度元数据
    meta = SUPPORTED_DIVERGENCES[method]
    r_star_val = meta["r_star"](delta)
    
    # 安全提取 p
    lp = torch.clamp(old_log_prob, MIN_LOGP, 0.0)
    p = torch.exp(lp)
    
    # -----------------------------------------------------
    # 1. 核心映射 (Analytical or Numerical)
    # -----------------------------------------------------
    if meta["has_analytical"]:
        # 解析解通道
        if method == "bandtv":
            lower_raw, upper_raw = _solve_bandtv_analytical(p, delta)
        elif method == "bandchi2":
            lower_raw, upper_raw = _solve_bandchi2_analytical(p, delta)
    else:
        # 通用数值求解通道
        lower_raw, upper_raw = universal_bisection_solver(p, delta, f_func=meta["f"])


    # -----------------------------------------------------
    # 2. 极限处理 (Limit handling for p -> 1 and p -> 0)
    # -----------------------------------------------------
    # 为了防止数值溢出，对极端的 p 应用理论极限进行强行覆盖
    eps_limit = 1e-6
    mask_to_1 = p > (1.0 - eps_limit)
    mask_to_0 = p < eps_limit
    
    # p -> 1 极限覆盖
    upper = torch.where(mask_to_1, torch.ones_like(p), upper_raw)
    lower = torch.where(mask_to_1, torch.full_like(p, r_star_val), lower_raw)
    
    # p -> 0 极限覆盖 (此时上界趋近正无穷，因此只被最后一步的 upper_bound_max 兜底，这里先给上单纯形上界)
    upper = torch.where(mask_to_0, 1.0 / torch.clamp(p, min=1e-12), upper)
    lower = torch.where(mask_to_0, torch.zeros_like(p), lower)

    # -----------------------------------------------------
    # 3. 启发式松弛与最大安全限幅 (Heuristic adjustments)
    # -----------------------------------------------------
    if does_relax_high_p_bound:
        # 针对高概率区间的启发式松弛，将 Band 区间与标准的 clip(1-eps, 1+eps) 求交并放大
        lower = torch.minimum(lower, torch.tensor(1.0 - eps_clip_low, dtype=dtype, device=device))
        upper = torch.maximum(upper, torch.tensor(1.0 + eps_clip_high, dtype=dtype, device=device))
        
    # 最终的安全兜底：确保不低于0，不超过预设的极大值
    lower = torch.clamp(lower, min=0.0, max=1.0)
    upper = torch.clamp(upper, min=1.0, max=upper_bound_max)

    return lower.to(dtype), upper.to(dtype)