# -*- coding: utf-8 -*-
# File: verl/bandpo/soft/softclip.py
#
# 7个soft-clip算法的工程实现（逐token、可微、数值稳定）：
#   1) soft_clip_1seg
#   2) soft_clip_2seg
#   3) soft_clip_3seg_control_converge(lower_method="converge")
#   4) soft_clip_3seg_control_converge(lower_method="reach")  # g 自适应扩大至可解区
#   5) soft_clip_3seg（封闭解，C^1 拼接）
#   6) soft_clip_3seg_rollback（中段恒等）
#   7) soft_clip_3seg_rollback（中段激活版）
#   6) soft_clip_3seg_rollback_plus（中段恒等）
#   7) soft_clip_3seg_rollback_plus（中段激活版）
#
# 另提供 apply_soft_clip(...) 统一分发，便于训练代码中替换 clamp。

from __future__ import annotations
from typing import Tuple, Literal, Optional
import torch

# ------------------------- 数值辅助 -------------------------
def _const_like(x: torch.Tensor, val: float) -> torch.Tensor:
    return torch.tensor(val, dtype=x.dtype, device=x.device)

def _atanh_safe(x: torch.Tensor, eps: Optional[float] = None) -> torch.Tensor:
    """稳定的 atanh；把输入夹到 (-1+eps, 1-eps) 再用 0.5*(log1p(x)-log1p(-x))."""
    if eps is None:
        eps = float(torch.finfo(x.dtype).eps)
    x = torch.clamp(x, min=-(1.0 - eps), max=(1.0 - eps))
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def _split_points(bound_low: torch.Tensor,
                  bound_high: torch.Tensor,
                  rho: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    e_high = rho + (1-rho)*bound_high ∈ [1, bound_high]
    e_low  = rho + (1-rho)*bound_low  ∈ [bound_low, 1]
    """
    # 选一个形状做基准以便广播
    base = bound_high if bound_high.numel() >= bound_low.numel() else bound_low
    one = _const_like(base, 1.0)
    rho_t = _const_like(base, float(rho))
    e_high = rho_t + (one - rho_t) * bound_high
    e_low  = rho_t + (one - rho_t) * bound_low
    return e_high, e_low

# ------------------------- 1) 单段 tanh-clamp（全域 C∞） -------------------------
def soft_clip_1seg(
    ratio: torch.Tensor,
    bound_low: torch.Tensor,
    bound_high: torch.Tensor,
    k: Optional[float] = None,      # 若为 None，默认 k = 2 / (bound_high - bound_low)
    tiny: float = 1e-12,
) -> torch.Tensor:
    """
    y(r) = m + 0.5*(B-A) * tanh( k * (r - m) ), 其中 m=(A+B)/2, A=bound_low, B=bound_high.
    - 全域平滑；两端渐近 A/B。
    - 在中段不是恒等（与PPO语义略有偏差），但实现最简单、梯度处处存在。
    """
    ratio      = ratio
    bound_low  = torch.as_tensor(bound_low,  dtype=ratio.dtype, device=ratio.device)
    bound_high = torch.as_tensor(bound_high, dtype=ratio.dtype, device=ratio.device)

    A = bound_low
    B = bound_high
    m = 0.5 * (A + B)
    n = 0.5 * (B - A)  # 半宽
    n_safe = torch.where(n.abs() < _const_like(n, tiny), _const_like(n, tiny), n)

    if k is None:
        # 与你numpy默认一致：k = 1/n = 2/(B-A)
        k_val = 1.0 / n_safe
    else:
        k_val = _const_like(ratio, float(k))

    y = m + n * torch.tanh(k_val * (ratio - m))
    return y

# ------------------------- 2) 两段式（以 r=1 为分界），C¹ 拼接 -------------------------
def soft_clip_2seg(
    ratio: torch.Tensor,
    bound_low: torch.Tensor,
    bound_high: torch.Tensor,
    k: float = 1.0,
    tiny: float = 1e-12,
) -> torch.Tensor:
    """
    r>=1: y = 1 + eps_high * tanh( (k/eps_high)*(r-1) )
    r<=1: y = 1 + eps_low  * tanh( (k/eps_low) *(r-1) )
    在 r=1 处函数值与导数均连续。
    """
    ratio      = ratio
    one        = _const_like(ratio, 1.0)
    bound_low  = torch.as_tensor(bound_low,  dtype=ratio.dtype, device=ratio.device)
    bound_high = torch.as_tensor(bound_high, dtype=ratio.dtype, device=ratio.device)

    eps_high = (bound_high - one)
    eps_low  = (one - bound_low)

    eps_high_safe = torch.where(eps_high.abs() < _const_like(ratio, tiny), _const_like(ratio, tiny), eps_high)
    eps_low_safe  = torch.where(eps_low.abs()  < _const_like(ratio, tiny), _const_like(ratio, tiny), eps_low)

    z = ratio - one
    y_hi = one + eps_high * torch.tanh((_const_like(ratio, float(k))/eps_high_safe) * z)
    y_lo = one + eps_low  * torch.tanh((_const_like(ratio, float(k))/eps_low_safe)  * z)

    y = torch.where(ratio >= one, y_hi, y_lo)
    # 可选：把极端 eps≈0 的侧面退化成恒等
    y = torch.where(eps_high.abs() < _const_like(ratio, tiny), ratio, y)
    y = torch.where(eps_low.abs()  < _const_like(ratio, tiny), ratio, y)
    return y

# ------------------------- 3/4) 三段式（控制收敛/抵达），中段恒等 -------------------------
def _params_high_converge(high_bound: torch.Tensor, e_high: torch.Tensor, g: torch.Tensor):
    # a = (B - e_h)*(1+E)/2, b = (1+e^{-2g})/(2(B-e_h)), c = e_h - [2g(B-e_h)E/(1+E)], d = B - a
    E = torch.exp(2.0 * g)
    a = (high_bound - e_high) * (1.0 + E) / 2.0
    b = (1.0 + torch.exp(-2.0 * g)) / (2.0 * torch.clamp(high_bound - e_high, min=torch.finfo(high_bound.dtype).eps))
    c = e_high - (2.0 * g * (high_bound - e_high) * E) / (1.0 + E)
    d = high_bound - a
    return a, b, c, d

def _params_low_converge(low_bound: torch.Tensor, e_low: torch.Tensor, g: torch.Tensor):
    # a = (e_l - A)*(1+E)/(2E), b = (1+E)/(2(e_l-A)), c = e_l - [2g(e_l-A)/(1+E)], d = A + a
    E = torch.exp(2.0 * g)
    a = (e_low - low_bound) * (1.0 + E) / (2.0 * E)
    b = (1.0 + E) / (2.0 * torch.clamp(e_low - low_bound, min=torch.finfo(low_bound.dtype).eps))
    c = e_low - (2.0 * g * (e_low - low_bound)) / (1.0 + E)
    d = low_bound + a
    return a, b, c, d

def _coth(x: torch.Tensor, tiny: float) -> torch.Tensor:
    # coth(x) = cosh(x)/sinh(x); 保护 |x| 很小
    tiny_t = _const_like(x, tiny)
    x_safe = torch.where(x.abs() < tiny_t, torch.sign(x) * tiny_t, x)
    return torch.cosh(x_safe) / torch.sinh(x_safe)

def _find_g_for_reach(e_low: torch.Tensor,
                      low_bound: torch.Tensor,
                      g_init: float,
                      tol: float = 1e-7,
                      max_iter: int = 60) -> torch.Tensor:
    """
    向量化搜索：T(g) = coth(g) - e_low / ( g (e_low - low_bound) ) ∈ (-1,1)。
    若不满足，则将 g 按几何级数放大直到 T(g) > -1 + tol。
    返回逐元素 g_eff。
    """
    # 常量/初始化
    g = torch.full_like(e_low, float(g_init))
    tiny = torch.finfo(e_low.dtype).eps
    denom = torch.clamp(e_low - low_bound, min=tiny)
    S = e_low / denom

    # 掩码：需要增大 g 的位置
    def T_of(gg: torch.Tensor) -> torch.Tensor:
        return _coth(gg, tiny) - (S / torch.clamp(gg, min=_const_like(gg, tiny)))

    T = T_of(g)
    need = (T <= -1.0 + tol)

    it = 0
    # 逐步放大 g；向量化更新，仅在 need 的位置更新
    while need.any() and it < max_iter:
        g = torch.where(need, g * 1.5, g)
        T = T_of(g)
        need = (T <= -1.0 + tol)
        it += 1
    return g

def _params_low_reach(low_bound: torch.Tensor,
                      e_low: torch.Tensor,
                      g_init: float,
                      tol: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reach 版下侧参数：
      T(g) := coth(g) - e_low/[ g (e_low - low_bound) ] ∈ (-1, 1)
      u = atanh(T)
      a = e_low / [ g (1 - T^2) ] = e_low * cosh(u)^2 / g
      b = g / e_low
      c = e_low - u / b = e_low - e_low*u / g
      d = e_low - a*T
    返回：a, b, c, d, g_eff
    """
    # 找逐元素 g_eff 使 |T|<1
    g_eff = _find_g_for_reach(e_low, low_bound, g_init=g_init, tol=tol)
    T = _coth(g_eff, torch.finfo(e_low.dtype).eps) - e_low / (torch.clamp(g_eff, min=_const_like(e_low, tol)) * torch.clamp(e_low - low_bound, min=_const_like(e_low, tol)))
    # T 必在 (-1,1)，数值上仍做一层极小的保护
    T = torch.clamp(T, min=-1.0 + 1e-9, max=1.0 - 1e-9)
    u = _atanh_safe(T, eps=1e-9)

    a = e_low / (torch.clamp(g_eff, min=_const_like(e_low, 1e-9)) * (1.0 - T * T))
    b = torch.clamp(g_eff, min=_const_like(e_low, 1e-9)) / torch.clamp(e_low, min=_const_like(e_low, 1e-9))
    c = e_low - (u / torch.clamp(b, min=_const_like(e_low, 1e-9)))
    d = e_low - a * T
    return a, b, c, d, g_eff

def soft_clip_3seg_control_converge(
    ratio: torch.Tensor,
    bound_low: torch.Tensor,
    bound_high: torch.Tensor,
    rho: float = 0.10,
    g: float = 1.0,
    lower_method: Literal["converge", "reach"] = "converge",
    tiny: float = 1e-12,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Method B（三段）：(r <= e_low) / (e_low < r < e_high) / (r >= e_high)
      - 高侧：tanh 收敛到 high_bound，满足 y(e_high)=e_high, y'(e_high)=1
      - 中段：恒等 y=r
      - 低侧：两种可选：
          * "converge": tanh 收敛到 low_bound
          * "reach":    通过 g_eff 使 y(0)=low_bound 且 y(e_low)=e_low, y'(e_low)=1
    返回：y, (e_high, e_low), g_eff
    """
    ratio      = ratio
    dtype      = ratio.dtype
    device     = ratio.device
    one        = _const_like(ratio, 1.0)

    bound_low  = torch.as_tensor(bound_low,  dtype=dtype, device=device)
    bound_high = torch.as_tensor(bound_high, dtype=dtype, device=device)

    e_high, e_low = _split_points(bound_low, bound_high, rho)

    # 高侧参数（converge）
    aH, bH, cH, dH = _params_high_converge(bound_high, e_high, _const_like(ratio, float(g)))

    # 低侧参数
    lower_method = str(lower_method).lower()
    if lower_method == "converge":
        aL, bL, cL, dL = _params_low_converge(bound_low, e_low, _const_like(ratio, float(g)))
        g_eff = _const_like(ratio, float(g))
    elif lower_method == "reach":
        aL, bL, cL, dL, g_eff = _params_low_reach(bound_low, e_low, g_init=float(g))
    else:
        raise ValueError("lower_method must be 'converge' or 'reach'.")

    # 三段输出：低/中/高
    mask_lo  = (ratio <= e_low)
    mask_hi  = (ratio >= e_high)
    mask_mid = (~mask_lo) & (~mask_hi)

    y = torch.empty_like(ratio)
    # 低侧（并保证非负）：y := max(y, 0)
    y_lo = aL * torch.tanh(bL * (ratio - cL)) + dL
    y_lo = torch.maximum(_const_like(ratio, 0.0), y_lo)

    # 中段：恒等
    y_mid = ratio

    # 高侧
    y_hi = aH * torch.tanh(bH * (ratio - cH)) + dH

    y = torch.where(mask_lo, y_lo, torch.where(mask_hi, y_hi, y_mid))
    return y, (e_high, e_low), g_eff

# ------------------------- 5) 三段式封闭解（C¹ 拼接，中段恒等） -------------------------
def soft_clip_3seg(
    ratio: torch.Tensor,
    bound_low: torch.Tensor,
    bound_high: torch.Tensor,
    rho: float = 0.10,
    tiny: float = 1e-12,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    高侧/低侧用封闭解（kappa, c）保证在 e_high/e_low 处 y=e, y'=1；中段恒等。
      r >= e_high: y = 1 + eps_high * tanh( kappa_h*(r-1) + c_h )
      r <= e_low : y = 1 + eps_low  * tanh( kappa_l*(r-1) + c_l )
      e_low < r < e_high: y=r
    其中：
      eps_high = B-1, eps_low = 1-A;
      delta_e_high = e_high - 1, delta_e_low = 1 - e_low;
      kappa_h = eps_high / (eps_high^2 - delta_e_high^2);
      c_h     = atanh(delta_e_high/eps_high) - kappa_h*delta_e_high;
      kappa_l = eps_low  / (eps_low^2  - delta_e_low^2 );
      c_l     = -atanh(delta_e_low/eps_low) + kappa_l*delta_e_low.
    """
    ratio      = ratio
    dtype      = ratio.dtype
    device     = ratio.device
    one        = _const_like(ratio, 1.0)
    eps_t      = _const_like(ratio, tiny)

    bound_low  = torch.as_tensor(bound_low,  dtype=dtype, device=device)
    bound_high = torch.as_tensor(bound_high, dtype=dtype, device=device)

    eps_high = (bound_high - one)
    eps_low  = (one - bound_low)

    e_high, e_low = _split_points(bound_low, bound_high, rho)
    delta_e_high = e_high - one
    delta_e_low  = one - e_low

    den_h = eps_high * eps_high - delta_e_high * delta_e_high
    den_l = eps_low  * eps_low  - delta_e_low  * delta_e_low
    den_h = torch.where(den_h.abs() < eps_t, torch.sign(den_h) * eps_t, den_h)
    den_l = torch.where(den_l.abs() < eps_t, torch.sign(den_l) * eps_t, den_l)

    eps_high_safe = torch.where(eps_high.abs() < eps_t, eps_t, eps_high)
    eps_low_safe  = torch.where(eps_low.abs()  < eps_t, eps_t, eps_low)

    t_h = torch.clamp(delta_e_high / eps_high_safe, min=-(1.0 - tiny), max=(1.0 - tiny))
    t_l = torch.clamp(-delta_e_low  / eps_low_safe,  min=-(1.0 - tiny), max=(1.0 - tiny))

    kappa_h = eps_high / den_h
    c_h     = _atanh_safe(t_h, eps=tiny) - kappa_h * delta_e_high

    kappa_l = eps_low  / den_l
    c_l     = -_atanh_safe(-t_l, eps=tiny) + kappa_l * delta_e_low

    mask_lo  = (ratio <= e_low)
    mask_hi  = (ratio >= e_high)
    mask_mid = (~mask_lo) & (~mask_hi)

    y_lo = one + eps_low  * torch.tanh(kappa_l * (ratio - one) + c_l)
    y_hi = one + eps_high * torch.tanh(kappa_h * (ratio - one) + c_h)

    # 当 eps_*≈0（边界≈1）时，该侧退化为恒等，避免噪声
    y_lo = torch.where(eps_low.abs()  < eps_t, ratio, y_lo)
    y_hi = torch.where(eps_high.abs() < eps_t, ratio, y_hi)

    y_mid = ratio
    y = torch.where(mask_lo, y_lo, torch.where(mask_hi, y_hi, y_mid))
    return y, (e_high, e_low)

# ------------------------- 6/7) 回拉版（含中段激活可选） -------------------------
def soft_clip_3seg_rollback(
    ratio: torch.Tensor,
    bound_low: torch.Tensor,
    bound_high: torch.Tensor,
    alpha: float = 0.10,
    use_activate_function: bool = False,
    alpha_in: float = 2.0,
    tiny: float = 1e-12,
) -> torch.Tensor:
    """
    越界区用回拉项（-alpha * tanh(r-1)）把比率往区间拉回；中段默认恒等。
    若 use_activate_function=True，则中段在 1 左右用平滑tanh激活（两侧曲率可调）。
    """
    ratio      = ratio
    dtype      = ratio.dtype
    device     = ratio.device
    one        = _const_like(ratio, 1.0)
    eps_t      = _const_like(ratio, tiny)

    bound_low  = torch.as_tensor(bound_low,  dtype=dtype, device=device)
    bound_high = torch.as_tensor(bound_high, dtype=dtype, device=device)

    eps_high = (bound_high - one)
    eps_low  = (one - bound_low)

    # mask：直接用边界分段
    mask_lo  = (ratio <= bound_low)
    mask_hi  = (ratio >= bound_high)
    mask_mid = (~mask_lo) & (~mask_hi)

    # 高侧回拉放大（避免 bound_low→0 时不稳定）
    bl_safe = torch.where(bound_low.abs() < eps_t, eps_t, bound_low)
    alpha_low  = _const_like(ratio, float(alpha))
    alpha_high = _const_like(ratio, float(alpha)) * (bound_high / bl_safe) ** 2

    # 越界两侧
    y_lo = -alpha_low  * torch.tanh(ratio - one) + bound_low  - alpha_low  * torch.tanh(eps_low)
    y_hi = -alpha_high * torch.tanh(ratio - one) + bound_high + alpha_high * torch.tanh(eps_high)

    # 中段
    if not use_activate_function:
        y_mid = ratio
    else:
        eps_low_safe  = torch.where(eps_low.abs()  < eps_t, eps_t, eps_low)
        eps_high_safe = torch.where(eps_high.abs() < eps_t, eps_t, eps_high)
        a_in = _const_like(ratio, float(alpha_in))
        mask_mid_low  = mask_mid & (ratio <= one)
        mask_mid_high = mask_mid & (ratio >  one)
        y_mid = ratio.clone()
        y_mid_low  = one + (eps_low / torch.tanh(a_in))  * torch.tanh((a_in / eps_low_safe)  * (ratio - one))
        y_mid_high = one + (eps_high/ torch.tanh(a_in))  * torch.tanh((a_in / eps_high_safe) * (ratio - one))
        y_mid = torch.where(mask_mid_low,  y_mid_low,  y_mid)
        y_mid = torch.where(mask_mid_high, y_mid_high, y_mid)

    y = torch.where(mask_lo, y_lo, torch.where(mask_hi, y_hi, y_mid))
    return y
def soft_clip_3seg_rollback_plus(
    ratio: torch.Tensor,
    bound_low: torch.Tensor,
    bound_high: torch.Tensor,
    alpha: float = 0.10,
    gamma: float = 0.20,
    use_activate_function: bool = False,
    alpha_in: float = 2.0,
    tiny: float = 1e-12,
) -> torch.Tensor:
    """
    越界区用回拉项（-alpha * tanh(r-1)）把比率往区间拉回；中段默认恒等。
    若 use_activate_function=True，则中段在 1 左右用平滑tanh激活（两侧曲率可调）。
    """
    ratio      = ratio
    dtype      = ratio.dtype
    device     = ratio.device
    one        = _const_like(ratio, 1.0)
    eps_t      = _const_like(ratio, tiny)

    bound_low  = torch.as_tensor(bound_low,  dtype=dtype, device=device)
    bound_high = torch.as_tensor(bound_high, dtype=dtype, device=device)

    eps_high = (bound_high - one)
    eps_low  = (one - bound_low)
    eps_high_safe = torch.where(eps_high.abs() < eps_t, eps_t, eps_high)
    eps_low_safe  = torch.where(eps_low.abs()  < eps_t, eps_t, eps_low)
    # mask：直接用边界分段
    mask_lo  = (ratio <= bound_low)
    mask_hi  = (ratio >= bound_high)
    mask_mid = (~mask_lo) & (~mask_hi)

    # 高侧回拉放大（避免 bound_low→0 时不稳定）
    bl_safe = torch.where(bound_low.abs() < eps_t, eps_t, bound_low)
    alpha_low  = _const_like(ratio, float(alpha))
    alpha_high = _const_like(ratio, float(alpha)) * (eps_high / eps_low)

    # 越界两侧
    gamma_t = _const_like(ratio, float(gamma))
    y_lo = -alpha_low  * torch.tanh(gamma_t * (ratio - bound_low)) + bound_low # bound_low + alpha
    y_hi = -alpha_high * torch.tanh(gamma_t * (ratio - bound_high)) + bound_high # bound_high - alpha * (bound_high-1) / eps_low

    # 中段
    if not use_activate_function:
        y_mid = ratio
    else:
        eps_low_safe  = torch.where(eps_low.abs()  < eps_t, eps_t, eps_low)
        eps_high_safe = torch.where(eps_high.abs() < eps_t, eps_t, eps_high)
        a_in = _const_like(ratio, float(alpha_in))
        mask_mid_low  = mask_mid & (ratio <= one)
        mask_mid_high = mask_mid & (ratio >  one)
        y_mid = ratio.clone()
        y_mid_low  = one + (eps_low / torch.tanh(a_in))  * torch.tanh((a_in / eps_low_safe)  * (ratio - one))
        y_mid_high = one + (eps_high/ torch.tanh(a_in))  * torch.tanh((a_in / eps_high_safe) * (ratio - one))
        y_mid = torch.where(mask_mid_low,  y_mid_low,  y_mid)
        y_mid = torch.where(mask_mid_high, y_mid_high, y_mid)

    y = torch.where(mask_lo, y_lo, torch.where(mask_hi, y_hi, y_mid))
    return y

# ------------------------- 统一分发器（工程调用友好） -------------------------
def apply_soft_clip(
    ratio: torch.Tensor,
    bound_low: torch.Tensor | float,
    bound_high: torch.Tensor | float,
    method: Literal[
        "hard",
        "1seg",
        "2seg",
        "3seg",
        "3seg_control_converge",
        "3seg_control_reach",
        "rollback",
        "rollback_activate",
        "rollback_plus",
        "rollback_activate_plus",
    ] = "3seg",
    *,
    # 共用可选超参
    rho: float = 0.1,
    g: float = 1.0,
    k: Optional[float] = None,
    # 当 alpha大于eps_low=（1-bound_low）时会导致上界的收敛值小于1。本试验下界一般为0.8，因此建议小于0.2
    alpha: float = 0.05, # 强烈建议alpha小于eps_low，目测决定0.2
    gamma: float = 0.20, # 决定梯度大小，控制收敛的快慢，目测决定0.2
    # alpha: float = 0.10,
    # gamma: float = 0.20,
    alpha_in: float = 2.0,
    tiny: float = 1e-12,
):
    """
    统一接口：返回 y = soft_clip_method(ratio; bounds, hyper-params).
    """
    
    if torch.is_tensor(bound_low):
        pass
    else:
        bound_low = _const_like(ratio, bound_low)
    if bound_low.requires_grad: bound_low  = bound_low.detach()
    if torch.is_tensor(bound_high):
        pass
    else:
        bound_high = _const_like(ratio, bound_high)
    if bound_high.requires_grad: bound_high = bound_high.detach()
    
    m = method.lower()
    if m == "hard":
        return torch.clamp(ratio, min=bound_low, max=bound_high)
    elif m == "1seg":
        return soft_clip_1seg(ratio, bound_low, bound_high, k=k, tiny=tiny)
    elif m == "2seg":
        return soft_clip_2seg(ratio, bound_low, bound_high, k=1.0 if k is None else float(k), tiny=tiny)
    elif m == "3seg":
        y, _ = soft_clip_3seg(ratio, bound_low, bound_high, rho=rho, tiny=tiny)
        return y
    elif m == "3seg_control_converge":
        y, _, _ = soft_clip_3seg_control_converge(ratio, bound_low, bound_high, rho=rho, g=g, lower_method="converge", tiny=tiny)
        return y
    elif m == "3seg_control_reach":
        y, _, _ = soft_clip_3seg_control_converge(ratio, bound_low, bound_high, rho=rho, g=g, lower_method="reach", tiny=tiny)
        return y
    elif m == "rollback":
        return soft_clip_3seg_rollback(ratio, bound_low, bound_high, alpha=alpha, use_activate_function=False, alpha_in=alpha_in, tiny=tiny)
    elif m == "rollback_activate":
        return soft_clip_3seg_rollback(ratio, bound_low, bound_high, alpha=alpha, use_activate_function=True,  alpha_in=alpha_in, tiny=tiny)
    elif m == "rollback_plus":
        return soft_clip_3seg_rollback_plus(ratio, bound_low, bound_high, alpha=alpha, gamma=gamma, use_activate_function=False, alpha_in=alpha_in, tiny=tiny)
    elif m == "rollback_activate_plus":
        return soft_clip_3seg_rollback_plus(ratio, bound_low, bound_high, alpha=alpha, gamma=gamma, use_activate_function=True,  alpha_in=alpha_in, tiny=tiny)
    else:
        raise ValueError(f"Unknown soft-clip method: {method}")