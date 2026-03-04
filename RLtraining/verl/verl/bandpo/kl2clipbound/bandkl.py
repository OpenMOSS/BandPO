# -*- coding: utf-8 -*-
# bandkl: 从向前 KL 约束 D_KL(pi_old || pi) = δ 推导出的逐 token 自适应 clip。
# 我们实现“上界根求解器”（两种：保序牛顿 / 纯二分；两域：概率域 / log 域），
# 再用精确镜像公式： l(p) = [1 - (1-p)*u(1-p)] / p  得到下界。
#
# 数值要点：
# - 端点与极端概率的最终处理放在 tokenwise_bounds.py 统一做；
# - 本文件内部仅对"有效子集"求解，避免在不合法点上评估 g/g'；
# - 全向量化、GPU友好，固定迭代步数 + 可选阈值提前停止。

from __future__ import annotations
import torch
import time
import warnings
import math
# ---------- 基础数值原语 ----------

def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """稳定计算 log(1 - exp(x)) for x<=0."""
    LOG_HALF = -0.6931471805599453  # log(0.5)
    return torch.where(
        x < LOG_HALF,
        torch.log1p(-torch.exp(x)),
        torch.log(-torch.expm1(x)),
    )


def _g_gp_prob_domain(
    p: torch.Tensor, x: torch.Tensor,
    TINY_ABS: float, REL_MARGIN: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    概率域稳定计算：
      g(p,x)  = (1-p)[log(1-p) - log(1-px)] - p*log x
      g'(p,x) = p*(x-1)/(x*(1-px))
    """
    dtype = p.dtype

    p = torch.clamp(p, min=torch.finfo(dtype).tiny, max=1.0 - 1e-16)
    x = torch.clamp(x, min=1.0 + 1e-12)

    px = torch.minimum(p * x, torch.tensor(1.0 - REL_MARGIN, dtype=dtype, device=p.device))
    g = (1.0 - p) * (torch.log1p(-p) - torch.log1p(-px)) - p * torch.log(x)

    den = x * (1.0 - px)
    gp = p * (x - 1.0) / torch.clamp(den, min=TINY_ABS)
    return g, gp


def _g_gp_log_domain(
    lp: torch.Tensor, lx: torch.Tensor,
    TINY_ABS: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    log 域稳定计算：
      p=exp(lp), X=exp(lx), s=lp+lx<=0.
      g = (1-p)[log(1-p) - log(1-pX)] - p*log X
        = (1-p)*(log1mexp(lp) - log1mexp(lp+lx)) - p*lx
      g' = ∂g/∂lx = p*(X-1)/(1-pX)
                 = p*expm1(lx) / (1 - exp(lp+lx))
                 = p*expm1(lx) / (-expm1(lp+lx))
    """
    p = torch.exp(lp)
    log1mp = _log1mexp(lp)
    s = lp + lx                      # s<=0 (保证在括号内)
    log1mpx = _log1mexp(s)

    one_minus_p = -torch.expm1(lp)   # 1 - p
    g = one_minus_p * (log1mp - log1mpx) - p * lx

    denom = -torch.expm1(s)
    gp = p * torch.expm1(lx) / torch.clamp(denom, min=TINY_ABS)
    return g, gp


# ---------- 括号构造 ----------

def _upper_bracket_prob(p: torch.Tensor, REL_MARGIN: float, BRACKET_ABS: float) -> tuple[torch.Tensor, torch.Tensor]:
    # x ∈ (1+BRACKET_ABS, (1-REL_MARGIN)/p)
    L = torch.ones_like(p) + BRACKET_ABS
    U = (1.0 - REL_MARGIN) / torch.clamp(p, min=torch.finfo(p.dtype).tiny)
    U = torch.where(U <= L + 1e-15, L * (1.0 + 1e-8), U)
    return L, U


def _upper_bracket_log(lp: torch.Tensor, BRACKET_ABS: float) -> tuple[torch.Tensor, torch.Tensor]:
    # lx ∈ (BRACKET_ABS, -lp - BRACKET_ABS)
    L = torch.full_like(lp, BRACKET_ABS)
    U = -lp - BRACKET_ABS
    U = torch.where(U <= L + 1e-15, L + 1e-8, U)
    return L, U


# ---------- 纯二分（上界根） ----------

def _upper_root_bisection_prob(
    p: torch.Tensor, delta: float,
    max_iter: int, tol: float,
    REL_MARGIN: float, BRACKET_ABS: float, TINY_ABS: float
) -> torch.Tensor:
    dtype = p.dtype
    device = p.device
    L, U = _upper_bracket_prob(p, REL_MARGIN, BRACKET_ABS)
    x = 0.5 * (L + U)
    for _ in range(max_iter):
        M = 0.5 * (L + U)
        gM, _ = _g_gp_prob_domain(p, M, TINY_ABS, REL_MARGIN)
        move_right = gM < delta      # g 在 (1,1/p) 单调递增
        L = torch.where(move_right, M, L)
        U = torch.where(move_right, U, M)
        if torch.max(U - L).item() <= tol:
            break
        x = 0.5 * (L + U)
    return x.to(dtype=device.dtype if hasattr(device, 'dtype') else dtype)


def _upper_root_bisection_log(
    lp: torch.Tensor, delta: float,
    max_iter: int, tol: float,
    BRACKET_ABS: float, TINY_ABS: float
) -> torch.Tensor:
    L, U = _upper_bracket_log(lp, BRACKET_ABS)
    lx = 0.5 * (L + U)
    for _ in range(max_iter):
        M = 0.5 * (L + U)
        gM, _ = _g_gp_log_domain(lp, M, TINY_ABS)
        move_right = gM < delta      # g 在 (0,-lp) 单调递增
        L = torch.where(move_right, M, L)
        U = torch.where(move_right, U, M)
        if torch.max(U - L).item() <= tol:
            break
        lx = 0.5 * (L + U)
    return torch.exp(lx)


# ---------- 保序牛顿（上界根） ----------

def _upper_root_safeguarded_newton_prob(
    p: torch.Tensor, delta: float,
    max_iter: int, tol: float,
    REL_MARGIN: float, BRACKET_ABS: float, TINY_ABS: float
) -> torch.Tensor:
    # 括号
    L, U = _upper_bracket_prob(p, REL_MARGIN, BRACKET_ABS)

    # 初值（概率域二阶近似）
    x0 = 1.0 + torch.sqrt(torch.clamp(2.0 * delta * (1.0 - p) / torch.clamp(p, min=torch.finfo(p.dtype).tiny), min=0.0))
    x = torch.clamp(x0, min=L * (1.0 + 1e-15), max=U * (1.0 - 1e-15))

    for _ in range(max_iter):
        g, gp = _g_gp_prob_domain(p, x, TINY_ABS, REL_MARGIN)

        # 更新括号（单调递增）
        L = torch.where(g < delta, x, L)
        U = torch.where(g >= delta, x, U)

        # 牛顿候选
        step = (g - delta) / torch.clamp(gp, min=torch.finfo(p.dtype).tiny)
        x_new = x - step

        # 仅当 inside 且残差下降才接受，否则用二分点
        inside = (x_new > L) & (x_new < U) & torch.isfinite(x_new)
        g_new, _ = _g_gp_prob_domain(p, torch.where(inside, x_new, 0.5 * (L + U)), TINY_ABS, REL_MARGIN)
        better = torch.abs(g_new - delta) < torch.abs(g - delta)
        x = torch.where(inside & better, x_new, 0.5 * (L + U))

        if torch.max(U - L).item() <= tol:
            break

    return torch.clamp(x, min=L * (1.0 + 1e-15), max=U * (1.0 - 1e-15))


def _upper_root_safeguarded_newton_log(
    lp: torch.Tensor, delta: float,
    max_iter: int, tol: float,
    BRACKET_ABS: float, TINY_ABS: float
) -> torch.Tensor:
    # 括号
    L, U = _upper_bracket_log(lp, BRACKET_ABS)

    # 初值（log 域二阶近似）：lx0 = sqrt(2δ(1-p)/p)
    p = torch.exp(lp)
    lx0 = torch.sqrt(torch.clamp(2.0 * delta * (1.0 - p) / torch.clamp(p, min=torch.finfo(p.dtype).tiny), min=0.0))
    lx = torch.clamp(lx0, min=L * (1.0 + 1e-15), max=U * (1.0 - 1e-15))

    for _ in range(max_iter):
        g, gp = _g_gp_log_domain(lp, lx, TINY_ABS)

        # 更新括号（单调递增）
        L = torch.where(g < delta, lx, L)
        U = torch.where(g >= delta, lx, U)

        # 牛顿候选
        step = (g - delta) / torch.clamp(gp, min=torch.finfo(lp.dtype).tiny)
        lx_new = lx - step

        inside = (lx_new > L) & (lx_new < U) & torch.isfinite(lx_new)
        g_new, _ = _g_gp_log_domain(lp, torch.where(inside, lx_new, 0.5 * (L + U)), TINY_ABS)
        better = torch.abs(g_new - delta) < torch.abs(g - delta)
        lx = torch.where(inside & better, lx_new, 0.5 * (L + U))

        if torch.max(U - L).item() <= tol:
            break

    lx = torch.clamp(lx, min=L * (1.0 + 1e-15), max=U * (1.0 - 1e-15))
    return torch.exp(lx)


# ---------- 镜像：用 u(1-p) 推出 l(p) ----------

def _mirror_lower_from_upper(p: torch.Tensor, u_comp: torch.Tensor) -> torch.Tensor:
    # l(p) = [1 - (1-p)*u(1-p)] / p
    return (1.0 - (1.0 - p) * u_comp) / torch.clamp(p, min=torch.finfo(p.dtype).tiny)

# —— 打印索引及其对应的 p 值（仅前100个）——
def _print_first_ids_and_vals(tag: str, ids: torch.Tensor, x: torch.Tensor, limit: int = 100):
    # ids 可能在 GPU；统一到 CPU 并展开为1D
    ids = ids.reshape(-1).to(torch.long)
    total = ids.numel()
    if total == 0:
        print(f"{tag}: none")
        return
    k = min(total, limit)
    ids_k = ids[:k].detach().cpu()

    # 先把被索引的张量搬到 CPU 再取值，避免跨设备索引
    x_cpu = x.detach().cpu().ravel()
    vals_k = x_cpu[ids_k]

    pairs = ", ".join(f"({int(i)}, {float(v)})" for i, v in zip(ids_k, vals_k))
    print(f"{tag}: count={total}; first_{k} (id, val): [{pairs}]")
# ===== 快速检查 & 修复 cp / lcp 的 NaN 与非有效项 =====
def _cnt(x, name):
    n_nan = torch.isnan(x).sum().item()
    n_posi = torch.isposinf(x).sum().item()
    n_negi = torch.isneginf(x).sum().item()
    print(f"[dbg:{name}] nan={n_nan}, +inf={n_posi}, -inf={n_negi}")
# ---------- 面向上层的 bandkl 计算（仅求解“有效子集”，其余留给上层处理） ----------

@torch.no_grad()
def compute_tokenwise_ratio_bounds_by_bandkl(
    old_log_prob: torch.Tensor,
    *,
    delta: float,
    solve: str = "bisect",          # {"newton","bisect"}
    use_log_domain: bool = True,
    max_iter_bisect: int = 80,
    max_iter_newton: int = 12,
    tol: float = 1e-10,
    TINY_ABS: float = 1e-18,
    REL_MARGIN: float = 1e-12,
    BRACKET_ABS: float = 1e-12,
    MIN_LOGP: float = -700.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    返回 (lower, upper)，与 old_log_prob 同形。注意：
    - 端点/极端概率的最终赋值交由上层统一处理；
    - 本函数在“有效子集”上求解，其它位置输出 NaN，由上层覆盖。
    """
    assert tol>REL_MARGIN
    start = time.perf_counter_ns()
    dtype = old_log_prob.dtype
    device = old_log_prob.device

    # 安全 p, log p
    lp = torch.clamp(old_log_prob, MIN_LOGP, 0.0)
    p = torch.exp(lp)

    # 有效子集：远离端点，避免 0 除或括号退化
    p_eps = torch.tensor(tol, dtype=dtype, device=device)
    # p_eps = torch.tensor(0.1, dtype=dtype, device=device)
    # p_eps = torch.tensor(0.0, dtype=dtype, device=device)
    valid = (p >= p_eps) & (p <= 1.0 - p_eps)
    mask_lo_ori = p < p_eps
    mask_hi_ori = p > 1.0 - p_eps
    n_lo = int(mask_lo_ori.sum().item())
    n_hi = int(mask_hi_ori.sum().item())
    if n_lo or n_hi:
        # warnings.warn(
        #     f"[compute_tokenwise_ratio_bounds_by_bandkl] "
        #     f"clamping {n_lo} probs <= p_eps and {n_hi} probs >= 1-p_eps to interior boundaries."
        # )
        # 压到开区间里：乘一点相对裕度，避免等号导致 valid=False 或括号退化
        p_lo = p_eps              # > p_eps
        p_hi = 1.0 - p_eps        # < 1 - p_eps
        # 写回 p，并保持 dtype/device，一次性克隆避免原地写 alias
        p = p.clone()
        p[mask_lo_ori] = p_lo
        p[mask_hi_ori] = p_hi

        # 同步更新 log p，确保后续 _log1mexp(lp) 与 p 一致
        lp = torch.log(p)
    valid = (p >= p_eps) & (p <= 1.0 - p_eps)
    mask_lo = p < p_eps
    mask_hi = p > 1.0 - p_eps
    n_lo = int(mask_lo.sum().item())
    n_hi = int(mask_hi.sum().item())
    assert n_lo == n_hi == 0

    # --- 初始化输出 ---
    upper = torch.full_like(p, float("nan"))
    lower = torch.full_like(p, float("nan"))

    # --- 只在有效子集上求解上界 u(p) ---
    if valid.any():
        pv = p[valid]
        lpv = lp[valid]

        if use_log_domain:
            if solve == "newton":
                u_valid = _upper_root_safeguarded_newton_log(
                    lpv, delta, max_iter_newton, tol, BRACKET_ABS, TINY_ABS
                )
            else:
                u_valid = _upper_root_bisection_log(
                    lpv, delta, max_iter_bisect, tol, BRACKET_ABS, TINY_ABS
                )
        else:
            if solve == "newton":
                u_valid = _upper_root_safeguarded_newton_prob(
                    pv, delta, max_iter_newton, tol, REL_MARGIN, BRACKET_ABS, TINY_ABS
                )
            else:
                u_valid = _upper_root_bisection_prob(
                    pv, delta, max_iter_bisect, tol, REL_MARGIN, BRACKET_ABS, TINY_ABS
                )
        upper[valid] = u_valid

    # --- 为了镜像求下界：先对 1-p 求上界 u(1-p) ---
    cp = 1.0 - p
    lcp = _log1mexp(lp)               # log(1-p)（稳定）
    # valid_cp = (cp >= p_eps) & (cp <= 1.0 - p_eps)
    valid_cp = valid

    u_comp = torch.full_like(p, float("nan"))
    if valid_cp.any():
        cpv = cp[valid_cp]
        lcpv = lcp[valid_cp]
        if use_log_domain:
            if solve == "newton":
                u_comp_valid = _upper_root_safeguarded_newton_log(
                    lcpv, delta, max_iter_newton, tol, BRACKET_ABS, TINY_ABS
                )
            else:
                u_comp_valid = _upper_root_bisection_log(
                    lcpv, delta, max_iter_bisect, tol, BRACKET_ABS, TINY_ABS
                )
        else:
            if solve == "newton":
                u_comp_valid = _upper_root_safeguarded_newton_prob(
                    cpv, delta, max_iter_newton, tol, REL_MARGIN, BRACKET_ABS, TINY_ABS
                )
            else:
                u_comp_valid = _upper_root_bisection_prob(
                    cpv, delta, max_iter_bisect, tol, REL_MARGIN, BRACKET_ABS, TINY_ABS
                )
        u_comp[valid_cp] = u_comp_valid

    # 镜像得到下界 l(p)
    lower = _mirror_lower_from_upper(p, u_comp)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000  # float 毫秒

    # 端点极限统一处理：p≈0 → (0, upper_bound_max)； p≈1 → (e^(-delta),1),下面三行代码因mask_hi_ori失效而失效
    upper = torch.where(mask_hi_ori, torch.ones_like(upper), upper)
    theory_lower_upbounds = math.exp(-delta)
    lower = torch.where(mask_hi_ori, torch.full_like(lower, theory_lower_upbounds), lower)
    lower = torch.where(mask_lo_ori, torch.zeros_like(lower), lower)
    # print(f"bandkl {solve} elapsed_ms: {elapsed_ms}\n")

    # 展平成一维，保证索引一致
    p_flat = p.reshape(-1)
    lower_flat = lower.reshape(-1)
    # 找出 lower 为 NaN 的索引
    nan_idx = torch.isnan(lower_flat).nonzero(as_tuple=True)[0]
    # 对应的原始 p 值、以及 lower(就是 NaN)
    p_nan = p_flat[nan_idx]
    lower_nan = lower_flat[nan_idx]  # 全是 NaN
    p_nan_tolist = p_nan.tolist()
    if p_nan_tolist:
        print("nan lower的 p 值：", p_nan_tolist)
    else:
        pass

    return lower.to(dtype), upper.to(dtype)
