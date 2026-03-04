# Extended validation: add bisection curves and DCPO (SAC-style) adaptive clip; make a 1x2 subplot figure.
import numpy as np
import matplotlib.pyplot as plt

# ----- Config -----
delta = 0.5
# Standard PPO constant clip
eps_clip_high = 0.28
eps_clip_low  = 0.20
bandkl_upper_bound_max = 100.0
# DCPO parameters (use the same highs/lows for fair comparison)
dcpo_eps_high = eps_clip_high
dcpo_eps_low  = eps_clip_low
dcpo_upper_bound_max = 10.0

# ===== Numerics (robust) =====
TINY_ABS     = 1e-18         # <<< 小分母保护
REL_MARGIN   = 1e-12         # <<< 相对留边，保证 p*x <= 1-REL_MARGIN
BRACKET_ABS  = 1e-12         # <<< 下界 L = 1 + BRACKET_ABS
tol = 1e-12
max_iter_bisect = 80
max_iter_newton = 12
MIN_LOGP = -700.0            # <<< logp -> p 防下溢

# Probability grid (101 points from 0 to 1 inclusive)
p_grid = np.linspace(0.00, 1.0, 101, dtype=np.float64)

# ===== Log-domain primitives (stable) =====
LOG_HALF = -0.6931471805599453  # log(0.5)

def log1mexp(x):
    """
    Stable log(1 - exp(x)) for x <= 0.
    Piecewise:
      if x < log(0.5):  log1p(-exp(x))
      else:             log(-expm1(x))
    """
    return np.where(x < LOG_HALF, np.log1p(-np.exp(x)), np.log(-np.expm1(x)))

def g_and_gprime_log(lp, lx):
    """
    g(lp,lx) = (1-p)[log(1-p) - log(1-p*X)] - p*log X,  with p=exp(lp), X=exp(lx)
    ∂g/∂lx   = p*(X-1)/(1-pX)
    说明：
    - 全程在 log 域里计算，避免 1-p、1-pX 的灾难性消减
    - 使用 log1mexp/expm1/log1p 的稳定写法
    """
    # clamp lp 以避免数值端点 (仍然允许非常小/大的概率)
    lp = np.clip(lp, np.log(1e-300), np.log(1.0 - 1e-16))
    p  = np.exp(lp)
    log1mp  = log1mexp(lp)          # log(1 - p)
    s       = lp + lx               # log(pX)
    log1mpx = log1mexp(s)           # log(1 - pX)
    one_minus_p = -np.expm1(lp)     # 1 - p
    g  = one_minus_p * (log1mp - log1mpx) - p * lx
    denom = -np.expm1(s)            # 1 - pX = -(exp(lp+lx)-1)
    gp = p * np.expm1(lx) / np.maximum(denom, TINY_ABS)  # (X-1)/(1-pX)
    return g, gp

def _upper_bracket_log(lp):
    """
    上界根区间：lx ∈ (0, -lp)
    注：不强行加入相对边界 (1-REL_MARGIN)，避免 p≈1 时括号退化；
       端点仍留有 BRACKET_ABS 安全余量。
    """
    L = np.full_like(lp, BRACKET_ABS)            # just above 0
    U = -lp - BRACKET_ABS                         # just below -lp
    bad = U <= L + 1e-15
    U = np.where(bad, L + 1e-8, U)               # ensure L < U
    return L, U

# ==== 可选：若你的输入是 logp，可用这个函数把 logp 转成安全的 p ====
def safe_p_from_logp(logp):
    lp = np.clip(logp, MIN_LOGP, 0.0)
    p  = np.exp(lp)
    return np.clip(p, 1e-300, 1.0 - 1e-16)

# ===== 稳定版 g 与 g'（绝不在 px>=1 处评估）=====
def g_and_gprime_safe(p, x):
    """
    g(p,x) = (1-p)[log(1-p) - log(1-px)] - p*log(x)
    g'(p,x) = p * (x-1) / (x * (1 - p*x))
    改进点：
      - px = min(p*x, 1-REL_MARGIN)
      - log 用 log1p；x 用下限 1+BRACKET_ABS
      - 分母加 TINY_ABS 保护
    """
    p  = np.clip(p, 1e-300, 1.0 - 1e-16)
    x  = np.maximum(x, 1.0 + BRACKET_ABS)
    px = np.minimum(p * x, 1.0 - REL_MARGIN)
    g  = (1.0 - p) * (np.log1p(-p) - np.log1p(-px)) - p * np.log(x)
    den = x * (1.0 - px)
    gp = p * (x - 1.0) / np.maximum(den, TINY_ABS)
    return g, gp

# ===== 相对安全括号：上界根区间 (1, (1-REL_MARGIN)/p) =====
def _upper_bracket(p):
    L = np.ones_like(p) + BRACKET_ABS
    U = (1.0 - REL_MARGIN) / np.clip(p, 1e-300, 1.0)   # <<< 关键：相对留边
    bad = U <= L + 1e-15
    U = np.where(bad, L * (1.0 + 1e-8), U)             # 确保 L<U
    return L, U

# ===== 二分法（稳健）=====
def upper_root_bisection(p, delta, tol=1e-12, max_iter=80):
    x = np.full_like(p, np.nan, dtype=np.float64)
    valid = (p > 0.0) & (p < 1.0)
    if not np.any(valid):
        # 端点极限返回
        x[p <= 0.0] = bandkl_upper_bound_max           # p=0 上界视为很大（再由y轴截断/或upper_bound_max）
        x[p >= 1.0] = 1.0 + BRACKET_ABS
        return x

    pp = p[valid]
    L, U = _upper_bracket(pp)

    for _ in range(max_iter):
        M = 0.5 * (L + U)
        gM, _ = g_and_gprime_safe(pp, M)
        move_right = gM < delta   # g 在 (1,1/p) 单调递增
        L = np.where(move_right, M, L)
        U = np.where(move_right, U, M)
        if np.max(U - L) <= tol:
            break
    x_valid = 0.5 * (L + U)
    x[valid] = x_valid

    # 端点极限值（确定返回，非 NaN/inf）
    x[p <= 0.0] = bandkl_upper_bound_max
    x[p >= 1.0] = 1.0 + BRACKET_ABS
    return x

# ===== 保序牛顿（从不越界评估，越界即回退二分）=====
def upper_root_safeguarded_newton(p, delta, tol=1e-12, max_iter=12):
    x = np.full_like(p, np.nan, dtype=np.float64)
    valid = (p > 0.0) & (p < 1.0)
    if not np.any(valid):
        x[p <= 0.0] = bandkl_upper_bound_max
        x[p >= 1.0] = 1.0 + BRACKET_ABS
        return x

    pp = p[valid]
    L, U = _upper_bracket(pp)

    # 初值：二阶近似；并剪裁到括号内
    x0 = 1.0 + np.sqrt(2.0 * delta * (1.0 - pp) / np.maximum(pp, 1e-300))
    xv = np.minimum(np.maximum(x0, L * (1.0 + 1e-15)), U * (1.0 - 1e-15))

    for _ in range(max_iter):
        g_val, gp = g_and_gprime_safe(pp, xv)
        # 括号随单调性更新
        L = np.where(g_val < delta, xv, L)
        U = np.where(g_val >= delta, xv, U)

        # 牛顿候选，不立刻评估
        step  = (g_val - delta) / np.where(np.abs(gp) > 0.0, gp, 1e-300)
        x_new = xv - step

        # 只有 inside 才评估 g_new；否则用二分点
        inside = (x_new > L) & (x_new < U) & np.isfinite(x_new)
        x_try  = np.where(inside, x_new, 0.5 * (L + U))
        g_new, _ = g_and_gprime_safe(pp, x_try)

        better = np.abs(g_new - delta) < np.abs(g_val - delta)
        xv = np.where(inside & better, x_new, 0.5 * (L + U))

        if np.max(U - L) < tol:
            break

    xv = np.minimum(np.maximum(xv, L * (1.0 + 1e-15)), U * (1.0 - 1e-15))
    out = np.full_like(p, np.nan, dtype=np.float64)
    out[valid] = xv
    out[p <= 0.0] = bandkl_upper_bound_max
    out[p >= 1.0] = 1.0 + BRACKET_ABS
    return out

# ===== Log-domain bisection (upper root) =====
def upper_root_bisection_log(lp, delta, tol=1e-12, max_iter=80):
    """
    直接用 log p = lp 求解 g(lp, lx) = delta 的上界根 (lx in (0, -lp))，返回 r=exp(lx)。
    单调性：在该区间 g 对 lx 严格递增，二分必收敛。
    """
    out_r = np.full_like(lp, np.nan, dtype=np.float64)
    L, U = _upper_bracket_log(lp)
    for _ in range(max_iter):
        M = 0.5 * (L + U)
        gM, _ = g_and_gprime_log(lp, M)
        move_right = gM < delta      # g increasing on (0, -lp)
        L = np.where(move_right, M, L)
        U = np.where(move_right, U, M)
        if np.max(U - L) <= tol:
            break
    lx = 0.5 * (L + U)
    out_r = np.exp(lx)               # 返回正常比例 r
    return out_r

# ===== Log-domain safeguarded Newton (upper root, recommended) =====
def upper_root_safeguarded_newton_log(lp, delta, tol=1e-12, max_iter=12):
    """
    保序牛顿：在 log 空间迭代，保持 (L,U) 括号，越界或残差不降则回退到二分点。
    初值来自在 lx≈0 处的二阶近似：g ≈ p/(2(1-p)) * lx^2。
    返回 r=exp(lx)。
    """
    # Brackets
    L, U = _upper_bracket_log(lp)

    # 初值（log 域二阶近似）
    p  = np.exp(lp)
    one_minus_p = -np.expm1(lp)
    lx0 = np.sqrt(np.maximum(2.0 * delta * one_minus_p / np.maximum(p, 1e-300), 0.0))
    lx  = np.minimum(np.maximum(lx0, L * (1.0 + 1e-15)), U * (1.0 - 1e-15))

    for _ in range(max_iter):
        g, gp = g_and_gprime_log(lp, lx)

        # 依据单调性更新括号 (g increasing)
        L = np.where(g < delta, lx, L)
        U = np.where(g >= delta, lx, U)

        # Newton candidate
        step  = (g - delta) / np.where(np.abs(gp) > 0.0, gp, 1e-300)
        lx_new = lx - step

        # 仅当在括号内且残差下降才接受，否则回退二分
        inside = (lx_new > L) & (lx_new < U) & np.isfinite(lx_new)
        g_new, _ = g_and_gprime_log(lp, np.where(inside, lx_new, 0.5 * (L + U)))
        better = np.abs(g_new - delta) < np.abs(g - delta)
        lx = np.where(inside & better, lx_new, 0.5 * (L + U))

        if np.max(U - L) <= tol:
            break

    lx = np.minimum(np.maximum(lx, L * (1.0 + 1e-15)), U * (1.0 - 1e-15))
    return np.exp(lx)  # 返回正常比例 r

# ===== 镜像映射下界（精确）=====
def mirror_lower_from_upper(p, u_upper_1_minus_p):
    out = np.full_like(p, np.nan, dtype=np.float64)
    valid = (p > 0.0) & (p < 1.0)
    if np.any(valid):
        pp = p[valid]
        out[valid] = (1.0 - (1.0 - pp) * u_upper_1_minus_p[valid]) / pp
    # 端点极限
    out[p <= 0.0] = 0.0
    out[p >= 1.0] = 1.0 - 1e-12
    return out

# --- Compute bandkl bounds (Newton and Bisection) Mirror lowers---
u_newton = upper_root_safeguarded_newton(p_grid, delta, tol=tol, max_iter=max_iter_newton)
u_newton_comp = upper_root_safeguarded_newton(1.0 - p_grid, delta, tol=tol, max_iter=max_iter_newton)
l_newton = mirror_lower_from_upper(p_grid, u_newton_comp)

u_bisect = upper_root_bisection(p_grid, delta, tol=tol, max_iter=max_iter_bisect)
u_bisect_comp = upper_root_bisection(1.0 - p_grid, delta, tol=tol, max_iter=max_iter_bisect)
l_bisect = mirror_lower_from_upper(p_grid, u_bisect_comp)

# --- Optional: combine direct bounds with epsilon (as before) ---
upper_adj = np.minimum(np.maximum(u_newton, 1.0 + eps_clip_high), bandkl_upper_bound_max)
lower_adj = np.minimum(l_newton, 1.0 - eps_clip_low)

# ===== Log-domain solving: from original p -> logp, then solve in log space, output normal r =====
# 有效区间掩码（避免 log(0/1)）
valid = (p_grid > 0.0) & (p_grid < 1.0)

# 稳定的 logp 与 log(1-p)
lp_grid  = np.zeros_like(p_grid)
lcp_grid = np.zeros_like(p_grid)
lp_grid[valid]  = np.log(np.clip(p_grid[valid], 1e-300, 1.0 - 1e-16))
# log(1-p) 用稳定原语；边界处随后直接给极限值
def _log1mexp_vec(lp_vec):
    return np.where(lp_vec < LOG_HALF, np.log1p(-np.exp(lp_vec)), np.log(-np.expm1(lp_vec)))
lcp_grid[valid] = _log1mexp_vec(lp_grid[valid])
# ---- Upper bounds via log-domain Newton/Bisection ----
u_newton_log_valid  = upper_root_safeguarded_newton_log(lp_grid[valid], delta, tol=tol, max_iter=max_iter_newton)
u_bisect_log_valid  = upper_root_bisection_log(lp_grid[valid],        delta, tol=tol, max_iter=max_iter_bisect)
u_newton_log = np.full_like(p_grid, np.nan, dtype=np.float64)
u_bisect_log = np.full_like(p_grid, np.nan, dtype=np.float64)
u_newton_log[valid] = u_newton_log_valid
u_bisect_log[valid] = u_bisect_log_valid
# 端点极限（与现有实现一致）
u_newton_log[~valid & (p_grid <= 0.0)] = bandkl_upper_bound_max
u_newton_log[~valid & (p_grid >= 1.0)] = 1.0 + BRACKET_ABS
u_bisect_log[~valid & (p_grid <= 0.0)] = bandkl_upper_bound_max
u_bisect_log[~valid & (p_grid >= 1.0)] = 1.0 + BRACKET_ABS
# ---- Lower bounds by exact mirror from u(1-p) ----
# 先算 u(1-p) 的 log 版上界（用 lcp_grid）
u_comp_newton_log_valid = upper_root_safeguarded_newton_log(lcp_grid[valid], delta, tol=tol, max_iter=max_iter_newton)
u_comp_bisect_log_valid = upper_root_bisection_log(lcp_grid[valid],        delta, tol=tol, max_iter=max_iter_bisect)
u_comp_newton_log = np.full_like(p_grid, np.nan, dtype=np.float64)
u_comp_bisect_log = np.full_like(p_grid, np.nan, dtype=np.float64)
u_comp_newton_log[valid] = u_comp_newton_log_valid
u_comp_bisect_log[valid] = u_comp_bisect_log_valid
# 端点极限
u_comp_newton_log[~valid & (p_grid >= 1.0)] = bandkl_upper_bound_max   # 当 (1-p)=0 → 上界无穷大（裁到 upper_bound_max）
u_comp_newton_log[~valid & (p_grid <= 0.0)] = 1.0 + BRACKET_ABS
u_comp_bisect_log[~valid & (p_grid >= 1.0)] = bandkl_upper_bound_max
u_comp_bisect_log[~valid & (p_grid <= 0.0)] = 1.0 + BRACKET_ABS
# 利用精确镜像公式得到下界（正常比例），与现有工具函数 mirror_lower_from_upper 复用
l_newton_log = mirror_lower_from_upper(p_grid, u_comp_newton_log)
l_bisect_log = mirror_lower_from_upper(p_grid, u_comp_bisect_log)
# ---- 数值一致性检查（log 域 vs 原 p 域）----
mask_valid = ~np.isnan(u_newton_log) & ~np.isnan(u_newton)
max_abs_diff_u_newton_log = np.nanmax(np.abs(u_newton_log[mask_valid] - u_newton[mask_valid]))
max_abs_diff_l_newton_log = np.nanmax(np.abs(l_newton_log[mask_valid] - l_newton[mask_valid]))
print(f"[Check] log-Newton vs p-Newton: max |Δu|={max_abs_diff_u_newton_log:.2e}, max |Δl|={max_abs_diff_l_newton_log:.2e}")
mask_valid_b = ~np.isnan(u_bisect_log) & ~np.isnan(u_bisect)
max_abs_diff_u_bisect_log = np.nanmax(np.abs(u_bisect_log[mask_valid_b] - u_bisect[mask_valid_b]))
max_abs_diff_l_bisect_log = np.nanmax(np.abs(l_bisect_log[mask_valid_b] - l_bisect[mask_valid_b]))
print(f"[Check] log-Bisect vs p-Bisect: max |Δu|={max_abs_diff_u_bisect_log:.2e}, max |Δl|={max_abs_diff_l_bisect_log:.2e}")



# Diagnostics: Newton vs Bisection max abs diff (where both valid)
mask_valid_u = ~np.isnan(u_bisect) & ~np.isnan(u_newton)
max_abs_diff = np.nanmax(np.abs(u_bisect[mask_valid_u] - u_newton[mask_valid_u]))

# --- Standard constant clip ---
std_upper = np.full_like(p_grid, 1.0 + eps_clip_high, dtype=np.float64)
std_lower = np.full_like(p_grid, 1.0 - eps_clip_low,  dtype=np.float64)

# --- Theoretical lines: 1/p and 0 ---
with np.errstate(divide='ignore', invalid='ignore'):
    theory_upper = np.where(p_grid > 0.0, 1.0 / p_grid, np.inf)
theory_lower = np.zeros_like(p_grid, dtype=np.float64)

# --- DCPO (SAC-style) adaptive clip ---
with np.errstate(divide='ignore', invalid='ignore'):
    denom = np.maximum(p_grid, 1e-300)   # <<< 防 0 除
    disc_low = np.maximum(1.0 - 4.0 * dcpo_eps_low  / denom, 0.0)  # ensure non-negative
    disc_high= 1.0 + 4.0 * dcpo_eps_high / denom
    r_dcpo_low  = 0.5 + 0.5 * np.sqrt(disc_low)
    r_dcpo_high = 0.5 + 0.5 * np.sqrt(disc_high)
# Apply r_max
r_dcpo_high = np.minimum(r_dcpo_high, dcpo_upper_bound_max)

# ----- Plot: 1 row, 2 columns -----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ---- Subplot (a): ratio bounds vs p ----
ax1.plot(p_grid, u_newton, linestyle='--', color='tab:orange', label="bandkl Upper (Newton)")
ax1.plot(p_grid, l_newton, linestyle='--', color='tab:blue',   label="bandkl Lower (mirror, Newton)")

# Add bisection curves
ax1.plot(p_grid, u_bisect, linestyle='-.', color='tab:orange', label="bandkl Upper (Bisection)")
ax1.plot(p_grid, l_bisect, linestyle='-.', color='tab:blue',   label="bandkl Lower (mirror, Bisection)")

# Adjusted bounds after combining with epsilon (optional visualization)
ax1.plot(p_grid, upper_adj, linestyle='--', color='tab:purple', label="Upper (min/max with 1+ε)")
ax1.plot(p_grid, lower_adj, linestyle='--', color='tab:pink',   label="Lower (min with 1-ε)")

ax1.plot(p_grid, u_newton_log, alpha=0.35, linestyle='-.', color='tab:orange', linewidth=1.0, label="bandkl Upper (Newton, log)")
ax1.plot(p_grid, l_newton_log, alpha=0.35, linestyle='-.', color='tab:blue',   linewidth=1.0, label="bandkl Lower (log mirror)")
ax1.plot(p_grid, u_bisect_log, alpha=0.35, linestyle=':', color='tab:orange', linewidth=1.0, label="bandkl Upper (Bisection, log)")
ax1.plot(p_grid, l_bisect_log, alpha=0.35, linestyle=':', color='tab:blue',   linewidth=1.0, label="bandkl Lower (Bisection, log mirror)")

# DCPO bounds
ax1.plot(p_grid, r_dcpo_high, linestyle='-', color="tab:olive",  alpha=0.7, label="DCPO upper")
ax1.plot(p_grid, r_dcpo_low,  linestyle='-', color="tab:green",  alpha=0.7, label="DCPO lower")

# Standard PPO constant clip
ax1.plot(p_grid, std_upper, linestyle=':', color="tab:gray",  label="Std clip upper (1+ε_high)")
ax1.plot(p_grid, std_lower, linestyle=':', color="tab:brown", label="Std clip lower (1-ε_low)")

# Theoretical lines
ax1.plot(p_grid, theory_upper, linestyle=':', color="tab:cyan", label="Theoretical upper 1/p")
ax1.plot(p_grid, theory_lower, linestyle=':', color="black",    label="Theoretical lower 0")

ax1.set_xlim(0.0, 1.0)
# ax1.set_ylim(0.0, 1.0)
ax1.set_ylim(0.0, 2.0)
# ax1.set_ylim(0.0, 20.0)
# ax1.set_ylim(0.0, bandkl_upper_bound_max*1.2)
ax1.set_xlabel("old probability p")
ax1.set_ylabel("ratio r = π/π_old")
ax1.set_title(f"(a) Ratio bounds vs p (δ={delta})\nbandkl Newton vs Bisection: max |Δ|={max_abs_diff:.2e}")
ax1.legend(fontsize=8, ncol=2)

# ---- Subplot (b): delta_up/down = r*p - p vs p ----
def deltas_from_bounds(upper, lower, p):
    with np.errstate(invalid='ignore'):
        delta_up = upper * p - p
        delta_dn = lower * p - p
    return delta_up, delta_dn

du_newton, dd_newton = deltas_from_bounds(u_newton, l_newton, p_grid)
du_bisect, dd_bisect = deltas_from_bounds(u_bisect, l_bisect, p_grid)
du_adj,  dd_adj      = deltas_from_bounds(upper_adj, lower_adj, p_grid)
du_newton_log, dd_newton_log = deltas_from_bounds(u_newton_log, l_newton_log, p_grid)
du_bisect_log, dd_bisect_log = deltas_from_bounds(u_bisect_log, l_bisect_log, p_grid)
du_dcpo, dd_dcpo     = deltas_from_bounds(r_dcpo_high, r_dcpo_low, p_grid)
du_std,  dd_std      = deltas_from_bounds(std_upper, std_lower, p_grid)
du_theo, dd_theo     = deltas_from_bounds(theory_upper, theory_lower, p_grid)

ax2.plot(p_grid, du_newton, label="bandkl upper (Newton)")
ax2.plot(p_grid, dd_newton, label="bandkl lower (Newton)")
ax2.plot(p_grid, du_bisect, linestyle='--', label="bandkl upper (Bisection)")
ax2.plot(p_grid, dd_bisect, linestyle='--', label="bandkl lower (Bisection)")
ax2.plot(p_grid, du_adj,  linestyle=':', label="Adj upper (min/max with 1+ε)")
ax2.plot(p_grid, dd_adj,  linestyle=':', label="Adj lower (min with 1-ε)")
ax2.plot(p_grid, du_newton_log, alpha=0.35, linestyle='-.', linewidth=1.0, label="bandkl upper (Newton, log)")
ax2.plot(p_grid, dd_newton_log, alpha=0.35, linestyle='-.', linewidth=1.0, label="bandkl lower (log mirror)")
ax2.plot(p_grid, du_bisect_log, alpha=0.35, linestyle=':', linewidth=1.0, label="bandkl upper (Bisection, log)")
ax2.plot(p_grid, dd_bisect_log, alpha=0.35, linestyle=':', linewidth=1.0, label="bandkl lower (Bisection, log mirror)")
ax2.plot(p_grid, du_dcpo, label="DCPO upper")
ax2.plot(p_grid, dd_dcpo, label="DCPO lower")
ax2.plot(p_grid, du_std,  linestyle=':', label="Std clip upper")
ax2.plot(p_grid, dd_std,  linestyle=':', label="Std clip lower")
ax2.plot(p_grid, du_theo, linestyle='-.', label="Theoretical upper (1/p)")
ax2.plot(p_grid, dd_theo, linestyle='-.', label="Theoretical lower (0)")

ax2.set_xlim(0.0, 1.0)
ax2.set_ylim(-1.0, 1.0)
ax2.set_xlabel("old probability p")
ax2.set_ylabel("Δ = r·p - p")
ax2.set_title("(b) Δ-up / Δ-down vs p")
ax2.legend(fontsize=8, ncol=2)

plt.tight_layout()
out_path = "/remote-home1/yli/Workspace/BandPO/RLtraining/verl/verl/bandpo/kl2clipbound/theory/robust_bandkl_bounds_with_dcpo_1x2.png"
plt.savefig(out_path, dpi=160)
plt.show()
