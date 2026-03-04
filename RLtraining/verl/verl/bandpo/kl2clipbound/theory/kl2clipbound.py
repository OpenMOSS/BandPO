# Extended validation: add bisection curves and DCPO (SAC-style) adaptive clip; make a 1x2 subplot figure.
import numpy as np
import matplotlib.pyplot as plt

# ----- Config -----
delta = 1
# Standard PPO constant clip
eps_clip_high = 0.28
eps_clip_low  = 0.20
bandkl_upper_bound_max = 100.0
# DCPO parameters (use the same highs/lows for fair comparison)
dcpo_eps_high = eps_clip_high
dcpo_eps_low  = eps_clip_low
dcpo_upper_bound_max = 10.0

# Numerics
p_min = 1e-12
p_max = 1 - 1e-12
tol = 1e-12
max_iter_bisect = 80
max_iter_newton = 12

# Probability grid (101 points from 0 to 1 inclusive)
p_grid = np.linspace(0.00, 1.0, 101, dtype=np.float64)

def g_and_gprime(p, x):
    """
    g(p,x) = (1-p)[log(1-p) - log(1-px)] - p*log(x)
    g'(p,x) = p * (x-1) / (x * (1 - p*x))
    Use log1p for stability.
    """
    g = (1.0 - p) * (np.log1p(-p) - np.log1p(-p * x)) - p * np.log(x)
    gp = p * (x - 1.0) / (x * (1.0 - p * x))
    return g, gp

# def upper_root_bisection(p, delta, tol=1e-12, max_iter=80):
#     """
#     Solve g(p,x)=delta for the upper root x in (1, 1/p) by bisection.
#     Returns NaN for p outside (0,1).
#     """
#     x = np.full_like(p, np.nan, dtype=np.float64)
#     valid = (p > 0.0) & (p < 1.0)
#     if not np.any(valid):
#         return x
#     pp = np.clip(p[valid], p_min, p_max)
#     L = np.ones_like(pp) + 1e-12
#     U = 1.0 / pp - 1e-12

#     # Bisection loop
#     for _ in range(max_iter):
#         M = 0.5 * (L + U)
#         gM, _ = g_and_gprime(pp, M)
#         # g increasing on (1, 1/p): if g(M) < delta, root is to the right
#         move_right = gM < delta
#         L = np.where(move_right, M, L)
#         U = np.where(move_right, U, M)
#         if np.max(U - L) <= tol:
#             break
#     x_valid = 0.5 * (L + U)
#     x[valid] = x_valid
#     return x

# def upper_root_safeguarded_newton(p, delta, tol=1e-12, max_iter=12):
#     """
#     Solve g(p,x)=delta for the upper root x in (1, 1/p) with safeguarded Newton.
#     Keeps a bracket [L,U] and only accepts Newton steps that stay in (L,U) AND improve residual; otherwise bisection.
#     """
#     x = np.full_like(p, np.nan, dtype=np.float64)
#     valid = (p > 0.0) & (p < 1.0)
#     if not np.any(valid):
#         return x

#     pp = np.clip(p[valid], p_min, p_max)
#     L = np.ones_like(pp) + 1e-12
#     U = 1.0 / pp - 1e-12

#     # Initial guess from quadratic expansion: x0 = 1 + sqrt(2*delta*(1-p)/p), clamped to (L,U)
#     x0 = 1.0 + np.sqrt(2.0 * delta * (1.0 - pp) / pp)
#     x0 = np.minimum(np.maximum(x0, L + 1e-15), U - 1e-15)
#     xv = x0

#     for _ in range(max_iter):
#         g_val, gp = g_and_gprime(pp, xv)
#         step = (g_val - delta) / np.where(np.abs(gp) > 0.0, gp, 1e-300)
#         x_new = xv - step

#         # Update bracket by monotonicity (g increasing on (1,1/p))
#         L = np.where(g_val < delta, xv, L)
#         U = np.where(g_val >= delta, xv, U)

#         # Check acceptability: inside (L,U) and reduces residual
#         g_new, _ = g_and_gprime(pp, x_new)
#         better = np.abs(g_new - delta) < np.abs(g_val - delta)
#         inside = (x_new > L) & (x_new < U) & np.isfinite(x_new)
#         accept = inside & better

#         # Fall back to bisection if Newton is not acceptable
#         x_bisect = 0.5 * (L + U)
#         xv = np.where(accept, x_new, x_bisect)

#         # Convergence (optional early stop)
#         if np.max(np.abs(g_val - delta)) < tol and np.max(U - L) < tol:
#             break

#     xv = np.minimum(np.maximum(xv, L + 1e-15), U - 1e-15)
#     out = np.full_like(p, np.nan, dtype=np.float64)
#     out[valid] = xv
#     return out

def upper_root_bisection(p, delta, tol=1e-12, max_iter=80):
    """
    Solve g(p,x)=delta for the upper root x in (1, 1/p) by bisection.

    额外规则（按你的要求）：
      - 若 p < p_min      -> 直接返回 bandkl_upper_bound_max
      - 若 p > p_max      -> 直接返回 1.0
      - 其它 p 正常计算
    """
    p = np.asarray(p, dtype=np.float64)
    x = np.full_like(p, np.nan, dtype=np.float64)

    # 快捷返回
    small = p < p_min
    large = p > p_max
    x[small] = bandkl_upper_bound_max
    x[large] = 1.0

    # 剩余需要计算的 p（仍要求在 (0,1) 内）
    mid = (~small) & (~large) & (p > 0.0) & (p < 1.0)
    if not np.any(mid):
        return x

    pp = p[mid]
    L = np.ones_like(pp) + 1e-12
    U = 1.0 / pp - 1e-12

    # Bisection loop（g 在 (1,1/p) 上单调递增）
    for _ in range(max_iter):
        M = 0.5 * (L + U)
        gM, _ = g_and_gprime(pp, M)
        move_right = gM < delta
        L = np.where(move_right, M, L)
        U = np.where(move_right, U, M)
        if np.max(U - L) <= tol:
            break

    x[mid] = 0.5 * (L + U)
    return x

def upper_root_safeguarded_newton(p, delta, tol=1e-12, max_iter=12):
    """
    Solve g(p,x)=delta for the upper root x in (1, 1/p) with safeguarded Newton.
    在牛顿步越界或不改进时回退二分。

    额外规则（按你的要求）：
      - 若 p < p_min      -> 直接返回 bandkl_upper_bound_max
      - 若 p > p_max      -> 直接返回 1.0
      - 其它 p 正常计算
    """
    p = np.asarray(p, dtype=np.float64)
    x = np.full_like(p, np.nan, dtype=np.float64)

    # 快捷返回
    small = p < p_min
    large = p > p_max
    x[small] = bandkl_upper_bound_max
    x[large] = 1.0

    # 剩余需要计算的 p（仍要求在 (0,1) 内）
    mid = (~small) & (~large) & (p > 0.0) & (p < 1.0)
    if not np.any(mid):
        return x

    pp = p[mid]
    L = np.ones_like(pp) + 1e-12
    U = 1.0 / pp - 1e-12

    # 初值（二阶近似），并夹在 (L,U) 内
    x0 = 1.0 + np.sqrt(2.0 * delta * (1.0 - pp) / pp)
    x0 = np.minimum(np.maximum(x0, L + 1e-15), U - 1e-15)
    xv = x0

    for _ in range(max_iter):
        g_val, gp = g_and_gprime(pp, xv)
        step = (g_val - delta) / np.where(np.abs(gp) > 0.0, gp, 1e-300)
        x_new = xv - step

        # 利用单调性更新括号
        L = np.where(g_val < delta, xv, L)
        U = np.where(g_val >= delta, xv, U)

        # 接受准则：在 (L,U) 内且残差更小，否则用二分
        g_new, _ = g_and_gprime(pp, x_new)
        better = np.abs(g_new - delta) < np.abs(g_val - delta)
        inside = (x_new > L) & (x_new < U) & np.isfinite(x_new)
        accept = inside & better

        xv = np.where(accept, x_new, 0.5 * (L + U))

        if np.max(np.abs(g_val - delta)) < tol and np.max(U - L) < tol:
            break

    xv = np.minimum(np.maximum(xv, L + 1e-15), U - 1e-15)
    x[mid] = xv
    return x

def mirror_lower_from_upper(p, u_upper_1_minus_p):
    """
    Mirror mapping (exact):
    l(p) = [1 - (1-p) * u(1-p)] / p
    For p in (0,1); for p==0 or p==1, returns NaN.
    """
    out = np.full_like(p, np.nan, dtype=np.float64)
    valid = (p > 0.0) & (p < 1.0)
    if not np.any(valid):
        return out
    pp = p[valid]
    out[valid] = (1.0 - (1.0 - pp) * u_upper_1_minus_p[valid]) / pp
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

# Diagnostics: Newton vs Bisection max abs diff (where both valid)
# print(np.isnan(u_newton))
# print(np.isnan(u_bisect))
mask_valid_u = ~np.isnan(u_bisect) & ~np.isnan(u_newton)
max_abs_diff = np.nanmax(np.abs(u_bisect[mask_valid_u] - u_newton[mask_valid_u]))

# --- Standard constant clip ---
std_upper = np.full_like(p_grid, 1.0 + eps_clip_high, dtype=np.float64)
std_lower = np.full_like(p_grid, 1.0 - eps_clip_low,  dtype=np.float64)

# --- Theoretical lines: 1/p and 0 ---
with np.errstate(divide='ignore', invalid='ignore'):
    theory_upper = np.where(p_grid > 0.0, 1.0 / p_grid, np.nan)
theory_lower = np.zeros_like(p_grid, dtype=np.float64)

# --- DCPO (SAC-style) adaptive clip ---
with np.errstate(divide='ignore', invalid='ignore'):
    disc_low = np.maximum(1.0 - 4.0 * dcpo_eps_low / p_grid, 0.0)  # ensure non-negative
    disc_high = 1.0 + 4.0 * dcpo_eps_high / p_grid
    r_dcpo_low = 0.5 + 0.5 * np.sqrt(disc_low)
    r_dcpo_high = 0.5 + 0.5 * np.sqrt(disc_high)
# Apply r_max
r_dcpo_high = np.minimum(r_dcpo_high, dcpo_upper_bound_max)



# ----- Plot: 1 row, 2 columns -----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ---- Subplot (a): ratio bounds vs p ----
ax1.plot(p_grid, u_newton, linestyle='--', color='tab:orange', label="bandkl Upper (Newton)")
ax1.plot(p_grid, l_newton, linestyle='--', color='tab:blue', label="bandkl Lower (mirror, Newton)")

# Add bisection curves
ax1.plot(p_grid, u_bisect, linestyle='-.', color='tab:orange', label="bandkl Upper (Bisection)")
ax1.plot(p_grid, l_bisect, linestyle='-.', color='tab:blue', label="bandkl Lower (mirror, Bisection)")

# Adjusted bounds after combining with epsilon (optional visualization)
ax1.plot(p_grid, upper_adj, linestyle='--', color='tab:purple', label="Upper (min/max with 1+ε)")
ax1.plot(p_grid, lower_adj, linestyle='--', color='tab:pink', label="Lower (min with 1-ε)")

# DCPO bounds
ax1.plot(p_grid, r_dcpo_high, linestyle='-', color="tab:olive", alpha=0.7, label="DCPO upper")
ax1.plot(p_grid, r_dcpo_low,  linestyle='-', color="tab:green", alpha=0.7, label="DCPO lower")

# Standard PPO constant clip
ax1.plot(p_grid, std_upper, linestyle=':', color="tab:gray", label="Std clip upper (1+ε_high)")
ax1.plot(p_grid, std_lower, linestyle=':', color="tab:brown", label="Std clip lower (1-ε_low)")

# Theoretical lines
ax1.plot(p_grid, theory_upper, linestyle=':', color="tab:cyan", label="Theoretical upper 1/p")
ax1.plot(p_grid, theory_lower, linestyle=':', color="black", label="Theoretical lower 0")

ax1.set_xlim(0.0, 1.0)
# ax1.set_ylim(0.0, bandkl_upper_bound_max*2)
# ax1.set_ylim(0.0, 20)
ax1.set_ylim(0.0, 2)
ax1.set_xlabel("old probability p")
ax1.set_ylabel("ratio r = π/π_old")
ax1.set_title(f"(a) Ratio bounds vs p (δ={delta})\nbandkl Newton vs Bisection: max |Δ|={max_abs_diff:.2e}")
ax1.legend(fontsize=8, ncol=2)

# ---- Subplot (b): delta_up/down = r*p - p vs p ----
# Prepare helper to compute delta arrays safely
def deltas_from_bounds(upper, lower, p):
    with np.errstate(invalid='ignore'):
        delta_up = upper * p - p
        delta_dn = lower * p - p
    return delta_up, delta_dn

# Compute deltas for each method
du_newton, dd_newton = deltas_from_bounds(u_newton, l_newton, p_grid)
du_bisect, dd_bisect = deltas_from_bounds(u_bisect, l_bisect, p_grid)
du_adj, dd_adj = deltas_from_bounds(upper_adj, lower_adj, p_grid)
du_dcpo,  dd_dcpo  = deltas_from_bounds(r_dcpo_high, r_dcpo_low, p_grid)
du_std,   dd_std   = deltas_from_bounds(std_upper, std_lower, p_grid)
du_theo,  dd_theo  = deltas_from_bounds(theory_upper, theory_lower, p_grid)

# Plot deltas
ax2.plot(p_grid, du_newton, label="bandkl upper (Newton)")
ax2.plot(p_grid, dd_newton, label="bandkl lower (Newton)")
ax2.plot(p_grid, du_bisect, linestyle='--', label="bandkl upper (Bisection)")
ax2.plot(p_grid, dd_bisect, linestyle='--', label="bandkl lower (Bisection)")
ax2.plot(p_grid, du_adj,  linestyle=':', label="Adj upper (min/max with 1+ε)")
ax2.plot(p_grid, dd_adj,  linestyle=':', label="Adj lower (min with 1-ε)")

ax2.plot(p_grid, du_dcpo,  label="DCPO upper")
ax2.plot(p_grid, dd_dcpo,  label="DCPO lower")

ax2.plot(p_grid, du_std, linestyle=':', label="Std clip upper")
ax2.plot(p_grid, dd_std, linestyle=':', label="Std clip lower")

ax2.plot(p_grid, du_theo, linestyle='-.', label="Theoretical upper (1/p)")
ax2.plot(p_grid, dd_theo, linestyle='-.', label="Theoretical lower (0)")

ax2.set_xlim(0.0, 1.0)
ax2.set_ylim(-1.0, 1.0)
ax2.set_xlabel("old probability p")
ax2.set_ylabel("Δ = r·p - p")
ax2.set_title("(b) Δ-up / Δ-down vs p")
ax2.legend(fontsize=8, ncol=2)

plt.tight_layout()
out_path = "/remote-home1/yli/Workspace/BandPO/RLtraining/verl/verl/bandpo/kl2clipbound/theory/bandkl_bounds_with_dcpo_1x2.png"
plt.savefig(out_path, dpi=160)
plt.show()
