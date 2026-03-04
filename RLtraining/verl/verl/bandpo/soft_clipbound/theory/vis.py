import numpy as np
import matplotlib.pyplot as plt

# --------------------- Tunable parameters ---------------------
high_bound  = 10
low_bound   = 0.8
eps_high = high_bound - 1.0
eps_low  = 1.0 - low_bound
k   = 1.0
g    = 10.0
rho  = 0.1
r_max = high_bound*1.2
# --------------------- Utility: safe hyperbolics ---------------------
def arctanh(x):
    return 0.5 * np.log((1.0 + x) / (1.0 - x))
def find_g_for_reach(e, low_bound, g_init):
    """
    For reach: need T(g) = coth(g) - e / ( g (e - low_bound) ) in (-1, 1).
    We monotonically increase g from g_init until T(g) > -1 + tol.
    """
    S = e / (e - low_bound)
    g = max(float(g_init), 1e-6)
    def T_of(gv): return (np.cosh(gv)/np.sinh(gv)) - (S / gv)
    # If g too small, T << -1. Increase g until feasible.
    tol = 1e-7
    max_g = 1e6
    # First ensure T < 1 (always true for large g since T -> 1^-), we need lower bound
    while T_of(g) <= -1 + tol and g < max_g:
        g *= 1.5  # geometric growth for speed
    return g
def get_split_points(high_bound, low_bound, rho):
    e_high = rho + high_bound*(1-rho) # high_bound - (high_bound-1)*rho
    e_low = rho + low_bound*(1-rho) # low_bound  + (1- low_bound)*rho
    return e_high, e_low
# Method baseline: 
# Tanh Clamp:
#     y(x) = m + 0.5 (b - a) tanh[k (x - m)],  m = (a + b)/2
# 全域 C^∞，两端平滑逼近 a, b。
def soft_clip_1seg(r, k: float = None, high_bound=1.28, low_bound=0.8):
    r = np.asarray(r, dtype=float)
    m = (high_bound+low_bound) / 2
    n = (high_bound - low_bound) / 2
    if not k:
        k=1/n
        print(f"Using default k={k} for tanh_clamp_ratio, which is 2/(high_bound - low_bound) = {2/(high_bound - low_bound)}")
    else:
        print(f"Using user-defined k={k} for tanh_clamp_ratio")
    return m + n * np.tanh(k * (r - m))

# Method A:  (original piecewise; split at r = 1)
#   r >= 1:   y = 1 + eps_high * tanh( k/eps_high * (r - 1) )
#   r <  1:   y = 1 + eps_low  * tanh( k/eps_low  * (r - 1) )
#   — continuous and C^1 at r=1.
def soft_clip_2seg(r, k=1.0, high_bound=1.28, low_bound=0.8):
    eps_high = high_bound - 1.0
    eps_low  = 1.0 - low_bound
    r = np.asarray(r, dtype=float)
    y = np.full_like(r, np.nan, dtype=float)
    mask_hi = r >= 1.0
    mask_lo = r <= 1.0
    # higher segment (r >= 1)
    y[mask_hi] = 1.0 + eps_high * np.tanh( k/eps_high * (r[mask_hi] - 1.0) )
    # lower segment (r <= 1)
    y[mask_lo] = 1.0 + eps_low * np.tanh( k/eps_low * (r[mask_lo] - 1.0) )
    return y
# Method B
# y(x) = a * tanh( b (x - c) ) + d
# Segmentation rule for the New method:
    # r <= e_low : lower side (converge or reach), then y := max(y, 0)
    # e_low < r < e_high : identity y = r
    # r >= e_high  : higher side (converge to high_bound)
def soft_clip_3seg_control_converge(r, high_bound=1.28, low_bound=0.8, rho=0.1, g=1.0, lower_method="converge"):
    e_high, e_low = get_split_points(high_bound, low_bound, rho)
    # constraints for higher bounds:
    # 1. y(e_high)=e_high
    # 2. y'(e_high)=1
    # 3. lim(x->+infinite)y(x) = high_bound
    # parameters: e_high, high_bound, rou, g (control the converging speed) 
    # let e_high = rho + high_bound*(1-rho); E = exp(2g):
    #       a = (high_bound - e_high) * (1 + E) / 2
    #       b = (1 + exp(-2g)) / (2 (high_bound - e_high))
    #       c = e_high - [ 2 g (high_bound - e_high) E / (1 + E) ]
    #       d = high_bound - a
    def get_high_bound_param_converge(high_bound, e_high, g):
        E = np.exp(2.0 * g)
        a = (high_bound - e_high) * (1.0 + E) / 2.0
        b = (1.0 + np.exp(-2.0 * g)) / (2.0 * (high_bound - e_high))
        c = e_high - (2.0 * g * (high_bound - e_high) * E) / (1.0 + E)
        d = high_bound - a
        return a, b, c, d
    # constraints for lower bounds (way 1: converge):
    # 1. y(e_low)=e_low
    # 2. y'(e_low)=1
    # 3. lim(x->-infinite)y(x) = low_bound
    # parameters: e_low, low_bound, rou, g (control the converging speed) 
    # let e_low = rho + low_bound*(1-rho); E = exp(2g):
    #           a = (e_low - low_bound) * (1 + E) / (2E)
    #           b = (1 + E) / (2 (e_low - low_bound))
    #           c = e_low - [ 2 g (e_low - low_bound) / (1 + E) ]
    #           d = low_bound + a
    def get_low_bound_param_converge(low_bound, e_low, g):
        E = np.exp(2.0 * g)
        a = (e_low - low_bound) * (1.0 + E) / (2.0 * E)
        b = (1.0 + E) / (2.0 * (e_low - low_bound))
        c = e_low - (2.0 * g * (e_low - low_bound)) / (1.0 + E)
        d = low_bound + a
        return a, b, c, d
    # constraints for lower bounds (way 2: reach):
    # 1. y(e_low)=e_low
    # 2. y'(e_low)=1
    # 3. y(0) = low_bound
    # parameters: e_low, low_bound, rou, g (control the converging speed)
    # let e_low = e_low = rho + low_bound*(1-rho); T(g) := coth(g) - e_low/[ g (e_low - low_bound) ] ∈ (-1, 1).:
        # u = artanh(T),   
        # a = e_low / [ g (1 - T^2) ] = e_low * cosh(u)^2 / g,
        # b = g / e_low,
        # c = e_low - u / b  = e_low - e_low * u / g,
        # d = e_low - a T.
        # If |T(g)| >= 1 for the requested g, we *increase* g to the minimal g_eff where T(g_eff) ∈ (-1,1) (monotone search). (Clipping T breaks y(0)=low_bound, so we do NOT clip T.)
    def get_low_bound_param_reach(low_bound, e_low, g):
        # Adjust g if needed so that |T| < 1 (no clipping; keep exact constraints).
        g_eff = find_g_for_reach(e_low, low_bound, g)
        coth_g = np.cosh(g_eff)/np.sinh(g_eff)
        T = coth_g - e_low / (g_eff * (e_low - low_bound))  # guaranteed in (-1,1)
        u = arctanh(T)
        a = e_low / ( g_eff * (1.0 - T**2) )     # = e_low * cosh(u)^2 / g_eff
        b = g_eff / e_low
        c = e_low - u / b                        # = e_low - e_low*u/g_eff
        d = e_low - a * T
        return a, b, c, d, g_eff
    r = np.asarray(r, dtype=float)
    y = np.full_like(r, np.nan, dtype=float)
    # Higher side (r >= e_high): converge to high_bound
    aH, bH, cH, dH = get_high_bound_param_converge(high_bound=high_bound, e_high=e_high, g=g)
    # Lower side (r <= e_low)
    if lower_method == "converge":
        aL, bL, cL, dL = get_low_bound_param_converge(low_bound=low_bound, e_low=e_low, g=g)
        g_eff = g  # unchanged
    elif lower_method == "reach":
        aL, bL, cL, dL, g_eff = get_low_bound_param_reach(low_bound=low_bound, e_low=e_low, g=g)
    else:
        raise ValueError("lower_method must be 'converge' or 'reach'.")
    mask_lo  = r <= e_low
    mask_mid = (r > e_low) & (r < e_high)
    mask_hi  = r >= e_high
    # Lower (with max(y,0))
    y_lo = aL * np.tanh( bL * (r[mask_lo] - cL) ) + dL
    y[mask_lo] = np.maximum(0.0, y_lo)
    # Middle: identity
    y[mask_mid] = r[mask_mid]
    # Higher
    y[mask_hi]  = aH * np.tanh( bH * (r[mask_hi] - cH) ) + dH
    # Return y plus split info & (possibly adjusted) g for reach
    return y, (e_high, e_low), g_eff
# Method C: 
# Segmentation rule for the New method:
    # r <= e_low : lower side (converge or reach), then y := max(y, 0)
    # e_low < r < e_high : identity y = r
    # r >= e_high  : higher side (converge to high_bound)

    # Higher:   F_up(r)   = 1 + eps_high * tanh( kappa_h (r-1) + c_h )
    # constraints for higher bounds:
    # 1. y(e_high)=e_high
    # 2. y'(e_high)=1
    # parameters: e_high, high_bound, rou(eps_high = high_bound-1, delta_e_high = e_high-1)
    # let e_high = rho + high_bound*(1-rho); t_h = tanh(kappa_h*delta_e_high + c_h) = delta_e_high/eps_high:
        # then 
            # kappa_h = eps_high / (eps_high^2 - delta_e_high^2),
            # c_h     = artanh(delta_e_high/eps_high) - kappa_h*delta_e_high.
    # Lower:   F_down(r) = 1 + eps_low  * tanh( kappa_l (r-1) + c_l )
    # constraints for lower bounds:
    # 1. y(eps_low)=eps_low
    # 2. y'(eps_low)=1
    # parameters: eps_low, low_bound, rou(eps_low = 1-low_bound, delta_e_low = 1-e_low)
    # let e_low = e_low = rho + low_bound*(1-rho); t_l = tanh(c_l - kappa_l*delta_e_low) = -delta_e_low/eps_low:
        # then 
        #     kappa_l = eps_low  / (eps_low^2  - delta_e_low^2 ),
        #     c_l     = -artanh(delta_e_low/eps_low) + kappa_l*delta_e_low.
def soft_clip_3seg(r, high_bound=1.28, low_bound=0.8, rho=0.1):
    r = np.asarray(r, dtype=float)
    y = np.full_like(r, np.nan, dtype=float)
    eps_high = high_bound - 1.0
    eps_low  = 1.0 - low_bound
    e_high, e_low = get_split_points(high_bound, low_bound, rho)
    delta_e_high = e_high-1
    delta_e_low = 1-e_low
    eps = 1e-12
    delta_e_high = np.clip(delta_e_high, eps, eps_high - eps)
    delta_e_low  = np.clip(delta_e_low,  eps, eps_low  - eps)

    kappa_h = eps_high / (eps_high**2 - delta_e_high**2)
    c_h     = arctanh(delta_e_high/eps_high) - kappa_h*delta_e_high
    kappa_l = eps_low  / (eps_low**2  - delta_e_low**2 )
    c_l     = -arctanh(delta_e_low/eps_low) + kappa_l*delta_e_low

    mask_lo  = r <= e_low
    mask_mid = (r > e_low) & (r < e_high)
    mask_hi  = r >= e_high
    y[mask_lo]  = 1.0 + eps_low  * np.tanh( kappa_l * (r[mask_lo]  - 1.0) + c_l )
    y[mask_mid] = r[mask_mid]
    y[mask_hi]  = 1.0 + eps_high * np.tanh( kappa_h * (r[mask_hi] - 1.0) + c_h )

    return y, (e_high, e_low)
def soft_clip_3seg_rollback(r, high_bound=1.28, low_bound=0.8, alpha=0.1, use_activate_function=False, alpha_in=2):
    # recommanded: alpha is belong to [-1, 1], where the positive values can give rollback force.
    r = np.asarray(r, dtype=float)
    y = np.full_like(r, np.nan, dtype=float)
    eps_high = high_bound - 1.0
    eps_low  = 1.0 - low_bound
    alpha_high=alpha*(high_bound/low_bound)**2
    alpha_low=alpha

    mask_lo  = r <= low_bound
    mask_mid = (r > low_bound) & (r < high_bound)
    mask_hi  = r >= high_bound
    y[mask_lo]  = -alpha_low*np.tanh(r[mask_lo]-1) + low_bound - alpha_low * np.tanh(eps_low)
    y[mask_mid] = r[mask_mid]
    y[mask_hi]  = -alpha_high*np.tanh(r[mask_hi]-1) + high_bound + alpha_high * np.tanh(eps_high)
    if use_activate_function:
        mask_mid_low  = (r > low_bound) & (r <= 1)
        mask_mid_high = (r > 1) & (r < high_bound)
        y[mask_mid_low] = 1 + (eps_low / np.tanh(alpha_in)) * np.tanh((alpha_in/eps_low) * (r[mask_mid_low]-1))
        y[mask_mid_high] = 1 + (eps_high / np.tanh(alpha_in)) * np.tanh((alpha_in/eps_high) * (r[mask_mid_high]-1))
    return y
def soft_clip_3seg_rollback_plus(r, high_bound=1.28, low_bound=0.8, alpha=0.1, gamma=0.1, use_activate_function=False, alpha_in=2):
    # recommanded: alpha is belong to [-1, 1], where the positive values can give rollback force.
    r = np.asarray(r, dtype=float)
    y = np.full_like(r, np.nan, dtype=float)
    eps_high = high_bound - 1.0
    eps_low  = 1.0 - low_bound
    alpha_high=alpha*(eps_high/eps_low)
    alpha_low=alpha

    mask_lo  = r <= low_bound
    mask_mid = (r > low_bound) & (r < high_bound)
    mask_hi  = r >= high_bound
    y[mask_lo]  = -alpha_low*np.tanh(gamma*(r[mask_lo]-low_bound)) + low_bound
    y[mask_mid] = r[mask_mid]
    y[mask_hi]  = -alpha_high*np.tanh(gamma*(r[mask_hi]-high_bound)) + high_bound
    if use_activate_function:
        mask_mid_low  = (r > low_bound) & (r <= 1)
        mask_mid_high = (r > 1) & (r < high_bound)
        y[mask_mid_low] = 1 + (eps_low / np.tanh(alpha_in)) * np.tanh((alpha_in/eps_low) * (r[mask_mid_low]-1))
        y[mask_mid_high] = 1 + (eps_high / np.tanh(alpha_in)) * np.tanh((alpha_in/eps_high) * (r[mask_mid_high]-1))
    return y

r = np.linspace(0.00, r_max, (int(r_max*100)+1), dtype=np.float64)

# Evaluate all four methods (single y per method)
y = r
y_soft_clip_1seg = soft_clip_1seg(r, k, high_bound, low_bound)
y_soft_clip_2seg = soft_clip_2seg(r, k, high_bound, low_bound)
y_soft_clip_3seg_control_converge_lower_by_converge, (e_high_1, e_low_1), g_eff = soft_clip_3seg_control_converge(r, high_bound, low_bound, rho, g, lower_method="converge")
y_soft_clip_3seg_control_converge_lower_by_reach, (e_high_2, e_low_2), g_eff    = soft_clip_3seg_control_converge(r, high_bound, low_bound, rho, g, lower_method="reach")
y_soft_clip_3seg, (e_high_3, e_low_3) = soft_clip_3seg(r, high_bound, low_bound, rho)
# y_soft_clip_3seg_rollback = soft_clip_3seg_rollback(r, high_bound, low_bound, alpha=0.1, use_activate_function=False, alpha_in=2)
# y_soft_clip_3seg_rollback_activate = soft_clip_3seg_rollback(r, high_bound, low_bound, alpha=0.1, use_activate_function=True, alpha_in=2)
y_soft_clip_3seg_rollback = soft_clip_3seg_rollback_plus(r, high_bound, low_bound, alpha=0.1, gamma=0.1, use_activate_function=False, alpha_in=2)
y_soft_clip_3seg_rollback_activate = soft_clip_3seg_rollback_plus(r, high_bound, low_bound, alpha=0.1, gamma=0.1, use_activate_function=True, alpha_in=2)

# --------------------- Plot (4 legends only) ----------------------
plt.figure(figsize=(10.5, 6.2))
plt.plot(r, y,                                                   label=f"y=x", alpha=0.4)
plt.plot(r, y_soft_clip_1seg,                                    label=f"soft_clip_1seg")
plt.plot(r, y_soft_clip_2seg,                                    label=f"soft_clip_2seg")
plt.plot(r, y_soft_clip_3seg_control_converge_lower_by_converge, label=f"soft_clip_3seg_converge", linestyle="--")
plt.plot(r, y_soft_clip_3seg_control_converge_lower_by_reach,    label=f"soft_clip_3seg_reach", linestyle=":")
plt.plot(r, y_soft_clip_3seg,                                    label=f"soft_clip_3seg", alpha=0.4)
plt.plot(r, y_soft_clip_3seg_rollback,                           label=f"y_soft_clip_3seg_rollback")
plt.plot(r, y_soft_clip_3seg_rollback_activate,                  label=f"y_soft_clip_3seg_rollback_activate")

# Reference/split lines
plt.axhline(high_bound, linestyle="--", linewidth=1, label="Higher bound high_bound")
plt.axhline(low_bound, linestyle="--", linewidth=1, label="Lower bound low_bound")
plt.axvline(1.0, linestyle=":", linewidth=1, label="Baseline split r=1")
plt.axvline(e_low_1, linestyle=":", linewidth=1, label="split e_low")
plt.axvline(e_high_1,  linestyle=":", linewidth=1, label="split e_high")
plt.axvline(high_bound, linestyle=":", linewidth=1, label="split high_bound")
plt.axvline(low_bound,  linestyle=":", linewidth=1, label="split low_bound")
# print(f"e_low_1:{e_low_1},e_high_1:{e_high_1},e_low_2:{e_low_2},e_high_2:{e_high_2},e_high_3:{e_high_3},e_low_3:{e_low_3}.")
print(f"Recommanded Method: soft_clip_3seg")
plt.xlim(0, r_max)
# plt.xlim(0, 1)
plt.ylim(0, 1.2*high_bound)
# plt.ylim(0, 1)
plt.xlabel("r")
plt.ylabel("soft clipped r")
plt.title(f"soft clipping methods(k:{k}, rho:{rho}, (g,g_eff): ({g},{g_eff}))")
plt.legend(ncol=2, frameon=True)
plt.grid(True, linestyle=":")

# Save & show
plt.savefig("/remote-home1/yli/Workspace/BandPO/RLtraining/verl/verl/bandpo/soft_clipbound/theory/vis.png", dpi=160)
plt.show()
