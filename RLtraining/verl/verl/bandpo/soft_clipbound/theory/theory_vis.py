# 分段绘制（只画各自片段）——把 4 个方法都按“分段”的方式显示；
# - 老板的（初版）：以 r=1 为分界线（两段）。
# - 新版（converge / reach）：以 e_low、e_up 为分界线（三段，且中段 y=r）。
# - 三段式 tanh：以 a=1-δ_low、b=1+δ_high 为分界线（三段，且中段 y=r）。
#
# 依然只用 matplotlib，且不指定颜色；用线型区分中段（y=r）。
#
# ---------------- 数学公式（注释备查） ----------------
# 统一记号：f=1.28, h=0.8；ε_high=f-1, ε_low=1-h。
#
# [老板的/初版] piecewise：
#   r >= 1:  y = 1 + ε_high * tanh( k (r-1) )
#   r <  1:  y = 1 + ε_low  * tanh( (ε_high/ε_low) k (r-1) )
#
# [新版] 母式 y = a*tanh( b(x-c) ) + d，满足 y(e)=e, y'(e)=1；
#   上半（收敛到 f）：令 e_up = ρ + f(1-ρ)，E=e^{2g}
#       a = (f - e)*(1+E)/2
#       b = (1 + e^{-2g}) / (2 (f - e))
#       c = e - [ 2g (f-e) E / (1+E) ]
#       d = f - a
#   下半-converge（收敛到 h）：令 e_low = ρ + h(1-ρ)，E=e^{2g}
#       a = (e - h)*(1+E)/(2E)
#       b = (1+E)/(2 (e - h))
#       c = e - [ 2g (e-h) / (1+E) ]
#       d = h + a
#   下半-reach（经 (0,h) ）：设 T = coth(g) - e/[ g (e-h) ], u = artanh(T)
#       b = g / e
#       a = e / [ g (1 - T^2) ]
#       c = e - u / b = e - e u / g
#       d = e - a T
#   新版分段绘制规则：
#       r <= e_low  用“下半”（converge 或 reach）；
#       e_low < r < e_up  用 y=r；
#       r >= e_up   用“上半”（收敛到 f）。
#   备注：下半最终与 0 取 max：y = max(0, y_lower)。
#
# [三段式 tanh（闭式）]（你提供的方案）：
#   设 δ_high ∈ (0, ε_high), δ_low ∈ (0, ε_low)，连接点 b=1+δ_high, a=1-δ_low；
#   F_up(r)   = 1 + ε_high * tanh( κ_h (r-1) + c_h )
#   F_down(r) = 1 + ε_low  * tanh( κ_l (r-1) + c_l )
#   条件：F_up(b)=b, F'_up(b)=1, F_down(a)=a, F'_down(a)=1
#   闭式：
#       κ_h = ε_high / (ε_high^2 - δ_high^2),
#       c_h = artanh(δ_high/ε_high) - κ_h δ_high,
#       κ_l = ε_low  / (ε_low^2  - δ_low^2 ),
#       c_l = -artanh(δ_low/ε_low) + κ_l δ_low.
#   三段合成：
#       r <= a 用 F_down； a < r < b 用 y=r； r >= b 用 F_up。
# ----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# --------------------- Tunable parameters ---------------------
f = 1.28
h = 0.8

# Baseline parameters
k = 1.0

# New method parameters
g = 1.0
rho = 0.1  # e_low = rho + h*(1-rho), e_up = rho + f*(1-rho)

# Three-segment parameters (as ratios; must satisfy delta < epsilon)
delta_high_ratio = 0.5  # delta_high = ratio * (f-1)
delta_low_ratio  = 0.5  # delta_low  = ratio * (1-h)
# -------------------------------------------------------------

# --------------------- Formulas (comments) ---------------------
# Baseline (your original continuous & differentiable piecewise):
#   eps_high = f - 1, eps_low = 1 - h
#   r >= 1:  y = 1 + eps_high * tanh( k * (r - 1) )
#   r <  1:  y = 1 + eps_low  * tanh( (eps_high/eps_low) * k * (r - 1) )
#
# New (unified y = a*tanh(b(x-c)) + d; enforce y(e)=e, y'(e)=1):
#   Upper side (asymptote to f): let e_up = rho + f*(1-rho), E = exp(2g):
#       a = (f - e)*(1 + E) / 2
#       b = (1 + exp(-2g)) / (2*(f - e))
#       c = e - [2*g*(f - e)*E / (1 + E)]
#       d = f - a
#   Lower-converge (asymptote to h): let e_low = rho + h*(1-rho), E = exp(2g):
#       a = (e - h)*(1 + E) / (2*E)
#       b = (1 + E) / (2*(e - h))
#       c = e - [2*g*(e - h) / (1 + E)]
#       d = h + a
#   Lower-reach (also passes (0,h)): define T = coth(g) - e / [ g*(e - h) ], u = artanh(T)
#       b = g / e
#       a = e / [ g*(1 - T^2) ]    (= e*cosh(u)^2 / g)
#       c = e - u / b              (= e - e*u/g)
#       d = e - a*T
#   New segmentation rule:
#       r <= e_low: use lower side (converge or reach)
#       e_low < r < e_up: use identity y=r
#       r >= e_up: use upper side (converge to f)
#   Note: lower side finally uses max(y, 0).
#
# Three-segment tanh (closed-form, your design). Define eps_high=f-1, eps_low=1-h.
#   Pick deltas: 0 < delta_high < eps_high, 0 < delta_low < eps_low.
#   Connection points: a = 1 - delta_low, b = 1 + delta_high.
#   Upper segment:   F_up(r)   = 1 + eps_high * tanh( kappa_h*(r-1) + c_h )
#   Lower segment:   F_down(r) = 1 + eps_low  * tanh( kappa_l*(r-1) + c_l )
#   Conditions: F_up(b)=b, F'_up(b)=1; F_down(a)=a, F'_down(a)=1.
#   Closed form:
#       kappa_h = eps_high / (eps_high^2 - delta_high^2)
#       c_h     = artanh(delta_high/eps_high) - kappa_h*delta_high
#       kappa_l = eps_low  / (eps_low^2  - delta_low^2 )
#       c_l     = -artanh(delta_low/eps_low) + kappa_l*delta_low
#   Composition:
#       r <= a: use F_down;  a < r < b: use y=r;  r >= b: use F_up.
# ---------------------------------------------------------------

# --------------------- Helper functions ---------------------
def initial_segments(r, k=1.0, f=1.28, h=0.8):
    eps_high = f - 1.0
    eps_low  = 1.0 - h
    r = np.asarray(r)

    mask_lo = r <= 1.0
    mask_hi = r >= 1.0

    y_lo = 1.0 + eps_low  * np.tanh( (eps_high/eps_low) * k * (r[mask_lo] - 1.0) )
    y_hi = 1.0 + eps_high * np.tanh( k * (r[mask_hi] - 1.0) )

    return {
        "lo": (r[mask_lo], y_lo),
        "hi": (r[mask_hi], y_hi),
        "split": 1.0
    }

def params_upper_converge(f, e, g):
    E = np.exp(2.0 * g)
    a = (f - e) * (1.0 + E) / 2.0
    b = (1.0 + np.exp(-2.0 * g)) / (2.0 * (f - e))
    c = e - (2.0 * g * (f - e) * E) / (1.0 + E)
    d = f - a
    return a, b, c, d

def params_lower_converge(h, e, g):
    E = np.exp(2.0 * g)
    a = (e - h) * (1.0 + E) / (2.0 * E)
    b = (1.0 + E) / (2.0 * (e - h))
    c = e - (2.0 * g * (e - h)) / (1.0 + E)
    d = h + a
    return a, b, c, d

def params_lower_reach(h, e, g):
    coth_g = np.cosh(g) / np.power(np.sinh(g), 1)
    T = coth_g - e / (g * (e - h))
    T = np.clip(T, -0.999999, 0.999999)  # stabilize
    u = 0.5 * np.log((1.0 + T) / (1.0 - T))  # artanh(T)
    b = g / e
    a = e / ( g * (1.0 - T**2) )
    c = e - u / b
    d = e - a * T
    return a, b, c, d

def new_segments(r, g=1.0, rho=0.1, f=1.28, h=0.8, lower_method="converge"):
    r = np.asarray(r)
    e_low = rho + h * (1.0 - rho)
    e_up  = rho + f * (1.0 - rho)

    aU, bU, cU, dU = params_upper_converge(f=f, e=e_up, g=g)
    if lower_method == "converge":
        aL, bL, cL, dL = params_lower_converge(h=h, e=e_low, g=g)
    elif lower_method == "reach":
        aL, bL, cL, dL = params_lower_reach(h=h, e=e_low, g=g)
    else:
        raise ValueError("lower_method must be 'converge' or 'reach'.")

    mask_lo  = r <= e_low
    mask_mid = (r > e_low) & (r < e_up)
    mask_hi  = r >= e_up

    y_lo_raw = aL * np.tanh( bL * (r[mask_lo] - cL) ) + dL
    y_lo = np.maximum(0.0, y_lo_raw)
    y_mid = r[mask_mid]
    y_hi  = aU * np.tanh( bU * (r[mask_hi] - cU) ) + dU

    return {
        "lo":  (r[mask_lo],  y_lo),
        "mid": (r[mask_mid], y_mid),
        "hi":  (r[mask_hi],  y_hi),
        "splits": (e_low, e_up)
    }

def three_segment_params(f=1.28, h=0.8, delta_high_ratio=0.5, delta_low_ratio=0.5):
    eps_high = f - 1.0
    eps_low  = 1.0 - h
    eps = 1e-9
    delta_high = np.clip(delta_high_ratio * eps_high, eps, eps_high - eps)
    delta_low  = np.clip(delta_low_ratio  * eps_low,  eps, eps_low  - eps)

    a_conn = 1.0 - delta_low
    b_conn = 1.0 + delta_high

    kappa_h = eps_high / (eps_high**2 - delta_high**2)
    c_h     = np.arctanh(delta_high/eps_high) - kappa_h*delta_high

    kappa_l = eps_low  / (eps_low**2  - delta_low**2 )
    c_l     = -np.arctanh(delta_low/eps_low) + kappa_l*delta_low

    return (kappa_h, c_h), (kappa_l, c_l), (a_conn, b_conn)

def three_segment_segments(r, f=1.28, h=0.8, delta_high_ratio=0.5, delta_low_ratio=0.5):
    r = np.asarray(r)
    eps_high = f - 1.0
    eps_low  = 1.0 - h

    (kappa_h, c_h), (kappa_l, c_l), (a_conn, b_conn) = three_segment_params(
        f=f, h=h, delta_high_ratio=delta_high_ratio, delta_low_ratio=delta_low_ratio
    )

    mask_lo  = r <= a_conn
    mask_mid = (r > a_conn) & (r < b_conn)
    mask_hi  = r >= b_conn

    y_lo  = 1.0 + eps_low  * np.tanh( kappa_l * (r[mask_lo]  - 1.0) + c_l )
    y_mid = r[mask_mid]
    y_hi  = 1.0 + eps_high * np.tanh( kappa_h * (r[mask_hi] - 1.0) + c_h )

    return {
        "lo":  (r[mask_lo],  y_lo),
        "mid": (r[mask_mid], y_mid),
        "hi":  (r[mask_hi],  y_hi),
        "splits": (a_conn, b_conn)
    }

# --------------------- Axes range (auto) ---------------------
def seg_y_minmax(seg):
    vals = []
    for key in ("lo","mid","hi"):
        if key in seg and len(seg[key][0])>0:
            vals.append(np.min(seg[key][1]))
            vals.append(np.max(seg[key][1]))
    return (min(vals) if vals else 0.0, max(vals) if vals else 1.0)

# Estimate characteristic length ~ 1/slope-scale
eps_high = f - 1.0
eps_low  = 1.0 - h
inv_scales = [1.0/max(k,1e-6), 1.0/max((eps_high/eps_low)*k,1e-6)]

def estimate_new_scales(g, rho, f, h):
    e_low = rho + h*(1.0 - rho)
    e_up  = rho + f*(1.0 - rho)
    aU, bU, _, _ = params_upper_converge(f, e_up, g)
    aLc, bLc, _, _ = params_lower_converge(h, e_low, g)
    aLr, bLr, _, _ = params_lower_reach(h, e_low, g)
    return [1.0/max(bU,1e-6), 1.0/max(bLc,1e-6), 1.0/max(bLr,1e-6)], (e_low, e_up)

new_scales, (e_low_tmp, e_up_tmp) = estimate_new_scales(g, rho, f, h)
inv_scales.extend(new_scales)

(kappa_h, _), (kappa_l, _), (a_conn_tmp, b_conn_tmp) = three_segment_params(
    f, h, delta_high_ratio, delta_low_ratio
)
inv_scales.append(1.0/max(abs(kappa_h),1e-6))
inv_scales.append(1.0/max(abs(kappa_l),1e-6))

L = 4.0 * max(inv_scales)
r_min = min(1.0, e_low_tmp, e_up_tmp, a_conn_tmp, b_conn_tmp) - L
r_max = max(1.0, e_low_tmp, e_up_tmp, a_conn_tmp, b_conn_tmp) + L
r_grid = np.linspace(r_min, r_max, 1600)

# --------------------- Build segments and plot ---------------------
seg_base       = initial_segments(r_grid, k=k, f=f, h=h)
seg_new_conv   = new_segments(r_grid, g=g, rho=rho, f=f, h=h, lower_method="converge")
seg_new_reach  = new_segments(r_grid, g=g, rho=rho, f=f, h=h, lower_method="reach")
seg_three      = three_segment_segments(r_grid, f=f, h=h,
                                        delta_high_ratio=delta_high_ratio,
                                        delta_low_ratio=delta_low_ratio)

ymins = [h, 0.0]
ymaxs = [f]
for s in [seg_base, seg_new_conv, seg_new_reach, seg_three]:
    mn, mx = seg_y_minmax(s)
    ymins.append(mn); ymaxs.append(mx)
y_min = min(ymins)
y_max = max(ymaxs)
pad = 0.1 * (y_max - y_min) if y_max > y_min else 0.1

plt.figure(figsize=(10, 6))

# Baseline: two segments
r_lo, y_lo = seg_base["lo"]
r_hi, y_hi = seg_base["hi"]
plt.plot(r_lo, y_lo, label=f"Baseline - lower (r<=1, k={k:g})")
plt.plot(r_hi, y_hi, label=f"Baseline - upper (r>=1, k={k:g})")

# New (converge): three segments; middle y=r uses dashed line
# for part, ls, lbl in [
#     ("lo", "-",  f"New-converge - lower (r<=e_low, g={g:g}, rho={rho:g})"),
#     ("mid","--", f"New-converge - middle (e_low<r<e_up, y=r)"),
#     ("hi", "-",  f"New-converge - upper (r>=e_up)"),
# ]:
#     r_seg, y_seg = seg_new_conv[part]
#     if len(r_seg)>0:
#         plt.plot(r_seg, y_seg, linestyle=ls, label=lbl)

# New (reach): three segments; middle y=r also dashed; lower/upper dotted to distinguish
for part, ls, lbl in [
    ("lo", ":",  f"New-reach - lower (r<=e_low, g={g:g}, rho={rho:g})"),
    ("mid","--", f"New-reach - middle (e_low<r<e_up, y=r)"),
    ("hi", ":",  f"New-reach - upper (r>=e_up)"),
]:
    r_seg, y_seg = seg_new_reach[part]
    if len(r_seg)>0:
        plt.plot(r_seg, y_seg, linestyle=ls, label=lbl)

# Three-segment tanh: three segments; middle y=r dashed
# for part, ls, lbl in [
#     ("lo", "-",  f"Three-segment - lower (r<=a)"),
#     ("mid","--", f"Three-segment - middle (a<r<b, y=r)"),
#     ("hi", "-",  f"Three-segment - upper (r>=b)"),
# ]:
#     r_seg, y_seg = seg_three[part]
#     if len(r_seg)>0:
#         plt.plot(r_seg, y_seg, linestyle=ls, label=lbl)

# Reference and split lines
plt.axhline(f, linestyle="--", linewidth=1, label="Upper bound f")
plt.axhline(h, linestyle="--", linewidth=1, label="Lower bound h")
plt.axvline(1.0, linestyle=":", linewidth=1, label="Baseline split r=1")
e_low, e_up = seg_new_conv["splits"]
plt.axvline(e_low, linestyle=":", linewidth=1, label="New split e_low")
plt.axvline(e_up,  linestyle=":", linewidth=1, label="New split e_up")
a_conn, b_conn = seg_three["splits"]
plt.axvline(a_conn, linestyle=":", linewidth=1, label="Three-segment a=1-delta_low")
plt.axvline(b_conn, linestyle=":", linewidth=1, label="Three-segment b=1+delta_high")

plt.xlim(r_min, r_max)
plt.ylim(y_min - pad, y_max + pad)
plt.xlabel("r")
plt.ylabel("y")
plt.title("Segmented Visualization: Baseline / New (converge & reach) / Three-segment")
plt.legend(ncol=2)
plt.grid(True, linestyle=":")

# Save and show
plt.savefig("/remote-home1/yli/Workspace/BandPO/RLtraining/verl/verl/bandpo/soft_clipbound/theory/theory_vis.png", dpi=160)
plt.show()
