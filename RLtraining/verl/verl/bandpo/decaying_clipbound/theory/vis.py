# Comprehensive visualization of four families (A/B/C/D) of clip_range (ε) schedules,
# each with multiple variants, *all plotted on a single chart*.
#
# Conventions:
# - SB3 passes `progress_remaining = p ∈ [1→0]`. We visualize against training fraction s ∈ [0→1],
#   with p = 1 - s.
# - We DO NOT set explicit colors (per instruction). We use matplotlib with one figure (no subplots).
# - Each function is documented with the mathematical form and short theoretical note.
#
# Legend keys:
#   A*: Linear family
#   B*: Cosine family
#   C*: Polynomial family
#   D*: Cosine-with-Restarts (SGDR-style) family

import numpy as np
import matplotlib.pyplot as plt
from math import pi

# --------------------
# A) Linear schedules
# --------------------
def clip_linear_to_zero(p, eps0=0.20):
    """
    Linear-to-zero:
      ε(p) = eps0 * p
    Theory:
      Classic step-size decay à la Robbins–Monro: gradually shrink the trust region in policy space.
    """
    return eps0 * p

def clip_linear_with_floor(p, eps0=0.20, eps_min=0.02):
    """
    Linear with floor:
      ε(p) = eps_min + (eps0 - eps_min) * p
    Theory:
      Ensures a non-zero late-stage trust region to avoid premature freezing.
    """
    return eps_min + (eps0 - eps_min) * p

def clip_piecewise_hold_then_linear(p, eps0=0.20, eps_min=0.02, hold_frac=0.3):
    """
    Piecewise: hold-then-decay
      Let s be training fraction, s = 1 - p.
      For s ∈ [0, hold_frac]: ε = eps0   (hold constant early)
      For s ∈ (hold_frac, 1]:   ε linear from eps0 → eps_min
    Theory:
      "Warm-hold" maintains exploration early, then linearly tighten for exploitation.
    """
    s = 1.0 - p
    eps = np.where(
        s <= hold_frac,
        eps0,
        eps_min + (eps0 - eps_min) * (1.0 - s) / (1.0 - hold_frac)
    )
    return eps

# ----------------------
# B) Cosine schedules
# ----------------------
def clip_cosine_with_floor(p, eps_max=0.20, eps_min=0.02):
    """
    Cosine anneal (no restart):
      ε(p) = eps_min + (eps_max - eps_min)/2 * (1 + cos(pi*(1 - p)))
    Theory:
      Smooth early descent (keeps exploration), faster late descent → stable finishing.
    """
    return eps_min + 0.5 * (eps_max - eps_min) * (1 + np.cos(np.pi * (1 - p)))

def clip_cosine_to_zero(p, eps_max=0.20):
    """
    Cosine-to-zero (floor=0):
      ε(p) = (eps_max)/2 * (1 + cos(pi*(1 - p)))
    """
    return 0.5 * eps_max * (1 + np.cos(np.pi * (1 - p)))

# -------------------------
# C) Polynomial schedules
# -------------------------
def clip_poly(p, eps_max=0.20, eps_min=0.02, q=2.0):
    """
    Polynomial:
      ε(p) = eps_min + (eps_max - eps_min) * p^q , q > 0
    Theory:
      Shape control: q>1 gives "slow-then-fast" tightening; 0<q<1 gives "fast-then-slow".
    """
    return eps_min + (eps_max - eps_min) * (p ** q)

# ---------------------------------------------
# D) Cosine with (Warm) Restarts (SGDR-style)
# ---------------------------------------------
def _sgdr_periods(T0=0.20, T_mult=2.0, s_end=1.0):
    """
    Helper to generate growing cycle lengths (in s-space) that cover [0, s_end].
    Returns arrays: starts[], lengths[]
    """
    periods = []
    T = float(T0)
    total = 0.0
    while total < s_end - 1e-12:
        Ti = min(T, s_end - total)
        periods.append(Ti)
        total += Ti
        T *= T_mult
    starts = np.cumsum([0.0] + periods[:-1])
    return np.array(starts), np.array(periods)

def clip_cosine_restarts_fixed(p, eps_max=0.20, eps_min=0.02, T0=0.20):
    """
    SGDR with fixed period (T_mult=1):
      For each cycle in s-space of length T0:
        ε = eps_min + (eps_max - eps_min)/2 * (1 + cos(pi * phase))
      where phase = (s - cycle_start)/T0 ∈ [0,1]
    Theory:
      Periodic "heat-and-cool" to re-explore new basins at each restart.
    """
    s = 1.0 - p
    starts, lengths = _sgdr_periods(T0=T0, T_mult=1.0, s_end=1.0)
    # Locate cycle index for each s
    idx = np.searchsorted(starts[1:], s, side='right')
    phase = (s - starts[idx]) / lengths[idx]
    return eps_min + 0.5 * (eps_max - eps_min) * (1 + np.cos(np.pi * phase))

def clip_cosine_restarts_expanding(p, eps_max=0.20, eps_min=0.02, T0=0.15, T_mult=2.0):
    """
    SGDR with expanding periods:
      Period lengths grow: T0, T0*T_mult, T0*T_mult^2, ...
      Same cosine formula within each period.
    Theory:
      Early frequent restarts encourage exploration; later longer periods stabilize refinement.
    """
    s = 1.0 - p
    starts, lengths = _sgdr_periods(T0=T0, T_mult=T_mult, s_end=1.0)
    idx = np.searchsorted(starts[1:], s, side='right')
    phase = (s - starts[idx]) / lengths[idx]
    return eps_min + 0.5 * (eps_max - eps_min) * (1 + np.cos(np.pi * phase))

def clip_cosine_restarts_expanding_amp_decay(p, eps_max=0.20, eps_min=0.02, T0=0.15, T_mult=2.0, gamma=0.8):
    """
    SGDR with expanding periods + amplitude decay:
      At cycle k, effective (eps_max_k - eps_min) = gamma^k * (eps_max - eps_min)
      Then apply the same cosine inside the cycle.
    Theory:
      Each restart is milder than the previous one → diminishing exploratory bursts over time.
    """
    s = 1.0 - p
    starts, lengths = _sgdr_periods(T0=T0, T_mult=T_mult, s_end=1.0)
    idx = np.searchsorted(starts[1:], s, side='right')
    phase = (s - starts[idx]) / lengths[idx]
    amp = (eps_max - eps_min) * (gamma ** idx)
    return eps_min + 0.5 * amp * (1 + np.cos(np.pi * phase))

# ---------------------
# Build and plot curves
# ---------------------
s = np.linspace(0.0, 1.0, 1200)   # training fraction
p = 1.0 - s                       # SB3 progress_remaining

# Common parameter choices (you can tweak for your runs)
eps0 = 0.20
eps_min = 0.02
eps_max = eps0

# A-family
A1 = clip_linear_to_zero(p, eps0=eps0)
A2 = clip_linear_with_floor(p, eps0=eps0, eps_min=eps_min)
A3 = clip_piecewise_hold_then_linear(p, eps0=eps0, eps_min=eps_min, hold_frac=0.30)

# B-family
B1 = clip_cosine_with_floor(p, eps_max=eps_max, eps_min=eps_min)
B2 = clip_cosine_to_zero(p, eps_max=eps_max)

# C-family (polynomials with different q)
C1 = clip_poly(p, eps_max=eps_max, eps_min=eps_min, q=0.5)  # fast-then-slow
C2 = clip_poly(p, eps_max=eps_max, eps_min=eps_min, q=2.0)  # slow-then-fast
C3 = clip_poly(p, eps_max=eps_max, eps_min=eps_min, q=3.0)  # even more late tightening

# D-family (SGDR variants)
D1 = clip_cosine_restarts_fixed(p, eps_max=eps_max, eps_min=eps_min, T0=0.20)
D2 = clip_cosine_restarts_expanding(p, eps_max=eps_max, eps_min=eps_min, T0=0.12, T_mult=2.0)
D3 = clip_cosine_restarts_expanding_amp_decay(p, eps_max=eps_max, eps_min=eps_min, T0=0.12, T_mult=2.0, gamma=0.8)

# Single chart with all curves
plt.figure(figsize=(12, 7))
# plt.plot(s, A1, label="A1 Linear→0", alpha=0.35, linestyle="-", color="tab:olive")
# plt.plot(s, A2, label="A2 Linear with floor")
# plt.plot(s, A3, label="A3 Hold(30%)→Linear")

# tab:olive
# tab:cyan
# tab:green
# brown
# black

plt.plot(s, B1, label="B1 Cosine (floor)", linestyle="-.")
plt.plot(s, B2, label="B2 Cosine→0", linestyle="-.")

plt.plot(s, C1, label="C1 Poly q=0.5", linestyle="-.")
plt.plot(s, C2, label="C2 Poly q=2", linestyle="-.")
plt.plot(s, C3, label="C3 Poly q=3", linestyle="-.")

plt.plot(s, D1, label="D1 SGDR fixed T0=0.20", linestyle=":")
plt.plot(s, D2, label="D2 SGDR expanding T0=0.12,×2", linestyle=":")
plt.plot(s, D3, label="D3 SGDR expanding + amp decay", linestyle=":")

plt.title("Clip-range ε schedules (SB3-style): A/B/C/D families with multiple variants\nx-axis: training fraction s ∈ [0,1];  p = 1 - s is SB3 'progress_remaining'")
plt.xlabel("training fraction s (0→1)")
plt.ylabel("clip_range ε")
plt.grid(True)
plt.legend(ncol=2, fontsize=10)
plt.ylim(0.0, max(0.25, float(np.max([A1.max(), B1.max(), C1.max(), D1.max()])) * 1.05))
plt.savefig("/remote-home1/yli/Workspace/BandPO/RLtraining/verl/verl/bandpo/decaying_clipbound/theory/theory_vis.png", dpi=160)
plt.show()
