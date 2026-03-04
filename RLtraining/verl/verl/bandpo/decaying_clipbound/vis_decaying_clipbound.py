# -*- coding: utf-8 -*-
# File: verl/bandpo/decaying_clipbound/vis_decaying_clip.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from verl.bandpo.decaying_clipbound.decaying_clipbound import (
    decaying_clip_linear,
    decaying_clip_hold_then_linear,
    decaying_clip_cosine,
    decaying_clip_poly,
    decaying_clip_restarts,
)

def plot_all_decaying_clips(
    *,
    start_value: float = 0.20,
    end_value: float = 0.02,
    save_path: str | None = None,
):
    """
    单图综合可视化：A/B/C/D 四大家族
      A: 线性、保持后线性
      B: 余弦
      C: 多项式（q=0.5, 2, 3）
      D: 统一重启（cosine/constant × fixed/expanding × 是否衰减）
    """
    s = np.linspace(0.0, 1.0, 1200)

    # ---- A-family ----
    A1 = decaying_clip_linear(s, start_value, end_value)
    A2 = decaying_clip_hold_then_linear(s, start_value, end_value, hold_frac=0.30)

    # ---- B-family ----
    B1 = decaying_clip_cosine(s, start_value, end_value)

    # ---- C-family ----
    C1 = decaying_clip_poly(s, start_value, end_value, exponent=0.5)
    C2 = decaying_clip_poly(s, start_value, end_value, exponent=2.0)
    C3 = decaying_clip_poly(s, start_value, end_value, exponent=3.0)

    # ---- D-family: unified restarts ----
    # Cosine 内核
    D1  = decaying_clip_restarts(s, start_value, end_value,
                                 mode="fixed", T0=0.20,
                                 decay=False, shape="cosine")
    D1d = decaying_clip_restarts(s, start_value, end_value,
                                 mode="fixed", T0=0.20,
                                 decay=True,  gamma=0.8, shape="cosine")   # ← 新增 fixed+decay
    D2  = decaying_clip_restarts(s, start_value, end_value,
                                 mode="expanding", T0=0.12, T_mult=2.0,
                                 decay=False, shape="cosine")
    D3  = decaying_clip_restarts(s, start_value, end_value,
                                 mode="expanding", T0=0.12, T_mult=2.0,
                                 decay=True,  gamma=0.8, shape="cosine")

    # Constant 内核（阶梯：按周期分段衰减，保证最后一段==end）
    D4  = decaying_clip_restarts(s, start_value, end_value,
                                 mode="fixed", T0=0.10,
                                 decay=False, shape="constant")            # 常数= start
    D5  = decaying_clip_restarts(s, start_value, end_value,
                                 mode="fixed", T0=0.10,
                                 decay=True,  shape="constant")            # ← 新增 fixed+decay（阶梯）
    D6  = decaying_clip_restarts(s, start_value, end_value,
                                 mode="expanding", T0=0.10, T_mult=2.0,
                                 decay=True,  shape="constant")            # 扩张+阶梯

    # ---- Plot (single chart; no explicit colors) ----
    plt.figure(figsize=(12, 7))

    # A
    plt.plot(s, A1, label="A1 Linear")
    plt.plot(s, A2, label="A2 Hold→Linear", linestyle="--")

    # B
    plt.plot(s, B1, label="B1 Cosine", linestyle="-.")

    # C
    plt.plot(s, C1, label="C1 Poly q=0.5", linestyle=":")
    plt.plot(s, C2, label="C2 Poly q=2",   linestyle=":")
    plt.plot(s, C3, label="C3 Poly q=3",   linestyle=":")

    # D: cosine restarts
    plt.plot(s, D1,  label="D1 Restarts cosine fixed T0=0.20")
    plt.plot(s, D1d, label="D1d Restarts cosine fixed + decay γ=0.8", linestyle="--")
    plt.plot(s, D2,  label="D2 Restarts cosine expanding T0=0.12×2")
    plt.plot(s, D3,  label="D3 Restarts cosine expanding + decay γ=0.8", linestyle="--")

    # D: constant restarts (阶梯)
    plt.plot(s, D4,  label="D4 Restarts constant fixed (no decay)", linestyle="-.")
    plt.plot(s, D5,  label="D5 Restarts constant fixed + steps→end", linestyle="-.")
    plt.plot(s, D6,  label="D6 Restarts constant expanding + steps→end", linestyle="-.")

    plt.title("Clip-range ε schedules: A/B/C/D families\nwith unified restarts (cosine & constant/steps)")
    plt.xlabel("training fraction s (0→1)")
    plt.ylabel("clip_range ε")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=10)

    ymax = float(np.max([
        A1.max(), A2.max(), B1.max(), C1.max(), C2.max(), C3.max(),
        D1.max(), D1d.max(), D2.max(), D3.max(), D4.max(), D5.max(), D6.max()
    ]) * 1.05)
    plt.ylim(0.0, max(0.25, ymax))

    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


if __name__ == "__main__":
    plot_all_decaying_clips(
        start_value=0.20,
        end_value=0.02,
        save_path="/remote-home1/yli/Workspace/BandPO/RLtraining/verl/verl/bandpo/decaying_clipbound/theory_vis.png",
    )
