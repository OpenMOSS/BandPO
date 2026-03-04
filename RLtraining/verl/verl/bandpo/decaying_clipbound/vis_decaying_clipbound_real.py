# -*- coding: utf-8 -*-
# File: verl/bandpo/decaying_clipbound/vis_decaying_clip.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# 仅使用统一分发接口
from verl.bandpo.decaying_clipbound.decaying_clipbound import apply_decaying_clip

def plot_all_decaying_clips(
    *,
    start_value: float = 0.20,
    end_value: float = 0.02,
    save_path: str | None = None,
):
    """
    单图综合可视化：A/B/C/D 四大家族（均经由 apply_decaying_clip）
      A: 线性、保持后线性
      B: 余弦
      C: 多项式（q=0.5, 2, 3）
      D: 统一重启（cosine/constant × fixed/expanding × 是否衰减）
    """
    s = np.linspace(0.0, 1.0, 1200)  # 训练进度 process_ratio

    # ---- A-family（线性）----
    A1 = apply_decaying_clip(
        process_ratio=s, start_value=start_value, end_value=end_value,
        method="linear",
    )
    A2 = apply_decaying_clip(
        process_ratio=s, start_value=start_value, end_value=end_value,
        method="hold_linear"
    )

    # ---- B-family（余弦）----
    B1 = apply_decaying_clip(
        process_ratio=s, start_value=start_value, end_value=end_value,
        method="cosine",
    )

    # ---- C-family（多项式）----
    C1 = apply_decaying_clip(
        process_ratio=s, start_value=start_value, end_value=end_value,
        method="poly", exponent=0.5,  # fast-then-slow
    )
    C2 = apply_decaying_clip(
        process_ratio=s, start_value=start_value, end_value=end_value,
        method="poly", exponent=2.0,  # slow-then-fast
    )
    C3 = apply_decaying_clip(
        process_ratio=s, start_value=start_value, end_value=end_value,
        method="poly", exponent=3.0,  # 更晚收紧
    )
    C = apply_decaying_clip(
        process_ratio=s, start_value=start_value, end_value=end_value,
        method="poly"
    )

    # ---- D-family（统一重启）----
    # Cosine 内核
    D1  = apply_decaying_clip(  # fixed, no decay
        process_ratio=s, start_value=start_value, end_value=end_value,
        method="restarts", mode="fixed", T0=0.20,
        decay=False, shape="cosine",
    )
    D1d = apply_decaying_clip(  # fixed + decay（几何衰减）
        process_ratio=s, start_value=start_value, end_value=end_value,
        method="restarts", mode="fixed", T0=0.20,
        decay=True, gamma=0.8, shape="cosine",
    )
    D2  = apply_decaying_clip(  # expanding, no decay
        process_ratio=s, start_value=start_value, end_value=end_value,
        method="restarts", mode="expanding", T0=0.12, T_mult=2.0,
        decay=False, shape="cosine",
    )
    D3  = apply_decaying_clip(  # expanding + decay
        process_ratio=s, start_value=start_value, end_value=end_value,
        method="restarts", mode="expanding", T0=0.12, T_mult=2.0,
        decay=True, gamma=0.8, shape="cosine",
    )

    # Constant 内核（阶梯：按周期分段衰减，保证最后一段==end）
    D4  = apply_decaying_clip(  # fixed, no decay（整段常数 = start）
        process_ratio=s, start_value=start_value, end_value=end_value,
        method="restarts", mode="fixed", T0=0.10,
        decay=False, shape="constant",
    )
    D5  = apply_decaying_clip(  # fixed + steps→end
        process_ratio=s, start_value=start_value, end_value=end_value,
        method="restarts", mode="fixed", T0=0.10,
        decay=True, shape="constant",
    )
    D6  = apply_decaying_clip(  # expanding + steps→end
        process_ratio=s, start_value=start_value, end_value=end_value,
        method="restarts", mode="expanding", T0=0.10, T_mult=2.0,
        decay=True, shape="constant",
    )

    # ---- 绘图（单图，无显式颜色）----
    plt.figure(figsize=(12, 7))

    # A
    plt.plot(s, A1, label="A1 Linear")
    plt.plot(s, A2, label="A2 Hold→Linear", linestyle="--")

    # B
    plt.plot(s, B1, label="B1 Cosine", linestyle="-.")

    # C
    # plt.plot(s, C1, label="C1 Poly q=0.5", linestyle=":")
    # plt.plot(s, C2, label="C2 Poly q=2",   linestyle=":")
    # plt.plot(s, C3, label="C3 Poly q=3",   linestyle=":")
    plt.plot(s, C3, label="C3 Poly q=default",   linestyle=":")

    # D: cosine restarts
    plt.plot(s, D1,  label="D1 Restarts cosine fixed T0=0.20")
    plt.plot(s, D1d, label="D1d Restarts cosine fixed + decay γ=0.8", linestyle="--")
    plt.plot(s, D2,  label="D2 Restarts cosine expanding T0=0.12×2")
    plt.plot(s, D3,  label="D3 Restarts cosine expanding + decay γ=0.8", linestyle="--")

    # D: constant restarts (阶梯)
    plt.plot(s, D4,  label="D4 Restarts constant fixed (no decay)", linestyle="-.")
    plt.plot(s, D5,  label="D5 Restarts constant fixed + steps→end", linestyle="-.")
    plt.plot(s, D6,  label="D6 Restarts constant expanding + steps→end", linestyle="-.")

    plt.title("Clip-range ε schedules: A/B/C/D (all via apply_decaying_clip)\nwith unified restarts (cosine & constant/steps)")
    plt.xlabel("training fraction s (0→1)")
    plt.ylabel("clip_range ε")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=10)

    # 统一设定 y 轴上界
    curves = [A1, A2, B1, C1, C2, C3, D1, D1d, D2, D3, D4, D5, D6]
    ymax = float(np.max([c.max() for c in curves]) * 1.05)
    plt.ylim(0.0, max(0.25, ymax))

    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


if __name__ == "__main__":
    plot_all_decaying_clips(
        start_value=0.20,
        end_value=0.02,
        # 你也可以改为 None 直接显示
        save_path="/remote-home1/yli/Workspace/BandPO/RLtraining/verl/verl/bandpo/decaying_clipbound/theory_vis_real.png",
    )
