# -*- coding: utf-8 -*-
# File: verl/bandpo/soft/vis_softclip.py
from __future__ import annotations
import torch
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from verl.bandpo.soft_clipbound.soft_clipbound import (
    soft_clip_1seg,
    soft_clip_2seg,
    soft_clip_3seg_control_converge,
    soft_clip_3seg,
    soft_clip_3seg_rollback,
    soft_clip_3seg_rollback_plus,
    apply_soft_clip,
)

@torch.no_grad()
def plot_all_softclips(
    *,
    low_bound: float = 0.8,
    high_bound: float = 1.28,
    r_max: float = None,    # 若 None 默认 1.2*high_bound
    n_points: int = 1201,
    device=None,
    dtype=torch.float64,
    save_path: Optional[str] = None,
):
    """
    画出 7条 soft-clip 映射：y(r) vs r，便于与 numpy 可视化对齐。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A = torch.tensor(low_bound,  dtype=dtype, device=device)
    # B = torch.tensor(high_bound, dtype=dtype, device=device)
    A = low_bound
    B = high_bound
    if r_max is None:
        r_max = float(high_bound) * 1.2
    r = torch.linspace(0.0, float(r_max), n_points, dtype=dtype, device=device)

    # ---- 逐方法计算 ----
    # y1 = soft_clip_1seg(r, A, B, k=k)
    # y2 = soft_clip_2seg(r, A, B, k=1.0 if k is None else float(k))
    # y3c, (e_high_1, e_low_1), g_eff_c = soft_clip_3seg_control_converge(r, A, B, rho=rho, g=g, lower_method="converge")
    # y3r, (e_high_2, e_low_2), g_eff_r = soft_clip_3seg_control_converge(r, A, B, rho=rho, g=g, lower_method="reach")
    # y3, (e_high_3, e_low_3)           = soft_clip_3seg(r, A, B, rho=rho)
    # y4   = soft_clip_3seg_rollback(r, A, B, alpha=alpha, use_activate_function=False, alpha_in=alpha_in)
    # y4a  = soft_clip_3seg_rollback(r, A, B, alpha=alpha, use_activate_function=True, alpha_in=alpha_in)
    # y4_plus   = soft_clip_3seg_rollback_plus(r, A, B, alpha=alpha, gamma=0.2, use_activate_function=False, alpha_in=alpha_in)
    # y4a_plus  = soft_clip_3seg_rollback_plus(r, A, B, alpha=alpha, gamma=0.2, use_activate_function=True, alpha_in=alpha_in)
    y = apply_soft_clip(r,A,B, "hard")
    y1 = apply_soft_clip(r,A,B, "1seg")
    y2 = apply_soft_clip(r,A,B, "2seg")
    y3c = apply_soft_clip(r,A,B, "3seg")
    y3r = apply_soft_clip(r,A,B, "3seg_control_converge")
    y3 = apply_soft_clip(r,A,B, "3seg_control_reach")
    y4 = apply_soft_clip(r,A,B, "rollback")
    y4a = apply_soft_clip(r,A,B, "rollback_activate")
    y4_plus = apply_soft_clip(r,A,B, "rollback_plus")
    y4a_plus = apply_soft_clip(r,A,B, "rollback_activate_plus")

    # ---- 画图 ----
    plt.figure(figsize=(12.5, 7.0))
    plt.plot(r.cpu(), y.cpu(),                        label="0) hard", alpha=0.35)
    plt.plot(r.cpu(), y1.cpu(),                       label="1) soft_clip_1seg")
    plt.plot(r.cpu(), y2.cpu(),                       label="2) soft_clip_2seg")
    plt.plot(r.cpu(), y3c.cpu(),                      label="3) 3seg_control_converge  (g)", linestyle="-", alpha=0.6)
    plt.plot(r.cpu(), y3r.cpu(),                      label="4) 3seg_control_reach     (g_eff)", linestyle=":")
    plt.plot(r.cpu(), y3.cpu(),                       label="5) 3seg (closed-form)", alpha=0.6, linestyle=":")
    plt.plot(r.cpu(), y4.cpu(),                       label="6) 3seg_rollback", alpha=0.6)
    plt.plot(r.cpu(), y4a.cpu(),                      label="7) 3seg_rollback_activate", alpha=0.6, linestyle=":")
    plt.plot(r.cpu(), y4_plus.cpu(),                  label="8) 3seg_rollback_plus", alpha=0.6)
    plt.plot(r.cpu(), y4a_plus.cpu(),                 label="9) 3seg_rollback_activate_plus", alpha=0.6, linestyle=":")

    # 参考线
    # plt.axhline(float(B), linestyle="--", linewidth=1, label="Upper bound", alpha=0.4)
    # plt.axhline(float(A), linestyle="--", linewidth=1, label="Lower bound", alpha=0.4)
    plt.axvline(1.0,      linestyle=":",  linewidth=1, label="r = 1", alpha=0.4)

    # 分割点（以封闭解的 e_high/e_low 为例）
    # plt.axvline(float(e_low_3.detach().cpu()),  linestyle=":", linewidth=1, label="e_low")
    # plt.axvline(float(e_high_3.detach().cpu()), linestyle=":", linewidth=1, label="e_high")

    plt.xlim(0.0, float(r_max))
    plt.ylim(0.0, float(B) * 1.2)
    plt.xlabel("ratio r")
    plt.ylabel("soft-clipped y(r)")
    # title = f"Soft-clip mappings (rho={rho}, g={g}, g_eff={float(g_eff_r.mean().cpu()):.3f}, k={'auto' if k is None else k}, alpha={alpha}, alpha_in={alpha_in})"
    title = f"Soft-clip mappings"
    plt.title(title)
    plt.legend(ncol=2, frameon=True, fontsize=9)
    plt.grid(True, linestyle=":")

    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()

# 直接运行示例（开发期可用）
if __name__ == "__main__":
    plot_all_softclips(
        low_bound=0.8,
        # high_bound=1.28,
        high_bound=10,
        r_max=20,
        n_points=1201,
        save_path="/remote-home1/yli/Workspace/BandPO/RLtraining/verl/verl/bandpo/soft_clipbound/all_soft_method_vis.png",
    )