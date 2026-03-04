# ====== NEW: exhaustive visualization with all variants ======
from __future__ import annotations
import torch
import numpy as np
import matplotlib.pyplot as plt

from verl.bandpo.kl2clipbound.tokenwise_clipbound import compute_tokenwise_ratio_bounds_core

@torch.no_grad()
def plot_all_bounds_1x2(
    *,
    # ---- Core hyper-params ----
    delta: float = 0.5,
    eps_clip_high: float = 0.28,
    eps_clip_low: float = 0.20,
    upper_bound_max_bandkl: float = 100.0,
    upper_bound_max_dcpo: float = 10.0,
    # ---- Numerics / solver ----
    max_iter_newton: int = 12,
    max_iter_bisect: int = 80,
    tol: float = 1e-10,
    TINY_ABS: float = 1e-18,
    REL_MARGIN: float = 1e-12,
    BRACKET_ABS: float = 1e-12,
    MIN_LOGP: float = -700.0,
    # ---- Grid & plotting ----
    n_points: int = 201,
    device=None,
    dtype=torch.float64,
    ylim_ratio: tuple[float|None, float|None] = (1.0, 2.0),
    show_std_clip: bool = True,
    show_theory: bool = True,
    save_path: str | None = None,
    title_suffix: str = "",
):
    """
    画出以下所有曲线（左：ratio；右：Δ=r*p-p），一行两列：
      - 标准常数 clip: upper=1+ε_high, lower=1-ε_low
      - 理论极限：upper=1/p, lower=0（可选）
      - DCPO (SAC-style)：上/下界
      - bandkl（8 条组合的上下界均画出）：
          solver ∈ {newton, bisect}
          domain ∈ {log-domain=True, log-domain=False}
          relax ∈ {does_relax_high_p_bound True/False}

    备注：
    - 全部向量化，GPU 友好；
    - bandkl 内部调用 compute_tokenwise_ratio_bounds_core 保持与你训练时一致的后处理；
    - upper_bound_max 分别对 bandkl 和 DCPO 可单独配置。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- p grid & log p -----
    p = torch.linspace(0.0, 1.0, n_points, device=device, dtype=dtype)
    lp = torch.empty_like(p)
    mask = (p > 0) & (p <= 1)
    lp[mask] = torch.log(torch.clamp(p[mask], min=torch.finfo(dtype).tiny, max=1.0))
    lp[~mask] = MIN_LOGP

    # ----- helpers -----
    def _deltas(upper: torch.Tensor, lower: torch.Tensor, p_: torch.Tensor):
        return upper * p_ - p_, lower * p_ - p_

    # ----- Standard constant clip -----
    std_upper = torch.full_like(p, 1.0 + eps_clip_high)
    std_lower = torch.full_like(p, 1.0 - eps_clip_low)

    # ----- Theory -----
    if show_theory:
        theory_upper = torch.where(p > 0, 1.0 / torch.clamp(p, min=torch.finfo(dtype).tiny), torch.tensor(float("inf"), dtype=dtype, device=device))
    else:
        theory_upper = None
    theory_lower = torch.zeros_like(p)

    # --- DCPO （两条：no-relax 与 relax） ---
    l_dcpo_norelax, u_dcpo_norelax = compute_tokenwise_ratio_bounds_core(
        lp,
        delta=delta,                       # 不用，但作为统一签名传递
        eps_clip_high=eps_clip_high,
        eps_clip_low=eps_clip_low,
        does_relax_high_p_bound=False,     # 不放宽
        upper_bound_max=upper_bound_max_dcpo,
        method="dcpo",
        band_numerical_solver="newton",      # 无关项
        band_use_log_domain=True,        # 无关项
    )
    l_dcpo_relax,  u_dcpo_relax  = compute_tokenwise_ratio_bounds_core(
        lp,
        delta=delta,
        eps_clip_high=eps_clip_high,
        eps_clip_low=eps_clip_low,
        does_relax_high_p_bound=True,      # 放宽
        upper_bound_max=upper_bound_max_dcpo,
        method="dcpo",
        band_numerical_solver="newton",
        band_use_log_domain=True,
    )


    # ----- bandkl variants -----
    # 8 组组合：solver ∈ {newton,bisect} × use_log_domain ∈ {True,False} × relax ∈ {False,True}
    combos = [
        # ("newton", True,  False),
        ("bisect", True,  False),

        # ("newton", True,  True),
        ("bisect", True,  True),

        # ("newton", False, False),
        # ("bisect", False, False),

        # ("newton", False, True),
        # ("bisect", False, True),
    ]

    # 给每组分配风格（颜色/线型/透明度）
    # 颜色按 solver：newton=tab:orange 系，bisect=tab:blue 系
    # 线型按 domain：log=True 用 '-'，log=False 用 '--'
    # 透明度按 relax：False=0.6，True=1.0（“放松”版本更醒目）
    def _style(solver: str, use_log: bool, relax: bool):
        if solver == "newton":
            color = "tab:orange"
        else:
            color = "tab:blue"
        ls = "-" if use_log else "--"
        alpha = 1.0 if relax else 0.6
        return color, ls, alpha

    bandkl_results = []  # list of dicts: {"label":..., "upper":Tensor, "lower":Tensor, "style":(color,ls,alpha)}
    for solver, use_log, relax in combos:
        lower, upper = compute_tokenwise_ratio_bounds_core(
            lp,
            delta=delta,
            eps_clip_high=eps_clip_high,
            eps_clip_low=eps_clip_low,
            does_relax_high_p_bound=relax,
            upper_bound_max=upper_bound_max_bandkl,
            method="bandkl",
            band_numerical_solver=solver,
            band_use_log_domain=use_log,
            max_iter_bisect=max_iter_bisect,
            max_iter_newton=max_iter_newton,
            tol=tol,
            TINY_ABS=TINY_ABS, REL_MARGIN=REL_MARGIN, BRACKET_ABS=BRACKET_ABS, MIN_LOGP=MIN_LOGP,
        )
        color, ls, alpha = _style(solver, use_log, relax)
        label = f"bandkl {solver}, {'log' if use_log else 'prob'}, {'relax' if relax else 'no-relax'}"
        bandkl_results.append({
            "label": label,
            "upper": upper,
            "lower": lower,
            "style": (color, ls, alpha),
        })

    # ===== Plotting =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- (a) ratio bounds vs p ---
    # bandkl 8 组
    for item in bandkl_results:
        color, ls, alpha = item["style"]
        ax1.plot(p.cpu(), item["upper"].cpu(), linestyle=ls, color=color, alpha=alpha, label=item["label"] + " (upper)")
        ax1.plot(p.cpu(), item["lower"].cpu(), linestyle=ls, color=color, alpha=alpha*0.85, label=item["label"] + " (lower)")

    # DCPO
    # ax1.plot(p.cpu(), u_dcpo.cpu(), linestyle="-.", color="tab:olive",  label="DCPO upper")
    # ax1.plot(p.cpu(), l_dcpo.cpu(), linestyle="-.", color="tab:green",  label="DCPO lower")
    ax1.plot(p.cpu(), u_dcpo_norelax.cpu(), linestyle="-.", color="tab:olive",  alpha=0.75, label="DCPO upper (no-relax)")
    ax1.plot(p.cpu(), l_dcpo_norelax.cpu(), linestyle="-.", color="tab:green",  alpha=0.75, label="DCPO lower (no-relax)")
    ax1.plot(p.cpu(), u_dcpo_relax.cpu(),   linestyle="-.", color="tab:olive",  alpha=1.00, label="DCPO upper (relax)")
    ax1.plot(p.cpu(), l_dcpo_relax.cpu(),   linestyle="-.", color="tab:green",  alpha=1.00, label="DCPO lower (relax)")

    # Standard clip
    if show_std_clip:
        ax1.plot(p.cpu(), std_upper.cpu(), linestyle=":", color="gray",  label="Std clip upper (1+ε)")
        ax1.plot(p.cpu(), std_lower.cpu(), linestyle=":", color="brown", label="Std clip lower (1-ε)")

    # Theory
    if show_theory:
        ax1.plot(p.cpu(), theory_upper.cpu(), linestyle=":", color="tab:cyan", label="Theoretical upper 1/p")
        ax1.plot(p.cpu(), theory_lower.cpu(), linestyle=":", color="black",    label="Theoretical lower 0")

    ax1.set_xlim(0.0, 1.0)
    if ylim_ratio is not None:
        ymin, ymax = ylim_ratio
        if ymin is not None or ymax is not None:
            ax1.set_ylim(*(ymin if ymin is not None else ax1.get_ylim()[0],
                           ymax if ymax is not None else ax1.get_ylim()[1]))
    ax1.set_ylim(0.0, 3.0)
    ax1.set_xlabel("old probability p")
    ax1.set_ylabel("ratio r = π/π_old")
    ax1.set_title(f"(a) Ratio bounds vs p (δ={delta}){(' ' + title_suffix) if title_suffix else ''}")
    ax1.legend(fontsize=8, ncol=2)

    # --- (b) Δ = r·p - p vs p ---
    # bandkl 8 组
    for item in bandkl_results:
        color, ls, alpha = item["style"]
        du, dd = _deltas(item["upper"], item["lower"], p)
        ax2.plot(p.cpu(), du.cpu(), linestyle=ls, color=color, alpha=alpha,       label=item["label"] + " (Δ-up)")
        ax2.plot(p.cpu(), dd.cpu(), linestyle=ls, color=color, alpha=alpha * 0.8, label=item["label"] + " (Δ-down)")

    # DCPO
    du_dcpo_nr, dd_dcpo_nr = u_dcpo_norelax * p - p, l_dcpo_norelax * p - p
    du_dcpo_rx, dd_dcpo_rx = u_dcpo_relax  * p - p, l_dcpo_relax  * p - p
    ax2.plot(p.cpu(), du_dcpo_nr.cpu(), linestyle="-.", color="tab:olive", alpha=0.75, label="DCPO Δ-up (no-relax)")
    ax2.plot(p.cpu(), dd_dcpo_nr.cpu(), linestyle="-.", color="tab:green", alpha=0.75, label="DCPO Δ-down (no-relax)")
    ax2.plot(p.cpu(), du_dcpo_rx.cpu(), linestyle="-.", color="tab:olive", alpha=1.00, label="DCPO Δ-up (relax)")
    ax2.plot(p.cpu(), dd_dcpo_rx.cpu(), linestyle="-.", color="tab:green", alpha=1.00, label="DCPO Δ-down (relax)")

    # Std & Theory
    if show_std_clip:
        du_std, dd_std = _deltas(std_upper, std_lower, p)
        ax2.plot(p.cpu(), du_std.cpu(), linestyle=":", color="gray",  label="Std clip Δ-up")
        ax2.plot(p.cpu(), dd_std.cpu(), linestyle=":", color="brown", label="Std clip Δ-down")

    if show_theory:
        du_theo, dd_theo = _deltas(theory_upper, theory_lower, p)
        ax2.plot(p.cpu(), du_theo.cpu(), linestyle=":", color="tab:cyan", label="Theoretical Δ-up")
        ax2.plot(p.cpu(), dd_theo.cpu(), linestyle=":", color="black",    label="Theoretical Δ-down")

    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_xlabel("old probability p")
    ax2.set_ylabel("Δ = r·p - p")
    ax2.set_title("(b) Δ-up / Δ-down vs p")
    ax2.legend(fontsize=8, ncol=2)

    # ===== 在 x=0.1 处标注并打印所有曲线数值 =====
    x_mark = 1.0
    # 找到最接近 x_mark 的网格点索引（鲁棒于 n_points 变化/浮点误差）
    idx = torch.argmin(torch.abs(p - x_mark)).item()
    x_val = float(p[idx].detach().cpu())
    def _annotate_and_print(ax, kind: str, label: str, y_tensor: torch.Tensor, color: str, *,
                            text_dx: int = 5, text_dy: int = 5):
        """在 ax 上标点并打印。kind 用于区分 'ratio' 或 'Δ'。"""
        y = float(y_tensor[idx].detach().cpu())
        ax.scatter([x_val], [y], s=25, color=color, zorder=6, edgecolors="none")
        ax.annotate(f"{y:.4g}",
                    xy=(x_val, y),
                    xytext=(text_dx, text_dy),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.65, edgecolor="none"))
        print(f"[p={x_val:.3f}] [{kind}] {label}: {y:.6g}")
    # —— (a) ratio 子图：bandkl
    for item in bandkl_results:
        color, ls, alpha = item["style"]
        _annotate_and_print(ax1, "ratio", item["label"] + " (upper)", item["upper"], color)
        _annotate_and_print(ax1, "ratio", item["label"] + " (lower)", item["lower"], color)

    # —— (a) ratio 子图：DCPO（两组 relax / no-relax）
    _annotate_and_print(ax1, "ratio", "DCPO upper (no-relax)", u_dcpo_norelax, "tab:olive")
    _annotate_and_print(ax1, "ratio", "DCPO lower (no-relax)", l_dcpo_norelax, "tab:green")
    _annotate_and_print(ax1, "ratio", "DCPO upper (relax)",   u_dcpo_relax,   "tab:olive")
    _annotate_and_print(ax1, "ratio", "DCPO lower (relax)",   l_dcpo_relax,   "tab:green")

    # —— (a) ratio 子图：Std & Theory（如果开启）
    if show_std_clip:
        _annotate_and_print(ax1, "ratio", "Std clip upper (1+ε)", std_upper, "gray")
        _annotate_and_print(ax1, "ratio", "Std clip lower (1-ε)", std_lower, "brown")
    if show_theory and theory_upper is not None:
        _annotate_and_print(ax1, "ratio", "Theoretical upper 1/p", theory_upper, "tab:cyan")
        _annotate_and_print(ax1, "ratio", "Theoretical lower 0",   theory_lower, "black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()

plot_all_bounds_1x2(
    delta=0.02,
    eps_clip_high=0.28,
    eps_clip_low=0.20,
    upper_bound_max_bandkl=10.0,
    upper_bound_max_dcpo=10.0,
    max_iter_newton=80,
    max_iter_bisect=80,
    tol=1e-10,
    n_points=101,
    ylim_ratio=(0.0, 20),
    show_std_clip=True,
    show_theory=True,
    save_path="/remote-home1/yli/Workspace/BandPO/RLtraining/verl/verl/bandpo/kl2clipbound/all_bounds_1x2.png",
    title_suffix=" | all variants"
)
