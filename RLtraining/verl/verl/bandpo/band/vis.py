# -*- coding: utf-8 -*-
"""
Visualization script for BandPO bounds.
Generates two plots:
1. Comparison of BandKL, BandTV, and BandChi2 (V2 pure math).
2. Precision & Relaxation Comparison: V1 (Old) pure KL vs V2 (New) pure KL vs V2 Relaxed KL.
"""

import math
import torch
import matplotlib.pyplot as plt

# 导入 V2 的新方法
from verl.bandpo.band.band import band
# 导入 V1 的老方法 (用于精度对比)
from verl.bandpo.kl2clipbound.tokenwise_clipbound import compute_tokenwise_ratio_bounds_core as v1_compute_bounds

def plot_divergences_comparison():
    # 生成 0.001 到 1.0 之间的高密度的 p
    p_vals = torch.linspace(0.001, 1.0, 500)
    log_probs = torch.log(p_vals)
    delta = 0.05
    
    plt.figure(figsize=(8, 6))
    
    methods = ["bandkl", "bandtv", "bandchi2"]
    colors = {"bandkl": "blue", "bandtv": "green", "bandchi2": "red"}
    
    for method in methods:
        low, up = band(
            log_probs, 
            method=method, 
            delta=delta, 
            does_relax_high_p_bound=False, # 纯理论绘制
            upper_bound_max=10.0
        )
        
        p_np = p_vals.numpy()
        up_np = up.numpy()
        low_np = low.numpy()
        
        plt.plot(p_np, up_np, label=f'{method} Upper', color=colors[method], linestyle='-', linewidth=2)
        plt.plot(p_np, low_np, label=f'{method} Lower', color=colors[method], linestyle='--', linewidth=2)
        
    # 绘制理论单纯形极限 1/p
    simplex_limit = (1.0 / p_vals).numpy()
    plt.plot(p_vals.numpy(), simplex_limit, label='Simplex Limit (1/p)', color='black', linestyle=':', linewidth=2)
    
    plt.ylim(0, 13.0)
    plt.xlim(0, 1.0)
    plt.xlabel('Action Probability $p$ (old policy)')
    plt.ylabel('Ratio $r$ Bounds')
    plt.title(f'BandPO V2 Bounds across Divergences ($\delta={delta}$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bandpo_v2_divergences.png', dpi=300)
    print("Saved 'bandpo_v2_divergences.png'")


def plot_kl_relaxation_and_precision_effect():
    p_vals = torch.linspace(0.001, 1.0, 500)
    log_probs = torch.log(p_vals)
    delta = 0.05
    eps_high = 0.28
    eps_low = 0.20
    
    plt.figure(figsize=(10, 7))
    p_np = p_vals.numpy()

    # ---------------------------------------------------------
    # 1. 纯净版 V2 KL (New Generic Solver)
    # ---------------------------------------------------------
    low_v2_pure, up_v2_pure = band(
        log_probs, method="bandkl", delta=delta, 
        does_relax_high_p_bound=False, upper_bound_max=10.0
    )
    
    # ---------------------------------------------------------
    # 2. 纯净版 V1 KL (Old Mirror-based Solver) 
    # ---------------------------------------------------------
    low_v1_pure, up_v1_pure = v1_compute_bounds(
        log_probs, 
        method="bandkl", 
        delta=delta, 
        does_relax_high_p_bound=False, 
        upper_bound_max=10.0,
        band_numerical_solver="bisect",
        band_use_log_domain=False # 尽量和V2在同一条件下对比
    )

    # ---------------------------------------------------------
    # 3. 松弛版 V2 KL (Relaxed KL)
    # ---------------------------------------------------------
    low_v2_relax, up_v2_relax = band(
        log_probs, method="bandkl", delta=delta, 
        eps_clip_high=eps_high, eps_clip_low=eps_low,
        does_relax_high_p_bound=True, upper_bound_max=10.0
    )
    
    # ======= 开始绘制 =======
    
    # 绘制 V2 Pure (粗实线/虚线，作为底色)
    plt.plot(p_np, up_v2_pure.numpy(), label='V2 Pure KL Upper (Generic Solver)', color='blue', linestyle='-', linewidth=4, alpha=0.5)
    plt.plot(p_np, low_v2_pure.numpy(), label='V2 Pure KL Lower (Generic Solver)', color='blue', linestyle='-', linewidth=4, alpha=0.5)

    # 绘制 V1 Pure (黑色点划线，叠加在V2上，如果不偏离则完美重合于中心)
    plt.plot(p_np, up_v1_pure.numpy(), label='V1 Pure KL Upper (Mirror Solver)', color='black', linestyle='-.', linewidth=1.5)
    plt.plot(p_np, low_v1_pure.numpy(), label='V1 Pure KL Lower (Mirror Solver)', color='black', linestyle='-.', linewidth=1.5)
    
    # 计算 V1 和 V2 之间的最大绝对误差 (MAE)，打印在控制台确认精度
    mae_up = torch.max(torch.abs(up_v2_pure - up_v1_pure)).item()
    mae_low = torch.max(torch.abs(low_v2_pure - low_v1_pure)).item()
    print(f"[Precision Check] Max Absolute Error between V1 & V2 - Upper: {mae_up:.2e}, Lower: {mae_low:.2e}")

    # 绘制 V2 Relaxed
    plt.plot(p_np, up_v2_relax.numpy(), label='V2 Relaxed KL Upper', color='orange', linestyle='--', linewidth=2)
    plt.plot(p_np, low_v2_relax.numpy(), label='V2 Relaxed KL Lower', color='orange', linestyle='--', linewidth=2)
    
    # 绘制基线 Fixed Clip Bound 
    plt.axhline(1 + eps_high, color='gray', linestyle=':', linewidth=2, label='Fixed Clip Upper (e.g., GRPO)')
    plt.axhline(1 - eps_low, color='gray', linestyle=':', linewidth=2, label='Fixed Clip Lower (e.g., GRPO)')
    
    plt.ylim(0, 13.0)
    plt.xlim(0, 1.0)
    plt.xlabel('Action Probability $p$ (old policy)')
    plt.ylabel('Ratio $r$ Bounds')
    plt.title(f'BandPO KL: Precision (V1 vs V2) & Relaxation Effect ($\delta={delta}$)')
    
    # 把图例放到外面一点或者用小字体，防止遮挡曲线
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bandpo_v2_kl_precision_and_relaxation.png', dpi=300)
    print("Saved 'bandpo_v2_kl_precision_and_relaxation.png'")

if __name__ == "__main__":
    plot_divergences_comparison()
    plot_kl_relaxation_and_precision_effect()