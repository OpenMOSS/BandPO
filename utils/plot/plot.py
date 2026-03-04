import numpy as np
import matplotlib.pyplot as plt

def plot_bounds(save_path: str, delta: float = 0.1):
    eps = 1e-3  # avoid p=0 singularity

    # Domains
    p_neg = np.linspace(-1, -eps, 2000)
    p_pos = np.linspace(eps, 2, 3000)

    # chi-square term is real only when (1-p)/p >= 0 for delta>0 -> p in (0,1]
    p_chi = np.linspace(eps, 1, 2500)

    # TV bounds
    rbar_tv_neg = 1 + delta / p_neg
    r_tv_neg    = 1 - delta / p_neg
    rbar_tv_pos = 1 + delta / p_pos
    r_tv_pos    = 1 - delta / p_pos

    # Chi-square bounds
    chi_term = np.sqrt(delta * (1 - p_chi) / p_chi)
    rbar_chi = 1 + chi_term
    r_chi    = 1 - chi_term

    plt.figure(figsize=(8, 4.8))

    # TV (split across 0)
    plt.plot(p_neg, rbar_tv_neg, label=r'$\bar r_{\mathrm{TV},\delta}(p)=1+\delta/p$')
    plt.plot(p_neg, r_tv_neg,    label=r'$r_{\mathrm{TV},\delta}(p)=1-\delta/p$')
    plt.plot(p_pos, rbar_tv_pos)
    plt.plot(p_pos, r_tv_pos)

    # Chi-square (only on (0,1])
    plt.plot(p_chi, rbar_chi, label=r'$\bar r_{\chi^2,\delta}(p)=1+\sqrt{\delta(1-p)/p}$')
    plt.plot(p_chi, r_chi,    label=r'$r_{\chi^2,\delta}(p)=1-\sqrt{\delta(1-p)/p}$')

    plt.xlim(-1, 2)
    plt.ylim(-1, 5)
    plt.xlabel('p')
    plt.ylabel('y')
    plt.title(rf'$\delta={delta}$')
    plt.grid(True, linewidth=0.5)
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()

    # Save
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    plot_bounds("bounds_plot.png", delta=0.01)

