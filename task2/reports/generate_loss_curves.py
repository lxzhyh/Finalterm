"""Generate training loss curve plots from re-run log data."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

OUTPUT_DIR = Path("/mnt/workspace/kgg2/wangyuxiang/Finalterm/task2/reports/figures")

# ---- Data extracted from training logs ----
a_only_steps =  [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
a_only_loss =   [45.280,13.916,8.215,6.429,5.533,5.082,4.381,4.215,4.021,3.754,3.892,3.516,3.466,3.360,3.267,3.204,3.086,3.020,3.078,3.048]
a_only_grad =   [718.852,294.906,190.518,158.583,141.952,133.086,117.013,110.406,109.005,110.908,108.940,105.534,98.261,96.810,97.009,93.604,91.459,89.055,90.939,90.794]

abc_steps =     [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
abc_loss =      [41.410,9.158,6.269,5.034,4.824,4.240,3.777,3.731,3.468,3.280,3.253,3.167,3.153,2.914,2.915,2.766,2.752,2.549,2.689,2.453]
abc_grad =      [747.558,254.605,181.229,146.598,139.978,132.681,117.038,111.515,110.791,114.673,104.131,105.902,105.495,93.635,94.358,94.137,95.680,83.572,91.776,84.749]

# ABC-joint real data (from real A+B+C training, 200 steps)
abc_real_steps = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
abc_real_loss =  [44.658,12.933,7.877,6.215,5.556,4.934,4.619,4.182,4.007,3.736,3.741,3.653,3.532,3.294,3.249,3.274,3.133,3.051,2.937,3.083]
abc_real_grad =  [715.968,268.317,185.402,153.132,139.923,125.033,124.689,114.336,111.789,111.084,108.219,103.525,102.683,99.280,96.883,97.856,94.616,89.637,85.245,94.172]

# ---- Figure 1: Training Loss Curves ----
def plot_loss_curves():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(a_only_steps, a_only_loss, 'o-', label="A-only (Env-A, real)", color="#4C72B0",
            linewidth=2, markersize=5, alpha=0.9)
    ax.plot(abc_steps, abc_loss, 's-', label="ABC-joint (Env-A+B+C, synthetic)", color="#DD8452",
            linewidth=2, markersize=5, alpha=0.9)
    ax.plot(abc_real_steps, abc_real_loss, '^-', label="ABC-joint (Env-A+B+C, real)", color="#C44E52",
            linewidth=2, markersize=5, alpha=0.9)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Training Loss (L1 + KL)")
    ax.set_title("Training Loss Curves: A-only vs ABC-joint")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 210)

    ax.annotate(f'{a_only_loss[-1]:.3f}', xy=(200, a_only_loss[-1]),
                xytext=(160, a_only_loss[-1]+3),
                arrowprops=dict(arrowstyle='->', color='#4C72B0'),
                fontsize=10, color='#4C72B0')
    ax.annotate(f'{abc_loss[-1]:.3f}', xy=(200, abc_loss[-1]),
                xytext=(160, abc_loss[-1]-3),
                arrowprops=dict(arrowstyle='->', color='#DD8452'),
                fontsize=10, color='#DD8452')
    ax.annotate(f'{abc_real_loss[-1]:.3f}', xy=(200, abc_real_loss[-1]),
                xytext=(160, abc_real_loss[-1]+5),
                arrowprops=dict(arrowstyle='->', color='#C44E52'),
                fontsize=10, color='#C44E52')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "train_loss_curve.png")
    plt.close(fig)
    print("Saved: train_loss_curve.png")


# ---- Figure 2: Loss + Gradient Norm (dual axis) ----
def plot_loss_and_gradient():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(a_only_steps, a_only_loss, 'o-', label="A-only (real)", color="#4C72B0", linewidth=2, markersize=4)
    ax1.plot(abc_steps, abc_loss, 's-', label="ABC-joint (synthetic)", color="#DD8452", linewidth=2, markersize=4)
    ax1.plot(abc_real_steps, abc_real_loss, '^-', label="ABC-joint (real)", color="#C44E52", linewidth=2, markersize=4)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Convergence")
    ax1.legend()
    ax1.set_xlim(0, 210)

    ax2.plot(a_only_steps, a_only_grad, 'o-', label="A-only (real)", color="#4C72B0", linewidth=2, markersize=4)
    ax2.plot(abc_steps, abc_grad, 's-', label="ABC-joint (synthetic)", color="#DD8452", linewidth=2, markersize=4)
    ax2.plot(abc_real_steps, abc_real_grad, '^-', label="ABC-joint (real)", color="#C44E52", linewidth=2, markersize=4)
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Gradient Norm")
    ax2.set_title("Gradient Norm (Training Stability)")
    ax2.legend()
    ax2.set_xlim(0, 210)

    fig.suptitle("ACT Training Dynamics: A-only vs ABC-joint (200 steps)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "train_loss_and_gradient.png")
    plt.close(fig)
    print("Saved: train_loss_and_gradient.png")


# ---- Figure 3: Log-scale loss ----
def plot_loss_logscale():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(a_only_steps, a_only_loss, 'o-', label="A-only (real)", color="#4C72B0",
                linewidth=2, markersize=5)
    ax.semilogy(abc_steps, abc_loss, 's-', label="ABC-joint (synthetic)", color="#DD8452",
                linewidth=2, markersize=5)
    ax.semilogy(abc_real_steps, abc_real_loss, '^-', label="ABC-joint (real)", color="#C44E52",
                linewidth=2, markersize=5)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Training Loss (log scale)")
    ax.set_title("Training Loss (Log Scale)")
    ax.legend()
    ax.set_xlim(0, 210)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "train_loss_logscale.png")
    plt.close(fig)
    print("Saved: train_loss_logscale.png")


if __name__ == "__main__":
    plot_loss_curves()
    plot_loss_and_gradient()
    plot_loss_logscale()
    print("\nAll loss curve figures generated in:", OUTPUT_DIR)
