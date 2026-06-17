"""Generate publication-quality figures for the experiment report."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

OUTPUT_DIR = Path("/mnt/workspace/kgg2/wangyuxiang/Finalterm/assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load eval data ----
eval_dir = Path("/mnt/workspace/kgg2/wangyuxiang/Finalterm/task2/outputs/eval")

with open(eval_dir / "act_A_only_D_offline.json") as f:
    a_only = json.load(f)
with open(eval_dir / "act_ABC_joint_D_offline.json") as f:
    abc_joint = json.load(f)
with open(eval_dir / "act_A_only_real_D_offline_full.json") as f:
    a_real = json.load(f)

abc_real_path = eval_dir / "act_ABC_real_D_offline.json"
if abc_real_path.exists():
    with open(abc_real_path) as f:
        abc_real = json.load(f)
    HAS_ABC_REAL = True
else:
    abc_real = None
    HAS_ABC_REAL = False

# ---- Figure 1: Overall metrics comparison bar chart ----
def plot_metrics_comparison():
    metrics = ["mean_action_l1", "position_l1", "rotation_l1", "gripper_error"]
    labels = ["Mean Action L1", "Position L1", "Rotation L1", "Gripper Error"]

    vals_a = [a_only[m] for m in metrics]
    vals_abc = [abc_joint[m] for m in metrics]
    vals_real = [a_real[m] for m in metrics]

    n_groups = 4 if HAS_ABC_REAL else 3
    x = np.arange(len(metrics))
    width = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=(12, 5))
    offset = -width * (n_groups - 1) / 2
    ax.bar(x + offset, vals_a, width, label="A-only (syn, 200 steps)", color="#4C72B0", edgecolor="white")
    ax.bar(x + offset + width, vals_abc, width, label="ABC-joint (syn, 200 steps)", color="#DD8452", edgecolor="white")
    ax.bar(x + offset + 2*width, vals_real, width, label="A-only (real, 20k steps)", color="#55A868", edgecolor="white")
    if HAS_ABC_REAL:
        vals_abc_r = [abc_real[m] for m in metrics]
        ax.bar(x + offset + 3*width, vals_abc_r, width, label="ABC-joint (real, 2k steps)", color="#C44E52", edgecolor="white")

    ax.set_ylabel("Error Value")
    ax.set_title("Offline Evaluation on Env-D (Zero-shot)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")

    for bar in ax.patches:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, rotation=45)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "metrics_comparison.png")
    plt.close(fig)
    print("Saved: metrics_comparison.png")


# ---- Figure 2: Per-dimension error breakdown ----
def plot_per_dim_error():
    dims = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
    dim_labels = [r"$\Delta x$", r"$\Delta y$", r"$\Delta z$",
                  r"$\Delta r_x$", r"$\Delta r_y$", r"$\Delta r_z$", "Gripper"]

    err_a = [abs(a_only["per_dimension_error"][d]["mean"]) for d in dims]
    err_abc = [abs(abc_joint["per_dimension_error"][d]["mean"]) for d in dims]
    err_real = [abs(a_real["per_dimension_error"][d]["mean"]) for d in dims]

    n_groups = 4 if HAS_ABC_REAL else 3
    x = np.arange(len(dims))
    width = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=(14, 5))
    offset = -width * (n_groups - 1) / 2
    ax.bar(x + offset, err_a, width, label="A-only (syn, 200 steps)", color="#4C72B0", edgecolor="white")
    ax.bar(x + offset + width, err_abc, width, label="ABC-joint (syn, 200 steps)", color="#DD8452", edgecolor="white")
    ax.bar(x + offset + 2*width, err_real, width, label="A-only (real, 20k steps)", color="#55A868", edgecolor="white")
    if HAS_ABC_REAL:
        err_abc_r = [abs(abc_real["per_dimension_error"][d]["mean"]) for d in dims]
        ax.bar(x + offset + 3*width, err_abc_r, width, label="ABC-joint (real, 2k steps)", color="#C44E52", edgecolor="white")

    ax.set_ylabel("|Mean Error|")
    ax.set_title("Per-Dimension Error Breakdown on Env-D")
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_dimension_error.png")
    plt.close(fig)
    print("Saved: per_dimension_error.png")


# ---- Figure 3: Action chunking analysis ----
def plot_chunking_analysis():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if HAS_ABC_REAL:
        names = ["A-only\n(syn, 200)", "ABC-joint\n(syn, 200)", "A-only\n(real, 20k)", "ABC-joint\n(real, 2k)"]
        boundary = [a_only["chunk_boundary_delta"], abc_joint["chunk_boundary_delta"],
                     a_real["chunk_boundary_delta"], abc_real["chunk_boundary_delta"]]
        inner_var = [a_only["chunk_inner_variance"], abc_joint["chunk_inner_variance"],
                      a_real["chunk_inner_variance"], abc_real["chunk_inner_variance"]]
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    else:
        names = ["A-only\n(syn, 200)", "ABC-joint\n(syn, 200)", "A-only\n(real, 20k)"]
        boundary = [a_only["chunk_boundary_delta"], abc_joint["chunk_boundary_delta"], a_real["chunk_boundary_delta"]]
        inner_var = [a_only["chunk_inner_variance"], abc_joint["chunk_inner_variance"], a_real["chunk_inner_variance"]]
        colors = ["#4C72B0", "#DD8452", "#55A868"]

    ax1 = axes[0]
    bars = ax1.bar(names, boundary, color=colors, edgecolor="white", width=0.6)
    ax1.set_ylabel("L2 Distance")
    ax1.set_title("Chunk Boundary Delta\n(Lower = Smoother Transitions)")
    for bar, val in zip(bars, boundary):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    ax2 = axes[1]
    bars = ax2.bar(names, inner_var, color=colors, edgecolor="white", width=0.6)
    ax2.set_ylabel("Variance")
    ax2.set_title("Chunk Inner Variance\n(Lower = More Consistent Actions)")
    ax2.set_yscale("log")
    for bar, val in zip(bars, inner_var):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5,
                 f'{val:.2e}', ha='center', va='bottom', fontsize=10)

    fig.suptitle("Action Chunking Smoothness Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chunking_analysis.png")
    plt.close(fig)
    print("Saved: chunking_analysis.png")


# ---- Figure 4: Error distribution box plots ----
def plot_error_std_comparison():
    dims = ["dx", "dy", "dz", "drx", "dry", "drz"]
    dim_labels = ["dx", "dy", "dz", "drx", "dry", "drz"]

    std_a = [a_only["per_dimension_error"][d]["std"] for d in dims]
    std_abc = [abc_joint["per_dimension_error"][d]["std"] for d in dims]
    std_real = [a_real["per_dimension_error"][d]["std"] for d in dims]

    n_groups = 4 if HAS_ABC_REAL else 3
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(dims))
    width = 0.8 / n_groups
    offset = -width * (n_groups - 1) / 2

    ax.bar(x + offset, std_a, width, label="A-only (syn, 200)", color="#4C72B0", edgecolor="white")
    ax.bar(x + offset + width, std_abc, width, label="ABC-joint (syn, 200)", color="#DD8452", edgecolor="white")
    ax.bar(x + offset + 2*width, std_real, width, label="A-only (real, 20k)", color="#55A868", edgecolor="white")
    if HAS_ABC_REAL:
        std_abc_r = [abc_real["per_dimension_error"][d]["std"] for d in dims]
        ax.bar(x + offset + 3*width, std_abc_r, width, label="ABC-joint (real, 2k)", color="#C44E52", edgecolor="white")

    ax.set_ylabel("Standard Deviation of Error")
    ax.set_title("Error Variance per Dimension (Lower = More Consistent)")
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_variance.png")
    plt.close(fig)
    print("Saved: error_variance.png")


# ---- Figure 5: Architecture diagram (text-based) ----
def plot_architecture():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis('off')

    # ACT architecture flow
    components = {
        "static_img": (0.05, 0.7, "Static Image\n(3, 200, 200)"),
        "gripper_img": (0.05, 0.4, "Gripper Image\n(3, 84, 84)"),
        "state": (0.05, 0.1, "Robot State\n(15,)"),
        "cnn1": (0.25, 0.7, "ResNet-18\n(pretrained)"),
        "cnn2": (0.25, 0.4, "ResNet-18\n(pretrained)"),
        "state_enc": (0.25, 0.1, "Linear\nEncoder"),
        "concat": (0.45, 0.45, "Feature\nConcatenation"),
        "cvae_enc": (0.6, 0.6, "CVAE Encoder\n(4 layers, d=32)"),
        "transformer": (0.6, 0.25, "Transformer\nEncoder (4L)\n+ Decoder (1L)\nd=512, h=8"),
        "cvae_dec": (0.8, 0.45, "CVAE Decoder\n(Action Chunk)"),
        "output": (0.95, 0.45, "Action Chunk\n(100, 7)"),
    }

    colors = {
        "static_img": "#E8D5B7", "gripper_img": "#E8D5B7", "state": "#B7D5E8",
        "cnn1": "#C4E8B7", "cnn2": "#C4E8B7", "state_enc": "#C4E8B7",
        "concat": "#E8E8B7", "cvae_enc": "#D5B7E8", "transformer": "#E8B7D5",
        "cvae_dec": "#D5B7E8", "output": "#FFD5D5",
    }

    for key, (x, y, text) in components.items():
        rect = plt.Rectangle((x - 0.05, y - 0.06), 0.12, 0.14,
                             fill=True, facecolor=colors[key],
                             edgecolor="black", linewidth=1.2, alpha=0.8,
                             transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        ax.text(x + 0.01, y + 0.01, text, ha='center', va='center',
                fontsize=8, transform=ax.transAxes, zorder=3)

    # Arrows
    arrow_pairs = [
        ("static_img", "cnn1"), ("gripper_img", "cnn2"), ("state", "state_enc"),
        ("cnn1", "concat"), ("cnn2", "concat"), ("state_enc", "concat"),
        ("concat", "cvae_enc"), ("concat", "transformer"),
        ("cvae_enc", "cvae_dec"), ("transformer", "cvae_dec"), ("cvae_dec", "output"),
    ]

    for src, dst in arrow_pairs:
        sx, sy, _ = components[src]
        dx, dy, _ = components[dst]
        ax.annotate("", xy=(dx - 0.05, dy + 0.01), xytext=(sx + 0.07, sy + 0.01),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
                    zorder=1)

    # KL divergence arrow
    ax.text(0.6, 0.78, "KL Divergence\nRegularization", ha='center', va='center',
            fontsize=8, color="purple", style='italic', transform=ax.transAxes)

    ax.set_title("ACT (Action Chunking with Transformers) Architecture", fontsize=14, pad=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "act_architecture.png")
    plt.close(fig)
    print("Saved: act_architecture.png")


# ---- Figure 6: Dataset statistics comparison ----
def plot_dataset_stats():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    datasets = ["A\n(train)", "B\n(train)", "C\n(train)", "D\n(test)"]
    episodes = [1000, 997, 715, 1000]
    frames = [60164, 60044, 42427, 60552]
    data_types = ["Real", "Real", "Real", "Real"]

    ax = axes[0]
    colors4 = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax.bar(datasets, episodes, color=colors4, edgecolor="white")
    ax.set_ylabel("Episodes")
    ax.set_title("Dataset: Episode Count")
    for bar, val, dt in zip(bars, episodes, data_types):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val}\n({dt})', ha='center', va='bottom', fontsize=9)

    ax = axes[1]
    bars = ax.bar(datasets, frames, color=colors4, edgecolor="white")
    ax.set_ylabel("Frames")
    ax.set_title("Dataset: Frame Count")
    for bar, val in zip(bars, frames):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{val:,}', ha='center', va='bottom', fontsize=9)

    ax = axes[2]
    # Feature dimensions
    features = ["Static\nImage", "Gripper\nImage", "State", "Action"]
    dims = [3*200*200, 3*84*84, 15, 7]
    ax.barh(features, dims, color=["#8172B2", "#8172B2", "#55A868", "#C44E52"], edgecolor="white")
    ax.set_xlabel("Dimension")
    ax.set_title("Feature Dimensions")
    ax.set_xscale("log")
    for i, (f, d) in enumerate(zip(features, dims)):
        ax.text(d * 1.1, i, str(d), va='center', fontsize=9)

    fig.suptitle("CALVIN Dataset Statistics", fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "dataset_stats.png")
    plt.close(fig)
    print("Saved: dataset_stats.png")


if __name__ == "__main__":
    plot_metrics_comparison()
    plot_per_dim_error()
    plot_chunking_analysis()
    plot_error_std_comparison()
    plot_architecture()
    plot_dataset_stats()
    print("\nAll figures generated in:", OUTPUT_DIR)
