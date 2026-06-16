"""
Export training plots from WandB/SwanLab logs for the report.

Usage:
    python scripts/export_plots.py \
        --wandb_project act-calvin \
        --output_dir reports/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
})


def plot_loss_curves(log_a, log_abc, output_path, metric="action_l1_loss", split="train"):
    """Plot training/validation loss curves for A-only vs ABC joint."""
    fig, ax = plt.subplots(figsize=(10, 6))

    key = f"{split}/{metric}"

    if key in log_a:
        steps_a = [e["step"] for e in log_a[key]]
        vals_a = [e["value"] for e in log_a[key]]
        ax.plot(steps_a, vals_a, label="ACT A-only", alpha=0.85, linewidth=2)

    if key in log_abc:
        steps_abc = [e["step"] for e in log_abc[key]]
        vals_abc = [e["value"] for e in log_abc[key]]
        ax.plot(steps_abc, vals_abc, label="ACT ABC-joint", alpha=0.85, linewidth=2)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel(f"{split.capitalize()} {metric.replace('_', ' ').title()}")
    ax.set_title(f"{split.capitalize()} {metric.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_eval_comparison(results_list, output_path):
    """Bar chart comparing eval metrics across models."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = [r["experiment"] for r in results_list]
    success_rates = [r.get("success_rate", 0.0) for r in results_list]
    action_l1 = [r.get("mean_action_l1", 0.0) for r in results_list]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, success_rates, width, label="Success Rate", color="#4C72B0")
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width / 2, action_l1, width, label="Action L1 Error", color="#DD8452")

    ax.set_ylabel("Success Rate")
    ax2.set_ylabel("Action L1 Error")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Env-D Zero-shot Evaluation")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def load_jsonl(path):
    """Load JSONL log file."""
    data = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            key = entry.get("metric", entry.get("_step", ""))
            if key not in data:
                data[key] = []
            data[key].append(entry)
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Export experiment plots")
    parser.add_argument("--log_dir", type=str, default="outputs/train",
                        help="Directory containing training logs")
    parser.add_argument("--eval_dir", type=str, default="outputs/eval",
                        help="Directory containing eval results")
    parser.add_argument("--output_dir", type=str, default="reports/figures",
                        help="Output directory for plots")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training logs
    log_a_path = Path(args.log_dir) / "act_A_only_seed42" / "metrics.jsonl"
    log_abc_path = Path(args.log_dir) / "act_ABC_joint_seed42" / "metrics.jsonl"

    log_a = load_jsonl(log_a_path) if log_a_path.exists() else {}
    log_abc = load_jsonl(log_abc_path) if log_abc_path.exists() else {}

    # Training loss curves
    print("Generating loss curves...")
    plot_loss_curves(log_a, log_abc, output_dir / "train_action_l1_curve.png",
                     metric="action_l1_loss", split="train")
    plot_loss_curves(log_a, log_abc, output_dir / "val_action_l1_curve.png",
                     metric="action_l1_loss", split="val")

    # Evaluation comparison
    eval_dir = Path(args.eval_dir)
    results = []
    for eval_file in sorted(eval_dir.glob("*.json")):
        with open(eval_file) as f:
            results.append(json.load(f))

    if results:
        print("Generating evaluation comparison...")
        plot_eval_comparison(results, output_dir / "env_D_evaluation.png")

    print("\nAll plots saved to:", output_dir)


if __name__ == "__main__":
    main()
