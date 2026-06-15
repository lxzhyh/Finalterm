"""
Dataset visualization and quality checking utilities.

Usage:
    python -m src.calvin_lerobot.visualization \
        --dataset data/lerobot_calvin/calvin_A_train \
        --num_samples 16 \
        --output reports/figures/dataset_preview_A.png
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def load_dataset_samples(dataset_path: Path, num_samples: int = 16, seed: int = 42):
    """Load random samples from a LeRobotDataset directory."""
    data_dir = dataset_path / "data"
    videos_dir = dataset_path / "videos"

    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    rng = np.random.RandomState(seed)
    selected = rng.choice(parquet_files, size=min(num_samples, len(parquet_files)), replace=False)

    samples = []
    for pf in selected:
        table = pq.read_table(pf)
        df = table.to_pandas()

        # Load corresponding video frame (first frame)
        ep_name = pf.stem  # episode_000000
        static_video = videos_dir / f"{ep_name}_static.mp4"
        gripper_video = videos_dir / f"{ep_name}_gripper.mp4"

        static_frame = _extract_first_frame(static_video)
        gripper_frame = _extract_first_frame(gripper_video)

        sample = {
            "episode": ep_name,
            "state": np.array(df["observation.state"].iloc[0], dtype=np.float32),
            "action": np.array(df["action"].iloc[0], dtype=np.float32),
            "static_rgb": static_frame,
            "gripper_rgb": gripper_frame,
            "num_frames": len(df),
        }
        samples.append(sample)

    return samples


def _extract_first_frame(video_path: Path) -> np.ndarray | None:
    """Extract the first frame from a video file."""
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def plot_dataset_preview(samples: list[dict], output_path: Path):
    """Create a grid visualization of dataset samples."""
    n = len(samples)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    # Each sample gets 3 columns: static, gripper, action
    fig, axes = plt.subplots(rows, 3 * cols, figsize=(4 * cols, 3 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for i, sample in enumerate(samples):
        r, c_base = i // cols, (i % cols) * 3

        # Static camera
        ax = axes[r, c_base]
        if sample["static_rgb"] is not None:
            ax.imshow(sample["static_rgb"])
        ax.set_title(f'Ep {sample["episode"][-3:]} static', fontsize=8)
        ax.axis("off")

        # Gripper camera
        ax = axes[r, c_base + 1]
        if sample["gripper_rgb"] is not None:
            ax.imshow(sample["gripper_rgb"])
        ax.set_title("gripper", fontsize=8)
        ax.axis("off")

        # Action bar
        ax = axes[r, c_base + 2]
        action = sample["action"]
        dim_names = ["dx", "dy", "dz", "drx", "dry", "drz", "gr"]
        colors = ["#4C72B0"] * 3 + ["#DD8452"] * 3 + ["#55A868"]
        ax.barh(dim_names, action, color=colors)
        ax.set_title("action", fontsize=8)
        ax.set_xlim(-1, 1)

    # Hide unused subplots
    for i in range(n, rows * cols):
        r, c_base = i // cols, (i % cols) * 3
        for offset in range(3):
            axes[r, c_base + offset].axis("off")

    fig.suptitle(f"Dataset Preview ({n} samples)", fontsize=16)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dataset preview to: {output_path}")


def validate_dataset(dataset_path: Path) -> list[str]:
    """Run quality checks on a converted LeRobotDataset."""
    issues = []

    meta_dir = dataset_path / "meta"
    data_dir = dataset_path / "data"
    videos_dir = dataset_path / "videos"

    # Check directory structure
    if not meta_dir.exists():
        issues.append("Missing meta/ directory")
    if not data_dir.exists():
        issues.append("Missing data/ directory")

    # Check info.json
    info_path = meta_dir / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        if "features" not in info:
            issues.append("info.json missing 'features' key")
    else:
        issues.append("Missing meta/info.json")

    # Check stats.json
    if not (meta_dir / "stats.json").exists():
        issues.append("Missing meta/stats.json")

    # Check parquet files
    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    if not parquet_files:
        issues.append("No parquet files in data/")
    else:
        # Validate first file
        table = pq.read_table(parquet_files[0])
        columns = table.column_names
        required = ["observation.state", "action", "timestamp", "episode_index"]
        for col in required:
            if col not in columns:
                issues.append(f"Parquet missing column: {col}")

        # Check action dimension
        if "action" in columns:
            actions = table.column("action").to_pylist()
            for a in actions[:5]:
                if len(a) != 7:
                    issues.append(f"Action dim mismatch: expected 7, got {len(a)}")
                    break

    # Check video files
    video_files = list(videos_dir.glob("*.mp4"))
    n_parquet = len(parquet_files)
    n_videos = len(video_files)
    if n_videos == 0:
        issues.append("No video files in videos/")

    # Summary
    print(f"\n=== Dataset Validation: {dataset_path.name} ===")
    print(f"  Parquet files: {n_parquet}")
    print(f"  Video files:   {n_videos}")
    if issues:
        print(f"  Issues ({len(issues)}):")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  All checks passed!")

    return issues


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize and validate LeRobotDataset")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to LeRobotDataset directory")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for preview image")
    parser.add_argument("--validate_only", action="store_true",
                        help="Only run validation, skip visualization")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)

    # Always run validation
    issues = validate_dataset(dataset_path)

    if not args.validate_only and not issues:
        samples = load_dataset_samples(dataset_path, num_samples=args.num_samples)
        if args.output:
            plot_dataset_preview(samples, Path(args.output))


if __name__ == "__main__":
    main()
