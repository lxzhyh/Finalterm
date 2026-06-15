"""
Convert CALVIN dataset to LeRobotDataset format.

Usage:
    python scripts/convert_calvin_to_lerobot.py \
        --calvin_root /path/to/calvin/dataset \
        --output_root data/lerobot_calvin \
        --splits A ABC D \
        --val_ratio 0.1 \
        --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from calvin_lerobot.convert import CalvinToLeRobotConverter
from calvin_lerobot.split import EpisodeSplitter


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CALVIN to LeRobotDataset")
    parser.add_argument("--calvin_root", type=str, required=True,
                        help="Path to CALVIN dataset root")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Output directory for LeRobot datasets")
    parser.add_argument("--splits", nargs="+", default=["A", "ABC", "D"],
                        help="Dataset splits to create")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation split ratio per environment")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splitting")
    parser.add_argument("--cameras", nargs="+", default=["static", "gripper"],
                        help="Camera views to include")
    parser.add_argument("--action_space", type=str, default="relative_cartesian",
                        help="Action space: relative_cartesian")
    parser.add_argument("--static_size", type=int, nargs=2, default=[200, 200],
                        help="Static camera image size (H, W)")
    parser.add_argument("--gripper_size", type=int, nargs=2, default=[84, 84],
                        help="Gripper camera image size (H, W)")
    parser.add_argument("--max_episodes_per_env", type=int, default=None,
                        help="Max episodes per env (for debug runs)")
    return parser.parse_args()


def main():
    args = parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    splitter = EpisodeSplitter(seed=args.seed, val_ratio=args.val_ratio)

    for split_name in args.splits:
        envs = list(split_name)  # "ABC" -> ["A", "B", "C"]
        print(f"\n=== Processing split: {split_name} (envs={envs}) ===")

        # Split episodes into train/val per environment
        splits = splitter.split_environments(
            calvin_root=Path(args.calvin_root),
            environments=envs,
            max_episodes_per_env=args.max_episodes_per_env,
        )

        # Create train dataset
        train_episodes = []
        for env, s in splits.items():
            train_episodes.extend(s["train"])

        converter = CalvinToLeRobotConverter(
            cameras=args.cameras,
            action_space=args.action_space,
            static_size=tuple(args.static_size),
            gripper_size=tuple(args.gripper_size),
        )

        train_output = output_root / f"calvin_{split_name}_train"
        converter.convert(train_episodes, train_output, envs)
        print(f"  Train: {len(train_episodes)} episodes -> {train_output}")

        # Create val dataset (skip for D — test only)
        if split_name != "D":
            val_episodes = []
            for env, s in splits.items():
                val_episodes.extend(s["val"])

            val_output = output_root / f"calvin_{split_name}_val"
            converter.convert(val_episodes, val_output, envs)
            print(f"  Val:   {len(val_episodes)} episodes -> {val_output}")
        else:
            # D is test-only
            test_output = output_root / "calvin_D_test"
            converter.convert(train_episodes, test_output, envs)
            print(f"  Test:  {len(train_episodes)} episodes -> {test_output}")

    print("\n=== Conversion complete ===")
    print(f"Output directory: {args.output_root}")
    print("Contents:")
    for d in sorted(output_root.iterdir()):
        if d.is_dir():
            n_parquet = len(list(d.glob("data/*.parquet")))
            print(f"  {d.name}/  ({n_parquet} parquet files)")


if __name__ == "__main__":
    main()
