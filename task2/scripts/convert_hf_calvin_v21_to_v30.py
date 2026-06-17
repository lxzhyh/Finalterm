"""Convert xiaoma26/calvin-lerobot (v2.1) to LeRobot v3.0 format."""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
import io


def read_v21_episode(parquet_path: str, tasks_map: dict = None) -> list[dict]:
    """Read a v2.1 parquet episode file and return list of frames."""
    table = pq.read_table(parquet_path)
    frames = []
    for i in range(table.num_rows):
        row = {col: table.column(col)[i].as_py() for col in table.column_names}

        # Decode PNG images
        static_img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        gripper_img = Image.open(io.BytesIO(row["wrist_image"]["bytes"])).convert("RGB")

        # Convert to CHW float32 tensors (normalized to [0, 1])
        static_np = np.array(static_img, dtype=np.float32) / 255.0  # HWC
        gripper_np = np.array(gripper_img, dtype=np.float32) / 255.0  # HWC
        static_chw = static_np.transpose(2, 0, 1)  # CHW
        gripper_chw = gripper_np.transpose(2, 0, 1)  # CHW

        task_idx = row["task_index"]
        task_str = tasks_map.get(task_idx, f"task_{task_idx}") if tasks_map else f"task_{task_idx}"

        frame = {
            "observation.images.static": torch.from_numpy(static_chw),
            "observation.images.gripper": torch.from_numpy(gripper_chw),
            "observation.state": torch.tensor(row["state"], dtype=torch.float32),
            "action": torch.tensor(row["actions"], dtype=torch.float32),
            "task": task_str,
        }
        frames.append(frame)
    return frames


def convert_split(input_dir: str, output_root: str, repo_id: str, split_name: str, max_episodes: int = None):
    """Convert a single split from v2.1 to v3.0."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    meta_dir = os.path.join(input_dir, "meta")
    data_dir = os.path.join(input_dir, "data")

    # Read v2.1 info
    with open(os.path.join(meta_dir, "info.json")) as f:
        v21_info = json.load(f)

    total_episodes = v21_info["total_episodes"]
    fps = v21_info["fps"]

    # Find all parquet files sorted
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "chunk-*/episode_*.parquet")))
    if max_episodes is not None:
        parquet_files = parquet_files[:max_episodes]
    print(f"  Found {len(parquet_files)} parquet files (expect {total_episodes} episodes)")

    # Load tasks map
    tasks_map = {}
    tasks_file = os.path.join(meta_dir, "tasks.jsonl")
    if os.path.exists(tasks_file):
        with open(tasks_file) as f:
            for line in f:
                entry = json.loads(line.strip())
                tasks_map[entry["task_index"]] = entry["task"]
        print(f"  Loaded {len(tasks_map)} tasks")

    # Define features for v3.0 (shapes must be tuples to match numpy .shape)
    features = {
        "observation.images.static": {
            "dtype": "image",
            "shape": (3, 200, 200),
            "names": ["channels", "height", "width"],
        },
        "observation.images.gripper": {
            "dtype": "image",
            "shape": (3, 84, 84),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {"dtype": "float32", "shape": (15,)},
        "action": {"dtype": "float32", "shape": (7,)},
    }

    output_dir = os.path.join(output_root, f"calvin_{split_name}")

    # Create dataset
    ds = LeRobotDataset.create(
        repo_id=f"calvin_{split_name}",
        fps=fps,
        features=features,
        root=output_dir,
        robot_type="franka_emika",
        use_videos=False,
    )
    print(f"  Created dataset at {output_dir}")

    # Convert episodes
    from tqdm import tqdm
    for pf in tqdm(parquet_files, desc=f"Converting {split_name}"):
        frames = read_v21_episode(pf, tasks_map=tasks_map)
        for frame in frames:
            ds.add_frame(frame)
        ds.save_episode()

    print(f"  Converted {len(parquet_files)} episodes, {ds.num_frames} frames")
    return ds.num_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True, help="Path to downloaded HF dataset")
    parser.add_argument("--output_root", required=True, help="Output path for v3.0 datasets")
    parser.add_argument("--splits", nargs="+", default=["A", "D"],
                        help="Splits to convert (A, B, C, D)")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Max episodes to convert per split (None = all)")
    args = parser.parse_args()

    split_map = {"A": "splitA", "B": "splitB", "C": "splitC", "D": "splitD"}

    for split in args.splits:
        hf_split = split_map[split]
        input_dir = os.path.join(args.input_root, hf_split)
        if not os.path.exists(input_dir):
            print(f"  Skipping {split} ({input_dir} not found)")
            continue
        print(f"\n=== Converting {split} ({hf_split}) ===")
        n_frames = convert_split(input_dir, args.output_root, "xiaoma26/calvin-lerobot", split,
                                 max_episodes=args.max_episodes)
        print(f"  Done: {n_frames} frames")


if __name__ == "__main__":
    main()
