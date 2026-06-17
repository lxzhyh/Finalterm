"""Create combined ABC dataset from v2.1 HF parquet files, converting to v3.0 on-the-fly.

Reads episodes directly from v2.1 parquet files and writes v3.0 format.
Much faster than converting then merging separately.

Usage:
    python scripts/create_abc_real_dataset.py \
        --hf_root data/calvin_lerobot_hf \
        --output data/lerobot_calvin/calvin_ABC_real \
        --max_per_env 1000
"""

import argparse
import glob
import io
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm


def read_v21_frames(parquet_path, tasks_map=None):
    """Read all frames from a v2.1 parquet episode file."""
    table = pq.read_table(parquet_path)
    frames = []
    for i in range(table.num_rows):
        row = {col: table.column(col)[i].as_py() for col in table.column_names}

        static_img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        gripper_img = Image.open(io.BytesIO(row["wrist_image"]["bytes"])).convert("RGB")

        static_np = np.array(static_img, dtype=np.float32) / 255.0
        gripper_np = np.array(gripper_img, dtype=np.float32) / 255.0

        task_idx = row.get("task_index", 0)
        task_str = tasks_map.get(task_idx, f"task_{task_idx}") if tasks_map else f"task_{task_idx}"

        frames.append({
            "observation.images.static": torch.from_numpy(static_np.transpose(2, 0, 1)),
            "observation.images.gripper": torch.from_numpy(gripper_np.transpose(2, 0, 1)),
            "observation.state": torch.tensor(row["state"], dtype=torch.float32),
            "action": torch.tensor(row["actions"], dtype=torch.float32),
            "task": task_str,
        })
    return frames


def create_abc_dataset(hf_root, output_dir, max_per_env=1000):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output_dir = Path(output_dir)
    hf_root = Path(hf_root)

    splits = {
        "A": {"data": hf_root / "splitA" / "data", "meta": hf_root / "splitA" / "meta"},
        "B": {"data": hf_root / "splitB" / "data", "meta": hf_root / "splitB" / "meta"},
        "C": {"data": hf_root / "splitC" / "data", "meta": hf_root / "splitC" / "meta"},
    }

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

    ds = LeRobotDataset.create(
        repo_id="calvin_ABC_real",
        fps=10,
        features=features,
        root=str(output_dir),
        robot_type="franka_emika",
        use_videos=False,
    )

    total_episodes = 0
    total_frames = 0

    for env_name, split_info in splits.items():
        data_dir = split_info["data"]
        meta_dir = split_info["meta"]
        if not data_dir.exists():
            print(f"  WARNING: {data_dir} not found, skipping Env-{env_name}")
            continue

        tasks_map = {}
        tasks_file = meta_dir / "tasks.jsonl"
        if tasks_file.exists():
            with open(tasks_file) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    tasks_map[entry["task_index"]] = entry["task"]
            print(f"  Loaded {len(tasks_map)} tasks for Env-{env_name}")

        parquet_files = sorted(glob.glob(str(data_dir / "chunk-*" / "episode_*.parquet")))
        if max_per_env:
            parquet_files = parquet_files[:max_per_env]

        print(f"\n=== Processing Env-{env_name}: {len(parquet_files)} episodes ===")

        for pf in tqdm(parquet_files, desc=f"Env-{env_name}"):
            try:
                frames = read_v21_frames(pf, tasks_map=tasks_map)
                for frame in frames:
                    ds.add_frame(frame)
                ds.save_episode()
                total_episodes += 1
                total_frames += len(frames)
            except Exception as e:
                print(f"  ERROR reading {pf}: {e}")
                continue

        print(f"  Env-{env_name}: {total_episodes} episodes so far, {total_frames} frames")

    print(f"\n=== Final ABC dataset ===")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output: {output_dir}")

    # Verify
    try:
        ds2 = LeRobotDataset(repo_id="calvin_ABC_real", root=str(output_dir))
        print(f"  Verification: {len(ds2)} frames loaded OK")
    except Exception as e:
        print(f"  Verification note: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_root", default="data/calvin_lerobot_hf")
    parser.add_argument("--output", default="data/lerobot_calvin/calvin_ABC_real")
    parser.add_argument("--max_per_env", type=int, default=1000,
                        help="Max episodes per environment (None = all)")
    args = parser.parse_args()
    create_abc_dataset(args.hf_root, args.output, args.max_per_env)


if __name__ == "__main__":
    main()
