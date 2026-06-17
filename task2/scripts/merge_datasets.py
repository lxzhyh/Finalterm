"""Merge multiple v3.0 LeRobot datasets into one combined dataset.

Usage:
    python scripts/merge_datasets.py \
        --inputs data/lerobot_calvin/calvin_A data/lerobot_calvin/calvin_B data/lerobot_calvin/calvin_C \
        --output data/lerobot_calvin/calvin_ABC_real \
        --repo_id calvin_ABC_real
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa


def merge_datasets(input_dirs: list[str], output_dir: str, repo_id: str):
    """Merge multiple LeRobot v3.0 datasets into one."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    total_frames = 0
    total_tasks = 0
    all_task_names = []
    all_episodes_meta = []

    # Copy data chunks and reindex episodes
    chunk_idx = 0
    episode_in_chunk = 0
    max_per_chunk = 1000

    data_out = output_dir / "data"
    data_out.mkdir(parents=True, exist_ok=True)

    global_episode_idx = 0
    global_frame_idx = 0

    for input_dir in input_dirs:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            print(f"  WARNING: {input_dir} not found, skipping")
            continue

        # Read source info
        with open(input_dir / "meta" / "info.json") as f:
            src_info = json.load(f)

        src_episodes = src_info["total_episodes"]
        src_frames = src_info["total_frames"]
        print(f"  Processing {input_dir.name}: {src_episodes} episodes, {src_frames} frames")

        # Find all parquet files
        parquet_files = sorted(input_dir.rglob("episode_*.parquet"))
        if not parquet_files:
            print(f"  WARNING: No parquet files found in {input_dir}")
            continue

        # Copy parquet files, reindexing episodes
        for pf in parquet_files:
            # Determine output chunk
            chunk_dir = data_out / f"chunk-{chunk_idx:03d}"
            chunk_dir.mkdir(exist_ok=True)

            # Copy the file with new name
            dst = chunk_dir / f"episode_{global_episode_idx:06d}.parquet"
            shutil.copy2(pf, dst)

            all_episodes_meta.append({
                "episode_index": global_episode_idx,
                "tasks": [],
                "length": 0,
            })

            global_episode_idx += 1
            episode_in_chunk += 1

            if episode_in_chunk >= max_per_chunk:
                chunk_idx += 1
                episode_in_chunk = 0

        total_episodes += src_episodes
        total_frames += src_frames

        # Collect tasks
        tasks_file = input_dir / "meta" / "tasks.jsonl"
        if tasks_file.exists():
            with open(tasks_file) as f:
                for line in f:
                    task = json.loads(line.strip())
                    if task["task"] not in all_task_names:
                        all_task_names.append(task["task"])
                        total_tasks += 1

        # Copy videos if present
        videos_in = input_dir / "videos"
        if videos_in.exists():
            videos_out = output_dir / "videos"
            videos_out.mkdir(exist_ok=True)
            for src_video in videos_in.rglob("*.mp4"):
                rel = src_video.relative_to(videos_in)
                dst_video = videos_out / rel
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_video, dst_video)

    # Write merged meta files
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(exist_ok=True)

    # info.json
    merged_info = {
        "codebase_version": "v3.0",
        "robot_type": "franka_emika",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": 10,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
    }

    # Copy features from first input
    first_info_path = Path(input_dirs[0]) / "meta" / "info.json"
    if first_info_path.exists():
        with open(first_info_path) as f:
            first_info = json.load(f)
        if "features" in first_info:
            merged_info["features"] = first_info["features"]

    with open(meta_dir / "info.json", "w") as f:
        json.dump(merged_info, f, indent=4)

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for i, name in enumerate(all_task_names):
            f.write(json.dumps({"task_index": i, "task": name}) + "\n")

    # episodes.jsonl (simplified)
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for i in range(total_episodes):
            f.write(json.dumps({"episode_index": i, "tasks": [], "length": 0}) + "\n")

    print(f"\n=== Merged dataset ===")
    print(f"  Episodes: {total_episodes}")
    print(f"  Frames: {total_frames}")
    print(f"  Tasks: {total_tasks}")
    print(f"  Chunks: {chunk_idx + 1}")
    print(f"  Output: {output_dir}")

    # Try to verify by loading
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(repo_id=repo_id, root=str(output_dir))
        print(f"  Verification: loaded {len(ds)} frames successfully")
    except Exception as e:
        print(f"  Verification failed: {e}")
        print(f"  (dataset may still work for training)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Input dataset directories")
    parser.add_argument("--output", required=True, help="Output merged dataset directory")
    parser.add_argument("--repo_id", required=True, help="Repo ID for the merged dataset")
    args = parser.parse_args()
    merge_datasets(args.inputs, args.output, args.repo_id)


if __name__ == "__main__":
    main()
