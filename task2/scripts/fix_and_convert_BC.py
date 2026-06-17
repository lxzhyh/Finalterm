"""
Fix nested splitB/splitC paths, generate metadata, and convert to v3.0.
"""
import json
import os
import glob
import sys
from pathlib import Path

HF_ROOT = Path("/mnt/workspace/kgg2/wangyuxiang/Finalterm/task2/data/calvin_lerobot_hf")
OUTPUT_ROOT = Path("/mnt/workspace/kgg2/wangyuxiang/Finalterm/task2/data/lerobot_calvin")


def fix_nested_paths():
    """Fix splitB/splitC nested directory structure."""
    for split in ["splitB", "splitC"]:
        outer = HF_ROOT / split
        inner = outer / split
        if inner.is_dir() and not (outer / "data").exists():
            print(f"Fixing nested path: {inner} -> {outer}")
            # Move inner contents to outer level
            for item in inner.iterdir():
                if item.name == ".cache":
                    continue
                target = outer / item.name
                if target.exists():
                    import shutil
                    shutil.rmtree(target) if target.is_dir() else target.unlink()
                item.rename(target)
            # Clean up empty inner dir
            try:
                inner.rmdir()
            except OSError:
                pass
            print(f"  Fixed: {outer}")
        elif (outer / "data").exists():
            print(f"{split}: already at correct level")
        else:
            print(f"{split}: no data found")


def count_parquet_episodes(data_dir):
    """Count episodes from parquet files."""
    parquets = sorted(glob.glob(str(data_dir / "chunk-*" / "episode_*.parquet")))
    return len(parquets), parquets


def read_episode_metadata_from_parquet(parquet_path):
    """Read task_index and episode_index from a parquet file."""
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path, columns=["task_index", "episode_index"])
    task_indices = sorted(set(table.column("task_index").to_pylist()))
    ep_idx = table.column("episode_index")[0].as_py()
    length = table.num_rows
    return ep_idx, length, task_indices


def generate_metadata(split_dir, scene_label):
    """Generate minimal meta files for a split by reading parquet data."""
    meta_dir = split_dir / "meta"
    data_dir = split_dir / "data"

    if not data_dir.exists():
        print(f"  No data dir at {data_dir}")
        return False

    os.makedirs(meta_dir, exist_ok=True)

    parquets = sorted(glob.glob(str(data_dir / "chunk-*" / "episode_*.parquet")))
    if not parquets:
        print(f"  No parquet files found")
        return False

    print(f"  Reading {len(parquets)} parquet files...")

    total_frames = 0
    all_task_indices = set()
    episodes = []

    for i, pf in enumerate(parquets):
        ep_idx, length, task_idxs = read_episode_metadata_from_parquet(pf)
        total_frames += length
        all_task_indices.update(task_idxs)
        episodes.append({
            "episode_index": i,
            "tasks": [f"task_{t}" for t in task_idxs],
            "length": length,
            "source_episode_index": ep_idx,
            "source_start_frame": 0,
            "source_end_frame": length,
            "scene": scene_label,
        })
        if (i + 1) % 200 == 0:
            print(f"    Processed {i+1}/{len(parquets)} episodes...")

    print(f"  Total: {len(parquets)} episodes, {total_frames} frames, {len(all_task_indices)} unique tasks")

    # Write info.json (copy format from splitA)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka_emika",
        "total_episodes": len(parquets),
        "total_frames": total_frames,
        "total_tasks": len(all_task_indices),
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 10,
        "splits": {"train": f"0:{len(parquets)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": json.loads(open(str(HF_ROOT / "splitA" / "meta" / "info.json")).read())["features"],
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Wrote info.json")

    # Write episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    print(f"  Wrote episodes.jsonl ({len(episodes)} entries)")

    # Write tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for tidx in sorted(all_task_indices):
            f.write(json.dumps({"task_index": tidx, "task": f"task_{tidx}"}) + "\n")
    print(f"  Wrote tasks.jsonl ({len(all_task_indices)} entries)")

    return True


def convert_to_v30(split_name, scene_label):
    """Convert a split from v2.1 to v3.0 using existing script."""
    split_dir = HF_ROOT / split_name
    if not (split_dir / "meta" / "info.json").exists():
        print(f"  Metadata not found, generating...")
        if not generate_metadata(split_dir, scene_label):
            return False
    else:
        print(f"  Metadata already exists")

    # Run conversion
    cmd = (
        f"python scripts/convert_hf_calvin_v21_to_v30.py "
        f"--input_root {split_dir} "
        f"--output_root {OUTPUT_ROOT} "
        f"--splits {scene_label}"
    )
    print(f"  Running: {cmd}")
    os.system(cmd)
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Fix nested paths")
    print("=" * 60)
    fix_nested_paths()

    print("\n" + "=" * 60)
    print("Step 2: Convert splitB -> calvin_B_train")
    print("=" * 60)
    convert_to_v30("splitB", "B")

    print("\n" + "=" * 60)
    print("Step 3: Convert splitC -> calvin_C_train")
    print("=" * 60)
    convert_to_v30("splitC", "C")

    print("\n" + "=" * 60)
    print("Done! Check output:")
    print(f"  {OUTPUT_ROOT}")
    print("=" * 60)
