"""Merge A+B+C LeRobot v3.0 datasets into a single combined dataset."""

import argparse
import shutil
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/lerobot_calvin")
    parser.add_argument("--output_name", default="calvin_ABC_real_train")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = data_root / args.output_name
    if output_dir.exists():
        print(f"Removing existing output: {output_dir}")
        shutil.rmtree(output_dir)

    splits = ["calvin_A_train", "calvin_B", "calvin_C"]
    datasets = []
    for name in splits:
        ds_dir = data_root / name
        if not ds_dir.exists():
            print(f"ERROR: {ds_dir} not found!")
            return
        ds = LeRobotDataset(repo_id=name, root=str(ds_dir))
        print(f"  {name}: {ds.num_episodes} episodes, {ds.num_frames} frames")
        datasets.append(ds)

    print(f"\nMerging {len(datasets)} datasets into {output_dir}...")
    merged = merge_datasets(
        datasets=datasets,
        output_repo_id=args.output_name,
        output_dir=str(output_dir),
    )
    print(f"Done: {merged.num_episodes} episodes, {merged.num_frames} frames")


if __name__ == "__main__":
    main()
