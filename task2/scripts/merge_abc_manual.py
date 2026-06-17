"""Custom merge of A+B+C datasets by directly copying parquet files."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa


def merge_datasets_manually(data_root: str, output_name: str = "calvin_ABC_real_train"):
    data_root = Path(data_root)
    output_dir = data_root / output_name

    sources = [
        ("calvin_A_train", data_root / "calvin_A_train"),
        ("calvin_B", data_root / "calvin_B"),
        ("calvin_C", data_root / "calvin_C"),
    ]

    for name, path in sources:
        info = json.load(open(path / "meta" / "info.json"))
        print(f"  {name}: {info['total_episodes']} episodes, {info['total_frames']} frames")

    if output_dir.exists():
        os.system(f"find {output_dir} -type f -delete 2>/dev/null")
        os.system(f"find {output_dir} -type d -empty -delete 2>/dev/null")
        if output_dir.exists():
            os.system(f"rm -rf {output_dir}")

    os.makedirs(output_dir / "data" / "chunk-000", exist_ok=True)
    os.makedirs(output_dir / "meta" / "episodes" / "chunk-000", exist_ok=True)

    ep_offset = 0
    idx_offset = 0
    file_idx = 0
    total_episodes = 0
    total_frames = 0
    all_tasks = []
    task_offset = 0

    for src_name, src_path in sources:
        src_info = json.load(open(src_path / "meta" / "info.json"))
        src_data_dir = src_path / "data" / "chunk-000"
        parquet_files = sorted(src_data_dir.glob("file-*.parquet"))

        src_tasks_path = src_path / "meta" / "tasks.parquet"
        if src_tasks_path.exists():
            tasks_table = pq.read_table(str(src_tasks_path))
            task_col = "task" if "task" in tasks_table.column_names else "__index_level_0__"
            for i in range(tasks_table.num_rows):
                task_str = tasks_table.column(task_col)[i].as_py()
                task_idx = tasks_table.column("task_index")[i].as_py()
                all_tasks.append({"task_index": task_idx + task_offset, "task": task_str})
            n_tasks = tasks_table.num_rows
        else:
            n_tasks = 0

        print(f"  Processing {src_name}: {len(parquet_files)} files, {n_tasks} tasks, ep_offset={ep_offset}")

        for pf in parquet_files:
            table = pq.read_table(str(pf))
            df_dict = {col: table.column(col).to_pylist() for col in table.column_names}

            old_ep_indices = sorted(set(df_dict["episode_index"]))
            ep_map = {old: old + ep_offset for old in old_ep_indices}

            new_ep_indices = [ep_map[e] for e in df_dict["episode_index"]]
            new_indices = [i + idx_offset for i in df_dict["index"]]
            new_task_indices = [t + task_offset for t in df_dict["task_index"]]

            new_columns = {}
            for col in table.column_names:
                if col == "episode_index":
                    new_columns[col] = new_ep_indices
                elif col == "index":
                    new_columns[col] = new_indices
                elif col == "task_index":
                    new_columns[col] = new_task_indices
                else:
                    new_columns[col] = df_dict[col]

            new_table = pa.table(new_columns)
            out_path = output_dir / "data" / "chunk-000" / f"file-{file_idx:03d}.parquet"
            pq.write_table(new_table, str(out_path))

            n_rows = table.num_rows
            n_eps = len(old_ep_indices)
            total_frames += n_rows
            total_episodes += n_eps
            ep_offset += n_eps
            idx_offset += n_rows
            file_idx += 1

            print(f"    {pf.name}: {n_rows} rows, {n_eps} episodes -> ep {ep_offset-n_eps}-{ep_offset-1}")

        task_offset += n_tasks

    # Write tasks
    if all_tasks:
        import pandas as pd
        df = pd.DataFrame({"task_index": [t["task_index"] for t in all_tasks]},
                          index=[t["task"] for t in all_tasks])
        df.index.name = None
        df.to_parquet(str(output_dir / "meta" / "tasks.parquet"))

    # Write info.json
    info = {
        "codebase_version": "v3.0",
        "robot_type": "franka_emika",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(all_tasks),
        "total_videos": 0,
        "fps": 10,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "features": {
            "observation.images.static": {"dtype": "image", "shape": [3, 200, 200], "names": ["channels", "height", "width"]},
            "observation.images.gripper": {"dtype": "image", "shape": [3, 84, 84], "names": ["channels", "height", "width"]},
            "observation.state": {"dtype": "float32", "shape": [15], "names": None},
            "action": {"dtype": "float32", "shape": [7], "names": None},
        },
    }
    with open(output_dir / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Copy stats.json from A_train as base
    src_stats = data_root / "calvin_A_train" / "meta" / "stats.json"
    if src_stats.exists():
        shutil.copy(str(src_stats), str(output_dir / "meta" / "stats.json"))

    print(f"\nDone: {total_episodes} episodes, {total_frames} frames, {len(all_tasks)} tasks")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    merge_datasets_manually("data/lerobot_calvin")
