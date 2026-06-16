#!/usr/bin/env python3
"""
Inspect CALVIN dataset directory structure and sample data.

Run this AFTER downloading CALVIN to understand the real file layout
before writing any conversion code.

Usage:
    # After downloading debug split:
    bash scripts/download_calvin.sh debug
    python tools/inspect_calvin_dataset.py --root third_party/calvin/dataset

    # Or with ABCD split:
    python tools/inspect_calvin_dataset.py --root /path/to/calvin/dataset --depth 4
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np


def print_tree(root: Path, max_depth: int = 3, prefix: str = ""):
    """Print directory tree up to max_depth levels."""
    if max_depth < 0:
        return

    try:
        entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    except PermissionError:
        print(f"{prefix}[permission denied]")
        return

    # Limit display for large directories
    max_show = 30
    if len(entries) > max_show:
        shown = entries[:max_show]
        remaining = len(entries) - max_show
    else:
        shown = entries
        remaining = 0

    for i, entry in enumerate(shown):
        is_last = (i == len(shown) - 1) and remaining == 0
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "

        if entry.is_dir():
            print(f"{prefix}{connector}{entry.name}/")
            print_tree(entry, max_depth - 1, prefix + extension)
        else:
            size = entry.stat().st_size
            size_str = _format_size(size)
            print(f"{prefix}{connector}{entry.name}  ({size_str})")

    if remaining > 0:
        print(f"{prefix}└── ... and {remaining} more entries")


def _format_size(size_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def find_data_files(root: Path, extensions: list[str], max_count: int = 20) -> dict[str, list[Path]]:
    """Find files with given extensions, return grouped by extension."""
    results = {ext: [] for ext in extensions}
    for ext in extensions:
        for f in root.rglob(f"*{ext}"):
            results[ext].append(f)
            if len(results[ext]) >= max_count:
                break
    return results


def inspect_npz(path: Path):
    """Print contents of an npz file."""
    try:
        data = np.load(path, allow_pickle=True)
        print(f"\n  File: {path}")
        print(f"  Type: npz")
        print(f"  Keys: {list(data.keys())}")
        for key in data.keys():
            arr = data[key]
            print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}")
    except Exception as e:
        print(f"  Error reading {path}: {e}")


def inspect_npy(path: Path):
    """Print shape/dtype of an npy file."""
    try:
        arr = np.load(path, allow_pickle=True)
        print(f"\n  File: {path}")
        print(f"  Type: npy")
        if hasattr(arr, "shape"):
            print(f"  Shape: {arr.shape}, dtype: {arr.dtype}")
            if arr.ndim == 1 and len(arr) > 0:
                print(f"  First element: {arr[0]}")
        else:
            print(f"  Value: {arr} (type: {type(arr).__name__})")
    except Exception as e:
        print(f"  Error reading {path}: {e}")


def inspect_pkl(path: Path):
    """Print structure of a pickle file."""
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"\n  File: {path}")
        print(f"  Type: pkl")
        print(f"  Python type: {type(data).__name__}")
        if isinstance(data, dict):
            print(f"  Keys: {list(data.keys())[:20]}")
            for k, v in list(data.items())[:3]:
                if hasattr(v, "shape"):
                    print(f"    {k}: ndarray shape={v.shape} dtype={v.dtype}")
                else:
                    val_str = str(v)[:100]
                    print(f"    {k}: {type(v).__name__} = {val_str}")
        elif isinstance(data, (list, tuple)):
            print(f"  Length: {len(data)}")
            if len(data) > 0:
                first = data[0]
                if hasattr(first, "shape"):
                    print(f"  [0]: ndarray shape={first.shape} dtype={first.dtype}")
                else:
                    print(f"  [0]: {type(first).__name__} = {str(first)[:100]}")
        else:
            print(f"  Value: {str(data)[:200]}")
    except Exception as e:
        print(f"  Error reading {path}: {e}")


def inspect_json(path: Path):
    """Print structure of a JSON file."""
    import json
    try:
        with open(path) as f:
            data = json.load(f)
        print(f"\n  File: {path}")
        print(f"  Type: json")
        print(f"  Python type: {type(data).__name__}")
        if isinstance(data, dict):
            print(f"  Keys: {list(data.keys())[:20]}")
        elif isinstance(data, list):
            print(f"  Length: {len(data)}")
            if len(data) > 0:
                print(f"  [0]: {str(data[0])[:100]}")
    except Exception as e:
        print(f"  Error reading {path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Inspect CALVIN dataset structure")
    parser.add_argument("--root", type=str, required=True,
                        help="Path to CALVIN dataset root directory")
    parser.add_argument("--depth", type=int, default=3,
                        help="Directory tree depth to print (default: 3)")
    parser.add_argument("--sample_count", type=int, default=5,
                        help="Max sample files to inspect per type")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Error: {root} does not exist")
        print("Download CALVIN first: bash scripts/download_calvin.sh debug")
        return

    print("=" * 70)
    print(f"CALVIN Dataset Inspection")
    print(f"Root: {root}")
    print("=" * 70)

    # 1. Directory tree
    print("\n--- Directory Tree ---")
    print_tree(root, max_depth=args.depth)

    # 2. Find data files by type
    print("\n--- Data Files by Type ---")
    extensions = [".npz", ".npy", ".pkl", ".json", ".txt", ".yaml"]
    found = find_data_files(root, extensions, max_count=args.sample_count)
    for ext, files in found.items():
        if files:
            print(f"\n  {ext}: {len(files)} file(s) found")
            for f in files:
                rel = f.relative_to(root)
                size = _format_size(f.stat().st_size)
                print(f"    {rel}  ({size})")

    # 3. Inspect sample files
    print("\n--- Sample File Contents ---")
    inspectors = {
        ".npz": inspect_npz,
        ".npy": inspect_npy,
        ".pkl": inspect_pkl,
        ".json": inspect_json,
    }
    for ext, files in found.items():
        if ext in inspectors:
            print(f"\n[Inspecting {ext} files]")
            for f in files[:args.sample_count]:
                inspectors[ext](f)

    # 4. Check for CALVIN-specific patterns
    print("\n--- CALVIN Structure Check ---")

    # Check for split directories
    for pattern in ["task_*_*", "env_*"]:
        matches = list(root.glob(pattern))
        if matches:
            print(f"  Pattern '{pattern}': {len(matches)} matches")
            for m in matches[:10]:
                print(f"    {m.name}/")

    # Check for episode-like directories
    ep_dirs = list(root.rglob("episode_*"))
    if ep_dirs:
        print(f"\n  episode_* directories: {len(ep_dirs)} total")
        print(f"  First: {ep_dirs[0].relative_to(root)}")
        # Inspect first episode
        print(f"  Contents of first episode:")
        for f in sorted(ep_dirs[0].iterdir()):
            size = _format_size(f.stat().st_size)
            print(f"    {f.name}  ({size})")
    else:
        print("  No episode_* directories found (data may be in flat npz/npy files)")

    # Check for language files
    lang_files = list(root.rglob("*language*")) + list(root.rglob("*lang*"))
    if lang_files:
        print(f"\n  Language annotation files: {len(lang_files)}")
        for f in lang_files[:3]:
            print(f"    {f.relative_to(root)}")

    print("\n" + "=" * 70)
    print("Inspection complete. Use this output to update:")
    print("  - src/calvin_lerobot/split.py  (directory structure)")
    print("  - src/calvin_lerobot/convert.py (field names and shapes)")
    print("=" * 70)


if __name__ == "__main__":
    main()
