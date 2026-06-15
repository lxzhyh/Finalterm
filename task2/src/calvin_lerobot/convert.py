"""
Convert CALVIN raw episodes to LeRobotDataset format using official API.

IMPORTANT: This script uses LeRobot's dataset creation API.
The exact API signatures may vary between versions. Run the verification
step below after installation to confirm:

    python -c "
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import inspect
    print(inspect.signature(LeRobotDataset.create))
    "

Expected API (LeRobot >= 0.5):
    dataset = LeRobotDataset.create(
        repo_id="calvin_A_train",
        fps=30,
        features={...},
        root="data/lerobot_calvin/calvin_A_train",
    )
    dataset.add_frame({...})
    dataset.save_episode()

If the API differs, this script will print the actual signatures and exit
with instructions for manual adjustment.

Usage:
    python scripts/convert_calvin_to_lerobot.py \
        --calvin_root third_party/calvin/dataset \
        --output_root data/lerobot_calvin \
        --splits A ABC D \
        --val_ratio 0.1 \
        --seed 42

    # Dry run (verify API without converting):
    python scripts/convert_calvin_to_lerobot.py --check_api
"""

import argparse
import inspect
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from calvin_lerobot.split import EpisodeInfo, EpisodeSplitter


# ─────────────────────────────────────────────────────────
# LeRobot API Discovery
# ─────────────────────────────────────────────────────────

def discover_lerobot_api():
    """
    Discover the actual LeRobot dataset creation API.
    Returns a dict of available methods and their signatures.
    """
    api_info = {"available": False, "methods": {}, "errors": []}

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        api_info["available"] = True
        api_info["class"] = LeRobotDataset

        # Check for creation methods
        for method_name in ["create", "from_dict", "__init__"]:
            method = getattr(LeRobotDataset, method_name, None)
            if method is not None:
                try:
                    sig = inspect.signature(method)
                    api_info["methods"][method_name] = str(sig)
                except (ValueError, TypeError):
                    api_info["methods"][method_name] = "signature unknown"

        # Check for frame/episode methods
        for method_name in ["add_frame", "save_episode", "save_video",
                           "push_to_hub", "from_preloaded"]:
            method = getattr(LeRobotDataset, method_name, None)
            if method is not None:
                try:
                    sig = inspect.signature(method)
                    api_info["methods"][method_name] = str(sig)
                except (ValueError, TypeError):
                    api_info["methods"][method_name] = "signature unknown"

    except ImportError as e:
        api_info["errors"].append(f"Cannot import LeRobotDataset: {e}")
    except Exception as e:
        api_info["errors"].append(f"Error discovering API: {e}")

    return api_info


def print_api_report(api_info: dict):
    """Print a report of discovered API methods."""
    print("=" * 60)
    print("LeRobot Dataset API Discovery Report")
    print("=" * 60)

    if not api_info["available"]:
        print("\nLeRobot is NOT available.")
        for err in api_info["errors"]:
            print(f"  Error: {err}")
        print("\nInstall lerobot:")
        print('  pip install "lerobot[aloha]"')
        return False

    print(f"\nLeRobotDataset class found.")
    print(f"\nAvailable methods:")
    for name, sig in sorted(api_info["methods"].items()):
        print(f"  {name}{sig}")

    # Check critical methods
    required = ["create", "add_frame", "save_episode"]
    missing = [m for m in required if m not in api_info["methods"]]
    if missing:
        print(f"\nWARNING: Missing expected methods: {missing}")
        print("The LeRobot API may have changed. Check the documentation:")
        print("  https://huggingface.co/docs/lerobot")
        return False

    print("\nAll required methods found. Ready to convert.")
    return True


# ─────────────────────────────────────────────────────────
# CALVIN Data Reading
# ─────────────────────────────────────────────────────────

# CALVIN action: [dx, dy, dz, drx, dry, drz, gripper]
ACTION_DIM = 7

# CALVIN state: ee_pos(3) + ee_orn(3) + gripper_width(1) + joints(7) + gripper_act(1)
STATE_DIM = 15


def read_calvin_episode(ep_info: EpisodeInfo) -> dict:
    """
    Read a single CALVIN episode into memory.

    Returns:
        {
            "rgb_static": np.ndarray (T, H, W, 3) uint8,
            "rgb_gripper": np.ndarray (T, H, W, 3) uint8,
            "robot_obs": np.ndarray (T, STATE_DIM) float32,
            "rel_actions": np.ndarray (T, ACTION_DIM) float32,
            "language": list[str],
        }
    """
    ep_dir = ep_info.path

    # Required files
    rgb_static = np.load(ep_dir / "rgb_static.npy")
    rgb_gripper = np.load(ep_dir / "rgb_gripper.npy")
    robot_obs = np.load(ep_dir / "robot_obs.npy")
    rel_actions = np.load(ep_dir / "rel_actions.npy")

    # Optional: language annotations
    language = []
    for lang_file in ["language.pkl", "lang_annotations.pkl"]:
        lang_path = ep_dir / lang_file
        if lang_path.exists():
            import pickle
            with open(lang_path, "rb") as f:
                lang_data = pickle.load(f)
            if isinstance(lang_data, dict):
                # CALVIN format: {ind: {lang: [annotations]}}
                for ind_data in lang_data.values():
                    if isinstance(ind_data, dict) and "lang" in ind_data:
                        language.extend(ind_data["lang"])
            elif isinstance(lang_data, (list, tuple)):
                language.extend([str(x) for x in lang_data])
            break

    if not language:
        language = ["unlabeled_task"]

    return {
        "rgb_static": rgb_static.astype(np.uint8),
        "rgb_gripper": rgb_gripper.astype(np.uint8),
        "robot_obs": robot_obs.astype(np.float32),
        "rel_actions": rel_actions.astype(np.float32),
        "language": language,
    }


def resize_images(images: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize (T, H, W, 3) images to target size."""
    h, w = target_hw
    if images.shape[1] == h and images.shape[2] == w:
        return images
    resized = np.zeros((len(images), h, w, 3), dtype=images.dtype)
    for i in range(len(images)):
        resized[i] = cv2.resize(images[i], (w, h), interpolation=cv2.INTER_AREA)
    return resized


# ─────────────────────────────────────────────────────────
# LeRobot Conversion
# ─────────────────────────────────────────────────────────

def define_features(
    static_hw: tuple[int, int] = (200, 200),
    gripper_hw: tuple[int, int] = (84, 84),
    state_dim: int = STATE_DIM,
    action_dim: int = ACTION_DIM,
    fps: int = 30,
) -> dict:
    """
    Define LeRobot feature schema.

    IMPORTANT: Verify these match your LeRobot version's expected format.
    Check with:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset("some_existing_dataset")
        print(ds.features)
    """
    return {
        "observation.images.static": {
            "dtype": "image",
            "shape": (3, *static_hw),
            "names": ["channels", "height", "width"],
        },
        "observation.images.gripper": {
            "dtype": "image",
            "shape": (3, *gripper_hw),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
        },
    }


def convert_episodes(
    episodes: list[EpisodeInfo],
    repo_id: str,
    output_root: Path,
    static_hw: tuple[int, int] = (200, 200),
    gripper_hw: tuple[int, int] = (84, 84),
    fps: int = 30,
):
    """
    Convert CALVIN episodes to LeRobotDataset using official API.

    Args:
        episodes: list of EpisodeInfo from splitter
        repo_id: dataset identifier (e.g., "calvin_A_train")
        output_root: base output directory
        static_hw: static camera image size (H, W)
        gripper_hw: gripper camera image size (H, W)
        fps: frames per second
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    features = define_features(static_hw, gripper_hw, fps=fps)
    output_dir = output_root / repo_id

    # Create dataset using official API
    # NOTE: If this fails, check API signature with:
    #   python scripts/convert_calvin_to_lerobot.py --check_api
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=str(output_dir),
    )

    print(f"Created dataset: {repo_id}")
    print(f"  Output: {output_dir}")
    print(f"  Episodes to convert: {len(episodes)}")

    for ep_idx, ep_info in enumerate(tqdm(episodes, desc="Converting")):
        data = read_calvin_episode(ep_info)

        # Resize images
        static_imgs = resize_images(data["rgb_static"], static_hw)  # (T, H, W, 3)
        gripper_imgs = resize_images(data["rgb_gripper"], gripper_hw)  # (T, H, W, 3)

        T = len(static_imgs)
        task_text = data["language"][0] if data["language"] else "unlabeled_task"

        # Add frames one by one
        for t in range(T):
            # Convert (H, W, 3) -> (3, H, W) and normalize to [0, 1]
            static_img = static_imgs[t].transpose(2, 0, 1).astype(np.float32) / 255.0
            gripper_img = gripper_imgs[t].transpose(2, 0, 1).astype(np.float32) / 255.0

            frame = {
                "observation.images.static": static_img,
                "observation.images.gripper": gripper_img,
                "observation.state": data["robot_obs"][t],
                "action": data["rel_actions"][t],
                "task": task_text,
            }
            dataset.add_frame(frame)

        dataset.save_episode()

        if (ep_idx + 1) % 10 == 0 or ep_idx == len(episodes) - 1:
            print(f"  Converted {ep_idx + 1}/{len(episodes)} episodes")

    print(f"\nDataset complete: {len(dataset)} frames from {len(episodes)} episodes")
    return dataset


def verify_dataset(repo_id: str, output_root: Path):
    """
    Verify the converted dataset can be loaded by LeRobot.

    This is a critical check — if this fails, the training pipeline
    will also fail.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    output_dir = output_root / repo_id

    try:
        ds = LeRobotDataset(repo_id, root=str(output_dir))
        print(f"\n=== Verification: {repo_id} ===")
        print(f"  Loaded successfully!")
        print(f"  Total frames: {len(ds)}")

        if len(ds) > 0:
            sample = ds[0]
            print(f"  Sample keys: {list(sample.keys())}")
            for key, value in sample.items():
                if hasattr(value, "shape"):
                    print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"    {key}: {type(value).__name__} = {str(value)[:50]}")
        return True

    except Exception as e:
        print(f"\n=== Verification FAILED: {repo_id} ===")
        print(f"  Error: {e}")
        print(f"\n  Possible causes:")
        print(f"    1. LeRobot API version mismatch")
        print(f"    2. Incorrect feature schema")
        print(f"    3. Missing required metadata files")
        print(f"\n  Debug steps:")
        print(f"    python scripts/convert_calvin_to_lerobot.py --check_api")
        print(f"    ls -la {output_dir}/meta/")
        print(f"    cat {output_dir}/meta/info.json")
        return False


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Convert CALVIN to LeRobotDataset")
    parser.add_argument("--calvin_root", type=str,
                        help="Path to CALVIN dataset root")
    parser.add_argument("--output_root", type=str, default="data/lerobot_calvin",
                        help="Output directory for LeRobot datasets")
    parser.add_argument("--splits", nargs="+", default=["A", "ABC", "D"],
                        help="Dataset splits to create")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_episodes_per_env", type=int, default=None,
                        help="Limit episodes per env (for debug)")
    parser.add_argument("--static_size", type=int, nargs=2, default=[200, 200])
    parser.add_argument("--gripper_size", type=int, nargs=2, default=[84, 84])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--check_api", action="store_true",
                        help="Only check LeRobot API, don't convert")
    parser.add_argument("--skip_verify", action="store_true",
                        help="Skip dataset verification after conversion")
    return parser.parse_args()


def main():
    args = parse_args()

    # API check mode
    if args.check_api:
        api_info = discover_lerobot_api()
        print_api_report(api_info)
        return

    if not args.calvin_root:
        print("Error: --calvin_root is required for conversion")
        print("Use --check_api to verify LeRobot installation first")
        sys.exit(1)

    # Verify LeRobot is available
    api_info = discover_lerobot_api()
    if not print_api_report(api_info):
        print("\nCannot proceed without LeRobot. Install and retry.")
        sys.exit(1)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Split episodes
    splitter = EpisodeSplitter(seed=args.seed, val_ratio=args.val_ratio)
    static_hw = tuple(args.static_size)
    gripper_hw = tuple(args.gripper_size)

    for split_name in args.splits:
        envs = list(split_name)
        is_test_only = (split_name == "D")

        print(f"\n{'='*60}")
        print(f"Processing split: {split_name} (envs={envs})")
        print(f"{'='*60}")

        # Determine train/val/test environments
        if is_test_only:
            splits = splitter.split(
                calvin_root=Path(args.calvin_root),
                train_envs=[],       # no training
                test_env="D",
                max_episodes_per_env=args.max_episodes_per_env,
            )
            # For D-only split, use test episodes as the dataset
            episodes = splits["test"]
            repo_id = f"calvin_D_test"
        else:
            splits = splitter.split(
                calvin_root=Path(args.calvin_root),
                train_envs=envs,
                test_env="D",
                max_episodes_per_env=args.max_episodes_per_env,
            )
            episodes = splits["train"]
            repo_id = f"calvin_{split_name}_train"

        if not episodes:
            print(f"  No episodes found for {split_name}, skipping")
            continue

        # Convert train episodes
        print(f"\nConverting {len(episodes)} episodes -> {repo_id}")
        convert_episodes(
            episodes=episodes,
            repo_id=repo_id,
            output_root=output_root,
            static_hw=static_hw,
            gripper_hw=gripper_hw,
            fps=args.fps,
        )

        # Verify
        if not args.skip_verify:
            verify_dataset(repo_id, output_root)

        # Convert val episodes (skip for D)
        if not is_test_only and splits["val"]:
            val_repo_id = f"calvin_{split_name}_val"
            print(f"\nConverting {len(splits['val'])} val episodes -> {val_repo_id}")
            convert_episodes(
                episodes=splits["val"],
                repo_id=val_repo_id,
                output_root=output_root,
                static_hw=static_hw,
                gripper_hw=gripper_hw,
                fps=args.fps,
            )
            if not args.skip_verify:
                verify_dataset(val_repo_id, output_root)

    print(f"\n{'='*60}")
    print(f"All conversions complete!")
    print(f"Output directory: {args.output_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
