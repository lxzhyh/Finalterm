"""
Convert CALVIN raw episodes to LeRobotDataset v3 format.

LeRobotDataset v3 structure:
    dataset_root/
    ├── meta/
    │   ├── info.json       # feature schema, fps, paths
    │   ├── stats.json      # normalization statistics
    │   └── tasks.jsonl     # task text & task_id mapping
    ├── data/
    │   └── episode_*.parquet   # state, action, timestamps
    └── videos/
        └── episode_*.mp4       # camera video streams

CALVIN episode format:
    Each episode directory contains:
        - rgb_static.npy          (T, H, W, 3) uint8
        - rgb_gripper.npy         (T, H, W, 3) uint8
        - depth_static.npy        (T, H, W)    float32
        - depth_gripper.npy       (T, H, W)    float32
        - robot_obs.npy           (T, state_dim) float32
        - rel_actions.npy         (T, 7) float32
        - scene_obs.npy           (T, scene_dim) float32
        - language.pkl            list of language annotations
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# CALVIN relative cartesian action: [dx, dy, dz, drx, dry, drz, gripper]
ACTION_DIM = 7

# CALVIN proprioceptive state:
#   ee_pos(3) + ee_orn_euler(3) + gripper_width(1) + joint_positions(7) + gripper_action(1) = 15
STATE_DIM = 15


class CalvinToLeRobotConverter:
    """Convert CALVIN episodes into LeRobotDataset v3 format."""

    def __init__(
        self,
        cameras: list[str] = None,
        action_space: str = "relative_cartesian",
        static_size: tuple[int, int] = (200, 200),
        gripper_size: tuple[int, int] = (84, 84),
        fps: int = 30,
    ):
        self.cameras = cameras or ["static", "gripper"]
        self.action_space = action_space
        self.static_size = static_size
        self.gripper_size = gripper_size
        self.fps = fps

    def convert(self, episodes: list[Path], output_dir: Path, environments: list[str]):
        """Convert a list of CALVIN episode paths to LeRobotDataset."""
        output_dir.mkdir(parents=True, exist_ok=True)

        meta_dir = output_dir / "meta"
        data_dir = output_dir / "data"
        videos_dir = output_dir / "videos"
        for d in [meta_dir, data_dir, videos_dir]:
            d.mkdir(exist_ok=True)

        # Build feature schema
        info = self._build_info(environments)
        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        # Collect all tasks across episodes
        all_tasks = {}
        task_id_counter = 0

        # Global stats accumulators
        action_sum = np.zeros(ACTION_DIM, dtype=np.float64)
        action_sq_sum = np.zeros(ACTION_DIM, dtype=np.float64)
        action_min = np.full(ACTION_DIM, np.inf)
        action_max = np.full(ACTION_DIM, -np.inf)
        state_sum = np.zeros(STATE_DIM, dtype=np.float64)
        state_sq_sum = np.zeros(STATE_DIM, dtype=np.float64)
        state_min = np.full(STATE_DIM, np.inf)
        state_max = np.full(STATE_DIM, -np.inf)
        total_frames = 0

        for ep_idx, ep_path in enumerate(tqdm(episodes, desc="Converting episodes")):
            # Load CALVIN data
            rgb_static = np.load(ep_path / "rgb_static.npy")
            rgb_gripper = np.load(ep_path / "rgb_gripper.npy")
            robot_obs = np.load(ep_path / "robot_obs.npy")
            rel_actions = np.load(ep_path / "rel_actions.npy")

            T = len(rgb_static)

            # Resize images if needed
            if "static" in self.cameras:
                rgb_static = self._resize_batch(rgb_static, self.static_size)
            if "gripper" in self.cameras:
                rgb_gripper = self._resize_batch(rgb_gripper, self.gripper_size)

            # Load language annotations
            tasks = self._load_language(ep_path)
            for task_text in tasks:
                if task_text not in all_tasks:
                    all_tasks[task_text] = task_id_counter
                    task_id_counter += 1

            # Assign task_index (use first annotation)
            task_index = all_tasks.get(tasks[0], 0) if tasks else 0

            # Extract environment name from path
            env_name = self._extract_env_name(ep_path)

            # Write video files
            if "static" in self.cameras:
                self._write_video(rgb_static, videos_dir / f"episode_{ep_idx:06d}_static.mp4")
            if "gripper" in self.cameras:
                self._write_video(rgb_gripper, videos_dir / f"episode_{ep_idx:06d}_gripper.mp4")

            # Build parquet table
            timestamps = np.arange(T, dtype=np.float32) / self.fps
            frame_indices = np.arange(T, dtype=np.int64)
            episode_indices = np.full(T, ep_idx, dtype=np.int64)
            task_indices = np.full(T, task_index, dtype=np.int64)
            next_done = np.zeros(T, dtype=np.bool_)
            next_done[-1] = True

            table = pa.table({
                "timestamp": timestamps,
                "frame_index": frame_indices,
                "episode_index": episode_indices,
                "task_index": task_indices,
                "next.done": next_done,
                "observation.state": [row.tolist() for row in robot_obs.astype(np.float32)],
                "action": [row.tolist() for row in rel_actions.astype(np.float32)],
            })

            pq.write_table(table, data_dir / f"episode_{ep_idx:06d}.parquet")

            # Accumulate stats
            actions = rel_actions.astype(np.float64)
            states = robot_obs.astype(np.float64)
            action_sum += actions.sum(axis=0)
            action_sq_sum += (actions ** 2).sum(axis=0)
            action_min = np.minimum(action_min, actions.min(axis=0))
            action_max = np.maximum(action_max, actions.max(axis=0))
            state_sum += states.sum(axis=0)
            state_sq_sum += (states ** 2).sum(axis=0)
            state_min = np.minimum(state_min, states.min(axis=0))
            state_max = np.maximum(state_max, states.max(axis=0))
            total_frames += T

        # Write tasks.jsonl
        with open(meta_dir / "tasks.jsonl", "w") as f:
            for task_text, tid in sorted(all_tasks.items(), key=lambda x: x[1]):
                f.write(json.dumps({"task_index": tid, "task": task_text}) + "\n")

        # Write stats.json
        n = max(total_frames, 1)
        action_mean = action_sum / n
        action_std = np.sqrt(action_sq_sum / n - action_mean ** 2)
        state_mean = state_sum / n
        state_std = np.sqrt(state_sq_sum / n - state_mean ** 2)

        stats = {
            "observation.state": {
                "mean": state_mean.tolist(),
                "std": state_std.tolist(),
                "min": state_min.tolist(),
                "max": state_max.tolist(),
            },
            "action": {
                "mean": action_mean.tolist(),
                "std": action_std.tolist(),
                "min": action_min.tolist(),
                "max": action_max.tolist(),
            },
        }
        with open(meta_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"  Converted {len(episodes)} episodes, {total_frames} frames")

    def _build_info(self, environments: list[str]) -> dict:
        """Build LeRobotDataset info.json."""
        features = {
            "timestamp": {"dtype": "float32", "shape": (1,)},
            "frame_index": {"dtype": "int64", "shape": (1,)},
            "episode_index": {"dtype": "int64", "shape": (1,)},
            "task_index": {"dtype": "int64", "shape": (1,)},
            "next.done": {"dtype": "bool", "shape": (1,)},
            "observation.state": {"dtype": "float32", "shape": (STATE_DIM,)},
            "action": {"dtype": "float32", "shape": (ACTION_DIM,)},
        }

        if "static" in self.cameras:
            h, w = self.static_size
            features["observation.images.static"] = {
                "dtype": "video",
                "shape": (3, h, w),
                "names": ["channels", "height", "width"],
                "video.fps": self.fps,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
            }
        if "gripper" in self.cameras:
            h, w = self.gripper_size
            features["observation.images.gripper"] = {
                "dtype": "video",
                "shape": (3, h, w),
                "names": ["channels", "height", "width"],
                "video.fps": self.fps,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
            }

        return {
            "codebase_version": "v3",
            "robot_type": "calvin",
            "fps": self.fps,
            "chunks_size": 1,
            "total_episodes": 0,  # filled after conversion
            "total_frames": 0,
            "total_tasks": 0,
            "total_chunks": 0,
            "features": features,
            "environments": environments,
            "action_space": self.action_space,
        }

    def _resize_batch(self, images: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """Resize a batch of images (T, H, W, 3) -> (T, H', W', 3)."""
        h, w = target_size
        if images.shape[1] == h and images.shape[2] == w:
            return images
        resized = np.zeros((len(images), h, w, 3), dtype=np.uint8)
        for i, img in enumerate(images):
            resized[i] = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return resized

    def _write_video(self, frames: np.ndarray, path: Path):
        """Write RGB frames as MP4 video."""
        h, w = frames.shape[1], frames.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self.fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    def _load_language(self, ep_path: Path) -> list[str]:
        """Load language annotations from CALVIN episode."""
        lang_file = ep_path / "language.pkl"
        if lang_file.exists():
            import pickle
            with open(lang_file, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, list):
                return [str(x) for x in data]
            return [str(data)]
        return ["unlabeled_task"]

    def _extract_env_name(self, ep_path: Path) -> str:
        """Extract environment name (A/B/C/D) from episode path."""
        # CALVIN data layout: .../task_D_D/episode_XXXXX or .../env_A/...
        parts = ep_path.parts
        for part in parts:
            for env in ["A", "B", "C", "D"]:
                if f"_{env}_" in part or part.endswith(f"_{env}") or f"env_{env}" in part:
                    return env
        return "unknown"
