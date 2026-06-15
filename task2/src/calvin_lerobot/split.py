"""
Episode-level data splitting for CALVIN environments.

Splitting rules:
    - Unit: episode (NOT frame — prevents data leakage)
    - Each environment split independently: 90% train, 10% val
    - Env-D: 100% test (no train/val)
    - Seed fixed for reproducibility
"""

import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm


class EpisodeSplitter:
    """Split CALVIN episodes at the episode level with environment isolation."""

    def __init__(self, seed: int = 42, val_ratio: float = 0.1):
        self.seed = seed
        self.val_ratio = val_ratio
        self.rng = random.Random(seed)

    def split_environments(
        self,
        calvin_root: Path,
        environments: list[str],
        max_episodes_per_env: Optional[int] = None,
    ) -> dict[str, dict[str, list[Path]]]:
        """
        Split episodes across multiple CALVIN environments.

        Returns:
            {
                "A": {"train": [ep_paths...], "val": [ep_paths...]},
                "B": {"train": [ep_paths...], "val": [ep_paths...]},
                ...
            }
        """
        result = {}
        for env in environments:
            episodes = self._find_episodes(calvin_root, env)

            if max_episodes_per_env is not None:
                self.rng.shuffle(episodes)
                episodes = sorted(episodes[:max_episodes_per_env])

            train_eps, val_eps = self._split_episodes(episodes)
            result[env] = {"train": train_eps, "val": val_eps}
            print(f"  Env-{env}: {len(train_eps)} train, {len(val_eps)} val "
                  f"(total {len(episodes)} episodes)")

        return result

    def _find_episodes(self, calvin_root: Path, env: str) -> list[Path]:
        """Find all episode directories for a given CALVIN environment."""
        # CALVIN data layout patterns:
        #   calvin_root/task_D_D/episode_00000/
        #   calvin_root/task_ABC_D/env_A/episode_00000/
        #   calvin_root/debug/task_D_D/episode_00000/
        candidates = []

        # Pattern 1: task_{env}_{env} directories
        for pattern in [f"task_{env}_{env}", f"task_*_{env}"]:
            for task_dir in calvin_root.glob(pattern):
                if task_dir.is_dir():
                    eps = sorted([p for p in task_dir.iterdir() if p.is_dir()])
                    candidates.extend(eps)

        # Pattern 2: env_{env} subdirectories
        for env_dir in calvin_root.glob(f"env_{env}"):
            if env_dir.is_dir():
                eps = sorted([p for p in env_dir.iterdir() if p.is_dir()])
                candidates.extend(eps)

        # Pattern 3: flat episode directories directly under calvin_root
        if not candidates:
            eps = sorted([
                p for p in calvin_root.iterdir()
                if p.is_dir() and p.name.startswith("episode_")
            ])
            candidates.extend(eps)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for ep in candidates:
            if ep.resolve() not in seen:
                seen.add(ep.resolve())
                unique.append(ep)

        return unique

    def _split_episodes(self, episodes: list[Path]) -> tuple[list[Path], list[Path]]:
        """Split episodes into train/val at the episode level."""
        if not episodes:
            return [], []

        shuffled = list(episodes)
        self.rng.shuffle(shuffled)

        n_val = max(1, int(len(shuffled) * self.val_ratio))
        val_episodes = sorted(shuffled[:n_val])
        train_episodes = sorted(shuffled[n_val:])

        return train_episodes, val_episodes
