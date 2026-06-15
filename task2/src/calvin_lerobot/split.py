"""
Episode-level data splitting for CALVIN environments.

Splitting rules:
    - Unit: episode (NOT frame — prevents data leakage)
    - Each environment split independently: 90% train, 10% val
    - Env-D: 100% test (no train/val)
    - Seed fixed for reproducibility

CALVIN directory structure (verified via tools/inspect_calvin_dataset.py):
    calvin_root/
    ├── task_D_D/          # D-only split for evaluation
    │   ├── episode_000000/
    │   ├── episode_000001/
    │   └── ...
    ├── task_ABC_D/        # ABC split for training
    │   ├── episode_000000/
    │   └── ...
    └── debug/             # debug split
        ├── task_D_D/
        └── task_ABC_D/

Each episode directory contains:
    rgb_static.npy, rgb_gripper.npy, robot_obs.npy, rel_actions.npy, etc.

Environment assignment:
    - task_D_D → Env-D (test only)
    - task_ABC_D → Env-A/B/C (train/val)
    - Environment label extracted from episode metadata or path

IMPORTANT:
    Env-D data must NEVER enter train or val sets.
    This is enforced by hard assertions.
"""

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class EpisodeInfo:
    """Metadata for a single CALVIN episode."""
    env: str              # "A", "B", "C", or "D"
    episode_id: str       # e.g., "episode_000000"
    path: Path            # full path to episode directory
    source_split: str     # original CALVIN split name, e.g., "task_ABC_D"


class EpisodeSplitter:
    """
    Split CALVIN episodes with strict environment isolation.

    Usage:
        splitter = EpisodeSplitter(seed=42, val_ratio=0.1)
        splits = splitter.split(calvin_root, train_envs=["A", "B", "C"], test_env="D")

        # splits["train"] → list[EpisodeInfo] for A/B/C (90%)
        # splits["val"]   → list[EpisodeInfo] for A/B/C (10%)
        # splits["test"]  → list[EpisodeInfo] for D (100%)
    """

    def __init__(self, seed: int = 42, val_ratio: float = 0.1):
        self.seed = seed
        self.val_ratio = val_ratio
        self.rng = random.Random(seed)

    def split(
        self,
        calvin_root: Path,
        train_envs: list[str] = None,
        test_env: str = "D",
        max_episodes_per_env: Optional[int] = None,
    ) -> dict[str, list[EpisodeInfo]]:
        """
        Main entry point: split episodes into train/val/test.

        Args:
            calvin_root: path to CALVIN dataset root
            train_envs: environments for training (default: ["A", "B", "C"])
            test_env: environment for testing (default: "D")
            max_episodes_per_env: limit episodes per env (for debug runs)

        Returns:
            {"train": [...], "val": [...], "test": [...]}
        """
        train_envs = train_envs or ["A", "B", "C"]
        calvin_root = Path(calvin_root)

        if not calvin_root.exists():
            raise FileNotFoundError(f"CALVIN root not found: {calvin_root}")

        # Discover all episodes and assign environments
        all_episodes = self._discover_episodes(calvin_root)

        if not all_episodes:
            raise ValueError(
                f"No episodes found under {calvin_root}. "
                f"Run: python tools/inspect_calvin_dataset.py --root {calvin_root}"
            )

        # Group by environment
        by_env: dict[str, list[EpisodeInfo]] = {}
        for ep in all_episodes:
            by_env.setdefault(ep.env, []).append(ep)

        print(f"Discovered episodes by environment:")
        for env in sorted(by_env.keys()):
            print(f"  Env-{env}: {len(by_env[env])} episodes")

        # Apply per-env limit (for debug)
        if max_episodes_per_env is not None:
            for env in by_env:
                self.rng.shuffle(by_env[env])
                by_env[env] = sorted(by_env[env][:max_episodes_per_env], key=lambda e: e.episode_id)

        # Split train environments
        train_episodes = []
        val_episodes = []
        for env in train_envs:
            if env not in by_env:
                print(f"  Warning: Env-{env} not found in dataset, skipping")
                continue
            eps = by_env[env]
            tr, va = self._split_episodes(eps)
            train_episodes.extend(tr)
            val_episodes.extend(va)
            print(f"  Env-{env}: {len(tr)} train, {len(va)} val")

        # Test environment (100% test, no train/val)
        test_episodes = by_env.get(test_env, [])
        if not test_episodes:
            print(f"  Warning: Env-{test_env} not found in dataset")
        else:
            print(f"  Env-{test_env}: {len(test_episodes)} test (all episodes)")

        # Hard safety assertions
        self._assert_env_isolation(train_episodes, val_episodes, test_episodes, test_env)

        return {
            "train": train_episodes,
            "val": val_episodes,
            "test": test_episodes,
        }

    def _discover_episodes(self, calvin_root: Path) -> list[EpisodeInfo]:
        """
        Find all episodes under calvin_root and assign environment labels.

        Searches for directories named episode_* at any depth.
        Environment is inferred from parent directory structure.
        """
        episodes = []

        # Find all episode directories
        ep_dirs = sorted(calvin_root.rglob("episode_*"))
        # Filter to actual directories (not files)
        ep_dirs = [d for d in ep_dirs if d.is_dir()]

        for ep_dir in ep_dirs:
            env = self._infer_environment(ep_dir, calvin_root)
            if env is None:
                continue  # skip episodes we can't classify

            # Extract episode_id from directory name
            episode_id = ep_dir.name

            # Determine source split from parent path
            source_split = self._infer_source_split(ep_dir, calvin_root)

            episodes.append(EpisodeInfo(
                env=env,
                episode_id=episode_id,
                path=ep_dir,
                source_split=source_split,
            ))

        return episodes

    def _infer_environment(self, ep_dir: Path, calvin_root: Path) -> Optional[str]:
        """
        Infer environment label (A/B/C/D) from episode path.

        Strategy (in priority order):
        1. Check for environment marker in parent directory names
        2. Check for CALVIN split naming convention (task_D_D, task_ABC_D)
        3. Check inside episode for metadata file

        CALVIN split naming:
            task_D_D    → all episodes are Env-D
            task_ABC_D  → episodes are Env-A/B/C (need further assignment)
        """
        rel_path = ep_dir.relative_to(calvin_root)
        parts = list(rel_path.parts)

        # Strategy 1: Look for task_*_* pattern in parent dirs
        for part in parts[:-1]:  # exclude the episode dir itself
            match = re.match(r"task_([A-D]+)_([A-D]+)$", part)
            if match:
                train_envs = match.group(1)  # e.g., "ABC" or "D"
                # If training envs is just "D", all episodes are D
                if train_envs == "D":
                    return "D"
                # Otherwise, episodes belong to one of the training envs
                # Need to determine which one from episode content or index
                return self._assign_env_from_multi(ep_dir, train_envs)

        # Strategy 2: Look for env_A, env_B, etc. directories
        for part in parts[:-1]:
            match = re.match(r"env_([A-D])$", part)
            if match:
                return match.group(1)

        # Strategy 3: Check if we're directly under calvin_root
        # (some CALVIN versions have flat episode dirs)
        # In this case, we can't determine environment reliably
        return None

    def _assign_env_from_multi(self, ep_dir: Path, envs: str) -> str:
        """
        When a split contains multiple environments (e.g., "ABC"),
        assign each episode to a specific environment.

        Approach: use episode index modulo to distribute evenly,
        or check episode metadata if available.
        """
        # Try to read environment from episode metadata
        for meta_file in ["env.txt", "metadata.json", "info.json"]:
            meta_path = ep_dir / meta_file
            if meta_path.exists():
                try:
                    text = meta_path.read_text().strip()
                    for env_char in envs:
                        if env_char in text:
                            return env_char
                except Exception:
                    pass

        # Fallback: use episode index for round-robin assignment
        # WARNING: This is a heuristic and may not reflect true environment labels
        ep_num = self._extract_episode_number(ep_dir.name)
        env_list = list(envs)
        assigned_env = env_list[ep_num % len(env_list)]

        # Log first few assignments for verification
        if ep_num < 10:
            print(f"    [WARNING] No metadata found for {ep_dir.name}, "
                  f"assigned to Env-{assigned_env} (modulo assignment)")

        return assigned_env

    def _infer_source_split(self, ep_dir: Path, calvin_root: Path) -> str:
        """Get the CALVIN split name from the path."""
        rel_path = ep_dir.relative_to(calvin_root)
        parts = list(rel_path.parts)
        for part in parts[:-1]:
            if part.startswith("task_"):
                return part
        return parts[0] if len(parts) > 1 else "unknown"

    def _extract_episode_number(self, name: str) -> int:
        """Extract numeric index from episode directory name."""
        match = re.search(r"(\d+)", name)
        return int(match.group(1)) if match else 0

    def _split_episodes(self, episodes: list[EpisodeInfo]) -> tuple[list[EpisodeInfo], list[EpisodeInfo]]:
        """Split episodes into train/val at episode level."""
        if not episodes:
            return [], []

        shuffled = list(episodes)
        self.rng.shuffle(shuffled)

        n_val = max(1, int(len(shuffled) * self.val_ratio))
        val_eps = sorted(shuffled[:n_val], key=lambda e: e.episode_id)
        train_eps = sorted(shuffled[n_val:], key=lambda e: e.episode_id)

        return train_eps, val_eps

    def _assert_env_isolation(
        self,
        train: list[EpisodeInfo],
        val: list[EpisodeInfo],
        test: list[EpisodeInfo],
        test_env: str,
    ):
        """Hard assertion: test environment never leaks into train/val."""
        train_envs = {ep.env for ep in train}
        val_envs = {ep.env for ep in val}
        test_envs = {ep.env for ep in test}

        assert test_env not in train_envs, (
            f"FATAL: Env-{test_env} found in train set! "
            f"Train environments: {train_envs}"
        )
        assert test_env not in val_envs, (
            f"FATAL: Env-{test_env} found in val set! "
            f"Val environments: {val_envs}"
        )

        # Also verify test set only contains test env
        if test_envs:
            assert test_envs == {test_env}, (
                f"FATAL: Test set contains non-test environments: "
                f"{test_envs - {test_env}}"
            )


def main():
    """CLI entry for testing the splitter."""
    import argparse

    parser = argparse.ArgumentParser(description="Test CALVIN episode splitting")
    parser.add_argument("--calvin_root", type=str, required=True)
    parser.add_argument("--train_envs", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--test_env", type=str, default="D")
    parser.add_argument("--max_episodes_per_env", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    splitter = EpisodeSplitter(seed=args.seed, val_ratio=args.val_ratio)
    splits = splitter.split(
        calvin_root=Path(args.calvin_root),
        train_envs=args.train_envs,
        test_env=args.test_env,
        max_episodes_per_env=args.max_episodes_per_env,
    )

    print(f"\n=== Split Summary ===")
    print(f"  Train: {len(splits['train'])} episodes")
    print(f"  Val:   {len(splits['val'])} episodes")
    print(f"  Test:  {len(splits['test'])} episodes")

    if splits["train"]:
        print(f"\n  First train episode: {splits['train'][0]}")
    if splits["test"]:
        print(f"  First test episode:  {splits['test'][0]}")


if __name__ == "__main__":
    main()
