"""
Generate synthetic CALVIN-format data for pipeline testing.

Creates a minimal dataset with the correct directory structure and file formats
to verify the full conversion + training pipeline works end-to-end.

Usage:
    python tools/generate_synthetic_calvin.py --output_dir third_party/calvin/dataset
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np


# CALVIN data specs
STATIC_H, STATIC_W = 200, 200
GRIPPER_H, GRIPPER_W = 84, 84
STATE_DIM = 15  # ee_pos(3) + ee_orn(3) + gripper_width(1) + joints(7) + gripper_act(1)
ACTION_DIM = 7  # dx, dy, dz, drx, dry, drz, gripper
FPS = 30
FRAMES_PER_EPISODE = 50  # ~1.67 seconds at 30fps


def generate_episode(ep_dir: Path, ep_idx: int, env_label: str):
    """Generate a single synthetic episode."""
    ep_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(ep_idx)

    # Generate synthetic RGB images with environment-specific color tint
    env_colors = {"A": (100, 150, 200), "B": (200, 100, 150), "C": (150, 200, 100), "D": (180, 180, 100)}
    base_color = env_colors.get(env_label, (128, 128, 128))

    static_frames = np.zeros((FRAMES_PER_EPISODE, STATIC_H, STATIC_W, 3), dtype=np.uint8)
    gripper_frames = np.zeros((FRAMES_PER_EPISODE, GRIPPER_H, GRIPPER_W, 3), dtype=np.uint8)

    for t in range(FRAMES_PER_EPISODE):
        # Static camera: background + moving "object"
        static_frames[t] = base_color
        # Add a moving rectangle to simulate robot arm
        x = int(80 + 40 * np.sin(t / 10.0 + ep_idx))
        y = int(80 + 40 * np.cos(t / 10.0 + ep_idx))
        static_frames[t, max(0, y-10):y+10, max(0, x-10):x+10] = (255, 50, 50)

        # Gripper camera: close-up view
        gripper_frames[t] = (base_color[0] // 2, base_color[1] // 2, base_color[2] // 2)
        # Add gripper fingers
        gripper_frames[t, 30:50, 20:30] = (200, 200, 200)
        gripper_frames[t, 30:50, 54:64] = (200, 200, 200)

    # Generate synthetic robot observations
    robot_obs = np.zeros((FRAMES_PER_EPISODE, STATE_DIM), dtype=np.float32)
    for t in range(FRAMES_PER_EPISODE):
        robot_obs[t, 0:3] = [0.5 + 0.1 * np.sin(t / 10.0), 0.3, 0.4]  # ee_pos
        robot_obs[t, 3:6] = [0.0, 0.0, 0.0]  # ee_orn_euler
        robot_obs[t, 6] = 0.04 + 0.02 * np.sin(t / 5.0)  # gripper_width
        robot_obs[t, 7:14] = rng.randn(7) * 0.1  # joint_positions
        robot_obs[t, 14] = 1.0 if t % 20 < 10 else 0.0  # gripper_action

    # Generate synthetic relative actions
    rel_actions = np.zeros((FRAMES_PER_EPISODE, ACTION_DIM), dtype=np.float32)
    for t in range(FRAMES_PER_EPISODE):
        rel_actions[t, 0:3] = [0.01 * np.cos(t / 10.0), 0.005, 0.01 * np.sin(t / 10.0)]  # dpos
        rel_actions[t, 3:6] = [0.0, 0.0, 0.0]  # dorn
        rel_actions[t, 6] = 1.0 if t % 20 < 10 else -1.0  # gripper

    # Save files
    np.save(ep_dir / "rgb_static.npy", static_frames)
    np.save(ep_dir / "rgb_gripper.npy", gripper_frames)
    np.save(ep_dir / "robot_obs.npy", robot_obs)
    np.save(ep_dir / "rel_actions.npy", rel_actions)

    # Language annotation
    tasks = ["pick up the red block", "place the block on the table", "push the object left"]
    lang_data = {t: {"lang": [tasks[t % len(tasks)]]} for t in range(FRAMES_PER_EPISODE)}
    with open(ep_dir / "lang_annotations.pkl", "wb") as f:
        pickle.dump(lang_data, f)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CALVIN data")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory (e.g., third_party/calvin/dataset)")
    parser.add_argument("--episodes_per_env", type=int, default=10,
                        help="Number of episodes per environment")
    args = parser.parse_args()

    output = Path(args.output_dir)

    # Generate task_ABC_D split (training environments A, B, C)
    abc_dir = output / "task_ABC_D"
    env_labels = ["A", "B", "C"]
    for i in range(args.episodes_per_env * len(env_labels)):
        env_label = env_labels[i % len(env_labels)]
        ep_dir = abc_dir / f"episode_{i:06d}"
        generate_episode(ep_dir, i, env_label)
        print(f"  Generated {ep_dir.name} (Env-{env_label})")

    print(f"\nGenerated {args.episodes_per_env * 3} episodes in {abc_dir}")

    # Generate task_D_D split (test environment D)
    d_dir = output / "task_D_D"
    for i in range(args.episodes_per_env):
        ep_dir = d_dir / f"episode_{i:06d}"
        generate_episode(ep_dir, i + 1000, "D")
        print(f"  Generated {ep_dir.name} (Env-D)")

    print(f"\nGenerated {args.episodes_per_env} episodes in {d_dir}")
    print(f"\nDone! Synthetic CALVIN data at: {output}")
    print(f"\nNext steps:")
    print(f"  python tools/inspect_calvin_dataset.py --root {output}")
    print(f"  python scripts/convert_calvin_to_lerobot.py \\")
    print(f"    --calvin_root {output} --output_root data/lerobot_calvin --splits A ABC D")


if __name__ == "__main__":
    main()
