"""
Rollout evaluation on CALVIN Env-D using closed-loop simulation.

This requires a working CALVIN installation:
    git clone --recurse-submodules https://github.com/mees/calvin.git third_party/calvin
    cd third_party/calvin && sh install.sh

CALVIN evaluation uses a CustomModel interface:
    class CustomModel:
        def reset(self):           # called at episode start
            ...
        def step(self, obs, goal): # returns action
            ...

This script wraps the LeRobot ACT policy to implement that interface,
with proper action chunking buffer management.

Usage:
    python -m src.calvin_lerobot.rollout_eval \
        --calvin_root third_party/calvin \
        --checkpoint outputs/train/act_A_only_seed42/checkpoints/best_model.pt \
        --output outputs/eval/act_A_only_D_rollout.json \
        --num_sequences 100 \
        --seed 42

Prerequisites:
    1. CALVIN installed and importable
    2. ACT policy trained and checkpoint saved
    3. Run offline_eval.py first as a sanity check
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from calvin_lerobot.offline_eval import load_policy


class ACTPolicyRolloutWrapper:
    """
    Wrap LeRobot ACT policy for CALVIN rollout evaluation.

    Implements CALVIN's CustomModel interface with action chunking:
        - reset(): clear action buffer at episode start
        - step(obs, goal): return next action from buffer or re-query policy

    Action chunking logic:
        1. When buffer is empty, call policy to predict chunk_size future actions
        2. Each env step consumes the next action from buffer
        3. When buffer exhausted, re-query policy
        4. Buffer cleared at each episode start (reset)
    """

    def __init__(self, policy, device: torch.device, chunk_size: int = 100):
        self.policy = policy
        self.device = device
        self.chunk_size = chunk_size
        self._action_buffer = []
        self._buffer_idx = 0

    def reset(self):
        """Clear action buffer at the start of each evaluation episode."""
        self._action_buffer = []
        self._buffer_idx = 0

    def step(self, obs: dict, goal: str = "") -> np.ndarray:
        """
        Get next action for CALVIN environment step.

        Args:
            obs: CALVIN observation dict with keys:
                - rgb_static: (H, W, 3) uint8
                - rgb_gripper: (H, W, 3) uint8
                - robot_obs: (state_dim,) float32
            goal: language instruction string

        Returns:
            action: (7,) float32 array
        """
        if self._buffer_idx >= len(self._action_buffer):
            self._action_buffer = self._predict_chunk(obs)
            self._buffer_idx = 0

        action = self._action_buffer[self._buffer_idx]
        self._buffer_idx += 1
        return action

    @torch.no_grad()
    def _predict_chunk(self, obs: dict) -> list[np.ndarray]:
        """Query ACT policy for a chunk of future actions."""
        # Prepare observation tensors
        rgb_static = torch.from_numpy(obs["rgb_static"]).float().permute(2, 0, 1).unsqueeze(0)
        rgb_gripper = torch.from_numpy(obs["rgb_gripper"]).float().permute(2, 0, 1).unsqueeze(0)
        state = torch.from_numpy(obs["robot_obs"]).float().unsqueeze(0)

        batch = {
            "observation.images.static": (rgb_static / 255.0).to(self.device),
            "observation.images.gripper": (rgb_gripper / 255.0).to(self.device),
            "observation.state": state.to(self.device),
        }

        # Forward pass
        actions = self.policy.select_action(batch)

        # Handle output shape: (1, chunk_size, action_dim) or (1, action_dim)
        if actions.dim() == 3:
            actions = actions.squeeze(0)  # (chunk_size, action_dim)
        elif actions.dim() == 2:
            actions = actions  # (1, action_dim) — single action

        actions_np = actions.cpu().numpy()
        return [actions_np[i] for i in range(len(actions_np))]


def run_rollout_evaluation(
    wrapper: ACTPolicyRolloutWrapper,
    calvin_root: str,
    dataset_path: str,
    num_sequences: int = 100,
    seed: int = 42,
) -> dict:
    """
    Run closed-loop rollout evaluation using CALVIN simulator.

    Returns:
        dict with success_rate, mean_completed_tasks, per_task_success, etc.
    """
    try:
        # Import CALVIN evaluation utilities
        # NOTE: The exact import paths depend on CALVIN version.
        # Common paths:
        #   from calvin_agent.evaluation.evaluate_policy import evaluate_policy
        #   from calvin_agent.evaluation.multistep_sequences import get_sequences
        from calvin_agent.evaluation.evaluate_policy import evaluate_policy
        from calvin_agent.evaluation.multistep_sequences import get_sequences

        sequences = get_sequences(num_sequences, seed=seed)

        # CALVIN's evaluate_policy expects a model with .reset() and .step()
        results = evaluate_policy(
            model=wrapper,
            dataset_path=dataset_path,
            eval_sequences=sequences,
        )

        return {
            "success_count": results.get("success_count", 0),
            "success_rate": results.get("avg_seq_len", 0.0),
            "mean_completed_tasks": results.get("avg_seq_len", 0.0),
            "per_task_success": results.get("per_task_success", {}),
            "num_sequences": num_sequences,
        }

    except ImportError as e:
        print(f"CALVIN evaluation import failed: {e}")
        print()
        print("Install CALVIN:")
        print("  git clone --recurse-submodules https://github.com/mees/calvin.git third_party/calvin")
        print("  cd third_party/calvin && sh install.sh")
        print()
        print("If CALVIN install fails due to Python version conflicts,")
        print("use offline evaluation instead:")
        print("  python -m src.calvin_lerobot.offline_eval ...")
        return {
            "error": f"calvin_import_failed: {e}",
            "success_rate": None,
            "fallback": "use_offline_eval",
        }
    except Exception as e:
        print(f"CALVIN evaluation error: {e}")
        return {
            "error": f"calvin_eval_error: {e}",
            "success_rate": None,
            "fallback": "use_offline_eval",
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Rollout evaluation on CALVIN Env-D")
    parser.add_argument("--calvin_root", type=str, default="third_party/calvin",
                        help="Path to CALVIN installation")
    parser.add_argument("--dataset_path", type=str,
                        help="Path to CALVIN Env-D dataset (e.g., task_D_D)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained ACT checkpoint")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path for results")
    parser.add_argument("--num_sequences", type=int, default=100,
                        help="Number of evaluation sequences")
    parser.add_argument("--chunk_size", type=int, default=100,
                        help="Action chunk size for ACT inference")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"=== Rollout Evaluation on CALVIN Env-D ===")
    print(f"  CALVIN root: {args.calvin_root}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Sequences: {args.num_sequences}")
    print(f"  Device: {device}")

    # Load policy
    policy = load_policy(args.checkpoint, device)

    # Wrap for CALVIN interface
    wrapper = ACTPolicyRolloutWrapper(policy, device, chunk_size=args.chunk_size)

    # Determine dataset path
    dataset_path = args.dataset_path
    if dataset_path is None:
        # Try common CALVIN dataset locations
        calvin_root = Path(args.calvin_root)
        for candidate in [
            calvin_root / "dataset" / "task_D_D",
            calvin_root / "task_D_D",
        ]:
            if candidate.exists():
                dataset_path = str(candidate)
                break
        if dataset_path is None:
            print("Error: Cannot find CALVIN Env-D dataset.")
            print("Specify with --dataset_path")
            sys.exit(1)

    # Run evaluation
    metrics = run_rollout_evaluation(
        wrapper=wrapper,
        calvin_root=args.calvin_root,
        dataset_path=dataset_path,
        num_sequences=args.num_sequences,
        seed=args.seed,
    )

    # Build output
    checkpoint_name = Path(args.checkpoint).parent.name
    result = {
        "experiment": checkpoint_name,
        "checkpoint": args.checkpoint,
        "test_env": "D",
        "eval_mode": "rollout",
        "num_sequences": args.num_sequences,
        "chunk_size": args.chunk_size,
        **metrics,
        "seed": args.seed,
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
