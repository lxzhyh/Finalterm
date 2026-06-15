"""
Evaluate ACT policy on CALVIN Env-D.

Supports two evaluation modes:
    1. Rollout evaluation: closed-loop success rate via CALVIN simulator
    2. Offline evaluation: open-loop action L1 error on Env-D demonstrations

Usage:
    # Rollout evaluation (preferred)
    python -m src.calvin_lerobot.eval_policy \
        --calvin_root third_party/calvin \
        --dataset_path /path/to/calvin/dataset/task_D_D \
        --checkpoint outputs/train/act_A_only_seed42/checkpoints/best_model.pt \
        --output outputs/eval/act_A_only_D.json \
        --num_sequences 100 \
        --seed 42

    # Offline fallback (if rollout env unavailable)
    python -m src.calvin_lerobot.offline_eval \
        --dataset data/lerobot_calvin/calvin_D_test \
        --checkpoint outputs/train/act_A_only_seed42/checkpoints/best_model.pt \
        --output outputs/eval/act_A_only_D_offline.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from calvin_lerobot.metrics import compute_eval_metrics


class ACTPolicyWrapper:
    """
    Wrap LeRobot ACT policy for CALVIN evaluation.

    CALVIN CustomModel interface:
        model.reset()              # called at episode start
        model.step(obs, goal)      # returns action given observation + language goal
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda", chunk_size: int = 100):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.chunk_size = chunk_size

        # Load policy
        self.policy = self._load_policy(checkpoint_path)
        self.policy.eval()

        # Action chunking buffer
        self._action_buffer = []
        self._buffer_idx = 0

    def _load_policy(self, checkpoint_path: str):
        """Load ACT policy from LeRobot checkpoint."""
        try:
            from lerobot.common.policies.act import ACTConfig, ACTPolicy

            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            config = ACTConfig(**ckpt.get("config", {}))
            policy = ACTPolicy(config)
            policy.load_state_dict(ckpt["model_state_dict"])
            policy.to(self.device)
            return policy
        except Exception as e:
            print(f"Warning: Could not load via LeRobot API ({e})")
            print("Falling back to raw checkpoint loading...")
            return torch.load(checkpoint_path, map_location=self.device, weights_only=False)

    def reset(self):
        """Clear action buffer at the start of each evaluation episode."""
        self._action_buffer = []
        self._buffer_idx = 0

    def step(self, obs: dict, goal: str = "") -> np.ndarray:
        """
        Get next action from ACT policy with action chunking.

        When the buffer is exhausted, query the policy for a new chunk
        of `chunk_size` future actions. Otherwise, return the next
        buffered action.

        Args:
            obs: dict with keys 'rgb_static', 'rgb_gripper', 'robot_obs'
            goal: language instruction string

        Returns:
            action: np.ndarray of shape (7,)
        """
        if self._buffer_idx >= len(self._action_buffer):
            # Re-query policy for new action chunk
            self._action_buffer = self._predict_chunk(obs)
            self._buffer_idx = 0

        action = self._action_buffer[self._buffer_idx]
        self._buffer_idx += 1
        return action

    @torch.no_grad()
    def _predict_chunk(self, obs: dict) -> list[np.ndarray]:
        """Run ACT forward pass to predict a chunk of actions."""
        # Prepare inputs
        rgb_static = torch.from_numpy(obs["rgb_static"]).float().permute(2, 0, 1).unsqueeze(0)
        rgb_gripper = torch.from_numpy(obs["rgb_gripper"]).float().permute(2, 0, 1).unsqueeze(0)
        state = torch.from_numpy(obs["robot_obs"]).float().unsqueeze(0)

        rgb_static = rgb_static.to(self.device) / 255.0
        rgb_gripper = rgb_gripper.to(self.device) / 255.0
        state = state.to(self.device)

        batch = {
            "observation.images.static": rgb_static,
            "observation.images.gripper": rgb_gripper,
            "observation.state": state,
        }

        # Forward pass
        actions = self.policy.select_action(batch)  # (1, chunk_size, action_dim)
        actions = actions.squeeze(0).cpu().numpy()  # (chunk_size, action_dim)

        return [actions[i] for i in range(len(actions))]


def evaluate_rollout(
    model: ACTPolicyWrapper,
    dataset_path: str,
    num_sequences: int = 100,
    seed: int = 42,
) -> dict:
    """
    Run closed-loop rollout evaluation using CALVIN simulator.

    Falls back to offline evaluation if CALVIN environment cannot be loaded.
    """
    try:
        from calvin_agent.evaluation.evaluate_policy import evaluate_policy
        from calvin_agent.evaluation.multistep_sequences import get_sequences

        # Use CALVIN's evaluation wrapper
        sequences = get_sequences(num_sequences, seed=seed)
        results = evaluate_policy(
            model=model,
            dataset_path=dataset_path,
            eval_sequences=sequences,
        )
        return {
            "success_count": results.get("success_count", 0),
            "success_rate": results.get("avg_seq_len", 0.0),
            "mean_completed_tasks": results.get("avg_seq_len", 0.0),
        }
    except ImportError:
        print("CALVIN evaluation not available. Install CALVIN for rollout evaluation.")
        print("Run: git clone --recurse-submodules https://github.com/mees/calvin.git third_party/calvin")
        print("     cd third_party/calvin && sh install.sh")
        return {"error": "calvin_eval_unavailable", "success_rate": None}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ACT policy on CALVIN Env-D")
    parser.add_argument("--calvin_root", type=str, default="third_party/calvin")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_sequences", type=int, default=100)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true",
                        help="Use offline action L1 evaluation instead of rollout")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"=== Evaluating checkpoint: {args.checkpoint} ===")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Mode: {'offline' if args.offline else 'rollout'}")

    # Load model
    model = ACTPolicyWrapper(args.checkpoint, device=args.device, chunk_size=args.chunk_size)

    if args.offline:
        from calvin_lerobot.offline_eval import offline_evaluate
        metrics = offline_evaluate(model, args.dataset_path)
    else:
        metrics = evaluate_rollout(
            model, args.dataset_path,
            num_sequences=args.num_sequences,
            seed=args.seed,
        )

    # Build output
    checkpoint_name = Path(args.checkpoint).parent.parent.name
    result = {
        "experiment": checkpoint_name,
        "checkpoint": args.checkpoint,
        "test_env": "D",
        "num_sequences": args.num_sequences,
        "eval_mode": "offline" if args.offline else "rollout",
        **metrics,
        "seed": args.seed,
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
