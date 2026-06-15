"""
Offline action L1 evaluation on Env-D demonstrations.

Reads REAL images from the converted LeRobotDataset and feeds them
to the trained ACT policy. Computes open-loop action prediction error.

IMPORTANT LIMITATION (must state in report):
    This is an OPEN-LOOP evaluation. It does NOT account for compounding
    errors in closed-loop execution. Cannot measure actual task success rate.
    Useful for quantifying visual distribution shift effects on action prediction.

Usage:
    python -m src.calvin_lerobot.offline_eval \
        --dataset_root data/lerobot_calvin/calvin_D_test \
        --dataset_repo_id calvin_D_test \
        --checkpoint outputs/train/act_A_only_seed42/checkpoints/best_model.pt \
        --output outputs/eval/act_A_only_D_offline.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from calvin_lerobot.metrics import compute_eval_metrics, compute_per_dim_stats


def load_policy(checkpoint_dir: str, device: torch.device) -> "ACTPolicy":
    """
    Load ACT policy using LeRobot's official loading API.

    LeRobot saves policies as directories containing:
        config.json, model.safetensors (or .pt), etc.

    Loading method (verify with your version):
        from lerobot.common.policies.act.modeling_act import ACTPolicy
        policy = ACTPolicy.from_pretrained(checkpoint_dir)

    If the import path differs, try:
        python -c "
        import lerobot.common.policies as p
        import pkgutil
        print([m.name for m in pkgutil.iter_modules(p.__path__)])
        "
    """
    # Try standard LeRobot loading
    try:
        from lerobot.common.policies.act.modeling_act import ACTPolicy
        policy = ACTPolicy.from_pretrained(checkpoint_dir)
        policy.to(device)
        policy.eval()
        print(f"Loaded policy via ACTPolicy.from_pretrained()")
        return policy
    except (ImportError, AttributeError) as e:
        print(f"ACTPolicy.from_pretrained failed: {e}")

    # Fallback: try loading from checkpoint directory
    try:
        from lerobot.common.policies.act.modeling_act import ACTPolicy
        ckpt_path = Path(checkpoint_dir)

        # Look for model file in checkpoint dir
        for fname in ["model.safetensors", "model.pt", "last_model.safetensors",
                       "best_model.safetensors", "pytorch_model.bin"]:
            model_file = ckpt_path / fname
            if model_file.exists():
                print(f"Loading model weights from: {model_file}")
                break

        # Load config and create policy
        config_file = ckpt_path / "config.json"
        if config_file.exists():
            import json
            with open(config_file) as f:
                config_dict = json.load(f)
            from lerobot.common.policies.act.configuration import ACTConfig
            config = ACTConfig(**config_dict)
            policy = ACTPolicy(config)
            # Load weights
            if model_file.suffix == ".safetensors":
                from safetensors.torch import load_file
                state_dict = load_file(str(model_file))
            else:
                state_dict = torch.load(str(model_file), map_location=device,
                                        weights_only=True)
            policy.load_state_dict(state_dict)
            policy.to(device)
            policy.eval()
            print(f"Loaded policy from checkpoint files")
            return policy
    except Exception as e:
        print(f"Fallback loading also failed: {e}")

    raise RuntimeError(
        f"Cannot load policy from {checkpoint_dir}. "
        f"Check LeRobot version and checkpoint format. "
        f"See: python -c 'import lerobot; print(lerobot.__version__)'"
    )


def offline_evaluate(
    policy,
    dataset_root: str,
    dataset_repo_id: str,
    device: torch.device,
    max_samples: int = None,
) -> dict:
    """
    Run offline evaluation by iterating over Env-D dataset samples.

    For each sample:
        1. Load real observation (images + state) from LeRobotDataset
        2. Feed to policy to get predicted action
        3. Compare with ground-truth action

    Args:
        policy: loaded ACT policy (in eval mode)
        dataset_root: local path to LeRobotDataset directory
        dataset_repo_id: dataset repo identifier
        device: torch device
        max_samples: limit number of samples (None = all)

    Returns:
        dict of evaluation metrics
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # Load dataset
    ds = LeRobotDataset(dataset_repo_id, root=dataset_root)
    print(f"Loaded dataset: {len(ds)} frames")

    if max_samples is not None:
        n_samples = min(max_samples, len(ds))
        indices = np.random.choice(len(ds), n_samples, replace=False)
    else:
        n_samples = len(ds)
        indices = range(n_samples)

    all_pred_actions = []
    all_gt_actions = []

    for idx in tqdm(indices, desc="Offline evaluation"):
        sample = ds[idx]

        # Build batch with real observations from dataset
        batch = {
            "observation.images.static": sample["observation.images.static"].unsqueeze(0).to(device),
            "observation.images.gripper": sample["observation.images.gripper"].unsqueeze(0).to(device),
            "observation.state": sample["observation.state"].unsqueeze(0).to(device),
        }

        # Predict action
        with torch.no_grad():
            pred_action = policy.select_action(batch)  # (1, action_dim) or (1, chunk, action_dim)

        # Handle action chunking output: take the first action of the chunk
        if pred_action.dim() == 3:
            pred_action = pred_action[:, 0, :]  # (1, action_dim)

        pred_np = pred_action.squeeze(0).cpu().numpy()
        gt_np = sample["action"].numpy()

        all_pred_actions.append(pred_np)
        all_gt_actions.append(gt_np)

    pred_actions = np.array(all_pred_actions)
    gt_actions = np.array(all_gt_actions)

    # Compute metrics
    metrics = compute_eval_metrics(pred_actions, gt_actions)
    per_dim = compute_per_dim_stats(pred_actions - gt_actions)
    metrics["per_dimension_error"] = per_dim

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Offline action L1 evaluation on Env-D")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Local path to Env-D LeRobotDataset")
    parser.add_argument("--dataset_repo_id", type=str, default="calvin_D_test",
                        help="Dataset repo identifier")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained ACT checkpoint directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (None = all)")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"=== Offline Evaluation ===")
    print(f"  Dataset: {args.dataset_root}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")
    print()
    print("  NOTE: This is an OPEN-LOOP metric.")
    print("  It does NOT substitute closed-loop success rate from rollout.")
    print()

    # Load policy
    policy = load_policy(args.checkpoint, device)

    # Evaluate
    metrics = offline_evaluate(
        policy=policy,
        dataset_root=args.dataset_root,
        dataset_repo_id=args.dataset_repo_id,
        device=device,
        max_samples=args.max_samples,
    )

    # Build output
    checkpoint_name = Path(args.checkpoint).parent.name
    result = {
        "experiment": checkpoint_name,
        "checkpoint": args.checkpoint,
        "test_env": "D",
        "eval_mode": "offline",
        "device": str(device),
        **metrics,
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
