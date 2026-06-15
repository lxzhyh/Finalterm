"""
Offline action L1 evaluation on Env-D demonstrations.

Fallback when CALVIN rollout environment is unavailable.
Computes open-loop action prediction error on Env-D demonstration data.

Limitations (must be stated in report):
    - Open-loop: does not account for compounding errors in closed-loop execution
    - Cannot measure actual task success rate
    - Still useful for quantifying visual distribution shift effects

Usage:
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
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from calvin_lerobot.metrics import compute_eval_metrics, compute_per_dim_stats


def offline_evaluate(model, dataset_path: str) -> dict:
    """
    Run offline evaluation: predict actions for each frame in the dataset
    and compare against ground-truth actions.

    Args:
        model: ACTPolicyWrapper instance
        dataset_path: path to LeRobotDataset directory

    Returns:
        dict of metrics
    """
    dataset_path = Path(dataset_path)
    data_dir = dataset_path / "data"

    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")

    all_pred_actions = []
    all_gt_actions = []

    for pf in tqdm(parquet_files, desc="Offline evaluation"):
        table = pq.read_table(pf)
        df = table.to_pandas()

        states = np.array(df["observation.state"].tolist(), dtype=np.float32)
        gt_actions = np.array(df["action"].tolist(), dtype=np.float32)

        for t in range(len(states)):
            # Build minimal observation for model
            obs = {
                "rgb_static": np.zeros((200, 200, 3), dtype=np.uint8),  # placeholder
                "rgb_gripper": np.zeros((84, 84, 3), dtype=np.uint8),   # placeholder
                "robot_obs": states[t],
            }

            model.reset()
            pred = model.step(obs)
            all_pred_actions.append(pred)
            all_gt_actions.append(gt_actions[t])

    pred_actions = np.array(all_pred_actions)
    gt_actions = np.array(all_gt_actions)

    metrics = compute_eval_metrics(pred_actions, gt_actions)
    per_dim = compute_per_dim_stats(pred_actions - gt_actions)
    metrics["per_dimension_error"] = per_dim

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Offline action L1 evaluation")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to LeRobotDataset (Env-D test)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained ACT checkpoint")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path for results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--chunk_size", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"=== Offline evaluation ===")
    print(f"  Dataset: {args.dataset}")
    print(f"  Checkpoint: {args.checkpoint}")
    print()
    print("Note: This is an OPEN-LOOP evaluation metric.")
    print("It cannot substitute closed-loop success rate from rollout evaluation.")
    print()

    from calvin_lerobot.eval_policy import ACTPolicyWrapper
    model = ACTPolicyWrapper(args.checkpoint, device=args.device, chunk_size=args.chunk_size)

    metrics = offline_evaluate(model, args.dataset)

    checkpoint_name = Path(args.checkpoint).parent.parent.name
    result = {
        "experiment": checkpoint_name,
        "checkpoint": args.checkpoint,
        "test_env": "D",
        "eval_mode": "offline",
        **metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
