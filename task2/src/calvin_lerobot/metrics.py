"""
Evaluation metrics for ACT policy cross-environment generalization.

Metrics:
    - mean_action_l1: average L1 error between predicted and ground-truth actions
    - position_l1: L1 error on xyz position deltas
    - rotation_l1: L1 error on rotation deltas (euler angles)
    - gripper_error: binary error on gripper action
    - chunk_boundary_delta: action jump at chunk boundaries (smoothness)
    - chunk_inner_variance: variance within a chunk (action consistency)
"""

import numpy as np


def compute_eval_metrics(
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
    chunk_size: int = 100,
) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        pred_actions: (N, 7) predicted actions
        gt_actions: (N, 7) ground-truth actions
        chunk_size: action chunk size for boundary analysis

    Returns:
        dict with metric names -> float values
    """
    assert pred_actions.shape == gt_actions.shape
    assert pred_actions.shape[-1] == 7

    # Per-dimension absolute errors
    abs_errors = np.abs(pred_actions - gt_actions)

    # Mean action L1
    mean_action_l1 = float(abs_errors.mean())

    # Position L1 (dims 0-2: dx, dy, dz)
    position_l1 = float(abs_errors[:, :3].mean())

    # Rotation L1 (dims 3-5: drx, dry, drz)
    rotation_l1 = float(abs_errors[:, 3:6].mean())

    # Gripper error (dim 6)
    gripper_error = float(abs_errors[:, 6].mean())

    # Chunk boundary analysis
    chunk_boundary_delta = _chunk_boundary_delta(pred_actions, chunk_size)

    # Chunk inner variance
    chunk_inner_variance = _chunk_inner_variance(pred_actions, chunk_size)

    return {
        "mean_action_l1": round(mean_action_l1, 6),
        "median_action_l1": round(float(np.median(abs_errors.mean(axis=1))), 6),
        "position_l1": round(position_l1, 6),
        "rotation_l1": round(rotation_l1, 6),
        "gripper_error": round(gripper_error, 6),
        "chunk_boundary_delta": round(chunk_boundary_delta, 6),
        "chunk_inner_variance": round(chunk_inner_variance, 6),
        "num_samples": len(pred_actions),
    }


def _chunk_boundary_delta(actions: np.ndarray, chunk_size: int) -> float:
    """
    Measure action discontinuity at chunk boundaries.

    At the transition between two consecutive chunks, compute the L2 distance
    between the last action of the previous chunk and the first action of the
    new chunk. Lower values indicate smoother transitions.
    """
    n = len(actions)
    if n <= chunk_size:
        return 0.0

    deltas = []
    for i in range(chunk_size, n, chunk_size):
        if i < n:
            d = np.linalg.norm(actions[i] - actions[i - 1])
            deltas.append(float(d))

    return float(np.mean(deltas)) if deltas else 0.0


def _chunk_inner_variance(actions: np.ndarray, chunk_size: int) -> float:
    """
    Measure action smoothness within each chunk.

    For each chunk, compute the variance of consecutive action differences.
    Lower values indicate smoother, more consistent action trajectories.
    """
    n = len(actions)
    variances = []

    for start in range(0, n - chunk_size + 1, chunk_size):
        chunk = actions[start:start + chunk_size]
        if len(chunk) < 2:
            continue
        diffs = np.diff(chunk, axis=0)
        variances.append(float(np.var(diffs)))

    return float(np.mean(variances)) if variances else 0.0


def compute_per_dim_stats(actions: np.ndarray) -> dict:
    """Compute per-dimension statistics for action analysis."""
    dim_names = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
    stats = {}
    for i, name in enumerate(dim_names):
        col = actions[:, i]
        stats[name] = {
            "mean": round(float(col.mean()), 6),
            "std": round(float(col.std()), 6),
            "min": round(float(col.min()), 6),
            "max": round(float(col.max()), 6),
        }
    return stats
