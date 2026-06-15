#!/bin/bash
# Train ACT on CALVIN Env-A only (baseline experiment)
#
# This trains the baseline model using only Environment A data.
# Expected to take several hours on a GPU.
#
# Usage: bash scripts/train_act_A.sh
set -euo pipefail

OUTPUT_DIR="outputs/train/act_A_only"
mkdir -p "${OUTPUT_DIR}"

echo "=== Training ACT on CALVIN Env-A (baseline) ==="
echo "Output: ${OUTPUT_DIR}"
echo "Steps: 100000, Batch size: 8"
echo ""

# LeRobot 0.4.4 CLI parameters (verified via lerobot-train --help)
lerobot-train \
  --dataset.repo_id=calvin_A \
  --dataset.root="data/lerobot_calvin/calvin_A" \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=act_A_only \
  --batch_size=8 \
  --steps=100000 \
  --seed=42 \
  --eval_freq=5000 \
  --log_freq=100 \
  --wandb.enable=true \
  --wandb.project=act-calvin \
  2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "=== Training complete ==="
echo "Check results: ls -la ${OUTPUT_DIR}/"
echo "Check log: tail -50 ${OUTPUT_DIR}/train.log"
