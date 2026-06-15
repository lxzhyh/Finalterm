#!/bin/bash
# Smoke test: verify training pipeline works with minimal data
#
# Prerequisites:
#   1. Install lerobot: pip install "lerobot[aloha]"
#   2. Generate synthetic data or download debug data
#   3. Convert data to LeRobot format
#
# Usage: bash scripts/train_debug.sh
set -euo pipefail

OUTPUT_DIR="outputs/train/debug_act"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "=== Smoke Test: 20 steps, batch_size=2, CPU ==="
echo ""

lerobot-train \
  --dataset.repo_id=calvin_A_train \
  --dataset.root="data/lerobot_calvin/calvin_A_train" \
  --policy.type=act \
  --policy.device=cpu \
  --policy.push_to_hub=false \
  --policy.repo_id="local/debug_act" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=debug_act \
  --batch_size=2 \
  --steps=20 \
  --seed=42 \
  --wandb.enable=false \
  2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "=== Smoke test complete ==="
echo "Checkpoint: ${OUTPUT_DIR}/checkpoints/last/pretrained_model/"
echo "Log: ${OUTPUT_DIR}/train.log"
