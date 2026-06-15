#!/bin/bash
# Train ACT on CALVIN Env-A only
#
# IMPORTANT: Before first run, verify CLI params with:
#   lerobot-train --help
#
# Adjust any parameter names that don't match your LeRobot version.
set -euo pipefail

EXP_NAME="act_A_only_seed42"
OUTPUT_DIR="outputs/train/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"

echo "=== Training ${EXP_NAME} ==="
echo "Dataset: calvin_A_train (Env-A only)"
echo "Steps: 100000, Batch size: 8"
echo ""

lerobot-train \
  --dataset.repo_id=calvin_A_train \
  --dataset.root="data/lerobot_calvin/calvin_A_train" \
  --policy.type=act \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${EXP_NAME}" \
  --policy.device=cuda \
  --seed=42 \
  --training.steps=100000 \
  --training.batch_size=8 \
  --wandb.enable=true \
  --wandb.project=act-calvin \
  2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "=== Training complete: ${EXP_NAME} ==="
echo "Checkpoints: ${OUTPUT_DIR}/checkpoints/"
echo "Log: ${OUTPUT_DIR}/train.log"
