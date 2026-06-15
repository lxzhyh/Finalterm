#!/bin/bash
# Train ACT on CALVIN Env-A + B + C (joint)
# Same hyperparameters as train_act_A.sh — only dataset differs
set -euo pipefail

EXP_NAME="act_ABC_joint_seed42"
OUTPUT_DIR="outputs/train/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"

echo "=== Training ${EXP_NAME} ==="
echo "Dataset: calvin_ABC_train (Env-A + B + C)"
echo "Steps: 100000, Batch size: 8"
echo ""

lerobot-train \
  --dataset.repo_id=calvin_ABC_train \
  --dataset.root="data/lerobot_calvin/calvin_ABC_train" \
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
