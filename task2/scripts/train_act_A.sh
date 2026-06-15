#!/bin/bash
# Train ACT on CALVIN Env-A only
set -euo pipefail

EXP_NAME="act_A_only_seed42"

echo "=== Training ${EXP_NAME} ==="

lerobot-train \
  --dataset.repo_id=calvin_A_train \
  --policy.type=act \
  --output_dir=outputs/train/${EXP_NAME} \
  --job_name=${EXP_NAME} \
  --policy.device=cuda \
  --wandb.enable=true \
  --wandb.project=act-calvin \
  --seed=42 \
  --batch_size=8 \
  --steps=100000 \
  --log_freq=50 \
  --eval_freq=5000 \
  --save_freq=10000 \
  2>&1 | tee outputs/train/${EXP_NAME}/train.log

echo "=== Training complete: ${EXP_NAME} ==="
