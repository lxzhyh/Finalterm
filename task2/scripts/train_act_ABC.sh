#!/bin/bash
# Train ACT on CALVIN Env-A + B + C (joint)
# Same hyperparameters as train_act_A.sh — only dataset differs
set -euo pipefail

EXP_NAME="act_ABC_joint_seed42"

echo "=== Training ${EXP_NAME} ==="

lerobot-train \
  --dataset.repo_id=calvin_ABC_train \
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
