#!/bin/bash
# Train ACT on CALVIN Env-A/B/C combined (multi-environment experiment)
#
# This trains the multi-environment model using combined A+B+C data.
# Expected to take several hours on a GPU.
#
# Usage: bash scripts/train_act_ABC.sh
set -euo pipefail

OUTPUT_DIR="outputs/train/act_ABC_joint_seed42"
mkdir -p "${OUTPUT_DIR}"

echo "=== Training ACT on CALVIN Env-A/B/C (multi-environment) ==="
echo "Output: ${OUTPUT_DIR}"
echo "Steps: 100000, Batch size: 8"
echo ""

# LeRobot 0.4.4 CLI parameters (verified via lerobot-train --help)
lerobot-train \
  --dataset.repo_id=calvin_ABC_train \
  --dataset.root="data/lerobot_calvin/calvin_ABC_train" \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=act_ABC_joint_seed42 \
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
