#!/bin/bash
# Evaluate ABC-joint real model on Env-D (real data)
set -euo pipefail

CHECKPOINT="outputs/train/act_ABC_real_seed42/checkpoints/last/pretrained_model"
OUTPUT="outputs/eval/act_ABC_real_D_offline.json"

echo "=== Evaluating ABC-joint (real) on Env-D ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT}"
echo ""

source .venv/bin/activate
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m src.calvin_lerobot.offline_eval \
  --dataset_root data/lerobot_calvin/calvin_D \
  --dataset_repo_id calvin_D \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT}" \
  --device cuda \
  --max_samples 5000

echo ""
echo "=== Evaluation complete ==="
cat "${OUTPUT}"
