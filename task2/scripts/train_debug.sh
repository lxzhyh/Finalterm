#!/bin/bash
# Smoke test: verify training pipeline works with minimal data
#
# Prerequisites:
#   1. Install lerobot: pip install "lerobot[aloha]"
#   2. Download debug data: bash scripts/download_calvin.sh debug
#   3. Convert: python scripts/convert_calvin_to_lerobot.py \
#         --calvin_root third_party/calvin/dataset \
#         --output_root data/lerobot_calvin \
#         --splits debug \
#         --max_episodes_per_env 5
#
# Usage: bash scripts/train_debug.sh
set -euo pipefail

OUTPUT_DIR="outputs/train/debug_act"
mkdir -p "${OUTPUT_DIR}"

echo "=== Smoke Test: 200 steps, batch_size=2 ==="
echo ""

# IMPORTANT: Before running, verify CLI params with:
#   lerobot-train --help
#
# The parameters below are based on LeRobot documentation.
# If they don't match your version, adjust accordingly.

lerobot-train \
  --dataset.repo_id=calvin_debug \
  --dataset.root="data/lerobot_calvin/calvin_debug" \
  --policy.type=act \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=debug_act \
  --policy.device=cuda \
  --training.steps=200 \
  --training.batch_size=2 \
  --wandb.enable=false \
  2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "=== Smoke test complete ==="
echo "Check results: ls -la ${OUTPUT_DIR}/"
echo "Check log: cat ${OUTPUT_DIR}/train.log"
