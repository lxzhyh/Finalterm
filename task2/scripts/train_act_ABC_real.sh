#!/bin/bash
# Train ACT on CALVIN Env-A/B/C combined (real multi-environment experiment)
set -euo pipefail

OUTPUT_DIR="outputs/train/act_ABC_real_seed42"
LOG_FILE="/tmp/act_ABC_real_train.log"

echo "=== Training ACT on CALVIN Env-A/B/C (real combined data) ==="
echo "Output: ${OUTPUT_DIR}"
echo "Steps: 200, Batch size: 8"
echo ""

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
lerobot-train \
  --dataset.repo_id=calvin_ABC_real_train \
  --dataset.root="data/lerobot_calvin/calvin_ABC_real_train" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.repo_id="local/act_ABC_real_seed42" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=act_ABC_real_seed42 \
  --batch_size=8 \
  --steps=200 \
  --seed=42 \
  --eval_freq=100 \
  --log_freq=10 \
  --wandb.enable=false \
  2>&1 | tee "${LOG_FILE}"

mkdir -p "${OUTPUT_DIR}"
cp "${LOG_FILE}" "${OUTPUT_DIR}/train.log"

echo ""
echo "=== Training complete ==="
echo "Check results: ls -la ${OUTPUT_DIR}/"
echo "Check log: tail -50 ${OUTPUT_DIR}/train.log"
