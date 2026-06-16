#!/bin/bash
# Evaluate both models on Env-D
# Runs offline evaluation first, then rollout if CALVIN is available.
#
# LeRobot saves checkpoints as directories:
#   outputs/train/<exp>/checkpoints/last/pretrained_model/
set -euo pipefail

CKPT_A="outputs/train/act_A_only_seed42/checkpoints/last/pretrained_model"
CKPT_ABC="outputs/train/act_ABC_joint_seed42/checkpoints/last/pretrained_model"

echo "=== Step 1: Offline Evaluation (action L1 on Env-D demos) ==="
echo ""

echo "--- act_A_only ---"
if [ -d "${CKPT_A}" ]; then
  python -m src.calvin_lerobot.offline_eval \
    --dataset_root data/lerobot_calvin/calvin_D_test \
    --dataset_repo_id calvin_D_test \
    --checkpoint "${CKPT_A}" \
    --output outputs/eval/act_A_only_D_offline.json
else
  echo "  SKIP: checkpoint not found at ${CKPT_A}"
  echo "  Run: bash scripts/train_act_A.sh"
fi

echo ""
echo "--- act_ABC_joint ---"
if [ -d "${CKPT_ABC}" ]; then
  python -m src.calvin_lerobot.offline_eval \
    --dataset_root data/lerobot_calvin/calvin_D_test \
    --dataset_repo_id calvin_D_test \
    --checkpoint "${CKPT_ABC}" \
    --output outputs/eval/act_ABC_joint_D_offline.json
else
  echo "  SKIP: checkpoint not found at ${CKPT_ABC}"
  echo "  Run: bash scripts/train_act_ABC.sh"
fi

echo ""
echo "=== Step 2: Rollout Evaluation (if CALVIN simulator available) ==="
echo ""
echo "To run rollout eval:"
echo "  python -m src.calvin_lerobot.rollout_eval \\"
echo "    --checkpoint ${CKPT_A} \\"
echo "    --output outputs/eval/act_A_only_D_rollout.json"
echo ""
echo "  python -m src.calvin_lerobot.rollout_eval \\"
echo "    --checkpoint ${CKPT_ABC} \\"
echo "    --output outputs/eval/act_ABC_joint_D_rollout.json"
echo ""
echo "=== Done ==="
echo "Results:"
echo "  act_A_only:  outputs/eval/act_A_only_D_offline.json"
echo "  act_ABC:     outputs/eval/act_ABC_joint_D_offline.json"
