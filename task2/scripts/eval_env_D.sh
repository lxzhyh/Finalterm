#!/bin/bash
# Evaluate both models on Env-D
# Runs offline evaluation first, then rollout if CALVIN is available.
set -euo pipefail

echo "=== Step 1: Offline Evaluation (action L1 on Env-D demos) ==="
echo ""

echo "--- act_A_only ---"
python -m src.calvin_lerobot.offline_eval \
  --dataset_root data/lerobot_calvin/calvin_D_test \
  --dataset_repo_id calvin_D_test \
  --checkpoint outputs/train/act_A_only_seed42/checkpoints/best_model.pt \
  --output outputs/eval/act_A_only_D_offline.json

echo ""
echo "--- act_ABC_joint ---"
python -m src.calvin_lerobot.offline_eval \
  --dataset_root data/lerobot_calvin/calvin_D_test \
  --dataset_repo_id calvin_D_test \
  --checkpoint outputs/train/act_ABC_joint_seed42/checkpoints/best_model.pt \
  --output outputs/eval/act_ABC_joint_D_offline.json

echo ""
echo "=== Step 2: Rollout Evaluation (if CALVIN available) ==="
echo ""
echo "To run rollout eval, first install CALVIN:"
echo "  git clone --recurse-submodules https://github.com/mees/calvin.git third_party/calvin"
echo "  cd third_party/calvin && sh install.sh"
echo ""
echo "Then run:"
echo "  python -m src.calvin_lerobot.rollout_eval \\"
echo "    --calvin_root third_party/calvin \\"
echo "    --checkpoint outputs/train/act_A_only_seed42/checkpoints/best_model.pt \\"
echo "    --output outputs/eval/act_A_only_D_rollout.json"
echo ""
echo "  python -m src.calvin_lerobot.rollout_eval \\"
echo "    --calvin_root third_party/calvin \\"
echo "    --checkpoint outputs/train/act_ABC_joint_seed42/checkpoints/best_model.pt \\"
echo "    --output outputs/eval/act_ABC_joint_D_rollout.json"
echo ""
echo "=== Offline evaluation complete ==="
echo "Results:"
echo "  act_A_only:  outputs/eval/act_A_only_D_offline.json"
echo "  act_ABC:     outputs/eval/act_ABC_joint_D_offline.json"
