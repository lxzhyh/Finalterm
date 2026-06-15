#!/bin/bash
# Evaluate both models on Env-D (zero-shot)
set -euo pipefail

echo "=== Evaluating act_A_only on Env-D ==="
python -m src.calvin_lerobot.eval_policy \
  --calvin_root third_party/calvin \
  --dataset_path ./data/lerobot_calvin/calvin_D_test \
  --checkpoint outputs/train/act_A_only_seed42/checkpoints/best_model.pt \
  --output outputs/eval/act_A_only_D.json \
  --num_sequences 100 \
  --seed 42

echo ""
echo "=== Evaluating act_ABC_joint on Env-D ==="
python -m src.calvin_lerobot.eval_policy \
  --calvin_root third_party/calvin \
  --dataset_path ./data/lerobot_calvin/calvin_D_test \
  --checkpoint outputs/train/act_ABC_joint_seed42/checkpoints/best_model.pt \
  --output outputs/eval/act_ABC_joint_D.json \
  --num_sequences 100 \
  --seed 42

echo ""
echo "=== Evaluation complete ==="
echo "Results:"
echo "  act_A_only:  outputs/eval/act_A_only_D.json"
echo "  act_ABC:     outputs/eval/act_ABC_joint_D.json"
