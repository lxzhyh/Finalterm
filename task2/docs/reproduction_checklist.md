# Reproduction Checklist

Follow these steps in order to reproduce the experiments. Each step has verification criteria.

## 1. Environment Setup

- [ ] `conda create -n hw3-act python=3.10 -y && conda activate hw3-act`
- [ ] `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- [ ] `pip install "lerobot[aloha]"`
- [ ] `pip install -r requirements.txt`

**Verify:**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import lerobot; print(f'LeRobot {lerobot.__version__}')"
```

## 2. Verify LeRobot API

- [ ] `python scripts/convert_calvin_to_lerobot.py --check_api`

**Verify:** Output shows available methods like `create`, `add_frame`, `save_episode`. If API differs from expectations, update `src/calvin_lerobot/convert.py` accordingly.

## 3. CALVIN Installation (for rollout eval)

- [ ] `git clone --recurse-submodules https://github.com/mees/calvin.git third_party/calvin`
- [ ] `cd third_party/calvin && sh install.sh && cd ../..`

**Note:** CALVIN may require Python 3.8. If installation fails, you can still run offline evaluation. Consider creating a separate env: `conda create -n calvin-eval python=3.8`.

**Verify:**
```bash
python -c "import calvin_agent; print('CALVIN imported successfully')"
```

## 4. Download Data

### Debug split (for testing):
- [ ] `bash scripts/download_calvin.sh debug`

### Full dataset:
- [ ] `bash scripts/download_calvin.sh ABCD`

**Verify:**
```bash
ls third_party/calvin/dataset/
# Should see task_ABC_D/, task_D_D/, etc.
```

## 5. Inspect Data Structure

- [ ] `python tools/inspect_calvin_dataset.py --root third_party/calvin/dataset`

**Verify:** Output shows:
- Directory tree with episode_* subdirectories
- File types: .npy files (rgb_static, rgb_gripper, robot_obs, rel_actions)
- Shapes: rgb_static (T, 200, 200, 3), rgb_gripper (T, 84, 84, 3), rel_actions (T, 7)
- Language annotations in .pkl files

**If structure differs:** Update `src/calvin_lerobot/split.py` and `convert.py` to match actual layout.

## 6. Convert Data

### Debug conversion:
- [ ] `python scripts/convert_calvin_to_lerobot.py --calvin_root third_party/calvin/dataset --output_root data/lerobot_calvin --splits debug --max_episodes_per_env 5`

### Full conversion:
- [ ] `python scripts/convert_calvin_to_lerobot.py --calvin_root third_party/calvin/dataset --output_root data/lerobot_calvin --splits A ABC D --val_ratio 0.1 --seed 42`

**Verify:**
```bash
# Check output structure
ls data/lerobot_calvin/
# Should see: calvin_A_train/, calvin_A_val/, calvin_ABC_train/, calvin_ABC_val/, calvin_D_test/

# Verify dataset can be loaded
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('calvin_A_train', root='data/lerobot_calvin/calvin_A_train')
print(f'Loaded {len(ds)} frames')
sample = ds[0]
print(f'Keys: {list(sample.keys())}')
print(f'action shape: {sample[\"action\"].shape}')
"
```

## 7. Data Validation

- [ ] `python -m src.calvin_lerobot.visualization --dataset data/lerobot_calvin/calvin_A_train --validate_only`
- [ ] `python -m src.calvin_lerobot.visualization --dataset data/lerobot_calvin/calvin_ABC_train --validate_only`
- [ ] `python -m src.calvin_lerobot.visualization --dataset data/lerobot_calvin/calvin_D_test --validate_only`

**Verify:**
- [ ] No Env-D data in training sets (A, ABC)
- [ ] Action shape is (7,) for all samples
- [ ] Image shapes match (static: 3×200×200, gripper: 3×84×84)
- [ ] No NaN values in actions or states

**Environment isolation check:**
```bash
python -c "
import sys
sys.path.insert(0, 'src')
from calvin_lerobot.split import EpisodeSplitter
from pathlib import Path

splitter = EpisodeSplitter(seed=42, val_ratio=0.1)
splits = splitter.split(Path('third_party/calvin/dataset'))

train_envs = {ep.env for ep in splits['train']}
val_envs = {ep.env for ep in splits['val']}
test_envs = {ep.env for ep in splits['test']}

print(f'Train envs: {train_envs}')
print(f'Val envs: {val_envs}')
print(f'Test envs: {test_envs}')

assert 'D' not in train_envs, 'FATAL: D in train!'
assert 'D' not in val_envs, 'FATAL: D in val!'
assert test_envs == {'D'}, 'FATAL: test has non-D envs!'
print('✓ Environment isolation verified')
"
```

## 8. Smoke Test Training

- [ ] `bash scripts/train_debug.sh`

**Verify:**
- [ ] Training runs without errors
- [ ] Loss decreases (not NaN)
- [ ] Checkpoint saved in `outputs/train/debug_act/checkpoints/`
- [ ] Can reload checkpoint:
```bash
python -c "
from lerobot.common.policies.act.modeling_act import ACTPolicy
policy = ACTPolicy.from_pretrained('outputs/train/debug_act/checkpoints/last/pretrained_model')
print('✓ Checkpoint loaded successfully')
"
```

## 9. Full Training

- [ ] `bash scripts/train_act_A.sh` (Exp1: Env-A only, ~100k steps)
- [ ] `bash scripts/train_act_ABC.sh` (Exp2: Env-A/B/C joint, ~100k steps)

**Verify:**
- [ ] Both use identical hyperparameters (check `configs/act_A_only.yaml` vs `configs/act_ABC_joint.yaml`)
- [ ] Only difference is `dataset.repo_id`
- [ ] Loss curves saved in WandB or local logs
- [ ] Best checkpoints saved:
  - `outputs/train/act_A_only_seed42/checkpoints/best_model.pt`
  - `outputs/train/act_ABC_joint_seed42/checkpoints/best_model.pt`

## 10. Offline Evaluation

- [ ] `bash scripts/eval_env_D.sh`

This runs open-loop action L1 evaluation on Env-D demonstrations.

**Verify:**
- [ ] Results saved:
  - `outputs/eval/act_A_only_D_offline.json`
  - `outputs/eval/act_ABC_joint_D_offline.json`
- [ ] Metrics include: `mean_action_l1`, `position_l1`, `rotation_l1`, `gripper_error`
- [ ] Compare: does ABC joint have lower L1 error than A-only?

## 11. Rollout Evaluation (if CALVIN available)

- [ ] `python -m src.calvin_lerobot.rollout_eval --calvin_root third_party/calvin --checkpoint outputs/train/act_A_only_seed42/checkpoints/best_model.pt --output outputs/eval/act_A_only_D_rollout.json --num_sequences 100`
- [ ] `python -m src.calvin_lerobot.rollout_eval --calvin_root third_party/calvin --checkpoint outputs/train/act_ABC_joint_seed42/checkpoints/best_model.pt --output outputs/eval/act_ABC_joint_D_rollout.json --num_sequences 100`

**Verify:**
- [ ] Results saved with `success_rate` and `mean_completed_tasks`
- [ ] Compare: does ABC joint achieve higher success rate than A-only?

## 12. Generate Plots

- [ ] `python scripts/export_plots.py --log_dir outputs/train --eval_dir outputs/eval --output_dir reports/figures`

**Verify:**
- [ ] `reports/figures/train_action_l1_curve.png` shows both experiments
- [ ] `reports/figures/val_action_l1_curve.png` shows validation loss
- [ ] `reports/figures/env_D_evaluation.png` compares offline/rollout metrics

## 13. Report Preparation

- [ ] Include all figures from `reports/figures/`
- [ ] Include hyperparameter table (from configs)
- [ ] Include dataset split table (episode counts per environment)
- [ ] Include Action Chunking analysis (Section 10 from plan)
- [ ] Clearly state offline vs rollout evaluation differences
- [ ] Upload best checkpoints to cloud storage (Google Drive / Baidu Pan)
- [ ] Add download links to README

## Final Verification

Before submission, verify:

- [ ] Both experiments use identical network architecture
- [ ] Both experiments use identical hyperparameters (except dataset)
- [ ] Env-D data never entered training
- [ ] Report includes GitHub repo link
- [ ] Report includes model weight download links
- [ ] All commands in README can be copy-pasted and run
- [ ] PDF report is readable and complete
- [ ] Deadline: 2026-06-24 23:59 (Beijing time)
