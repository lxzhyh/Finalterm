# Reproduction Checklist

Follow these steps in order to reproduce the experiments.

## 1. Environment

- [ ] `conda create -n hw3-act python=3.10 -y && conda activate hw3-act`
- [ ] `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
- [ ] `pip install "lerobot>=0.5.1"`
- [ ] `pip install -r requirements.txt`

## 2. CALVIN (for rollout eval)

- [ ] `git clone --recurse-submodules https://github.com/mees/calvin.git third_party/calvin`
- [ ] `cd third_party/calvin && sh install.sh && cd ../..`

## 3. Data

- [ ] `bash scripts/download_calvin.sh ABCD`
- [ ] `python scripts/convert_calvin_to_lerobot.py --calvin_root third_party/calvin/dataset --output_root data/lerobot_calvin --splits A ABC D`

## 4. Data Validation

- [ ] `python -m src.calvin_lerobot.visualization --dataset data/lerobot_calvin/calvin_A_train --validate_only`
- [ ] `python -m src.calvin_lerobot.visualization --dataset data/lerobot_calvin/calvin_ABC_train --validate_only`
- [ ] `python -m src.calvin_lerobot.visualization --dataset data/lerobot_calvin/calvin_D_test --validate_only`
- [ ] Verify: no Env-D data in training sets
- [ ] Verify: action shape is (7,) for all samples

## 5. Training

- [ ] `bash scripts/train_act_A.sh`  (Exp1: Env-A only)
- [ ] `bash scripts/train_act_ABC.sh`  (Exp2: Env-A/B/C joint)
- [ ] Verify both experiments use identical hyperparameters

## 6. Evaluation

- [ ] `bash scripts/eval_env_D.sh`  (both models on Env-D)
- [ ] Check results in `outputs/eval/`

## 7. Plots

- [ ] `python scripts/export_plots.py --output_dir reports/figures`

## 8. Report

- [ ] Include all figures from `reports/figures/`
- [ ] Include hyperparameter table
- [ ] Include dataset split table
- [ ] Include Action Chunking analysis
- [ ] Upload best checkpoints to cloud storage
