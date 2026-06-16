# HW3 Task 2: LeRobot ACT Cross-Environment Generalization on CALVIN

## Overview

This project implements cross-environment generalization experiments for the ACT (Action Chunking with Transformers) policy using the LeRobot framework on the CALVIN benchmark.

**Core comparison:**

| Experiment | Training Data | Test Data | Purpose |
|---|---|---|---|
| `act_A_only` | Env-A | Env-D (zero-shot) | Baseline single-environment policy |
| `act_ABC_joint` | Env-A + B + C | Env-D (zero-shot) | Multi-environment joint training |

**Key question:** Does multi-environment training improve zero-shot generalization to unseen environments?

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n hw3-act python=3.10 -y
conda activate hw3-act

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install LeRobot with ACT support
pip install "lerobot[aloha]"

# Install additional dependencies
pip install -r requirements.txt
```

### 2. Verify LeRobot API

Before proceeding, check that LeRobot is correctly installed and inspect the dataset API:

```bash
python scripts/convert_calvin_to_lerobot.py --check_api
```

This will print available methods like `LeRobotDataset.create()`, `add_frame()`, etc. If the API has changed, you may need to adjust `src/calvin_lerobot/convert.py`.

### 3. Download CALVIN Data

For initial testing, use the debug split:

```bash
bash scripts/download_calvin.sh debug
```

For full experiments:

```bash
bash scripts/download_calvin.sh ABCD
```

### 4. Inspect Data Structure

Understand the actual CALVIN directory layout:

```bash
python tools/inspect_calvin_dataset.py --root third_party/calvin/dataset
```

This prints the file tree and shows the shape/dtype of sample data files. Use this to verify that `split.py` and `convert.py` match your CALVIN version.

### 5. Convert to LeRobot Format

```bash
# Debug conversion (5 episodes per environment)
python scripts/convert_calvin_to_lerobot.py \
  --calvin_root third_party/calvin/dataset \
  --output_root data/lerobot_calvin \
  --splits debug \
  --max_episodes_per_env 5

# Full conversion
python scripts/convert_calvin_to_lerobot.py \
  --calvin_root third_party/calvin/dataset \
  --output_root data/lerobot_calvin \
  --splits A ABC D \
  --val_ratio 0.1 \
  --seed 42
```

The script will:
- Split episodes by environment (A/B/C for train, D for test)
- Convert to LeRobotDataset format using official API
- Verify the dataset can be loaded

### 6. Smoke Test Training

Verify the training pipeline works:

```bash
bash scripts/train_debug.sh
```

This runs 200 steps on debug data with batch_size=2. Check that:
- Loss decreases (not NaN)
- Checkpoints are saved
- No shape mismatches

### 7. Full Training

Train the two main experiments:

```bash
# Experiment 1: Env-A only (baseline)
bash scripts/train_act_A.sh

# Experiment 2: Env-A/B/C joint training
bash scripts/train_act_ABC.sh
```

Both use identical hyperparameters — only the training data differs.

### 8. Evaluation

**Offline evaluation** (action L1 on Env-D demonstrations):

```bash
bash scripts/eval_env_D.sh
```

This runs open-loop evaluation on Env-D data. Results are saved to `outputs/eval/`.

**Rollout evaluation** (closed-loop success rate, requires CALVIN):

```bash
# Install CALVIN simulator
git clone --recurse-submodules https://github.com/mees/calvin.git third_party/calvin
cd third_party/calvin && sh install.sh && cd ../..

# Run rollout evaluation
python -m src.calvin_lerobot.rollout_eval \
  --calvin_root third_party/calvin \
  --checkpoint outputs/train/act_A_only_seed42/checkpoints/best_model.pt \
  --output outputs/eval/act_A_only_D_rollout.json \
  --num_sequences 100

python -m src.calvin_lerobot.rollout_eval \
  --calvin_root third_party/calvin \
  --checkpoint outputs/train/act_ABC_joint_seed42/checkpoints/best_model.pt \
  --output outputs/eval/act_ABC_joint_D_rollout.json \
  --num_sequences 100
```

### 9. Generate Plots

Export training curves and evaluation comparisons:

```bash
python scripts/export_plots.py \
  --log_dir outputs/train \
  --eval_dir outputs/eval \
  --output_dir reports/figures
```

## Repository Structure

```
Finalterm/
└── task2/
    ├── README.md                      # This file
    ├── requirements.txt               # Python dependencies
    ├── environment.yml                # Conda environment spec
    ├── configs/                       # Training & eval YAML configs
    │   ├── act_A_only.yaml
    │   ├── act_ABC_joint.yaml
    │   └── eval_D.yaml
    ├── scripts/                       # Runnable entry scripts
    │   ├── download_calvin.sh
    │   ├── convert_calvin_to_lerobot.py
    │   ├── train_debug.sh            # Smoke test
    │   ├── train_act_A.sh
    │   ├── train_act_ABC.sh
    │   ├── eval_env_D.sh
    │   └── export_plots.py
    ├── src/calvin_lerobot/            # Core implementation
    │   ├── convert.py                 # CALVIN → LeRobotDataset
    │   ├── split.py                   # Episode-level splitting
    │   ├── offline_eval.py            # Open-loop action L1 eval
    │   ├── rollout_eval.py            # Closed-loop success rate
    │   ├── metrics.py                 # Evaluation metrics
    │   └── visualization.py           # Dataset preview & validation
    ├── tools/                         # Diagnostic utilities
    │   └── inspect_calvin_dataset.py  # Inspect CALVIN structure
    ├── reports/                       # Figures, tables, LaTeX report
    ├── outputs/                       # Training outputs & eval results
    ├── docs/                          # Experiment log & checklist
    └── third_party/                   # CALVIN dependency (gitignored)
```

## Important Notes

### Environment Isolation

The data splitting logic enforces strict environment isolation:
- **Train/Val:** Environments A, B, C only
- **Test:** Environment D only (never seen during training)

This is verified by hard assertions in `split.py`. See `docs/reproduction_checklist.md` for validation steps.

### Offline vs Rollout Evaluation

- **Offline evaluation** (`offline_eval.py`): Open-loop action L1 error on Env-D demonstrations. Fast, no simulator needed, but doesn't measure actual task success.
- **Rollout evaluation** (`rollout_eval.py`): Closed-loop success rate in CALVIN simulator. More meaningful but requires CALVIN installation.

Report both metrics. Clearly state in the report that offline L1 cannot substitute for closed-loop success rate.

### Action Chunking

ACT predicts `chunk_size` future actions at once (default: 100). During inference:
1. Policy predicts a chunk of actions
2. Environment executes actions one by one from the buffer
3. When buffer is exhausted, re-query policy

This is implemented in `rollout_eval.py:ACTPolicyRolloutWrapper`.

## Results

TBD after experiments.

## Model Weights

TBD after training.

## References

- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [ACT: Action Chunking with Transformers](https://tonyzhaozh.github.io/aloha/)
- [CALVIN Benchmark](https://github.com/mees/calvin)

## Troubleshooting

### LeRobot API mismatch

If `convert.py` fails with API errors:

```bash
python scripts/convert_calvin_to_lerobot.py --check_api
```

Check the printed signatures and update `src/calvin_lerobot/convert.py` accordingly.

### CALVIN installation fails

CALVIN has older dependencies (Python 3.8 recommended). If installation fails:
- Use offline evaluation instead of rollout
- Or create a separate conda env for CALVIN: `conda create -n calvin-eval python=3.8`

### Training OOM

Reduce batch size in training scripts:
```bash
--training.batch_size=4  # or 2
```

Both experiments must use the same batch size for fair comparison.

### Data conversion errors

Run the inspection tool first:
```bash
python tools/inspect_calvin_dataset.py --root third_party/calvin/dataset
```

Check that file names and shapes match expectations in `convert.py`.
