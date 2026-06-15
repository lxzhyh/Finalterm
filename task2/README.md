# HW3 Task 2: LeRobot ACT Cross-Environment Generalization on CALVIN

## Overview

This project implements cross-environment generalization experiments for the ACT (Action Chunking with Transformers) policy using the LeRobot framework on the CALVIN benchmark.

**Core comparison:**

| Experiment | Training Data | Test Data | Purpose |
|---|---|---|---|
| `act_A_only` | Env-A | Env-D (zero-shot) | Baseline single-environment policy |
| `act_ABC_joint` | Env-A + B + C | Env-D (zero-shot) | Multi-environment joint training |

## Environment Setup

```bash
conda create -n hw3-act python=3.10 -y
conda activate hw3-act

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install LeRobot
pip install "lerobot>=0.5.1"

# Install additional dependencies
pip install -r requirements.txt
```

Install CALVIN (for rollout evaluation):

```bash
git clone --recurse-submodules https://github.com/mees/calvin.git third_party/calvin
cd third_party/calvin
sh install.sh
cd ../..
```

## Dataset Preparation

Download CALVIN dataset:

```bash
bash scripts/download_calvin.sh
```

## Convert CALVIN to LeRobotDataset

```bash
python scripts/convert_calvin_to_lerobot.py \
  --calvin_root /path/to/calvin/dataset \
  --output_root data/lerobot_calvin \
  --splits A ABC D \
  --val_ratio 0.1 \
  --seed 42 \
  --cameras static gripper \
  --action_space relative_cartesian
```

## Train ACT on Env-A

```bash
bash scripts/train_act_A.sh
```

## Train ACT on Env-A/B/C

```bash
bash scripts/train_act_ABC.sh
```

## Evaluate on Env-D

```bash
bash scripts/eval_env_D.sh
```

## Results

TBD after experiments.

## Model Weights

TBD after training.

## Repository Structure

```
Finalterm/
└── task2/
    ├── README.md               # This file
    ├── requirements.txt        # Python dependencies
    ├── environment.yml         # Conda environment
    ├── configs/                # Training & eval configs
    ├── scripts/                # Runnable entry scripts
    ├── src/calvin_lerobot/     # Data conversion & evaluation code
    ├── reports/                # Figures, tables, LaTeX report
    ├── outputs/                # Training outputs & eval results
    ├── docs/                   # Experiment log & checklist
    └── third_party/            # CALVIN dependency
```

## References

- [LeRobot](https://github.com/huggingface/lerobot)
- [ACT: Action Chunking with Transformers](https://tonyzhaozh.github.io/aloha/)
- [CALVIN Benchmark](https://github.com/mees/calvin)
