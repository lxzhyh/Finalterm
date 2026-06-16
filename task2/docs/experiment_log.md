# Experiment Log

## Setup

- **Date started:** 2026-06-15
- **Platform:** TBD (local MacBook M4 / cloud GPU)
- **GPU:** TBD
- **Python:** 3.10
- **PyTorch:** TBD (>=2.0)
- **LeRobot:** TBD (>=0.5.1, installed via `pip install "lerobot[aloha]"`)
- **CUDA:** TBD

## Data Preparation

- **CALVIN version:** TBD
- **Data split:** ABCD
- **Download date:** TBD
- **Conversion date:** TBD
- **Episode counts:**
  - Env-A train: TBD
  - Env-A val: TBD
  - Env-B train: TBD
  - Env-B val: TBD
  - Env-C train: TBD
  - Env-C val: TBD
  - Env-D test: TBD

## Experiment 1: act_A_only (Baseline)

- **Config:** `configs/act_A_only.yaml`
- **Dataset:** `calvin_A_train` (Env-A only)
- **Validation:** `calvin_A_val`
- **Start time:** TBD
- **End time:** TBD
- **Total training time:** TBD hours
- **Final train steps:** TBD
- **WandB run:** TBD
- **Best checkpoint:** `outputs/train/act_A_only_seed42/checkpoints/best_model.pt`
- **Best val action L1:** TBD
- **Best val epoch:** TBD

### Training Observations

- Loss curve shape: TBD
- Convergence speed: TBD
- Any anomalies: TBD

## Experiment 2: act_ABC_joint (Multi-Environment)

- **Config:** `configs/act_ABC_joint.yaml`
- **Dataset:** `calvin_ABC_train` (Env-A + B + C)
- **Validation:** `calvin_ABC_val`
- **Start time:** TBD
- **End time:** TBD
- **Total training time:** TBD hours
- **Final train steps:** TBD
- **WandB run:** TBD
- **Best checkpoint:** `outputs/train/act_ABC_joint_seed42/checkpoints/best_model.pt`
- **Best val action L1:** TBD
- **Best val epoch:** TBD

### Training Observations

- Loss curve shape: TBD
- Convergence speed: TBD
- Comparison to A-only: TBD (faster/slower convergence?)
- Any anomalies: TBD

## Evaluation: Env-D Zero-shot

### Offline Evaluation (Action L1 on Env-D demos)

| Model | Mean Action L1 | Position L1 | Rotation L1 | Gripper Error |
|-------|---------------|-------------|-------------|---------------|
| act_A_only | TBD | TBD | TBD | TBD |
| act_ABC_joint | TBD | TBD | TBD | TBD |

**Results files:**
- `outputs/eval/act_A_only_D_offline.json`
- `outputs/eval/act_ABC_joint_D_offline.json`

**Observations:**
- Does ABC have lower L1 error? TBD
- Which action dimensions show biggest improvement? TBD
- Chunk boundary analysis: TBD

### Rollout Evaluation (Closed-loop Success Rate)

| Model | Success Rate | Mean Completed Tasks | Num Sequences |
|-------|-------------|---------------------|---------------|
| act_A_only | TBD | TBD | 100 |
| act_ABC_joint | TBD | TBD | 100 |

**Results files:**
- `outputs/eval/act_A_only_D_rollout.json`
- `outputs/eval/act_ABC_joint_D_rollout.json`

**Note:** If CALVIN installation failed, this section may be incomplete. Offline evaluation still provides useful signal.

**Observations:**
- Does ABC achieve higher success rate? TBD
- Failure modes observed: TBD
- Action chunking behavior: TBD

## Action Chunking Analysis

### Hypothesis

ACT's action chunking should provide:
1. Smoother trajectories (lower variance within chunks)
2. Better handling of visual distribution shift (chunk buffer reduces per-frame dependency)
3. More stable grasping (multi-step planning)

### Observations

**Chunk boundary delta:**
- act_A_only: TBD
- act_ABC_joint: TBD

**Chunk inner variance:**
- act_A_only: TBD
- act_ABC_joint: TBD

**Interpretation:**
- Does multi-env training improve chunk smoothness? TBD
- Are chunk boundaries more stable in ABC model? TBD

## Key Findings

1. **Convergence:** TBD (which model converges faster/better?)
2. **Generalization:** TBD (does ABC generalize better to Env-D?)
3. **Action Chunking:** TBD (how does chunking affect robustness?)
4. **Offline vs Rollout:** TBD (do offline metrics correlate with rollout success?)

## Issues Encountered

### Issue 1: TBD
- **Description:** TBD
- **Solution:** TBD
- **Impact on results:** TBD

### Issue 2: TBD
- **Description:** TBD
- **Solution:** TBD
- **Impact on results:** TBD

## Hyperparameter Table

| Parameter | Value |
|-----------|-------|
| Policy | ACT |
| Seed | 42 |
| Batch size | 8 |
| Training steps | 100000 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.0 |
| LR scheduler | cosine |
| Warmup steps | 500 |
| Grad clip norm | 10.0 |
| Loss | L1 |
| Chunk size | 100 |
| Cameras | static + gripper |
| Static image size | 200×200 |
| Gripper image size | 84×84 |
| Action space | relative cartesian (7D) |
| State space | proprioceptive (15D) |

## Dataset Split Table

| Environment | Train Episodes | Val Episodes | Test Episodes |
|-------------|---------------|--------------|---------------|
| A | TBD | TBD | 0 |
| B | TBD | TBD | 0 |
| C | TBD | TBD | 0 |
| D | 0 | 0 | TBD |

## Next Steps

- [ ] Complete all TBD fields after experiments
- [ ] Add failure case visualizations
- [ ] Write up report based on findings
- [ ] Upload checkpoints to cloud storage
