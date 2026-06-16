# Outputs Directory

This directory stores training outputs and evaluation results.

**Do NOT commit large files (checkpoints, videos) to git.**

## Structure

```
outputs/
├── train/
│   ├── act_A_only_seed42/
│   │   ├── checkpoints/
│   │   │   └── best_model.pt      # Best checkpoint by val loss
│   │   ├── config.yaml            # Training config
│   │   └── train.log              # Training log
│   └── act_ABC_joint_seed42/
│       ├── checkpoints/
│       │   └── best_model.pt
│       ├── config.yaml
│       └── train.log
├── eval/
│   ├── act_A_only_D.json          # Rollout eval results
│   ├── act_ABC_joint_D.json
│   ├── act_A_only_D_offline.json  # Offline L1 eval (fallback)
│   └── act_ABC_joint_D_offline.json
└── README.md
```

## Model Weights

Best checkpoints are uploaded to cloud storage separately:

```
act_A_only best checkpoint: <网盘链接 TBD>
act_ABC_joint best checkpoint: <网盘链接 TBD>
extraction code: <TBD>
```
