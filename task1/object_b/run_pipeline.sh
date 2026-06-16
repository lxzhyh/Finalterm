#!/bin/bash
# 物体 B: text-to-3d 流水线 (threestudio + SDS)
set -e

# CUDA 环境修复 (conda CUDA 13 + nerfacc JIT)
export CUDA_HOME="${CONDA_PREFIX:-$HOME/miniconda3/envs/3d}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH}"
# HF 镜像（国内网络必需）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 参数 ----
CONFIG_PATH="configs/text-to-3d.yaml"
PROMPT="a high quality 3D render of a cute baby dragon, white background"
OUTPUT_DIR="output"
MAX_STEPS=10000
GPU_ID=0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max_steps) MAX_STEPS="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "物体 B: config=$CONFIG_PATH prompt='$PROMPT' max_steps=$MAX_STEPS gpu=$GPU_ID"

# 训练
echo ""
echo "--- text-to-3d ---"
python train_text_to_3d.py \
    --config "$CONFIG_PATH" \
    --prompt "$PROMPT" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS" \
    --gpu_id "$GPU_ID" \
    --export_mesh

echo ""
echo "物体 B 搞定，模型在 $OUTPUT_DIR"
