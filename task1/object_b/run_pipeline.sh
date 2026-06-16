#!/bin/bash
# ============================================
# 物体 B: 文本到 3D 生成 完整流水线
# threestudio + Stable Diffusion + SDS Loss
# ============================================

set -e

# ---- CUDA 环境修复 (conda CUDA 13 + gcc 14 + nerfacc JIT) ----
# cuda_runtime.h 在 conda 下的实际路径
export CUDA_HOME="${CONDA_PREFIX:-$HOME/miniconda3/envs/3d}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH}"
# HF 镜像（国内网络必需）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 配置参数 ----
CONFIG_PATH="configs/text-to-3d.yaml"
PROMPT="a high quality 3D render of a cute baby dragon, white background"
OUTPUT_DIR="output"
MAX_STEPS=10000
GPU_ID=0

# ---- 参数解析 ----
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

echo "============================================"
echo " 物体 B: 文本到 3D 生成流水线"
echo "============================================"
echo " 配置文件: $CONFIG_PATH"
echo " Prompt: $PROMPT"
echo " 最大步数: $MAX_STEPS"
echo " GPU: $GPU_ID"
echo "============================================"

# ---- 训练 ----
echo ""
echo "[Step 1/1] 运行 threestudio 文本到 3D 生成..."
python train_text_to_3d.py \
    --config "$CONFIG_PATH" \
    --prompt "$PROMPT" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS" \
    --gpu_id "$GPU_ID" \
    --export_mesh

echo ""
echo "============================================"
echo " 物体 B 生成完成!"
echo " 模型输出: $OUTPUT_DIR"
echo "============================================"
