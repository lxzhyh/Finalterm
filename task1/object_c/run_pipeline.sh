#!/bin/bash
# 物体 C: 单图到 3D 流水线 (去背景 + Zero123 + threestudio)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# CUDA 环境修复 (threestudio + nerfacc JIT)
export CUDA_HOME="${CONDA_PREFIX:-$HOME/miniconda3/envs/3d}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ---- 参数 ----
INPUT_IMAGE=""                   # 输入图像路径 (必填)
PROCESSED_DIR="data/processed"   # 处理后图像目录
CONFIG_PATH="configs/image-to-3d.yaml"
OUTPUT_DIR="output"
MAX_STEPS=5000
GPU_ID=0
BG_METHOD="rembg"               # 背景去除方法: rembg 或 sam

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --image) INPUT_IMAGE="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max_steps) MAX_STEPS="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --bg_method) BG_METHOD="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ -z "$INPUT_IMAGE" ]; then
    echo "用法: bash run_pipeline.sh --image <图像路径>"
    echo "  --config       配置文件 (默认 configs/image-to-3d.yaml)"
    echo "  --output_dir   输出目录 (默认 output)"
    echo "  --max_steps    最大步数 (默认 5000)"
    echo "  --gpu          GPU ID (默认 0)"
    echo "  --bg_method    去背景方法: rembg/sam (默认 rembg)"
    exit 1
fi

# 获取文件名（不含扩展名）
BASENAME=$(basename "$INPUT_IMAGE")
FILENAME="${BASENAME%.*}"
PROCESSED_IMAGE="$PROCESSED_DIR/${FILENAME}_nobg.png"

echo "物体 C: image=$INPUT_IMAGE bg=$BG_METHOD config=$CONFIG_PATH max_steps=$MAX_STEPS gpu=$GPU_ID"

# Step 1: 去背景
echo ""
echo "--- 去背景 ---"
python remove_background.py \
    --input "$INPUT_IMAGE" \
    --output "$PROCESSED_IMAGE" \
    --method "$BG_METHOD" \
    --crop

# Step 2: Zero123
echo ""
echo "--- Zero123 ---"
python train_image_to_3d.py \
    --config "$CONFIG_PATH" \
    --image "$PROCESSED_IMAGE" \
    --output_dir "$OUTPUT_DIR" \
    --max_steps "$MAX_STEPS" \
    --gpu_id "$GPU_ID" \
    --export_mesh

echo ""
echo "物体 C 搞定，处理后图像: $PROCESSED_IMAGE，模型在 $OUTPUT_DIR"
