#!/bin/bash
# 物体 A: COLMAP + 3DGS 多视角重建流水线
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 参数 ----
VIDEO_PATH=""                    # 输入视频路径 (必填)
IMAGE_DIR="data/images"          # 帧图像输出目录
COLMAP_WORKSPACE="data"          # COLMAP 工作目录
OUTPUT_DIR="output"              # 3DGS 模型输出目录
FRAME_INTERVAL=10                # 抽帧间隔 (视频序列需足够视差)
MAX_FRAMES=300                   # 最大帧数
ITERATIONS=30000                 # 3DGS 训练迭代次数
RESOLUTION=-1                    # 图像分辨率 (-1 为原始)
MAX_SIZE=1280                    # 抽帧时最大边长 (避免 COLMAP CPU OOM)
GPU_ID=0                         # GPU 设备 ID

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --video) VIDEO_PATH="$2"; shift 2 ;;
        --interval) FRAME_INTERVAL="$2"; shift 2 ;;
        --max_frames) MAX_FRAMES="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --resolution) RESOLUTION="$2"; shift 2 ;;
        --max_size) MAX_SIZE="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ -z "$VIDEO_PATH" ]; then
    echo "用法: bash run_pipeline.sh --video <视频路径>"
    echo "  --interval     抽帧间隔 (默认 2)"
    echo "  --max_frames   最大帧数 (默认 300)"
    echo "  --iterations   3DGS 迭代数 (默认 30000)"
    echo "  --resolution   图像分辨率 (默认 -1)"
    echo "  --max_size     抽帧最大边长 (默认 1280)"
    echo "  --gpu          GPU ID (默认 0)"
    exit 1
fi

echo "物体 A: 视频=$VIDEO_PATH 帧间隔=$FRAME_INTERVAL 最大帧=$MAX_FRAMES iter=$ITERATIONS max_size=$MAX_SIZE gpu=$GPU_ID"

# Step 1: 抽帧
echo ""
echo "--- 抽帧 ---"
python extract_frames.py \
    --video "$VIDEO_PATH" \
    --output_dir "$IMAGE_DIR" \
    --interval "$FRAME_INTERVAL" \
    --max_frames "$MAX_FRAMES" \
    --max_size "$MAX_SIZE"

# Step 2: COLMAP
echo ""
echo "--- COLMAP ---"
python run_colmap.py \
    --image_dir "$IMAGE_DIR" \
    --workspace "$COLMAP_WORKSPACE" \
    --gpu_id "$GPU_ID"

# Step 3: 3DGS
echo ""
echo "--- 3DGS ---"
DENSE_DIR="$COLMAP_WORKSPACE/dense"
python train_3dgs.py \
    --source_path "$DENSE_DIR" \
    --output_path "$OUTPUT_DIR" \
    --iterations "$ITERATIONS" \
    --resolution "$RESOLUTION" \
    --gpu_id "$GPU_ID" \
    --render

echo ""
echo "物体 A 搞定，模型在 $OUTPUT_DIR"
