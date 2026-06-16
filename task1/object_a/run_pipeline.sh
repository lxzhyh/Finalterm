#!/bin/bash
# ============================================
# 物体 A: 真实多视角重建 完整流水线
# COLMAP 位姿提取 + 3DGS 高斯溅射重建
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 配置参数 ----
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

# ---- 参数解析 ----
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
    echo "用法: bash run_pipeline.sh --video <视频路径> [选项]"
    echo ""
    echo "选项:"
    echo "  --video <path>       输入视频路径 (必填)"
    echo "  --interval <int>     抽帧间隔 (默认: 2)"
    echo "  --max_frames <int>   最大帧数 (默认: 300)"
    echo "  --iterations <int>   3DGS 训练迭代 (默认: 30000)"
    echo "  --resolution <int>   图像分辨率 (默认: -1)"
    echo "  --max_size <int>     抽帧最大边长 (默认: 1280)"
    echo "  --gpu <int>          GPU ID (默认: 0)"
    exit 1
fi

echo "============================================"
echo " 物体 A: 真实多视角重建流水线"
echo "============================================"
echo " 视频: $VIDEO_PATH"
echo " 抽帧间隔: $FRAME_INTERVAL"
echo " 最大帧数: $MAX_FRAMES"
echo " 3DGS 迭代: $ITERATIONS"
echo " 图像最大边长: $MAX_SIZE"
echo " GPU: $GPU_ID"
echo "============================================"

# ---- Step 1: 抽帧 ----
echo ""
echo "[Step 1/3] 从视频中抽取帧图像..."
python extract_frames.py \
    --video "$VIDEO_PATH" \
    --output_dir "$IMAGE_DIR" \
    --interval "$FRAME_INTERVAL" \
    --max_frames "$MAX_FRAMES" \
    --max_size "$MAX_SIZE"

# ---- Step 2: COLMAP ----
echo ""
echo "[Step 2/3] 运行 COLMAP SfM..."
python run_colmap.py \
    --image_dir "$IMAGE_DIR" \
    --workspace "$COLMAP_WORKSPACE" \
    --gpu_id "$GPU_ID"

# ---- Step 3: 3DGS 训练 ----
echo ""
echo "[Step 3/3] 训练 3D Gaussian Splatting..."
DENSE_DIR="$COLMAP_WORKSPACE/dense"
python train_3dgs.py \
    --source_path "$DENSE_DIR" \
    --output_path "$OUTPUT_DIR" \
    --iterations "$ITERATIONS" \
    --resolution "$RESOLUTION" \
    --gpu_id "$GPU_ID" \
    --render

echo ""
echo "============================================"
echo " 物体 A 重建完成!"
echo " 模型输出: $OUTPUT_DIR"
echo "============================================"
