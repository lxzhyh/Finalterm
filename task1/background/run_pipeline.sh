#!/bin/bash
# Train a Mip-NeRF 360 background scene with Graphdeco 3DGS.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SCENE_PATH="data/garden"
IMAGES="images_4"
OUTPUT_PATH="output/garden"
ITERATIONS=30000
RESOLUTION=-1
GPU_ID=0
RENDER=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene_path) SCENE_PATH="$2"; shift 2 ;;
        --images) IMAGES="$2"; shift 2 ;;
        --output_path) OUTPUT_PATH="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --resolution) RESOLUTION="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --render) RENDER=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

CMD=(
    python train_background_3dgs.py
    --scene_path "$SCENE_PATH"
    --images "$IMAGES"
    --output_path "$OUTPUT_PATH"
    --iterations "$ITERATIONS"
    --resolution "$RESOLUTION"
    --gpu_id "$GPU_ID"
)

if [[ "$RENDER" == "1" ]]; then
    CMD+=(--render)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
