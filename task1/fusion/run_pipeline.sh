#!/bin/bash
# Merge prepared Gaussian assets and render a lightweight flythrough preview.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="configs/scene_layout.yaml"
OUTPUT_MODEL="output/fused_scene"
ITERATION=30000
RENDER_PREVIEW=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --output_model) OUTPUT_MODEL="$2"; shift 2 ;;
        --iteration) ITERATION="$2"; shift 2 ;;
        --no_preview) RENDER_PREVIEW=0; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

python merge_gaussians.py \
    --config "$CONFIG" \
    --output_model "$OUTPUT_MODEL" \
    --iteration "$ITERATION"

PLY="$OUTPUT_MODEL/point_cloud/iteration_$ITERATION/point_cloud.ply"

if [[ "$RENDER_PREVIEW" == "1" ]]; then
    python render_flythrough.py \
        --ply "$PLY" \
        --output "$OUTPUT_MODEL/flythrough_preview.mp4" \
        --trajectory_json "$OUTPUT_MODEL/flythrough_trajectory.json" \
        --frames 120 \
        --stride 4
fi
