#!/bin/bash
# Download CALVIN dataset
# Usage: bash scripts/download_calvin.sh [SPLIT]
#   SPLIT: ABCD (default), ABC, D, or debug

set -euo pipefail

SPLIT=${1:-ABCD}
CALVIN_DIR="${CALVIN_DIR:-./third_party/calvin}"
DATASET_DIR="${CALVIN_DIR}/dataset"

echo "=== Downloading CALVIN dataset: split=${SPLIT} ==="

if [ ! -d "${CALVIN_DIR}" ]; then
  echo "CALVIN not found at ${CALVIN_DIR}, cloning..."
  git clone --recurse-submodules https://github.com/mees/calvin.git "${CALVIN_DIR}"
fi

cd "${DATASET_DIR}"
sh download_data.sh "${SPLIT}"

echo "=== Download complete: ${DATASET_DIR} ==="
echo "Next step: convert to LeRobot format"
echo "  python scripts/convert_calvin_to_lerobot.py \\"
echo "    --calvin_root ${DATASET_DIR} \\"
echo "    --output_root ./data/lerobot_calvin \\"
echo "    --splits A ABC D"
