#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <parts_dir> <part_01> <part_02> [part_03 ...]" >&2
    exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTS_DIR="$1"
shift
PART_NAMES=("$@")

PYTHON_BIN="${PYTHON_BIN:-python}"
VOXEL_SIZE="${VOXEL_SIZE:-0.05}"
OVERLAP_FRAMES="${OVERLAP_FRAMES:-120}"
SKIP_FRAMES="${SKIP_FRAMES:-3}"
MIN_FITNESS="${MIN_FITNESS:-0.30}"
MAX_RMSE="${MAX_RMSE:-0.10}"

PARTS_PARENT="$(cd "$(dirname "$PARTS_DIR")" && pwd)"
PARTS_NAME="$(basename "$PARTS_DIR")"
TRANSFORMATIONS="$PARTS_PARENT/${PARTS_NAME}_transformations.json"
ALIGNED_PREVIEW="$PARTS_PARENT/${PARTS_NAME}_aligned_preview.ply"
TRANSFORMED_DIR="$PARTS_PARENT/${PARTS_NAME}_transformed"
MERGED_DIR="$PARTS_PARENT/${PARTS_NAME}_merged"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "[1/4] Validating capture parts"
"$PYTHON_BIN" tools/multipart_capture.py validate-room \
    --room-dir "$PARTS_DIR" \
    --part-names "${PART_NAMES[@]}" \
    --require-timestamps

echo "[2/4] Registering adjacent overlap windows"
"$PYTHON_BIN" tools/align_pointclouds.py \
    --parts-dir "$PARTS_DIR" \
    --part-names "${PART_NAMES[@]}" \
    --output "$TRANSFORMATIONS" \
    --save-aligned-pcd "$ALIGNED_PREVIEW" \
    --pose-mode planar \
    --voxel-size "$VOXEL_SIZE" \
    --overlap-frames "$OVERLAP_FRAMES" \
    --skip-frames "$SKIP_FRAMES" \
    --min-fitness "$MIN_FITNESS" \
    --max-rmse "$MAX_RMSE"

echo "[3/4] Converting every camera pose into the global HOV-SG frame"
"$PYTHON_BIN" tools/transform_poses.py \
    --parts-dir "$PARTS_DIR" \
    --part-names "${PART_NAMES[@]}" \
    --transformations "$TRANSFORMATIONS" \
    --output-dir "$TRANSFORMED_DIR"

echo "[4/4] Merging RGB, depth, global poses, and frame metadata"
"$PYTHON_BIN" tools/merge_multipart_data.py \
    --transformed-dir "$TRANSFORMED_DIR" \
    --part-names "${PART_NAMES[@]}" \
    --output-dir "$MERGED_DIR"

echo "Registration: $TRANSFORMATIONS"
echo "Aligned preview: $ALIGNED_PREVIEW"
echo "Merged HOV-SG dataset: $MERGED_DIR"
