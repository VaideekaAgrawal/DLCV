#!/usr/bin/env bash
# scripts/cache_dinov2_features.sh
# Launch offline DINOv2 feature caching for MegaDepth training images.
#
# This pre-extracts DINOv2-B features and saves to HDF5, allowing
# training without loading DINOv2 on the GPU (saves ~1.5 GB VRAM).
#
# Usage:
#   bash scripts/cache_dinov2_features.sh [--variant vitb14_reg] [--max_images N]
#
# Estimated time: ~2-4 hours for 50k images on RTX 3060

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source .venv/bin/activate
fi

# Default arguments
VARIANT="${VARIANT:-vitb14_reg}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_IMAGES="${MAX_IMAGES:-}"
DEVICE="${DEVICE:-cuda}"

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case $1 in
        --variant) VARIANT="$2"; shift 2;;
        --batch_size) BATCH_SIZE="$2"; shift 2;;
        --max_images) MAX_IMAGES="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# Get data root from glue-factory
DATA_ROOT=$(python -c "
try:
    from gluefactory.settings import DATA_PATH
    print(DATA_PATH)
except:
    import pathlib; print(pathlib.Path.home() / 'data' / 'gluefactory')
")

IMAGE_DIR="$DATA_ROOT/megadepth/train"
OUTPUT_FILE="$PROJECT_ROOT/cache/megadepth_dinov2_${VARIANT}_features.h5"

echo "======================================================"
echo " DINOv2 Feature Caching"
echo " Variant:    $VARIANT"
echo " Image dir:  $IMAGE_DIR"
echo " Output:     $OUTPUT_FILE"
echo " Batch size: $BATCH_SIZE"
echo " Device:     $DEVICE"
[[ -n "$MAX_IMAGES" ]] && echo " Max images: $MAX_IMAGES"
echo " Free disk:  $(df -h . | awk 'NR==2{print $4}')"
echo "======================================================"

if [[ ! -d "$IMAGE_DIR" ]]; then
    echo "WARNING: Image directory not found: $IMAGE_DIR"
    echo "Download MegaDepth training data first."
    echo "Exiting without caching."
    exit 0
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"

MAX_ARGS=""
[[ -n "$MAX_IMAGES" ]] && MAX_ARGS="--max_images $MAX_IMAGES"

python -m src.cache_features \
    --image_dir "$IMAGE_DIR" \
    --output_file "$OUTPUT_FILE" \
    --variant "$VARIANT" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    $MAX_ARGS

echo ""
echo "Caching complete!"
echo "Output file: $OUTPUT_FILE"
if [[ -f "$OUTPUT_FILE" ]]; then
    SIZE=$(du -sh "$OUTPUT_FILE" | cut -f1)
    echo "File size: $SIZE"
fi
