#!/usr/bin/env bash
# scripts/setup_env.sh
# One-shot environment setup for the DINOv2+LightGlue project.
# Run once after cloning: bash scripts/setup_env.sh
# Assumes: virtual environment already activated at .venv/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================"
echo " DINOv2 + LightGlue — Environment Setup"
echo " Project root: $PROJECT_ROOT"
echo "======================================================"

# Verify we're in the venv
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

echo ""
echo "[1/5] Checking PyTorch and CUDA..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  VRAM: {mem:.1f} GB')
"

echo ""
echo "[2/5] Installing missing packages..."
pip install --quiet \
    scipy \
    h5py \
    einops \
    tqdm \
    omegaconf \
    tensorboard \
    wandb \
    albumentations \
    seaborn \
    joblib \
    "scikit-learn~=1.3.0" \
    timm \
    poselib

echo "  Core packages installed."

echo ""
echo "[3/5] Installing glue-factory in editable mode..."
if python -c "import gluefactory" 2>/dev/null; then
    echo "  glue-factory already installed."
else
    pip install --quiet -e glue-factory/
    echo "  glue-factory installed."
fi

echo ""
echo "[4/5] Installing LightGlue in editable mode..."
if python -c "import lightglue" 2>/dev/null; then
    echo "  LightGlue already installed."
else
    pip install --quiet -e LightGlue/
    echo "  LightGlue installed."
fi

echo ""
echo "[5/5] Creating data directories..."
mkdir -p data/megadepth data/scannet data/hpatches checkpoints cache experiments logs

echo ""
echo "======================================================"
echo " Verifying imports..."
python -c "
import torch
import torchvision
import numpy as np
import cv2
import scipy
import h5py
import tqdm
import omegaconf
import albumentations
import kornia
import poselib
print('  All core imports OK')

try:
    import gluefactory
    print('  gluefactory OK')
except ImportError as e:
    print(f'  WARNING: gluefactory not available: {e}')

try:
    import lightglue
    print('  lightglue OK')
except ImportError as e:
    print(f'  WARNING: lightglue not available: {e}')
"

echo ""
echo "======================================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Download eval data:   bash scripts/download_data.sh"
echo "   2. Open notebook:        jupyter lab Exp1.ipynb"
echo "   3. Run Day 1 cells to verify GPU and environment"
echo "======================================================"
