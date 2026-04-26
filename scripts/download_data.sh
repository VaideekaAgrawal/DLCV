#!/usr/bin/env bash
# scripts/download_data.sh
# Download all evaluation datasets for the project.
#
# Datasets:
#   - MegaDepth-1500 (~1.5 GB) — primary pose evaluation
#   - ScanNet-1500   (~1.1 GB) — indoor pose evaluation
#   - HPatches       (~1.8 GB) — homography estimation
#
# Total: ~4.4 GB
# Uses glue-factory's auto-download via dataset loaders.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GLUEFACTORY="$PROJECT_ROOT/glue-factory"
cd "$PROJECT_ROOT"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source .venv/bin/activate
fi

echo "======================================================"
echo " Downloading evaluation datasets"
echo " Storage required: ~4.4 GB"
echo " Free disk: $(df -h . | awk 'NR==2{print $4}')"
echo "======================================================"

# Check glue-factory is available
if ! python -c "import gluefactory" 2>/dev/null; then
    echo "ERROR: gluefactory not installed. Run: bash scripts/setup_env.sh"
    exit 1
fi

# Read data root from glue-factory settings (or use default)
DATA_ROOT=$(python -c "
from pathlib import Path
try:
    from gluefactory.settings import DATA_PATH
    print(DATA_PATH)
except:
    print(Path.home() / 'data' / 'gluefactory')
")
echo "Data root: $DATA_ROOT"
mkdir -p "$DATA_ROOT"

echo ""
echo "[1/3] MegaDepth-1500 evaluation split..."
cd "$GLUEFACTORY"
python -c "
from gluefactory.datasets.megadepth import MegaDepth1500
# Triggers auto-download of the 1500-pair split
try:
    ds = MegaDepth1500({'root': '$DATA_ROOT/megadepth', 'split': 'test'})
    print(f'  MegaDepth-1500: {len(ds)} pairs loaded')
except Exception as e:
    print(f'  Download via eval script...')
    import subprocess
    result = subprocess.run(
        ['python', '-m', 'gluefactory.datasets.megadepth', '--download'],
        capture_output=True, text=True
    )
    print(result.stdout)
" 2>&1 || echo "  Note: MegaDepth will download on first eval run."

echo ""
echo "[2/3] ScanNet-1500 evaluation split..."
python -c "
try:
    from gluefactory.datasets.scannet import ScanNet1500
    ds = ScanNet1500({'root': '$DATA_ROOT/scannet', 'split': 'test'})
    print(f'  ScanNet-1500: {len(ds)} pairs loaded')
except Exception as e:
    print(f'  ScanNet-1500 will download on first eval run: {e}')
" 2>&1 || echo "  Note: ScanNet will download on first eval run."

echo ""
echo "[3/3] HPatches evaluation split..."
python -c "
try:
    from gluefactory.datasets.hpatches import HPatches
    ds = HPatches({'root': '$DATA_ROOT/hpatches', 'split': 'test'})
    print(f'  HPatches: {len(ds)} pairs loaded')
except Exception as e:
    print(f'  HPatches will download on first eval run: {e}')
" 2>&1 || echo "  Note: HPatches will download on first eval run."

cd "$PROJECT_ROOT"

echo ""
echo "======================================================"
echo " Data download initiated. Large files download during"
echo " the first eval run."
echo ""
echo " To trigger downloads and run baseline evaluation:"
echo "   cd glue-factory"
echo "   python -m gluefactory.eval.megadepth1500 \\"
echo "     --conf superpoint+lightglue-official \\"
echo "     --overwrite"
echo ""
echo " Expected baseline (SuperPoint+LightGlue):"
echo "   MegaDepth-1500: AUC@5=66.8 / @10=79.3 / @20=87.9"
echo "   HPatches:       AUC@1=37.1 / @3=67.4 / @5=77.8"
echo "======================================================"
