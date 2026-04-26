# DINOv2 + LightGlue: Foundation Model Descriptors for Sparse Feature Matching

> **"Foundation Model Descriptors Meet Sparse Matching: Bridging DINOv2 and LightGlue for Robust Feature Correspondence"**
>
> Target Venues: CVPR/ECCV Workshop (IMCW, LSCV) · WACV / BMVC 2026
>
> Hardware: NVIDIA RTX 3060 (12 GB VRAM) · Timeline: 25–30 Days

---

## Overview

This project investigates whether DINOv2 (Meta's self-supervised Vision Transformer) patch features can replace SuperPoint's learned descriptors in the LightGlue sparse feature matcher, yielding more robust correspondences — especially under large viewpoint changes, illumination shifts, and cross-domain scenarios.

**Key insight:** RoMa (CVPR 2024) proved frozen DINOv2 dramatically improves *dense* matching. This project closes the gap for *sparse* matching with LightGlue's efficient adaptive architecture.

---

## Hardware & Storage

| Resource | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 (12 GB VRAM) |
| RAM | 16 GB |
| Disk (free) | ~113 GB on `/dev/nvme0n1p5` |
| PyTorch | 2.11.0+cu130 |
| Python | 3.11 |

---

## Repository Structure

```
DLCV_VEDANSH/
├── README.md                       ← You are here
├── CONTRIBUTING.md                 ← Git workflow & branching rules (READ BEFORE COMMITTING)
├── plan.md                         ← Full 30-day research master plan
├── requirement.txt                 ← Python package requirements
├── Exp1.ipynb                      ← Main experiment notebook (Week 1: Days 1–7)
│
├── .github/
│   ├── AGENTS.md                   ← AI agent git rules (READ BEFORE ANY GIT OPERATION)
│   └── workflows/                  ← (future CI/CD)
│
├── docs/
│   ├── sessionlogs.md              ← AI coding session log (auto-updated by agent)
│   └── changelog.md                ← Code change history
│
├── src/                            ← Custom Python modules
│   ├── __init__.py
│   ├── dinov2_extractor.py         ← DINOv2 feature extraction module
│   ├── feature_sampling.py         ← Bilinear sampling of patch features at keypoints
│   ├── projection.py               ← Projection MLP variants (Linear / MLP-1 / MLP-2)
│   ├── pipeline.py                 ← End-to-end SuperPoint+DINOv2+LightGlue pipeline
│   ├── cache_features.py           ← Offline DINOv2 feature caching to HDF5
│   ├── evaluate.py                 ← Evaluation utilities
│   ├── models/
│   │   └── __init__.py
│   └── utils/
│       ├── __init__.py
│       └── viz.py                  ← Visualization helpers
│
├── configs/
│   ├── dinov2_lightglue_base.yaml  ← Training config: DINOv2-B + LightGlue
│   ├── dinov2_lightglue_small.yaml ← Training config: DINOv2-S + LightGlue
│   └── ablation/
│       ├── proj_linear.yaml
│       ├── proj_mlp1.yaml
│       └── proj_mlp2.yaml
│
├── scripts/
│   ├── setup_env.sh                ← One-shot environment setup script
│   ├── download_data.sh            ← Download all evaluation datasets
│   └── cache_dinov2_features.sh    ← Launch offline DINOv2 feature caching
│
├── experiments/
│   ├── exp1_1/                     ← Zero-shot descriptor quality results
│   ├── exp1_2/                     ← Pretrained LG + DINOv2 results
│   └── ...
│
├── data/                           ← (gitignored) Downloaded datasets
│   ├── megadepth/
│   ├── scannet/
│   └── hpatches/
│
├── checkpoints/                    ← (gitignored) Model checkpoints
│
├── cache/                          ← (gitignored) Cached DINOv2 HDF5 features
│
├── glue-factory/                   ← Upstream: cvg/glue-factory (submodule)
└── LightGlue/                      ← Upstream: cvg/LightGlue (submodule)
```

---

## Quick Start

### 1. Clone with submodules
```bash
git clone --recurse-submodules <repo-url>
cd DLCV_VEDANSH
```

### 2. Set up the environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate

# Install all dependencies
bash scripts/setup_env.sh
```

### 3. Verify GPU
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 3060
```

### 4. Download evaluation datasets (~4.4 GB)
```bash
bash scripts/download_data.sh
```

### 5. Run the SuperPoint+LightGlue baseline
```bash
cd glue-factory
python -m gluefactory.eval.megadepth1500 \
    --conf superpoint+lightglue-official \
    --overwrite
```

### 6. Run Week 1 experiments
Open `Exp1.ipynb` in JupyterLab or VS Code and run cells top-to-bottom.

---

## Environment Setup (Manual)

```bash
source .venv/bin/activate

# Core dependencies
pip install scipy h5py einops tqdm omegaconf tensorboard wandb
pip install albumentations seaborn joblib "scikit-learn~=1.3.0"
pip install timm                          # for DINOv2 loading via timm (alt)

# Install glue-factory in editable mode
pip install -e glue-factory/

# Install LightGlue in editable mode
pip install -e LightGlue/

# Verify
python -c "import gluefactory; import lightglue; print('All OK')"
```

---

## Experiment Phases

| Phase | Days | Goal | Key Deliverable |
|---|---|---|---|
| **Phase 1** | 1–7 | Proof of concept | Zero-shot DINOv2 quality, Go/No-Go gate |
| **Phase 2** | 8–15 | Core training | Fine-tuned DINOv2+LightGlue model |
| **Phase 3** | 16–20 | Full evaluation | Complete benchmark tables |
| **Phase 4** | 21–28 | Paper writing | Draft paper |
| **Phase 5** | 29–30 | Buffer & submission | Camera-ready |

---

## Key Design Decisions

1. **Keypoint detector:** SuperPoint (frozen, keypoints only — not descriptors)
2. **Descriptors:** DINOv2-B/14 patch features sampled via bilinear interpolation at keypoint locations
3. **Projection:** MLP `768→512→256` with LayerNorm + GELU (best ablation — see Plan §4)
4. **LightGlue init:** From SuperPoint pretrained weights (transformer layers are dim-agnostic)
5. **Training:** Freeze DINOv2, train projection MLP + LightGlue transformer on MegaDepth subset

---

## Disk Space Budget

| Item | Estimated Size |
|---|---|
| Evaluation datasets (MegaDepth-1500, ScanNet-1500, HPatches) | ~4.4 GB |
| MegaDepth training subset (~50 scenes) | ~30–50 GB |
| Cached DINOv2 features (HDF5) | ~15–25 GB |
| Model checkpoints | ~2 GB |
| Code, logs, results | ~5 GB |
| **Total** | **~60–85 GB** ⚠️ Leave buffer from 113 GB free |

---

## Branching Strategy

See [CONTRIBUTING.md](CONTRIBUTING.md) and [.github/AGENTS.md](.github/AGENTS.md) for the full git workflow. In brief:

- `main` — protected, only merge via PR
- `develop` — integration branch
- `feature/<name>` — feature work
- `exp/<name>` — experiment branches
- `hotfix/<name>` — critical fixes

---

## License

This project uses:
- [glue-factory](https://github.com/cvg/glue-factory) — Apache 2.0
- [LightGlue](https://github.com/cvg/LightGlue) — Apache 2.0
- [DINOv2](https://github.com/facebookresearch/dinov2) — Apache 2.0

Research code in `src/` is released under Apache 2.0.
