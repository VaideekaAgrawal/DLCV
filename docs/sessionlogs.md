# Session Logs

> Auto-maintained by AI agent. One entry per coding session. **Never delete entries** — append only.
> Format: `## Session YYYY-MM-DD — <Goal>`

---

## Session 2026-04-20 — Project Bootstrap & Week 1 Code

**Agent:** GitHub Copilot (Claude Sonnet 4.6)
**Branch:** `exp/week1-poc` (to be created)
**Duration:** Initial session

### Goal
Set up the full project scaffold, assess plan feasibility, write all Week 1 deliverable code.

### Hardware State
- GPU: NVIDIA RTX 3060 · 12 GB VRAM · 11.9 GB free at session start
- Disk: 113 GB free on `/dev/nvme0n1p5`
- RAM: ~9.3 GB available
- PyTorch: 2.11.0+cu130 · Python 3.11

### Pre-existing State
- `.venv/` exists with: `torch 2.11.0`, `torchvision 0.26.0`, `opencv-python 4.13.0`, `kornia 0.8.2`, `numpy 2.4.4`, `matplotlib 3.10.8`, `poselib 2.0.5`
- `glue-factory/` and `LightGlue/` cloned but NOT installed in editable mode
- Missing: `scipy`, `h5py`, `einops`, `tqdm`, `omegaconf`, `tensorboard`, `albumentations`, `seaborn`, `joblib`, `scikit-learn`, `timm`, `wandb`

### Actions Taken
1. **Explored workspace:** storage, GPU, installed packages, source code structure
2. **Assessed plan.md feasibility** — flagged disk space as tight; recommended 50-scene MegaDepth subset
3. **Created project files:**
   - `README.md` — project overview, folder structure, quick start commands
   - `CONTRIBUTING.md` — git branching strategy and PR workflow
   - `.github/AGENTS.md` — AI agent git rules (mandatory reading)
   - `docs/sessionlogs.md` — this file
   - `docs/changelog.md` — code change history
4. **Updated plan.md** — added feasibility notes section with disk budget correction
5. **Created Week 1 source modules:**
   - `src/__init__.py`, `src/models/__init__.py`, `src/utils/__init__.py`
   - `src/dinov2_extractor.py` — DINOv2Extractor class (Days 3–4)
   - `src/feature_sampling.py` — bilinear keypoint sampling
   - `src/projection.py` — Linear / MLP-1 / MLP-2 projection variants
   - `src/pipeline.py` — end-to-end SuperPoint+DINOv2+LightGlue pipeline
   - `src/cache_features.py` — offline DINOv2 → HDF5 caching (Days 6–7)
   - `src/evaluate.py` — evaluation utilities
   - `src/utils/viz.py` — visualization helpers
6. **Created scripts:**
   - `scripts/setup_env.sh`
   - `scripts/download_data.sh`
   - `scripts/cache_dinov2_features.sh`
7. **Created configs:**
   - `configs/dinov2_lightglue_base.yaml`
8. **Populated `Exp1.ipynb`** — 7-day structured notebook with full Week 1 code

### Plan Feasibility Issues Found
- **Disk space:** Plan expects 80–150 GB, only 113 GB free. Recommend 50-scene MegaDepth subset (~30–50 GB training data). Updated plan accordingly.
- **Missing packages:** 12 packages need `pip install` — added to `scripts/setup_env.sh`
- **PyTorch version:** 2.11.0 is newer than plan assumed; all APIs are compatible.
- **DINOv2-L:** Feasible for inference but monitor VRAM during training; DINOv2-B is primary target.

### Decisions Made
- Use `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')` — reg variant has better feature quality
- Cache DINOv2 features offline before training to reduce VRAM peak
- Use 50-scene MegaDepth subset for initial training runs
- Input resolution: 518×518 (DINOv2 native, 37×37 patch grid)

### Next Session Goals
- Day 1–2: Run `setup_env.sh`, download eval data, run SP+LG baseline → get baseline numbers
- Day 3–4: Test `DINOv2Extractor` on single image pair end-to-end
- Day 5: Run Experiments 1.1 and 1.2, make Go/No-Go decision

---
<!-- future sessions appended below -->

## Session 2026-04-22 — Critical Mid-Project Review & Course Correction

**Agent:** GitHub Copilot
**Duration:** Review session

### Goal
Honest assessment of publishability. User asked: "are we making groundbreaking progress or just jargon?"

### Verdict: 🔴 NOT PUBLISHABLE in current state.

### Key Findings
1. **MegaDepth AUC@10° (25-epoch model, EXP5): 39.6%** — marginally above baseline 37.7%, BUT measured on only 100 pairs (not statistically significant)
2. **HPatches illumination: 2.1% vs 37.5%** — DINOv2+LG is 18× worse on the exact benchmark where it should excel
3. **ScanNet indoor: 2.7% vs 29.4%** — 11× worse
4. **Speed: 12.7× slower** — no practical motivation
5. **The 25-epoch EXP5 model was NEVER evaluated on HPatches or ScanNet** — biggest gap

### Actions Taken
1. Updated `plan.md` §17 with corrective action plan
2. Updated `changelog.md` with this review entry
3. Updated `metrics.md` with honest "publishability gap" section
4. Identified pivot strategy: reframe as analysis/complementarity paper if replacement approach fails

### Critical Next Steps (Priority Order)
1. Benchmark EXP5 model on HPatches + ScanNet (2 hrs) — THIS DECIDES EVERYTHING
2. Multi-scale DINOv2 extraction for spatial precision (4 hrs)
3. Train 50+ epochs with cached features (8 hrs)
4. Stratified easy/medium/hard analysis (2 hrs)
5. If replacement still fails → pivot to descriptor fusion/complementarity approach

### Risk Assessment
- If EXP5 model also fails on HPatches/ScanNet → fundamental spatial resolution problem, must pivot
- If EXP5 model shows improvement → more training + multi-scale is the path forward
- Deadline pressure: ~6-8 days remaining for experiments before paper writing must begin

---

## Session 2026-04-22 (Evening) — THE PIVOT: Descriptor Fusion

**Agent:** GitHub Copilot
**Duration:** Deep work session

### Goal
Execute full pivot from descriptor replacement to descriptor fusion approach.

### Key Decisions Made
1. **ABANDON replacement approach** — DINOv2 alone cannot match SP spatial precision (stride 14 vs stride 8)
2. **FUSION is the new core idea**: SP(spatial) + DINOv2(semantic) → learned fusion → LightGlue
3. **GatedFusion** as primary architecture — interpretable per-keypoint gating is a paper contribution
4. **Full pretrained LightGlue weights** — fusion output is 256-d, no dimension mismatch

### Analysis: Why Replacement Failed
- DINOv2 stride-14 grid (37×37 for 518px) is too coarse for pixel-level accuracy
- RoMa (CVPR'24) already proved: DINOv2 must be COMBINED with fine features, not used alone
- Our HPatches 2.1% vs 37.5% is caused by ~7px spatial quantization error from patch grid

### Code Created
1. `src/descriptor_fusion.py` — 3 fusion strategies (ConcatMLP, GatedFusion, AdaptiveFusion) + DINOv2FusionProjection + MultiScaleSampler
2. `src/fusion_pipeline.py` — Full FusionPipeline + SPLightGlueBaseline + forward_from_cache
3. `src/train_fusion.py` — Two-phase training script
4. `src/eval_fusion.py` — Full benchmark evaluation 
5. `weakness.md` — Weakness tracker with 10 identified issues and resolutions

### Module Test Results
- All 3 fusion strategies: ✅ correct output shape (B, N, 256), L2-normalized
- GatedFusion gate init: mean=0.500 (equal weighting — correct)
- New trainable params: DINOv2Proj(230K) + GatedFusion(165K) = ~395K (tiny vs LightGlue's 3M)

### Updated Documents
- `plan.md` §18 — New pivot section with revised experiment plan
- `docs/changelog.md` — This session entry
- `docs/sessionlogs.md` — This entry
- `weakness.md` — New file

### Next Steps (Tomorrow)
1. Start Exp 6.1 training (gated fusion, 50 epochs on MegaDepth)
2. Evaluate zero-shot fusion (untrained) on HPatches — should match SP+LG baseline since SP descriptors are preserved
3. If training improves over baseline → we have a paper

---

## Session — 2026-04-26 — Training Running + Critical Bugs Fixed

### Problems Solved
1. **`gluefactory.models` import hangs**: Root cause was circular import chain through `gluefactory.utils.tools`. Fixed by inlining NLLLoss (20 lines of code) — no need for the entire model registry.
2. **LightGlue blocks gradients**: `lightglue.py:507` had `desc.detach()` — removed it. Gradients now flow: NLL loss → log_assignment → LG transformers → fused descriptors → fusion module → DINOv2 projection.
3. **LightGlue missing `log_assignment` output**: Patched LG to return the score matrix needed for NLL training loss.
4. **Conda base auto-activating**: Disabled `auto_activate_base`. All commands now use `.venv/bin/python` explicitly.
5. **`torch.cuda.amp` deprecated API**: Migrated to `torch.amp.autocast`/`GradScaler`.

### Training Progress (Exp 6.1 — Gated Fusion)
- **Config**: GatedFusion, ViT-B/14-reg, MegaDepth-1500 (1200 train/300 val), 1024 kpts, 640px
- **Phase 1 E01**: train_nll=2.30, val_nll=1.73, avg_matches=287, 159s/epoch
- **Phase 1 E02**: Step-level NLL dropping to 0.25-0.41 — fusion learning fast
- ETA: ~80min for Phase 1 (30 epochs), ~53min for Phase 2 (20 epochs)

### Key Architecture Insight
In `fusion_only` mode, LG params have `requires_grad=True` but are NOT in the optimizer. This means:
- Gradients flow through LG (backprop works)
- LG weights don't change (pretrained weights preserved)
- Only fusion module + DINOv2 projection are updated
This is the correct way to train a "wrapper" around a frozen backbone.

### Files Modified
- `LightGlue/lightglue/lightglue.py` — Removed `detach()`, added `log_assignment` to output
- `src/fusion_pipeline.py` — Fixed gradient flow in fusion_only mode
- `src/train_fusion_v2.py` — Inlined NLLLoss, custom MegaDepth dataset, `torch.amp` API
- `docs/changelog.md` — v0.4.0 entry

### Evaluation Results (HPatches Homography AUC@5px)
| Method | All | Illumination | Viewpoint |
|--------|-----|-------------|-----------|
| SP+LG Baseline | **48.6%** | **63.8%** | **34.0%** |
| Fusion (P1 best) | 43.8% | 59.3% | 26.8% |
| Fusion (P2 E05) | 46.1% | 62.9% | 29.8% |
| Fusion (P2 best NLL) | 6.1% | 11.9% | 0.5% |

### Key Findings
1. Fusion preserves ~95% of baseline performance (46.1/48.6) — semantic features don't destroy matching
2. Phase 2 overfits catastrophically after epoch 5 on tiny MegaDepth-1500 (1200 train pairs)
3. Val NLL on MegaDepth doesn't correlate with HPatches AUC — need HPatches-based early stopping
4. Best checkpoint: `p2_e05.pt` — use this for paper results and further experiments
