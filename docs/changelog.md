# Changelog

> All notable code changes to this project. Format: `## [Date] - <description>`.
> Maintained by both human contributors and AI agents. **Append only.**

---

## [2026-04-20] — Project Bootstrap (Session 1)

**Author:** GitHub Copilot · **Branch:** `exp/week1-poc`

### Added
- `README.md` — Full project README with folder structure, quick start, hardware specs, disk budget
- `CONTRIBUTING.md` — Git workflow: branching strategy, commit conventions, PR process, code review checklist
- `.github/AGENTS.md` — AI agent git rules: permitted operations, mandatory pre-commit checks, file ownership matrix
- `docs/sessionlogs.md` — Session log file (this bootstrap entry)
- `docs/changelog.md` — This file
- `src/__init__.py` — Package init
- `src/models/__init__.py` — Models subpackage init
- `src/utils/__init__.py` — Utils subpackage init
- `src/utils/viz.py` — Visualization utilities: `draw_keypoints()`, `draw_matches()`, `plot_descriptor_tsne()`
- `src/dinov2_extractor.py` — `DINOv2Extractor` class: loads DINOv2 (S/B/L variants), extracts patch feature maps, exposes `forward()` and `extract_features_batch()`
- `src/feature_sampling.py` — `sample_descriptors_bilinear()`: bilinear interpolation of DINOv2 patch features at arbitrary keypoint locations; handles normalization and edge cases
- `src/projection.py` — Three projection variants: `LinearProjection`, `MLP1Projection`, `MLP2Projection`; factory function `build_projection()`
- `src/pipeline.py` — `DINOv2LightGluePipeline`: end-to-end SuperPoint(kpts only) + DINOv2(frozen) + Projection + LightGlue(trainable)
- `src/cache_features.py` — `DINOv2FeatureCache`: offline extraction of DINOv2 features for all training images → saves to HDF5; resume-capable
- `src/evaluate.py` — `evaluate_matching_quality()`: MegaDepth-1500 / HPatches / ScanNet-1500 evaluation wrappers; AUC computation helpers
- `configs/dinov2_lightglue_base.yaml` — Full training configuration for DINOv2-B + MLP-2 + LightGlue fine-tune
- `scripts/setup_env.sh` — One-shot environment setup: install all missing packages, editable installs of glue-factory and LightGlue
- `scripts/download_data.sh` — Download and verify all evaluation datasets via glue-factory
- `scripts/cache_dinov2_features.sh` — Launch offline DINOv2 feature caching job
- `Exp1.ipynb` — Week 1 experiment notebook (7 days, fully populated)

### Modified
- `plan.md` — Added §17 "Feasibility Assessment" with disk budget correction, package install notes, DINOv2-L caution
- `requirement.txt` — Added all missing dependencies

### Architecture Decisions Recorded
- DINOv2 variant: `dinov2_vitb14_reg` (register variant, better downstream features)
- Input resolution: 518×518 → 37×37 patch grid → 768-dim per patch
- Projection: MLP-2 (`768→512 LayerNorm GELU →256 L2Norm`) as primary; Linear and MLP-1 as ablations
- Training: freeze DINOv2, train projection + all LightGlue transformer layers
- LightGlue init: from SuperPoint pretrained weights (skip `input_proj` layer due to dim mismatch)
- Caching: HDF5 keyed by image path hash; allows incremental runs

---
<!-- future entries appended below -->

## [2026-04-22] — Critical Mid-Project Review & Course Correction

**Author:** GitHub Copilot · **Context:** User-requested honest assessment

### Critical Assessment
- Reviewed ALL experiment metrics. Project is **NOT PUBLISHABLE** in current state.
- HPatches illumination: DINOv2+LG scores 2.1% vs SP+LG 37.5% (18× worse)
- ScanNet indoor: 2.7% vs 29.4% (11× worse)
- Only MegaDepth shows marginal promise (39.6% vs 37.7%), but on 100-pair subset only
- 25-epoch EXP5 model was never benchmarked on HPatches or ScanNet

### Modified
- `plan.md` — Added §17 "Critical Mid-Project Review" with corrective action plan, pivot strategy, and revised paper angle
- `docs/sessionlogs.md` — Added this review session with priority action items
- `docs/changelog.md` — This entry

### Decisions
- Priority 1: Benchmark EXP5 on all eval sets before any other work
- Pivot option: reframe as analysis paper ("when do semantic features help?") if replacement approach continues to fail
- Multi-scale DINOv2 extraction needed to fix spatial precision problem

## [2026-04-22 Evening] — THE PIVOT: Descriptor Fusion Architecture

**Author:** GitHub Copilot · **Context:** Fundamental approach change based on weakness analysis

### Critical Decision
- **ABANDONED** the descriptor replacement approach (DINOv2 instead of SP)
- **PIVOTED** to descriptor fusion (SP + DINOv2 combined)
- Root cause of failure: stride-14 spatial resolution loss kills all pixel-level benchmarks

### Added
- `src/descriptor_fusion.py` — ConcatFusionMLP, GatedFusion, AdaptiveFusion, DINOv2FusionProjection, MultiScaleDINOv2Sampler
- `src/fusion_pipeline.py` — FusionPipeline (end-to-end), SPLightGlueBaseline, forward_from_cache training support
- `src/train_fusion.py` — Two-phase training: fusion-only (30ep) → end-to-end (20ep)
- `src/eval_fusion.py` — Full benchmark evaluation on HPatches, ScanNet, MegaDepth
- `weakness.md` — 10 identified weaknesses with resolution actions and priority queue

### Modified
- `plan.md` — Added §18 "Critical Pivot" with new architecture, experiment plan, and timeline
- `docs/sessionlogs.md` — Pivot session log
- `docs/changelog.md` — This entry

### Architecture Change
```
OLD: Image → SP(kpts only) + DINOv2(desc) → Proj → LightGlue [FAILED: 2.1% HPatches]
NEW: Image → SP(kpts+desc) + DINOv2(semantic) → GatedFusion → LightGlue [FULL SP weights]
```

### Key Insight
The fusion output is 256-dim, matching SuperPoint exactly. This means we can load FULL pretrained SP-LightGlue weights with ZERO dimension mismatch — the only new parameters are the fusion module (~395K params). This is a massive advantage over the replacement approach.

## [0.4.0] — 2026-04-26 — Training Pipeline Fixed & Running

### Fixed
- **LightGlue `detach()` bug**: Removed `desc.detach()` in `LightGlue/lightglue/lightglue.py:507-508` that was blocking gradient flow from NLL loss back to fusion module
- **LightGlue `log_assignment` output**: Patched LightGlue to return `log_assignment` matrix (needed for NLL training loss) — was only used internally before
- **`gluefactory.models` import hang**: Bypassed entirely by inlining `NLLLoss` and `weight_loss` into `train_fusion_v2.py`
- **`torch.cuda.amp.autocast` API**: Migrated to `torch.amp.autocast`/`GradScaler` (new PyTorch API)
- **Gradient flow in fusion_only mode**: LG params keep `requires_grad=True` for gradient passthrough but are excluded from optimizer
- **Conda/venv conflict**: Disabled conda `auto_activate_base`, all scripts use `.venv` Python 3.11 exclusively

### Added
- Custom `MegaDepth1500Dataset`: Reads `pairs_calibrated.txt` directly (img paths, K0, K1, R, t), loads images + h5 depth maps, scales intrinsics
- Inlined `NLLMatchingLoss` with `weight_loss` — identical to glue-factory's but without the import chain that hangs
- `compute_gt()` function: Builds Camera/Pose wrappers and calls `gt_matches_from_pose_depth`

### Training Status (Exp 6.1)
- **Phase 1 Epoch 1**: train_nll=2.30, val_nll=1.73, avg_matches=287, 159s/epoch
- **Phase 1 Epoch 2**: Loss dropping rapidly (step-level: 0.25-0.41)
- Configuration: GatedFusion, ViT-B/14-reg, 1024 max keypoints, 640px, grad_accum=4
- Data: 1200 train / 300 val pairs from MegaDepth-1500

## [0.3.0] — 2026-04-25 — Fusion Architecture & Pivot

### Changed
- **Fusion module**: Replaced ConcatFusionMLP with GatedFusion (better performance)
- **Training strategy**: Two-phase training (fusion-only + end-to-end) for stability and performance
- **Evaluation**: Added full benchmark evaluation on HPatches, ScanNet, MegaDepth

### Fixed
- **LightGlue integration**: Resolved issues with LightGlue parameters not updating during training
- **Gradient flow**: Ensured proper gradient flow through the fusion module and LightGlue
