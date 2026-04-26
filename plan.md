# 🔬 Research Master Plan: DINOv2 Foundation Features for Sparse Feature Matching with LightGlue

> **Project Title:** "Foundation Model Descriptors Meet Sparse Matching: Bridging DINOv2 and LightGlue for Robust Feature Correspondence"
>
> **Timeline:** 25–30 Days | **Hardware:** NVIDIA RTX 3060 (12 GB VRAM) | **Target:** Workshop / WACV / BMVC 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Questions & Hypotheses](#2-research-questions--hypotheses)
3. [Background & Motivation](#3-background--motivation)
4. [Technical Architecture](#4-technical-architecture)
5. [Baseline Systems & Benchmark Numbers](#5-baseline-systems--benchmark-numbers)
6. [Dataset Plan](#6-dataset-plan)
7. [Experiment Plan](#7-experiment-plan)
8. [Day-by-Day Schedule](#8-day-by-day-schedule)
9. [VRAM & Compute Budget](#9-vram--compute-budget)
10. [Codebase Architecture](#10-codebase-architecture)
11. [Evaluation Protocol](#11-evaluation-protocol)
12. [Paper Outline](#12-paper-outline)
13. [Risk Register & Mitigations](#13-risk-register--mitigations)
14. [Go/No-Go Checkpoints](#14-gono-go-checkpoints)
15. [Tools & Dependencies](#15-tools--dependencies)
16. [References](#16-references)

---

## 1. Executive Summary

### Core Idea

Replace SuperPoint's learned 256-dimensional local descriptors with features extracted from DINOv2 (a self-supervised Vision Transformer foundation model) at SuperPoint keypoint locations, and fine-tune LightGlue's transformer matcher to work with these foundation-model descriptors. The hypothesis is that DINOv2's rich, semantically-aware features — trained on 142M images via self-supervised learning — will produce more robust correspondences than task-specific local descriptors, especially under challenging conditions (large viewpoint changes, illumination shifts, cross-domain scenarios).

### Why This Matters

- **RoMa (CVPR 2024)** proved that frozen DINOv2 features dramatically improve *dense* matching
- **MASt3R (ECCV 2024)** builds 3D matching on DINOv2/ViT-L backbones
- **Nobody has done this for sparse matching with LightGlue** — this is the gap
- Sparse matching remains critical for real-time SLAM, SfM, and visual localization where dense methods are too slow

### Acceptance Chances

| Venue | Estimated Probability |
|---|---|
| CVPR/ECCV Workshop (e.g., IMCW, LSCV) | **~75%** |
| WACV / BMVC Main Conference | **~50-60%** |
| CVPR / ECCV Main Conference | **~15-20%** |

---

## 2. Research Questions & Hypotheses

### Primary Research Questions

**RQ1:** Can DINOv2 patch features, sampled at sparse keypoint locations, serve as effective replacements for SuperPoint descriptors in the LightGlue matching pipeline?

**RQ2:** Does fine-tuning LightGlue with DINOv2 descriptors improve matching robustness under challenging conditions (large viewpoint changes, illumination shifts, day-night) compared to SuperPoint+LightGlue?

**RQ3:** What is the optimal integration strategy — frozen DINOv2 with projection, or light fine-tuning of DINOv2's last layers?

### Hypotheses

- **H1:** DINOv2-B (768-dim) features projected to 256-dim will match or exceed SuperPoint descriptors on MegaDepth-1500 pose AUC, because DINOv2 encodes richer semantic and geometric information.
- **H2:** The improvement will be most pronounced on cross-domain benchmarks (Aachen Day-Night, HPatches illumination), where semantic understanding matters most.
- **H3:** A lightweight projection MLP (768→256) with frozen DINOv2 will outperform a simple linear projection, capturing non-linear descriptor adaptation.
- **H4:** Even DINOv2-S (384-dim, 21M params) will be competitive with SuperPoint (256-dim), providing a low-compute alternative.

---

## 3. Background & Motivation

### LightGlue Architecture (ICCV 2023)

LightGlue is a sparse feature matcher that takes keypoints + descriptors from two images and outputs correspondences.

**Key architecture details (from source code):**
```
Default config:
  input_dim: 256           # descriptor dimension (per-feature-type, auto-set)
  descriptor_dim: 256      # internal transformer dimension
  n_layers: 9              # transformer layers (self-attn + cross-attn each)
  num_heads: 4             # attention heads
  flash: True              # FlashAttention support
  depth_confidence: 0.95   # early stopping threshold
  width_confidence: 0.99   # point pruning threshold
  filter_threshold: 0.1    # match score threshold
```

**Supported feature types and their dimensions:**
| Feature | input_dim | add_scale_ori | Weights |
|---|---|---|---|
| SuperPoint | 256 | False | `superpoint_lightglue` |
| DISK | 128 | False | `disk_lightglue` |
| ALIKED | 128 | False | `aliked_lightglue` |
| SIFT | 128 | True | `sift_lightglue` |
| DoGHardNet | 128 | True | `doghardnet_lightglue` |

**Critical code path (lightglue.py:393-398):**
```python
if conf.input_dim != conf.descriptor_dim:
    self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
else:
    self.input_proj = nn.Identity()
```
→ LightGlue **already supports arbitrary input dimensions** via a learned linear projection. This is our entry point.

**Descriptor assertion (lightglue.py:512-513):**
```python
assert desc0.shape[-1] == self.conf.input_dim
assert desc1.shape[-1] == self.conf.input_dim
```

### DINOv2 Foundation Model (Meta, 2023)

| Variant | Params | Feature Dim | Patch Size | VRAM (inference) |
|---|---|---|---|---|
| ViT-S/14 | 21M | 384 | 14×14 | ~0.3 GB |
| ViT-B/14 | 86M | 768 | 14×14 | ~1.2 GB |
| ViT-L/14 | 300M | 1024 | 14×14 | ~4.5 GB |
| ViT-g/14 | 1.1B | 1536 | 14×14 | ~16 GB ❌ |

**Our primary model: DINOv2-B/14** (768-dim, 86M params, ~1.2 GB VRAM for inference)
**Comparison model: DINOv2-S/14** (384-dim, 21M params, ~0.3 GB VRAM)

**Loading:**
```python
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# or with registers: 'dinov2_vitb14_reg'
```

**Feature extraction:** For an image resized to 518×518 → produces 37×37 = 1,369 patch tokens of dim 768.

### Key Prior Work

| Paper | Year | Venue | Relevance |
|---|---|---|---|
| RoMa | 2024 | CVPR | Frozen DINOv2 for *dense* matching (coarse features) + ConvNet fine features |
| MASt3R | 2024 | ECCV | 3D matching on DINOv2/ViT-L encoder, massive multi-dataset training |
| DUSt3R | 2024 | CVPR | 3D reconstruction from 2 views, DINOv2-based encoder |
| LightGlue-ONNX | 2026 | GitHub | FP8 PTQ for LightGlue (quantization angle covered) |
| PTQ4ViT | 2022 | NeurIPS | Post-training quantization for ViTs |

**Gap:** RoMa uses DINOv2 for *dense* matching. Nobody has studied DINOv2 features for *sparse* keypoint matching with LightGlue's efficient adaptive architecture.

---

## 4. Technical Architecture

### Pipeline Overview

```
Image → SuperPoint (keypoint detection only) → Keypoint locations (x, y)
                                                      ↓
Image → DINOv2-B/14 (frozen) → Patch feature map (37×37×768)
                                                      ↓
                              Bilinear interpolation at keypoint locations
                                                      ↓
                              768-dim descriptors per keypoint
                                                      ↓
                              Projection MLP: 768 → 256
                                                      ↓
                              LightGlue matcher (fine-tuned) → Matches
```

### Design Decisions

**D1: Keypoint Detector = SuperPoint (frozen)**
- We keep SuperPoint's keypoint detector (not descriptors) because it's proven reliable
- This lets us do apples-to-apples comparison: same keypoints, different descriptors
- SuperPoint outputs: `keypoints [B×N×2]`, `keypoint_scores [B×N]`, `descriptors [B×N×256]` — we only use the first two

**D2: DINOv2 Feature Sampling via Bilinear Interpolation**
- DINOv2 outputs patch-level features at stride 14 (one token per 14×14 pixel region)
- We bilinearly interpolate these features at sub-patch keypoint locations
- This is the same strategy used by SuperPoint itself (`sample_descriptors` with `grid_sample`)

**D3: Projection Network Variants (ablation study)**

| Variant | Architecture | Trainable Params |
|---|---|---|
| **Linear** | `Linear(768, 256)` | 196,864 |
| **MLP-1** | `Linear(768, 512) → ReLU → Linear(512, 256)` | ~524K |
| **MLP-2** | `Linear(768, 512) → LayerNorm → GELU → Linear(512, 256) → L2Norm` | ~525K |
| **Identity** (if using DINOv2-S) | `Linear(384, 256)` | 98,560 |

**D4: LightGlue Configuration for DINOv2**
```python
# New feature entry to add:
features["dinov2"] = {
    "weights": None,  # train from scratch (or fine-tune from SP weights)
    "input_dim": 256,  # after projection
}
# Or if feeding raw DINOv2:
features["dinov2_raw"] = {
    "weights": None,
    "input_dim": 768,  # let LightGlue's input_proj handle it
}
```

**D5: Training Strategy — Two Options**

| Strategy | Description | Pros | Cons |
|---|---|---|---|
| **A: External projection + LG fine-tune** | Train projection MLP externally, set `input_dim=256`, fine-tune all of LightGlue | Clean separation, can init LG from SP weights | May lose some info in projection |
| **B: Let LG project internally** | Set `input_dim=768`, LG's built-in `input_proj` (Linear 768→256) handles it, train end-to-end | Simpler, LG optimizes projection jointly | Can't init from SP pretrained weights (dim mismatch in input_proj) |

**→ We go with Strategy A as primary, Strategy B as ablation.**

For Strategy A, we can initialize LightGlue from the pretrained SuperPoint weights (all transformer layers are dim-256 regardless of input), only the `input_proj` differs.

---

## 5. Baseline Systems & Benchmark Numbers

### MegaDepth-1500 Pose Estimation (AUC@5°/10°/20°, PoseLib estimator)

| Method | AUC@5° | AUC@10° | AUC@20° |
|---|---|---|---|
| SuperPoint + SuperGlue | - | - | - |
| **SuperPoint + LightGlue** | **66.8** | **79.3** | **87.9** |
| SIFT (4K) + LightGlue | 65.9 | 78.6 | 87.4 |
| ALIKED + LightGlue | 66.4 | 79.0 | 87.5 |
| SuperPoint + GlueStick | 64.4 | 77.5 | 86.5 |

### MegaDepth-1500 (OpenCV RANSAC estimator)

| Method | AUC@5° | AUC@10° | AUC@20° |
|---|---|---|---|
| **SuperPoint + LightGlue** | **51.0** | **68.1** | **80.7** |
| SIFT (4K) + LightGlue | 49.9 | 67.3 | 80.3 |

### HPatches Homography Estimation (AUC@1/3/5 px, PoseLib)

| Method | @1px | @3px | @5px |
|---|---|---|---|
| **SuperPoint + LightGlue** | **37.1** | **67.4** | **77.8** |
| SuperPoint + SuperGlue | 37.0 | 68.2 | 78.7 |

### ScanNet-1500 Indoor Pose (AUC@5°/10°/20°, PoseLib)

| Method | AUC@5° | AUC@10° | AUC@20° |
|---|---|---|---|
| **SuperPoint + LightGlue** | **21.9** | **39.8** | **55.7** |
| SuperPoint + SuperGlue | 22.7 | 39.5 | 54.3 |
| DISK + LightGlue | 12.1 | 23.1 | 35.0 |

**Our target: Match or exceed SuperPoint+LightGlue on MegaDepth-1500 and HPatches, with significant improvement on cross-domain/challenging conditions.**

---

## 6. Dataset Plan

### Evaluation Datasets (Must Download)

| Dataset | Size | Use | Auto-download |
|---|---|---|---|
| **MegaDepth-1500** | ~1.5 GB | Primary pose evaluation (outdoor) | ✅ via glue-factory |
| **ScanNet-1500** | ~1.1 GB | Indoor pose evaluation | ✅ via glue-factory |
| **HPatches** | ~1.8 GB | Homography estimation | ✅ via glue-factory |

**Total eval data: ~4.4 GB**

### Training Datasets

| Dataset | Full Size | Our Subset | Use |
|---|---|---|---|
| **MegaDepth (full)** | ~420 GB | ~420 GB (auto-managed by glue-factory) | Fine-tuning |
| **Oxford-Paris 1M** | ~450 GB | ❌ **Skip this** | Homography pre-training |

**CRITICAL DECISION: Skip homography pre-training.**

Rationale:
1. 450 GB download is impractical for our timeline and disk space
2. DINOv2 features are already trained on diverse data — they don't need the synthetic augmentation that task-specific SuperPoint descriptors needed
3. We can initialize LightGlue's transformer layers from the SuperPoint pretrained weights (they're dimension-agnostic at 256-dim internal)
4. Go straight to MegaDepth fine-tuning

**For MegaDepth training with cached features:**
- glue-factory supports `data.load_features.do=True` to cache extracted features
- We'll cache DINOv2 features offline → stored as HDF5, ~150 GB for full MegaDepth
- **Alternative:** Use a subset of MegaDepth scenes (~50-100 scenes, ~30-50 GB)

### Disk Space Budget

| Item | Size |
|---|---|
| Evaluation datasets | ~4.4 GB |
| MegaDepth training (auto-downloaded, partial) | ~50-100 GB |
| Cached DINOv2 features | ~20-40 GB (for subset) |
| Model checkpoints | ~2 GB |
| Code, logs, results | ~5 GB |
| **Total needed** | **~80-150 GB** |

---

## 7. Experiment Plan

### Phase 1: Proof of Concept (Days 1-5)

#### Experiment 1.1: Zero-Shot DINOv2 Descriptor Quality
**Goal:** Measure raw DINOv2 descriptor quality without any training.

**Method:**
1. Extract SuperPoint keypoints from MegaDepth-1500 image pairs
2. Sample DINOv2-B features at keypoint locations via bilinear interpolation
3. Apply simple linear projection (768→256, random init or PCA)
4. Run nearest-neighbor matching (no LightGlue) using cosine similarity
5. Compute match precision and recall vs. ground-truth correspondences

**What we learn:** Are DINOv2 features inherently more discriminative than SuperPoint descriptors at sparse locations? This tells us if the whole project is viable.

**Go/No-Go Gate:** If DINOv2-NN matching precision is < 50% of SuperPoint-NN matching, reconsider approach.

#### Experiment 1.2: Pretrained LightGlue with DINOv2 (No Training)
**Goal:** See what happens when we naively feed DINOv2 descriptors to a SuperPoint-pretrained LightGlue.

**Method:**
1. Use SuperPoint-LightGlue pretrained weights
2. Replace SuperPoint descriptors with projected DINOv2 descriptors (768→256 via learned linear)
3. Run evaluation on MegaDepth-1500

**Expected result:** Degraded performance (LightGlue hasn't adapted), but if it still works somewhat, the transformer is robust to descriptor distribution shifts.

### Phase 2: Core Training (Days 6-15)

#### Experiment 2.1: Fine-tune LightGlue with DINOv2 (Strategy A)
**Goal:** Train the full pipeline: DINOv2(frozen) → ProjectionMLP → LightGlue(fine-tune)

**Configuration:**
```yaml
model:
  extractor:
    name: extractors.superpoint_open   # keypoint detection only
    max_num_keypoints: 1024
  matcher:
    name: matchers.lightglue
    input_dim: 256                      # after our projection
    descriptor_dim: 256
    n_layers: 9
    depth_confidence: -1                # disable for training
    width_confidence: -1                # disable for training

data:
  name: megadepth
  batch_size: 1                         # RTX 3060 constraint
  preprocessing:
    resize: 1024
    side: long

train:
  epochs: 40
  lr: 1e-4
  optimizer: adam
  lr_schedule:
    type: exp
    exp_div_10: 10
  eval_every_iter: 1000
  save_every_iter: 5000
  clip_grad: 10
```

**Training strategy:**
- **Freeze:** DINOv2-B backbone (86M params)
- **Train:** Projection MLP (~525K params) + LightGlue transformer (all layers, ~4M params)
- **Initialize:** LightGlue transformer layers from SuperPoint-LightGlue pretrained weights, only skip `input_proj`
- **Batch size:** 1 (RTX 3060), use gradient accumulation of 8 to simulate batch_size=8
- **Mixed precision:** FP16 with autocast for memory savings

#### Experiment 2.2: Projection Network Ablation
**Goal:** Find optimal projection architecture.

Run 3 variants, each for 20 epochs:
1. **Linear:** `Linear(768, 256)` — simplest baseline
2. **MLP-1:** `Linear(768, 512) → ReLU → Linear(512, 256)`
3. **MLP-2:** `Linear(768, 512) → LayerNorm → GELU → Linear(512, 256) → L2Norm`

Evaluate on MegaDepth-1500 after each. Pick the best for remaining experiments.

#### Experiment 2.3: DINOv2 Variant Comparison
**Goal:** Compare DINOv2 model sizes.

| Config | DINOv2 | Projection | Expected VRAM |
|---|---|---|---|
| Small | ViT-S/14 (384-dim) | 384→256 | ~4 GB total |
| **Base** | **ViT-B/14 (768-dim)** | **768→256** | **~7 GB total** |
| Large | ViT-L/14 (1024-dim) | 1024→256 | ~10 GB (tight!) |

Run each for 20 epochs on MegaDepth, evaluate.

#### Experiment 2.4: LightGlue Init Strategy Ablation
**Goal:** Does initializing LightGlue from SuperPoint weights help?

| Config | Init |
|---|---|
| A | SuperPoint-LightGlue pretrained (skip input_proj layer) |
| B | Random initialization |

### Phase 3: Extended Evaluation (Days 16-20)

#### Experiment 3.1: Full Benchmark Suite
Run the best model from Phase 2 on all benchmarks:
- MegaDepth-1500 (outdoor pose, AUC@5/10/20)
- ScanNet-1500 (indoor pose, AUC@5/10/20)
- HPatches (homography, AUC@1/3/5px)

#### Experiment 3.2: Robustness Analysis
**Goal:** Where does DINOv2 shine over SuperPoint?

Analysis dimensions:
1. **Overlap binning:** Split MegaDepth pairs by visual overlap (low/medium/high) → plot AUC per bin
2. **Illumination:** HPatches illumination-only subset vs. viewpoint-only subset
3. **Match count analysis:** Do we get more/fewer matches? Higher precision?
4. **Adaptive behavior:** How does LightGlue's early stopping behave with DINOv2 descriptors?
   - Plot average number of layers used per pair
   - Compare depth_confidence curves

#### Experiment 3.3: Feature Visualization & Analysis
- t-SNE visualization of DINOv2 vs. SuperPoint descriptors colored by semantic category
- Attention map visualization from LightGlue's cross-attention layers
- Descriptor similarity heatmaps for matched/unmatched points

#### Experiment 3.4: Aachen Day-Night (if time permits)
**Goal:** Cross-domain robustness on visual localization

- Requires HLoc pipeline integration
- Only attempt if main results are strong
- This would be the strongest evidence for the paper's claims

### Phase 4: Paper Writing (Days 21-28)

#### Experiment 4.1: Final Ablation Table
Consolidate all ablation results into clean tables:
- Projection type ablation
- DINOv2 size ablation  
- Init strategy ablation
- Max keypoints ablation (512 vs. 1024 vs. 2048)

#### Experiment 4.2: Inference Speed Benchmark
- Measure FPS for the full pipeline (SuperPoint + DINOv2 + Projection + LightGlue)
- Compare against SuperPoint + LightGlue baseline
- Report on various keypoint counts

### Phase 5: Buffer & Submission (Days 29-30)

- Final paper revision
- Supplementary material (visualizations, additional tables)
- Code cleanup for potential release

---

## 8. Day-by-Day Schedule

### Week 1: Setup & Proof of Concept (Days 1-7)

| Day | Session | Tasks | Deliverables |
|---|---|---|---|
| **1** | Environment Setup | Install PyTorch, glue-factory, DINOv2. Verify GPU. Clone repos. | Working env, all imports pass |
| **2** | Data Pipeline | Download MegaDepth-1500, HPatches, ScanNet-1500 eval sets. Run SP+LG baseline eval. | Baseline numbers reproduced |
| **3** | DINOv2 Integration | Write DINOv2 feature extractor module. Test on single image pair. Verify dimensions. | `DINOv2Extractor` class working |
| **4** | Feature Sampling | Implement bilinear sampling of DINOv2 features at SP keypoint locations. Debug dimensions. | Feature extraction pipeline end-to-end |
| **5** | Exp 1.1 + 1.2 | Run zero-shot descriptor quality test. Run pretrained-LG-with-DINOv2 test. | Go/No-Go decision data |
| **6** | Training Pipeline | Write custom training script or adapt glue-factory config. Implement DINOv2 feature caching. | Training loop runs without crash |
| **7** | Buffer / Debug | Fix any issues from days 1-6. Start DINOv2 feature caching for MegaDepth training subset. | Cache generation started |

### Week 2: Core Training (Days 8-14)

| Day | Session | Tasks | Deliverables |
|---|---|---|---|
| **8** | Exp 2.1 Start | Launch main training: DINOv2-B + MLP-2 projection + LightGlue fine-tune. | Training running, loss decreasing |
| **9** | Monitor + Exp 2.2 | Monitor Exp 2.1 training curves. Start projection ablation (Linear variant). | Training logs, ablation started |
| **10** | Exp 2.2 Cont. | Run MLP-1 projection variant. Evaluate first checkpoints from Exp 2.1. | Early evaluation numbers |
| **11** | Exp 2.3 | Run DINOv2-S variant training. Compare with DINOv2-B early results. | Size comparison data |
| **12** | Exp 2.4 | Init strategy ablation (pretrained vs. random). | Init comparison data |
| **13** | Evaluation | Run best model on MegaDepth-1500. Compare with baselines. | Core result table |
| **14** | Analysis | Compile all Phase 2 results. Pick best configuration. | Decision on final model |

### Week 3: Deep Evaluation & Writing (Days 15-21)

| Day | Session | Tasks | Deliverables |
|---|---|---|---|
| **15** | Exp 3.1 | Full benchmark: MegaDepth-1500 + ScanNet-1500 + HPatches | Complete benchmark table |
| **16** | Exp 3.2 | Robustness analysis: overlap binning, illumination split | Robustness plots |
| **17** | Exp 3.3 | Feature visualization: t-SNE, attention maps, similarity heatmaps | Visualization figures |
| **18** | Speed Benchmark | Inference speed measurement across keypoint counts | Speed table |
| **19** | Paper: Intro + Related | Write Introduction and Related Work sections | Draft sections |
| **20** | Paper: Method | Write Method section with architecture figures | Draft section + figures |
| **21** | Paper: Experiments | Write Experiments section, compile all tables and plots | Draft section |

### Week 4: Paper Polish & Submit (Days 22-28+)

| Day | Session | Tasks | Deliverables |
|---|---|---|---|
| **22** | Paper: Results + Discussion | Write remaining sections, abstract, conclusion | Full draft v1 |
| **23** | Revision | Self-review, fix inconsistencies, improve writing | Draft v2 |
| **24** | Figures & Tables | Polish all figures, ensure consistent formatting | Final figures |
| **25** | Supplementary | Additional experiments, failure case analysis | Supplementary material |
| **26** | Final Revision | Address any weak points, strengthen claims | Draft v3 |
| **27** | Proofread | Grammar, references, formatting check | Final paper |
| **28** | Submit | Camera-ready preparation, code cleanup | Submission |
| **29-30** | Buffer | Emergency fixes, rebuttal prep notes | |

---

## 9. VRAM & Compute Budget

### Memory Analysis for RTX 3060 (12 GB)

#### Training VRAM Breakdown

| Component | VRAM (FP16) | Notes |
|---|---|---|
| DINOv2-B inference (frozen) | ~1.5 GB | Forward pass only, no gradients |
| DINOv2-B model weights | ~0.35 GB | FP16 |
| SuperPoint inference (frozen) | ~0.1 GB | Keypoint detection only |
| Projection MLP | ~0.01 GB | Tiny |
| LightGlue (model + gradients + optimizer) | ~3.0 GB | 4M params, Adam optimizer states |
| Input images + feature maps | ~0.5 GB | 1024×1024 images |
| Attention matrices (9 layers) | ~1.5 GB | For 1024 keypoints |
| Gradient accumulation buffer | ~0.5 GB | 8-step accumulation |
| PyTorch overhead | ~1.5 GB | CUDA context, memory fragmentation |
| **Total** | **~9.0 GB** | **Fits in 12 GB ✅** |

#### Offline Feature Caching Strategy (Recommended)

To avoid loading DINOv2 during training:
1. **Phase A (offline):** Run DINOv2-B on all training images → save 768-dim features per keypoint to HDF5
2. **Phase B (training):** Load cached features → Projection MLP → LightGlue → Loss

This reduces training VRAM to:
| Component | VRAM |
|---|---|
| Cached features (loaded from disk) | ~0.01 GB |
| Projection MLP + LightGlue (train) | ~3.5 GB |
| Everything else | ~2.0 GB |
| **Total with caching** | **~5.5 GB** ✅ |

#### Max Keypoints Constraint

| Max Keypoints | Attention VRAM | Total VRAM | Feasible? |
|---|---|---|---|
| 512 | ~0.4 GB | ~7 GB | ✅ Very comfortable |
| **1024** | **~1.5 GB** | **~9 GB** | **✅ Primary setting** |
| 2048 | ~5.5 GB | ~13 GB | ❌ Too tight |

**→ Use max_num_keypoints=1024 for training, 2048 for evaluation (inference is cheaper).**

### Compute Time Estimates

| Task | Time (RTX 3060) |
|---|---|
| DINOv2 feature caching (MegaDepth subset, 50 scenes) | ~6-8 hours |
| LightGlue fine-tuning (40 epochs, 1024 kpts, BS=1, grad_accum=8) | ~24-48 hours |
| Single evaluation run (MegaDepth-1500) | ~20-30 minutes |
| Full benchmark suite (MD + SN + HP) | ~1-2 hours |
| Projection ablation (3 variants × 20 epochs) | ~36 hours total |
| DINOv2 size ablation (3 sizes × 20 epochs) | ~36 hours total |

**Total GPU hours: ~150-200 hours over 30 days (~5-7 hrs/day average)**

---

## 10. Codebase Architecture

### Repository Structure

```
lightglue/
├── plan.md                          # This document
├── alternative_ideas.md             # Other explored ideas
├── start.txt                        # Original QAT idea
│
├── src/                             # Our research code
│   ├── __init__.py
│   ├── dinov2_extractor.py          # DINOv2 feature extraction module
│   ├── feature_sampler.py           # Bilinear sampling at keypoint locations
│   ├── projection.py                # Projection MLP variants
│   ├── dinov2_lightglue.py          # Combined pipeline (SP-kpts + DINOv2-desc + LG)
│   ├── cache_features.py            # Offline DINOv2 feature caching script
│   ├── train.py                     # Custom training script (or glue-factory adapter)
│   ├── evaluate.py                  # Evaluation runner
│   ├── visualize.py                 # t-SNE, attention maps, match visualizations
│   └── configs/
│       ├── dinov2b_lightglue_megadepth.yaml
│       ├── dinov2s_lightglue_megadepth.yaml
│       └── ablation_configs/
│           ├── proj_linear.yaml
│           ├── proj_mlp1.yaml
│           └── proj_mlp2.yaml
│
├── experiments/                     # Experiment logs and results
│   ├── exp1.1_zero_shot/
│   ├── exp1.2_pretrained_lg/
│   ├── exp2.1_main_training/
│   ├── exp2.2_projection_ablation/
│   ├── exp2.3_dinov2_size/
│   ├── exp2.4_init_strategy/
│   └── exp3_benchmarks/
│
├── paper/                           # LaTeX paper
│   ├── main.tex
│   ├── figures/
│   └── tables/
│
└── scripts/                         # Utility scripts
    ├── setup_env.sh
    ├── download_data.sh
    └── run_all_benchmarks.sh
```

### Key Module: `dinov2_extractor.py`

```python
class DINOv2Extractor:
    """Extract DINOv2 features at sparse keypoint locations."""
    
    def __init__(self, model_name='dinov2_vitb14', device='cuda'):
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval().to(device)
        self.patch_size = 14
        self.feat_dim = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024
        }[model_name]
    
    @torch.no_grad()
    def extract_features(self, image, keypoints):
        """
        Args:
            image: [B, 3, H, W] normalized to [0, 1]
            keypoints: [B, N, 2] in pixel coordinates
        Returns:
            descriptors: [B, N, feat_dim]
        """
        # Resize to make dimensions divisible by patch_size
        # Get patch tokens from DINOv2
        # Reshape to spatial grid
        # Bilinear interpolate at keypoint locations
        # Return per-keypoint descriptors
```

### Key Module: `projection.py`

```python
class ProjectionMLP(nn.Module):
    """Project DINOv2 features to LightGlue descriptor space."""
    
    def __init__(self, input_dim=768, output_dim=256, variant='mlp2'):
        # variant in ['linear', 'mlp1', 'mlp2']
        ...
    
    def forward(self, x):
        # x: [B, N, input_dim] → [B, N, output_dim]
        ...
```

---

## 11. Evaluation Protocol

### Primary Metrics

#### Pose Estimation (MegaDepth-1500, ScanNet-1500)
- **AUC@5°, AUC@10°, AUC@20°** of the relative pose error
- Estimator: PoseLib (RANSAC, threshold=1.0 for MD, threshold=12.0 for SN)
- Also report with OpenCV RANSAC for comparison

#### Homography Estimation (HPatches)
- **H_error AUC@1px, @3px, @5px**
- Both DLT and RANSAC estimators

#### Matching Quality
- **Precision@k**: fraction of predicted matches that are correct at various thresholds
- **Number of matches**: average inlier count after RANSAC
- **Inlier ratio**: matches / keypoints

### Secondary Metrics

- **Inference speed** (FPS) at various keypoint counts
- **Number of LightGlue layers used** (depth_confidence analysis)
- **Match count** distribution

### Evaluation Commands (glue-factory)

```bash
# MegaDepth-1500
python -m gluefactory.eval.megadepth1500 --conf <our_config> eval.estimator=poselib eval.ransac_th=1.0

# ScanNet-1500
python -m gluefactory.eval.scannet1500 --conf <our_config> eval.estimator=poselib eval.ransac_th=12.0

# HPatches
python -m gluefactory.eval.hpatches --conf <our_config> --overwrite
```

### Statistical Significance
- Report mean and standard deviation over 3 random seeds for training
- Use the same evaluation pairs as all baselines (standard splits)

---

## 12. Paper Outline

### Title Options
1. "Foundation Model Descriptors Meet Sparse Matching: Bridging DINOv2 and LightGlue for Robust Feature Correspondence"
2. "Beyond Task-Specific Descriptors: DINOv2 Features for Efficient Sparse Feature Matching"
3. "DINOGlue: Leveraging Self-Supervised Foundation Features for Adaptive Sparse Matching"

### Structure (8 pages + references)

**Abstract** (150 words)
- Problem: Task-specific local descriptors (SuperPoint, DISK) limit matching robustness
- Insight: Foundation model features (DINOv2) encode richer visual priors
- Method: Replace SuperPoint descriptors with DINOv2 features, fine-tune LightGlue
- Results: Improved robustness under challenging conditions, competitive/superior on standard benchmarks

**1. Introduction** (1 page)
- Local feature matching is fundamental to 3D vision
- Current sparse matchers rely on task-specific descriptors with limited training data
- Foundation models (DINOv2) trained on 142M images offer richer representations
- Dense matchers (RoMa) already benefit from DINOv2, but sparse matching hasn't been explored
- We propose using DINOv2 features as drop-in descriptors for LightGlue
- Contributions:
  1. First study of foundation model features for sparse keypoint matching
  2. Lightweight integration via projection MLP (zero overhead at descriptor extraction)
  3. Systematic analysis of where foundation features improve over task-specific ones

**2. Related Work** (1 page)
- 2.1 Local Feature Matching (SuperGlue, LightGlue, LoFTR, MatchFormer)
- 2.2 Foundation Models in Vision (DINO, DINOv2, MAE, CLIP)
- 2.3 Foundation Features for Correspondence (RoMa, MASt3R, DUSt3R)
- 2.4 Descriptor Design (SuperPoint, DISK, ALIKED, R2D2)

**3. Method** (1.5 pages)
- 3.1 Pipeline Overview (figure)
- 3.2 Foundation Feature Extraction (DINOv2 at keypoint locations)
- 3.3 Descriptor Projection (MLP variants)
- 3.4 Matcher Adaptation (LightGlue fine-tuning strategy)
- 3.5 Training Details

**4. Experiments** (3 pages)
- 4.1 Experimental Setup (datasets, metrics, baselines, implementation details)
- 4.2 Main Results (Table: MD-1500, SN-1500, HPatches)
- 4.3 Ablation Studies
  - 4.3.1 Projection Architecture
  - 4.3.2 DINOv2 Model Size
  - 4.3.3 Initialization Strategy
- 4.4 Robustness Analysis
  - Per-overlap-bin analysis
  - Illumination vs. viewpoint (HPatches)
- 4.5 Qualitative Results (match visualizations, failure cases)
- 4.6 Inference Speed

**5. Discussion & Limitations** (0.5 pages)
- When do foundation features help vs. hurt?
- Computational overhead of DINOv2 backbone
- Limitations: requires DINOv2 inference, larger model footprint

**6. Conclusion** (0.5 pages)

**Supplementary**
- Additional benchmark results
- More visualizations
- Hyperparameter sensitivity

### Key Figures

1. **Fig 1:** Pipeline overview diagram (teaser figure)
2. **Fig 2:** Qualitative matching comparison (SP+LG vs DINOv2+LG on hard cases)
3. **Fig 3:** Per-overlap-bin AUC plot
4. **Fig 4:** t-SNE of descriptor spaces
5. **Fig 5:** Attention visualization from LightGlue layers

### Key Tables

1. **Table 1:** Main results (3 benchmarks × multiple methods)
2. **Table 2:** Projection ablation
3. **Table 3:** DINOv2 size ablation
4. **Table 4:** Inference speed comparison
5. **Table 5:** Init strategy ablation

---

## 13. Risk Register & Mitigations

| # | Risk | Impact | Probability | Mitigation |
|---|---|---|---|---|
| R1 | DINOv2 features at sparse locations lose spatial precision (14px patch stride is too coarse) | High | Medium | Multi-scale feature extraction; bilinear interpolation; use features from intermediate layers as well |
| R2 | LightGlue fine-tuning doesn't converge with new descriptors | High | Low | Init from SP pretrained weights; lower learning rate; try Strategy B (internal projection) |
| R3 | No improvement over SuperPoint+LightGlue baseline | High | Medium | Focus on challenging conditions where DINOv2 should shine; even neutral results + analysis = publishable at workshop |
| R4 | VRAM overflow during training | Medium | Low | Feature caching eliminates DINOv2 from GPU during training; reduce keypoints to 512; use gradient checkpointing |
| R5 | MegaDepth download/processing takes too long | Medium | Medium | Use subset of scenes; start download Day 1; have fallback to HPatches-only evaluation |
| R6 | DINOv2 inference too slow for practical use | Low | Medium | Report speed honestly; argue offline extraction can be cached; DINOv2-S as lightweight alternative |
| R7 | Reviewers say "obvious extension, limited novelty" | Medium | Medium | Emphasize systematic analysis + insights; show surprising results; careful framing as first study |
| R8 | Training instability with batch_size=1 | Medium | Medium | Gradient accumulation; careful learning rate tuning; gradient clipping |

### Contingency Plans

**If DINOv2 features don't help on standard benchmarks:**
→ Pivot to "analysis paper" angle: "When do foundation features help for sparse matching?"
→ Focus on cross-domain results where improvement is more likely
→ Still publishable at workshops

**If training is too slow:**
→ Use DINOv2-S instead of DINOv2-B (3× faster)
→ Train for fewer epochs (20 instead of 40)
→ Reduce MegaDepth subset

**If VRAM issues:**
→ Feature caching (removes DINOv2 from GPU during training)
→ Reduce max_keypoints to 512
→ Use gradient checkpointing (`checkpointed: True` in LightGlue config)

---

## 14. Go/No-Go Checkpoints

### Checkpoint 1: Day 5 (After Exp 1.1 + 1.2)

**GO if ANY of:**
- [ ] DINOv2 NN-matching precision ≥ 70% of SuperPoint NN-matching
- [ ] Pretrained LG with DINOv2 achieves >30 AUC@10° on MegaDepth (even without fine-tuning)
- [ ] Visual inspection shows DINOv2 matches are qualitatively sensible

**NO-GO if ALL of:**
- [ ] DINOv2 NN-matching is catastrophically bad (<20% of SP)
- [ ] Pretrained LG completely fails (<5 AUC@10°)
- [ ] Feature visualization shows no spatial discrimination
→ **Pivot to Idea 1 (Adaptive Matcher Fusion) or Idea 3 (Synthetic Domain Bridge)**

### Checkpoint 2: Day 14 (After Phase 2)

**GO if:**
- [ ] Best DINOv2+LG model achieves ≥ 95% of SP+LG on MegaDepth AUC@10°
- [ ] OR shows >2% improvement on any benchmark
- [ ] Training loss converged properly

**CONDITIONAL GO if:**
- [ ] Results are within 90-95% of baseline
→ Focus on robustness analysis (where DINOv2 wins) for paper angle

**NO-GO if:**
- [ ] Best model < 85% of baseline on all benchmarks after 40 epochs
→ Write up negative results as workshop paper (still publishable)

### Checkpoint 3: Day 20 (After Phase 3)

**GO for full paper if:**
- [ ] At least one benchmark shows clear improvement over SP+LG
- [ ] Ablations show clear trends
- [ ] Sufficient material for 8-page paper

---

## 15. Tools & Dependencies

### Python Environment

```
# Core
python >= 3.10
torch >= 2.0 (for FlashAttention)
torchvision
numpy
scipy
matplotlib
opencv-python
h5py
omegaconf
tensorboard

# Specific
glue-factory (git clone from cvg/glue-factory)
lightglue (git clone from cvg/LightGlue)
poselib (for pose estimation)
kornia (used by DISK, general CV ops)

# DINOv2
# Loaded via torch.hub, no separate install needed

# Paper
latex (texlive or overleaf)
```

### Setup Commands

```bash
# Create conda environment
conda create -n dinoglue python=3.10
conda activate dinoglue

# Install PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Clone repositories
git clone https://github.com/cvg/LightGlue.git
git clone https://github.com/cvg/glue-factory.git

# Install glue-factory
cd glue-factory
pip install -e .
pip install poselib

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0)); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')"

# Test DINOv2 loading
python -c "import torch; m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14'); print(f'DINOv2-B loaded: {sum(p.numel() for p in m.parameters())/1e6:.1f}M params')"
```

---

## 16. References

1. Lindenberger, P., Sarlin, P.-E., Pollefeys, M. (2023). "LightGlue: Local Feature Matching at Light Speed." ICCV 2023.
2. Oquab, M., et al. (2024). "DINOv2: Learning Robust Visual Features without Supervision." TMLR 2024.
3. Edstedt, J., et al. (2024). "RoMa: Robust Dense Feature Matching." CVPR 2024.
4. Leroy, V., et al. (2024). "Grounding Image Matching in 3D with MASt3R." ECCV 2024.
5. Wang, Z., et al. (2024). "DUSt3R: Geometric 3D Vision Made Easy." CVPR 2024.
6. DeTone, D., Malisiewicz, T., Rabinovich, A. (2018). "SuperPoint: Self-Supervised Interest Point Detection and Description." CVPRW 2018.
7. Tyszkiewicz, M., Fua, P., Trulls, E. (2020). "DISK: Learning Local Features with Policy Gradient." NeurIPS 2020.
8. Sarlin, P.-E., DeTone, D., Malisiewicz, T., Rabinovich, A. (2020). "SuperGlue: Learning Feature Matching with Graph Neural Networks." CVPR 2020.
9. Zhao, X., et al. (2022). "ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation." IEEE TIM 2023.
10. fabio-sim. (2026). "LightGlue-ONNX: ONNX/TensorRT/OpenVINO export with FP8 quantization." GitHub.

---

## Appendix A: Detailed LightGlue Architecture Reference

```
LightGlue default_conf:
  name: "lightglue"
  input_dim: 256
  descriptor_dim: 256
  n_layers: 9
  num_heads: 4
  flash: True (FlashAttention)
  mp: False (mixed precision)
  depth_confidence: 0.95
  width_confidence: 0.99
  filter_threshold: 0.1

Feature configs (lightglue.py):
  superpoint: {weights: "superpoint_lightglue", input_dim: 256}
  disk:       {weights: "disk_lightglue",       input_dim: 128}
  aliked:     {weights: "aliked_lightglue",     input_dim: 128}
  sift:       {weights: "sift_lightglue",       input_dim: 128, add_scale_ori: True}
  doghardnet: {weights: "doghardnet_lightglue", input_dim: 128, add_scale_ori: True}

Key flow:
  descriptors [B, N, input_dim]
    → input_proj (Linear if input_dim != 256, else Identity)
    → [B, N, 256]
    → 9× TransformerLayer (self_attn + cross_attn, 4 heads, dim=256)
    → MatchAssignment → scores → matches
```

## Appendix B: glue-factory Training Reference

```
Default training config (train.py):
  epochs: 1
  optimizer: adam
  lr: 0.001
  lr_schedule.type: None
  eval_every_iter: 1000
  save_every_iter: 5000
  log_every_iter: 200
  clip_grad: None
  best_key: "loss/total"

Training LightGlue:
  Stage 1: Homography pre-train (Oxford-Paris 1M, ~450GB) - WE SKIP THIS
  Stage 2: MegaDepth fine-tune (~420GB)
    - batch_size=32 default (for 2x3090), we use 1 + grad_accum=8
    - Supports cached features via: data.load_features.do=True
    - Cache requires ~150GB for full MegaDepth

Evaluation:
  python -m gluefactory.eval.megadepth1500 --conf <config>
  python -m gluefactory.eval.scannet1500 --conf <config>
  python -m gluefactory.eval.hpatches --conf <config>

Official LightGlue training notes:
  - Default batch_size=128 requires 2x 3090 GPUs (24GB each)
  - For 1x 1080: batch_size=32
  - For RTX 3060: batch_size=1 + gradient_accumulation=8
```

---

## 17. CRITICAL MID-PROJECT REVIEW (April 22, 2026)

### 🔴 HONEST STATUS: NOT PUBLISHABLE YET

#### What the numbers actually say

| Benchmark | SP+LG (baseline) | DINOv2+LG (5ep) | DINOv2+LG (25ep, EXP5) | Gap |
|---|---|---|---|---|
| MegaDepth AUC@10° (200 pairs) | **37.7%** | 28.2% | 39.6%* (100 pairs) | *Unverified on full set* |
| HPatches illum AUC@5px | **37.5%** | 2.1% | ❌ Not evaluated | **18× worse** |
| HPatches all AUC@5px | **18.4%** | 1.0% | ❌ Not evaluated | **18× worse** |
| ScanNet AUC@10° | **29.4%** | 2.7% | ❌ Not evaluated | **11× worse** |
| Inference speed | 11.7ms | — | 148.4ms | **12.7× slower** |

\* EXP5's 39.6% AUC@10° is on a 100-pair subset, not the standard 200-pair benchmark. This number is statistically unreliable.

#### Why this is not publishable

1. **The core hypothesis is failing**: DINOv2 was supposed to help on illumination/cross-domain. HPatches illumination shows it's **18× worse**, not better.
2. **ScanNet (indoor) is catastrophic**: 2.7% vs 29.4%. No reviewer accepts this.
3. **Only MegaDepth shows promise** — and only with 25 epochs of training on MegaDepth data. This could be overfitting to MegaDepth distribution.
4. **The 25-epoch model has never been evaluated on HPatches or ScanNet** — this is the single most critical gap.
5. **12.7× slower with no clear accuracy win** — no practical motivation.

#### Root causes

- **Undertrained**: 5 epochs on 150 pairs is absurdly little. The 25-epoch EXP5 model exists but was never fully benchmarked.
- **Resolution mismatch**: DINOv2 patch grid is 37×37 at 518px. SuperPoint operates at arbitrary resolution. Bilinear sampling from a coarse grid loses spatial precision — this explains HPatches failure (pixel-level accuracy matters).
- **No hard-case analysis**: We never separated easy vs hard pairs to see where DINOv2 actually helps.

### 🟡 CORRECTIVE ACTION PLAN

#### MUST DO (Days 22–25) — Make or Break

| # | Action | Why | Time |
|---|---|---|---|
| 1 | **Full benchmark of EXP5 (25-epoch) model on HPatches + ScanNet** | We literally don't know if more training fixes the gap | 2 hours |
| 2 | **Multi-scale DINOv2 features** (extract at 518px AND 336px, concat) | Coarse patch grid is killing spatial precision | 4 hours |
| 3 | **Train for 50+ epochs on larger MegaDepth subset** (use cached features) | 25 epochs is still not converged (loss still dropping) | 8 hours |
| 4 | **Stratified evaluation**: easy/medium/hard overlap bins per benchmark | Find the niche where DINOv2 wins — that's the paper story | 2 hours |

#### SHOULD DO (Days 25–28) — Strengthen the narrative

| # | Action | Why |
|---|---|---|
| 5 | **Feature caching for all eval sets** (HPatches, ScanNet) — not just MegaDepth | Speed up eval loop |
| 6 | **DINOv2 + SP descriptor fusion** (concat or weighted sum) | If replacement fails, complementarity might work |
| 7 | **Cross-dataset generalization test**: train on MegaDepth, test on ScanNet WITHOUT fine-tuning | Tests if DINOv2's semantic features generalize better |
| 8 | **Day-Night evaluation** (Aachen or equivalent) | This is the strongest case for semantic features |

#### PIVOT OPTION (if Action 1 shows 25-epoch model still fails on HPatches/ScanNet)

**Change the paper story from "replacement" to "complementarity":**
- SP descriptors for spatial precision + DINOv2 for semantic context
- Fuse both descriptor types before LightGlue
- This is a more defensible and novel contribution

### Revised Paper Angle

**Current angle (weak):** "DINOv2 can replace SP descriptors" — data doesn't support this.

**Better angle:** "When does semantic feature matching beat task-specific descriptors? An empirical study of DINOv2+LightGlue with analysis of failure modes and complementarity."

This reframes the work as an **analysis paper** (always publishable at workshops) rather than a **systems paper** (needs clear SOTA).

---

## 18. CRITICAL PIVOT — Session April 22, 2026 (Evening)

### Status: EXECUTING THE PIVOT

**Decision: Full pivot from "replacement" to "descriptor fusion".**

The replacement approach (DINOv2 instead of SP) has fundamentally failed due to stride-14 spatial resolution loss. We now pursue **learned descriptor fusion** — combining SP's spatial precision with DINOv2's semantic robustness.

### New Architecture

```
Image ─┬── SuperPoint ──── kpts, scores, SP_desc (256-d)     [SPATIAL PRECISION]
       └── DINOv2 (frozen) → sample at SP kpts → DINOv2Proj (128-d) [SEMANTIC CONTEXT]
                                          ↓
                         GatedFusion(SP_desc, DINOv2_proj) → fused (256-d)
                                          ↓
                         LightGlue (FULL pretrained SP weights) → matches
```

### Why This Fixes Everything

| Old Problem | How Fusion Fixes It |
|---|---|
| HPatches 2.1% (lost SP precision) | SP descriptors preserved; DINOv2 adds context |
| ScanNet 2.7% (lost spatial info) | SP spatial precision kept intact |
| 12.7× slower (no benefit) | Cost justified by accuracy GAIN on hard pairs |
| LightGlue dim mismatch | Output is 256-d, FULL pretrained weights used |
| "Obvious extension" novelty concern | Gated fusion with interpretable per-keypoint gate is novel |

### New Paper Title
**"Semantic-Geometric Descriptor Fusion: Enhancing Sparse Feature Matching with Foundation Model Features"**

### New Research Questions
- **RQ1:** Does fusing DINOv2 semantic features with SuperPoint local descriptors improve sparse matching on challenging benchmarks?
- **RQ2:** Can a learned gating mechanism identify when semantic vs local features should be prioritized?
- **RQ3:** What is the computational trade-off of adding DINOv2 to the sparse matching pipeline?

### New Experiment Plan (Exp6-Exp9)

| Exp | Description | Status |
|---|---|---|
| **Exp 6.1** | Fusion pipeline with GatedFusion, Phase 1 (30 ep fusion-only) + Phase 2 (20 ep e2e) | 🔧 Code ready |
| **Exp 6.2** | Fusion strategy ablation: concat_mlp vs gated vs adaptive | 🔧 Code ready |
| **Exp 6.3** | DINOv2 projection dim ablation: 64 vs 128 vs 256 | 🔧 Planned |
| **Exp 7.1** | Full HPatches eval (illum + viewpoint) | 🔧 Code ready |
| **Exp 7.2** | Full ScanNet-1500 eval | 🔧 Planned |
| **Exp 7.3** | Full MegaDepth-1500 eval | 🔧 Planned |
| **Exp 8.1** | Gate visualization: when does DINOv2 help? | 🔧 Planned |
| **Exp 8.2** | Multi-scale DINOv2 + fusion | 🔧 Code ready |
| **Exp 9.1** | Speed benchmark: SP+LG vs Fusion vs DINOv2-S fusion | 🔧 Planned |

### New Files Created

| File | Purpose |
|---|---|
| `src/descriptor_fusion.py` | ConcatFusionMLP, GatedFusion, AdaptiveFusion, DINOv2FusionProjection, MultiScaleDINOv2Sampler |
| `src/fusion_pipeline.py` | FusionPipeline (full pipeline), SPLightGlueBaseline, forward_from_cache |
| `src/train_fusion.py` | Two-phase training: fusion-only → end-to-end |
| `src/eval_fusion.py` | Full benchmark evaluation on all datasets |
| `weakness.md` | Weakness tracker with resolution actions |

### Timeline (Remaining ~6 days)

| Day | Action | Deliverable |
|---|---|---|
| Day 22 (today) | ✅ Build fusion code, test modules | Code ready |
| Day 23 | Train Exp 6.1 (gated fusion, 50 epochs) | Trained model |
| Day 24 | Full eval on HPatches + ScanNet + MegaDepth | Benchmark numbers |
| Day 25 | Ablations (Exp 6.2, 6.3), gate visualization | Ablation table |
| Day 26 | Multi-scale + speed benchmark | Complete results |
| Day 27-28 | Paper writing | 8-page draft |
