# Weakness Analysis & Resolution Tracker

> **Purpose:** Honest record of every weakness, its root cause, and planned fix.
> **Rule:** Every weakness MUST have a resolution action. No logging without fixing.

---

## 🔴 CRITICAL WEAKNESSES (Paper-blocking)

### W1: Catastrophic Spatial Precision Loss
- **Evidence:** HPatches illum AUC@5px: 2.1% (ours) vs 37.5% (SP+LG) — 18× worse
- **Root Cause:** DINOv2 stride=14 produces 37×37 feature grid for 518×518 image. Bilinear sampling at this coarse resolution loses sub-pixel precision. SuperPoint uses stride=8 with dense descriptor maps.
- **Impact:** Kills all pixel-level benchmarks (HPatches homography, ScanNet indoor)
- **Resolution:** ✅ **PIVOT to descriptor fusion** — keep SP descriptors for spatial precision, add DINOv2 for semantic robustness. Also implement multi-scale DINOv2 extraction.
- **Status:** 🔧 IN PROGRESS

### W2: Pure Replacement Approach is Fundamentally Wrong
- **Evidence:** ScanNet AUC@10°: 2.7% vs 29.4%. MegaDepth marginal gain (39.6% vs 37.7% on 100 pairs, not significant).
- **Root Cause:** Replacing proven local descriptors with coarse semantic features loses the spatial precision that makes sparse matching work. This is exactly why RoMa (CVPR 2024) uses DINOv2 as COARSE features combined with fine CNN features.
- **Impact:** No benchmark shows convincing improvement → unpublishable
- **Resolution:** ✅ **Fusion architecture** — concatenate SP (256-dim) + projected DINOv2 (128-dim) → learned fusion to 256-dim for LightGlue input.
- **Status:** 🔧 IN PROGRESS

### W3: Insufficient Training (3-25 epochs)
- **Evidence:** Loss still decreasing at epoch 25 (0.81). Ablations only 3 epochs each.
- **Root Cause:** Compute time constraints, lack of feature caching initially
- **Impact:** Models not converged, unfair comparisons
- **Resolution:** Use cached features + fusion pipeline (much faster training). Target 100 epochs.
- **Status:** 🔧 PLANNED

### W4: Evaluation on Insufficient Samples
- **Evidence:** MegaDepth "best result" (39.6%) on only 100 pairs. Standard is 1500 pairs.
- **Root Cause:** Time pressure, wanted quick results
- **Impact:** Not statistically significant, reviewer will reject immediately
- **Resolution:** Run all evaluations on FULL standard splits (1500 pairs MegaDepth, all HPatches, 1500 ScanNet)
- **Status:** 🔧 PLANNED

### W5: 12.7× Speed Overhead with No Accuracy Gain
- **Evidence:** DINOv2 backbone = 43% of total inference time. Net result is slower AND less accurate.
- **Root Cause:** ViT-B forward pass is expensive. Used at full 518×518 resolution.
- **Impact:** No practical deployment motivation
- **Resolution:** (a) Fusion approach amortizes cost — DINOv2 adds semantic benefit on top of SP. (b) Use DINOv2-S (ViT-S, 5× faster) for speed-sensitive configs. (c) Feature caching for offline SfM/SLAM use case.
- **Status:** 🔧 PLANNED

---

## 🟡 MODERATE WEAKNESSES (Strengthening needed)

### W6: No Confidence Intervals / Multiple Runs
- **Evidence:** All results are single-run numbers
- **Resolution:** Run 3× with different seeds, report mean ± std
- **Status:** PLANNED

### W7: Missing Hard-Case Stratified Analysis
- **Evidence:** No breakdown by overlap ratio, illumination severity, etc.
- **Resolution:** Stratify MegaDepth by overlap bins, HPatches by difficulty level
- **Status:** PLANNED

### W8: No Aachen Day-Night Evaluation
- **Evidence:** Skipped "if time permits" benchmark
- **Resolution:** This is WHERE semantic features should shine. Must evaluate if fusion works.
- **Status:** PLANNED (after fusion shows promise on HPatches)

### W9: Novelty Perception — "Obvious Extension"
- **Evidence:** Reviewers may say "RoMa already uses DINOv2 for matching"
- **Resolution:** Pivot framing: First systematic study of semantic+local descriptor FUSION for SPARSE matching. Complementarity analysis (when do semantic features help?) is the contribution.
- **Status:** 🔧 IN PROGRESS (new paper angle)

---

## 🟢 RESOLVED WEAKNESSES

### W10: Zero-shot DINOv2 fails with pretrained LightGlue
- **Evidence:** 0 matches in Exp 1.2
- **Resolution:** Expected — different descriptor distributions. Training mandatory.
- **Status:** ✅ RESOLVED (training works, loss decreases)

---

## Resolution Priority Queue

| Priority | Weakness | Action | Est. Time | Impact |
|---|---|---|---|---|
| 1 | W2 | Build fusion pipeline (SP+DINOv2→256) | 3 hrs | **Game-changer** |
| 2 | W1 | Multi-scale DINOv2 extraction | 2 hrs | Spatial precision |
| 3 | W3 | Train fusion model 100 epochs | 8 hrs | Converged results |
| 4 | W4 | Full eval on all benchmarks | 4 hrs | Statistical validity |
| 5 | W5 | Speed benchmark with DINOv2-S fusion | 1 hr | Practical story |
| 6 | W9 | Reframe paper angle | 1 hr | Reviewer perception |
| 7 | W6 | Multi-seed runs | 6 hrs | Robustness |
| 8 | W7 | Stratified analysis | 2 hrs | Insight depth |

---

## Key Insight for the Pivot

**RoMa's recipe applied to sparse matching:**
- RoMa: DINOv2 (coarse semantic) + CNN (fine local) → dense matching
- **Ours: DINOv2 (semantic context) + SuperPoint (spatial precision) → sparse matching via LightGlue**
- This is genuinely novel — nobody has done learned fusion of foundation model features with local descriptors for sparse matching
- The paper becomes: "When and How Do Semantic Features Complement Local Descriptors for Sparse Matching?"

**Why fusion should work:**
1. SP descriptors are great at spatial precision but fail on illumination/day-night (trained on synthetic data)
2. DINOv2 features are robust to appearance changes but coarse spatially
3. Concatenation + learned fusion lets the matcher decide WHEN to rely on which signal
4. On easy pairs: SP dominates (no overhead from bad DINOv2 spatial info)
5. On hard pairs: DINOv2 semantic signal rescues matches that SP alone would miss
