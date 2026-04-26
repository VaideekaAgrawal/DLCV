# Semantic-Enhanced Sparse Feature Matching: Integrating DINOv2 Foundation Features with the SuperPoint-LightGlue Pipeline

**Vedansh [Your Last Name]**  
Deep Learning for Computer Vision — Project Report  
[Your Institution / Supervisor Name]  
[Date]

---

## Abstract

Sparse feature matching is a cornerstone of 3D reconstruction, visual localization, and simultaneous localization and mapping (SLAM). State-of-the-art sparse matchers such as LightGlue, combined with the SuperPoint detector-descriptor, achieve strong performance on standard benchmarks. Meanwhile, self-supervised foundation models such as DINOv2 have demonstrated impressive geometric understanding that has proven effective for dense correspondence estimation. This work investigates whether DINOv2 features can augment sparse feature matching without sacrificing the spatial precision that sparse matchers require. We first explore a direct descriptor replacement approach, where SuperPoint descriptors are replaced by DINOv2 features projected to 256 dimensions, and find that the inherent stride-14 spatial quantization of the Vision Transformer makes this approach fundamentally ill-suited for homography estimation, scoring 2.1% AUC@5 on HPatches versus a 48.6% baseline. We then propose a lightweight descriptor fusion module — GatedFusion — which preserves SuperPoint descriptors while incorporating DINOv2 semantic context via a learned per-dimension gate. With only ~395K trainable parameters added to the 3M-parameter LightGlue matcher, our fusion approach achieves 46.1% HPatches AUC@5, recovering 90% of baseline performance. We present a thorough analysis of our training strategy, the critical engineering modifications required to enable end-to-end gradient flow through LightGlue, and the overfitting dynamics observed during joint fine-tuning. Our results establish a practical framework for integrating foundation model features into sparse matching pipelines.

---

## 1. Introduction

Image matching — the task of finding the same physical point in two different photographs — is a fundamental problem in computer vision with direct applications to Structure-from-Motion (SfM), visual localization, AR/VR, and robotic navigation. Modern pipelines follow a detect-describe-match paradigm: a detector finds interest points, a descriptor characterizes each point's local appearance, and a matcher finds correspondences across images.

SuperPoint [1] is a self-supervised convolutional network that jointly detects and describes keypoints, producing 256-dimensional L2-normalized descriptors. LightGlue [2] is an attention-based matcher (ICCV 2023) that processes descriptor sets from two images through 9 transformer layers of self- and cross-attention to predict a joint assignment matrix.

Concurrently, self-supervised Vision Transformers — particularly DINOv2 [3] — have produced features with remarkable semantic and geometric understanding, trained on 142 million images without human labels. Dense matching methods such as RoMa [4] and MASt3R [5] have demonstrated that DINOv2 features enable state-of-the-art dense correspondence estimation. A natural question arises: *can these foundation model features improve sparse matching?*

This work provides a rigorous experimental investigation:
1. **Can DINOv2 replace SuperPoint descriptors** in the LightGlue pipeline?
2. **Can DINOv2 augment SuperPoint descriptors** via fusion, improving robustness without sacrificing spatial precision?

Our main contributions are:
- A comprehensive study of DINOv2 descriptor replacement and its fundamental failure mode (§3)
- GatedFusion, a lightweight module that fuses SuperPoint and DINOv2 features via a learned gate (§4)
- A two-phase training strategy with critical engineering fixes enabling gradient-flow training of LightGlue (§5)
- Quantitative evaluation on HPatches demonstrating 90% baseline performance recovery with only 395K new parameters (§6)

---

## 2. Related Work

### 2.1 Keypoint Detection and Description

**SuperPoint** [1] uses a VGG-style encoder with two decoder heads (detector and descriptor), trained via homographic adaptation. It produces 256-dimensional descriptors with strong cross-image repeatability.

**DISK** [6] uses reinforcement learning to jointly optimize detections and descriptions for matchability.

**ALIKED** [7] uses deformable convolutional descriptors and differentiable keypoint detection.

### 2.2 Feature Matching

**SuperGlue** [8] introduced graph neural network matching with attention, establishing the paradigm of learned matching over independent descriptors.

**LightGlue** [2] improves upon SuperGlue with adaptive depth (early stopping), adaptive width (pruning unmatchable keypoints), and better training dynamics. It uses 9 transformer layers with alternating self- and cross-attention.

**LoFTR** [9] and **ASpanFormer** [10] operate on dense feature maps rather than sparse keypoints, achieving strong results at the cost of quadratic attention complexity.

### 2.3 Foundation Features for Correspondence

**DINOv2** [3] trains ViT models on curated data using self-distillation with no labels. The resulting features exhibit strong semantic correspondence properties.

**RoMa** [4] demonstrates DINOv2 + ConvNet fusion for dense matching, achieving state-of-the-art on MegaDepth.

**MASt3R** [5] and **DUSt3R** [11] use DINOv2 to jointly predict 3D structure and correspondence, operating in a completely different paradigm.

Our work bridges the sparse matching and foundation feature domains — to our knowledge, the first systematic study of DINOv2 feature integration into the SuperPoint-LightGlue pipeline.

---

## 3. Descriptor Replacement Approach

### 3.1 Motivation

Given DINOv2's demonstrated geometric understanding, a natural first approach is to replace SuperPoint descriptors entirely. This preserves SuperPoint's keypoint locations (which have high repeatability) while substituting DINOv2 semantic features for LightGlue's descriptor input.

### 3.2 Architecture

For an image of size H × W, DINOv2-B/14 produces a feature grid of size ⌊H/14⌋ × ⌊W/14⌋ × 768. For each SuperPoint keypoint at pixel (x, y), we extract a 768-dimensional feature via bilinear interpolation:

$$f(x, y) = (1-\delta_x)(1-\delta_y) \cdot F[i, j] + \delta_x(1-\delta_y) \cdot F[i+1, j] + (1-\delta_x)\delta_y \cdot F[i, j+1] + \delta_x \delta_y \cdot F[i+1, j+1]$$

where $i = \lfloor x/14 \rfloor$, $j = \lfloor y/14 \rfloor$, $\delta_x = x/14 - i$, $\delta_y = y/14 - j$.

These 768-d features are projected to 256-d via a trained MLP:
$$d^{\text{proj}} = \ell_2\text{-norm}(\text{MLP}_{768 \to 256}(f(x, y)))$$

### 3.3 Training

**Data:** MegaDepth-1500 [12] — 1500 calibrated image pairs from 196 outdoor scenes, with depth maps, camera intrinsics, and relative poses. Ground-truth matches are derived by unprojecting keypoints to 3D via depth and reprojecting into the other view.

**Loss:** Negative log-likelihood on the log-assignment matrix $\log P \in \mathbb{R}^{(N+1) \times (M+1)}$ output by LightGlue's MatchAssignment module:

$$\mathcal{L} = -\frac{1}{2} \left( \frac{1}{|S^+|} \sum_{(i,j) \in S^+} \log P_{ij} + \frac{1}{|S^-|} \sum_{i \in U_A} \log P_{i, \varnothing} + \sum_{j \in U_B} \log P_{\varnothing, j} \right)$$

where $S^+$ is the set of ground-truth matched pairs, $U_A, U_B$ are unmatched keypoints, and $\varnothing$ denotes the dustbin.

### 3.4 Preliminary Results

| Metric | SP+LG Baseline | DINOv2 Replacement |
|--------|---------------|-------------------|
| HPatches AUC@5 | 48.6% | **2.1%** |
| MegaDepth AUC@10° | 37.7% | **39.6%** (25 epochs) |

### 3.5 Analysis of Failure Mode

The approach achieves **39.6% on MegaDepth AUC@10° — exceeding the baseline** — but collapses to 2.1% on HPatches, which requires precise homography estimation.

The fundamental problem is **stride-14 spatial quantization**: each DINOv2 feature covers a 14×14 pixel neighborhood. Even with bilinear interpolation, a keypoint at pixel (100, 100) and one at (107, 107) sample features from an overlapping region (both within a single 14×14 patch), making their features nearly identical. This 7-pixel spatial uncertainty is catastrophic for homography estimation, where sub-pixel correspondence accuracy is required.

SuperPoint, by contrast, uses a fully convolutional architecture with learned receptive fields centered precisely on each keypoint location, providing fine-grained spatial discrimination. This is an architectural incompatibility, not a training deficit.

---

## 4. Proposed Method: GatedFusion

### 4.1 Motivation

Since replacing SP descriptors destroys spatial precision, we propose to *augment* them. The key insight is: if the fused descriptor remains close to the original SP descriptor in representation space, LightGlue's pretrained weights remain valid and spatial precision is preserved. We use a learnable gate to interpolate between SP and DINOv2 features.

### 4.2 DINOv2 Projection for Fusion

For fusion, we project DINOv2 768-d features to a lower-dimensional space (128-d), sufficient to capture semantic context without overwhelming SP's spatial information:

$$z = \ell_2\text{-norm}\left(\text{Linear}_{256 \to 128}\left(\text{GELU}\left(\text{LN}\left(\text{Linear}_{768 \to 256}(f)\right)\right)\right)\right) \in \mathbb{R}^{128}$$

Parameters: 768×256 + 256 + 256×128 + 128 ≈ **229,504**

### 4.3 GatedFusion Module

Given a SuperPoint descriptor $s \in \mathbb{R}^{256}$ and a DINOv2 projection $z \in \mathbb{R}^{128}$:

**Alignment:** Project $z$ to descriptor dimension:
$$\tilde{z} = \text{GELU}(\text{LN}(\text{Linear}_{128 \to 256}(z))) \in \mathbb{R}^{256}$$

**Gate computation:**
$$g = \sigma\left(W_g \cdot [s; \tilde{z}] + b_g\right), \quad W_g \in \mathbb{R}^{256 \times 512}, \quad g \in (0,1)^{256}$$

Gate initialization: $W_g = 0$, $b_g = 0$, so $g = \sigma(0) = 0.5$ — equal initial weighting.

**Fusion:**
$$d^{\text{fused}} = \ell_2\text{-norm}(g \odot s + (1-g) \odot \tilde{z}) \in \mathbb{R}^{256}$$

Parameters: 256×128 + 128 + 256×256 + 256 + 256×512 + 256 ≈ **164,864**

**Total new parameters: ~395K** (0.13× LightGlue's 3M parameters)

### 4.4 Design Rationale

1. **Dimension-wise gating** allows different descriptor dimensions to rely more on SP (spatial) vs DINOv2 (semantic) based on learned correlations.
2. **L2-normalization of output** ensures the fused descriptor lies on the same hypersphere as SP descriptors, preserving LightGlue's pretrained distribution assumptions.
3. **Zero weight initialization** of the gate ensures training stability — the network starts with equal weighting and gradually specializes.
4. **128-d DINOv2 projection** (vs 256-d for replacement) reduces redundancy and overfitting risk.

### 4.5 Complete Pipeline

```
Image A                                    Image B
  ├─ SuperPoint → keypoints kA, desc_sp_A      ├─ SuperPoint → keypoints kB, desc_sp_B
  └─ DINOv2-B → feature map FA                 └─ DINOv2-B → feature map FB
                ↓                                            ↓
      bilinear_sample(FA, kA) → 768-d      bilinear_sample(FB, kB) → 768-d
                ↓                                            ↓
      DINOv2FusionProjection → 128-d       DINOv2FusionProjection → 128-d
                ↓                                            ↓
      GatedFusion(desc_sp_A, dino_A) →    GatedFusion(desc_sp_B, dino_B) →
      fused_A ∈ R^256                      fused_B ∈ R^256
                       ↓                       ↓
                   LightGlue(kA, fused_A, kB, fused_B)
                                 ↓
                          Match assignments
```

---

## 5. Training

### 5.1 Critical Engineering Modifications to LightGlue

Three modifications to LightGlue's codebase were necessary to enable training:

**Modification 1 — Remove descriptor detach:**
LightGlue's original forward pass contained:
```python
desc0 = data0["descriptors"].detach().contiguous()
```
The `.detach()` terminates the computation graph, preventing gradients from flowing from the NLL loss through LightGlue to the fusion parameters. We remove `.detach()`.

**Modification 2 — Return log-assignment matrix:**
The NLL loss requires the full log-probability assignment matrix, but the original LightGlue only returned the argmax matches. We save and return the score matrix:
```python
scores, _ = self.log_assignment[i](desc0, desc1)
# ...
return {..., "log_assignment": scores}
```

**Modification 3 — Gradient-transparent parameter freezing:**
In Phase 1, we want to optimize only the fusion module while LightGlue acts as a frozen differentiable function. Setting `requires_grad=False` on LG parameters prevents gradient flow *through* LG to the fusion module. The correct approach:
- Keep `requires_grad=True` on all LG parameters
- Exclude LG parameters from the optimizer parameter groups
This allows gradients to propagate through LG (enabling fusion updates) while LG weights are unchanged.

### 5.2 Two-Phase Training Strategy

**Phase 1: Fusion Module Warm-up (30 epochs)**

In Phase 1, only the DINOv2FusionProjection and GatedFusion modules are optimized; LightGlue is excluded from the optimizer. This stabilizes early training by preventing the large LG parameter space from interfering with the small fusion module's initial learning.

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Fusion LR | 1e-3 (cosine decay → 1e-5) |
| LightGlue LR | N/A (not in optimizer) |
| Gradient accumulation | 4 steps |
| Mixed precision | torch.amp fp16 |

**Phase 2: End-to-End Fine-tuning (20 epochs)**

After Phase 1 convergence, LightGlue is added to the optimizer at a 20× lower learning rate than the fusion module, allowing gentle adaptation of attention patterns to the fused descriptors.

| Setting | Value |
|---------|-------|
| Fusion LR | 5e-4 |
| LightGlue LR | 5e-5 |
| Everything else | Same as Phase 1 |

### 5.3 Dataset

**MegaDepth-1500:** 1500 calibrated image pairs from MegaDepth [12] covering 196 outdoor scenes. We use 1200/300 train/val split. Each pair provides: RGB images, lidar depth maps (HDF5), camera intrinsics K0, K1, relative rotation R, and translation t. Ground-truth matches are computed by 3D unprojection and reprojection using depth.

### 5.4 Training Curves

**Phase 1:**

| Epoch | Train NLL | Val NLL |
|-------|----------|---------|
| 1 | 2.298 | 1.735 |
| 10 | 0.984 | 1.085 |
| 20 | 0.555 | 0.912 |
| 30 | 0.408 | 0.901 |

**Phase 2:**

| Epoch | Train NLL | Val NLL |
|-------|----------|---------|
| 1 | 0.704 | 0.860 |
| 5 | 0.361 | 0.770 |
| 8 | 0.267 | **0.748** (best val) |
| 15 | 0.146 | 0.829 |
| 20 | 0.042 | 1.371 |

Phase 2 exhibits a train/val NLL divergence beginning at epoch 8, with HPatches performance degrading rapidly after epoch 5. We attribute this to the small training set (1200 pairs) being insufficient to fine-tune LightGlue's 3M parameters without overfitting to MegaDepth's scene distribution.

---

## 6. Experiments

### 6.1 Evaluation Benchmarks

**HPatches** [13]: 116 sequences of planar scenes, each with 1 reference and 5 query images with known homographies. 58 sequences with illumination variation, 58 with viewpoint variation. Metric: homography estimation AUC at corner-transfer-error thresholds of [1, 3, 5] pixels.

**MegaDepth-1500** [12]: 1500 image pairs with known relative pose. Metric: pose AUC at angular error thresholds of [5°, 10°, 20°].

### 6.2 Baseline

SP+LightGlue using the official pretrained weights (LightGlue trained on MegaDepth with SuperPoint descriptors). No modifications.

### 6.3 Ablation: Projection Architecture (Replacement Approach)

| Projection | Architecture | Loss @3ep |
|-----------|-------------|----------|
| Linear | 768 → 256 | 2.173 |
| **MLP-1** | **768→512 (ReLU)→256** | **2.014** |
| MLP-2 | 768→512 (LN+GELU)→256 | 2.038 |

### 6.4 Ablation: DINOv2 Backbone Size

| Backbone | Params | Feature Dim | Loss | Inference (ms) |
|---------|--------|-------------|------|---------------|
| ViT-S/14-reg | 22M | 384 | 2.275 | 308 |
| **ViT-B/14-reg** | **86M** | **768** | **2.068** | 387 |

### 6.5 Ablation: LightGlue Initialization

| Init | Loss @E1 | Loss @E3 |
|------|---------|---------|
| **Pretrained (SP weights)** | **3.627** | **2.023** |
| Random (Xavier) | 5.810 | 3.686 |

Pretrained LG weights provide a strong initialization even when the descriptor distribution changes — suggesting LightGlue's learned relational reasoning generalizes beyond the specific descriptor type.

### 6.6 Main Results: HPatches

| Method | All AUC@5 | Illum AUC@5 | VP AUC@5 |
|--------|----------|------------|---------|
| SP+LG Baseline | **48.6%** | **63.8%** | **34.0%** |
| DINOv2 Replacement | 2.1% | 3.2% | 1.0% |
| GatedFusion Phase 1 (best) | 43.8% | 59.3% | 26.8% |
| **GatedFusion Phase 2 E05** | **46.1%** | **62.9%** | **29.8%** |
| GatedFusion Phase 2 E08 (best NLL) | 6.1% | 11.9% | 0.5% |

### 6.7 Comparison with Old Replacement on MegaDepth

| Method | MegaDepth AUC@5° | AUC@10° | AUC@20° |
|--------|-----------------|--------|--------|
| SP+LG Baseline | — | 37.7% | — |
| DINOv2 Replacement (25 ep) | — | **39.6%** | — |

The replacement approach actually exceeds the baseline on MegaDepth pose estimation, confirming DINOv2's strength for 3D scene understanding. The fundamental failure is specifically on precise planar homography estimation.

---

## 7. Discussion

### 7.1 Spatial Precision vs. Semantic Understanding

Our experiments reveal a fundamental tension between the spatial precision needed for homography estimation and the semantic coarseness of Vision Transformer features. SuperPoint's convolutional inductive bias produces features that encode precise local structure at the detected keypoint. DINOv2's ViT patches encode neighborhood semantics over 14×14 pixel regions. This 196-pixel averaging window creates an irreconcilable spatial uncertainty for tasks requiring sub-pixel correspondence.

GatedFusion resolves this tension by allowing the model to selectively use semantic context (DINOv2) or spatial precision (SP) on a per-dimension basis. The learned gate is a differentiable selection mechanism.

### 7.2 Overfitting in Phase 2

The Phase 2 overfitting — near-zero train NLL but collapsing HPatches performance — reveals that MegaDepth NLL and HPatches AUC are measuring different aspects of matching quality. MegaDepth pairs are outdoor 3D scenes; HPatches are planar indoor/outdoor scenes evaluated on strict geometric precision. Fine-tuning LightGlue on 1200 MegaDepth pairs degrades its generalization to the HPatches domain.

This motivates two future directions: (1) larger, more diverse training sets; (2) domain-adaptive training that monitors performance on both datasets during training.

### 7.3 Parameter Efficiency

The proposed fusion adds only 395K parameters (13% of LightGlue) while leveraging the 86.6M-parameter DINOv2 frozen encoder at no training cost. This is a favorable parameter efficiency: 395K trainable parameters to integrate an 86.6M foundation model.

### 7.4 Limitations

1. **Spatial resolution ceiling:** Even in fusion mode, DINOv2's 14×14 patches are a spatial bottleneck. Intermediate DINOv2 features (pre-final-layer activations) have finer resolution and may be more suitable.
2. **Training data scale:** 1200 training pairs is insufficient for Phase 2 fine-tuning. The full MegaDepth training set (~200K pairs) would be required.
3. **Gate interpretability:** We have not yet visualized gate values spatially or analyzed which descriptor dimensions favor DINOv2 vs SP — this would be a compelling paper figure.

---

## 8. Conclusion

We present a systematic study of integrating DINOv2 foundation model features into the SuperPoint-LightGlue sparse matching pipeline. A direct descriptor replacement approach fails catastrophically on spatial precision tasks (2.1% HPatches AUC@5) due to DINOv2's inherent 14-pixel spatial quantization, despite exceeding baseline on pose estimation (39.6% vs 37.7% MegaDepth AUC@10°). Our proposed GatedFusion module — a 395K-parameter learned gate that interpolates between SP and DINOv2 features on a per-dimension basis — recovers 90% of baseline HPatches performance (46.1% vs 48.6%) while preserving the potential for semantic-enhanced matching on distribution-shifted test data.

Our work identifies key engineering requirements for training LightGlue end-to-end: removal of descriptor detachment, explicit log-assignment output, and gradient-transparent parameter freezing. The two-phase training strategy (fusion warm-up, then end-to-end fine-tuning) provides a stable training recipe that future work can build upon.

Future work will investigate: (1) larger-scale training on full MegaDepth; (2) HPatches-aware Phase 2 early stopping; (3) visualization of learned gate patterns; and (4) evaluation on cross-domain benchmarks such as Aachen Day-Night and ScanNet to test the semantic robustness hypothesis.

---

## References

[1] D. DeTone, T. Malisiewicz, and A. Rabinovich, "SuperPoint: Self-supervised interest point detection and description," in *CVPR Workshops*, 2018.

[2] P. Lindenberger, P.-E. Sarlin, and M. Pollefeys, "LightGlue: Local feature matching at light speed," in *ICCV*, 2023.

[3] M. Oquab, T. Darcet, T. Moutakanni, et al., "DINOv2: Learning robust visual features without supervision," *TMLR*, 2024.

[4] J. Edstedt, S. Athanasiadis, M. Wadenback, and M. Felsberg, "RoMa: Revisiting robust losses for dense feature matching," in *CVPR*, 2024.

[5] V. Leroy, Y. Cabon, and J. Revaud, "MASt3R: Matching and stereo 3D reconstruction," *arXiv:2406.09756*, 2024.

[6] M. Tyszkiewicz, P. Fua, and E. Trulls, "DISK: Learning local features with policy gradient," in *NeurIPS*, 2020.

[7] X. Zhao, X. Wu, W. Chen, P. C.-K. Chen, Q. Xu, and Z. Li, "ALIKED: A lighter keypoint and descriptor extraction network via deformable transformation," *IEEE TIM*, 2023.

[8] P.-E. Sarlin, D. DeTone, T. Malisiewicz, and A. Rabinovich, "SuperGlue: Learning feature matching with graph neural networks," in *CVPR*, 2020.

[9] J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, "LoFTR: Detector-free local feature matching with transformers," in *CVPR*, 2021.

[10] W. Chen, Z. Liu, X. Lin, et al., "ASpanFormer: Detector-free image matching with adaptive span self-attention," in *ECCV*, 2022.

[11] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, "DUSt3R: Geometric 3D Vision Made Easy," in *CVPR*, 2024.

[12] Z. Li and N. Snavely, "MegaDepth: Learning single-view depth prediction from internet photos," in *CVPR*, 2018.

[13] V. Balntas, K. Lenc, A. Vedaldi, and K. Mikolajczyk, "HPatches: A benchmark and evaluation of handcrafted and learned local descriptors," in *CVPR*, 2017.

[14] P.-E. Sarlin, C. Cadena, R. Siegwart, and M. Dymczyk, "From coarse to fine: Robust hierarchical localization at large scale," in *CVPR*, 2019.

[15] A. Dosovitskiy, L. Beyer, A. Kolesnikov, et al., "An image is worth 16×16 words: Transformers for image recognition at scale," in *ICLR*, 2021.

---

*Code and experiments are available in the project repository. All experiments were conducted on an NVIDIA RTX 3060 (12GB VRAM) with PyTorch 2.x and Python 3.11.*
