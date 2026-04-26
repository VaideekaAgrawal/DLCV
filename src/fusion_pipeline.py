"""
src/fusion_pipeline.py

THE PIVOT: Descriptor Fusion Pipeline.

Instead of REPLACING SuperPoint descriptors with DINOv2 (which loses spatial 
precision), this pipeline FUSES them — keeping SP's sub-pixel accuracy while 
adding DINOv2's semantic robustness.

Architecture:
    Image ─┬── SuperPoint ──── kpts, scores, SP_desc (256-d)
           └── DINOv2 (frozen) ─── feat_map ─→ sample at kpts → raw (768-d)
                                                       ↓
                                            DINOv2Proj → dino_desc (128-d)
                                                       ↓
                                    FusionModule(SP_desc, dino_desc) → fused (256-d)
                                                       ↓
                                            LightGlue (pretrained SP weights!) → matches

KEY ADVANTAGE: LightGlue sees 256-dim descriptors just like with SuperPoint,
so we can use FULL pretrained SP-LightGlue weights without any dim mismatch.
Only the fusion module is new and trained.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dinov2_extractor import build_dinov2_extractor, DINOV2_NATIVE_SIZE
from src.feature_sampling import sample_descriptors_bilinear, preprocess_image_for_dinov2
from src.descriptor_fusion import (
    build_fusion, DINOv2FusionProjection, MultiScaleDINOv2Sampler,
    FusionStrategy,
)


class FusionPipeline(nn.Module):
    """
    SP + DINOv2 Descriptor Fusion Pipeline for LightGlue.
    
    This is the publishable version. Key design choices:
    1. SuperPoint provides keypoints AND descriptors (spatial precision preserved)
    2. DINOv2 provides semantic features sampled at SP keypoint locations
    3. Learned fusion combines both → 256-dim output for LightGlue
    4. LightGlue loaded with FULL pretrained SP weights (no dim mismatch!)
    
    Training modes:
    - "fusion_only": Freeze LightGlue, train fusion module + DINOv2 projection
    - "end_to_end": Train fusion + LightGlue jointly (after fusion warmup)
    """

    def __init__(
        self,
        dinov2_variant: str = "vitb14_reg",
        fusion_strategy: FusionStrategy = "gated",
        dino_proj_dim: int = 128,
        descriptor_dim: int = 256,
        max_keypoints: int = 1024,
        multi_scale: bool = False,
        device: str = "cuda",
        lightglue_weights: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.max_keypoints = max_keypoints
        self.descriptor_dim = descriptor_dim
        self.multi_scale = multi_scale

        # 1. DINOv2 feature extractor (frozen)
        self.dinov2 = build_dinov2_extractor(
            variant=dinov2_variant, device=device, freeze=True,
        )
        dinov2_dim = self.dinov2.get_feat_dim()

        # 2. DINOv2 projection (768 → 128, trainable)
        self.dino_projection = DINOv2FusionProjection(
            input_dim=dinov2_dim, output_dim=dino_proj_dim,
        ).to(device)

        # 3. Fusion module (trainable)
        self.fusion = build_fusion(
            strategy=fusion_strategy,
            sp_dim=descriptor_dim,
            dino_proj_dim=dino_proj_dim,
            output_dim=descriptor_dim,
        ).to(device)

        # 4. Multi-scale sampler (optional)
        if multi_scale:
            self.ms_sampler = MultiScaleDINOv2Sampler()
        else:
            self.ms_sampler = None

        # 5. SuperPoint (frozen, extracts keypoints + descriptors)
        self.superpoint = self._build_superpoint(device)

        # 6. LightGlue (with FULL pretrained SP weights — no dim mismatch!)
        self.lightglue = self._build_lightglue(descriptor_dim, lightglue_weights)

        self.to(device)

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        image0 = data["image0"].to(self.device)
        image1 = data["image1"].to(self.device)
        B, _, H0, W0 = image0.shape
        _, _, H1, W1 = image1.shape

        # Step 1: SuperPoint — keypoints + descriptors
        if "keypoints0" not in data:
            sp_out0 = self._run_superpoint(image0)
            sp_out1 = self._run_superpoint(image1)
            kpts0, scores0, sp_desc0 = sp_out0
            kpts1, scores1, sp_desc1 = sp_out1
        else:
            kpts0 = data["keypoints0"].to(self.device)
            kpts1 = data["keypoints1"].to(self.device)
            scores0 = data.get("keypoint_scores0", torch.ones(B, kpts0.shape[1], device=self.device))
            scores1 = data.get("keypoint_scores1", torch.ones(B, kpts1.shape[1], device=self.device))
            sp_desc0 = data["descriptors0"].to(self.device)
            sp_desc1 = data["descriptors1"].to(self.device)

        # Step 2: DINOv2 features at keypoint locations
        if self.ms_sampler is not None:
            dino_raw0 = self.ms_sampler(image0, kpts0, self.dinov2, (H0, W0))
            dino_raw1 = self.ms_sampler(image1, kpts1, self.dinov2, (H1, W1))
        else:
            dino_raw0, dino_raw1 = self._extract_dino_at_keypoints(
                image0, image1, kpts0, kpts1, (H0, W0), (H1, W1)
            )

        # Step 3: Project DINOv2 features
        dino_proj0 = self.dino_projection(dino_raw0)  # (B, N, 128)
        dino_proj1 = self.dino_projection(dino_raw1)

        # Step 4: Fuse SP + DINOv2 → 256-dim
        fused0 = self.fusion(sp_desc0, dino_proj0)  # (B, N, 256)
        fused1 = self.fusion(sp_desc1, dino_proj1)

        # Step 5: LightGlue matching with fused descriptors
        lg_input = {
            "image0": {
                "keypoints": kpts0, "descriptors": fused0,
                "image_size": torch.tensor([[H0, W0]], device=self.device).expand(B, -1),
            },
            "image1": {
                "keypoints": kpts1, "descriptors": fused1,
                "image_size": torch.tensor([[H1, W1]], device=self.device).expand(B, -1),
            },
        }
        if scores0 is not None:
            lg_input["image0"]["keypoint_scores"] = scores0
            lg_input["image1"]["keypoint_scores"] = scores1

        pred = self.lightglue(lg_input)

        pred.update({
            "keypoints0": kpts0, "keypoints1": kpts1,
            "descriptors0": fused0, "descriptors1": fused1,
            "sp_descriptors0": sp_desc0, "sp_descriptors1": sp_desc1,
            "dino_descriptors0": dino_proj0, "dino_descriptors1": dino_proj1,
        })

        return pred

    def forward_from_cache(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Forward pass using pre-cached SP and DINOv2 features.
        For fast training — no SP or DINOv2 forward pass needed.
        
        data must contain:
            keypoints0/1, keypoint_scores0/1, 
            descriptors0/1 (SP),
            dino_descriptors0/1 (raw DINOv2 at keypoints, 768-d),
            image_size0/1
        """
        kpts0 = data["keypoints0"].to(self.device)
        kpts1 = data["keypoints1"].to(self.device)
        scores0 = data["keypoint_scores0"].to(self.device)
        scores1 = data["keypoint_scores1"].to(self.device)
        sp_desc0 = data["descriptors0"].to(self.device)
        sp_desc1 = data["descriptors1"].to(self.device)
        dino_raw0 = data["dino_descriptors0"].to(self.device)
        dino_raw1 = data["dino_descriptors1"].to(self.device)

        B = kpts0.shape[0]
        
        # Project + Fuse
        dino_proj0 = self.dino_projection(dino_raw0)
        dino_proj1 = self.dino_projection(dino_raw1)
        fused0 = self.fusion(sp_desc0, dino_proj0)
        fused1 = self.fusion(sp_desc1, dino_proj1)

        # LightGlue matching
        H0, W0 = data["image_size0"]
        H1, W1 = data["image_size1"]
        lg_input = {
            "image0": {
                "keypoints": kpts0, "descriptors": fused0,
                "image_size": torch.tensor([[H0, W0]], device=self.device).expand(B, -1),
            },
            "image1": {
                "keypoints": kpts1, "descriptors": fused1,
                "image_size": torch.tensor([[H1, W1]], device=self.device).expand(B, -1),
            },
        }
        if scores0 is not None:
            lg_input["image0"]["keypoint_scores"] = scores0
            lg_input["image1"]["keypoint_scores"] = scores1

        pred = self.lightglue(lg_input)

        pred.update({
            "keypoints0": kpts0, "keypoints1": kpts1,
            "descriptors0": fused0, "descriptors1": fused1,
        })
        return pred

    # ------------------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------------------

    def get_trainable_params(self, mode: str = "fusion_only") -> list[dict]:
        """
        Get parameter groups for optimizer.
        
        Modes:
        - "fusion_only": Train fusion + dino_projection only (LG frozen)
        - "end_to_end": Train fusion + dino_projection + LightGlue
        - "lg_finetune": Lower LR for LightGlue, higher for fusion
        """
        fusion_params = (
            list(self.dino_projection.parameters()) + 
            list(self.fusion.parameters())
        )
        
        if mode == "fusion_only":
            # Don't freeze LG params — gradients must flow through LG
            # to reach fusion module. Just don't include LG in optimizer.
            return [{"params": fusion_params, "lr": 1e-3}]
        
        elif mode == "end_to_end":
            for p in self.lightglue.parameters():
                p.requires_grad = True
            return [
                {"params": fusion_params, "lr": 1e-3},
                {"params": list(self.lightglue.parameters()), "lr": 1e-4},
            ]
        
        elif mode == "lg_finetune":
            for p in self.lightglue.parameters():
                p.requires_grad = True
            return [
                {"params": fusion_params, "lr": 5e-4},
                {"params": list(self.lightglue.parameters()), "lr": 5e-5},
            ]
        
        raise ValueError(f"Unknown training mode: {mode}")

    def set_training_mode(self, mode: str) -> None:
        """Configure which modules are trainable."""
        # DINOv2 and SuperPoint always frozen
        self.dinov2.eval()
        if self.superpoint:
            self.superpoint.eval()
        
        # Fusion always trainable
        self.dino_projection.train()
        self.fusion.train()
        
        if mode == "fusion_only":
            # LG in eval mode but requires_grad=True so gradients flow
            # through LG back to fused descriptors. LG params are NOT
            # in the optimizer so they won't be updated.
            self.lightglue.eval()
            for p in self.lightglue.parameters():
                p.requires_grad = True  # needed for gradient flow!
        else:
            self.lightglue.train()
            for p in self.lightglue.parameters():
                p.requires_grad = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_dino_at_keypoints(
        self, image0, image1, kpts0, kpts1, size0, size1,
    ):
        """Extract DINOv2 features at SP keypoint locations."""
        img0_dino, _ = preprocess_image_for_dinov2(image0, DINOV2_NATIVE_SIZE)
        img1_dino, _ = preprocess_image_for_dinov2(image1, DINOV2_NATIVE_SIZE)

        with torch.no_grad():
            feat0 = self.dinov2(img0_dino)
            feat1 = self.dinov2(img1_dino)

        H0, W0 = size0
        H1, W1 = size1
        kpts0_sc = kpts0.clone()
        kpts0_sc[..., 0] *= DINOV2_NATIVE_SIZE / W0
        kpts0_sc[..., 1] *= DINOV2_NATIVE_SIZE / H0
        kpts1_sc = kpts1.clone()
        kpts1_sc[..., 0] *= DINOV2_NATIVE_SIZE / W1
        kpts1_sc[..., 1] *= DINOV2_NATIVE_SIZE / H1

        desc0 = sample_descriptors_bilinear(
            kpts0_sc, feat0, (DINOV2_NATIVE_SIZE, DINOV2_NATIVE_SIZE), normalize=True
        )
        desc1 = sample_descriptors_bilinear(
            kpts1_sc, feat1, (DINOV2_NATIVE_SIZE, DINOV2_NATIVE_SIZE), normalize=True
        )
        return desc0, desc1

    def _run_superpoint(self, image):
        """Run SuperPoint and return keypoints, scores, descriptors."""
        assert self.superpoint is not None, "SuperPoint not available"
        with torch.no_grad():
            result = self.superpoint({"image": image})
        return (
            result["keypoints"],
            result.get("keypoint_scores"),
            result["descriptors"],
        )

    def _build_superpoint(self, device):
        try:
            from lightglue import SuperPoint
            sp = SuperPoint(max_num_keypoints=self.max_keypoints).to(device).eval()
            for p in sp.parameters():
                p.requires_grad = False
            print("  SuperPoint loaded (keypoints + descriptors).")
            return sp
        except ImportError:
            print("  Warning: SuperPoint not available.")
            return None

    def _build_lightglue(self, descriptor_dim, weights):
        try:
            from lightglue import LightGlue as LG
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "lightglue",
                Path(__file__).parent.parent / "LightGlue" / "lightglue" / "lightglue.py",
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            LG = mod.LightGlue

        if weights is not None:
            lg = LG(features=None, input_dim=descriptor_dim, descriptor_dim=descriptor_dim)
            state = torch.load(weights, map_location="cpu", weights_only=True)
            lg.load_state_dict(state, strict=False)
            print(f"  LightGlue loaded from {weights}")
        else:
            # Use FULL pretrained SuperPoint-LightGlue weights!
            # descriptor_dim=256 matches SP exactly, no dim mismatch.
            lg = LG(features="superpoint")
            print("  LightGlue loaded with FULL pretrained SP weights (no dim mismatch)")
        
        return lg


# ---------------------------------------------------------------------------
# SP-only baseline (for fair comparison)
# ---------------------------------------------------------------------------

class SPLightGlueBaseline(nn.Module):
    """Pure SP+LG baseline with identical evaluation code."""
    
    def __init__(self, max_keypoints=1024, device="cuda"):
        super().__init__()
        from lightglue import SuperPoint, LightGlue as LG
        self.superpoint = SuperPoint(max_num_keypoints=max_keypoints).to(device).eval()
        self.lightglue = LG(features="superpoint").to(device).eval()
        self.device = device
    
    @torch.no_grad()
    def forward(self, data):
        image0 = data["image0"].to(self.device)
        image1 = data["image1"].to(self.device)
        out0 = self.superpoint({"image": image0})
        out1 = self.superpoint({"image": image1})
        
        B, _, H0, W0 = image0.shape
        _, _, H1, W1 = image1.shape
        
        lg_input = {
            "image0": {
                "keypoints": out0["keypoints"],
                "descriptors": out0["descriptors"],
                "image_size": torch.tensor([[H0, W0]], device=self.device).expand(B, -1),
            },
            "image1": {
                "keypoints": out1["keypoints"],
                "descriptors": out1["descriptors"],
                "image_size": torch.tensor([[H1, W1]], device=self.device).expand(B, -1),
            },
        }
        pred = self.lightglue(lg_input)
        pred.update({
            "keypoints0": out0["keypoints"], "keypoints1": out1["keypoints"],
            "descriptors0": out0["descriptors"], "descriptors1": out1["descriptors"],
        })
        return pred


if __name__ == "__main__":
    print("FusionPipeline module imports OK.")
    print("Run Exp2_fusion.ipynb to test the full pipeline.")
