"""
src/pipeline.py

End-to-end pipeline: SuperPoint (keypoints only) + DINOv2 (frozen, descriptors)
+ Projection MLP + LightGlue (fine-tunable matcher).

This is the core research contribution — see plan.md §4 for design decisions.

The pipeline accepts a standard LightGlue data dict and returns matches.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dinov2_extractor import DINOv2Extractor, build_dinov2_extractor, DINOV2_NATIVE_SIZE
from src.feature_sampling import sample_descriptors_bilinear, preprocess_image_for_dinov2
from src.projection import build_projection, ProjectionType


class DINOv2LightGluePipeline(nn.Module):
    """
    Full matching pipeline:

        Image0 ──┬── SuperPoint (kpts only, frozen) ──── kpts0, scores0
                 └── DINOv2 (frozen) ─── feat_map0 ─→ sample at kpts0 → desc0_raw
                                                              ↓
                                                     ProjectionMLP → desc0_proj (256-dim)
                                                              ↓
        Image1 ──(same) ──────────────────────────────── desc1_proj
                                                              ↓
                                                 LightGlue (trainable) → matches

    Args:
        dinov2_variant:   DINOv2 variant to use (default: "vitb14_reg")
        proj_type:        Projection type: "linear" | "mlp1" | "mlp2"
        input_dim:        DINOv2 feature dim (auto-set from variant)
        descriptor_dim:   LightGlue working dim (default: 256)
        max_keypoints:    Max keypoints per image (passed to SuperPoint)
        device:           "cuda" or "cpu"
        lightglue_weights: Path to LightGlue .pth checkpoint (optional)
    """

    def __init__(
        self,
        dinov2_variant: str = "vitb14_reg",
        proj_type: ProjectionType = "mlp2",
        descriptor_dim: int = 256,
        max_keypoints: int = 1024,
        device: str = "cuda",
        lightglue_weights: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.max_keypoints = max_keypoints
        self.descriptor_dim = descriptor_dim

        # ----------------------------------------------------------------
        # 1. DINOv2 feature extractor (frozen)
        # ----------------------------------------------------------------
        self.dinov2 = build_dinov2_extractor(
            variant=dinov2_variant,
            device=device,
            freeze=True,
        )
        dinov2_dim = self.dinov2.get_feat_dim()

        # ----------------------------------------------------------------
        # 2. Projection network (trainable)
        # ----------------------------------------------------------------
        self.projection = build_projection(
            proj_type=proj_type,
            input_dim=dinov2_dim,
            output_dim=descriptor_dim,
        ).to(device)

        # ----------------------------------------------------------------
        # 3. LightGlue matcher (trainable, strategy A: external projection)
        # ----------------------------------------------------------------
        self.lightglue = self._build_lightglue(
            descriptor_dim=descriptor_dim,
            weights=lightglue_weights,
        )

        # ----------------------------------------------------------------
        # 4. SuperPoint keypoint extractor (frozen, keypoints only)
        # ----------------------------------------------------------------
        self.superpoint = self._build_superpoint(device=device)

        self.to(device)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Run the full pipeline on a pair of images.

        Args:
            data: dict with keys:
                "image0": (B, 3, H, W) float32 in [0, 1]
                "image1": (B, 3, H, W) float32 in [0, 1]

                Optionally pre-extracted keypoints (skips SuperPoint):
                "keypoints0": (B, N, 2) float32
                "keypoints1": (B, M, 2) float32
                "keypoint_scores0": (B, N) float32
                "keypoint_scores1": (B, M) float32

        Returns:
            pred: dict with:
                "keypoints0", "keypoints1": (B, N/M, 2)
                "descriptors0", "descriptors1": (B, N/M, 256) — projected DINOv2
                "matches0": (B, N) — index into kpts1 for each kpt0, -1 = unmatched
                "matches1": (B, M) — index into kpts0 for each kpt1, -1 = unmatched
                "matching_scores0": (B, N)
                "matching_scores1": (B, M)
        """
        image0 = data["image0"].to(self.device)
        image1 = data["image1"].to(self.device)
        B, _, H0, W0 = image0.shape
        _, _, H1, W1 = image1.shape

        # ----------------------------------------------------------------
        # Step 1: Extract keypoints with SuperPoint
        # ----------------------------------------------------------------
        if "keypoints0" not in data:
            kpts0, scores0 = self._extract_keypoints(image0)
            kpts1, scores1 = self._extract_keypoints(image1)
        else:
            kpts0 = data["keypoints0"].to(self.device)
            kpts1 = data["keypoints1"].to(self.device)
            scores0 = data.get("keypoint_scores0", torch.ones(B, kpts0.shape[1], device=self.device))
            scores1 = data.get("keypoint_scores1", torch.ones(B, kpts1.shape[1], device=self.device))

        # ----------------------------------------------------------------
        # Step 2: Extract DINOv2 features
        # ----------------------------------------------------------------
        # Resize images to DINOv2 native resolution
        img0_dino, orig_size0 = preprocess_image_for_dinov2(image0, DINOV2_NATIVE_SIZE)
        img1_dino, orig_size1 = preprocess_image_for_dinov2(image1, DINOV2_NATIVE_SIZE)

        with torch.no_grad():
            feat_map0 = self.dinov2(img0_dino)  # (B, D, 37, 37)
            feat_map1 = self.dinov2(img1_dino)

        # Scale keypoints from original resolution to DINOv2 input resolution
        kpts0_scaled = self._scale_keypoints(kpts0, orig_size0, (DINOV2_NATIVE_SIZE, DINOV2_NATIVE_SIZE))
        kpts1_scaled = self._scale_keypoints(kpts1, orig_size1, (DINOV2_NATIVE_SIZE, DINOV2_NATIVE_SIZE))

        # ----------------------------------------------------------------
        # Step 3: Sample DINOv2 features at keypoint locations
        # ----------------------------------------------------------------
        desc0_raw = sample_descriptors_bilinear(
            kpts0_scaled, feat_map0,
            image_size=(DINOV2_NATIVE_SIZE, DINOV2_NATIVE_SIZE),
            normalize=False,   # projection will normalize
        )
        desc1_raw = sample_descriptors_bilinear(
            kpts1_scaled, feat_map1,
            image_size=(DINOV2_NATIVE_SIZE, DINOV2_NATIVE_SIZE),
            normalize=False,
        )

        # ----------------------------------------------------------------
        # Step 4: Project to 256-dim descriptor space
        # ----------------------------------------------------------------
        desc0 = self.projection(desc0_raw)  # (B, N, 256)
        desc1 = self.projection(desc1_raw)  # (B, M, 256)

        # ----------------------------------------------------------------
        # Step 5: LightGlue matching
        # ----------------------------------------------------------------
        lg_input = {
            "image0": {"keypoints": kpts0, "descriptors": desc0, "image_size": (H0, W0)},
            "image1": {"keypoints": kpts1, "descriptors": desc1, "image_size": (H1, W1)},
        }
        # Scores/confidence passed if available
        if scores0 is not None:
            lg_input["image0"]["keypoint_scores"] = scores0
            lg_input["image1"]["keypoint_scores"] = scores1

        pred = self.lightglue(lg_input)

        # Augment with our keypoints and descriptors
        pred.update({
            "keypoints0": kpts0,
            "keypoints1": kpts1,
            "descriptors0": desc0,
            "descriptors1": desc1,
        })

        return pred

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def get_trainable_params(self) -> list:
        """Return only parameters that should be trained."""
        params = list(self.projection.parameters())
        params += list(self.lightglue.parameters())
        return params

    def freeze_lightglue(self) -> None:
        for p in self.lightglue.parameters():
            p.requires_grad = False

    def unfreeze_lightglue(self) -> None:
        for p in self.lightglue.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_lightglue(self, descriptor_dim: int, weights: Optional[Path]) -> nn.Module:
        """Build LightGlue configured for our descriptor dimension."""
        try:
            # Try importing from the installed LightGlue package
            from lightglue import LightGlue
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "lightglue",
                Path(__file__).parent.parent / "LightGlue" / "lightglue" / "lightglue.py",
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            LightGlue = mod.LightGlue

        # Create LightGlue with our custom input_dim
        # We pass features=None to skip the built-in feature type lookup
        # and set input_dim manually
        lg = LightGlue(
            features=None,
            input_dim=descriptor_dim,
            descriptor_dim=descriptor_dim,
        )

        if weights is not None:
            state = torch.load(weights, map_location="cpu", weights_only=True)
            # Load partial weights (skip input_proj if dim mismatch)
            missing, unexpected = lg.load_state_dict(state, strict=False)
            if missing:
                print(f"  LightGlue: {len(missing)} missing keys (expected if loading SP weights)")
        else:
            # Load pretrained SuperPoint-LightGlue weights and transfer
            # compatible layers (all transformer layers — they're dim-agnostic at 256)
            self._load_superpoint_lightglue_weights(lg)

        return lg

    def _load_superpoint_lightglue_weights(self, lg: nn.Module) -> None:
        """
        Transfer SuperPoint-LightGlue pretrained weights to our LightGlue instance.
        All transformer layers work at descriptor_dim=256 regardless of input_dim,
        so they can be transferred. Only input_proj is skipped (dim mismatch).
        """
        try:
            from lightglue import LightGlue
            sp_lg = LightGlue(features="superpoint")
            sp_lg.eval()

            our_state = lg.state_dict()
            sp_state = sp_lg.state_dict()

            transferred = 0
            skipped = 0
            for key in our_state:
                if key in sp_state and sp_state[key].shape == our_state[key].shape:
                    our_state[key] = sp_state[key]
                    transferred += 1
                else:
                    skipped += 1  # input_proj will be skipped due to dim

            lg.load_state_dict(our_state, strict=True)
            print(f"  Transferred {transferred} layers from SP-LightGlue, "
                  f"skipped {skipped} (input_proj — expected).")
            del sp_lg
        except Exception as e:
            print(f"  Warning: Could not load SP-LightGlue weights: {e}. "
                  f"Using random init.")

    def _build_superpoint(self, device: str) -> Optional[nn.Module]:
        """Build SuperPoint extractor (keypoints only)."""
        try:
            from lightglue import SuperPoint
            sp = SuperPoint(max_num_keypoints=self.max_keypoints).to(device).eval()
            for p in sp.parameters():
                p.requires_grad = False
            print("  SuperPoint loaded and frozen.")
            return sp
        except ImportError:
            print("  Warning: SuperPoint not available. "
                  "Pass keypoints0/keypoints1 explicitly.")
            return None

    def _extract_keypoints(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract keypoints using SuperPoint."""
        if self.superpoint is None:
            raise RuntimeError(
                "SuperPoint is not available. Pass 'keypoints0'/'keypoints1' "
                "in the data dict directly."
            )
        with torch.no_grad():
            result = self.superpoint({"image": image})
        return result["keypoints"], result.get("keypoint_scores")

    @staticmethod
    def _scale_keypoints(
        kpts: torch.Tensor,
        original_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Scale keypoint coordinates from original_size to target_size."""
        H_orig, W_orig = original_size
        H_tgt, W_tgt = target_size

        scale = torch.tensor(
            [W_tgt / W_orig, H_tgt / H_orig],
            dtype=kpts.dtype,
            device=kpts.device,
        )
        return kpts * scale.unsqueeze(0).unsqueeze(0)

    def get_vram_usage_mb(self) -> float:
        """Return current VRAM usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0


# ---------------------------------------------------------------------------
# Quick test (no actual model loading — just structure)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Pipeline structure test (no model download)...")
    print("To test the full pipeline, run Exp1.ipynb Day 3 cells.")
    print("Module imports OK.")
