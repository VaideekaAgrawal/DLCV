"""
src/dinov2_extractor.py

DINOv2 feature extractor module.
Loads a DINOv2 ViT backbone and returns dense patch feature maps
that can be sampled at arbitrary keypoint locations.

Day 3 deliverable — see plan.md §7 Experiment 1.1
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Supported DINOv2 variants
# ---------------------------------------------------------------------------
DINOV2_VARIANTS = {
    "vits14":     {"hub_name": "dinov2_vits14",     "feat_dim": 384,  "patch_size": 14},
    "vits14_reg": {"hub_name": "dinov2_vits14_reg", "feat_dim": 384,  "patch_size": 14},
    "vitb14":     {"hub_name": "dinov2_vitb14",     "feat_dim": 768,  "patch_size": 14},
    "vitb14_reg": {"hub_name": "dinov2_vitb14_reg", "feat_dim": 768,  "patch_size": 14},
    "vitl14":     {"hub_name": "dinov2_vitl14",     "feat_dim": 1024, "patch_size": 14},
    "vitl14_reg": {"hub_name": "dinov2_vitl14_reg", "feat_dim": 1024, "patch_size": 14},
}

# Default variant — ViT-B/14 with registers (best quality for downstream)
DEFAULT_VARIANT = "vitb14_reg"
# Native input resolution that produces integer 37×37 grid (518 = 14×37)
DINOV2_NATIVE_SIZE = 518


class DINOv2Extractor(nn.Module):
    """
    Wraps a DINOv2 backbone for keypoint-level feature extraction.

    Usage::

        extractor = DINOv2Extractor(variant="vitb14_reg", device="cuda")
        # image: (B, 3, H, W) float32 in [0, 1]
        feature_map = extractor(image)
        # feature_map: (B, feat_dim, H//14, W//14)  e.g. (B, 768, 37, 37) for 518×518

    The feature map can be passed to `sample_descriptors_bilinear` to get
    per-keypoint descriptors.
    """

    def __init__(
        self,
        variant: str = DEFAULT_VARIANT,
        device: str = "cuda",
        freeze: bool = True,
        local_weights: Optional[Path] = None,
    ) -> None:
        super().__init__()

        if variant not in DINOV2_VARIANTS:
            raise ValueError(
                f"Unknown DINOv2 variant '{variant}'. "
                f"Choose from: {list(DINOV2_VARIANTS.keys())}"
            )

        self.variant = variant
        self.feat_dim: int = DINOV2_VARIANTS[variant]["feat_dim"]
        self.patch_size: int = DINOV2_VARIANTS[variant]["patch_size"]
        hub_name: str = DINOV2_VARIANTS[variant]["hub_name"]

        # ----------------------------------------------------------------
        # Load backbone
        # ----------------------------------------------------------------
        if local_weights is not None:
            # Load from local .pth file (no internet required)
            model = torch.hub.load(
                "facebookresearch/dinov2",
                hub_name,
                source="local",
                pretrained=False,
            )
            state = torch.load(local_weights, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
        else:
            model = torch.hub.load(
                "facebookresearch/dinov2",
                hub_name,
                verbose=False,
            )

        self.model = model.to(device)

        if freeze:
            self._freeze()

        self.device = device

        # ImageNet normalization (DINOv2 standard)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract DINOv2 patch feature map.

        Args:
            image: (B, 3, H, W) float32, values in [0, 1].
                   H and W must be divisible by patch_size (14).
                   Recommended: resize to 518×518 first.

        Returns:
            feature_map: (B, feat_dim, h, w) where h=H//14, w=W//14.
        """
        B, C, H, W = image.shape

        if H % self.patch_size != 0 or W % self.patch_size != 0:
            warnings.warn(
                f"Image size ({H}×{W}) not divisible by patch_size={self.patch_size}. "
                f"Padding to nearest multiple.",
                stacklevel=2,
            )
            H_pad = (self.patch_size - H % self.patch_size) % self.patch_size
            W_pad = (self.patch_size - W % self.patch_size) % self.patch_size
            image = F.pad(image, (0, W_pad, 0, H_pad))
            _, _, H, W = image.shape

        # ImageNet normalization
        x = (image - self.mean) / self.std

        # DINOv2 forward → get patch tokens
        # get_intermediate_layers returns a list of (B, N_patches, feat_dim)
        features = self.model.get_intermediate_layers(
            x,
            n=1,              # last layer only
            reshape=True,     # reshape to (B, feat_dim, h, w) — requires DINOv2 >= 2023.09
        )
        # features is a tuple of length 1
        feature_map = features[0]  # (B, feat_dim, h, w)
        return feature_map

    def extract_features_batch(
        self,
        images: torch.Tensor,
        batch_size: int = 4,
    ) -> torch.Tensor:
        """
        Memory-efficient batch extraction. Splits a large batch into chunks.

        Args:
            images: (N, 3, H, W) float32 in [0, 1]
            batch_size: chunk size for forward passes

        Returns:
            feature_maps: (N, feat_dim, h, w)
        """
        all_features = []
        was_training = self.training
        self.eval()

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                chunk = images[i : i + batch_size].to(self.device)
                feat = self.forward(chunk)
                all_features.append(feat.cpu())

        if was_training:
            self.train()

        return torch.cat(all_features, dim=0)

    def get_feat_dim(self) -> int:
        return self.feat_dim

    def get_patch_size(self) -> int:
        return self.patch_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def unfreeze_last_n_blocks(self, n: int) -> None:
        """Optionally unfreeze last N transformer blocks for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = False

        blocks = list(self.model.blocks)  # DINOv2 has .blocks attribute
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

        print(f"Unfroze last {n} DINOv2 transformer blocks.")

    def train(self, mode: bool = True):
        """Keep DINOv2 in eval mode if frozen."""
        super().train(mode)
        if mode:
            # DINOv2 backbone stays in eval (BatchNorm etc.)
            self.model.eval()
        return self


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_dinov2_extractor(
    variant: str = DEFAULT_VARIANT,
    device: str = "cuda",
    freeze: bool = True,
    local_weights: Optional[Path] = None,
) -> DINOv2Extractor:
    """Factory function to build a DINOv2Extractor."""
    print(f"Loading DINOv2 variant: {variant}  "
          f"(feat_dim={DINOV2_VARIANTS[variant]['feat_dim']})")
    extractor = DINOv2Extractor(
        variant=variant,
        device=device,
        freeze=freeze,
        local_weights=local_weights,
    )
    n_params = sum(p.numel() for p in extractor.model.parameters()) / 1e6
    print(f"  Loaded. Parameters: {n_params:.1f}M | Frozen: {freeze}")
    return extractor


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    extractor = build_dinov2_extractor(variant="vitb14_reg", device=device, freeze=True)

    # Simulate a batch of 2 images at 518×518
    dummy_images = torch.rand(2, 3, DINOV2_NATIVE_SIZE, DINOV2_NATIVE_SIZE, device=device)

    with torch.no_grad():
        feature_map = extractor(dummy_images)

    print(f"Input:        {dummy_images.shape}")   # (2, 3, 518, 518)
    print(f"Feature map:  {feature_map.shape}")    # (2, 768, 37, 37)
    assert feature_map.shape == (2, 768, 37, 37), "Unexpected output shape!"
    print("✓ DINOv2Extractor test passed.")
