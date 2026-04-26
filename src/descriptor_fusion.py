"""
src/descriptor_fusion.py

Descriptor Fusion Module: Combines SuperPoint local descriptors with 
DINOv2 semantic features for enhanced sparse matching.

THE KEY PIVOT: Instead of REPLACING SP descriptors with DINOv2 (which loses
spatial precision), we FUSE them — keeping SP's sub-pixel accuracy while 
adding DINOv2's semantic robustness.

Architecture:
    SP desc (256-d) ─────────────────────┐
                                          ├── concat (256+128=384) → FusionMLP → 256-d
    DINOv2 feat (768-d) → ProjMLP (128) ─┘

The fusion output is 256-dim, directly compatible with pretrained LightGlue.
"""

from __future__ import annotations
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------

FusionStrategy = Literal["concat_mlp", "gated", "adaptive"]


class ConcatFusionMLP(nn.Module):
    """
    Concatenation + MLP fusion.
    
    SP(256) + DINOv2_proj(128) → concat(384) → MLP → 256
    
    Simple but effective. The MLP learns which descriptor dimensions
    to emphasize based on the joint representation.
    """

    def __init__(
        self,
        sp_dim: int = 256,
        dino_proj_dim: int = 128,
        output_dim: int = 256,
        hidden_dim: int = 384,
    ) -> None:
        super().__init__()
        input_dim = sp_dim + dino_proj_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )
        self._init_weights()

    def forward(self, sp_desc: torch.Tensor, dino_desc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sp_desc:   (..., 256) SuperPoint descriptors, L2-normalized
            dino_desc: (..., 128) Projected DINOv2 features, L2-normalized
        Returns:
            fused: (..., 256) L2-normalized fused descriptor
        """
        x = torch.cat([sp_desc, dino_desc], dim=-1)
        out = self.net(x)
        return F.normalize(out, p=2, dim=-1)

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


class GatedFusion(nn.Module):
    """
    Gated fusion with learned attention over SP vs DINOv2 contributions.
    
    Learns a per-dimension gate: output = gate * SP + (1-gate) * DINOv2_proj
    The gate is predicted from the concatenation of both descriptors.
    
    This is more interpretable — we can visualize which keypoints
    rely more on semantic vs local features.
    """

    def __init__(
        self,
        sp_dim: int = 256,
        dino_proj_dim: int = 128,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.sp_dim = sp_dim
        self.output_dim = output_dim
        
        # Project DINOv2 to same dim as SP
        self.dino_align = nn.Sequential(
            nn.Linear(dino_proj_dim, output_dim, bias=True),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        
        # Gate predictor: from concat of both → per-dim gate in [0,1]
        self.gate = nn.Sequential(
            nn.Linear(sp_dim + output_dim, output_dim, bias=True),
            nn.Sigmoid(),
        )
        self._init_weights()

    def forward(self, sp_desc: torch.Tensor, dino_desc: torch.Tensor) -> torch.Tensor:
        dino_aligned = self.dino_align(dino_desc)
        gate_input = torch.cat([sp_desc, dino_aligned], dim=-1)
        g = self.gate(gate_input)  # (..., 256) in [0, 1]
        fused = g * sp_desc + (1 - g) * dino_aligned
        return F.normalize(fused, p=2, dim=-1)

    def forward_with_gate(self, sp_desc: torch.Tensor, dino_desc: torch.Tensor):
        """Return fused descriptor AND gate values for visualization."""
        dino_aligned = self.dino_align(dino_desc)
        gate_input = torch.cat([sp_desc, dino_aligned], dim=-1)
        g = self.gate(gate_input)
        fused = g * sp_desc + (1 - g) * dino_aligned
        return F.normalize(fused, p=2, dim=-1), g.mean(dim=-1)  # avg gate per keypoint

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Initialize gate bias so it starts near 0.5 (equal weighting)
        # The last linear in self.gate
        nn.init.zeros_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, 0.0)  # sigmoid(0) = 0.5


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion with cross-attention between SP and DINOv2 features.
    
    Uses a lightweight cross-attention: DINOv2 features attend to SP features
    to determine what semantic context to inject.
    """

    def __init__(
        self,
        sp_dim: int = 256,
        dino_proj_dim: int = 128,
        output_dim: int = 256,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        
        # Align DINOv2 to output dim
        self.dino_proj = nn.Linear(dino_proj_dim, output_dim, bias=True)
        
        # Cross-attention: SP queries, DINOv2 keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        
        # Final fusion
        self.ffn = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
        self._init_weights()

    def forward(self, sp_desc: torch.Tensor, dino_desc: torch.Tensor) -> torch.Tensor:
        dino_proj = self.dino_proj(dino_desc)
        # Cross-attention: SP attends to DINOv2 context
        attn_out, _ = self.cross_attn(sp_desc, dino_proj, dino_proj)
        # Residual + FFN
        fused = self.ffn(torch.cat([sp_desc, attn_out], dim=-1))
        return F.normalize(sp_desc + fused, p=2, dim=-1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# DINOv2 projection for fusion (smaller than replacement — only 128-dim)
# ---------------------------------------------------------------------------

class DINOv2FusionProjection(nn.Module):
    """
    Lightweight projection: DINOv2 768-dim → 128-dim for fusion.
    Smaller than the replacement projection (256-dim) because we only need 
    to capture semantic signal, not replace spatial precision.
    """

    def __init__(self, input_dim: int = 768, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256, bias=True),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, output_dim, bias=True),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Multi-scale DINOv2 feature extraction
# ---------------------------------------------------------------------------

class MultiScaleDINOv2Sampler(nn.Module):
    """
    Extract DINOv2 features at multiple resolutions and combine them.
    
    This addresses the spatial precision problem: at higher resolution,
    we get finer spatial grids (more patches), improving bilinear sampling.
    
    Scales: [518, 728, 364] → grids [37×37, 52×52, 26×26]
    Features are sampled at each scale and averaged.
    """

    def __init__(
        self,
        scales: list[int] = None,
        patch_size: int = 14,
    ) -> None:
        super().__init__()
        if scales is None:
            # Resolutions must be divisible by 14
            self.scales = [518, 728]  # 37×37 and 52×52
        else:
            self.scales = scales
        self.patch_size = patch_size

    def forward(
        self,
        image: torch.Tensor,
        keypoints: torch.Tensor,
        dinov2: nn.Module,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        Extract multi-scale DINOv2 features at keypoint locations.
        
        Args:
            image: (B, 3, H, W) original image
            keypoints: (B, N, 2) keypoint locations in original pixel coords
            dinov2: DINOv2Extractor module
            image_size: (H, W) of original image
            
        Returns:
            descriptors: (B, N, D) averaged multi-scale descriptors
        """
        from src.feature_sampling import sample_descriptors_bilinear
        
        all_descs = []
        
        for scale in self.scales:
            # Resize image to this scale
            img_scaled = F.interpolate(
                image, size=(scale, scale), mode="bilinear", align_corners=False
            )
            
            # Extract features
            with torch.no_grad():
                feat_map = dinov2(img_scaled)  # (B, D, scale//14, scale//14)
            
            # Scale keypoints to this resolution
            H, W = image_size
            kpts_scaled = keypoints.clone()
            kpts_scaled[..., 0] = kpts_scaled[..., 0] * scale / W
            kpts_scaled[..., 1] = kpts_scaled[..., 1] * scale / H
            
            # Sample at keypoints
            desc = sample_descriptors_bilinear(
                kpts_scaled, feat_map,
                image_size=(scale, scale),
                normalize=True,
            )
            all_descs.append(desc)
        
        # Average across scales
        stacked = torch.stack(all_descs, dim=0)  # (S, B, N, D)
        return stacked.mean(dim=0)  # (B, N, D)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_FUSION_CLASSES = {
    "concat_mlp": ConcatFusionMLP,
    "gated": GatedFusion,
    "adaptive": AdaptiveFusion,
}


def build_fusion(
    strategy: FusionStrategy = "gated",
    sp_dim: int = 256,
    dino_proj_dim: int = 128,
    output_dim: int = 256,
) -> nn.Module:
    """Build a descriptor fusion module."""
    if strategy not in _FUSION_CLASSES:
        raise ValueError(f"Unknown fusion strategy '{strategy}'. "
                         f"Choose from {list(_FUSION_CLASSES.keys())}")
    
    cls = _FUSION_CLASSES[strategy]
    fusion = cls(sp_dim=sp_dim, dino_proj_dim=dino_proj_dim, output_dim=output_dim)
    
    n_params = sum(p.numel() for p in fusion.parameters())
    print(f"Built '{strategy}' fusion: SP({sp_dim})+DINOv2({dino_proj_dim})→{output_dim}  "
          f"({n_params:,} trainable params)")
    return fusion


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, N = 2, 100
    sp_desc = F.normalize(torch.randn(B, N, 256), dim=-1)
    dino_desc = F.normalize(torch.randn(B, N, 128), dim=-1)

    for strategy in ["concat_mlp", "gated", "adaptive"]:
        fusion = build_fusion(strategy)
        out = fusion(sp_desc, dino_desc)
        assert out.shape == (B, N, 256), f"{strategy}: wrong shape {out.shape}"
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), f"{strategy}: not normalized"
        print(f"  ✓ {strategy}: {out.shape}, L2-norm OK")

    # Test gated with gate visualization
    gated = GatedFusion()
    fused, gate_vals = gated.forward_with_gate(sp_desc, dino_desc)
    print(f"  ✓ Gate values: mean={gate_vals.mean():.3f}, std={gate_vals.std():.3f}")
    
    # Test DINOv2 projection
    proj = DINOv2FusionProjection(768, 128)
    raw = torch.randn(B, N, 768)
    projected = proj(raw)
    assert projected.shape == (B, N, 128)
    print(f"  ✓ DINOv2FusionProjection: {raw.shape} → {projected.shape}")

    print("\n✓ All descriptor_fusion.py tests passed.")
