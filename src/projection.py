"""
src/projection.py

Projection network variants for mapping DINOv2 features (768-dim) to the
descriptor space used by LightGlue (256-dim).

Three variants for ablation study:
  - LinearProjection:  Linear(768 → 256)
  - MLP1Projection:    Linear(768 → 512) → ReLU → Linear(512 → 256)
  - MLP2Projection:    Linear(768 → 512) → LayerNorm → GELU → Linear(512 → 256) → L2Norm
                       [Primary — see plan.md §4 Design Decision D3]

Day 3 deliverable / Phase 2 ablation (Experiment 2.2)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Projection variants
# ---------------------------------------------------------------------------

class LinearProjection(nn.Module):
    """
    Simplest possible projection: single linear layer.
    Used as the baseline ablation.
    Trainable params: ~196K (768×256 + 256 bias)
    """

    def __init__(self, input_dim: int = 768, output_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=True)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., input_dim) — arbitrary leading dims
        Returns:
            out: (..., output_dim) L2-normalized
        """
        out = self.proj(x)
        return F.normalize(out, p=2, dim=-1)

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def extra_repr(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, params={n:,}"


class MLP1Projection(nn.Module):
    """
    Two-layer MLP with ReLU.
    Trainable params: ~524K
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return F.normalize(out, p=2, dim=-1)

    def _init_weights(self) -> None:
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def extra_repr(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, params={n:,}"


class MLP2Projection(nn.Module):
    """
    Primary projection network (plan.md §4 D3 Variant MLP-2).
    Two-layer MLP with LayerNorm + GELU + L2Norm output.
    Empirically best quality-efficiency trade-off.
    Trainable params: ~525K
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., input_dim)
        Returns:
            out: (..., output_dim) L2-normalized
        """
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=-1)

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def extra_repr(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}, params={n:,}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

ProjectionType = Literal["linear", "mlp1", "mlp2"]

_PROJECTION_CLASSES = {
    "linear": LinearProjection,
    "mlp1": MLP1Projection,
    "mlp2": MLP2Projection,
}


def build_projection(
    proj_type: ProjectionType = "mlp2",
    input_dim: int = 768,
    output_dim: int = 256,
    hidden_dim: int = 512,
) -> nn.Module:
    """
    Build a projection network by name.

    Args:
        proj_type:  "linear" | "mlp1" | "mlp2"
        input_dim:  Input feature dimension (DINOv2 output dim)
        output_dim: Output descriptor dimension (LightGlue input dim)
        hidden_dim: Hidden layer dimension (only for mlp1/mlp2)

    Returns:
        Projection nn.Module (randomly initialized)
    """
    if proj_type not in _PROJECTION_CLASSES:
        raise ValueError(f"Unknown projection type '{proj_type}'. "
                         f"Choose from {list(_PROJECTION_CLASSES.keys())}")

    cls = _PROJECTION_CLASSES[proj_type]
    if proj_type == "linear":
        proj = cls(input_dim=input_dim, output_dim=output_dim)
    else:
        proj = cls(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    n_params = sum(p.numel() for p in proj.parameters())
    print(f"Built '{proj_type}' projection: {input_dim}→{output_dim}  "
          f"({n_params:,} trainable params)")
    return proj


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, N, D_in, D_out = 2, 512, 768, 256

    x = torch.randn(B, N, D_in)

    for ptype in ["linear", "mlp1", "mlp2"]:
        proj = build_projection(proj_type=ptype, input_dim=D_in, output_dim=D_out)
        out = proj(x)
        assert out.shape == (B, N, D_out), f"Shape mismatch for {ptype}"
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"{ptype}: output not unit-normalized!"
        print(f"  ✓ {ptype}: output shape {out.shape}")

    print("✓ projection.py tests passed.")
