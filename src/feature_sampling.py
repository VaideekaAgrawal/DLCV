"""
src/feature_sampling.py

Bilinear sampling of DINOv2 patch feature maps at sparse keypoint locations.

This is analogous to SuperPoint's `sample_descriptors` function but operates
on a coarse patch-level feature map (stride=14) rather than a dense 8× map.

Day 4 deliverable — see plan.md §4 Design Decision D2
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def sample_descriptors_bilinear(
    keypoints: torch.Tensor,
    feature_map: torch.Tensor,
    image_size: tuple[int, int],
    normalize: bool = True,
) -> torch.Tensor:
    """
    Sample descriptors from a dense feature map at sparse keypoint locations
    using bilinear interpolation.

    This is the core bridging operation: DINOv2 produces features on a
    (H/14 × W/14) grid; we need per-keypoint descriptors at arbitrary locations.

    Args:
        keypoints:   (B, N, 2) float32. Keypoint (x, y) coords in pixel space,
                     where x ∈ [0, W) and y ∈ [0, H). (x is column, y is row)
        feature_map: (B, D, h, w) float32. DINOv2 patch feature map.
                     Typically D=768, h=H/14, w=W/14.
        image_size:  (H, W) — original image dimensions used for normalization.
        normalize:   If True, L2-normalize the output descriptors.

    Returns:
        descriptors: (B, N, D) float32. Per-keypoint feature vectors.

    Notes:
        - Uses F.grid_sample with align_corners=False (consistent with torchvision).
        - Keypoints outside image bounds are clamped to border.
        - This follows the same convention as SuperPoint's sample_descriptors().
    """
    B, N, _ = keypoints.shape
    B_f, D, h, w = feature_map.shape
    H, W = image_size

    assert B == B_f, f"Batch size mismatch: keypoints has {B}, feature_map has {B_f}"

    # Normalize keypoint coordinates to [-1, 1] for grid_sample
    # grid_sample expects (x, y) in [-1, 1] where:
    #   -1 = left/top edge of the leftmost/topmost pixel
    #   +1 = right/bottom edge of the rightmost/bottommost pixel
    # With align_corners=False:
    #   normalized_x = (x + 0.5) / W * 2 - 1
    kp = keypoints.clone()
    kp[..., 0] = (kp[..., 0] + 0.5) / W * 2.0 - 1.0   # x → [-1, 1]
    kp[..., 1] = (kp[..., 1] + 0.5) / H * 2.0 - 1.0   # y → [-1, 1]

    # grid_sample expects grid of shape (B, N, 1, 2) for a list of N points
    # Output will be (B, D, N, 1)
    grid = kp.unsqueeze(2)  # (B, N, 1, 2) — grid_sample expects (B, H_out, W_out, 2)

    # Permute feature_map to ensure correct format: (B, D, h, w) is already correct
    sampled = F.grid_sample(
        feature_map,        # (B, D, h, w)
        grid,               # (B, N, 1, 2)
        mode="bilinear",
        padding_mode="border",   # clamp to border for keypoints near edges
        align_corners=False,
    )
    # sampled: (B, D, N, 1)
    descriptors = sampled.squeeze(-1).permute(0, 2, 1)  # (B, N, D)

    if normalize:
        descriptors = F.normalize(descriptors, p=2, dim=-1)

    return descriptors


def sample_descriptors_bilinear_v2(
    keypoints: torch.Tensor,
    feature_map: torch.Tensor,
    image_size: tuple[int, int],
    normalize: bool = True,
) -> torch.Tensor:
    """
    Alternative implementation using explicit nearest-patch + bilinear blend.
    More transparent about what's happening at patch boundaries.
    Produces identical results to sample_descriptors_bilinear.
    Used for unit testing / verification.
    """
    B, N, _ = keypoints.shape
    _, D, h, w = feature_map.shape
    H, W = image_size

    # Map keypoint pixel coords to patch-level coords
    # Patch (i, j) covers pixels [(i*14, (i+1)*14) × (j*14, (j+1)*14)]
    # Patch center is at ((i+0.5)*14 - 0.5, (j+0.5)*14 - 0.5) in pixel coords
    # Fractional patch coordinate for pixel x: fx = (x + 0.5) / 14 - 0.5
    px = (keypoints[..., 0] + 0.5) / (W / w) - 0.5  # (B, N) fractional patch x
    py = (keypoints[..., 1] + 0.5) / (H / h) - 0.5  # (B, N) fractional patch y

    # Clamp to valid range
    px = px.clamp(0, w - 1)
    py = py.clamp(0, h - 1)

    x0 = px.long().clamp(0, w - 2)
    y0 = py.long().clamp(0, h - 2)
    x1 = (x0 + 1).clamp(0, w - 1)
    y1 = (y0 + 1).clamp(0, h - 1)

    # Bilinear weights
    wx1 = (px - x0.float()).unsqueeze(-1)   # (B, N, 1)
    wy1 = (py - y0.float()).unsqueeze(-1)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    # Gather features at 4 corners: (B, N, D)
    def gather(xi, yi):
        idx = (yi * w + xi).unsqueeze(1).expand(-1, D, -1)  # (B, D, N)
        fm_flat = feature_map.view(B, D, h * w)
        return fm_flat.gather(2, idx).permute(0, 2, 1)      # (B, N, D)

    f00 = gather(x0, y0)
    f10 = gather(x1, y0)
    f01 = gather(x0, y1)
    f11 = gather(x1, y1)

    descriptors = wx0 * wy0 * f00 + wx1 * wy0 * f10 + wx0 * wy1 * f01 + wx1 * wy1 * f11

    if normalize:
        descriptors = F.normalize(descriptors, p=2, dim=-1)

    return descriptors


def preprocess_image_for_dinov2(
    image: torch.Tensor,
    target_size: int = 518,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Resize an image to DINOv2's native resolution (multiple of 14).

    Args:
        image: (B, 3, H, W) or (3, H, W) float32 in [0, 1].
        target_size: Target resolution (must be a multiple of 14). Default: 518.

    Returns:
        resized_image: (B, 3, target_size, target_size)
        original_size: (H, W) original image dimensions
    """
    assert target_size % 14 == 0, f"target_size must be divisible by 14, got {target_size}"

    if image.dim() == 3:
        image = image.unsqueeze(0)

    B, C, H, W = image.shape
    original_size = (H, W)

    resized = F.interpolate(
        image,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    return resized, original_size


def compute_descriptor_similarity(
    desc0: torch.Tensor,
    desc1: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between two sets of descriptors.

    Args:
        desc0: (B, N, D) L2-normalized descriptors from image 0
        desc1: (B, M, D) L2-normalized descriptors from image 1

    Returns:
        sim: (B, N, M) cosine similarity matrix
    """
    # Assumes L2-normalized inputs → dot product = cosine similarity
    return torch.bmm(desc0, desc1.transpose(1, 2))


def mutual_nearest_neighbor_matching(
    desc0: torch.Tensor,
    desc1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform mutual nearest neighbor (MNN) matching between two descriptor sets.

    Args:
        desc0: (N, D) float32 descriptors (L2-normalized)
        desc1: (M, D) float32 descriptors (L2-normalized)

    Returns:
        matches0: (K,) indices into desc0 for the K mutual matches
        matches1: (K,) indices into desc1 for the K mutual matches
    """
    sim = torch.mm(desc0, desc1.t())   # (N, M)

    # Nearest neighbor in desc1 for each desc0
    nn0 = sim.argmax(dim=1)  # (N,) indices into desc1
    # Nearest neighbor in desc0 for each desc1
    nn1 = sim.argmax(dim=0)  # (M,) indices into desc0

    # Mutual check: i → j AND j → i
    ids0 = torch.arange(len(desc0), device=desc0.device)
    mask = nn1[nn0] == ids0

    matches0 = ids0[mask]
    matches1 = nn0[mask]

    return matches0, matches1


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    B, N, D, h, w, H, W = 2, 512, 768, 37, 37, 518, 518
    kpts = torch.rand(B, N, 2)
    kpts[..., 0] *= W
    kpts[..., 1] *= H
    feat_map = torch.randn(B, D, h, w)

    desc = sample_descriptors_bilinear(kpts, feat_map, (H, W), normalize=True)
    desc2 = sample_descriptors_bilinear_v2(kpts, feat_map, (H, W), normalize=True)

    print(f"Descriptors shape: {desc.shape}")  # (2, 512, 768)
    assert desc.shape == (B, N, D)

    # Verify L2 normalization
    norms = desc.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Not unit normalized!"

    # Verify both implementations agree (within float32 precision)
    max_diff = (desc - desc2).abs().max().item()
    print(f"Max diff between implementations: {max_diff:.6f}")
    assert max_diff < 1e-4, f"Implementations disagree: max_diff={max_diff}"

    print("✓ feature_sampling tests passed.")
