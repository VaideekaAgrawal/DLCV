"""
src/evaluate.py

Evaluation utilities for the DINOv2+LightGlue pipeline.

Wraps glue-factory's evaluation functions and adds:
- MNN (mutual nearest neighbor) baseline evaluation
- Per-pair metrics for analysis
- AUC computation

Used in Experiments 1.1, 1.2, and all Phase 3 evaluations.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# AUC computation
# ---------------------------------------------------------------------------

def compute_auc(errors: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    """
    Compute AUC at multiple thresholds.

    Args:
        errors:     (N,) array of pose errors (degrees) or pixel errors
        thresholds: List of threshold values, e.g. [5, 10, 20] for pose

    Returns:
        dict: {"auc@5": float, "auc@10": float, ...}
    """
    sort_idx = np.argsort(errors)
    errors_sorted = errors[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)

    results = {}
    for thresh in thresholds:
        last_ind = np.searchsorted(errors_sorted, thresh)
        y = np.zeros(last_ind + 1)
        x = np.zeros(last_ind + 1)
        y[1:] = recall[:last_ind]
        x[1:] = errors_sorted[:last_ind]
        # Recall at threshold
        recall_at_thresh = last_ind / len(errors)
        # Area under curve (normalized to [0, thresh])
        # np.trapezoid is the NumPy 2.x name (trapz removed in 2.0)
        trapz_fn = getattr(np, "trapezoid", np.trapz) if hasattr(np, "trapz") else np.trapezoid
        auc_val = trapz_fn(y, x) / thresh if thresh > 0 else 0.0
        results[f"auc@{thresh}"] = float(auc_val * 100)
        results[f"recall@{thresh}"] = float(recall_at_thresh * 100)

    return results


def compute_pose_auc(
    pose_errors: np.ndarray,
    thresholds: list[float] = [5.0, 10.0, 20.0],
) -> dict[str, float]:
    """Compute pose AUC (standard for MegaDepth-1500, ScanNet-1500)."""
    return compute_auc(pose_errors, thresholds)


def compute_homography_auc(
    corner_errors: np.ndarray,
    thresholds: list[float] = [1.0, 3.0, 5.0],
) -> dict[str, float]:
    """Compute homography AUC (standard for HPatches)."""
    return compute_auc(corner_errors, thresholds)


# ---------------------------------------------------------------------------
# Descriptor matching quality
# ---------------------------------------------------------------------------

def evaluate_descriptor_matching(
    desc0: torch.Tensor,
    desc1: torch.Tensor,
    gt_matches: torch.Tensor,
    matching_strategy: str = "mnn",
    ratio_threshold: float = 0.9,
) -> dict[str, float]:
    """
    Evaluate descriptor quality via nearest-neighbor matching.

    Args:
        desc0:      (N, D) float32, L2-normalized descriptors from image 0
        desc1:      (M, D) float32, L2-normalized descriptors from image 1
        gt_matches: (N,) long, gt_matches[i] = j if kpt_i matches kpt_j else -1
        matching_strategy: "mnn" | "nn" | "ratio"
        ratio_threshold: Lowe's ratio test threshold (only for "ratio")

    Returns:
        dict with "precision", "recall", "num_matches", "num_gt_matches"
    """
    desc0 = desc0.float()
    desc1 = desc1.float()

    # Cosine similarity
    sim = torch.mm(desc0, desc1.t())  # (N, M)

    if matching_strategy == "mnn":
        matches0, matches1 = _mnn(sim)
    elif matching_strategy == "nn":
        matches0 = torch.arange(len(desc0))
        matches1 = sim.argmax(dim=1)
    elif matching_strategy == "ratio":
        matches0, matches1 = _ratio_test(sim, ratio_threshold)
    else:
        raise ValueError(f"Unknown matching strategy: {matching_strategy}")

    # Compute precision and recall
    gt_valid = gt_matches >= 0
    num_gt = gt_valid.sum().item()

    if len(matches0) == 0:
        return {
            "precision": 0.0, "recall": 0.0,
            "num_matches": 0, "num_gt_matches": num_gt,
        }

    # Check which matches are correct
    pred_gt = gt_matches[matches0]   # GT match index for each predicted match source
    correct = (pred_gt == matches1).sum().item()

    precision = correct / len(matches0) if len(matches0) > 0 else 0.0
    recall = correct / num_gt if num_gt > 0 else 0.0

    return {
        "precision": float(precision * 100),
        "recall": float(recall * 100),
        "num_matches": len(matches0),
        "num_gt_matches": num_gt,
    }


def _mnn(sim: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mutual nearest neighbor matching from similarity matrix."""
    nn0 = sim.argmax(dim=1)   # (N,) best match in img1 for each kpt in img0
    nn1 = sim.argmax(dim=0)   # (M,) best match in img0 for each kpt in img1
    ids0 = torch.arange(len(nn0), device=sim.device)
    mask = nn1[nn0] == ids0
    return ids0[mask], nn0[mask]


def _ratio_test(
    sim: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lowe's ratio test."""
    # Sort similarities descending for each query
    topk = sim.topk(min(2, sim.shape[1]), dim=1)
    ratios = topk.values[:, 1] / (topk.values[:, 0] + 1e-8)
    keep = ratios < threshold
    ids0 = torch.where(keep)[0]
    ids1 = topk.indices[keep, 0]
    return ids0, ids1


# ---------------------------------------------------------------------------
# GlueFactory-based evaluation runners
# ---------------------------------------------------------------------------

def run_megadepth1500_eval(
    config_name: str,
    overwrite: bool = False,
    output_dir: Optional[Path] = None,
) -> dict[str, float]:
    """
    Run MegaDepth-1500 evaluation using glue-factory.

    Args:
        config_name: glue-factory config name (e.g., "superpoint+lightglue-official")
        overwrite:   Recompute even if results exist
        output_dir:  Where to save results (default: glue-factory output dir)

    Returns:
        results: dict with AUC@5/10/20 and other metrics
    """
    try:
        import subprocess
        import json

        cmd = [
            sys.executable, "-m", "gluefactory.eval.megadepth1500",
            "--conf", config_name,
        ]
        if overwrite:
            cmd.append("--overwrite")

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(Path(__file__).parent.parent / "glue-factory"))
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr[-1000:]}")
            raise RuntimeError(f"Evaluation failed with code {result.returncode}")

        # Parse AUC from stdout
        metrics = _parse_gluefactory_output(result.stdout)
        return metrics
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return {}


def _parse_gluefactory_output(stdout: str) -> dict[str, float]:
    """Parse AUC numbers from glue-factory stdout."""
    import re
    results = {}
    # Look for patterns like "AUC@5 : 66.8" or "auc@5: 66.8"
    patterns = [
        r"(?:AUC|auc)@(\d+)[°\s]*[:\s]+([0-9.]+)",
        r"(\d+)°\s+[:\s]+([0-9.]+)",
    ]
    for pat in patterns:
        for match in re.finditer(pat, stdout, re.IGNORECASE):
            key = f"auc@{match.group(1)}"
            results[key] = float(match.group(2))
    return results


# ---------------------------------------------------------------------------
# VRAM monitoring
# ---------------------------------------------------------------------------

class VRAMMonitor:
    """Context manager that reports peak VRAM usage."""

    def __init__(self, label: str = ""):
        self.label = label
        self.peak_mb = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            self.peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  [{self.label}] Peak VRAM: {self.peak_mb:.1f} MB")

    def report(self) -> float:
        return self.peak_mb


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

class Timer:
    """Simple timing context manager."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.t0
        print(f"  [{self.label}] Time: {self.elapsed*1000:.1f} ms")


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test AUC computation
    rng = np.random.default_rng(42)
    errors = rng.uniform(0, 30, size=1000).astype(np.float32)
    auc = compute_pose_auc(errors, thresholds=[5, 10, 20])
    print("AUC test:", auc)
    assert all(0 <= v <= 100 for v in auc.values())

    # Test descriptor evaluation
    N, M, D = 200, 300, 256
    desc0 = F.normalize(torch.randn(N, D), p=2, dim=-1)
    desc1 = F.normalize(torch.randn(M, D), p=2, dim=-1)
    gt = torch.full((N,), -1, dtype=torch.long)
    gt[:50] = torch.arange(50)  # 50 ground truth matches

    metrics = evaluate_descriptor_matching(desc0, desc1, gt)
    print("Descriptor matching metrics:", metrics)

    print("✓ evaluate.py tests passed.")
