"""
src/utils/viz.py

Visualization utilities for feature matching experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: Optional[np.ndarray] = None,
    color: tuple = (0, 255, 0),
    radius: int = 3,
    title: str = "",
) -> np.ndarray:
    """
    Draw keypoints on an image (numpy uint8 HWC).

    Args:
        image:     (H, W, 3) uint8 RGB image
        keypoints: (N, 2) float array of (x, y) keypoint coordinates
        scores:    (N,) float array of keypoint scores (used for color mapping)
        color:     Default keypoint color (R, G, B) if scores is None
        radius:    Circle radius in pixels

    Returns:
        vis: (H, W, 3) uint8 with keypoints drawn
    """
    import cv2
    vis = image.copy()
    if scores is not None:
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)

    for i, (x, y) in enumerate(keypoints):
        if scores is not None:
            c = plt.cm.plasma(scores_norm[i])[:3]
            c = tuple(int(v * 255) for v in c[::-1])  # RGB → BGR for cv2
        else:
            c = (color[2], color[1], color[0])  # BGR for cv2
        cv2.circle(vis, (int(x), int(y)), radius, c, -1, cv2.LINE_AA)

    if title:
        cv2.putText(vis, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def draw_matches(
    image0: np.ndarray,
    image1: np.ndarray,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    matches: np.ndarray,
    correct_mask: Optional[np.ndarray] = None,
    max_display: int = 200,
    title: str = "",
) -> np.ndarray:
    """
    Draw matching lines between two images side by side.

    Args:
        image0, image1: (H, W, 3) uint8 RGB
        kpts0, kpts1:   (N, 2), (M, 2) float arrays of keypoint coordinates
        matches:        (K, 2) int array of [idx0, idx1] match pairs
        correct_mask:   (K,) bool array, True = correct match (green), False = wrong (red)
        max_display:    Limit displayed matches for clarity

    Returns:
        canvas: (H, W0+W1, 3) uint8
    """
    import cv2
    H0, W0 = image0.shape[:2]
    H1, W1 = image1.shape[:2]
    H = max(H0, H1)

    canvas = np.zeros((H, W0 + W1, 3), dtype=np.uint8)
    canvas[:H0, :W0] = image0
    canvas[:H1, W0:] = image1

    if len(matches) == 0:
        return canvas

    # Subsample if too many matches
    if len(matches) > max_display:
        idx = np.random.choice(len(matches), max_display, replace=False)
        matches = matches[idx]
        if correct_mask is not None:
            correct_mask = correct_mask[idx]

    for i, (m0, m1) in enumerate(matches):
        x0, y0 = int(kpts0[m0, 0]), int(kpts0[m0, 1])
        x1, y1 = int(kpts1[m1, 0]) + W0, int(kpts1[m1, 1])

        if correct_mask is not None:
            color = (0, 255, 0) if correct_mask[i] else (0, 0, 255)  # green/red
        else:
            color = (255, 200, 0)  # gold

        cv2.line(canvas, (x0, y0), (x1, y1), color[::-1], 1, cv2.LINE_AA)
        cv2.circle(canvas, (x0, y0), 3, color[::-1], -1)
        cv2.circle(canvas, (x1, y1), 3, color[::-1], -1)

    if title:
        cv2.putText(canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def plot_descriptor_tsne(
    desc_sp: np.ndarray,
    desc_dino: np.ndarray,
    labels: Optional[np.ndarray] = None,
    n_samples: int = 2000,
    title: str = "SuperPoint vs DINOv2 Descriptor Space",
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot t-SNE comparison of SuperPoint and DINOv2 descriptor distributions.

    Args:
        desc_sp:   (N, 256) SuperPoint descriptors
        desc_dino: (N, 256) DINOv2 projected descriptors
        labels:    (N,) optional semantic labels for color coding
        n_samples: Number of points to sample for t-SNE
        save_path: If given, save figure to this path
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn required for t-SNE. Run: pip install scikit-learn")
        return

    N = min(n_samples, len(desc_sp))
    idx = np.random.choice(len(desc_sp), N, replace=False)

    desc_all = np.concatenate([desc_sp[idx], desc_dino[idx]], axis=0)
    source_labels = np.array([0] * N + [1] * N)

    print(f"Running t-SNE on {2*N} points...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    emb = tsne.fit_transform(desc_all)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, src, name in zip(axes, [0, 1], ["SuperPoint", "DINOv2"]):
        mask = source_labels == src
        if labels is not None:
            scatter = ax.scatter(emb[mask, 0], emb[mask, 1],
                                 c=labels[idx], cmap="tab20", s=2, alpha=0.7)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(emb[mask, 0], emb[mask, 1], s=2, alpha=0.5)
        ax.set_title(f"{name} descriptors (t-SNE)")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved t-SNE plot to {save_path}")
    else:
        plt.show()


def plot_match_statistics(
    stats_dict: dict,
    title: str = "Matching Statistics",
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot bar charts comparing matching statistics across methods.

    Args:
        stats_dict: {"method_name": {"precision": float, "recall": float, ...}}
    """
    methods = list(stats_dict.keys())
    metrics = ["precision", "recall", "num_matches"]
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    for ax, metric in zip(axes, metrics):
        values = [stats_dict[m].get(metric, 0) for m in methods]
        bars = ax.bar(methods, values, color=colors[:len(methods)])
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel(metric)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticklabels(methods, rotation=15, ha="right")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
