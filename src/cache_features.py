"""
src/cache_features.py

Offline DINOv2 feature caching to HDF5.

Pre-extracts DINOv2 patch features for all training images and saves them to
disk, allowing training without loading DINOv2 on the GPU.

This reduces training VRAM from ~9 GB → ~5 GB (see plan.md §9).

Day 6–7 deliverable.

Usage:
    python -m src.cache_features \
        --image_dir /path/to/megadepth/images \
        --output_file cache/megadepth_dinov2b_features.h5 \
        --variant vitb14_reg \
        --batch_size 8
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dinov2_extractor import build_dinov2_extractor, DINOV2_NATIVE_SIZE

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("WARNING: h5py not installed. Run: pip install h5py")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Image loading utilities
# ---------------------------------------------------------------------------

def load_image_rgb(path: Path, size: int = DINOV2_NATIVE_SIZE) -> Optional[torch.Tensor]:
    """
    Load an image as float32 (3, H, W) in [0, 1], resized to `size × size`.

    Returns None if loading fails.
    """
    try:
        if CV2_AVAILABLE:
            img = cv2.imread(str(path))
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Fallback to PIL
            from PIL import Image
            img = np.array(Image.open(path).convert("RGB"))

        # Resize to DINOv2 native size
        if img.shape[0] != size or img.shape[1] != size:
            if CV2_AVAILABLE:
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            else:
                from PIL import Image
                img_pil = Image.fromarray(img)
                img_pil = img_pil.resize((size, size), Image.BILINEAR)
                img = np.array(img_pil)

        # HWC uint8 → CHW float32 in [0, 1]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor
    except Exception as e:
        print(f"  Warning: Failed to load {path}: {e}")
        return None


def image_key(path: Path, base_dir: Optional[Path] = None) -> str:
    """
    Generate an HDF5 key for an image path.
    Uses the relative path from base_dir, with '/' replaced by '_'.
    """
    if base_dir is not None:
        try:
            rel = path.relative_to(base_dir)
            return str(rel).replace(os.sep, "/")
        except ValueError:
            pass
    return str(path)


def find_images(
    directory: Path,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"),
    max_images: Optional[int] = None,
) -> list[Path]:
    """Recursively find all image files under `directory`."""
    images = []
    for ext in extensions:
        images.extend(directory.rglob(f"*{ext}"))
    images = sorted(images)
    if max_images is not None:
        images = images[:max_images]
    return images


# ---------------------------------------------------------------------------
# Batch generator
# ---------------------------------------------------------------------------

def batch_generator(
    image_paths: list[Path],
    batch_size: int,
    size: int,
    existing_keys: Optional[set] = None,
    base_dir: Optional[Path] = None,
) -> Generator[tuple[list[Path], torch.Tensor], None, None]:
    """
    Yield (paths, batch_tensor) of size (B, 3, size, size).
    Skips images whose keys already exist in the HDF5 file.
    """
    batch_paths: list[Path] = []
    batch_tensors: list[torch.Tensor] = []

    for path in image_paths:
        key = image_key(path, base_dir)
        if existing_keys is not None and key in existing_keys:
            continue

        tensor = load_image_rgb(path, size=size)
        if tensor is None:
            continue

        batch_paths.append(path)
        batch_tensors.append(tensor)

        if len(batch_tensors) == batch_size:
            yield batch_paths, torch.stack(batch_tensors, dim=0)
            batch_paths = []
            batch_tensors = []

    if batch_tensors:
        yield batch_paths, torch.stack(batch_tensors, dim=0)


# ---------------------------------------------------------------------------
# Main caching class
# ---------------------------------------------------------------------------

class DINOv2FeatureCache:
    """
    Extracts DINOv2 patch feature maps for all images in a directory and
    stores them in an HDF5 file.

    HDF5 structure:
        /images/<relative_path>  → dataset of shape (D, h, w) float32
                                   e.g. (768, 37, 37) for vitb14

    Metadata stored as HDF5 attributes:
        /metadata/variant        → e.g. "vitb14_reg"
        /metadata/feat_dim       → 768
        /metadata/patch_size     → 14
        /metadata/image_size     → 518
    """

    def __init__(
        self,
        output_file: Path,
        variant: str = "vitb14_reg",
        device: str = "cuda",
        batch_size: int = 8,
        image_size: int = DINOV2_NATIVE_SIZE,
        compression: Optional[str] = "gzip",
        compression_opts: int = 4,
    ) -> None:
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required. Install with: pip install h5py")

        self.output_file = Path(output_file)
        self.variant = variant
        self.device = device
        self.batch_size = batch_size
        self.image_size = image_size
        self.compression = compression
        self.compression_opts = compression_opts

        # Load DINOv2 extractor
        self.extractor = build_dinov2_extractor(
            variant=variant,
            device=device,
            freeze=True,
        )
        self.feat_dim = self.extractor.get_feat_dim()
        self.extractor.eval()

    def cache_directory(
        self,
        image_dir: Path,
        base_dir: Optional[Path] = None,
        max_images: Optional[int] = None,
        show_progress: bool = True,
    ) -> dict:
        """
        Cache features for all images in `image_dir`.

        Args:
            image_dir:   Root directory containing images.
            base_dir:    Base path for relative key generation. Defaults to image_dir.
            max_images:  Process only first N images (for testing).
            show_progress: Show tqdm progress bar.

        Returns:
            stats: dict with "total", "cached", "skipped", "failed" counts.
        """
        if base_dir is None:
            base_dir = image_dir

        image_paths = find_images(image_dir, max_images=max_images)
        print(f"Found {len(image_paths)} images in {image_dir}")

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        stats = {"total": len(image_paths), "cached": 0, "skipped": 0, "failed": 0}

        with h5py.File(self.output_file, "a") as f:
            # Write metadata
            if "metadata" not in f:
                meta = f.create_group("metadata")
                meta.attrs["variant"] = self.variant
                meta.attrs["feat_dim"] = self.feat_dim
                meta.attrs["patch_size"] = self.extractor.get_patch_size()
                meta.attrs["image_size"] = self.image_size
                meta.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")

            if "images" not in f:
                f.create_group("images")

            # Get already-cached keys
            existing_keys = set(f["images"].keys()) if "images" in f else set()
            stats["skipped"] = len(existing_keys)

            pbar = tqdm(
                total=len(image_paths) - len(existing_keys),
                desc=f"Caching DINOv2 {self.variant}",
                disable=not show_progress,
            )

            for paths, batch in batch_generator(
                image_paths,
                batch_size=self.batch_size,
                size=self.image_size,
                existing_keys=existing_keys,
                base_dir=base_dir,
            ):
                batch = batch.to(self.device)

                with torch.no_grad():
                    features = self.extractor(batch)  # (B, D, h, w)

                features_np = features.cpu().numpy()

                for path, feat in zip(paths, features_np):
                    key = image_key(path, base_dir)
                    try:
                        if self.compression:
                            f["images"].create_dataset(
                                key,
                                data=feat,
                                compression=self.compression,
                                compression_opts=self.compression_opts,
                            )
                        else:
                            f["images"].create_dataset(key, data=feat)
                        stats["cached"] += 1
                    except Exception as e:
                        print(f"\n  Failed to save {key}: {e}")
                        stats["failed"] += 1

                pbar.update(len(paths))

            pbar.close()

        print(f"\nCaching complete: {stats}")
        print(f"Output: {self.output_file} "
              f"({self.output_file.stat().st_size / 1024**3:.2f} GB)")
        return stats

    def cache_image_list(
        self,
        image_paths: list[Path],
        base_dir: Optional[Path] = None,
        show_progress: bool = True,
    ) -> dict:
        """
        Cache features for a specific list of image paths.
        Useful for caching only the MegaDepth training subset.
        """
        stats = {"total": len(image_paths), "cached": 0, "skipped": 0, "failed": 0}

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(self.output_file, "a") as f:
            if "metadata" not in f:
                meta = f.create_group("metadata")
                meta.attrs["variant"] = self.variant
                meta.attrs["feat_dim"] = self.feat_dim
                meta.attrs["patch_size"] = self.extractor.get_patch_size()
                meta.attrs["image_size"] = self.image_size
                meta.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")

            if "images" not in f:
                f.create_group("images")

            existing_keys = set()
            self._collect_keys(f["images"], existing_keys)
            stats["skipped"] = len(existing_keys)

            pbar = tqdm(
                total=len(image_paths) - len(existing_keys),
                desc=f"Caching DINOv2 {self.variant}",
                disable=not show_progress,
            )

            for paths, batch in batch_generator(
                image_paths,
                batch_size=self.batch_size,
                size=self.image_size,
                existing_keys=existing_keys,
                base_dir=base_dir,
            ):
                batch = batch.to(self.device)
                with torch.no_grad():
                    features = self.extractor(batch)
                features_np = features.cpu().numpy()

                for path, feat in zip(paths, features_np):
                    key = image_key(path, base_dir)
                    try:
                        if self.compression:
                            f["images"].create_dataset(
                                key, data=feat,
                                compression=self.compression,
                                compression_opts=self.compression_opts,
                            )
                        else:
                            f["images"].create_dataset(key, data=feat)
                        stats["cached"] += 1
                    except Exception as e:
                        stats["failed"] += 1

                pbar.update(len(paths))

            pbar.close()

        return stats

    @staticmethod
    def load_feature(h5_file: Path, image_key_str: str) -> Optional[np.ndarray]:
        """
        Load cached features for a single image.

        Args:
            h5_file:       Path to HDF5 cache file
            image_key_str: Key string (relative image path)

        Returns:
            features: (D, h, w) float32 numpy array, or None if not found
        """
        with h5py.File(h5_file, "r") as f:
            if image_key_str not in f.get("images", {}):
                return None
            return f["images"][image_key_str][:]

    @staticmethod
    def _collect_keys(group, result_set: set, prefix: str = "") -> None:
        """Recursively collect all dataset keys in an HDF5 group."""
        for key, item in group.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(item, h5py.Dataset):
                result_set.add(full_key)
            elif isinstance(item, h5py.Group):
                DINOv2FeatureCache._collect_keys(item, result_set, full_key)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache DINOv2 features for a dataset of images."
    )
    parser.add_argument("--image_dir", type=Path, required=True,
                        help="Root directory of images to process.")
    parser.add_argument("--output_file", type=Path, required=True,
                        help="Output HDF5 file path.")
    parser.add_argument("--variant", type=str, default="vitb14_reg",
                        help="DINOv2 variant (default: vitb14_reg).")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Images per batch (default: 8).")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Process at most N images (for testing).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda).")
    parser.add_argument("--no_compress", action="store_true",
                        help="Disable gzip compression (faster but larger file).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cache = DINOv2FeatureCache(
        output_file=args.output_file,
        variant=args.variant,
        device=args.device,
        batch_size=args.batch_size,
        compression=None if args.no_compress else "gzip",
    )

    stats = cache.cache_directory(
        image_dir=args.image_dir,
        max_images=args.max_images,
    )

    print(f"\nDone! Stats: {stats}")
