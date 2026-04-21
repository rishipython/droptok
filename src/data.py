"""Datasets and per-patch masking utilities.

Two dataset modes are supported:

1. ``CachedDataset`` -- reads precomputed ``(image, teacher_feats)``
   blobs produced by ``scripts/precompute_teacher.py``. Fastest and
   deterministic but applies no augmentation: DINO targets were
   computed once for a fixed center crop.

2. ``ImageFolderDataset`` -- reads raw images from a directory tree,
   applies augmentation, and lets the training loop run the DINOv2
   teacher on-the-fly. This is required when the experiment relies on
   data augmentation (SimCLR-style or otherwise) because the reconstruction
   target MUST match the augmented view.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #

def make_patch_mask(B: int, H: int, W: int, ratio: float, device) -> torch.Tensor:
    """Independent Bernoulli mask per patch, per image. True = MASKED."""
    return torch.rand(B, H, W, device=device) < ratio


# --------------------------------------------------------------------------- #
# Cached dataset (precomputed DINO features)
# --------------------------------------------------------------------------- #

class CachedDataset(Dataset):
    """Each sample on disk is a ``.pt`` dict with keys
    ``image`` (3,H,W, normalised fp16) and ``teacher_feats`` (H,W,D fp16).
    """

    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root) / split
        self.files: List[Path] = sorted(self.root.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No cached samples under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        blob = torch.load(self.files[idx], weights_only=False, map_location="cpu")
        return {
            "image": blob["image"].float(),
            "teacher_feats": blob["teacher_feats"].float(),
            "id": self.files[idx].stem,
        }


# --------------------------------------------------------------------------- #
# Raw-image dataset (for live teacher + augmentation)
# --------------------------------------------------------------------------- #

class ImageFolderDataset(Dataset):
    """Read raw images from a directory and apply a given torchvision
    transform. Caller is responsible for making the transform output
    shape (3, image_size, image_size) normalised using the DINO stats
    (see ``augment.build_train_transform`` / ``build_eval_transform``).
    The training loop then runs the frozen teacher on the returned
    tensors to compute matching DINO targets.
    """

    def __init__(self, root: str | Path, transform, extensions: Tuple[str, ...] =
                 (".jpg", ".jpeg", ".png", ".webp")):
        self.root = Path(root)
        self.transform = transform
        self.files = sorted(
            p for p in self.root.rglob("*")
            if p.suffix.lower() in extensions and p.is_file()
        )
        if not self.files:
            raise FileNotFoundError(f"No images under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        from PIL import Image
        pil = Image.open(self.files[idx]).convert("RGB")
        img = self.transform(pil)       # (3, H, W) normalised tensor
        return {"image": img, "id": self.files[idx].stem}
