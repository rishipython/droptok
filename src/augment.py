"""Data augmentation pipelines.

Two presets:

- ``build_eval_transform``: deterministic resize + center crop +
  DINO-stat normalisation. Used for val/test and for the cache builder.

- ``build_train_transform``: SimCLR-style weak augmentation aimed at
  breaking positional memorisation without destroying the content the
  DINO teacher can read. Specifically: random resized crop,
  horizontal flip, moderate color jitter, and occasional grayscale.
  Strong augmentations (heavy blur, solarize, cutout) are skipped
  because the goal is to preserve teacher-feature semantics while
  varying the per-position distribution enough to break the
  "decoder memorises the dataset mean" shortcut.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torchvision import transforms as T

# DINOv2 preprocessing stats (identical to CLIP/ImageNet-ish).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_eval_transform(image_size: int = 224,
                         mean: Tuple[float, float, float] = IMAGENET_MEAN,
                         std: Tuple[float, float, float] = IMAGENET_STD):
    """Deterministic resize + center crop."""
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def build_train_transform(image_size: int = 224,
                          mean: Tuple[float, float, float] = IMAGENET_MEAN,
                          std: Tuple[float, float, float] = IMAGENET_STD,
                          strength: float = 1.0):
    """SimCLR-lite: random resized crop, hflip, color jitter, small
    grayscale probability. ``strength`` scales the color-jitter magnitude.
    """
    cj = T.ColorJitter(
        brightness=0.4 * strength,
        contrast=0.4 * strength,
        saturation=0.4 * strength,
        hue=0.1 * strength,
    )
    return T.Compose([
        T.RandomResizedCrop(
            image_size, scale=(0.5, 1.0),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([cj], p=0.8),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
