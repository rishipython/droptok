"""Thin wrapper around a DINOv2 teacher.

The teacher is frozen and used to

1. pre-compute patch features once per image (`precompute_teacher.py`
   CLI), stored as float16 blobs for fast, deterministic training, or
2. run on-the-fly in the training loop so that augmented image views
   produce matched augmented DINO targets (`extract_batch_tensor`). The
   live path is essential when using SimCLR-style augmentations because
   a cached static target cannot change view-by-view.

We default to ``facebook/dinov2-base`` (ungated). At 224x224 with
patch_size=14 this gives a 16x16 grid of 768-d patch tokens (CLS
dropped).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from PIL import Image


@dataclass
class TeacherConfig:
    model_name: str = "facebook/dinov2-base"
    image_size: int = 224
    grid_h: int = 16
    grid_w: int = 16
    teacher_dim: int = 768


# module-level cache so multiple callers in one process share one model
_model = None
_processor = None
_device = None
_image_mean: Optional[torch.Tensor] = None
_image_std: Optional[torch.Tensor] = None


def load(model_name: str = "facebook/dinov2-base", device: str = "cuda"):
    """Load (and cache) the teacher. Returns (model, processor)."""
    global _model, _processor, _device, _image_mean, _image_std
    if _model is not None and _device == device and _processor is not None:
        return _model, _processor
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(model_name)
    try:
        model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
    except (TypeError, ValueError):
        model = AutoModel.from_pretrained(model_name)
    model = model.to(device).eval()
    _model, _processor, _device = model, processor, device
    _image_mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
    _image_std = torch.tensor(processor.image_std).view(1, 3, 1, 1)
    return model, processor


def image_normalizer(device: str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (mean, std) broadcastable to (B, 3, H, W) on the requested
    device. Loads the teacher if needed."""
    if _image_mean is None:
        load(device=device)
    return _image_mean.to(device), _image_std.to(device)


def _num_register_tokens(model) -> int:
    cfg = model.config
    for attr in ("num_register_tokens", "register_tokens"):
        if hasattr(cfg, attr):
            v = getattr(cfg, attr)
            if isinstance(v, int):
                return v
    return 0


@torch.inference_mode()
def extract_from_pils(pil_imgs: List[Image.Image], cfg: TeacherConfig,
                      device: str = "cuda") -> torch.Tensor:
    """Run the teacher on a list of PIL images (cache builder path).
    Returns a (B, H, W, D) float16 patch-feature tensor on CPU."""
    model, processor = load(cfg.model_name, device)
    inputs = processor(
        images=pil_imgs,
        return_tensors="pt",
        size={"shortest_edge": cfg.image_size},
        do_center_crop=True,
        crop_size={"height": cfg.image_size, "width": cfg.image_size},
    )
    pixel_values = inputs["pixel_values"].to(device)
    out = model(pixel_values=pixel_values, return_dict=True)
    hs = out.last_hidden_state              # (B, 1+R+N, D)
    n_reg = _num_register_tokens(model)
    patch = hs[:, 1 + n_reg :, :]
    B = patch.shape[0]
    patch = patch.reshape(B, cfg.grid_h, cfg.grid_w, cfg.teacher_dim).contiguous()
    return patch.half().cpu()


@torch.inference_mode()
def extract_from_tensor(pixel_values: torch.Tensor, cfg: TeacherConfig
                        ) -> torch.Tensor:
    """Run the teacher on an ALREADY-NORMALISED image tensor
    (B, 3, image_size, image_size). Returns (B, H, W, D) on the same
    device in float32. Used on-the-fly during training so that augmented
    pixel views produce matching augmented DINO targets.
    """
    model, _ = load(cfg.model_name, pixel_values.device.type)
    out = model(pixel_values=pixel_values, return_dict=True)
    hs = out.last_hidden_state
    n_reg = _num_register_tokens(model)
    patch = hs[:, 1 + n_reg :, :]
    B = patch.shape[0]
    return patch.reshape(B, cfg.grid_h, cfg.grid_w, cfg.teacher_dim).contiguous()
