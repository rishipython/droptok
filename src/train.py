"""Training loop for DropTok.

Supports two input modes:

- ``mode="cached"``  : reads precomputed DINO features from a cache
  directory built by ``scripts/precompute_teacher.py``. No
  augmentation. Fastest.

- ``mode="live"``    : reads raw images, applies the augmentation
  transform, and runs the frozen DINOv2 teacher on each batch. Slower
  (one extra teacher forward per batch) but lets augmentation act on
  both the student input and the reconstruction target.

Outputs (written to ``out_dir``):

- ``summary.json``  -- metrics + config
- ``history.json``  -- per-epoch training/val metrics
- ``best.pt``       -- best-val checkpoint (state_dict + cfg + summary)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import CachedDataset, ImageFolderDataset, make_patch_mask
from .model import DropTok, DropTokConfig
from .teacher import TeacherConfig, extract_from_tensor, load as load_teacher
from .utils import seed_worker, set_seed


# --------------------------------------------------------------------------- #
# Config builder
# --------------------------------------------------------------------------- #

def build_config(base: Dict[str, Any]) -> DropTokConfig:
    """Build a ``DropTokConfig`` from a plain dict config, using only
    the keys the dataclass understands."""
    allowed = set(DropTokConfig.__dataclass_fields__.keys())
    kwargs = {k: v for k, v in base.items() if k in allowed}
    # allow passing a list/tuple for drop_after_layers from YAML
    if "drop_after_layers" in kwargs and kwargs["drop_after_layers"] is not None:
        kwargs["drop_after_layers"] = tuple(kwargs["drop_after_layers"])
    return DropTokConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #

def run(cfg: DropTokConfig,
        out_dir: str | Path,
        *,
        mode: str = "cached",
        cache_root: Optional[str] = None,
        image_root: Optional[str] = None,
        val_split: str = "val",
        test_split: str = "test",
        train_split: str = "train",
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 2e-4,
        weight_decay: float = 0.05,
        grad_clip: float = 1.0,
        num_workers: int = 4,
        seed: int = 7,
        strength: float = 1.0,
        device: Optional[str] = None) -> Dict[str, Any]:
    """Train once with the given config.

    When ``mode="live"``, ``image_root`` must point at a directory
    containing ``train/``, ``val/``, ``test/`` subfolders of raw images.
    When ``mode="cached"``, ``cache_root`` must point at a directory
    containing cached ``.pt`` blobs in the same split layout.
    """
    assert mode in {"cached", "live"}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    # --------------- data --------------- #
    if mode == "cached":
        assert cache_root, "cache_root required for mode='cached'"
        train_ds = CachedDataset(cache_root, train_split)
        val_ds = CachedDataset(cache_root, val_split)
        test_ds = CachedDataset(cache_root, test_split)
    else:
        from .augment import build_train_transform, build_eval_transform
        assert image_root, "image_root required for mode='live'"
        root = Path(image_root)
        train_tf = build_train_transform(cfg.image_size, strength=strength)
        eval_tf = build_eval_transform(cfg.image_size)
        train_ds = ImageFolderDataset(root / train_split, train_tf)
        val_ds = ImageFolderDataset(root / val_split, eval_tf)
        test_ds = ImageFolderDataset(root / test_split, eval_tf)

    def loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True, worker_init_fn=seed_worker,
        )

    train_loader = loader(train_ds, True)
    val_loader = loader(val_ds, False)
    test_loader = loader(test_ds, False)

    # --------------- model + optim --------------- #
    model = DropTok(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[droptok {cfg.method} K={cfg.K_total}] "
          f"params: {n_params / 1e6:.2f}M  "
          f"drop_sched={getattr(model, 'drop_sched', None)}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --------------- live-teacher setup --------------- #
    if mode == "live":
        teacher_cfg = TeacherConfig(
            image_size=cfg.image_size,
            grid_h=cfg.grid_h, grid_w=cfg.grid_w,
            teacher_dim=cfg.d_teacher,
        )
        load_teacher(teacher_cfg.model_name, device)

    is_pixel = cfg.recon_target == "pixel"

    def err_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if is_pixel:
            return ((pred.float() - target.float()) ** 2).mean(dim=-1)
        return 1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1)

    def prep_batch(batch):
        """Return (teacher, img) tensors for a batch, running the live
        teacher when needed. teacher: (B,H,W,D). img: (B,3,h,w) or None.
        """
        if mode == "cached":
            teacher = batch["teacher_feats"].to(device, non_blocking=True)
            img = (batch["image"].to(device, non_blocking=True)
                   if is_pixel else None)
            return teacher, img

        img = batch["image"].to(device, non_blocking=True)
        with torch.no_grad():
            teacher = extract_from_tensor(img, teacher_cfg).to(device)
        return teacher, (img if is_pixel else None)

    # --------------- train loop --------------- #
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        K_soft_sum = 0.0
        K_hard_sum = 0.0
        K_batches = 0

        for batch in train_loader:
            teacher, img = prep_batch(batch)
            B, H, W, _ = teacher.shape
            patch_mask = make_patch_mask(B, H, W, cfg.mask_ratio, device)
            loss, out, _ = model.compute_loss(teacher, patch_mask, img=img)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            total_loss += float(loss.item()) * B
            n_seen += B
            if cfg.method == "gated":
                K_soft_sum += float(out["n_surviving_soft"].item())
                K_hard_sum += float(out["n_surviving_hard"].item())
                K_batches += 1

        train_loss = total_loss / max(n_seen, 1)
        ep_soft = K_soft_sum / max(K_batches, 1)
        ep_hard = K_hard_sum / max(K_batches, 1)

        model.eval()
        v_sum, v_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                teacher, img = prep_batch(batch)
                B, H, W, _ = teacher.shape
                patch_mask = make_patch_mask(B, H, W, cfg.mask_ratio, device)
                _, out, target = model.compute_loss(teacher, patch_mask, img=img)
                if out["pred"].numel() == 0:
                    continue
                e = err_fn(out["pred"], target)
                v_sum += float(e.sum().item())
                v_n += e.numel()
        val_err = v_sum / max(v_n, 1)

        row = {"epoch": epoch, "train_loss": train_loss, "val_err": val_err}
        if cfg.method == "gated":
            row["train_soft_K"] = ep_soft
            row["train_hard_K"] = ep_hard
        history.append(row)
        extra = (f"  soft_K={ep_soft:.1f}  hard_K={ep_hard:.1f}"
                 if cfg.method == "gated" else "")
        print(f"  epoch {epoch}/{epochs}  train={train_loss:.4f}  "
              f"val_err={val_err:.4f}{extra}")

        if val_err < best_val:
            best_val = val_err
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # --------------- test --------------- #
    model.eval()
    t_sum, t_n = 0.0, 0
    tok_sum, img_n = 0.0, 0
    hardK_per_image: list = []
    with torch.no_grad():
        for batch in test_loader:
            teacher, img = prep_batch(batch)
            B, H, W, _ = teacher.shape
            patch_mask = make_patch_mask(B, H, W, cfg.mask_ratio, device)
            _, out, target = model.compute_loss(teacher, patch_mask, img=img)
            if out["pred"].numel() == 0:
                continue
            e = err_fn(out["pred"], target)
            t_sum += float(e.sum().item())
            t_n += e.numel()
            if cfg.method == "gated":
                per = out["n_surviving_per_image"].cpu().tolist()
                hardK_per_image.extend(per)
                tok_sum += sum(per)
            else:
                tok_sum += float(out["n_surviving_hard"].item()) * B
            img_n += B

    train_minutes = (time.time() - t0) / 60.0
    summary: Dict[str, Any] = {
        "method": cfg.method,
        "K_total": cfg.K_total,
        "mask_ratio": cfg.mask_ratio,
        "recon_target": cfg.recon_target,
        "mode": mode,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "n_params_M": n_params / 1e6,
        "best_val_err": best_val,
        "test_err_mean": t_sum / max(t_n, 1),
        "test_metric": ("cosine_err" if cfg.recon_target == "dino"
                        else "pixel_mse"),
        "avg_tokens": tok_sum / max(img_n, 1),
        "train_minutes": train_minutes,
        "drop_schedule": getattr(model, "drop_sched", None),
    }
    if cfg.method == "gated" and hardK_per_image:
        import statistics as _stats
        summary.update({
            "lambda_comp": cfg.lambda_comp,
            "gate_alpha": F.softplus(model.gate_alpha_raw).detach().cpu().tolist(),
            "gate_tau": model.gate_tau.detach().cpu().tolist(),
            "natural_K_mean": _stats.fmean(hardK_per_image),
            "natural_K_median": _stats.median(hardK_per_image),
            "natural_K_min": min(hardK_per_image),
            "natural_K_max": max(hardK_per_image),
        })

    torch.save({"state_dict": best_state, "cfg": asdict(cfg),
                "summary": summary}, out_dir / "best.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[droptok {cfg.method} K={cfg.K_total}] DONE  "
          f"test_err={summary['test_err_mean']:.4f}  "
          f"avg_tokens={summary['avg_tokens']:.2f}  "
          f"train_min={train_minutes:.1f}")
    return summary
