"""Run the DINOv2 teacher over every split and cache
(image, teacher_feats) pairs as float16 ``.pt`` blobs.

One blob per image so that small random-access data loaders can stream
efficiently without memmapping a giant tensor.

    python scripts/precompute_teacher.py \\
        --splits-root data/coco_subset \\
        --cache-root  data/coco_subset_cache \\
        --batch-size  32
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.teacher import TeacherConfig, extract_from_pils, load  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-root", required=True,
                   help="directory with train/ val/ test/ image subdirs")
    p.add_argument("--cache-root", required=True,
                   help="output directory for cached blobs")
    p.add_argument("--model", default="facebook/dinov2-base")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--grid-h", type=int, default=16)
    p.add_argument("--grid-w", type=int, default=16)
    p.add_argument("--teacher-dim", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    cfg = TeacherConfig(
        model_name=args.model, image_size=args.image_size,
        grid_h=args.grid_h, grid_w=args.grid_w, teacher_dim=args.teacher_dim,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, processor = load(cfg.model_name, device)

    splits_root = Path(args.splits_root)
    cache_root = Path(args.cache_root)

    t0 = time.time()
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for split in ("train", "val", "test"):
        sdir = splits_root / split
        if not sdir.exists():
            print(f"[skip] {sdir} not found")
            continue
        out = cache_root / split
        out.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in sdir.iterdir() if p.suffix.lower() in exts)
        print(f"[{split}] {len(files)} images -> {out}")

        for i in tqdm(range(0, len(files), args.batch_size), desc=split):
            batch = files[i : i + args.batch_size]
            pils = [Image.open(f).convert("RGB") for f in batch]
            patch = extract_from_pils(pils, cfg, device=device)

            inputs = processor(
                images=pils, return_tensors="pt",
                size={"shortest_edge": cfg.image_size},
                do_center_crop=True,
                crop_size={"height": cfg.image_size, "width": cfg.image_size},
            )
            imgs = inputs["pixel_values"].half()     # (B, 3, H, W) normalised
            for b, f in enumerate(batch):
                torch.save(
                    {"image": imgs[b], "teacher_feats": patch[b]},
                    out / f"{f.stem}.pt",
                )

    print(f"cache built in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
