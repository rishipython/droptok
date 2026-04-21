"""Build seeded train/val/test splits by symlinking/copying image files.

Example:

    python scripts/make_subset.py \\
        --src data/coco/val2017 \\
        --dst data/coco_subset \\
        --n 5000 --train 4000 --val 500 --test 500
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="source image directory")
    p.add_argument("--dst", required=True, help="output split root")
    p.add_argument("--n", type=int, required=True, help="total images to sample")
    p.add_argument("--train", type=int, required=True)
    p.add_argument("--val", type=int, required=True)
    p.add_argument("--test", type=int, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--copy", action="store_true",
                   help="copy files instead of symlinking (default: symlink)")
    args = p.parse_args()

    assert args.train + args.val + args.test == args.n, \
        "train+val+test must equal n"

    src = Path(args.src)
    dst = Path(args.dst)
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in exts)
    if len(files) < args.n:
        raise RuntimeError(f"Only found {len(files)} images in {src}, need {args.n}")

    rng = random.Random(args.seed)
    rng.shuffle(files)
    chosen = files[: args.n]
    splits = [
        ("train", chosen[: args.train]),
        ("val",   chosen[args.train : args.train + args.val]),
        ("test",  chosen[args.train + args.val :]),
    ]
    for name, items in splits:
        out = dst / name
        out.mkdir(parents=True, exist_ok=True)
        for f in items:
            tgt = out / f.name
            if tgt.exists():
                tgt.unlink()
            if args.copy:
                shutil.copy2(f, tgt)
            else:
                tgt.symlink_to(f.resolve())
        print(f"{name}: {len(items):>6} -> {out}")


if __name__ == "__main__":
    main()
