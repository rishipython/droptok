"""Launch a single DropTok training run from a YAML config + CLI overrides.

Example:

    python scripts/run.py \\
        --config configs/default.yaml \\
        --mode cached --cache-root data/coco_subset_cache \\
        --method gated --K 64 --lambda-comp 0.01 \\
        --out runs/gated_K64

Any CLI flag listed under "overrides" wins over the YAML config.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from src.train import build_config, run           # noqa: E402
from src.utils import load_config                 # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--out", required=True, help="output directory")

    # data mode
    p.add_argument("--mode", choices=("cached", "live"), default=None,
                   help="cached (precomputed features) or live (raw images + teacher)")
    p.add_argument("--cache-root", default=None)
    p.add_argument("--image-root", default=None)

    # common overrides
    p.add_argument("--method", default=None,
                   choices=("gated", "progressive_learned", "progressive_random"))
    p.add_argument("--K", type=int, default=None,
                   help="target K (fixed-K variants) or starting budget")
    p.add_argument("--mask-ratio", type=float, default=None)
    p.add_argument("--recon-target", default=None, choices=("dino", "pixel"))
    p.add_argument("--lambda-comp", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--strength", type=float, default=None,
                   help="augmentation strength (live mode only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = load_config(args.config)

    # collect overrides
    over = {
        "method": args.method,
        "K_total": args.K,
        "mask_ratio": args.mask_ratio,
        "recon_target": args.recon_target,
        "lambda_comp": args.lambda_comp,
    }
    for k, v in over.items():
        if v is not None:
            base[k] = v

    cfg = build_config(base)

    epochs = args.epochs or base.get("epochs", 30)
    batch_size = args.batch_size or base.get("batch_size", 32)
    lr = args.lr or base.get("lr", 2e-4)
    num_workers = args.num_workers if args.num_workers is not None \
        else base.get("num_workers", 4)
    seed = args.seed or base.get("seed", 7)
    strength = args.strength if args.strength is not None \
        else base.get("strength", 1.0)
    mode = args.mode or base.get("mode", "cached")

    run(
        cfg, args.out,
        mode=mode,
        cache_root=args.cache_root or base.get("cache_root"),
        image_root=args.image_root or base.get("image_root"),
        epochs=epochs, batch_size=batch_size, lr=lr,
        num_workers=num_workers, seed=seed, strength=strength,
    )


if __name__ == "__main__":
    main()
