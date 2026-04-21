#!/usr/bin/env bash
# Thin wrapper around scripts/run.py so a new user can get going with
# just one command. Defaults to the cached 5k-COCO setting produced by
# bash/prepare_coco.sh.
#
# Examples:
#
#   # progressive_learned at K=64 on the cached 5k split:
#   bash/train.sh --method progressive_learned --K 64 --out runs/prog_K64
#
#   # gated (variable-K) with compression penalty:
#   bash/train.sh --method gated --lambda-comp 0.01 --out runs/gated_lam001
#
#   # live teacher + augmentation on a bigger image directory:
#   bash/train.sh --mode live --image-root data/imagenet_subset \
#       --method gated --lambda-comp 0.01 --epochs 60 --out runs/live_gated
#
# Any flag supported by scripts/run.py works here.
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
python scripts/run.py --config "${CONFIG}" "$@"
