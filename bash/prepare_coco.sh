#!/usr/bin/env bash
# Download COCO val2017 (small, 5k images, no annotations needed for
# self-supervised training), build a train/val/test split, and
# precompute DINOv2 features.
#
# Variables you can override via env:
#   COCO_URL  : download URL (defaults to the standard mirror)
#   DATA_DIR  : where to place downloads + splits + cache (default: ./data)
#   N         : total images to sample (default: 5000, the whole val set)
#   TRAIN/VAL/TEST : split sizes (must sum to N)
#   BATCH_SIZE: teacher batch size (default: 32)

set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
COCO_URL="${COCO_URL:-http://images.cocodataset.org/zips/val2017.zip}"
N="${N:-5000}"
TRAIN="${TRAIN:-4000}"
VAL="${VAL:-500}"
TEST="${TEST:-500}"
BATCH_SIZE="${BATCH_SIZE:-32}"

mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

if [ ! -d coco/val2017 ]; then
    echo "[prepare_coco] downloading COCO val2017 (~1GB)"
    mkdir -p coco
    if [ ! -f coco/val2017.zip ]; then
        curl -L "${COCO_URL}" -o coco/val2017.zip
    fi
    (cd coco && unzip -q val2017.zip)
fi

cd - >/dev/null

echo "[prepare_coco] building splits (${TRAIN}/${VAL}/${TEST})"
python scripts/make_subset.py \
    --src "${DATA_DIR}/coco/val2017" \
    --dst "${DATA_DIR}/coco_subset" \
    --n "${N}" --train "${TRAIN}" --val "${VAL}" --test "${TEST}"

echo "[prepare_coco] precomputing DINOv2 features"
python scripts/precompute_teacher.py \
    --splits-root "${DATA_DIR}/coco_subset" \
    --cache-root  "${DATA_DIR}/coco_subset_cache" \
    --batch-size  "${BATCH_SIZE}"

echo "[prepare_coco] done."
echo "  images: ${DATA_DIR}/coco_subset/{train,val,test}/"
echo "  cache : ${DATA_DIR}/coco_subset_cache/{train,val,test}/"
