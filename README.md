# DropTok

A small, clean reference implementation of **DropTok**: a self-supervised
visual tokenizer that produces a variable number of tokens per image by
progressively dropping attention-dormant tokens inside a Vision
Transformer encoder, trained with masked reconstruction of frozen
DINOv2 features (or raw pixels).

> Banerjee, Bayiz, Athavale. *DropTok: Adaptive Visual Tokenization via
> Progressive Visual Token Dropping.* CS 280, UC Berkeley, Spring 2026.

The implementation here is the faithful core of that proposal, stripped
of one-off experiments: three methods (`gated`, `progressive_learned`,
`progressive_random`), two training modes (cached vs. live teacher),
and two reconstruction targets (DINO features vs. raw pixels).

---

## 1. What's in here

```
.
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml           # single config, every field CLI-overridable
├── src/
│   ├── model.py               # DropTok + decoder (one file, < 500 lines)
│   ├── teacher.py             # frozen DINOv2 wrapper
│   ├── data.py                # cached + raw-image datasets
│   ├── augment.py             # SimCLR-lite augmentation
│   ├── train.py               # training loop (cached OR live teacher)
│   └── utils.py               # seeding + yaml loader
├── scripts/
│   ├── make_subset.py         # seeded train/val/test split
│   ├── precompute_teacher.py  # cache DINO features to disk
│   └── run.py                 # one-liner trainer
├── bash/
│   ├── setup.sh               # conda env + pip install
│   ├── prepare_coco.sh        # download COCO val2017 + cache features
│   └── train.sh               # thin wrapper around scripts/run.py
└── modal_app.py               # optional Modal orchestration
```

## 2. Setup

### 2a. Local

Requires a CUDA GPU for anything non-trivial (teacher inference), but
will run on CPU for very small sanity checks.

```bash
git clone https://github.com/rishipython/droptok.git
cd droptok
bash bash/setup.sh              # creates `droptok` conda env + installs deps
conda activate droptok
```

### 2b. Modal (recommended for full training)

No VM management needed. Install and authenticate:

```bash
pip install modal
modal setup                      # opens a browser for auth
```

The first Modal call creates two persistent volumes (`droptok-data`,
`droptok-runs`) in your workspace.

---

## 3. Prepare data

We train from COCO val2017 (5k images). It's public, small (~1 GB),
and fast to cache.

### 3a. Local

```bash
bash bash/prepare_coco.sh
# -> data/coco/val2017/                (raw jpgs)
# -> data/coco_subset/{train,val,test} (4000/500/500 split)
# -> data/coco_subset_cache/...        (DINOv2 features, fp16)
```

Use `DATA_DIR=...` / `N=...` / `TRAIN=...` env vars to override
defaults. See `bash/prepare_coco.sh`.

### 3b. Modal

```bash
modal run modal_app.py::prepare
```

Runs the same pipeline inside Modal and stores everything in the
`droptok-data` volume. Safe to re-run -- each stage is skipped if its
output already exists.

---

## 4. Train

### 4a. Cached mode (fast, no augmentation)

One command:

```bash
bash bash/train.sh --method gated --lambda-comp 0.01 --out runs/gated_lam01
```

All flags are forwarded to `scripts/run.py`. The important ones:

| flag | default | meaning |
| --- | --- | --- |
| `--method` | `gated` | `gated`, `progressive_learned`, or `progressive_random` |
| `--K`  | 64 | fixed-K variants only; `gated` derives K from the penalty |
| `--mask-ratio` | 0.75 | fraction of input patches replaced with [MASK] |
| `--recon-target` | `dino` | `dino` (cosine loss) or `pixel` (MSE loss) |
| `--lambda-comp` | 0.0 | compression penalty weight (gated only) |
| `--epochs` | 30 | |
| `--batch-size` | 32 | |
| `--lr` | 2e-4 | |
| `--mode` | `cached` | `cached` (precomputed features) or `live` (raw images + teacher) |

On Modal:

```bash
modal run modal_app.py::train --method gated --lam 0.01 --out gated_lam01
```

### 4b. Live teacher + augmentation

When you want augmentation to also vary the reconstruction target (the
proper MAE-style setup, and the only way to train on augmented views
without leaking a fixed-crop DINO cache), use the `live` mode. The
student trains on augmented pixels and the frozen DINOv2 teacher is
run inside each training step so the target matches the student view.

```bash
bash bash/train.sh --mode live \
    --image-root data/coco_subset \
    --method gated --lambda-comp 0.01 \
    --epochs 60 --strength 1.0 \
    --out runs/live_gated_lam01
```

Or on Modal:

```bash
modal run modal_app.py::train_live_cmd --method gated --lam 0.01 \
    --out live_gated --epochs 60 --strength 1.0
```

Live mode is ~2x slower per step (one extra teacher forward per batch)
but breaks the "decoder memorises a position-conditional dataset mean"
shortcut that makes cached training collapse to K=1 under any
compression penalty.

---

## 5. Output layout

Each run writes:

```
runs/<out>/
├── summary.json       # final metrics + full DropTokConfig
├── history.json       # per-epoch train_loss / val_err / (K_soft, K_hard)
└── best.pt            # best-val checkpoint
```

The important fields in `summary.json`:

- `test_err_mean`: reconstruction error on the test split (cosine-err
  for `dino`, pixel MSE for `pixel`).
- `avg_tokens`: average number of surviving tokens per image on the
  test set. For fixed-K methods this is exactly `K_total`; for
  `gated` it's the learned, per-image count.
- `natural_K_mean/median/min/max`: `gated` only; distribution of
  surviving-token counts.
- `gate_alpha`, `gate_tau`: learned gate parameters per stage (`gated`).

---

## 6. Reproducing the proposal's core claims

### 6a. Progressive vs. one-shot (matched K)

```bash
bash bash/train.sh --method progressive_learned --K 64 --out runs/prog_learned_K64
bash bash/train.sh --method progressive_random  --K 64 --out runs/prog_random_K64
```

Progressive learned should reach a lower `test_err_mean` than
progressive random at the same K *and* produce content-dependent
survival maps (see `summary.json["final_positions"]` histogramming
over the test set -- plot is left to the user).

### 6b. Variable-K via λ sweep (gated)

Sweep `lambda_comp` to trace the rate--distortion curve:

```bash
for LAM in 0.0 0.001 0.003 0.01 0.03 0.1; do
    bash bash/train.sh --method gated --lambda-comp $LAM \
        --out runs/gated_lam${LAM}
done
```

Each run reports `natural_K_mean` and `test_err_mean`; scatter one
against the other for the Pareto front.

---

## 7. Results

Preliminary numbers from the same mechanism on cached COCO-val2017
subsets are in [`results/README.md`](results/README.md). Short version:
the fixed-K methods work as expected (`progressive_random` matches or
slightly beats `progressive_learned` at matched K, with a positional
prior driving the gap), and the variable-K `gated` method collapses to
K~1 under any compression penalty because the decoder's positional
queries can memorise a per-position dataset mean. Live teacher +
augmentation (`--mode live`) is the proposed fix; results for that
setup are pending.

## 8. Notes / gotchas

- **Pixel target + cached mode is slightly silly.** Cached images are
  center-cropped once; `pixel` targets on cached runs will give
  deterministic MSE. Use `pixel` with `mode=live` and augmentation if
  you want the target to vary.
- **`gated` can collapse to K=1** on small cached datasets. The
  decoder's learnable positional queries alone can memorise a
  position-conditional dataset mean and serve reconstruction from any
  single token. Fixes, in order of effort: bigger dataset, live
  teacher + augmentation, pixel target, or add a `K_min` floor to the
  compression penalty (not implemented here; trivially added in
  `DropTok.compute_loss`).
- The decoder is small (2 layers, d_model=256) but the `self.pos`
  positional queries give it effectively full per-position
  conditioning; that's the parameter you want to watch, not the
  decoder layer count.

## 9. Citation

If you use this code, please cite the original proposal:

```
@misc{droptok2026,
  title  = {DropTok: Adaptive Visual Tokenization via Progressive Visual Token Dropping},
  author = {Banerjee, Arjun and Bayiz, Ozan and Athavale, Rishi},
  year   = {2026},
  note   = {CS 280, UC Berkeley},
}
```

## 10. License

MIT.
