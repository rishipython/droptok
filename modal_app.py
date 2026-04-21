"""Minimal Modal orchestration for DropTok.

Three entrypoints, all self-contained:

    modal run modal_app.py::prepare_coco
        Download COCO val2017 + make a 4000/500/500 split + precompute
        DINOv2 features. Writes everything into the ``droptok-data``
        volume. Skips any stage whose output already exists.

    modal run modal_app.py::train --method gated --lam 0.01 --out gated_lam01
        Run a single training job on an A10. Reads the cache from the
        ``droptok-data`` volume; writes artifacts to the ``droptok-runs``
        volume at /runs/<out>/.

    modal run modal_app.py::train_live --out live_gated --method gated --lam 0.01
        Same as ``train`` but with live DINOv2 teacher + SimCLR-lite
        augmentation. Useful when you want augmentation to also vary the
        reconstruction target.

The Modal image copies the local project in; that's how the container
imports src/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import modal

APP_NAME = "droptok"
DATA_VOL = "droptok-data"
RUNS_VOL = "droptok-runs"

data_volume = modal.Volume.from_name(DATA_VOL, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOL, create_if_missing=True)

DATA_DIR = "/data"
RUNS_DIR = "/runs"
PROJECT_DIR = "/root/project"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "unzip", "curl")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers>=4.56.0",
        "numpy>=1.26",
        "pillow>=10",
        "tqdm>=4.66",
        "pyyaml>=6",
    )
    .add_local_dir(str(Path(__file__).parent / "src"), f"{PROJECT_DIR}/src")
    .add_local_dir(str(Path(__file__).parent / "scripts"),
                   f"{PROJECT_DIR}/scripts")
    .add_local_dir(str(Path(__file__).parent / "configs"),
                   f"{PROJECT_DIR}/configs")
)

app = modal.App(APP_NAME, image=image)
GPU = "A10"     # 24 GB, ~$1.10/hr; plenty for 4-layer ViT on DINO features.


# --------------------------------------------------------------------------- #
# Prepare data (download + split + precompute DINO features)
# --------------------------------------------------------------------------- #

@app.function(
    volumes={DATA_DIR: data_volume},
    timeout=60 * 60,
    cpu=4, memory=8192, gpu=GPU,
)
def prepare_coco(n: int = 5000, train: int = 4000, val: int = 500,
                 test: int = 500, batch_size: int = 64):
    """Idempotent COCO val2017 prep. Skips any stage whose output exists."""
    import subprocess
    sys.path.insert(0, PROJECT_DIR)
    coco_zip = Path(DATA_DIR) / "val2017.zip"
    coco_dir = Path(DATA_DIR) / "val2017"
    split_dir = Path(DATA_DIR) / "coco_subset"
    cache_dir = Path(DATA_DIR) / "coco_subset_cache"

    if not coco_dir.exists():
        if not coco_zip.exists():
            print("[prepare] downloading COCO val2017")
            subprocess.run(
                ["curl", "-L",
                 "http://images.cocodataset.org/zips/val2017.zip",
                 "-o", str(coco_zip)], check=True)
        print("[prepare] unzipping")
        subprocess.run(["unzip", "-q", str(coco_zip), "-d", str(DATA_DIR)],
                       check=True)
    else:
        print(f"[prepare] {coco_dir} already exists, skipping download")

    if not split_dir.exists():
        print(f"[prepare] making {train}/{val}/{test} split")
        subprocess.run(
            ["python", f"{PROJECT_DIR}/scripts/make_subset.py",
             "--src", str(coco_dir), "--dst", str(split_dir),
             "--n", str(n), "--train", str(train),
             "--val", str(val), "--test", str(test)],
            check=True)
    else:
        print(f"[prepare] {split_dir} already exists, skipping split")

    have_cache = (cache_dir / "train").exists() and any(
        (cache_dir / "train").glob("*.pt"))
    if not have_cache:
        print("[prepare] precomputing DINOv2 features")
        subprocess.run(
            ["python", f"{PROJECT_DIR}/scripts/precompute_teacher.py",
             "--splits-root", str(split_dir),
             "--cache-root", str(cache_dir),
             "--batch-size", str(batch_size)],
            check=True)
    else:
        print(f"[prepare] {cache_dir} already populated, skipping precompute")

    data_volume.commit()


# --------------------------------------------------------------------------- #
# Train (cached mode, no augmentation)
# --------------------------------------------------------------------------- #

@app.function(
    volumes={DATA_DIR: data_volume, RUNS_DIR: runs_volume},
    timeout=4 * 60 * 60, gpu=GPU, cpu=4, memory=16384,
)
def train_one(method: str = "gated", K: int = 64, lam: float = 0.0,
              mask_ratio: float = 0.75, epochs: int = 30,
              batch_size: int = 64, out: str = "run",
              recon_target: str = "dino", seed: int = 7):
    """Single cached-mode training job. See ``train_live`` for augmentation."""
    sys.path.insert(0, PROJECT_DIR)
    from src.model import DropTokConfig
    from src.train import run as run_train

    cfg = DropTokConfig(
        method=method, K_total=K, mask_ratio=mask_ratio,
        lambda_comp=lam, recon_target=recon_target,
    )
    summary = run_train(
        cfg, out_dir=f"{RUNS_DIR}/{out}", mode="cached",
        cache_root=f"{DATA_DIR}/coco_subset_cache",
        epochs=epochs, batch_size=batch_size, seed=seed,
    )
    runs_volume.commit()
    return summary


# --------------------------------------------------------------------------- #
# Train (live-teacher + SimCLR-lite augmentation)
# --------------------------------------------------------------------------- #

@app.function(
    volumes={DATA_DIR: data_volume, RUNS_DIR: runs_volume},
    timeout=6 * 60 * 60, gpu=GPU, cpu=4, memory=16384,
)
def train_live(method: str = "gated", K: int = 64, lam: float = 0.0,
               mask_ratio: float = 0.75, epochs: int = 30,
               batch_size: int = 64, out: str = "run_live",
               recon_target: str = "dino", strength: float = 1.0,
               seed: int = 7):
    """Live-teacher training job: raw images + augmentation + DINO
    forward per batch."""
    sys.path.insert(0, PROJECT_DIR)
    from src.model import DropTokConfig
    from src.train import run as run_train

    cfg = DropTokConfig(
        method=method, K_total=K, mask_ratio=mask_ratio,
        lambda_comp=lam, recon_target=recon_target,
    )
    summary = run_train(
        cfg, out_dir=f"{RUNS_DIR}/{out}", mode="live",
        image_root=f"{DATA_DIR}/coco_subset",
        epochs=epochs, batch_size=batch_size, seed=seed, strength=strength,
    )
    runs_volume.commit()
    return summary


# --------------------------------------------------------------------------- #
# Local entrypoints so you can kick these off from your laptop
# --------------------------------------------------------------------------- #

@app.local_entrypoint()
def prepare(n: int = 5000, train: int = 4000, val: int = 500,
            test: int = 500, batch_size: int = 64):
    prepare_coco.remote(n=n, train=train, val=val, test=test,
                        batch_size=batch_size)


@app.local_entrypoint()
def train(method: str = "gated", K: int = 64, lam: float = 0.0,
          mask_ratio: float = 0.75, epochs: int = 30,
          batch_size: int = 64, out: str = "run",
          recon_target: str = "dino", seed: int = 7):
    """Example:
        modal run modal_app.py::train --method gated --lam 0.01 --out gated_lam01
    """
    res = train_one.remote(
        method=method, K=K, lam=lam, mask_ratio=mask_ratio,
        epochs=epochs, batch_size=batch_size, out=out,
        recon_target=recon_target, seed=seed,
    )
    print(res)


@app.local_entrypoint()
def train_live_cmd(method: str = "gated", K: int = 64, lam: float = 0.0,
                   mask_ratio: float = 0.75, epochs: int = 30,
                   batch_size: int = 64, out: str = "run_live",
                   recon_target: str = "dino", strength: float = 1.0,
                   seed: int = 7):
    """Example:
        modal run modal_app.py::train_live_cmd --method gated --lam 0.01 \\
            --out live_gated --epochs 60 --strength 1.0
    """
    res = train_live.remote(
        method=method, K=K, lam=lam, mask_ratio=mask_ratio,
        epochs=epochs, batch_size=batch_size, out=out,
        recon_target=recon_target, strength=strength, seed=seed,
    )
    print(res)
