#!/usr/bin/env bash
# Create a conda env named `droptok` and install dependencies.
# Skip the conda step if you already have a Python 3.10+ environment active.
set -euo pipefail

ENV_NAME="${ENV_NAME:-droptok}"
PY_VERSION="${PY_VERSION:-3.11}"

if command -v conda >/dev/null 2>&1; then
    if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        echo "[setup] creating conda env ${ENV_NAME} (python ${PY_VERSION})"
        conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
fi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[setup] done. activate with:  conda activate ${ENV_NAME}"
