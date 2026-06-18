#!/bin/bash
# One-episode local inference for the AFB S1 head-camera Cosmos-Predict2.5 checkpoint.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/gyc/cosmos-predict2.5-CoRL}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/gyc/cosmos-predict2.5/.venv/bin/python3}"

export PYTHONPATH="${REPO_ROOT}:$REPO_ROOT/packages/cosmos-cuda:${PYTHONPATH:-}"
export H5PY_EXTRA_PATH="${H5PY_EXTRA_PATH:-/mnt/gyc/envs/cosmos-policy/lib/python3.10/site-packages}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cosmos-infer}"

VENV_CUDNN="${VENV_CUDNN:-/mnt/gyc/cosmos-predict2.5/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib}"
export LD_LIBRARY_PATH="${VENV_CUDNN}:/usr/local/cuda-12.2/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"

exec "${PYTHON_BIN}" scripts/infer_afb_s1_epoch1_one_episode.py "$@"
