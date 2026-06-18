#!/bin/bash
# Run one RobotWin validation episode for each of the three final checkpoints.
#
# Baidu Cloud 8-GPU job command:
#   bash /mnt/gyc/cosmos-predict2.5-CoRL/scripts/infer_robotwin_three_ckpts_8gpu.sh
#
# This launcher uses three single-GPU Python processes in parallel:
#   GPU0: pcp+pob chunk16
#   GPU1: all50 chunk16
#   GPU2: all50 chunk32

set -euo pipefail

REPO="/mnt/gyc/cosmos-predict2.5-CoRL"
cd "${REPO}"

VENV="/mnt/gyc/cosmos-predict2.5/.venv"
PYTHON="${VENV}/bin/python3"
if [ ! -x "${PYTHON}" ]; then
    echo "[ERROR] Python not found: ${PYTHON}"
    exit 1
fi

# Put venv cuDNN9 first to avoid system cuDNN conflicts.
VENV_CUDNN="${VENV}/lib/python3.10/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="${VENV_CUDNN}:/usr/local/cuda-12.2/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=".:packages/cosmos-cuda"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export MPLCONFIGDIR="/tmp/matplotlib-cosmos-infer"

ANN="${ANN:-/mnt/gyc_ckp/datasets/robotwin_clean50_pcp_pob/annotation/val/0.json}"
SAVE_ROOT="${SAVE_ROOT:-/mnt/gyc_ckp/infer_results/robotwin_three_ckpts}"
LOG_DIR="${SAVE_ROOT}/logs"
mkdir -p "${SAVE_ROOT}" "${LOG_DIR}" "${MPLCONFIGDIR}"
PIDS=()

echo "[INFO] Repo: ${REPO}"
echo "[INFO] Python: ${PYTHON}"
echo "[INFO] Annotation: ${ANN}"
echo "[INFO] Save root: ${SAVE_ROOT}"
echo "[INFO] Logs: ${LOG_DIR}"
echo "[INFO] CUDA_VISIBLE_DEVICES before launch: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[INFO] INFER_MODE: ${INFER_MODE:-parallel}"

run_one() {
    local gpu="$1"
    local name="$2"
    local log_file="${LOG_DIR}/${name}_$(date +%Y%m%d_%H%M%S).log"

    echo "[INFO] Launch ${name} on GPU ${gpu}; log=${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON}" scripts/infer_robotwin_three_ckpts.py \
        --only "${name}" \
        --ann "${ANN}" \
        --save-root "${SAVE_ROOT}" \
        2>&1 | tee "${log_file}" &
    PIDS+=("$!")
}

wait_all() {
    local failed=0
    local pid
    for pid in "${PIDS[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    PIDS=()
    if [ "${failed}" -ne 0 ]; then
        echo "[ERROR] One or more inference processes failed. Check logs in ${LOG_DIR}."
        exit 1
    fi
}

if [ "${INFER_MODE:-parallel}" = "serial" ]; then
    run_one 0 pcp_pob_chunk16
    wait_all
    run_one 0 all50_chunk16
    wait_all
    run_one 0 all50_chunk32
    wait_all
else
    run_one 0 pcp_pob_chunk16
    run_one 1 all50_chunk16
    run_one 2 all50_chunk32
    wait_all
fi

echo "[DONE] Inference finished."
echo "[DONE] Outputs:"
find "${SAVE_ROOT}" -maxdepth 1 -type f -name '*.mp4' -print | sort
