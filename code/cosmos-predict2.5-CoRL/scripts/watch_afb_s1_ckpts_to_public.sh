#!/bin/bash
# Watch AFB S1 Cosmos DCP checkpoints and copy completed epoch checkpoints to public storage.

set -euo pipefail

SRC="${SRC:-/mnt/gyc_ckp/cosmos_train_output/cosmos_predict2_action_conditioned_robotwin/afb_s1_family_balanced/cosmos_predict2p5_2B_afb_s1_family_balanced_3_1_1_1_chunk16_headcam/checkpoints}"
DEST_ROOT="${DEST_ROOT:-/mnt/public_ckp/cscsx_projects/cosmospredict2.5_infer/checkpoints}"
EPOCH_STEP="${EPOCH_STEP:-4277}"
FINAL_STEP="${FINAL_STEP:-40000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_FILE="${LOG_FILE:-/mnt/public_ckp/cscsx_projects/cosmospredict2.5_infer/watch_checkpoints.log}"

mkdir -p "${DEST_ROOT}" "$(dirname "${LOG_FILE}")"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" | tee -a "${LOG_FILE}"
}

is_complete_dcp() {
    local ckpt_dir="$1"
    [ -d "${ckpt_dir}/model" ] || return 1
    [ -f "${ckpt_dir}/model/.metadata" ] || return 1
    [ -f "${ckpt_dir}/trainer/.metadata" ] || return 1
    [ -f "${SRC}/latest_checkpoint.txt" ] || return 1
    return 0
}

copy_one() {
    local src_dir="$1"
    local base step epoch dst tmp

    base="$(basename "${src_dir}")"
    step="$((10#${base#iter_}))"
    if [ "${step}" -eq "${FINAL_STEP}" ]; then
        dst="${DEST_ROOT}/checkpoint-final-step${step}"
    elif [ "$((step % EPOCH_STEP))" -eq 0 ]; then
        epoch="$((step / EPOCH_STEP))"
        dst="${DEST_ROOT}/checkpoint-epoch${epoch}-step${step}"
    else
        return 0
    fi
    tmp="${dst}.tmp"

    if [ -f "${dst}/.copy_complete" ]; then
        return 0
    fi
    if ! is_complete_dcp "${src_dir}"; then
        return 0
    fi

    log "Copying ${src_dir} -> ${dst}"
    rm -rf "${tmp}"
    cp -a "${src_dir}" "${tmp}"
    printf 'source=%s\nstep=%s\nepoch_step=%s\nfinal_step=%s\ncopied_at=%s\n' \
        "${src_dir}" "${step}" "${EPOCH_STEP}" "${FINAL_STEP}" "$(date '+%Y-%m-%d %H:%M:%S %z')" > "${tmp}/copy_info.txt"
    touch "${tmp}/.copy_complete"
    rm -rf "${dst}"
    mv "${tmp}" "${dst}"
    log "Finished ${dst}"
}

log "Watching ${SRC}; copying epoch checkpoints every ${EPOCH_STEP} steps and final step ${FINAL_STEP} to ${DEST_ROOT}"

while true; do
    if [ -d "${SRC}" ]; then
        while IFS= read -r ckpt_dir; do
            copy_one "${ckpt_dir}"
        done < <(find "${SRC}" -maxdepth 1 -mindepth 1 -type d -name 'iter_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]' | sort)
    else
        log "Source directory not found yet: ${SRC}"
    fi
    sleep "${POLL_SECONDS}"
done
