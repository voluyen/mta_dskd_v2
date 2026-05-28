#!/usr/bin/env bash
set -u
set -o pipefail

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

log() { echo -e "\n[run.sh] $*"; }

# ---------------------------------------------------------------------------
# Python venv setup
# ---------------------------------------------------------------------------
ENV_DIR="${ENV_DIR:-env}"

if [ ! -d "${ENV_DIR}" ]; then
    log "creating venv at '${ENV_DIR}'"
    python -m venv "${ENV_DIR}"
fi
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
log "active python: $(which python) ($(python --version 2>&1))"

# Install dependencies (if not already done).
bash install.sh
bash download_model.sh

export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_DISABLED=True
export TOKENIZERS_PARALLELISM=false

# Disable NVLink SHARP (NVLS) multicast — server has no working NVSwitch/Fabric
# Manager, so NCCL otherwise fails with CUDA error 802 'system not yet initialized'.
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=1

# ---------------------------------------------------------------------------
# Job list: "script_path|gpu_id|master_port"
# - gpu_id    : passed as $1 to each script → overrides its built-in GPUS=()
# - master_port: injected via MASTER_PORT env var (each script respects
#               MASTER_PORT="${MASTER_PORT:-<random>}" so ports never collide)
# Adjust GPU IDs to match your server's available devices (here: 3, 4, 5).
# ---------------------------------------------------------------------------
declare -a JOBS=(
    "scripts/dolly/gpt2-340M/run_dskdv2_eta.sh|2|6700"
    "scripts/dolly/tinyllama-1.1B/run_dskdv2_eta.sh|3|6710"
    "scripts/dolly/gpt2-1.5B/run_dskdv2_eta.sh|2|6720"
    "scripts/dolly/opt-2.7B/run_dskdv2_eta.sh|3|6730"
)

log "Launching ${#JOBS[@]} jobs in pairs (2 GPUs × 2 rounds):"
for entry in "${JOBS[@]}"; do
    IFS='|' read -r s g p <<< "${entry}"
    echo "    GPU ${g}  port ${p}  ←  ${s#scripts/}"
done

# ---------------------------------------------------------------------------
# Run 2 jobs at a time (one per GPU), wait for the pair, then start the next
# ---------------------------------------------------------------------------
FAILED=()
total=${#JOBS[@]}

run_pair() {
    local i=$1
    local entry1="${JOBS[$i]}"
    local entry2="${JOBS[$((i+1))]}"
    local pair_pids=() pair_rels=() pair_logs=()

    for entry in "${entry1}" "${entry2}"; do
        IFS='|' read -r s g p <<< "${entry}"
        local rel="${s#scripts/}"
        local log_file="${LOG_DIR}/${rel%.sh}.log"
        mkdir -p "$(dirname "${log_file}")"
        pair_rels+=("${rel}")
        pair_logs+=("${log_file}")
        log "▶ GPU ${g} port ${p}: ${rel}  →  ${log_file}"
        MASTER_PORT="${p}" bash -o pipefail "${s}" "${g}" 2>&1 | tee "${log_file}" &
        pair_pids+=($!)
    done

    for j in 0 1; do
        wait "${pair_pids[$j]}"
        rc=$?
        if [ $rc -eq 0 ]; then
            log "✓ done : ${pair_rels[$j]}"
        else
            log "✗ FAILED: ${pair_rels[$j]} (exit=${rc}, see ${pair_logs[$j]})"
            FAILED+=("${pair_rels[$j]} (exit=${rc})")
        fi
    done
}

log "=== Pair 1/2 (GPU 2 + GPU 3) ==="
run_pair 0
log "=== Pair 2/2 (GPU 2 + GPU 3) ==="
run_pair 2

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "===================== summary ====================="
log "Total jobs : ${#JOBS[@]}"
log "Succeeded  : $(( ${#JOBS[@]} - ${#FAILED[@]} ))"
log "Failed     : ${#FAILED[@]}"
for f in "${FAILED[@]}"; do echo "    ✗ ${f}"; done
if [ ${#FAILED[@]} -eq 0 ]; then
    log "All jobs completed successfully ✓"
fi
log "Logs in    : ${LOG_DIR}/"
