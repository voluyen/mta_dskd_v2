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
    "scripts/dolly/gpt2-340M/run_dskdv2_eta.sh|4|6700"
    "scripts/dolly/gpt2-1.5B/run_dskdv2_eta.sh|4|6710"
    "scripts/dolly/opt-2.7B/run_dskdv2_eta.sh|4|6720"
    "scripts/dolly/tinyllama-1.1B/run_dskdv2_eta.sh|4|6730"
)

log "Launching ${#JOBS[@]} jobs sequentially:"
for entry in "${JOBS[@]}"; do
    IFS='|' read -r s g p <<< "${entry}"
    echo "    GPU ${g}  port ${p}  ←  ${s#scripts/}"
done

# ---------------------------------------------------------------------------
# Run jobs one at a time — wait for each to finish before starting the next
# ---------------------------------------------------------------------------
FAILED=()
SCRIPTS=()
LOG_FILES=()
idx=0

for entry in "${JOBS[@]}"; do
    IFS='|' read -r s g p <<< "${entry}"
    rel="${s#scripts/}"
    log_file="${LOG_DIR}/${rel%.sh}.log"
    mkdir -p "$(dirname "${log_file}")"
    SCRIPTS+=("${rel}")
    LOG_FILES+=("${log_file}")

    log "▶ [$(( idx+1 ))/${#JOBS[@]}] GPU ${g} port ${p}: ${rel}  →  ${log_file}"
    MASTER_PORT="${p}" bash -o pipefail "${s}" "${g}" 2>&1 | tee "${log_file}"
    rc=$?
    if [ $rc -eq 0 ]; then
        log "✓ done : ${rel}"
    else
        log "✗ FAILED: ${rel} (exit=${rc}, see ${log_file})"
        FAILED+=("${rel} (exit=${rc})")
    fi
    (( idx++ )) || true
done

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
