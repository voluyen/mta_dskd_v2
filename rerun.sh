#!/usr/bin/env bash
# Re-run tinyllama-1.1B → Mistral-7B MTA-DSKD experiment on a single GPU.
# Usage:
#   bash rerun.sh          # uses GPU_ID below
#   bash rerun.sh 3        # override GPU at runtime
set -u
set -o pipefail

# ---------------------------------------------------------------------------
# Config — adjust GPU_ID and MASTER_PORT to avoid conflicts with other jobs.
# ---------------------------------------------------------------------------
GPU_ID="${1:-3}"
MASTER_PORT="${MASTER_PORT:-6750}"

SCRIPT="scripts/dolly/tinyllamA-1.1B/run_mta_dskdv2_eta.sh"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/dolly/tinyllama-1.1B/run_mta_dskdv2_eta.log"

log() { echo -e "\n[rerun.sh] $*"; }

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [ ! -f "${SCRIPT}" ]; then
    echo "ERROR: script not found: ${SCRIPT}" >&2
    exit 1
fi

mkdir -p "$(dirname "${LOG_FILE}")"

# ---------------------------------------------------------------------------
# Python venv
# ---------------------------------------------------------------------------
ENV_DIR="${ENV_DIR:-env}"
if [ ! -d "${ENV_DIR}" ]; then
    log "creating venv at '${ENV_DIR}'"
    python -m venv "${ENV_DIR}"
fi
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
log "active python: $(which python) ($(python --version 2>&1))"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_DISABLED=True
export TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=1

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
log "GPU ${GPU_ID}  port ${MASTER_PORT}"
log "script  : ${SCRIPT}"
log "log     : ${LOG_FILE}"
log "save_dir: results/rerun/tinyllama/tinyllama-1.1B/mta_dskd_v2_eta/"

MASTER_PORT="${MASTER_PORT}" bash "${SCRIPT}" "${GPU_ID}" 2>&1 | tee "${LOG_FILE}"
rc=${PIPESTATUS[0]}

if [ $rc -eq 0 ]; then
    log "✓ done"
else
    log "✗ FAILED (exit=${rc}, see ${LOG_FILE})"
    exit $rc
fi
