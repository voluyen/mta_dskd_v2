#!/usr/bin/env bash
set -u
set -o pipefail

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

log() { echo -e "\n[run.sh] $*"; }

# ---------------------------------------------------------------------------
# Conda environment setup
# ---------------------------------------------------------------------------
ENV_NAME="${ENV_NAME:-mta_dskd}"
PY_VERSION="${PY_VERSION:-3.10}"

if command -v conda >/dev/null 2>&1; then
    log "conda detected: $(conda --version)"
    # Make `conda activate` usable from this non-interactive shell.
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"

    if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        log "conda env '${ENV_NAME}' already exists — reusing"
    else
        log "creating conda env '${ENV_NAME}' (python=${PY_VERSION})"
        conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
    fi
    conda activate "${ENV_NAME}"
    log "active python: $(which python) ($(python --version 2>&1))"
else
    log "conda not found — skipping env setup, using current python"
fi

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
    "scripts/dolly/gpt2-120M/run_mta_dskdv2_eta.sh|3|6610"
    "scripts/dolly/gpt2-340M/run_mta_dskdv2_eta.sh|3|6620"
    "scripts/dolly/tinyllamA-1.1B/run_mta_dskdv2_eta.sh|3|6630"
    "scripts/dolly/gpt2-1.5B/run_mta_dskdv2_eta.sh|4|6640"
    "scripts/dolly/opt-2.7B/run_mta_dskdv2_eta.sh|4|6650"
    "scripts/dolly/ablation/run_mta_dskdv2_wo_weight.sh|5|6660"
    "scripts/dolly/ablation/run_mta_dskdv2_word_level.sh|5|6670"
    "scripts/dolly/ablation/run_mta_dskdv2_phrase_level.sh|5|6680"
)

log "Launching ${#JOBS[@]} jobs simultaneously:"
for entry in "${JOBS[@]}"; do
    IFS='|' read -r s g p <<< "${entry}"
    echo "    GPU ${g}  port ${p}  ←  ${s#scripts/}"
done

# ---------------------------------------------------------------------------
# Launch all jobs in the background, each with its own dedicated port
# ---------------------------------------------------------------------------
PIDS=()
SCRIPTS=()
LOG_FILES=()

for entry in "${JOBS[@]}"; do
    IFS='|' read -r s g p <<< "${entry}"
    rel="${s#scripts/}"
    log_file="${LOG_DIR}/${rel%.sh}.log"
    mkdir -p "$(dirname "${log_file}")"

    log "▶ GPU ${g} port ${p}: ${rel}  →  ${log_file}"
    # MASTER_PORT env var is consumed by the script's:
    #   MASTER_PORT="${MASTER_PORT:-66$(($RANDOM%90+10))}"
    MASTER_PORT="${p}" bash -o pipefail "${s}" "${g}" 2>&1 | tee "${log_file}" &
    PIDS+=($!)
    SCRIPTS+=("${rel}")
    LOG_FILES+=("${log_file}")
done

log "All ${#PIDS[@]} jobs launched — waiting for completion..."

# ---------------------------------------------------------------------------
# Wait for every job and collect failures
# ---------------------------------------------------------------------------
FAILED=()
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    rc=$?
    if [ $rc -eq 0 ]; then
        log "✓ done : ${SCRIPTS[$i]}"
    else
        log "✗ FAILED: ${SCRIPTS[$i]} (exit=${rc}, see ${LOG_FILES[$i]})"
        FAILED+=("${SCRIPTS[$i]} (exit=${rc})")
    fi
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
