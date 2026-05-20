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

ALL_SCRIPTS=(
    "scripts/dolly/gpt2-120M/run_dskdv2_eta.sh"
    "scripts/dolly/gpt2-120M/run_mta_dskdv2_eta.sh"
    "scripts/dolly/gpt2-340M/run_mta_dskdv2_eta.sh"
    "scripts/dolly/tinyllama-1.1B/run_mta_dskdv2_eta.sh"
    "scripts/dolly/ablation/run_mta_dskdv2_wo_weight.sh"
    "scripts/dolly/ablation/run_mta_dskdv2_word_level.sh"
    "scripts/dolly/ablation/run_mta_dskdv2_phrase_level.sh"
)

log "Will execute ${#ALL_SCRIPTS[@]} script(s):"
for s in "${ALL_SCRIPTS[@]}"; do echo "    - ${s}"; done

FAILED=()
for s in "${ALL_SCRIPTS[@]}"; do
    rel="${s#scripts/}"
    log_file="${LOG_DIR}/${rel%.sh}.log"
    mkdir -p "$(dirname "${log_file}")"

    log "▶ ${rel}  (log: ${log_file#/})"
    # All training scripts assume CWD == project root and BASE_PATH=. (relative).
    if bash "${s}" ${GPUS:+"${GPUS}"} 2>&1 | tee "${log_file}"; then
        log "✓ done: ${rel}"
    else
        log "✗ FAILED: ${rel} (see ${log_file})"
        FAILED+=("${rel}")
        break
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "===================== summary ====================="
log "Total scripts : ${#ALL_SCRIPTS[@]}"
log "Failed        : ${#FAILED[@]}"
for f in "${FAILED[@]}"; do echo "    ✗ ${f}"; done
log "Logs in       : ${LOG_DIR}"
