set -u
set -o pipefail

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

log() { echo -e "\n[run.sh] $*"; }

# Install dependencies (if not already done).
bash install.sh
bash download_model.sh


export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_DISABLED=True

ALL_SCRIPTS=(
    "scripts/dolly/gpt2-1.5B/run_mta_dskdv2_eta.sh"
    "scripts/dolly/opt-2.7B/run_mta_dskdv2_eta.sh"
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

# # ---------------------------------------------------------------------------
# # 5. Archive all training outputs (checkpoints, logs) into a single zip
# # ---------------------------------------------------------------------------
# RESULT_ZIP="${PROJECT_ROOT}/results_$(date +%Y%m%d_%H%M%S).zip"
# log "Zipping outputs/ + logs/ → ${RESULT_ZIP}"
# if command -v zip >/dev/null 2>&1; then
#     ( cd "${PROJECT_ROOT}" && zip -r "${RESULT_ZIP}" outputs logs >/dev/null )
# else
#     # Fallback when `zip` is not installed (common on minimal images).
#     RESULT_ZIP="${RESULT_ZIP%.zip}.tar.gz"
#     ( cd "${PROJECT_ROOT}" && tar -czf "${RESULT_ZIP}" outputs logs )
# fi
# log "Archive ready: ${RESULT_ZIP}"

# [[ ${#FAILED[@]} -eq 0 ]] || exit 1
