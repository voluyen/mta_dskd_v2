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
#
# VRAM estimates (single GPU, per job):
#   gpt2-base  (120M bf16) + Qwen1.5-1.8B  (fp16 frozen)     →  ~6–7  GB  GPU 6
#   gpt2-medium(340M bf16) + Qwen1.5-1.8B  (fp16 frozen)     → ~10–11 GB  GPU 6 (shared)
#   TinyLlama-1.1B + LoRA r=256 (bf16) + Mistral-7B  (fp16)  → ~20–22 GB  GPU 4
#   opt-2.7B   + LoRA r=256 (bf16) + Qwen2.5-7B (fp16)       → ~22–26 GB  GPU 5
#   gpt2-xl (1.5B) + LoRA r=256 (bf16) + Qwen2.5-7B (fp16)   → ~20–22 GB  GPU 7
#
# NOTE: gpt2-120M and gpt2-340M share GPU 6 — ensure sufficient VRAM (~17 GB combined).
# ---------------------------------------------------------------------------
declare -a JOBS=(
    "scripts/dolly/tinyllamA-1.1B/run_dskdv2_eta.sh|0|6700"       # TinyLlama-1.1B baseline → Mistral-7B  ~20–22 GB
    "scripts/dolly/tinyllamA-1.1B/run_mta_dskdv2_eta.sh|1|6750"   # TinyLlama-1.1B MTA     → Mistral-7B  ~20–22 GB
)

log "Launching ${#JOBS[@]} jobs simultaneously:"
for entry in "${JOBS[@]}"; do
    IFS='|' read -r s g p <<< "${entry}"
    echo "    GPU ${g}  port ${p}  ←  ${s#scripts/}"
done

# ---------------------------------------------------------------------------
# Launch all jobs in parallel
# ---------------------------------------------------------------------------
PIDS=()
RELS=()
LOGS=()
FAILED=()

for entry in "${JOBS[@]}"; do
    IFS='|' read -r s g p <<< "${entry}"
    rel="${s#scripts/}"
    log_file="${LOG_DIR}/${rel%.sh}.log"
    mkdir -p "$(dirname "${log_file}")"
    RELS+=("${rel}")
    LOGS+=("${log_file}")
    log "▶ GPU ${g} port ${p}: ${rel}  →  ${log_file}"
    ( MASTER_PORT="${p}" bash "${s}" "${g}" 2>&1 | tee "${log_file}" ) &
    PIDS+=($!)
done

log "All ${#PIDS[@]} jobs launched — waiting for completion..."

for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    rc=$?
    if [ $rc -eq 0 ]; then
        log "✓ done : ${RELS[$i]}"
    else
        log "✗ FAILED: ${RELS[$i]} (exit=${rc}, see ${LOGS[$i]})"
        FAILED+=("${RELS[$i]} (exit=${rc})")
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
[ ${#FAILED[@]} -eq 0 ] || exit 1
