#!/bin/bash
# =============================================================================
# MTA_DSKD_v2 — one-shot bootstrap + run-all script
#
# Usage (from the project root, i.e. the directory containing this file):
#   bash run.sh                    # uses all visible GPUs
#   bash run.sh 0,1,2,3            # restrict to specific GPUs
#
# What it does:
#   1. Creates / activates a Python venv (override with VENV_DIR=...).
#   2. Installs all Python dependencies (torch, transformers, deepspeed, ...).
#   3. Downloads every student model into model_hub/<type>/<name>/
#      (teachers are HF Hub IDs and stream directly via transformers).
#   4. Runs every shell script under scripts/ sequentially, with logs
#      tee'd into logs/<script-relpath>.log.
#
# Environment overrides:
#   VENV_DIR           default: ./.venv         set to "" to skip venv creation
#   SKIP_INSTALL=1     skip pip install step
#   SKIP_DOWNLOAD=1    skip model download step
#   ONLY_SCRIPTS=...   run only scripts whose relative path matches this glob
#                      (e.g. ONLY_SCRIPTS='dolly/gpt2-120M/*.sh')
#   STOP_ON_ERROR=1    abort after the first failing script (default: continue)
#   HF_TOKEN=...       passed through to huggingface-cli for gated repos
# =============================================================================

set -u
set -o pipefail

# -------- locate project root (the directory of this script) ----------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

GPUS="${1:-}"
if [[ -n "${GPUS}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPUS}"
fi

VENV_DIR="${VENV_DIR-${PROJECT_ROOT}/.venv}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
ONLY_SCRIPTS="${ONLY_SCRIPTS:-}"

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

log() { echo -e "\n[run.sh] $*"; }

# ---------------------------------------------------------------------------
# 1. Python environment
# ---------------------------------------------------------------------------
if [[ -n "${VENV_DIR}" ]]; then
    if [[ ! -d "${VENV_DIR}" ]]; then
        log "Creating venv at ${VENV_DIR}"
        python3 -m venv "${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    log "Using python: $(which python) ($(python --version 2>&1))"
fi

# ---------------------------------------------------------------------------
# 2. Install dependencies
# ---------------------------------------------------------------------------
if [[ "${SKIP_INSTALL}" != "1" ]]; then
    log "Upgrading pip / wheel / setuptools"
    python -m pip install --upgrade pip wheel setuptools

    log "Installing Python dependencies"
    # NB: torch is intentionally pinned loose so this works on most CUDA images.
    # On a fresh CUDA 12.1 server, you may want:
    #   pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    python -m pip install \
        "torch>=2.0.0" \
        "transformers>=4.40.0" \
        "deepspeed>=0.10.0" \
        "peft>=0.5.0" \
        "accelerate>=0.27.0" \
        "datasets" \
        "rouge-score" \
        "nltk" \
        "sentencepiece" \
        "protobuf" \
        "tqdm" \
        "huggingface_hub"

    python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" || true
else
    log "SKIP_INSTALL=1 — skipping pip install"
fi

# ---------------------------------------------------------------------------
# 3. Download student models into model_hub/<type>/<name>/
#    (Teacher checkpoints are HF Hub IDs — transformers downloads them
#     on-the-fly during training; we only mirror students locally because
#     the scripts hard-code the model_hub/<type>/<name> path.)
# ---------------------------------------------------------------------------
if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
    log "Preparing student models under model_hub/"

    # Silence interactive prompts from huggingface_hub (update check, telemetry).
    export HF_HUB_DISABLE_TELEMETRY=1
    export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
    export HF_HUB_DISABLE_PROGRESS_BARS=0
    export DO_NOT_TRACK=1
    export PYTHONUNBUFFERED=1

    # `huggingface-cli` is deprecated in huggingface_hub >= 1.x → use `hf`.
    # Fallback to the legacy CLI if `hf` is somehow unavailable.
    if command -v hf >/dev/null 2>&1; then
        HF_CLI=(hf)
    else
        HF_CLI=(huggingface-cli)
    fi
    log "Using HF CLI: ${HF_CLI[*]}"

    # All student / teacher checkpoints used here are public — no login needed.

    # Args: <hf_id> <type_subdir> <name_subdir> [--all]
    # By default, skip ONNX/TFLite/TF/Flax/Rust/GGUF/OpenVINO/Core ML weights.
    # Pass --all to download every file in the repo (e.g. for tinyllama).
    download_model() {
        local hf_id="$1"
        local target="${PROJECT_ROOT}/model_hub/$2/$3"
        local mode="${4:-filtered}"
        if [[ -f "${target}/config.json" ]]; then
            log "  ✓ already present: ${target}"
            return 0
        fi
        mkdir -p "${target}"

        if [[ "${mode}" == "--all" ]]; then
            log "  ↓ downloading ${hf_id} → ${target}  (full repo, no excludes)"
            # </dev/null prevents the "Do you want to update now?" prompt from blocking.
            "${HF_CLI[@]}" download "${hf_id}" --local-dir "${target}" </dev/null
            return
        fi

        log "  ↓ downloading ${hf_id} → ${target}  (PyTorch weights + tokenizer only)"
        "${HF_CLI[@]}" download "${hf_id}" --local-dir "${target}" \
            --exclude "*.onnx" "*.onnx_data" "onnx/*" \
            --exclude "*.tflite" "*tflite*" \
            --exclude "*.msgpack" "flax_model*" \
            --exclude "tf_model*" "*.h5" \
            --exclude "rust_model*" "*.ot" \
            --exclude "*.gguf" "*.ggml" \
            --exclude "openvino/*" "*.xml" "*.bin.openvino" \
            --exclude "coreml/*" "*.mlmodel" "*.mlpackage" \
            --exclude "*.msgpack" "*.npz" \
            </dev/null

        # If both safetensors and pytorch_model.bin were downloaded, drop the
        # legacy .bin to save disk (transformers prefers safetensors).
        if compgen -G "${target}/*.safetensors" >/dev/null 2>&1; then
            find "${target}" -maxdepth 2 -name 'pytorch_model*.bin' -delete 2>/dev/null || true
        fi
    }

    # Student checkpoints referenced by scripts/dolly/*/run_*.sh
    download_model "gpt2"                      gpt2      gpt2-base
    download_model "gpt2-medium"               gpt2      gpt2-medium
    download_model "gpt2-xl"                   gpt2      gpt2-xl
    download_model "facebook/opt-2.7b"         opt       opt-2.7b
    download_model "TinyLlama/TinyLlama_v1.1"  tinyllama tinyllama_v1.1  --all

    log "Pre-warming teacher tokenizers (optional, speeds up first run)"
    python - <<'PY' || true
from transformers import AutoTokenizer
for hf_id in [
    "VoCuc/Qwen1.5_1.8B_SFT_Dolly",
    "VoCuc/Qwen2.5-7B-Instruct-Dolly-SFT",
    "VoCuc/Mistral7B_Dolly_SFT",
]:
    try:
        AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        print(f"  ok: {hf_id}")
    except Exception as e:
        print(f"  skip: {hf_id} ({e})")
PY
else
    log "SKIP_DOWNLOAD=1 — skipping model download"
fi

# ---------------------------------------------------------------------------
# 4. Run every script under scripts/
# ---------------------------------------------------------------------------
export PYTHONPATH="${PROJECT_ROOT}"
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_DISABLED=True

mapfile -t ALL_SCRIPTS < <(find "${PROJECT_ROOT}/scripts" -type f -name '*.sh' | sort)

if [[ -n "${ONLY_SCRIPTS}" ]]; then
    FILTERED=()
    for s in "${ALL_SCRIPTS[@]}"; do
        rel="${s#${PROJECT_ROOT}/scripts/}"
        # shellcheck disable=SC2053
        if [[ "${rel}" == ${ONLY_SCRIPTS} ]]; then
            FILTERED+=("${s}")
        fi
    done
    ALL_SCRIPTS=("${FILTERED[@]}")
fi

log "Will execute ${#ALL_SCRIPTS[@]} script(s):"
for s in "${ALL_SCRIPTS[@]}"; do echo "    - ${s#${PROJECT_ROOT}/}"; done

FAILED=()
for s in "${ALL_SCRIPTS[@]}"; do
    rel="${s#${PROJECT_ROOT}/scripts/}"
    log_file="${LOG_DIR}/${rel%.sh}.log"
    mkdir -p "$(dirname "${log_file}")"

    log "▶ ${rel}  (log: ${log_file#${PROJECT_ROOT}/})"
    # All training scripts assume CWD == project root and BASE_PATH=. (relative).
    if bash "${s}" ${GPUS:+"${GPUS}"} 2>&1 | tee "${log_file}"; then
        log "✓ done: ${rel}"
    else
        log "✗ FAILED: ${rel} (see ${log_file})"
        FAILED+=("${rel}")
        if [[ "${STOP_ON_ERROR}" == "1" ]]; then
            log "STOP_ON_ERROR=1 — aborting."
            break
        fi
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

# ---------------------------------------------------------------------------
# 5. Archive all training outputs (checkpoints, logs) into a single zip
# ---------------------------------------------------------------------------
RESULT_ZIP="${PROJECT_ROOT}/results_$(date +%Y%m%d_%H%M%S).zip"
log "Zipping outputs/ + logs/ → ${RESULT_ZIP}"
if command -v zip >/dev/null 2>&1; then
    ( cd "${PROJECT_ROOT}" && zip -r "${RESULT_ZIP}" outputs logs >/dev/null )
else
    # Fallback when `zip` is not installed (common on minimal images).
    RESULT_ZIP="${RESULT_ZIP%.zip}.tar.gz"
    ( cd "${PROJECT_ROOT}" && tar -czf "${RESULT_ZIP}" outputs logs )
fi
log "Archive ready: ${RESULT_ZIP}"

[[ ${#FAILED[@]} -eq 0 ]] || exit 1
