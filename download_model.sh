log() { echo -e "\n[download_model.sh] $*"; }
log "Preparing student models under model_hub/"

export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export PYTHONUNBUFFERED=1

# Args: <hf_id> <type_subdir> <name_subdir> [--all]
# Uses Python snapshot_download so ignore_patterns is always enforced correctly.
download_model() {
    local hf_id="$1"
    local target="model_hub/$2/$3"
    local mode="${4:-filtered}"
    if [[ -f "${target}/config.json" ]]; then
        log "  ✓ already present: ${target}"
        return 0
    fi
    mkdir -p "${target}"
    log "  ↓ downloading ${hf_id} → ${target}"
    python - <<PY
from huggingface_hub import snapshot_download

ignore = None if "${mode}" == "--all" else [
    "*.onnx", "*.onnx_data", "onnx/*",
    "*.tflite", "*tflite*",
    "*.msgpack", "flax_model*",
    "tf_model*", "*.h5",
    "rust_model*", "*.ot",
    "*.gguf", "*.ggml",
    "openvino/*", "*.xml", "*.bin.openvino",
    "coreml/*", "*.mlmodel", "*.mlpackage",
    "*.npz",
]
snapshot_download(
    repo_id="${hf_id}",
    local_dir="${target}",
    ignore_patterns=ignore,
)
PY

    # Drop legacy .bin if safetensors are present.
    if compgen -G "${target}/*.safetensors" >/dev/null 2>&1; then
        find "${target}" -maxdepth 2 -name 'pytorch_model*.bin' -delete 2>/dev/null || true
    fi
}

# Student checkpoints referenced by scripts/dolly/*/run_*.sh
download_model "gpt2"                      gpt2      gpt2-base
download_model "gpt2-medium"               gpt2      gpt2-medium
download_model "gpt2-xl"                   gpt2      gpt2-xl
download_model "facebook/opt-2.7b"         opt       opt-2.7b
download_model "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"  tinyllama tinyllama-1.1B