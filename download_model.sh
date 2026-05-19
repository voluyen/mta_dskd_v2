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
        --exclude "*.onnx" \
        --exclude "*.onnx_data" \
        --exclude "onnx/*" \
        --exclude "*.tflite" \
        --exclude "*tflite*" \
        --exclude "*.msgpack" \
        --exclude "flax_model*" \
        --exclude "tf_model*" \
        --exclude "*.h5" \
        --exclude "rust_model*" \
        --exclude "*.ot" \
        --exclude "*.gguf" \
        --exclude "*.ggml" \
        --exclude "openvino/*" \
        --exclude "*.xml" \
        --exclude "*.bin.openvino" \
        --exclude "coreml/*" \
        --exclude "*.mlmodel" \
        --exclude "*.mlpackage" \
        --exclude "*.npz" \
        </dev/null

    # If both safetensors and pytorch_model.bin were downloaded, drop the
    # legacy .bin to save disk (transformers prefers safetensors).
    if compgen -G "${target}/*.safetensors" >/dev/null 2>&1; then
        find "${target}" -maxdepth 2 -name 'pytorch_model*.bin' -delete 2>/dev/null || true
    fi
}

# Student checkpoints referenced by scripts/dolly/*/run_*.sh
download_model "gpt2"                      gpt2      gpt2-base
# download_model "gpt2-medium"               gpt2      gpt2-medium
# download_model "gpt2-xl"                   gpt2      gpt2-xl
# download_model "facebook/opt-2.7b"         opt       opt-2.7b
# download_model "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"  tinyllama tinyllama-1.1B  --all
download_model "VoCuc/Qwen1.5_1.8B_SFT_Dolly"  qwen      Qwen1.5_1.8B_SFT_Dolly
