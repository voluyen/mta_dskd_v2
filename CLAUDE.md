# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

DSKD v2 — Dual-Space Knowledge Distillation for LLMs (paper: arXiv 2504.11426). This tree adds **MTA** (Multi-layer Token-Aligned) span/feature distillation on top of upstream DSKD v2. Supports same- and cross-tokenizer KD, off- and on-policy training, multiple divergences, LoRA, and DeepSpeed ZeRO.

The parent repo at `../../../` (`d:\cross-tokenizer-scoring`) is an unrelated paper implementation; do not pull instructions from its `CLAUDE.md` into this subproject.

## Setup

No `requirements.txt` is included; install manually (versions per upstream README):

```bash
pip install "torch>=2.0.0" "transformers>=4.30.0" "deepspeed>=0.10.0" "peft>=0.5.0" datasets rouge-score accelerate
```

Models are expected at `model_hub/<type>/<name>/` (e.g. `model_hub/gpt2/gpt2-base/`); teacher checkpoints can be HF Hub IDs (e.g. `VoCuc/Qwen1.5_1.8B_SFT_Dolly`). Data lives under `data/<task>/` as `train.jsonl` / `dev.jsonl` / optional `test.jsonl` with `{"prompt": ..., "response": ...}` entries.

## Running training

Training is launched via `torchrun` + DeepSpeed wrappers in `scripts/dolly/<student>/`:

```bash
bash scripts/dolly/gpt2-120M/run_mta_dskdv2_eta.sh        # all visible GPUs
bash scripts/dolly/gpt2-120M/run_mta_dskdv2_eta.sh 0,1    # restrict CUDA_VISIBLE_DEVICES
```

Each script sets `BASE_PATH=.` and expects to be invoked **from the project root** (`MTA_DSKD_v2/`), not from inside `scripts/`. Outputs land in `outputs/<ckpt_type>/<ckpt_name>/<task>/<setting>/` with `train.log` tee'd alongside checkpoints.

Three script variants per student:
- `run_dskdv2.sh` — DSKD v2 with cross-model attention (same-tokenizer baseline).
- `run_dskdv2_eta.sh` — DSKD v2 with Exact Token Alignment (`--criterion dual_space_kd_v2_with_eta`), the cross-tokenizer path.
- `run_mta_dskdv2_eta.sh` — adds `--MTA-mode` plus layer-mapping flags (`--teacher_layer_mapping`, `--student_layer_mapping`, `--split_layer_mapping`, `--w-span-loss`).

DeepSpeed config is auto-selected from `--model-dtype` (`bf16` → `configs/deepspeed/ds_config_bf16.json`, `fp16` → `ds_config.json`, `fp32` → `ds_config_fp32.json`). ZeRO-3 variant exists at `configs/deepspeed/ds_config_zero3_bf16.json` if you need parameter sharding (the save path in `distillation.py:389` handles ZeRO-3 gather).

## Evaluation

`code/evaluate_main.py` is the standalone eval entry point (uses `code/evaluate.py` + `code/rouge_metric.py`). Dev-set eval and ROUGE-L generation also run inline during training when `--do-valid --eval-gen` are set; checkpoints are pruned to top `--keep-best-n-checkpoints` by ROUGE-L (or eval loss if `--eval-gen` is off).

## Quick syntax check (no GPU)

```bash
python -c "import py_compile, pathlib; [py_compile.compile(str(p), doraise=True) for p in pathlib.Path('code').rglob('*.py')]"
```

There is no test suite.

## Architecture

### Entry point flow (`code/distillation.py`)
`main()` → parses `arguments.py` → builds `Distiller` → `prepare_dataset()` → `deepspeed.initialize(model=distiller, ...)` → `finetune()`. The `Distiller` itself is the `nn.Module` passed to DeepSpeed; `model.module.student_model` is the wrapped HF causal LM.

### Distiller (`code/distiller.py`)
Owns both the student and (optional) teacher HF models + tokenizers, plus all auxiliary trainable parameters:
- `t2s_projector` / `s2t_projector` — dual-space projectors instantiated when `--criterion` contains `dual_space`. Optional logit-identity init via `--init-t2s-projector` / `--init-s2t-projector`.
- `mta_projector_list` — one `nn.Linear(student_hidden, teacher_hidden)` per entry in `--teacher_layer_mapping`, used by MTA span/feature losses.
- Optional teacher↔student token/id mappings loaded from JSON (used by `min_edit_dis_kld` only).
- `add_optimizer_param_group` injects projector params with their own `--projector-lr`.

### Criterion registry (`code/criterions/__init__.py`)
`build_criterion(args)` dispatches `args.criterion` to one of:
`cross_entropy`, `various_divergence`, `dual_space_kd`, `dual_space_kd_v2`, `dual_space_kd_v2_with_eta`, `universal_logit_distillation`, `min_edit_dis_kld`. The cross-model-attention variant is registered but currently commented out. Each criterion's `forward(distiller, batch, logging_output)` returns `(loss, logging_output)`; `--kd-objective` (`forward_kl`, `reverse_kl`, `js_divergence`, `skewed_forward_kl`, `skewed_reverse_kl`, `adaptive_kl`) selects the divergence inside `various_divergence.py`.

The MTA-specific span/feature distillation losses are layered on inside the criterion when `--MTA-mode` is set; the layer mapping triples (`teacher_layer_mapping`, `student_layer_mapping`, `split_layer_mapping`) define which intermediate hidden states are aligned and weighted by `--w-span-loss`.

### Data pipeline (`code/data_utils/distill_datasets.py`)
`DistillDataset(args, split, student_tokenizer, teacher_tokenizer)` produces a per-batch dict with at least `input_batch`, `label_batch`, `teacher_input_batch`, `teacher_label_batch`, `prompt_batch`. The training loop expects `label_batch["label"]` to be `-100`-masked over prompt tokens; token-count normalization is global across grad-accum × DP world size (`distillation.py:284-289`). When on-policy is active, parallel `op_*_batch` keys are populated from student-generated rollouts re-tokenized for the teacher.

### On-policy distillation
Enabled by `--on-policy` (commented out in default scripts). After `--on-policy-after-n-epochs`, each batch with prob `--stu-gen-ratio` is replaced by student-generated continuations (or teacher-mixed for `--criterion minillm`). Same-tokenizer vs. cross-tokenizer paths diverge at `distillation.py:193` — cross-tokenizer re-runs the teacher tokenizer on decoded student text and rebuilds aligned `op_teacher_input_batch`.

### Checkpoint saving
Two paths: (1) ZeRO-3 collects partitioned params via `deepspeed.zero.GatheredParameters` before `student_model.save_pretrained` (handles `tie_word_embeddings` quirk for Qwen2-0.5B style models); (2) ZeRO-≤2 saves directly. `model.module.projectors.state_dict()` is dumped to `projector.pt` whenever the distiller has projectors. Use `--only-save-projector` to skip the base model (handy for LoRA/projector-only runs).

### Important arg conventions (`code/arguments.py`)
- `--save-interval -1` / `--eval-interval -1` mean "once per epoch" (resolved in `main()` to `train_iters_per_epoch`).
- `--topk-vocab -1` disables top-k truncation in dual-space losses.
- `--teacher-model-fp16` loads the teacher in fp16 regardless of `--model-dtype`.
- `--gradient-checkpointing` is available but disabled in the default scripts.
