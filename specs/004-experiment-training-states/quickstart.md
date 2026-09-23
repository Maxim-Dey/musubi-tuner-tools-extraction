# Quickstart: Validate Portable Experiment States

This is a local validation guide for the Stage 3 implementation. It uses small temporary fixtures and CPU Accelerate; it does not run the Qwen model, GPU training, downloads or remote transfer. See the [CLI contract](contracts/experiment-cli.md) and [state contract](contracts/experiment-states.md) for exact behavior.

## Prerequisites

- Python 3.12 environment with this repository's existing dependencies.
- Repository root as the current directory for the test commands below.
- Stage 1/2 validation tests passing or their known baseline failures recorded separately.

In PowerShell:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:WANDB_MODE = 'disabled'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$python = 'C:/Users/inbox/Desktop/musubi-tuner-flux2dev-lora/.venv/Scripts/python.exe'
```

## 1. Check the public commands

After implementation, inspect the actual accepted options:

```powershell
& $python -B -m musubi_tuner.qwen_image_train_network --help
& $python -B -m musubi_tuner.qwen_image_cache_latents --help
& $python -B -m musubi_tuner.qwen_image_cache_text_encoder_outputs --help
```

Expected: training accepts `--experiment_dir` with existing `--config_file`; each cache command accepts `--train_config`, `--experiment_dir`, and existing `--dataset_config`. The cache commands do not claim trainer `--config_file`.

The intended path syntax after supplying **real** model files and captioned images is:

```text
trainer:  --config_file C:/moved/portrait/train.toml
latents:  --train_config C:/moved/portrait/train.toml --dataset_config train-dataset.toml --vae <real VAE path>
latents:  --train_config C:/moved/portrait/train.toml --dataset_config val-dataset.toml --vae <real VAE path>
text:     --train_config C:/moved/portrait/train.toml --dataset_config train-dataset.toml --text_encoder <real text model path>
text:     --train_config C:/moved/portrait/train.toml --dataset_config val-dataset.toml --text_encoder <real text model path>
```

`train.toml` can set `experiment_dir = "."`; relative dataset/cache/model paths then use the resolved root. An absolute `--experiment_dir` may replace `--train_config` for cache commands. Do not execute the example caching/training invocations locally without the corresponding real user data and models.

## 2. Run focused CPU acceptance

These test modules are implementation targets. They should create their own temporary data, a small LoRA-compatible model, and a two-rank CPU Accelerate state round trip:

```powershell
& $python -B -m pytest -p no:cacheprovider -q --tb=short tests/test_qwen_image_experiment_paths.py tests/test_qwen_image_experiment_states.py
& $python -B -m pytest -p no:cacheprovider -q --tb=short tests/test_qwen_image_validation_resume.py tests/test_qwen_image_validation_training.py tests/test_qwen_image_training_invariants.py
```

Inspect failures and exit codes, not only console lines. Run affected configuration/cache legacy tests as well and compare any pre-existing failure against baseline; do not mark it as passed.

## 3. Confirm the state and path scenarios

The focused tests must prove all of the following with real filesystem inspection:

1. Move/rename a temporary experiment root and invoke both parser/preflight paths from another CWD. All relative paths resolve inside the new root; Stage 1 validation fingerprint remains unchanged. An absolute input stays absolute. No output is written at the old root.
2. Use both cache parsers with `val-dataset.toml`; each sees both role-labelled sources and rejects a missing caption/cache collision before model load. The unroled training TOML remains distinct.
3. Verify omitted/`fp32` save precision succeeds and `bf16`/`fp16` plus explicit `save_last_n_steps_state` fail before model construction. Without `experiment_dir`, existing CWD and save paths remain.
4. Save a small CPU model at a step with coincident periodic/final/new-best/sample reasons. Inspect **one** `<output_name>-step-<X>` directory across current and best, exactly one `model.safetensors`, optimizer/scheduler files, each rank RNG file, both JSON sidecars, and one PNG set under `samples/`. Check no standalone adapter or shared `output/sample` copy.
5. Load that exact `model.safetensors` through the Qwen LoRA adapter path and load the full package through Accelerate. Compare FP32 parameter tensors, optimizer/scheduler, each rank's Python/NumPy/torch RNG state, absolute step, and the next controlled update with an uninterrupted reference. Repeat from current and best locations.
6. Exercise a sample-only step and a new best at step 0 with `sample_at_first=false`. Each produces one complete package and one sample set when sampling is enabled; disabled sampling performs no generation.
7. Test strict best improvement, equal score, nonfinite/failed validation, sample/save write failure, and an older-current resume with a later best. The previous best remains valid after failures; its metric is never assigned to other weights.
8. At current step `X=12` with `save_last_n_steps=8`, keep an owned current package at step 4 and remove an owned one at step 3 as entire folders; keep an older best and every unrelated file. Verify no duplicate physical package for a step.

## 4. Review documentation and report limits

After code implementation, compare `README.ru.md` and `docs/qwen_image.md` with the actual parsers, package names, sample behavior, FP32 policy, all-rank resume and retention. Report the CPU commands/results and any known baseline failures. Record once that full-model GPU operation was not run locally; do not treat these tests as H200 performance or numerical verification.
