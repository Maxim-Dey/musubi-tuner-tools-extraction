# Quickstart and Validation Guide

This guide describes the intended retained workflow after implementation. The current planning phase has not corrected the two template references or changed the code. Commands that load models below are instructions for the later private-server stage, not commands to execute locally during planning or implementation.

## Prepare User Inputs

Use a compatible existing Python environment (`>=3.10,<3.13`) with the repository's retained dependencies and the selected PyTorch/backend configuration. The example needs bitsandbytes for AdamW8bit and TensorBoard for logging. No specific GPU model is required by the configuration contract. Weight compatibility and actual device capacity are checked in the later operational stage.

Provide original Qwen-Image DiT, original RGB VAE, Qwen2.5-VL weights, and captioned images. The text loader also uses the tokenizer assets for `Qwen/Qwen-Image`, subfolder `tokenizer`; prepare them in the server environment. Do not invoke that loader during local contract checks.

From the repository root, edit the existing three templates:

| File | Required user replacements / retained references |
| --- | --- |
| `config_for_qwen_image_lora/train.toml` | Replace `dit`, `vae`, `text_encoder` with actual server paths. Set `dataset_config="config_for_qwen_image_lora/dataset.toml"`, `sample_prompts="config_for_qwen_image_lora/sample_prompts.txt"` (implementation's approved reference correction). Choose output name/locations as needed |
| `config_for_qwen_image_lora/dataset.toml` | Replace `image_directory` with captioned images; choose a cache directory. Each image has its basename `.txt` caption, or use the existing image JSONL source format. Add dataset sections/overrides if needed |
| `config_for_qwen_image_lora/sample_prompts.txt` | Replace `TOK` and scene descriptions; choose prompt count, valid dimensions, seeds, steps, CFG and flow shift |

Keep `model_version="original"` and `network_module="networks.lora_qwen_image"` for this example. The template already contains the separately approved `lr_scheduler="constant_with_warmup"`, `lr_warmup_steps=200`. Do not revert it to constant while retaining positive warmup.

`qwen_image_lora/output`, `qwen_image_lora/logs` and `qwen_image_lora/cache/train` are valid independent output defaults; they need not be renamed to match the config directory or pre-exist. Paths are relative to the working directory unless absolute. Resolution, rank, precision and run duration remain user choices within existing supported constraints; template numbers are not fixed program requirements.

## Later Server Stage: Both Caches and Training

Run from the repository root in the prepared server environment. The following examples use a POSIX shell; replace all `/srv/...` paths and use the same VAE/text weights as in training TOML.

```bash
export PYTHONPATH="$PWD/src"
python qwen_image_cache_latents.py --dataset_config config_for_qwen_image_lora/dataset.toml --vae /srv/models/qwen_image_vae.safetensors --model_version original
python qwen_image_cache_text_encoder_outputs.py --dataset_config config_for_qwen_image_lora/dataset.toml --text_encoder /srv/models/qwen_2.5_vl_7b.safetensors --model_version original
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 qwen_image_train_network.py --config_file config_for_qwen_image_lora/train.toml
```

The bf16 launcher setting agrees with the example TOML; if changing supported precision, keep launcher and training configuration consistent. Explicit training CLI arguments override TOML; omitted options preserve it. Existing store-true flags do not provide a negative CLI override for a true TOML field—edit that field to disable it.

Neither cache command reads `train.toml` or accepts `--config_file`. Both accept applicable `--device`, `--num_workers` and `--batch_size`; the latter limits cache processing chunks. Text caching separately accepts `--fp8_vl`. Latent caching does not support an explicit `--vae_dtype` or the old Hunyuan tiling/chunk arguments.

On subsequent cache passes, `--skip_existing` skips existing files by existence only; it does not verify caption/model/config freshness. Recreate the affected caches after changing their inputs. `--keep_cache` retains files absent from the current dataset that default cache cleanup would remove. Use dataset-appropriate cache locations.

Expected artifacts and observations:

- Both latent and embedding safetensors are present in the selected dataset cache directory, with the formats in [data-model.md](data-model.md).
- Training consumes paired caches; an empty effective dataset is an early error. Existing missing-text-cache item warnings/skips are not evidence of successful complete caching.
- Samples appear under `<output_dir>/sample` at configured initial/step/epoch events. The supplied `sample_at_first=false` means no initial sample; `sample_every_n_steps=200` schedules later examples.
- Logs use the configured backend/directory. `log_with="tensorboard"` requires its installed dependency and a logging directory; existing wandb/all options retain their own prerequisites.

## Later Server Stage: Save and Resume

With the supplied example settings, step checkpoints use names such as `qwen_image_lora-step00000200.safetensors`. Separate Accelerate state directories use `qwen_image_lora-step00000200-state`. Final outputs are `qwen_image_lora.safetensors` and, with state saving enabled, `qwen_image_lora-state`.

`save_every_n_steps` controls the example cadence. `save_last_n_steps` and `save_last_n_steps_state` are independent retention windows, not a count of files; preserve the existing boundary/fallback rules. Actual retained files depend on the reached save boundary.

Resume from an existing compatible **state directory**, adjusting paths/output settings to the actual run:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 qwen_image_train_network.py --config_file config_for_qwen_image_lora/train.toml --resume qwen_image_lora/output/qwen_image_lora-step00000200-state
```

The current implementation restores saved adapter/optimizer/scheduler/RNG through Accelerate but restarts the trainer's local epoch/global-step counters and does not skip previously consumed batches. This plan preserves that baseline; do not interpret `max_train_steps` as an automatically inferred remaining budget or claim exact data-cursor continuation. Consider the existing checkpoint/sample naming behavior when choosing resume output locations.

To initialize a new run from an adapter only, use the existing `--network_weights /srv/adapters/initial.safetensors` instead; that does not restore optimizer/scheduler/RNG state. `base_weights` is yet another existing path for merging adapters into the base before training, not resume.

## Local Verification Without GPU, Weights or Network

These checks require a compatible dependency-complete CPU environment. The current checkout has no `.venv`; the inspected runtimes lack the ML dependencies, so real training/cache imports, parser integration and pytest regressions were **unavailable during planning**. The actual source checks and failed-import evidence are in [research.md](research.md). Do not install/download dependencies or claim these commands passed merely because they are documented.

For local PowerShell checks in an already prepared environment, select its Python executable as `python`, then set:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:WANDB_MODE = 'disabled'
```

Before changing substantial logic, capture the results of available existing checks:

```powershell
python -B qwen_image_cache_latents.py --help
python -B qwen_image_cache_text_encoder_outputs.py --help
python -B qwen_image_train_network.py --help
python -B -m pytest -p no:cacheprovider -q tests/test_save_precision.py tests/test_lora_dtype_bridging.py tests/test_grad_metrics.py tests/test_krea2_timesteps.py tests/test_ideogram4_timesteps.py
```

Help/import checks must import real modules; no fake torch/transformers/accelerate replacements. Do not invoke a cache or trainer without `--help` as a local smoke test. Record the original revision, template hashes, effective parser outputs and affected numerical/artifact behavior before extraction.

During implementation, add only focused regression checks for the changed contracts, using the real readers and existing helpers. Suggested cohesive test files are `tests/test_qwen_image_config.py`, `tests/test_qwen_image_dataset_cache.py` and `tests/test_qwen_image_training_invariants.py`; these are planned files, not files created in this phase. The implementation test command can then include them together with the retained existing checks.

| Scenario | Expected local evidence |
| --- | --- |
| All template keys + CLI override | Actual parser and TOML reader; temporary replacement inputs; assert final effective configuration, retained bool/numeric semantics, corrected references and non-template values |
| Unsupported TOML separately | Unknown keys, wrong types, invalid choices, excluded versions/modules and conflicting edit/original selectors fail before loader sentinels; no parser/reader stubs |
| Scheduler | Real CPU optimizer and existing scheduler factory: constant/0, constant/200 rejection, warmup schedule progression and ratio semantics; no training loop |
| Dataset/cache | Tiny real images/captions/JSONL and tensors; schema/blueprint, buckets/repeats, cache writer/reader metadata/shapes/dtype replacement, skip/keep behavior; no claimed encoding through real weights |
| Samples/logs/math | Real prompt formats/trigger/image saver/metrics; narrow callback only where generation is unavailable. Check original attention/packing/loss/LoRA gradients and state nonmutation at exercised boundaries |
| Save/resume | Real small adapter and CPU Accelerate state serialization/restoration, retention boundaries and filter hooks; record counter reset separately, without full-training simulation |
| Cleanup/docs | Retained imports and short/qualified dynamic adapters resolve; no excluded code/dispatch remains; README commands and shipped references agree with parsers; dependency manifests/any existing locks agree |

Validate corrected template links during implementation; they deliberately remain unedited in this planning phase. Use [the contract](contracts/cli-and-config.md) for supported values and early errors. Missing local prerequisites remain an explicit unperformed check, and cannot be counted toward completion of implementation.
