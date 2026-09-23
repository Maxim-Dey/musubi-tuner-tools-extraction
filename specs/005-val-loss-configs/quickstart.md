# Quickstart: Example Acceptance and User Run Sequence

This guide validates the [Stage 4 example](../../qwen_image_lora_val_example/) and records the intended user command sequence. The example is a template: supply real captioned images, original Qwen-Image model files and a replacement for `TOK` before an operational run. The training command prepares caches. The [example CLI contract](contracts/example-cli.md) is the authority for accepted flags. Local acceptance uses small temporary CPU fixtures and does not run the real model.

## 1. Check the physical example

The completed example must contain `train.toml`, `train-dataset.toml`, `val-dataset.toml`, and `sample_prompts.txt` under one independent `qwen_image_lora_val_example/` folder. Create the declared dataset/cache/output layout without filling it with invented images or caches. Supply an image and matching UTF-8 `.txt` caption for every training and validation image. The training command builds both cache types for all configured sources. `val_unfamiliar` must never appear in the train TOML.

Replace the three explicit absolute paths in `train.toml` with the server original BF16 DiT, Qwen VAE and text encoder files. The DiT is not a preconverted FP8 file. Replace `TOK` in both sample prompts with the intended trigger. Keep `lr_warmup_steps=200` as an integer. Keep BF16 compute and `save_precision="fp32"`: the one adapter file must retain FP32 values for exact resume. To disable sampling, remove both `sample_prompts` and `sample_every_n_steps` entries; do not use an empty prompt string.

## 2. Inspect accepted commands locally

From the repository root, using an existing Python 3.12 environment with project dependencies, run:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:WANDB_MODE = 'disabled'
$python = 'python'  # Python 3.12 from the activated project environment
& $python -B -m musubi_tuner.qwen_image_train_network --help
& $python -B -m musubi_tuner.qwen_image_cache_latents --help
& $python -B -m musubi_tuner.qwen_image_cache_text_encoder_outputs --help
```

Expected: the trainer accepts `--config_file` and obtains its datasets and model paths from the effective training TOML. The cache commands remain available for manual use but are not required for an ordinary run.

## 3. Operational command sequence after supplying real inputs

These commands are a documented later server procedure. They are not part of the local CPU check. Set `REPO` and `ROOT` to real absolute paths after configuring model paths in `train.toml`. The shell starts outside the repository and experiment to prove that invocation CWD is not the path anchor.

```bash
REPO=/workspace/musubi-tuner-tools-extraction
ROOT=/workspace/qwen_image_lora_val_example
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd /tmp

python "$REPO/qwen_image_train_network.py" --config_file "$ROOT/train.toml"
tensorboard --logdir "$ROOT/output/tensorboard"
```

The trainer checks both cache types for every declared dataset. It creates missing files and repairs stale validation caches before loading training weights. The literal placeholder paths shown above must be replaced; these commands are not evidence of a run.

## 4. Move the root and resume

Moving or renaming the whole prepared root, including its datasets, caches and output, changes only the external `ROOT` value. Its internal relative paths still refer to the moved folder. For example, after moving it to `/workspace/portrait-renamed`, run from another CWD:

```bash
ROOT=/workspace/portrait-renamed
cd /tmp
python "$REPO/qwen_image_train_network.py" --config_file "$ROOT/train.toml"
```

Use an actual complete published package and its exact absolute step. For example, if step 200 is current and step 400 is best:

```bash
CURRENT="$ROOT/output/current_training_states/qwen_image_lora-step-200"
BEST="$ROOT/output/val_training_states/val-loss/qwen_image_lora-step-400"
accelerate launch --mixed_precision bf16 "$REPO/qwen_image_train_network.py" --config_file "$ROOT/train.toml" --resume "$CURRENT"
accelerate launch --mixed_precision bf16 "$REPO/qwen_image_train_network.py" --config_file "$ROOT/train.toml" --resume "$BEST"
```

Choose one of the two resume commands for a run; they are alternatives. The package must exist and pass the Stage 3 integrity checks. Keep the validation images, captions, source-bound caches, and fixed noise controls intact so the validation identity remains compatible. Resume at saved absolute step `s` validates once at `s`; the first new training-loss point is `s+1`. `max_train_steps=B` is this invocation's completed-update budget (`B=1600` in the example), so its final absolute step is `s+B`. Exact data-loader position restoration is not promised.

## 5. Local CPU acceptance

After implementation, the example parser/reader checks and the composed controlled integration should be run with the existing environment:

```powershell
& $python -B -m pytest -p no:cacheprovider -q --tb=short tests/test_qwen_image_experiment_paths.py tests/test_qwen_image_validation_inputs.py
& $python -B -m pytest -p no:cacheprovider -q --tb=short tests/test_qwen_image_validation_training.py tests/test_qwen_image_validation_resume.py tests/test_qwen_image_experiment_states.py
& $python -B -m pytest -p no:cacheprovider -q --tb=short tests/test_qwen_image_config.py tests/test_qwen_image_dataset_cache.py tests/test_qwen_image_training_invariants.py
```

The Stage 4 implementation must add focused checks of the physical files and a composed CPU chain to these tests, reusing their fixtures. The chain uses a small model and two differently sized validation sets; it proves the fixed SHA-256-seeded 10-by-1 grid, exact levels without applying the training shift, shared weighted loss and image-first means, six tags, periodic/final coalescing, one strict best, one complete FP32 adapter state, and nonzero-step resume. Exercise the 0–1600 due-step grid through the existing scheduling seam; **do not run 1,600 CPU optimizer updates**. A separate short controlled loop covers the config-to-resume integration.

Expected fresh-run validation labels are `0, 200, 400, 600, 800, 1000, 1200, 1400, 1600`, with no duplicate at 1600. Successful events publish exactly `train_eval_loss_mean`, `train_eval_loss_low_noise`, `train_eval_loss_high_noise`, `val_loss_mean`, `val_loss_low_noise`, and `val_loss_high_noise`. Only strict finite improvement of unfamiliar `val_loss_mean` changes the one best. After save at absolute `X`, current packages with `s >= X-1000` remain whole; an older best remains protected. A valid step-0 best receives samples when sampling is enabled, even with `sample_at_first=false`.

Record the exact command results and any unavailable checks. The full H200 run, image quality, GPU memory and throughput belong to later private-server verification and are not claimed by these CPU checks.
