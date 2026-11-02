# Musubi Tuner: Krea2 Raw LoRA

This extraction supports captioned-image LoRA training for **Krea 2 Raw**: latent and text caching, training, Raw or opt-in Turbo sample images during training, logging, adapter/state saving and resume. It retains the Qwen-Image VAE and Qwen3-VL-4B conditioner required by this workflow.

## Environment and model files

Use Python 3.10–3.12 and a PyTorch/torchvision build compatible with your environment. Project dependencies are in `pyproject.toml`; optional CUDA extras select their corresponding PyTorch index. Install the project in your chosen environment, for example with `pip install -e .` after installing compatible PyTorch/torchvision. Development and optional TensorBoard/debug tools are in the `dev` dependency group. Attention backends, alternative optimizers and W&B require their own compatible optional packages.

Provide these local safetensors files:

- Krea 2 Raw DiT.
- Qwen-Image VAE.
- Qwen3-VL-4B-Instruct text encoder, in the official or supported ComfyUI single-file layout.

The encoder resolves its tokenizer using `Qwen/Qwen3-VL-4B-Instruct`; arrange the tokenizer cache before an offline run. The text encoder is needed for text caching and training samples. Model licenses apply separately. Example settings do not establish compatibility with any particular hardware.

Run all commands below **from the repository root**, in the same activated environment. Set the source path once:

```powershell
# PowerShell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
```

```sh
# POSIX shell
export PYTHONPATH="$PWD/src"
```

## Prepare captioned images

Place an image and its UTF-8 caption together, such as `photo01.png` and `photo01.txt`. Each caption describes its corresponding training image. Image JSONL sources are also supported; see [dataset configuration](docs/dataset_config.md).

The supplied experiment uses all 107 training and 20 independent validation images, with paired UTF-8 captions. Preserve this split and the ten existing sample subjects.

- [train.toml](config_for_krea2_raw_lora/train.toml): main 1000-step experiment; verified `dit`, `vae` and `text_encoder` paths are filled during remote preparation. Until then required model paths are unset and the configuration is not ready for training.
- [train-dataset.toml](config_for_krea2_raw_lora/train-dataset.toml): 1024 buckets, train batch 16, one repeat and `cache/train`.
- [val-dataset.toml](config_for_krea2_raw_lora/val-dataset.toml): same preprocessing, batch 1, one repeat and `cache/val`.
- [sample_prompts.txt](config_for_krea2_raw_lora/sample_prompts.txt): ten user subjects, 1024×1024, seed 42, 8 Turbo sampling steps and CFG 1.

With `experiment_mode=true`, paths in the main TOML resolve from its directory; paths inside each dataset TOML resolve from that dataset file. Absolute paths remain absolute. The legacy [dataset.toml](config_for_krea2_raw_lora/dataset.toml) remains an example for ordinary mode, whose paths retain their previous working-directory rules. Module names, model repository IDs and URLs are not rewritten as local paths.

## Caches and fixed validation

`auto_cache=true` prepares only missing compatible train/val caches using the configured VAE and Qwen3-VL encoder, then frees these models before loading DiT. Existing incompatible caches fail explicitly; new experiment modes reject old caches without sufficient source/model/preprocessing provenance. Do not reuse Qwen Image caches as Krea2 caches. With automatic caching disabled, all required compatible caches must already exist.

Preflight checks effective defaults→TOML→CLI settings and data before loading models. It rejects incomplete validation settings, missing captions, empty data, train/val overlap by SHA-256 image content and conflicting validation duplicates. Compatible content copies count once. The fixed validation manifest includes captions, preprocessing/buckets, cache identities and noise/component settings; changes fail on resume and subsequent validation. Moving an unchanged experiment or renaming unchanged image content does not change its noise seed.

Validation requires the explicit complete group `val_dataset_config`, `val_every_n_steps`, `val_seed_noise`, `val_level_noise_n`, `val_seed_noise_n`. Omit the entire group to disable validation. Period is an integer ≥1, seeds are integers, N1 is even ≥2 and N2 ≥1; booleans are rejected. New `experiment_mode` and `auto_cache` default to false.

## Train, metrics and samples

After verified model paths and the environment are prepared:

```sh
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 krea2_train_network.py --config_file config_for_krea2_raw_lora/train.toml
```

The main configuration retains AdamW, constant learning rate 0.0001, rank/alpha 32, bf16, SDPA, gradient checkpointing and training `krea2_shift`. It validates before training at optimizer step 0, every 200 completed optimizer steps and the final step if not already measured. Accumulation microbatches are not optimizer steps. The full training run is started by the user after the short diagnostics.

TensorBoard reports `val_loss_mean`, `val_loss_low_noise` and `val_loss_high_noise` against completed optimizer steps. Each unique image has equal weight. For each image, ten fixed midpoint noise levels between 0.05 and 0.95 and two noise realizations are measured; low means t<0.5 and high means t≥0.5. These measure flow-matching prediction error, independently of visual sampling and CFG. Validation uses no training shift and preserves training RNG/state. See [numerical and reproducibility details](docs/krea2.md#fixed-validation-loss).

Without `turbo_dit`, samples retain the existing Raw model behavior. The supplied main config now opts into the distilled Turbo base for samples with the current trained LoRA; Raw weights are restored before training/validation continues. `--l` sets CFG; without `--n`, the unconditional branch encodes an empty string. `--l 1` disables CFG. The supplied prompts use supported flags only. Samples at a saved step belong to that checkpoint's `samples/`; samples without a checkpoint use separate explicitly numbered output. See [sampling](docs/sampling_during_training.md).

The main config selects `turbo_dit="/workspace/models/Krea-2/diffusion_models/krea2_turbo_bf16.safetensors"` and `turbo_dit_cache=true`. The full Turbo file SHA256 and native BF16 GPU sampling acceptance passed in both resident and streaming modes. Turbo uses fixed mu=1.15; absent per-prompt settings default to 8 steps/CFG1, while explicit overrides remain respected. The ten supplied subjects now explicitly use those values without changing text, seed or dimensions. Only sampling changes: Raw training and deterministic Raw validation retain their previous mathematics and settings. See [Turbo sampling and memory](docs/krea2.md#turbo-samples-during-raw-training).

Install TensorBoard with the existing dev dependency group, for example `uv sync --group dev` with your chosen compatible CUDA extra, or install `tensorboard` in the chosen environment. Then:

```sh
tensorboard --logdir config_for_krea2_raw_lora/output/tensorboard --host 0.0.0.0 --port 6006
```

The verified remote address and persistent service commands are recorded only after remote checks in [feature quickstart](specs/004-krea2-deterministic-val/quickstart.md).

## Checkpoints and resume

With `experiment_mode=true` and `save_state=true`, each completed save produces `output/krea2_raw_lora-step 000200/` containing one FP32 `model.safetensors`, optimizer/scheduler/per-process RNG state, exact train position, manifest and completed validation metadata, plus same-step samples. The adapter loads separately with `--network_weights`; that does not restore optimizer state. Resume uses the complete state directory:

```sh
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 krea2_train_network.py --config_file config_for_krea2_raw_lora/train.toml --resume /absolute/path/to/krea2_raw_lora-step 000200
```

Use an actually completed checkpoint from the user experiment, never a diagnostic state. Model/data/topology settings must remain compatible. Resume restores completed optimizer count and consumed train position, and does not repeat an already recorded validation state. Restoring an older checkpoint branches logging to avoid mixing incompatible later trajectories. Incomplete writes are not valid checkpoints. The final actual step is saved once even when not periodic.

In experiment mode, `save_last_n_steps` alone sets the optimizer-step retention window for the complete checkpoint folder: adapter weights, optimizer/scheduler/RNG state, metadata and samples. The main value is 1000; omitting it keeps all complete checkpoints. Whole outdated folders are removed using the existing interval-boundary rule. Ordinary mode retains its previous adapter/state formats.

Local acceptance and required real GPU/remote diagnostics are tracked separately in [verification.md](specs/004-krea2-deterministic-val/verification.md). Diagnostic files live under [diagnostics](config_for_krea2_raw_lora/diagnostics/) and never replace main settings or user output.

## Optional settings

Applicable existing settings remain configurable even when disabled in the starter:

- Scaled FP8: use `--fp8_base --fp8_scaled` together.
- ConvRot INT8: `--convrot_int8`, with `--convrot_int8_bwd bf16` or opt-in `int8`; do not combine with FP8. Fused kernels require a compatible Triton installation; the standard backward path has an eager fallback.
- [Block swap](docs/block_swap.md), pinned transfers and activation checkpointing/CPU offload.
- Supported attention backends, gradient accumulation, optimizer/scheduler options, LoRA target patterns and LoRA+.
- [torch.compile](docs/torch_compile.md) and applicable Accelerate Dynamo settings.

See [Krea2 details](docs/krea2.md) and `python krea2_train_network.py --help`. Configuration order is defaults → TOML → explicit CLI. Unknown settings and unsupported combinations are errors before model loading.

## Local checks

With development dependencies installed and `PYTHONPATH` set as above:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTHONIOENCODING = 'utf-8'
python -m pytest -q tests
python krea2_cache_latents.py --help
python krea2_cache_text_encoder_outputs.py --help
python krea2_train_network.py --help
```

CPU/import/parser checks do not verify real model training. [Contributing](CONTRIBUTING.md) describes local development checks.

## License

Other code is under the Apache License 2.0. Some code is copied and modified from Diffusers. Retained Qwen-Image VAE files include the Qwen-Image/Wan/Hugging Face Apache-2.0 notices; preserve their source headers.

Code in `modules/convrot_int8_kernels.py` is modified from [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) (in turn derived from dxqb/OneTrainer and ComfyUI-Flux2-INT8). The license is under the Apache License 2.0.

Original upstream attribution is retained here for provenance of the extracted project: code under the former `hunyuan_model` directory was modified from [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) and followed their license; code under the former `hunyuan_video_1_5` directory was modified from [HunyuanVideo 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) and followed their license. Code under the former `wan` and `frame_pack` directories was modified from [Wan2.1](https://github.com/Wan-Video/Wan2.1) and [FramePack](https://github.com/lllyasviel/FramePack), respectively, under Apache License 2.0.
