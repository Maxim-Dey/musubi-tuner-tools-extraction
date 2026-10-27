# Musubi Tuner: Krea2 Raw LoRA

This extraction supports captioned-image LoRA training for **Krea 2 Raw**: latent and text caching, training, Raw sample images during training, logging, adapter/state saving and resume. It retains the Qwen-Image VAE and Qwen3-VL-4B conditioner required by this workflow.

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

Edit the starter files:

- [dataset.toml](config_for_krea2_raw_lora/dataset.toml): replace `REPLACE_WITH_IMAGE_DIRECTORY`.
- [train.toml](config_for_krea2_raw_lora/train.toml): replace the Raw DiT, VAE and text-encoder paths.
- [sample_prompts.txt](config_for_krea2_raw_lora/sample_prompts.txt): replace `REPLACE_WITH_TRAINING_CAPTION`.

Use the same VAE and text-encoder files in caching and training. Paths in these examples resolve from the repository root. The starter uses 1024 image buckets, rank/alpha 32, batch size 1 and 1000 steps; these are editable example choices.

## Cache latents and text

Replace the model placeholders in these commands with the same paths used in `train.toml`:

```sh
python krea2_cache_latents.py --dataset_config config_for_krea2_raw_lora/dataset.toml --vae REPLACE_WITH_QWEN_IMAGE_VAE.safetensors
python krea2_cache_text_encoder_outputs.py --dataset_config config_for_krea2_raw_lora/dataset.toml --text_encoder REPLACE_WITH_QWEN3_VL_4B.safetensors --batch_size 1
```

Both passes use `cache/krea2_raw_lora` as configured in the dataset file. Re-cache when source images/captions or the corresponding model representations change. `--skip_existing` is useful only when existing caches remain valid. Inspect each command's `--help` for device, batching, worker and cache/debug options.

## Train and inspect Raw samples

```sh
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 krea2_train_network.py --config_file config_for_krea2_raw_lora/train.toml
```

The starter selects AdamW, a constant learning rate, bf16, SDPA, gradient checkpointing and resolution-aware `krea2_shift` timestep sampling. It writes LoRA checkpoints and resumable state every 200 steps, requests Raw samples every 200 steps and saves the final adapter/state under `output/krea2_raw_lora`.

Samples use the same Raw model with the current trained adapter. Prompt `--l` sets CFG; a negative prompt supplied by `--n` enables its unconditional branch. See [sample prompts](docs/sampling_during_training.md). Sample images are saved under the output directory's `sample` folder.

The starter logs progress to the console. For TensorBoard, append `--logging_dir logs/krea2_raw_lora --log_with tensorboard`, with TensorBoard installed. W&B is available with `--log_with wandb` and its optional package/account configuration. [Advanced configuration](docs/advanced_config.md) describes logging, optimizers, adapter settings and saving.

## Resume

Append a saved **state directory** to the same training command:

```sh
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 krea2_train_network.py --config_file config_for_krea2_raw_lora/train.toml --resume REPLACE_WITH_SAVED_STATE_DIRECTORY
```

Keep compatible model, dataset and network settings. `--resume` restores Accelerate training state, including optimizer, scheduler and RNG. `--network_weights` initializes adapter weights only; it does not restore training state. Retain state directories along with the model/data settings needed to resume.

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
