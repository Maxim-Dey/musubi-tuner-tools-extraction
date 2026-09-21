# Musubi Tuner — FLUX.2 Dev LoRA

[日本語](README.ja.md) · [Русский](README.ru.md)

This fork supports exactly **FLUX.2 Dev image LoRA**: VAE latent caching, Mistral caption-output caching and training. The three root commands below also work as `python -m musubi_tuner.<script-name-without-py>`. Other models (including Klein), full fine-tuning, Self-Flow, LoHa/LoKr/LyCORIS, GUI, standalone generation, caption generation, converters, merge/export and post-hoc EMA tools are removed. Training-time PNG samples and internal standard-LoRA base-weight operations remain.

## Setup and prerequisites

Use Python 3.10–3.12 with compatible PyTorch/torchvision and hardware supporting the intended BF16/AdamW8bit run. The following is a **separate operational setup example**, not a command sequence executed by local contract checks. It uses an existing CUDA 12.8 option; other retained CUDA extras are in [pyproject.toml](pyproject.toml).

```sh
python -m venv .venv
# Activate .venv before the following commands.
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -m pip install tensorboard pytest "ruff>=0.12.10,<0.16"
python -m pip check
```

On Windows activate with `.\.venv\Scripts\Activate.ps1`; on POSIX use `source .venv/bin/activate`. Optional console previews/timestep plots use `ascii-magic==2.3.0` and `matplotlib==3.10.0`. Install selected optional backends/trackers separately. Missing bitsandbytes/TensorBoard/selected backends fail explicitly; settings are never silently substituted.

Prepare original Dev DiT and AE checkpoint files, **all ten Mistral shards** beside `model-00001-of-00010.safetensors`, and the cached processor/tokenizer/chat-template resources for `mistralai/Mistral-Small-3.1-24B-Instruct-2503`. Processor lookup is independent of the weight path, uses the normal Hugging Face cache, and is checked offline before weights. No tokenizer-path option exists. Resources are external; see [exact paths and preparation](docs/flux_2.md).

## Workflow

Run from the repository root. [train.toml](flux2dev_lora/train.toml), [dataset.toml](flux2dev_lora/dataset.toml) and [sample_prompts.txt](flux2dev_lora/sample_prompts.txt) supply a consistent configuration. Replace only external resource paths for your machine. The image directory needs user-prepared images with same-stem UTF-8 `.txt` captions. Cache, output and log destinations must be writable.

These **operational commands load real weights**; run both caches before training:

```sh
python flux_2_cache_latents.py --model_version dev --dataset_config ./flux2dev_lora/dataset.toml --vae ./models/flux2-dev/ae.safetensors --vae_dtype float32
python flux_2_cache_text_encoder_outputs.py --model_version dev --dataset_config ./flux2dev_lora/dataset.toml --text_encoder ./models/flux2-dev/text_encoder/model-00001-of-00010.safetensors
accelerate launch --num_processes 1 --mixed_precision bf16 flux_2_train_network.py --config_file ./flux2dev_lora/train.toml
```

The template remains rank=alpha=32, dropout .05, BF16, FP32 AE, SDPA/checkpointing, no FP8, `flux2_shift`, AdamW8bit at 1e-4, 100 warmup steps, 2000 updates, batch 1/accumulation 4, seed 42 and two persistent workers. `blocks_to_swap=20` stays commented. Useful non-template Dev options and both LoRA module spellings remain; unknown/excluded source keys fail before weights.

LoRA and full states save every 250 updates; both retention windows are 1000 updates. Samples run at first and every 250 updates, writing PNGs under `<output_dir>/sample`; the supplied prompt uses existing 256×256/20-step/guidance-4 defaults. TensorBoard uses the original prefix/tracker name under `./logs/flux2_dev_style`; view with `tensorboard --logdir ./logs/flux2_dev_style`.

A LoRA `.safetensors` file and full Accelerate state directory are different artifacts. `--resume` restores epoch, update count and data position from new states, continuing the remaining updates with the original configuration. Legacy states without progress metadata still load with an explicit warning that counters/data restart under the old behavior. `--dim_from_weights --network_weights <file>` restores adapter rank, alpha and weights. See [artifacts and compatibility](docs/flux_2.md).

## Guides and local checks

- [Complete Dev workflow](docs/flux_2.md)
- [Dataset and image controls](docs/dataset_config.md)
- [Advanced options and trackers](docs/advanced_config.md)
- [Training samples](docs/sampling_during_training.md)
- [Block swap](docs/block_swap.md) and [torch.compile](docs/torch_compile.md)

For local checks use an already prepared CPU environment with `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1` and UTF-8 console output:

```sh
python flux_2_cache_latents.py --help
python flux_2_cache_text_encoder_outputs.py --help
python flux_2_train_network.py --help
python -m pytest -p no:cacheprovider -q tests
```

The suite includes all six root/module help calls and tiny CPU/temp-file checks. No real training, GPU, weight/resource download, package build or server validation is a local smoke test. Actual results and unavailable processor/resource checks are recorded in [baseline.md](specs/001-scope-flux2-dev-lora/baseline.md), not inferred from parsing success.

## Development and attribution

Follow [CONTRIBUTING.md](CONTRIBUTING.md) and [日本語 development guide](CONTRIBUTING.ja.md). Use Ruff with the existing project settings; preserve shared numeric/precision assertions. Existing `.agents/skills` instructions, personal agent files and Spec Kit/CI tooling are preserved. The old README referenced `.ai` prompt files that are absent in this checkout; this guide does not require them.

Musubi Tuner by [kohya-ss](https://github.com/kohya-ss/musubi-tuner). Retained FLUX code derives from [Black Forest Labs](https://github.com/black-forest-labs/flux); some common code is copied/modified from Diffusers. Retained code uses Apache-2.0 notices in its sources; model/resource licenses are separate. Shared FP8/offloading credits remain in the [advanced](docs/advanced_config.md) and [block-swap](docs/block_swap.md) guides. This fork is unofficial and is not affiliated with model authors.

Upstream sponsor: [AiHUB](https://aihub.co.jp/top-en).

[![AiHUB](images/logo_aihub.png)](https://aihub.co.jp/top-en)

[Support upstream development](https://github.com/sponsors/kohya-ss/).
