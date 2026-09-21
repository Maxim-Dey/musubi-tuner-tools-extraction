# FLUX.2 Dev LoRA

This fork retains image LoRA training for FLUX.2 Dev, latent caching and Mistral caption-output caching. Root scripts and `python -m musubi_tuner.<script>` expose the same three operations. Standalone generation, full fine-tuning, Self-Flow, other models/adapters, GUI, converters, merge/export, caption generation and post-hoc EMA commands have been removed. PNG samples generated during training and internal standard-LoRA base-weight loading remain.

## Runtime and resources

Run commands from the repository root with Python 3.10–3.12 and compatible PyTorch/torchvision, Accelerate, bitsandbytes, Transformers, sentencepiece and TensorBoard. The template needs a runtime that supports BF16 and AdamW8bit; CPU contract checks do not prove that hardware capability. See [setup](../README.md) and [pyproject.toml](../pyproject.toml) for retained versions/extras. Optional attention/optimizer/tracker dependencies must be installed when selected; no optimizer fallback is performed.

Prepare these external resources before operational runs:

| Resource | Supplied example path or source |
| --- | --- |
| Original Dev DiT checkpoint | `./models/flux2-dev/flux2-dev.safetensors` |
| Original AE checkpoint (FP32) | `./models/flux2-dev/ae.safetensors` |
| Mistral 3 checkpoint | `./models/flux2-dev/text_encoder/model-00001-of-00010.safetensors`, with **all ten** matching shards in the same directory |
| Processor/tokenizer/chat template | Normal Hugging Face cache for `mistralai/Mistral-Small-3.1-24B-Instruct-2503`; independent of the checkpoint path |
| User images and UTF-8 captions | `./data/flux2_dev_style/images`, same-stem `.txt` captions |
| Writable cache/output/log directories | Paths in the two supplied TOMLs |

Original model resources: [FLUX.2 Dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) and [Mistral processor resources](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/tree/main). Checkpoint arguments take original checkpoint **files**, not Diffusers directories. The shard loader derives all companion names from the numbered filename. Preflight checks processor availability with `AutoProcessor.from_pretrained(..., use_fast=False, local_files_only=True)` and never downloads it. Populate `HF_HOME`/`HF_HUB_CACHE` separately. There is no `--tokenizer_path` option or newly required `tokenizer.model` file.

The [training template](../flux2dev_lora/train.toml) and [dataset template](../flux2dev_lora/dataset.toml) use working-directory-relative paths, not TOML-directory-relative paths. Replace external resource locations consistently. The supplied dataset/prompt references already resolve from the repository root.

## Operational examples — not local smoke tests

These commands load real weights and must run only in a separately prepared operational environment. Both caches are required before training:

```sh
python flux_2_cache_latents.py --model_version dev --dataset_config ./flux2dev_lora/dataset.toml --vae ./models/flux2-dev/ae.safetensors --vae_dtype float32
python flux_2_cache_text_encoder_outputs.py --model_version dev --dataset_config ./flux2dev_lora/dataset.toml --text_encoder ./models/flux2-dev/text_encoder/model-00001-of-00010.safetensors
accelerate launch --num_processes 1 --mixed_precision bf16 flux_2_train_network.py --config_file ./flux2dev_lora/train.toml
```

Configure Accelerate for the intended hardware; this example is one process. `--skip_existing` skips existing per-item caches; `--keep_cache` prevents stale-cache cleanup. Existing filenames, keys, metadata and dtype suffixes are retained; no cache migration is required. Mistral uses layers `[10,20,30]`, padded length 512 and feature width 15360. Per-item context is `[512,15360]`, collated context `[B,512,15360]`.

The supplied scenario remains rank=alpha=32, dropout=.05, BF16 training/Mistral, FP32 AE, SDPA, gradient checkpointing, `flux2_shift`, weighting `none`, AdamW8bit at `1e-4`, 100 warmup steps, 2000 updates, batch 1 and accumulation 4. Seed 42 and two persistent data-loader workers remain. All FP8 flags are false; `blocks_to_swap=20` remains commented. Effective batch is four images with one process. Missing dependencies do not disable any setting. Mistral FP8 is rejected before weights; optional DiT FP8 remains available.

## Samples, logging and artifacts

[Sample prompts](../flux2dev_lora/sample_prompts.txt) contain one ceramic-cup sentence. Default sample settings are 256×256, 20 denoising steps and guidance 4.0; these do not change dataset resolution 1024×1024. The template samples at first and every 250 optimizer updates, retaining sample RNG/model-mode restoration. Outputs are PNGs under the output directory's `sample` folder. See [sampling](sampling_during_training.md).

TensorBoard runs under `./logs/flux2_dev_style`, with prefix `flux2_dev_style_` plus a timestamp and tracker subdirectory `flux2_style`. Metrics include `loss/current`, `loss/average` and `lr/unet`. To view an operational run: `tensorboard --logdir ./logs/flux2_dev_style`. Optional gradient metrics and WandB settings are described in [advanced configuration](advanced_config.md).

LoRA weights are saved every 250 updates (`flux2_dev_style-step00000250.safetensors`, etc.), with the final adapter `flux2_dev_style.safetensors`. Default save precision remains FP32 unless explicitly selected otherwise. Full Accelerate states are separate directories (`flux2_dev_style-step00000250-state`, final `flux2_dev_style-state`), containing adapter/model state, optimizer, scheduler and RNG. Retention settings are **1000-update windows**, not file counts. State and weight retention use their existing independent settings.

`--resume ./output/flux2_dev_style/<state-directory>` loads full state; an adapter `.safetensors` file is not full state. New states include `training_progress.pt` beside the unchanged Accelerate files. It records epoch, completed updates, next batch, sampler RNG, dataset shuffle seeds/epochs, timestep-bucket state and moving loss. Resume skips completed batches without loading their caches and keeps scheduler, logging, save/retention and sample numbering aligned. It does not repeat the initial sample. Keep the same dataset/cache contents, total target steps, accumulation, process count and loader settings; the saved target is the total, not an additional number of updates. For the template, a step-1500 state continues to step 2000 with 500 updates remaining.

Legacy states without this sidecar retain their previous loading behavior: weights, optimizer, scheduler and RNG load, but counters and data restart from zero with an explicit warning. Exact data position cannot be recovered from those files or inferred safely from the directory name. A malformed/incompatible progress sidecar is an error, not a fallback to zero. Existing LoRA and Accelerate files need no migration.

`--network_weights <file>` loads initial adapter weights. Adding `--dim_from_weights` derives each adapter's rank and alpha from that file before loading its tensors; without the flag the configured network construction is unchanged. This is separate from full-state resume.

## Local evidence

Local checks use CPU, temporary fixtures, loader sentinels and no real-model weights. They cover input rejection, templates, cache contracts, toy LoRA/math, PNGs, TensorBoard and Accelerate state I/O. The resume regression uses five tiny cached items, six CPU optimizer updates and real trainer/Accelerate save/load paths. It compares exact parameters, optimizer, scheduler, Python/NumPy/torch RNG, counters, data order and subsequent log/save/sample events, including accumulated updates and persistent workers. This does not establish real-model/GPU equivalence or quality. Exact results and the still-blocked offline Mistral processor check are in [baseline.md](../specs/001-scope-flux2-dev-lora/baseline.md).
