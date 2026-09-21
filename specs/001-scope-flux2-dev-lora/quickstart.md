# Quickstart and validation guide

This guide describes the intended implementation result. The current planning invocation has not modified the templates, added the prompt or narrowed the code. Current local evidence is in [baseline.md](baseline.md); interface details are in [contracts/cli-and-config.md](contracts/cli-and-config.md).

## 1. Prepared runtime and installation

Installation commands below are for a separately prepared runtime, not commands executed during this local planning stage. Use Python 3.10-3.12; the example uses 3.12 within the existing project range. Use a Windows/Linux machine with hardware/driver support for the chosen PyTorch CUDA build and BF16/AdamW8bit scenario. No specific local GPU capability has been established.

From the repository root, create an environment with an installed supported Python:

```text
python3.12 -m venv .venv
```

On Windows the equivalent is `py -3.12 -m venv .venv`. Activate it with `source .venv/bin/activate` on POSIX or `.\.venv\Scripts\Activate.ps1` in PowerShell. The following commands then use its `python`:

```text
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -m pip install tensorboard pytest "ruff>=0.12.10,<0.16"
python -m pip check
```

The torch/torchvision CUDA 12.8 pair is published for Windows/Linux and satisfies an existing project CUDA option; this example does not change project version requirements. [PyTorch installation archive](https://pytorch.org/get-started/previous-versions/). bitsandbytes documents compatible Python/PyTorch and CUDA builds; actual device suitability must be checked in that runtime. [bitsandbytes installation](https://huggingface.co/docs/bitsandbytes/main/en/installation).

`pip install -e .` installs the retained declared runtime dependencies, including bitsandbytes and sentencepiece. TensorBoard must be explicit because it currently lives only in the dev group; pytest is also not declared. No torchaudio, Qwen caption package or extra attention backend is needed for these templates. Keep development tooling: optional console image previews/timestep plots require the existing `ascii-magic==2.3.0` and `matplotlib==3.10.0`. Other selected optional Dev backends/optimizers/trackers/codecs require their own existing dependencies; missing packages must produce errors before weights, not fallback settings.

Without creating tensors or loading models, verify the complete mandatory import path and dependency consistency:

```text
python -c "import torch, torchvision, accelerate, bitsandbytes, diffusers, einops, huggingface_hub, cv2, PIL, numpy, packaging, safetensors, toml, tqdm, transformers, voluptuous, sentencepiece, tensorboard; from transformers import Mistral3Config, AutoProcessor; getattr(__import__('transformers'), 'Mistral3ForConditional' + 'Generation'); from torch.utils.tensorboard import SummaryWriter; print('runtime imports OK')"
python -m pip check
```

This checks declared/transitive imports, including NumPy/packaging, but does not execute GPU kernels or prove a compatible processor cache. The planning environment lacks these packages; a successful install/solve/import result is not claimed.

## 2. Resources and both templates

After implementation, `flux2dev_lora/train.toml` must contain these corrected repository-relative references:

```toml
dataset_config = "./flux2dev_lora/dataset.toml"
sample_prompts = "./flux2dev_lora/sample_prompts.txt"
```

All commands run from the repository root. Template paths are relative to the working directory, not the TOML file's directory. Replace external resource paths consistently in templates and command arguments; supplied config/prompt files are repository deliverables, while these resources are not:

| Resource | Example and preparation |
| --- | --- |
| DiT | `./models/flux2-dev/flux2-dev.safetensors`, original Dev checkpoint |
| AE | `./models/flux2-dev/ae.safetensors`, original AE checkpoint, used in FP32 |
| Mistral weights | `./models/flux2-dev/text_encoder/model-00001-of-00010.safetensors`; all ten matching numbered shards alongside it. Existing loader derives companion names from the first shard. |
| Processor/tokenizer | Existing loader uses `mistralai/Mistral-Small-3.1-24B-Instruct-2503` via the normal Hugging Face cache, independently of `--text_encoder`. Supply online access or a prepopulated `HF_HOME`/`HF_HUB_CACHE` with processor/tokenizer/template resources. No `--tokenizer_path` exists. |
| Dataset | `./data/flux2_dev_style/images`; user-prepared images with same-stem UTF-8 `.txt` captions |
| Cache | `./data/flux2_dev_style/cache`; writable directory dedicated to the selected dataset |
| Outputs/logs | `./output/flux2_dev_style`, `./logs/flux2_dev_style`; writable destinations |

Model resources come from the documented [FLUX.2 Dev repository](https://huggingface.co/black-forest-labs/FLUX.2-dev). Do not substitute a Diffusers-format DiT/AE directory for the expected original checkpoint files. The [Mistral resource directory](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/tree/main) supplies config, processor/preprocessor, tokenizer JSON/config, special tokens and chat template files. Do not invent a mandatory `tokenizer.model` or change tokenizer backend during cleanup. Resource acquisition and real-model work belong to the later operational stage.

With processor resources already cached, this separate weight-free preflight verifies actual processor dependencies and tokenization without constructing `Mistral3Embedder` or loading its weights:

```text
python -c "from transformers import AutoProcessor; p=AutoProcessor.from_pretrained('mistralai/Mistral-Small-3.1-24B-Instruct-2503',use_fast=False,local_files_only=True); x=p.apply_chat_template([{'role':'user','content':[{'type':'text','text':'A ceramic cup.'}]}],tokenize=True,return_dict=True,return_tensors='pt',padding='max_length',truncation=True,max_length=512); print({k:tuple(v.shape) for k,v in x.items() if hasattr(v,'shape')})"
```

Missing cached resources make this check unavailable; they do not justify changing `sample_at_first` or the logging settings. Training still needs Mistral to prepare sample-prompt embeddings even after caption caches have been generated.

Implementation supplies `flux2dev_lora/sample_prompts.txt` with this minimal valid nonempty line:

```text
A ceramic cup on a wooden table.
```

This uses the current sampler defaults: 256x256, 20 steps and guidance 4.0. It does not alter the dataset's 1024x1024 resolution or training hyperparameters. Optional per-prompt `--w`, `--h`, `--d`, `--s`, `--g` and repeated `--ci` remain available; explicit values in a custom prompt are not the sampler defaults. A no-control prompt needs no extra reference-image file.

The following active template values must all remain effective:

| Area | Required values |
| --- | --- |
| Dataset | Resolution `[1024,1024]`, `.txt` captions, batch 1, buckets enabled, no upscale, repeats 1 |
| Adapter | `dev`, `networks.lora_flux_2`, rank 32, alpha 32, dropout .05 |
| Precision/memory | BF16 training/DiT/Mistral, FP32 VAE, all three FP8 flags false, SDPA and gradient checkpointing true; `blocks_to_swap=20` remains commented |
| Noise/optimization | `flux2_shift`, weighting `none`, AdamW8bit, LR `1e-4`, constant-with-warmup, 100 warmup steps, max gradient norm 1 |
| Duration/data | 2000 steps, accumulation 4, seed 42, two persistent data-loader workers; effective batch 4 on one GPU |
| Artifacts | Save every 250 optimizer updates; full state enabled including train end; both step retention windows 1000; original output name |
| Samples | Every 250 optimizer updates and at first; prompt file retained |
| Logging | TensorBoard, original logging directory, `flux2_dev_style_` prefix, tracker name `flux2_style` |

Tests must compare every actual active key from the original templates, not only this summary table.

## 3. Local checks without weights

Use an already equipped environment. Disable GPU selection and network downloads for these checks (`CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1`; set `PYTHONPATH=src` if not installed editable). Do not use full training or caching commands as smoke tests.

```text
python flux_2_cache_latents.py --help
python -m musubi_tuner.flux_2_cache_latents --help
python flux_2_cache_text_encoder_outputs.py --help
python -m musubi_tuner.flux_2_cache_text_encoder_outputs --help
python flux_2_train_network.py --help
python -m musubi_tuner.flux_2_train_network --help
python -m pytest -p no:cacheprovider -q tests/test_save_precision.py tests/test_lora_dtype_bridging.py tests/test_grad_metrics.py tests/test_datasource_item_extras.py
```

Run production parsing of the real training template without invoking `main`/`train`:

```text
python -c "import sys; from musubi_tuner.training.parser_common import setup_parser_common,read_config_from_file; from musubi_tuner.flux_2_train_network import flux2_setup_parser; sys.argv=['check','--config_file','./flux2dev_lora/train.toml']; p=flux2_setup_parser(setup_parser_common()); a=read_config_from_file(p.parse_args(),p); assert a.model_version=='dev'; assert a.network_module=='networks.lora_flux_2'; print('training TOML parsed')"
python -c "import argparse; from musubi_tuner.dataset.config_utils import ConfigSanitizer,BlueprintGenerator,load_user_config; from musubi_tuner.dataset.architectures import ARCHITECTURE_FLUX_2_DEV; b=BlueprintGenerator(ConfigSanitizer()).generate(load_user_config('./flux2dev_lora/dataset.toml'),argparse.Namespace(),architecture=ARCHITECTURE_FLUX_2_DEV); print(b)"
python -c "from musubi_tuner.training.sampling_prompts import load_prompts; p=load_prompts('./flux2dev_lora/sample_prompts.txt'); assert p and p[0]['prompt'].strip(); print('sample prompts parsed')"
```

These commands cover parsing only; implementation's focused tests must additionally invoke the final validator, compare all effective template/default/override values and instrument loaders. A parser-only success does not prove FR-011. Use a small number of tests (for example `tests/test_flux2_scope.py` and `tests/test_flux2_workflow_contracts.py`, to be created during implementation) for:

- CLI and TOML rejection matrix from the contract: all four Klein variants, excluded methods/modules/nested args, unknown/removed parameters, malformed structures/types, video/audio/Self-Flow, invalid dataset and all supported prompt formats. Preserve arbitrary one-level training groups, with positive `[model]`/`[lora]` and flattening/CLI precedence cases; unknown contained parameters still fail. Dataset/prompt/tracker sections retain their own schemas. Test final valid overrides separately from unknown source parameters; loader sentinels must remain untouched.
- Consumed `log_tracker_config` TOML: use temporary files for valid tracker settings and missing/unreadable files, malformed TOML, unknown tracker sections/initialization arguments and invalid structures/values. Check early source-qualified errors and untouched model-loader sentinels without starting trackers or contacting services. Preserve supported backend arguments and nested payload meanings.
- Tiny image/control/cache fixtures, LoRA/math/precision/gradient comparisons, moved PNG saver, interval/retention/RNG behavior, small TensorBoard events and Accelerate CPU state fixtures. Preserve baseline assertions, including visible resume limitations; no real model or full training invocation.
- Retained wrappers, metadata and removal inventory; no hidden `main`, registry or dynamic factory path for excluded capabilities. Run existing Ruff checks on changed retained Python and review all README variants/internal links.

Keep baseline failures separate. In the current environment pytest, CLI imports and production template parsers are blocked before execution; stdlib TOML parsing and AST checks do not replace these checks.

## 4. Full operational cycle — later stage only

The following commands load real models and are not local contract checks. Use them only in the separately authorized equipped runtime, with real resources and the implemented path corrections. Restore that runtime's intended GPU/network settings after CPU-only checks.

Cache image latents using FP32 AE:

```text
python flux_2_cache_latents.py --model_version dev --dataset_config ./flux2dev_lora/dataset.toml --vae ./models/flux2-dev/ae.safetensors --vae_dtype float32
```

Cache caption embeddings using the original Mistral BF16 path:

```text
python flux_2_cache_text_encoder_outputs.py --model_version dev --dataset_config ./flux2dev_lora/dataset.toml --text_encoder ./models/flux2-dev/text_encoder/model-00001-of-00010.safetensors --batch_size 1
```

Expected: existing Dev cache naming (`f2d`, `f2d_te`) and field/metadata contracts. Compatible caches need no regeneration due to cleanup. Existing `--skip_existing`/`--keep_cache` remain available; default cache cleanup can remove stale cache files for that dataset, so use the intended dedicated cache directory.

Train with the full template:

```text
accelerate launch --num_processes 1 --num_machines 1 --num_cpu_threads_per_process 1 --mixed_precision bf16 flux_2_train_network.py --config_file ./flux2dev_lora/train.toml
```

Expected: AdamW8bit and accumulation 4, training samples at first/every 250 updates under `output/flux2_dev_style/sample`, LoRA checkpoint files and Accelerate state directories at configured events, and TensorBoard events under the configured run prefix/tracker. No separate generation command is needed.

Inspect logs in another terminal:

```text
tensorboard --logdir ./logs/flux2_dev_style
```

Resume an unfinished run from an actual retained full-state directory; step 1500 is only an example and must exist:

```text
accelerate launch --num_processes 1 --num_machines 1 --num_cpu_threads_per_process 1 --mixed_precision bf16 flux_2_train_network.py --config_file ./flux2dev_lora/train.toml --resume ./output/flux2_dev_style/flux2_dev_style-step00001500-state
```

Do not pass the LoRA `.safetensors` as full state or assume an early step checkpoint survived the 1000-step retention window. Keep `max_train_steps=2000` and other active settings unchanged. The current source resets loop counters after loading state; this plan does not promise that the above resumes a remaining 500 updates or preserves global event numbering. Verify baseline-versus-cleanup state, RNG and event behavior and report any discrepancy against the specification, without silently correcting counters or weakening requirements.

Operational acceptance must verify both caching stages, actual optimization/dtypes/loss, samples, logging, LoRA/full-state saving and resume on a real model, together with baseline numerical/RNG comparison where feasible. A completed run or local synthetic result alone is insufficient. Training, GPU work, weights download, packaging, server transfer and server verification were not performed in this planning stage.
