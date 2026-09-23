# Example CLI and File Reader Contract

This contract specifies the commands for the Stage 4 example after the user replaces the three absolute server model paths, replaces `TOK`, supplies captioned images, and creates the required caches. `<root>` is the absolute path of the movable `qwen_image_lora_val_example/`; `<repo>` is the absolute repository path. The repository package must be installed or `<repo>/src` must be on `PYTHONPATH`.

## Field/file to reader and effective value/path

| Field or file | Reader | Effective value or path |
| --- | --- | --- |
| `--config_file <root>/train.toml` | Trainer `read_config_from_file`, then Qwen experiment preflight | Selects the only accepted training TOML; defaults < TOML < explicit CLI. |
| `train.toml: experiment_dir="."` | Trainer and both Qwen cache entrypoints (via cache `--train_config`) | Resolves to the selected train TOML's parent, `<root>`, independent of invocation CWD. |
| `train.toml: model_version`, `network_module` | Trainer parser and Qwen preflight | `original`; `networks.lora_qwen_image`. |
| `train.toml: dit`, `vae`, `text_encoder` | Trainer input preflight and model loaders | User-replaced absolute server files; source DiT remains original BF16. Cache commands do not import these fields. |
| `train.toml: dataset_config`, `val_dataset_config` | Trainer preflight and dataset reader | `<root>/train-dataset.toml`; `<root>/val-dataset.toml`. |
| `train.toml: output_dir`, `logging_dir` | Trainer experiment preflight | `<root>/output`; `<root>/output/tensorboard`. |
| `train.toml: output_name` | Trainer state publisher | `qwen_image_lora`, a basename for `<output_name>-step-<X>` packages. |
| `train.toml: mixed_precision`, `save_precision` | Trainer parser and experiment preflight | BF16 computation; FP32 sole resumable adapter. BF16 `save_precision` is rejected. |
| `train.toml: lr_warmup_steps`, training timestep fields | Trainer parser/training loop | Integer `200`; `shift`/`2.2` affect training only, not fixed validation levels. |
| `train.toml: val_*`, `save_*` | Trainer Stage 2/3 event and package logic | Validation every 200 completed steps, fixed Stage 1 seeds/levels; whole package every 200 and inclusive 1,000-step retention. `save_last_n_steps_state` is absent. |
| `train.toml: sample_prompts`, `sample_every_n_steps` | Trainer prompt preflight and sampler | `<root>/sample_prompts.txt`; every 200 steps. Remove both keys to disable samples; `sample_prompts=""` is invalid. |
| `train-dataset.toml` | `config_utils.load_user_config`, blueprint and image/caption readers | One unroled `dataset/train` source and `cache/train`; common 1024 bucket/caption/batch settings. |
| `val-dataset.toml` | Same readers plus Stage 1 role-aware preflight | Exactly `val_familiar` → `dataset/val_familiar`, `cache/val_familiar`; `val_unfamiliar` → corresponding separate paths. |
| `sample_prompts.txt` | Qwen `load_prompts` | Two valid `TOK` prompt lines; `TOK` is user-replaceable. |
| Cache `--dataset_config` | Each Qwen cache command's parser and reader | Explicitly selects `train-dataset.toml` or `val-dataset.toml` relative to `<root>`; the latter processes both validation roles. |
| Cache `--vae` / `--text_encoder` | Latent / text cache parser and model loader | Explicit user-replaced server path in each relevant command. |
| Trainer `--resume` | Experiment preflight and Accelerate state loader | A published current or best package directory; a relative value resolves from `<root>`. |

## Accepted command sequence from another CWD

The following POSIX shell template uses the repository's existing entry scripts. Set `REPO` and `ROOT` to real absolute paths; use the same paths that replaced the placeholders in `train.toml`. The user supplies data and model files before running the cache commands.

```sh
REPO=/absolute/path/to/musubi-tuner-tools-extraction
ROOT=/absolute/path/to/qwen_image_lora_val_example
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd /tmp

python "$REPO/qwen_image_cache_latents.py" --train_config "$ROOT/train.toml" --dataset_config train-dataset.toml --vae /absolute/server/path/to/vae.safetensors --model_version original
python "$REPO/qwen_image_cache_latents.py" --train_config "$ROOT/train.toml" --dataset_config val-dataset.toml --vae /absolute/server/path/to/vae.safetensors --model_version original
python "$REPO/qwen_image_cache_text_encoder_outputs.py" --train_config "$ROOT/train.toml" --dataset_config train-dataset.toml --text_encoder /absolute/server/path/to/text_encoder.safetensors --model_version original
python "$REPO/qwen_image_cache_text_encoder_outputs.py" --train_config "$ROOT/train.toml" --dataset_config val-dataset.toml --text_encoder /absolute/server/path/to/text_encoder.safetensors --model_version original

accelerate launch --mixed_precision bf16 "$REPO/qwen_image_train_network.py" --config_file "$ROOT/train.toml"
tensorboard --logdir "$ROOT/output/tensorboard"
```

The cache entrypoints accept `--train_config`, `--experiment_dir`, and `--dataset_config`; they do not accept trainer `--config_file` or `--val_dataset_config` in place of their dataset selector. An absolute cache `--experiment_dir "$ROOT"` may replace `--train_config` if desired. The trainer still requires `--config_file "$ROOT/train.toml"`.

## Resume and relocation

Use a real published package directory in one of these commands; `<X>` denotes its absolute completed step:

```sh
accelerate launch --mixed_precision bf16 "$REPO/qwen_image_train_network.py" --config_file "$ROOT/train.toml" --resume "$ROOT/output/current_training_states/qwen_image_lora-step-<X>"
accelerate launch --mixed_precision bf16 "$REPO/qwen_image_train_network.py" --config_file "$ROOT/train.toml" --resume "$ROOT/output/val_training_states/val-loss/qwen_image_lora-step-<X>"
```

After moving or renaming the whole experiment folder, change `ROOT` to its new absolute path and rerun commands without editing internal relative paths. A resumed package at step `s` with `max_train_steps=1600` validates once at `s`, logs the first new training loss at `s+1`, and reaches `s+1600` after 1,600 completed updates. Exact data-loader position restoration is not promised. Preserve fixed validation images, captions, caches, and noise controls for compatible resume.
