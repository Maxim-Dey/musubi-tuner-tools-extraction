# Example CLI and File Reader Contract

This contract specifies the ordinary single-command run for the Stage 4 example after the user replaces the three absolute server model paths, replaces `TOK`, and supplies captioned images. The training command creates required caches. `<root>` is the absolute path of the movable `qwen_image_lora_val_example/`; `<repo>` is the absolute repository path. The repository package must be installed or `<repo>/src` must be on `PYTHONPATH`. Revised 2026-09-24 after the user superseded the original four-command cache procedure.

## Field/file to reader and effective value/path

| Field or file | Reader | Effective value or path |
| --- | --- | --- |
| `--config_file <root>/train.toml` | Trainer `read_config_from_file`, then Qwen experiment preflight | Selects the only accepted training TOML; defaults < TOML < explicit CLI. |
| `train.toml: experiment_dir="."` | Trainer and automatic Qwen cache preparation | Resolves to the selected train TOML's parent, `<root>`, independent of invocation CWD. |
| `train.toml: model_version`, `network_module` | Trainer parser and Qwen preflight | `original`; `networks.lora_qwen_image`. |
| `train.toml: dit`, `vae`, `text_encoder` | Trainer input preflight, model loaders, and automatic cache preparation | User-replaced absolute server files; source DiT remains original BF16. The trainer passes effective VAE and text-encoder paths to cache subprocesses. |
| `train.toml: dataset_config`, `val_dataset_config` | Trainer preflight and dataset reader | `<root>/train-dataset.toml`; `<root>/val-dataset.toml`. |
| `train.toml: output_dir`, `logging_dir` | Trainer experiment preflight | `<root>/output`; `<root>/output/tensorboard`. |
| `train.toml: output_name` | Trainer state publisher | User-configured basename for `<output_name>-step-<X>` packages. |
| `train.toml: mixed_precision`, `save_precision` | Trainer parser and experiment preflight | BF16 computation; FP32 sole resumable adapter. BF16 `save_precision` is rejected. |
| `train.toml: lr_warmup_steps`, training timestep fields | Trainer parser/training loop | User-configured integer warmup; `shift`/`2.2` affect training only, not fixed validation levels. |
| `train.toml: max_train_steps`, `val_*`, `save_*` | Trainer Stage 2/3 event and package logic | User-configured update budget and intervals; validation runs at step 0, periodic boundaries, and final step, without duplicate events. Whole-package retention uses the configured inclusive window. `save_last_n_steps_state` is absent. |
| `train.toml: sample_prompts`, `sample_every_n_steps` | Trainer prompt preflight and sampler | `<root>/sample_prompts.txt`; cadence comes from the configuration. Remove both keys to disable samples; `sample_prompts=""` is invalid. |
| `train-dataset.toml` | `config_utils.load_user_config`, blueprint and image/caption readers | One unroled `dataset/train` source and `cache/train`; common 1024 bucket/caption/batch settings. |
| `val-dataset.toml` | Same readers plus Stage 1 role-aware preflight | Exactly `val_familiar` → `dataset/val_familiar`, `cache/val_familiar`; `val_unfamiliar` → corresponding separate paths. |
| `sample_prompts.txt` | Qwen `load_prompts` | Two valid `TOK` prompt lines; `TOK` is user-replaceable. |
| Cache subprocess dataset and model arguments | Existing Qwen latent/text cache readers | Derived by the trainer from effective `dataset_config`, `val_dataset_config`, `vae`, and `text_encoder`; the validation declaration processes both roles. |
| Trainer `--resume` | Experiment preflight and Accelerate state loader | A published current or best package directory; a relative value resolves from `<root>`. |

## Accepted command sequence from another CWD

The following POSIX shell template uses the Qwen training entry script. Set `REPO` and `ROOT` to real absolute paths. The user supplies captioned data and model files before the run.

```sh
REPO=/absolute/path/to/musubi-tuner-tools-extraction
ROOT=/absolute/path/to/qwen_image_lora_val_example
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd /tmp

python "$REPO/qwen_image_train_network.py" --config_file "$ROOT/train.toml"
tensorboard --logdir "$ROOT/output/tensorboard"
```

On launch, the trainer invokes both existing cache entrypoints for every configured dataset before loading the training model. Existing training cache files are skipped, missing files are created, and validation preflight rebuilds stale validation caches. The user's ordinary command needs no cache or rebuild flags. Repeated runs with complete caches do not load cache encoder weights.

## Resume and relocation

Use a real published package directory in one of these commands; `<X>` denotes its absolute completed step:

```sh
accelerate launch --mixed_precision bf16 "$REPO/qwen_image_train_network.py" --config_file "$ROOT/train.toml" --resume "$ROOT/output/current_training_states/<output_name>-step-<X>"
accelerate launch --mixed_precision bf16 "$REPO/qwen_image_train_network.py" --config_file "$ROOT/train.toml" --resume "$ROOT/output/val_training_states/val-loss/<output_name>-step-<X>"
```

After moving or renaming the whole experiment folder, change `ROOT` to its new absolute path and rerun commands without editing internal relative paths. Set `<output_name>` to the value in `train.toml`. A resumed package at step `s` with user-selected `max_train_steps=B` validates once at `s`, logs the first new training loss at `s+1`, and reaches `s+B` after `B` completed updates. Exact data-loader position restoration is not promised. Preserve fixed validation images, captions, caches, and noise controls for compatible resume.
