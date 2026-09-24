# Qwen-Image original adapter training

This repository trains original Qwen-Image adapters with the existing Musubi Tuner engine. The supported workflow is captioned images → latent cache → caption embedding cache → adapter training with optional sample PNGs. Use the [Russian README](../README.md) and the supplied [experiment configuration](../qwen_image_lora_val_example/train.toml).

Set model and dataset paths in the training TOML. Run from the repository root; without `experiment_dir`, relative paths resolve from the process CWD.

```bash
python qwen_image_train_network.py --config_file /path/to/experiment/train.toml
```

The package must be installed or `src` included in `PYTHONPATH`. The training command checks both caches for every configured dataset and creates missing files before loading the training model. The cache commands remain available for manual use.

Use the original DiT, RGB VAE and Qwen2.5-VL weights. The text loader obtains tokenizer assets from `Qwen/Qwen-Image`, subfolder `tokenizer`; prepare them in the operational environment. DiT execution uses bf16. The training VAE dtype default is bf16; the Qwen VAE loader retains its existing loading behavior. `num_layers` defaults to 60 and must match the checkpoint.

Set `max_train_steps`, training batch size, rank, learning rate, and validation/save/sample intervals for your run in the configuration files. Defaults are overridden by training TOML and then explicit CLI. Omitted CLI flags preserve TOML; store-true flags cannot disable a true TOML setting. Unknown fields/types, excluded selectors and invalid effective values fail early. The only model version is `original`. [Alternative adapters](loha_lokr.md) retain the Qwen model engine.

Both cache commands have independent CLI settings for manual use. `batch_size` caps encoding chunks; the dataset declaration controls training batch size. `num_workers` must be positive when explicitly set. Qwen cache commands skip existing files by default, for both training and validation datasets, and create missing files. The automatic check first fills missing files; validation then verifies cache contents against current images and captions and rebuilds its caches if needed. Training caches are checked by file existence. `keep_cache` preserves old files outside the current dataset. Latent cache debug supports `image` and `console`; Qwen latent caching does not accept an explicit `vae_dtype` or tiling/chunk options.

Latent caches preserve `[C,1,H,W]`; the singleton axis is required by the original VAE. Text caches keep variable-length `[L,D]` embeddings. Training pads text and preserves masks/split attention. Missing text caches are warned and skipped; an empty effective training dataset fails before models.

## Validation input preparation

For Qwen-Image original, provide a separate role-bearing `val-dataset.toml` with exactly one nonempty `val_familiar` and one nonempty `val_unfamiliar` dataset. Each has captioned images (or captioned JSONL records), its own cache directory, and effective `batch_size = 1` and `num_repeats = 1`. Set `val_dataset_config` in the training TOML. The training command prepares both caches and rejects missing captions and ambiguous cache names before loading weights. Validation checks that existing caches are bound to the current image bytes and captions.

Set `val_dataset_config` in the training TOML or CLI to enable validation; omit it to preserve the previous training behavior. Qwen-only controls are `val_every_n_steps` (default `200`, integer `>=1`), `val_seed_noise` (default `42`, signed integer), `val_level_noise_n` (default `10`, even integer `>=2`), and `val_seed_noise_n` (default `1`, integer `>=1`). Effective values follow defaults → TOML → explicit CLI; booleans and unknown keys are rejected. Invalid roles, empty sets, missing or changed images, captions or caches fail with source-labelled errors before model loading or on a later read. The prepared input has a portable identity and deterministic noise checks. See the [input contract](../specs/002-val-loss-core/contracts/validation-inputs.md).

Validation uses the current LoRA weights and the training forward and loss at each fixed noise level. It averages checks within each declared image occurrence, then averages image means within each role, independent of image size, bucket or training repeats. Low noise uses `t<0.5`; high noise uses `t>=0.5`. The existing tracker publishes six scalars after both roles complete:

| Role | Full mean | Low noise | High noise |
| --- | --- | --- | --- |
| `val_familiar` | `train_eval_loss_mean` | `train_eval_loss_low_noise` | `train_eval_loss_high_noise` |
| `val_unfamiliar` | `val_loss_mean` | `val_loss_low_noise` | `val_loss_high_noise` |

An enabled run validates at absolute completed-update step 0 before its first update, after positive steps divisible by `val_every_n_steps`, and once at its final completed step. Accumulation microbatches and skipped optimizer attempts do not advance that count or publish an event. Periodic and final events at the same step coalesce. Existing training-loss tags remain; enabled training-loss points use completed absolute steps. A failed or nonfinite event publishes none of its six values.

Each enabled Accelerate state directory contains `val_loss_state.json` with the explicit absolute completed step, Stage 1 fingerprint and effective validation controls. Enabled resume requires a valid sidecar and unchanged inputs and controls; a legacy state without one remains usable when validation is disabled. Resume from `s` validates at `s` before the next batch and places the first new training-loss point at `s+1`. `max_train_steps` remains the completed-update budget for this invocation, so the final absolute step is `s+max_train_steps`. Loading network weights without `--resume` starts a new timeline at 0. Exact training data-loader position restoration is outside this feature. See the [training and resume contract](../specs/003-val-loss-training/contracts/validation-training.md) and [local CPU checks](../specs/003-val-loss-training/quickstart.md).

## Portable Qwen-Image LoRA experiment

The optional `experiment_dir` mode applies to Qwen-Image `original` LoRA. Start from the separate [validation example](../qwen_image_lora_val_example/), or create your own `<root>/train.toml`, an unroled `<root>/train-dataset.toml`, and a two-role `<root>/val-dataset.toml`. The trainer requires `--config_file <root>/train.toml` and enabled validation, even if the root is absolute. The example already contains these fields; they do not change the supplied legacy template:

```toml
experiment_dir = "."
dataset_config = "train-dataset.toml"
val_dataset_config = "val-dataset.toml"
save_precision = "fp32"
save_last_n_steps = 1000
```

A relative root is anchored to the selected `train.toml`, independent of invocation CWD; an explicit CLI value overrides TOML. Relative model, dataset-config, image/cache paths within both dataset TOMLs, JSONL `image_path`, prompts, tracker config, and local resume paths resolve from the experiment root. Absolute paths stay absolute and the process CWD does not change. Output and TensorBoard directories are fixed at `<root>/output` and `<root>/output/tensorboard`.

The example is a template, not prepared training data. Replace its three marked absolute paths with the server's original BF16 DiT, VAE, and text encoder; replace `TOK` in both prompt lines; supply your own images and matching `.txt` captions in all three dataset directories. From another working directory, set `REPO` and `ROOT` to absolute paths:

```bash
REPO=/absolute/path/to/musubi-tuner-tools-extraction
ROOT=/absolute/path/to/qwen_image_lora_val_example
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd /tmp

python "$REPO/qwen_image_train_network.py" --config_file "$ROOT/train.toml"
```

In a separate terminal, monitor the six existing tags (`train_eval_loss_mean`, `train_eval_loss_low_noise`, `train_eval_loss_high_noise`, `val_loss_mean`, `val_loss_low_noise`, `val_loss_high_noise`):

```bash
tensorboard --logdir "$ROOT/output/tensorboard"
```

The training command uses the model paths and dataset declarations in `train.toml` to fill missing caches. The selected validation TOML processes both roles and preflights sources and cache collisions before model loading.

Fresh-run validation occurs at step 0, at each configured `val_every_n_steps` boundary, and at the final `max_train_steps` step. A coincident final and periodic step is evaluated once. The original BF16 DiT and BF16 training compute remain in use, while the sole resumable adapter is saved as FP32: BF16 saving would round that only copy and prevent exact state resume. To disable samples in `train.toml`, remove both `sample_prompts` and `sample_every_n_steps`; `sample_prompts = ""` is not a valid substitute. A step-0 new-best package still includes samples when sampling is enabled despite `sample_at_first = false`.

Each save produces one complete `<output_name>-step-<X>` directory under `output/current_training_states/` or, for the one best state, `output/val_training_states/val-loss/`. It has one FP32 LoRA `model.safetensors`, optimizer/scheduler files, RNG state for each rank, `val_loss_state.json`, `experiment_state.json`, and `samples/` with PNGs when sampling is enabled. Omitted, `float`, and `fp32` save precision are accepted; `fp16` and `bf16` are rejected because the sole model file must retain exact FP32 weights. There are no separate adapter exports or `*-state` duplicates in this mode.

Only a complete finite `val_unfamiliar` `val_loss_mean` selects best, and improvement must be strict: a tie keeps the earlier best. The first valid event can select best at step 0. Coincident periodic, epoch, final, new-best, and sample triggers create one package and one sample set. A sample-only trigger creates a full package; a required step-0 best includes PNGs when sampling is enabled even with `sample_at_first = false`. Sampling without prompts or a trigger is disabled.

`save_last_n_steps = 1000` retains whole current packages at steps `s >= X-1000` after publication at `X`; omitting the setting retains all. The best is protected even when older, and a former best returns to current only within the window. Resume from either published current or best package restores its own step and state; resuming an older current package still compares against the stored later best. Experiment mode rejects `save_last_n_steps_state`, `save_last_n_epochs`, `save_last_n_epochs_state`, and `save_state_to_huggingface`. See the [experiment CLI contract](../specs/004-experiment-training-states/contracts/experiment-cli.md) and [state contract](../specs/004-experiment-training-states/contracts/experiment-states.md).

To rename or move the entire prepared experiment, set `ROOT` to its new absolute path; the relative paths inside its TOMLs remain unchanged. From another CWD, choose **one** existing complete package at its actual absolute step `X` and run the matching command (the earlier `REPO` and `PYTHONPATH` settings still apply):

```bash
ROOT=/absolute/path/to/moved-qwen-image-experiment
cd /tmp
accelerate launch --mixed_precision bf16 "$REPO/qwen_image_train_network.py" --config_file "$ROOT/train.toml" --resume "$ROOT/output/current_training_states/<output_name>-step-<X>"
accelerate launch --mixed_precision bf16 "$REPO/qwen_image_train_network.py" --config_file "$ROOT/train.toml" --resume "$ROOT/output/val_training_states/val-loss/<output_name>-step-<X>"
```

Keep the validation images, captions, source-bound caches, and fixed noise controls intact for compatible resume. From saved step `s`, initial validation occurs once at `s`; the first new training-loss point is `s+1`. `max_train_steps=B` is this invocation's user-selected completed-update budget, so its final absolute step is `s+B`. Use the `output_name` configured in `train.toml` for the package path. Exact data-loader position restoration is not promised.

The full H200 run is a later private-server verification. Local CPU checks of this example do not establish real-model image quality, GPU memory use, or throughput. See the [example command contract](../specs/005-val-loss-configs/contracts/example-cli.md).

See [datasets](dataset_config.md), [sampling](sampling_during_training.md), [advanced options](advanced_config.md), [block swap](block_swap.md) and [compilation](torch_compile.md). No standalone inference, full finetuning, editing, layered, audio or video workflow is shipped.

## 日本語

元のQwen-Imageのアダプター学習のみを対象とします。画像とキャプションを用意し、latentとテキスト埋め込みの両方をキャッシュしてから学習します。`experiment_dir` を使わない従来のコマンドはリポジトリのルートで実行し、外部パスを置き換えてください。この場合、相対パスの基準は設定ファイルの場所ではなく、実行時の作業ディレクトリです。ポータブルな例の絶対パス `REPO` と `ROOT` を使うコマンドは、別の作業ディレクトリから実行できます。実モデルでの動作確認は別の運用段階です。
