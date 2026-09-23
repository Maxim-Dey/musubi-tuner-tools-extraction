# Qwen-Image original adapter training

This repository trains original Qwen-Image adapters with the existing Musubi Tuner engine. The supported workflow is captioned images → latent cache → caption embedding cache → adapter training with optional sample PNGs. Use the [Russian README](../README.ru.md) and the supplied [training template](../config_for_qwen_image_lora/train.toml).

For the existing workflow, run from the repository root. Replace all `/srv/...` paths with your inputs; without `experiment_dir`, paths are resolved from process CWD, including paths inside TOML and JSONL.

```bash
python qwen_image_cache_latents.py --dataset_config config_for_qwen_image_lora/dataset.toml --vae /srv/models/qwen_image_vae.safetensors --model_version original
python qwen_image_cache_text_encoder_outputs.py --dataset_config config_for_qwen_image_lora/dataset.toml --text_encoder /srv/models/qwen_2.5_vl_7b.safetensors --model_version original
accelerate launch --mixed_precision bf16 qwen_image_train_network.py --config_file config_for_qwen_image_lora/train.toml
```

The same commands are available as `python -m musubi_tuner.qwen_image_cache_latents`, `python -m musubi_tuner.qwen_image_cache_text_encoder_outputs`, and `python -m musubi_tuner.qwen_image_train_network`. The package must be installed or `src` included in `PYTHONPATH`.

Use the original DiT, RGB VAE and Qwen2.5-VL weights. The text loader obtains tokenizer assets from `Qwen/Qwen-Image`, subfolder `tokenizer`; prepare them in the operational environment. DiT execution uses bf16. The training VAE dtype default is bf16; the Qwen VAE loader retains its existing loading behavior. `num_layers` defaults to 60 and must match the checkpoint.

The supplied paths, H200 comment, rank 16, 1600 updates and batch/resolution values are examples, not fixed requirements. Defaults are overridden by training TOML and then explicit CLI. Omitted CLI flags preserve TOML; store-true flags cannot disable a true TOML setting. Unknown fields/types, excluded selectors and invalid effective values fail early. The only model version is `original`. [Alternative adapters](loha_lokr.md) retain the Qwen model engine.

Both cache commands have independent CLI settings. Without `--train_config`, they do not read training TOML; they do not accept the trainer's `--config_file`. `batch_size` caps encoding chunks; the dataset declaration controls training batch size. `num_workers` must be positive when explicitly set. `skip_existing` checks existence only; rebuild affected caches after changing images, captions, weights or relevant settings. `keep_cache` preserves stale files that normal cache cleanup removes. Latent cache debug supports `image` and `console`; Qwen latent caching does not accept an explicit `vae_dtype` or tiling/chunk options.

Latent caches preserve `[C,1,H,W]`; the singleton axis is required by the original VAE. Text caches keep variable-length `[L,D]` embeddings. Training pads text and preserves masks/split attention. Missing text caches are warned and skipped; an empty effective training dataset fails before models.

## Validation input preparation

For Qwen-Image original, provide a separate role-bearing `val-dataset.toml` with exactly one nonempty `val_familiar` and one nonempty `val_unfamiliar` dataset. Each has captioned images (or captioned JSONL records), its own cache directory, and effective `batch_size = 1` and `num_repeats = 1`. Pass this same file to both Qwen cache commands with `--dataset_config`, without `--skip_existing`; their role-aware preflight rejects missing captions and ambiguous cache names before loading weights. Validation requires both newly generated caches bound to the original image bytes. Legacy unroled cache and training configurations keep their existing behavior.

Set `val_dataset_config` in the training TOML or CLI to enable validation; omit it to preserve the previous training behavior. Qwen-only controls are `val_every_n_steps` (default `200`, integer `>=1`), `val_seed_noise` (default `42`, signed integer), `val_level_noise_n` (default `10`, even integer `>=2`), and `val_seed_noise_n` (default `1`, integer `>=1`). Effective values follow defaults → TOML → explicit CLI; booleans and unknown keys are rejected. Invalid roles, empty sets, missing or changed images, captions or caches fail with source-labelled errors before model loading or on a later read. The prepared input has a portable identity and deterministic noise checks. See the [input contract](../specs/002-val-loss-core/contracts/validation-inputs.md).

After preparing the ordinary training caches, create `val-dataset.toml` in the repository root and prepare both validation caches before training (replace `/srv/...` with your weights):

```bash
python qwen_image_cache_latents.py --dataset_config val-dataset.toml --vae /srv/models/qwen_image_vae.safetensors --model_version original
python qwen_image_cache_text_encoder_outputs.py --dataset_config val-dataset.toml --text_encoder /srv/models/qwen_2.5_vl_7b.safetensors --model_version original
accelerate launch --mixed_precision bf16 qwen_image_train_network.py --config_file config_for_qwen_image_lora/train.toml --val_dataset_config val-dataset.toml
```

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

The example is a template, not prepared training data. Replace its three marked absolute paths with the server's original BF16 DiT (not a preconverted FP8 file), VAE, and text encoder; replace `TOK` in both prompt lines; supply your own images and matching `.txt` captions in all three dataset directories. Then build latent and text caches for train and both validation roles. The val TOML handles both roles in each cache command. From another working directory, set `REPO` and `ROOT` to absolute paths and replace `VAE` and `TEXT_ENCODER` with the same server files specified in `train.toml`:

```bash
REPO=/absolute/path/to/musubi-tuner-tools-extraction
ROOT=/absolute/path/to/qwen_image_lora_val_example
VAE=/absolute/server/path/to/vae.safetensors
TEXT_ENCODER=/absolute/server/path/to/text_encoder.safetensors
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd /tmp

python "$REPO/qwen_image_cache_latents.py" --train_config "$ROOT/train.toml" --dataset_config train-dataset.toml --vae "$VAE" --model_version original
python "$REPO/qwen_image_cache_text_encoder_outputs.py" --train_config "$ROOT/train.toml" --dataset_config train-dataset.toml --text_encoder "$TEXT_ENCODER" --model_version original
python "$REPO/qwen_image_cache_latents.py" --train_config "$ROOT/train.toml" --dataset_config val-dataset.toml --vae "$VAE" --model_version original
python "$REPO/qwen_image_cache_text_encoder_outputs.py" --train_config "$ROOT/train.toml" --dataset_config val-dataset.toml --text_encoder "$TEXT_ENCODER" --model_version original

accelerate launch --mixed_precision bf16 "$REPO/qwen_image_train_network.py" --config_file "$ROOT/train.toml"
```

In a separate terminal, monitor the six existing tags (`train_eval_loss_mean`, `train_eval_loss_low_noise`, `train_eval_loss_high_noise`, `val_loss_mean`, `val_loss_low_noise`, `val_loss_high_noise`):

```bash
tensorboard --logdir "$ROOT/output/tensorboard"
```

The cache commands read only `experiment_dir` from `--train_config`; cache `--experiment_dir` overrides it. An absolute cache root works without `--train_config`, while a relative root requires that anchor. The selected validation TOML processes both roles and preflights sources and cache collisions before model loading.

With the example's 1,600 completed updates and 200-step validation interval, fresh-run validation steps are `0, 200, 400, 600, 800, 1000, 1200, 1400, 1600`; the final step is evaluated once. The original BF16 DiT and BF16 training compute remain in use, while the sole resumable adapter is saved as FP32: BF16 saving would round that only copy and prevent exact state resume. To disable samples in `train.toml`, remove both `sample_prompts` and `sample_every_n_steps`; `sample_prompts = ""` is not a valid substitute. A step-0 new-best package still includes samples when sampling is enabled despite `sample_at_first = false`.

Each save produces one complete `<output_name>-step-<X>` directory under `output/current_training_states/` or, for the one best state, `output/val_training_states/val-loss/`. It has one FP32 LoRA `model.safetensors`, optimizer/scheduler files, RNG state for each rank, `val_loss_state.json`, `experiment_state.json`, and `samples/` with PNGs when sampling is enabled. Omitted, `float`, and `fp32` save precision are accepted; `fp16` and `bf16` are rejected because the sole model file must retain exact FP32 weights. There are no separate adapter exports or `*-state` duplicates in this mode.

Only a complete finite `val_unfamiliar` `val_loss_mean` selects best, and improvement must be strict: a tie keeps the earlier best. The first valid event can select best at step 0. Coincident periodic, epoch, final, new-best, and sample triggers create one package and one sample set. A sample-only trigger creates a full package; a required step-0 best includes PNGs when sampling is enabled even with `sample_at_first = false`. Sampling without prompts or a trigger is disabled.

`save_last_n_steps = 1000` retains whole current packages at steps `s >= X-1000` after publication at `X`; omitting the setting retains all. The best is protected even when older, and a former best returns to current only within the window. Resume from either published current or best package restores its own step and state; resuming an older current package still compares against the stored later best. Experiment mode rejects `save_last_n_steps_state`, `save_last_n_epochs`, `save_last_n_epochs_state`, and `save_state_to_huggingface`. See the [experiment CLI contract](../specs/004-experiment-training-states/contracts/experiment-cli.md) and [state contract](../specs/004-experiment-training-states/contracts/experiment-states.md).

To rename or move the entire prepared experiment, set `ROOT` to its new absolute path; the relative paths inside its TOMLs remain unchanged. From another CWD, choose **one** existing complete package at its actual absolute step `X` and run the matching command (the earlier `REPO` and `PYTHONPATH` settings still apply):

```bash
ROOT=/absolute/path/to/moved-qwen-image-experiment
cd /tmp
accelerate launch --mixed_precision bf16 "$REPO/qwen_image_train_network.py" --config_file "$ROOT/train.toml" --resume "$ROOT/output/current_training_states/qwen_image_lora-step-<X>"
accelerate launch --mixed_precision bf16 "$REPO/qwen_image_train_network.py" --config_file "$ROOT/train.toml" --resume "$ROOT/output/val_training_states/val-loss/qwen_image_lora-step-<X>"
```

Keep the validation images, captions, source-bound caches, and fixed noise controls intact for compatible resume. From saved step `s`, initial validation occurs once at `s`; the first new training-loss point is `s+1`. `max_train_steps=B` is this invocation's completed-update budget, so its final absolute step is `s+B` (the example uses `B=1600`). Exact data-loader position restoration is not promised.

The full H200 run is a later private-server verification. Local CPU checks of this example do not establish real-model image quality, GPU memory use, or throughput. See the [example command contract](../specs/005-val-loss-configs/contracts/example-cli.md).

See [datasets](dataset_config.md), [sampling](sampling_during_training.md), [advanced options](advanced_config.md), [block swap](block_swap.md) and [compilation](torch_compile.md). No standalone inference, full finetuning, editing, layered, audio or video workflow is shipped.

## 日本語

元のQwen-Imageのアダプター学習のみを対象とします。画像とキャプションを用意し、latentとテキスト埋め込みの両方をキャッシュしてから学習します。`experiment_dir` を使わない従来のコマンドはリポジトリのルートで実行し、外部パスを置き換えてください。この場合、相対パスの基準は設定ファイルの場所ではなく、実行時の作業ディレクトリです。ポータブルな例の絶対パス `REPO` と `ROOT` を使うコマンドは、別の作業ディレクトリから実行できます。実モデルでの動作確認は別の運用段階です。
