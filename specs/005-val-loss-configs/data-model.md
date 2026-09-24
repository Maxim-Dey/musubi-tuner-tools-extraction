# Data Model: Portable Validation Example

## Experiment root

`qwen_image_lora_val_example/` is a movable folder. Its name has no special meaning to the code. It contains `train.toml`, `train-dataset.toml`, `val-dataset.toml`, `sample_prompts.txt`, and the `dataset/`, `cache/`, and `output/` subtrees. The user supplies real images with matching `.txt` captions; the training command builds caches before loading training weights. Empty source directories are not valid inputs. The three model paths in `train.toml` are explicitly replaceable absolute server paths.

## Configuration entities

| Entity | Fields and relationships | Validation |
| --- | --- | --- |
| Training configuration | `experiment_dir="."`; root-relative train/val TOMLs, prompts, output and TensorBoard paths; original Qwen-Image LoRA controls; absolute DiT/VAE/text encoder paths. | The selected file is `<root>/train.toml`; the effective fields are accepted after defaults, TOML and CLI merging. `output_dir="output"`, `logging_dir="output/tensorboard"`, and `save_precision="fp32"` satisfy experiment preflight. |
| Training dataset | One unroled `[[datasets]]` source with `image_directory="dataset/train"` and `cache_directory="cache/train"`; common resolution, bucket, caption, batch and repeat settings. | No validation role or source appears here. Each image has a caption and matching latent/text cache before training. |
| Validation dataset | Exactly two role-labelled `[[datasets]]` sources, `val_familiar` and `val_unfamiliar`, with separate image and cache directories under the root. | Both roles are nonempty, captions and source-bound caches match, and fixed validation membership/protocol is preserved on resume. No random transforms are configured. |
| Prompt file | Two lines using `TOK` and the supported `--w`, `--h`, `--d`, `--s`, `--l`, `--fs` syntax. | Replace `TOK` with the intended trigger. Remove both prompt and cadence settings to disable sampling; do not use an empty path. |
| Cache sets | Latent and text caches for train, familiar validation, and unfamiliar validation, each in its declared cache directory. | One training invocation checks all configured sets through the existing Qwen cache commands. It creates missing files and repairs stale validation caches before the training model loads. |
| Training state | Complete package in `output/current_training_states/<output_name>-step-<X>/` or the sole best in `output/val_training_states/val-loss/<output_name>-step-<X>/`. | One FP32 `model.safetensors`, optimizer/scheduler and per-rank RNG state, both JSON sidecars, and PNGs when sampling is enabled. Best changes only on strict finite improvement of `val_loss_mean`. |

## State and path transitions

1. The train TOML anchors the root. Dataset TOML image/cache paths and JSONL image paths, if any, resolve from that root; absolute model paths stay absolute.
2. Cache preparation fills the three distinct declared cache directories. Stage 1 preflight freezes validation sources and cache bindings before DiT loading.
3. A fresh user-selected `B`-update run validates at step 0, configured periodic boundaries, and step `B`, with one event when the final step is also periodic and six loss tags. Package reasons at one absolute step coalesce into one physical package and at most one sample set.
4. A published package at absolute step `s` can resume from either tree. Initial validation occurs at `s`; `B` newly completed updates end at `s+B`. The data-loader position is not promised to match an uninterrupted run.
5. After publication at step `X`, current packages with `step >= X-1000` remain within the inclusive retention window; the best is protected even when older. Moving or renaming the entire root changes only the command's root selector, not internal relative paths.
