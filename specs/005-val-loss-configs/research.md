# Research: Portable Qwen-Image Validation Example

## Decisions

| Decision | Rationale | Alternative considered |
| --- | --- | --- |
| Deliver a separate `qwen_image_lora_val_example/` with four user-facing files and empty data/cache locations only. | The example must not overwrite existing user inputs or imply fabricated images/caches are usable. | Reusing `config_for_qwen_image_lora/` would affect existing configurations. |
| Select `<root>/train.toml` with trainer `--config_file`; keep `experiment_dir="."` in that file. | The trainer anchors a relative experiment root to the selected TOML, independently of process CWD. It requires the selected file to be exactly `<root>/train.toml`. | An absolute `--experiment_dir` without `--config_file` is rejected for training. |
| Select each dataset TOML explicitly for both cache commands, using `--train_config <root>/train.toml` as the root anchor. Supply `--vae` or `--text_encoder` explicitly. | Cache commands read only `experiment_dir` from the train TOML. Selecting `val-dataset.toml` processes both validation roles in one invocation per cache type. | Cache `--config_file`/`--val_dataset_config` and importing model paths from train TOML are unsupported. |
| Save the sole adapter as `save_precision="fp32"` while keeping `mixed_precision="bf16"` and an original BF16 DiT. | Experiment preflight rejects BF16 adapter saving; rounding the only adapter file would compromise exact full-state resume. This is the correction authorized by the Stage 4 input. | The requested `save_precision="bf16"` cannot satisfy the single-file resume contract. |
| Disable sampling by removing both `sample_prompts` and `sample_every_n_steps` entries. | Empty `sample_prompts=""` rebases to the experiment root, then fails the prompt-file preflight. With prompts and cadence present, a step-0 new-best package receives samples even when `sample_at_first=false`. | An empty string is not a disable switch. |
| Document current/best resume as a package-directory `--resume`, with `max_train_steps` as the new invocation's completed-update budget. | Stage 2/3 preserve the saved absolute step, publish one complete package per step, and do not promise exact data-loader position restoration. | Treating `max_train_steps` as an absolute stopping step would misstate the implemented behavior. |

## Verification boundary

The example and guide use existing Qwen-Image parsers, dataset readers, validation, and state machinery. Local acceptance uses temporary small CPU fixtures and the existing tests. Real model files, captioned images, both cache types, and an H200 run are supplied or verified separately on the private server. No training algorithm or legacy mode changes are part of this feature.
