# Data Model: Portable Experiment and Complete States

## Experiment root

| Field | Meaning and validation |
| --- | --- |
| `root` | Absolute normalized directory. A relative `experiment_dir` is anchored to the selected train TOML parent; an absolute value is unchanged. |
| `train_config` | Selected `train.toml` when present. The trainer uses `--config_file`; cache commands use `--train_config`. |
| `dataset_config`, `val_dataset_config` | Root-resolved TOML files. The train file remains unroled; the validation file has exactly the two Stage 1 roles. |
| `output_dir`, `logging_dir` | Exactly `root/output` and `root/output/tensorboard` in experiment mode. Conflicting effective explicit assignments fail. |

Ownership is established only for a complete package under the selected root with the exact `output_name`/step filename, matching `output_name`, model version, fingerprint and controls in its package sidecar, and a matching Stage 2 sidecar. Unknown or incomplete directories are unrelated and cannot be pruned.

## Effective path set

The resolved effective configuration contains the root-absolute trainer paths for model files, dataset/validation TOMLs, prompts, tracker configuration, local resume, output and logging. Dataset TOML loading resolves each `image_directory`, `image_jsonl_file`, `cache_directory`, and relative JSONL `image_path` against the same root. Absolute paths stay absolute. No persistent rewritten TOML is needed; the normalized in-memory values feed the existing blueprint and Stage 1 frozen-item recheck. The Stage 1 fingerprint remains path independent.

## Validation event

| Field | Meaning |
| --- | --- |
| `absolute_completed_step` | Exact nonnegative integer `s+r` from Stage 2, where `s` is loaded and `r` counts completed updates in this run. |
| `metrics_at_step` | Optional six finite Stage 2 scalar values, published together only after a complete event. `val_loss_mean` is the sole best comparator. |
| `save_reasons` | Set drawn from `initial_best`, `periodic`, `epoch`, `final`, `new_best`, `sample_trigger`; it is collapsed to one package decision. |
| `sampling_enabled` | True only with valid prompts and configured sample trigger. A saved package then needs one PNG set regardless of which save reason won. |

A failed/nonfinite validation event cannot be a best candidate. A sample-only trigger is a save reason, so it creates one complete package. A step-0 best receives samples even when `sample_at_first=false`.

## Complete step package

Only a published, verified directory is a package. Its basename is `<output_name>-step-<absolute_completed_step>`; its parent is exactly one of `current_training_states` or `val_training_states/val-loss`.

| Required content | Source / rule |
| --- | --- |
| `model.safetensors` | Exactly one canonical FP32 LoRA tensor file; no frozen DiT, second model file, or separate adapter export. |
| `optimizer.bin`, `scheduler.bin` | Accelerate 1.6 names or verified actual equivalents; preserve optimizer/scheduler state. |
| `random_states_<rank>.pkl` | One for every participating rank; validate existence and required RNG fields before resume. |
| `val_loss_state.json` | Unmodified Stage 2 schema: absolute step, model version, frozen validation fingerprint and controls. |
| `experiment_state.json` | Versioned ownership, same step/protocol, reasons and optional metrics measured at this step. |
| `samples/` | One PNG set from these weights when sampling is enabled; empty directory allowed when disabled. |

`experiment_state.json` uses `version = "qwen-image-experiment-state-v1"`, `output_name`, `model_version = "original"`, `absolute_completed_step`, `validation_fingerprint`, the four effective Stage 2 controls, `save_reasons`, `metrics_at_step` (six finite values or null), and `sample_count` (exact nonnegative integer). `sample_count = 0` means sampling was disabled; a positive value requires exactly that many valid PNG files. Package verification compares the count before publication, load, best selection, and retention. The two sidecars' step/protocol must agree. After resume the authoritative current best is the complete package physically in `val-loss/` and its own `metrics_at_step.val_loss_mean`, never a value from an older current package.

## Best record

There is at most one owned complete package under `val-loss/`. Its own `metrics_at_step.val_loss_mean` is finite and is the current best score; its step is the best step. A candidate replaces it only when `candidate < current`. A tie, nonfinite value, partial event or failed package write leaves the best unchanged. On replacement the previous best moves by directory rename into current if `old_step >= current_step - save_last_n_steps`, or is removed whole after the replacement succeeds if outside the window. No weights or PNGs are copied.

When resuming from an older current package, the loaded model's step and own metrics come from that package. The comparator independently reads the currently published best. It never assigns best metrics to the loaded model. A conflicting package already present at the next step is a collision, not permission to overwrite it.

## Retention window

`save_last_n_steps` is optional. At completed step `X`, owned current package `s` is retained when unset or `s >= X-N`; otherwise it may be deleted whole. Equality is included. The published best is exempt regardless of age. Explicit `save_last_n_steps_state` is invalid in experiment mode. Dataset, cache, config, unknown folders and incomplete staging directories are never retention targets.

## Package state transitions

```text
no package for X
  → staging (samples + all-rank Accelerate files + sidecars)
  → verified complete
  → published current OR published best
  → retained current / moved to best / pruned current
```

Any staging failure ends without a published package and without changing the prior best. A best replacement stages the new package first, then moves directory locations; on a handled move error it restores the prior best and reports failure. The implementation must not claim atomicity across abrupt process or machine failure between multiple directory moves.
