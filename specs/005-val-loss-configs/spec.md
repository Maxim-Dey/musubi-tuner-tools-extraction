# Feature Specification: Qwen-Image Validation Example and Run Guide

**Feature Branch**: `001-scope-qwen-image-lora` (current branch; feature directory is independent)

**Created**: 2026-09-23

**Status**: Draft

**Input**: Stage 4.1 of `config_for_qwen_image_lora/val_loss_speckit_commands.md`: deliver physical example files and an exact guide for the completed Stage 1–3 contracts.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Prepare a portable experiment (Priority: P1)

The user copies a separate example, replaces clearly marked server model paths and `TOK`, supplies their own captioned images, and builds train and validation caches. The real Qwen commands accept the files without editing existing user configurations.

**Why this priority**: An example is useful only if the actual commands accept it and validation images stay out of training.

**Independent Test**: Parse all files with the current readers and preflight a relocated temporary folder with small real image, caption, and cache fixtures from another working directory.

**Acceptance Scenarios**:

1. **Given** the example with replaced server paths and real user data, **when** training configuration is selected, **then** all declared settings are accepted and relative paths resolve within the experiment root.
2. **Given** the two dataset configurations, **when** membership is inspected, **then** training has one unroled source and validation has exactly `val_familiar` and `val_unfamiliar`, each with a separate cache.
3. **Given** no user data, **when** the folder is inspected, **then** it contains no fabricated images or caches and clearly states what the user must supply.

---

### User Story 2 - Follow one complete run (Priority: P1)

The user follows exact commands to cache all three sets, launch original Qwen-Image LoRA training, monitor six validation series, and identify complete current and best packages.

**Why this priority**: The user needs executable instructions and clear metrics and state meanings.

**Independent Test**: Run the documented command path with temporary small CPU fixtures and controlled trainer collaborators through fixed inputs, evaluation, metrics, save, and resume.

**Acceptance Scenarios**:

1. **Given** prepared sources, **when** the four cache invocations are followed, **then** train caches remain separate and both validation roles are processed by each validation cache invocation.
2. **Given** a fresh run of 1,600 completed updates at interval 200, **when** validation succeeds, **then** events occur exactly at `0, 200, 400, 600, 800, 1000, 1200, 1400, 1600`, with one event at 1600 and six named series per event.
3. **Given** coincident periodic, final, sampled, and new-best reasons, **when** the step is saved, **then** one complete package and at most one sample set represent that absolute step; the lowest valid unfamiliar-set mean determines the only best.

---

### User Story 3 - Move and resume (Priority: P2)

The user can invoke commands from another working directory, rename the entire prepared folder, and resume from a current or best package. The guide distinguishes the saved absolute step from the new run's update budget.

**Why this priority**: Portable paths and truthful resume limits prevent confusing results.

**Independent Test**: Preflight a renamed fixture from two working directories and compare controlled current/best resume steps and fixed input identity.

**Acceptance Scenarios**:

1. **Given** a relocated root, **when** only the command's root path is changed, **then** internal relative dataset, cache, prompt, output, logging, and local resume paths resolve within the new root.
2. **Given** a package saved at absolute step `s`, **when** resumed with `max_train_steps=1600`, **then** initial validation runs once at `s`, new training loss starts at `s+1`, and the final absolute step is `s+1600` after 1,600 completed updates.
3. **Given** a current or best package, **when** resume is documented, **then** the guide names the correct location and does not promise restoration of the exact data-loader position.

### Edge Cases

- An unchanged placeholder model path, missing image/caption, or stale cache fails before weights load; empty data folders are not presented as ready data.
- Unknown, ill-typed, or conflicting example settings must fail in the real parser or preflight, not be ignored.
- The original requested `save_precision="bf16"` would round the only adapter file; the example explicitly uses `fp32` for exact resume while keeping BF16 training compute.
- `sample_at_first=false` suppresses a standalone initial sample; a valid step-0 best still has samples if sampling is enabled.
- Periodic and final validation coincide at 1600: publish one event.
- A current package at `X-save_last_n_steps` is retained, while an older best stays protected.
- Resume on a periodic boundary validates once at the saved absolute step.
- Disabling sampling removes both `sample_prompts` and `sample_every_n_steps` entries; an empty prompt string is not a valid substitute.
- Existing user files with matching basenames are not overwritten by the separate example folder.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Delivery MUST include physical `qwen_image_lora_val_example/train.toml`, `train-dataset.toml`, `val-dataset.toml`, and `sample_prompts.txt`, with coherent `dataset/train/`, `dataset/val_familiar/`, `dataset/val_unfamiliar/`, `cache/train/`, `cache/val_familiar/`, `cache/val_unfamiliar/`, and `output/` layout. The example folder name MUST NOT be hard-coded. Existing user configs/prompts MUST remain untouched; no invented image or cache may be offered as production data.
- **FR-002**: `train.toml` MUST select `model_version="original"`, `experiment_dir="."`, `dataset_config="train-dataset.toml"`, `val_dataset_config="val-dataset.toml"`, and `network_module="networks.lora_qwen_image"`. `dit`, `vae`, and `text_encoder` MUST be explicitly replaceable absolute server paths. The guide MUST identify the source DiT as BF16, not a preconverted FP8 checkpoint.
- **FR-003**: The example MUST contain the exact training, validation, logging, save, and sampling values in the Configuration Values table, plus only fields required by the implemented contract, including `output_dir="output"`. Every field MUST be accepted by the real Qwen trainer. `lr_warmup_steps=200` MUST remain an integer. Training `timestep_sampling="shift"` and `discrete_flow_shift=2.2` MUST NOT alter Stage 1's fixed validation levels.
- **FR-004**: The example MUST use `save_precision="fp32"` for the sole complete resumable adapter. The guide MUST explain this agreed change from the requested `bf16`: rounding the only adapter copy would prevent exact state resume. It MUST NOT add `save_last_n_steps_state` or duplicate adapter exports. `save_last_n_steps=1000` retains whole current packages on the inclusive absolute-step boundary and protects the current best.
- **FR-005**: `train-dataset.toml` MUST declare one unroled `image_directory="dataset/train"` and `cache_directory="cache/train"`, with `resolution=[1024,1024]`, `enable_bucket=true`, `bucket_no_upscale=true`, `caption_extension=".txt"`, `batch_size=1`, and `num_repeats=1`.
- **FR-006**: `val-dataset.toml` MUST use the same common settings and exactly two explicit roles: `val_familiar` with `dataset/val_familiar` and `cache/val_familiar`; `val_unfamiliar` with `dataset/val_unfamiliar` and `cache/val_unfamiliar`. It MUST have no random transformations. No validation source belongs in the training configuration.
- **FR-007**: `sample_prompts.txt` MUST contain at least the two exact lines below. The guide MUST explain replacing `TOK` and disabling samples by removing both `sample_prompts` and `sample_every_n_steps` from the training configuration, without using an empty prompt string.
- **FR-008**: The guide MUST provide four exact accepted cache commands: latent and text caching for each of `train-dataset.toml` and `val-dataset.toml`, with proper root and dataset selectors and an explicit `--vae` for each latent command or `--text_encoder` for each text command. `--train_config` MUST NOT be described as importing model paths into cache commands. The guide MUST also provide an `accelerate launch` command selecting `<root>/train.toml`, current and best resume commands, and a TensorBoard command for `<root>/output/tensorboard`. Commands MUST support invocation from another working directory.
- **FR-009**: The guide MUST explain whole-root moving/renaming without changing internal relative paths; preparation of captioned data and both cache types before training; and fixed validation membership, cache binding, and noise protocol for compatible resume.
- **FR-010**: The guide MUST name the fresh-run validation steps `0, 200, 400, 600, 800, 1000, 1200, 1400, 1600` without duplicating the last event and exactly six Stage 2 tags: `train_eval_loss_mean`, `train_eval_loss_low_noise`, `train_eval_loss_high_noise`, `val_loss_mean`, `val_loss_low_noise`, `val_loss_high_noise`. Only strict finite improvement of `val_loss_mean` selects best. Save and sample reasons coalesce by absolute completed step.
- **FR-011**: The guide MUST state that `max_train_steps` is a completed-update budget for this invocation, distinct from absolute step labels. On resume from `s` with budget `B`, validation runs at `s`, first new training loss is at `s+1`, and final absolute step is `s+B`; it MUST NOT promise exact data-loader position restoration.
- **FR-012**: Acceptance MUST parse every supplied file with the real readers, check flags and paths, source/cache separation, both roles, prompt syntax, and precision/retention constraints. Controlled CPU integration MUST exercise configuration → fixed inputs → evaluation → six metrics → complete save → resume with temporary small fixtures and no external weights/data. A full H200 run MUST be described as separate server verification, not performed or claimed locally.
- **FR-013**: The example and guide MUST preserve Stage 1–3 contracts and no-`experiment_dir` legacy behavior. They MUST NOT introduce new training mathematics, metric selectors, unsupported keys, fabricated user data, downloads, or support for other model families.

### Configuration Values

The physical `train.toml` must contain these exact effective values. The only agreed correction to the original list is FP32 adapter saving in FR-004.

| Controls | Exact values |
| --- | --- |
| Model and adapter | `model_version="original"`; `network_module="networks.lora_qwen_image"`; `network_dim=16`; `network_alpha=16`; replaceable absolute `dit`, `vae`, `text_encoder` |
| Paths | `experiment_dir="."`; `dataset_config="train-dataset.toml"`; `val_dataset_config="val-dataset.toml"`; `output_dir="output"`; `output_name="qwen_image_lora"` |
| Compute | `mixed_precision="bf16"`; `fp8_base=false`; `fp8_scaled=false`; `fp8_vl=false`; `blocks_to_swap=0`; `sdpa=true`; `gradient_checkpointing=true` |
| Updates | `max_train_steps=1600`; `gradient_accumulation_steps=1`; `seed=42`; `optimizer_type="adamw8bit"`; `learning_rate=5e-5`; `lr_scheduler="constant_with_warmup"`; `lr_warmup_steps=200`; `max_grad_norm=1.0` |
| Training noise and loading | `timestep_sampling="shift"`; `discrete_flow_shift=2.2`; `weighting_scheme="none"`; `max_data_loader_n_workers=0`; `persistent_data_loader_workers=false` |
| Fixed validation | `val_every_n_steps=200`; `val_seed_noise=42`; `val_level_noise_n=10`; `val_seed_noise_n=1` |
| Complete state and logging | `save_precision="fp32"`; `save_every_n_steps=200`; `save_last_n_steps=1000`; `save_state=true`; `log_with="tensorboard"`; `logging_dir="output/tensorboard"` |
| Sampling | `sample_prompts="sample_prompts.txt"`; `sample_every_n_steps=200`; `sample_at_first=false` |

Exact prompt lines:

```text
TOK, a gold coin on a plain background. --w 1024 --h 1024 --d 42 --s 30 --l 4.0 --fs 2.2
TOK, a copper teapot on a table. --w 1024 --h 1024 --d 42 --s 30 --l 4.0 --fs 2.2
```

### Key Entities

- **Example experiment folder**: Portable user-facing files and relative paths, except explicitly replaceable server model paths.
- **Training dataset declaration**: One unroled captioned source with its own cache.
- **Validation dataset declaration**: Two role-labelled captioned sources with separate caches and fixed membership.
- **Run guide**: Exact sequence for caching, training, monitoring, moving, and resuming the example.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All four physical example files exist; 100% of their settings and prompt lines pass the real readers after substitution of the three server paths and `TOK`; no existing user file is overwritten.
- **SC-002**: Local fixtures show exactly one unroled training source and exactly one nonempty source per validation role, with distinct declared caches and zero validation images in training.
- **SC-003**: Before and after renaming the root, commands run from two other working directories resolve 100% of internal relative paths within the current root.
- **SC-004**: A controlled scheduling check for a fresh 1,600-update budget yields exactly nine validation steps at `0, 200, 400, 600, 800, 1000, 1200, 1400, 1600`, with no duplicate at 1600. A separate short CPU integration exercises real validation events, six finite named loss values per successful event, save, and resume; it does not perform 1,600 optimizer updates.
- **SC-005**: At each exercised saved step, exactly one complete package and at most one sample set exist; only strict finite unfamiliar-set mean improvement changes best; the inclusive 1,000-step window retains all eligible owned current packages and the protected best.
- **SC-006**: Controlled CPU integration covers configuration through resume without external model files or downloads; the documented H200 run remains a separate server check.

## Assumptions

- [Stage 1](../002-val-loss-core/spec.md), [Stage 2](../003-val-loss-training/spec.md), and [Stage 3](../004-experiment-training-states/spec.md) supply existing validation, scheduling, and complete-state behavior. This feature delivers a usable example and guide, not new training behavior.
- The user supplies real images, `.txt` captions, server model files, and compatible caches. Empty directory structure is illustrative, not valid training input.
- The previously agreed FP32 save correction applies only to the sole resumable adapter; BF16 training computation and the original BF16 DiT remain requested.
- The H200 procedure is for later private-server verification. Local CPU fixtures do not establish real-model quality, memory use, or throughput.
