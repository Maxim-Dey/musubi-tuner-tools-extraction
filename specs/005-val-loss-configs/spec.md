# Feature Specification: Qwen-Image Validation Example and Run Guide

**Feature Branch**: `001-scope-qwen-image-lora` (current branch; feature directory is independent)

**Created**: 2026-09-23

**Status**: Draft

**Input**: Stage 4.1 of `config_for_qwen_image_lora/val_loss_speckit_commands.md`: deliver physical example files and an exact guide for the completed Stage 1–3 contracts.

**Revision (2026-09-24)**: The user subsequently required a single training command that reads the training TOML and prepares caches automatically. This supersedes the original four-command cache preparation requirement; the original implementation preceded this SpecKit revision.

**Revision (2026-09-24, user decision)**: The supplied experiment configuration is editable by its user. Training steps, batch size, rank, learning rate, validation cadence, and other numerical controls are not a fixed example contract. The removed `config_for_qwen_image_lora/` folder is intentionally absent.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Prepare a portable experiment (Priority: P1)

The user configures the supplied experiment folder, replaces clearly marked server model paths and `TOK`, supplies their own captioned images, and starts training from the training TOML. The run prepares train and validation caches automatically without editing unrelated user configurations.

**Why this priority**: An example is useful only if the actual commands accept it and validation images stay out of training.

**Independent Test**: Parse all files with the current readers and preflight a relocated temporary folder with small real image, caption, and cache fixtures from another working directory.

**Acceptance Scenarios**:

1. **Given** the example with replaced server paths and real user data, **when** training configuration is selected, **then** all declared settings are accepted and relative paths resolve within the experiment root.
2. **Given** the two dataset configurations, **when** membership is inspected, **then** training has one unroled source and validation has exactly `val_familiar` and `val_unfamiliar`, each with a separate cache.
3. **Given** no user data, **when** the folder is inspected, **then** it contains no fabricated images or caches and clearly states what the user must supply.

---

### User Story 2 - Follow one complete run (Priority: P1)

The user runs one original Qwen-Image LoRA training command with `--config_file`, then monitors six validation series and identifies complete current and best packages.

**Why this priority**: The user needs executable instructions and clear metrics and state meanings.

**Independent Test**: Run the documented command path with temporary small CPU fixtures and controlled trainer collaborators through fixed inputs, evaluation, metrics, save, and resume.

**Acceptance Scenarios**:

1. **Given** captioned train and validation sources, **when** the user starts training with the training TOML, **then** both cache types are checked for every configured dataset, existing train caches are skipped, missing caches are created, and stale validation caches are repaired before training weights load.
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

- An unchanged placeholder model path or missing image/caption fails before training weights load; empty data folders are not presented as ready data. A missing cache is created; a stale validation cache is rebuilt or reported if repair fails.
- Repeated launches with unchanged sources do not load cache encoder weights when all cache files are present. Training-cache freshness beyond file existence is not promised.
- Unknown, ill-typed, or conflicting example settings must fail in the real parser or preflight, not be ignored.
- The original requested `save_precision="bf16"` would round the only adapter file; the example explicitly uses `fp32` for exact resume while keeping BF16 training compute.
- `sample_at_first=false` suppresses a standalone initial sample; a valid step-0 best still has samples if sampling is enabled.
- When periodic and final validation coincide, publish one event. The 1,600/200 scheduling test exercises this case.
- A current package at `X-save_last_n_steps` is retained, while an older best stays protected.
- Resume on a periodic boundary validates once at the saved absolute step.
- Disabling sampling removes both `sample_prompts` and `sample_every_n_steps` entries; an empty prompt string is not a valid substitute.
- Existing user files with matching basenames are not overwritten by the separate example folder.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Delivery MUST include physical `qwen_image_lora_val_example/train.toml`, `train-dataset.toml`, `val-dataset.toml`, and `sample_prompts.txt`, with coherent `dataset/train/`, `dataset/val_familiar/`, `dataset/val_unfamiliar/`, `cache/train/`, `cache/val_familiar/`, `cache/val_unfamiliar/`, and `output/` layout. The example folder name MUST NOT be hard-coded. Existing user configs/prompts MUST remain untouched; no invented image or cache may be offered as production data.
- **FR-002**: `train.toml` MUST select `model_version="original"`, `experiment_dir="."`, `dataset_config="train-dataset.toml"`, `val_dataset_config="val-dataset.toml"`, and `network_module="networks.lora_qwen_image"`. `dit`, `vae`, and `text_encoder` MUST be explicitly replaceable absolute server paths. The guide MUST identify the source DiT as BF16, not a preconverted FP8 checkpoint.
- **FR-003**: The supplied `train.toml` MUST be an editable experiment configuration. The user chooses training steps and other numerical controls; tests MUST NOT require particular values from that file. Every field MUST be accepted by the real Qwen trainer. `lr_warmup_steps` MUST be an integer. Training `timestep_sampling="shift"` and `discrete_flow_shift=2.2` MUST NOT alter Stage 1's fixed validation levels.
- **FR-004**: The example MUST use `save_precision="fp32"` for the sole complete resumable adapter. The guide MUST explain this agreed change from the requested `bf16`: rounding the only adapter copy would prevent exact state resume. It MUST NOT add `save_last_n_steps_state` or duplicate adapter exports. The configured `save_last_n_steps` retains whole current packages on the inclusive absolute-step boundary and protects the current best.
- **FR-005**: `train-dataset.toml` MUST declare one unroled `image_directory="dataset/train"` and `cache_directory="cache/train"`, with `resolution=[1024,1024]`, `enable_bucket=true`, `bucket_no_upscale=true`, `caption_extension=".txt"`, and `num_repeats=1`. The user chooses a valid training `batch_size`.
- **FR-006**: `val-dataset.toml` MUST use the same resolution, bucketing, caption, and repeat settings, plus effective `batch_size=1`, and exactly two explicit roles: `val_familiar` with `dataset/val_familiar` and `cache/val_familiar`; `val_unfamiliar` with `dataset/val_unfamiliar` and `cache/val_unfamiliar`. It MUST have no random transformations. No validation source belongs in the training configuration.
- **FR-007**: `sample_prompts.txt` MUST contain at least the two exact lines below. The guide MUST explain replacing `TOK` and disabling samples by removing both `sample_prompts` and `sample_every_n_steps` from the training configuration, without using an empty prompt string.
- **FR-008**: The ordinary guide MUST require only `python qwen_image_train_network.py --config_file <root>/train.toml` after the user configures models and captioned datasets. The training command MUST take dataset and VAE/text-encoder paths from the effective training configuration, prepare both cache types for every configured dataset before training-model loading, skip existing training caches, create missing caches, and rebuild stale validation caches detected by validation preflight. The user MUST NOT need a cache command or rebuild flag for this path. The guide MUST also provide current and best resume locations and a TensorBoard command. The command MUST support invocation from another working directory when the package is installed or on `PYTHONPATH`.
- **FR-009**: The guide MUST explain whole-root moving/renaming without changing internal relative paths; preparation of captioned source data before the one training command; and fixed validation membership, cache binding, and noise protocol for compatible resume.
- **FR-010**: The guide MUST explain that validation runs at step 0, each configured `val_every_n_steps` boundary, and the configured final step, with no duplicate event when boundaries coincide. It MUST name exactly six Stage 2 tags: `train_eval_loss_mean`, `train_eval_loss_low_noise`, `train_eval_loss_high_noise`, `val_loss_mean`, `val_loss_low_noise`, `val_loss_high_noise`. Only strict finite improvement of `val_loss_mean` selects best. Save and sample reasons coalesce by absolute completed step.
- **FR-011**: The guide MUST state that `max_train_steps` is a completed-update budget for this invocation, distinct from absolute step labels. On resume from `s` with budget `B`, validation runs at `s`, first new training loss is at `s+1`, and final absolute step is `s+B`; it MUST NOT promise exact data-loader position restoration.
- **FR-012**: Acceptance MUST parse every supplied file with the real readers, check paths, source/cache separation, both roles, prompt syntax, and precision/retention constraints. Local tests MUST verify that the training command dispatches cache preparation from the effective TOML, an unchanged rerun skips encoder loading, missing files trigger encoding, and invalid validation caches trigger repair. Controlled CPU integration MUST exercise configuration → fixed inputs → evaluation → six metrics → complete save → resume with temporary small fixtures and no external weights/data. A full H200 run MUST be described as separate server verification, not performed or claimed locally.
- **FR-013**: The example and guide MUST preserve Stage 1–3 contracts and no-`experiment_dir` legacy training behavior when `--config_file` is absent. Automatic cache preparation MUST finish before the training model is loaded and MUST NOT consume the parent training process's RNG or retain cache encoder weights during training. They MUST NOT introduce new training mathematics, metric selectors, unsupported keys, fabricated user data, downloads, or support for other model families.

### Configuration Controls

The physical `train.toml` supplies starting values that the user may edit before training. `max_train_steps` is the user's completed-update budget; validation, save, and sample intervals are also read from that file. Tests may set 1,600 steps and a 200-step validation interval to exercise scheduling, but these values are not required in the supplied experiment. Model family, dataset roles, portable paths, BF16 computation, FP32 resumable adapter saving, and the fixed-validation algorithm remain governed by FR-002–FR-013.

Exact prompt lines:

```text
TOK, a gold coin on a plain background. --w 1024 --h 1024 --d 42 --s 30 --l 4.0 --fs 2.2
TOK, a copper teapot on a table. --w 1024 --h 1024 --d 42 --s 30 --l 4.0 --fs 2.2
```

### Key Entities

- **Example experiment folder**: Portable user-facing files and relative paths, except explicitly replaceable server model paths.
- **Training dataset declaration**: One unroled captioned source with its own cache.
- **Validation dataset declaration**: Two role-labelled captioned sources with separate caches and fixed membership.
- **Run guide**: One training command plus monitoring, moving, and resuming the example.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All four physical example files exist; 100% of their settings and prompt lines pass the real readers after substitution of the three server paths and `TOK`; no existing user file is overwritten.
- **SC-002**: Local fixtures show exactly one unroled training source and exactly one nonempty source per validation role, with distinct declared caches and zero validation images in training.
- **SC-003**: Before and after renaming the root, commands run from two other working directories resolve 100% of internal relative paths within the current root.
- **SC-004**: A controlled scheduling check for a fresh 1,600-update budget yields exactly nine validation steps at `0, 200, 400, 600, 800, 1000, 1200, 1400, 1600`, with no duplicate at 1600. A separate short CPU integration exercises real validation events, six finite named loss values per successful event, save, and resume; it does not perform 1,600 optimizer updates.
- **SC-005**: At each exercised saved step, exactly one complete package and at most one sample set exist; only strict finite unfamiliar-set mean improvement changes best; a controlled 1,000-step retention window retains all eligible owned current packages and the protected best.
- **SC-006**: Controlled CPU integration covers configuration through resume without external model files or downloads; the documented H200 run remains a separate server check.
- **SC-007**: With configured models and captioned datasets, one `--config_file` training invocation prepares all missing cache types across train and both validation roles without separate user commands. A repeated invocation with complete caches loads no cache encoder weights; an invalid validation cache is rebuilt before the training model loads.

## Assumptions

- [Stage 1](../002-val-loss-core/spec.md), [Stage 2](../003-val-loss-training/spec.md), and [Stage 3](../004-experiment-training-states/spec.md) supply existing validation, scheduling, and complete-state behavior. This feature delivers a usable example and guide, not new training behavior.
- The user supplies real images, `.txt` captions, and server model files. The training command prepares caches; empty directory structure is illustrative, not valid training input.
- The previously agreed FP32 save correction applies only to the sole resumable adapter; BF16 training computation and the original BF16 DiT remain requested.
- The H200 procedure is for later private-server verification. Local CPU fixtures do not establish real-model quality, memory use, or throughput.
