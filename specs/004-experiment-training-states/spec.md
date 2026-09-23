# Feature Specification: Portable Experiment and Complete Training States

**Feature Branch**: `001-scope-qwen-image-lora` (current branch; feature directory is independent)

**Created**: 2026-09-23

**Status**: Draft

**Input**: Stage 3.1 of `config_for_qwen_image_lora/val_loss_speckit_commands.md`: an explicitly enabled experiment folder and complete, nonduplicated training states for Qwen-Image original LoRA with the completed Stage 1 and Stage 2 validation contracts.

## Clarifications

### Session 2026-09-23

- Q: What happens when samples are due on a step without a periodic, final, or new-best save? → A: The sample trigger explicitly saves one complete package at that step, increasing the complete-package count by one while keeping one sample set and no detached or hidden checkpoint.
- Q: What happens when `sample_at_first=false` but a valid step-0 validation result becomes best? → A: The required step-0 best package includes one sample set when sampling is enabled; `sample_at_first=false` suppresses only a standalone initial sample request and creates no second package.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Move and run one experiment folder (Priority: P1)

The user keeps the training configuration, datasets, caches, prompts, logs, and results under one chosen experiment root. They can move that folder or start the commands from another working directory without changing relative paths. Training and both cache commands agree on the same train and validation datasets.

**Why this priority**: A single portable folder prevents training from silently using stale inputs or writing results elsewhere.

**Independent Test**: Run path and configuration preflight on a renamed folder from a different working directory, then check the resolved training, validation, cache, prompt, output, logging, and resume paths without loading model weights.

**Acceptance Scenarios**:

1. **Given** a selected `train.toml` and a relative `experiment_dir`, **when** the folder is moved and the user invokes a command from another working directory, **then** the root is found relative to that `train.toml`, all relative experiment paths still resolve inside the moved root, and absolute paths remain absolute.
2. **Given** an `experiment_dir` with no selected `<root>/train.toml` or with either required train/validation dataset TOML missing, **when** trainer configuration is checked, **then** it rejects the incomplete experiment before model loading and identifies the required file or setting. An absolute root removes only the need for a relative-path anchor, not the training-configuration requirement.
3. **Given** a single `val-dataset.toml` with both validation roles, **when** either Qwen cache command uses it, **then** it prepares the existing latent or text cache format for both roles without adding validation images to the training dataset.
4. **Given** no `experiment_dir`, **when** an existing training or cache configuration is used, **then** its prior path, output, and save behavior remains available.

---

### User Story 2 - Keep one complete state per saved step (Priority: P1)

The user can resume from one step-labelled folder containing the LoRA adapter, optimizer and scheduler state, each rank's random state, metadata, and that step's samples when sampling is enabled. A step that is periodic, final, or a new best still produces one package and one set of samples.

**Why this priority**: Complete packages remove ambiguity about which weights, metrics, and samples belong together and avoid storage duplication.

**Independent Test**: Trigger periodic, final, and new-best reasons on the same controlled step; inspect the package, count adapter files and sample generations, then resume from that package and compare the next controlled update.

**Acceptance Scenarios**:

1. **Given** several save reasons on absolute step `s`, **when** the save succeeds, **then** exactly one complete `output_name` and step-labelled package exists for `s`, with one `model.safetensors` and no second copy of the same adapter weights.
2. **Given** two ranks, **when** a package is complete, **then** it contains a random-state file for each rank and a resume restores weights, optimizer, scheduler, random states, and the absolute completed-update step.
3. **Given** an incomplete write, **when** the package is inspected or a best replacement is attempted, **then** it is unavailable for resume or best selection and the previous valid best remains intact.

---

### User Story 3 - Preserve the best validation state and whole-package retention (Priority: P1)

The user always has one best complete state selected by the lowest valid unfamiliar-set mean loss. Current packages remain for the requested optimizer-step window; an older best stays protected. Changing best moves folders without copying weights or regenerating samples.

**Why this priority**: The user needs an unambiguous best adapter and predictable storage use without losing resumable state.

**Independent Test**: Feed finite improving, tied, and failed validation events across steps, inspect current and best folders at the inclusive retention boundary, and resume from both locations.

**Acceptance Scenarios**:

1. **Given** the first complete validation result, including one at step 0, **when** its package is saved, **then** it becomes the single best state; a later strictly lower `val_loss_mean` replaces it, while an equal value leaves the previous best in place.
2. **Given** a best replacement, **when** the new package is complete, **then** the former best moves to current storage if still within the step window and otherwise is removed; no step exists in both locations.
3. **Given** `save_last_n_steps=N` and current step `X`, **when** retention runs, **then** current packages with step `s >= X-N` remain whole, older owned current packages are removed whole, and the current best remains regardless of age.

---

### User Story 4 - Keep requested samples with their weights (Priority: P2)

The user can inspect samples under the state package for the same step and weights, with one generation per absolute step. Sampling neither changes the training path nor creates detached copies in the old shared sample folder.

**Why this priority**: A sample is meaningful only when its source weights and step can be identified.

**Independent Test**: Exercise a sample trigger outside periodic saving and a step-0 best with `sample_at_first=false`; count complete packages and sample generations and compare the next controlled update against an unsampled run.

**Acceptance Scenarios**:

1. **Given** a sample trigger on a step without another save reason, **when** that step finishes, **then** the requested sample event is retained in a complete package at that step and is generated once.
2. **Given** enabled sampling, `sample_at_first=false`, and a valid step-0 best, **when** the best package is committed, **then** it contains one sample set for its step-0 weights even though no standalone initial sample event was requested.
3. **Given** sampling is disabled, **when** a package is saved, **then** no image generation occurs and an empty `samples/` folder is allowed.

### Edge Cases

- Relative `experiment_dir` and relative paths inside it are resolved from different anchors: the former from the selected `train.toml`, the latter from the resolved experiment root. Changing the working directory must not alter either result.
- A relative `experiment_dir` is supplied without a selected `train.toml`: reject it during preflight; an absolute root needs no path-resolution anchor but still requires the selected `<root>/train.toml` for trainer opt-in.
- Trainer opt-in is requested only from CLI, even with an absolute root, or without enabled `val_dataset_config` and both dataset TOMLs: reject the incomplete experiment before model loading. Cache commands may use an absolute root without `--train_config`.
- An explicit output or logging location conflicts with the experiment hierarchy: report the conflicting setting and correction before model loading rather than silently writing outside the root.
- Periodic, final, new-best, epoch, and sample triggers coincide: there is one package and one sample set for the absolute step.
- A sample-only step is made an explicit full-package save; it must not create an unnamed or partial checkpoint.
- The first valid validation result arrives at step 0 while `sample_at_first` is false: the mandatory best package is still complete, including its samples when sampling is enabled.
- The old best remains intact when a new-best package or its required samples fail to finish.
- A tied best metric, nonfinite value, or partial validation event does not replace the best.
- Retention reaches the inclusive boundary exactly; the best can remain outside the current-state window without a duplicate current package.
- Resume from a current or best package at a different step than the best: the loaded model keeps its own step and metrics, while best comparison retains the stored best value.
- An explicit `save_last_n_steps_state` appears with `experiment_dir`: reject it before model loading with a correction to use only `save_last_n_steps`.
- An explicit `save_last_n_epochs`, `save_last_n_epochs_state`, or `save_state_to_huggingface` appears with `experiment_dir`, or `save_last_n_steps` is negative: reject the setting before model loading with its source and correction.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The new behavior MUST apply only when `experiment_dir` is explicitly set for Qwen-Image `model_version=original` LoRA training or its two Qwen cache commands. Without it, existing configurations, path rules, save outputs, samples, and resume behavior MUST remain unchanged. Other models and future similarity or quality metrics are out of scope.
- **FR-002**: A relative `experiment_dir` MUST resolve from the directory of the trainer-selected `<root>/train.toml`; an absolute root MUST retain its meaning without a relative-path anchor. In either case, trainer opt-in MUST require selection of `<root>/train.toml` through `--config_file`; CLI-only opt-in without that selected file MUST fail in preflight. In experiment mode, relative dataset, cache, configuration, prompt, output, logging, and resume paths, including paths within both dataset configurations, MUST resolve from that root; absolute values MUST remain absolute. Resolution MUST NOT change the process working directory and MUST be independent of the invocation working directory and root folder name.
- **FR-003**: Effective settings MUST follow defaults, then the selected `train.toml`, then explicit command-line values. Experiment-mode training MUST have enabled `val_dataset_config` plus both existing train and validation dataset TOMLs. An assignment conflicting with the required output or logging hierarchy, or a missing required configuration, MUST report its source, cause, and correction before model loading rather than be silently ignored.
- **FR-004**: The experiment layout MUST use `<experiment_dir>/train.toml`, `train-dataset.toml`, `val-dataset.toml`, `sample_prompts.txt`, `dataset/train/`, `dataset/val_familiar/`, `dataset/val_unfamiliar/`, `cache/train/`, `cache/val_familiar/`, `cache/val_unfamiliar/`, `output/tensorboard/`, `output/current_training_states/`, and `output/val_training_states/val-loss/`. Image captions remain `.txt`; existing Qwen latent and text cache formats and names, including `_qi_te.safetensors`, MUST remain usable.
- **FR-005**: The trainer MUST select `<root>/train.toml` through its existing `--config_file`, even if `experiment_dir` is absolute. Both Qwen cache commands MUST accept an explicit `--train_config` selecting that file as the anchor for a relative `experiment_dir`, or an absolute `--experiment_dir` without `--train_config`; their existing `--dataset_config` MUST select either the root's `train-dataset.toml` or `val-dataset.toml`. Each cache command MUST process both role-labelled validation sets from the single `val-dataset.toml` when selected; validation membership MUST NOT be added to training. Cache commands without `experiment_dir` MUST keep their existing invocation behavior.
- **FR-006**: A saved state MUST be one folder named with `output_name` and its absolute completed optimizer step. It MUST contain one adapter file named `model.safetensors`, complete optimizer and scheduler state, rank-local random-state files, trustworthy completed-step and validation-protocol metadata, best/metric attribution, and `samples/` when applicable. Service-file names required for state restoration MUST remain compatible with the existing loader. The frozen base model MUST NOT be saved.
- **FR-007**: A complete package MUST have one physical copy of the LoRA weights and be usable both as an adapter and for exact training-state resume. Its `model.safetensors` MUST preserve the supported Qwen LoRA trainable weights in FP32; an effective lower-precision `save_precision` MUST fail in preflight with a correction to use `fp32` or `float`. Adapter export and resume MUST be verified rather than assumed equivalent. A separate adapter file outside the package, second LoRA file inside it, separate `*-state` duplicate, extra final duplicate, or shared `output/sample` copy is forbidden in experiment mode.
- **FR-008**: `save_every_n_steps` MUST trigger periodic complete packages by absolute completed optimizer step, and one final complete package MUST be saved after the last completed update. Periodic, epoch, final, new-best, and sample reasons on one step MUST coalesce into at most one package and one sample set for that step.
- **FR-009**: After a complete validation event, the minimum finite `val_loss_mean` for `val_unfamiliar` MUST select the only best state in `output/val_training_states/val-loss/`. Improvement MUST be strict; a tie MUST keep the earlier best. The first valid result, including step 0, MUST become best. A new best MUST be saved immediately even outside periodic cadence. No other validation metric selects a best state; failed or nonfinite events MUST never do so.
- **FR-010**: A step package MUST physically reside either under `output/current_training_states/` or under `output/val_training_states/val-loss/`, never both. On a successful best replacement, the former best MUST return to current storage only if it remains within retention; otherwise it MUST be removed as a whole package. This transition MUST NOT regenerate or copy samples or adapter weights.
- **FR-011**: In experiment mode, `save_last_n_steps` MUST be the only retention setting, MUST be a nonnegative integer when set, and MUST apply to each entire package. After successful retention at current step `X`, an owned current package at step `s` MUST remain when `s >= X-save_last_n_steps`, while older owned current packages MUST be removed as complete packages. An unset limit MUST retain all packages. The current best MUST be protected regardless of age. The supplied configuration MUST set `save_last_n_steps=1000`; explicit `save_last_n_steps_state`, `save_last_n_epochs`, `save_last_n_epochs_state`, or `save_state_to_huggingface` with `experiment_dir` MUST be rejected before loading weights with the source and correction to remove the conflicting setting and use only `save_last_n_steps` for retention.
- **FR-012**: When sampling is enabled, each saved complete package MUST contain one PNG sample set generated from that package's weights, including a new best saved between periodic steps. Coincident sample triggers MUST produce one set. A sample-only step MUST be an explicit complete-package save, so honoring the trigger increases the number of full packages by one at that step but never creates a detached sample or hidden partial checkpoint. Sampling MUST restore training modes and random states.
- **FR-013**: `sample_at_first=false` MUST suppress only a standalone initial sample request. If a valid step-0 result creates the mandatory best package while sampling is enabled, that package MUST still receive its one step-0 sample set; this does not create a second package. With sampling disabled, no generation occurs and an empty `samples/` folder is allowed.
- **FR-014**: A package MUST become available for resume or best selection only after all required files and samples finish successfully. Failure MUST leave the previous valid best intact. Retention MUST affect only complete packages owned by this experiment and MUST never remove datasets, caches, configurations, or unrelated files.
- **FR-015**: Every rank MUST participate in saving the random state and other required rank-local training state; main rank MUST coordinate common files. A two-rank package MUST include both rank random-state files, and resume MUST verify that all required rank files are present before accepting a package. Resume from current or best MUST restore model, optimizer, scheduler, random states, and absolute completed step according to the Stage 2 contract.
- **FR-016**: Best selection MUST continue from the stored prior best after resume, rather than restart from an empty best value. Resuming from another step MUST NOT assign old best metrics to the newly loaded weights. Changed Stage 1 validation inputs or protocol MUST continue to be rejected as in Stage 2.
- **FR-017**: The existing Stage 1 fixed validation inputs and Stage 2 loss, scheduling, logging, and isolation contracts MUST remain in force. The new folder and save policy MUST NOT change training mathematics, random state, optimizer updates, or the meaning of an absolute completed step. NN-search/SSCD, duplicates, CSD, and VQAScore MUST NOT be implemented or added as dependencies, settings, or placeholders in this stage.
- **FR-018**: Acceptance MUST include controlled path relocation and legacy compatibility checks; one-package/one-sample coalescing; adapter-file uniqueness; best improvement, tie, failure, and relocation cases; inclusive whole-package retention; a real small training-state save/load from both current and best locations; and a two-rank CPU state round trip confirming both rank random states. Local checks MUST NOT be described as real-model GPU verification.

### Key Entities *(include if feature involves data)*

- **Experiment root**: The user-selected portable directory that anchors all relative experiment paths and owns this mode's output hierarchy.
- **Complete step package**: A uniquely step-labelled folder containing one LoRA adapter, complete resumable training state, attributed metadata, and that step's samples when enabled; its location is current or best.
- **Best record**: The single complete package selected by the lowest valid unfamiliar-set mean validation loss, with the value and step retained across resume.
- **Retention window**: The inclusive completed-step range for owned current packages; the current best is exempt.
- **Sample event**: One requested PNG generation at a completed step, bound to that step's package and weights.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A relocated, renamed experiment resolves 100% of its relative input and output references identically from two different working directories; no experiment output appears at the old location. Existing configurations without `experiment_dir` retain their prior path behavior.
- **SC-002**: At every exercised saved absolute step, exactly one complete package exists across current and best, with exactly one LoRA adapter file and at most one sample set. Coincident save reasons never create a second package or sample generation.
- **SC-003**: For every exercised valid validation result, exactly one best package exists; only a strictly lower unfamiliar-set mean changes it. A tie, nonfinite result, interrupted event, or failed replacement leaves the previous best and its value unchanged.
- **SC-004**: For `X` and `N`, 100% of owned current packages with `s >= X-N` survive whole and 100% with `s < X-N` are removed whole after successful retention, except the one protected best; no unrelated path is removed.
- **SC-005**: Controlled saves from both current and best restore the same adapter weights, optimizer and scheduler states, each participating rank's random state, and absolute step; the next controlled update matches a reference without save/load.
- **SC-006**: Every enabled sample trigger yields exactly one sample set tied to its step's weights. A sample-only step adds one explicit complete package; a step-0 best with `sample_at_first=false` adds no second package and still has its sample set. Controlled training state after sampling matches its unsampled counterpart.
- **SC-007**: All prescribed local acceptance checks pass without model downloads or GPU training; the result is reported as local verification only.

## Assumptions

- The [Stage 1 input contract](../002-val-loss-core/contracts/validation-inputs.md) and [Stage 2 training contract](../003-val-loss-training/contracts/validation-training.md) provide valid role-labelled caches, trustworthy absolute steps, and completed validation results. This feature adds storage and path behavior, not new validation mathematics.
- Experiment mode is explicitly opted into by `experiment_dir`; the user supplies the experiment's data and model references. The documented folder tree is a layout contract, not a request to generate datasets, caches, or prompts automatically.
- `save_last_n_steps=1000` is a value in the supplied example configuration, not a new global default. Omitting it in another experiment retains all complete packages.
- Sampling is enabled only when sample prompts and a sampling trigger are configured; otherwise an empty `samples/` directory is acceptable and no PNG generation occurs.
- The supplied example configuration will use the supported `save_precision=fp32` spelling; lower-precision adapter export is incompatible with an exact, single-file resumable state in this mode.
- Only complete, attributable packages created for the selected experiment are eligible for retention or best moves. Existing unrelated files in the output tree are not owned packages.
- This is a local specification. Remote transfer, GPU training, model downloads, and full-model operational verification are outside this stage.
