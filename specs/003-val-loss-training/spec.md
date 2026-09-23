# Feature Specification: Deterministic Validation Loss During Training

**Feature Branch**: `001-scope-qwen-image-lora` (current branch; feature directory is independent)

**Created**: 2026-09-23

**Status**: Draft

**Input**: Stage 2.1 of `config_for_qwen_image_lora/val_loss_speckit_commands.md`: connect the completed Stage 1 validation input and noise contract to Qwen-Image original LoRA training, with comparable loss metrics, fixed event scheduling, and reliable resume.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compare familiar and unfamiliar validation loss (Priority: P1)

The user enables validation with the two fixed sets from Stage 1 and sees six named loss series. Each image contributes equally to its set, while low and high noise remain separately visible. The values use the same model weights and loss mathematics as training.

**Why this priority**: A graph is useful only if its values represent the requested checks and are comparable to the training objective.

**Independent Test**: Supply known per-check losses for images with different sizes and verify the six published values against an independent image-first calculation. With the current Qwen objective, verify the validation target and loss weighting against the training calculation at the same validation noise level.

**Acceptance Scenarios**:

1. **Given** both nonempty sets and valid Stage 1 checks, **when** a complete validation event finishes, **then** exactly the six specified scalar series are published once at that event's absolute step, with every declared image contributing equally.
2. **Given** different image sizes, buckets, or training repeat counts, **when** set losses are aggregated, **then** each image has one equal share of its set mean and the full mean equals the average of its low and high means within numerical tolerance.
3. **Given** a nonfinite check loss, changed input, or failed read, **when** an event is evaluated, **then** the event fails and publishes none of its six values as valid results.

---

### User Story 2 - Observe validation at predictable steps without disturbing training (Priority: P1)

The user sees validation before the first update, at the configured completed-update interval, and after the run's last update. Validation does not change training weights, gradients, optimizer, scheduler, random state, or the next controlled training update.

**Why this priority**: A correctly computed metric is still misleading if it shifts the training path or appears at the wrong step.

**Independent Test**: In a controlled run, compare training state and the next update with and without an intervening validation event; exercise accumulation, a skipped update, periodic/final overlap, mixed train/eval flags, dropout, and an event exception.

**Acceptance Scenarios**:

1. **Given** a fresh run, **when** training starts, **then** one validation event is recorded at step 0 before the first optimizer update.
2. **Given** completed update count `s`, **when** `s` is divisible by `val_every_n_steps` or is the run's final completed step, **then** exactly one event occurs after that update and before the next training step; intermediate microbatches and skipped updates do not trigger an event.
3. **Given** modules with different pre-event train/eval flags, **when** validation succeeds or raises an error, **then** all those flags and training random states are restored and no validation backward or parameter update occurs.
4. **Given** `val_dataset_config` is absent, **when** the same training configuration runs, **then** its existing behavior and loss graphs remain unchanged.

---

### User Story 3 - Continue a run with a trustworthy absolute timeline (Priority: P2)

The user resumes from a saved training state and sees validation at the saved absolute update count before another update. Training loss continues at the next absolute step, while the run's requested step budget and restored optimizer schedule retain their established meanings.

**Why this priority**: Reset or guessed steps and silently changed validation inputs break comparisons across sessions.

**Independent Test**: Save a controlled state after `s>0` completed updates, resume with a short new budget, and check the initial validation at `s`, first new training loss at `s+1`, subsequent periodic/final events, state restoration, and changed-input rejection.

**Acceptance Scenarios**:

1. **Given** a trustworthy state saved at absolute step `s` with unchanged validation input and protocol, **when** the run resumes, **then** it validates once at `s` before the next update and numbers later results by completed absolute updates.
2. **Given** a resumed run with a budget of `B` completed updates, **when** it completes, **then** its final absolute step is `s+B`; its optimizer and scheduler continue from the restored state rather than restarting warmup.
3. **Given** validation is enabled and an old state lacks a trustworthy completed-update count or its recorded validation identity differs, **when** resume is attempted, **then** it fails with a corrective error rather than guessing a step or accepting changed inputs.
4. **Given** only network weights are loaded without optimizer-state resume, **when** training starts, **then** the run uses a new timeline beginning at step 0.

### Edge Cases

- The periodic event and mandatory final event coincide: publish one event at that step.
- A run resumes at a step that is itself divisible by the interval: perform the resume-start event once in that run, then continue with later absolute boundaries.
- Gradient accumulation has several microbatches or an optimizer update is skipped: neither creates a completed-step validation boundary.
- An event fails after evaluating some images: publish no partial or zero-filled set values, and do not treat it as a best result.
- A set contains images with different resolutions or repeat counts: mean per image before mean per set; never average all pixels or buckets as peers.
- Model and LoRA modules begin with mixed train/eval flags: restore each original flag after success and exception.
- A multi-rank run uses padding or unequal local partitions: each declared occurrence contributes exactly once globally and only one publication is made.
- A resumed state records an absolute step but has a different Stage 1 input fingerprint or protocol: reject the resume before a validation result is published.
- A legacy resume has no trustworthy absolute step: it remains usable when validation is disabled, but cannot silently start enabled validation at zero.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: This feature MUST connect only the existing Qwen-Image `model_version=original` LoRA trainer to the completed [Stage 1 validation input and noise contract](../002-val-loss-core/contracts/validation-inputs.md). It MUST use that contract's roles, membership, frozen identity, grid, seeds, noise mixing, and public controls without redefining them. Output hierarchy, best-state selection, other models, and other validation modes are outside this stage.
- **FR-002**: `val_dataset_config` MUST enable validation. When it is absent, the existing training loss, data, noise and timestep sampling, precision, gradients, optimizer, scheduler, random state, logging, saving, and resume behavior MUST remain unchanged.
- **FR-003**: Each validation event MUST finish both role-labelled sets using the fixed Stage 1 items, one image at a time and exactly `N1*N2` checks per image. Validation MUST use current LoRA weights and MUST NOT construct a second model.
- **FR-004**: A check's scalar loss `L(x,i,j)` MUST use the training model's forward, target, and mathematical loss, including the same weighting and reduction. For the current Qwen flow-matching objective, the target MUST be `epsilon - latent`. An independently simplified loss that can diverge from training is prohibited.
- **FR-005**: For each image `x`, its full loss MUST be the arithmetic mean of its `N1*N2` scalar check losses. A set's full loss MUST be the arithmetic mean of its image losses, giving every image equal weight regardless of resolution, pixel count, bucket, or training repeats. Averaging pixels across the set or giving buckets equal weight is prohibited.
- **FR-006**: The low-noise image mean MUST use only checks with `t<0.5`; the high-noise image mean MUST use only checks with `t>=0.5`. Each set's low and high loss MUST be the arithmetic mean of the corresponding image means. Because the Stage 1 grid has equal-sized halves, `loss_mean = (loss_low_noise + loss_high_noise)/2` MUST hold within numerical tolerance.
- **FR-007**: On a complete event, the existing logging workflow MUST publish exactly these six scalar tags at the event's absolute step, once each: `train_eval_loss_mean`, `train_eval_loss_low_noise`, and `train_eval_loss_high_noise` for `val_familiar`; `val_loss_mean`, `val_loss_low_noise`, and `val_loss_high_noise` for `val_unfamiliar`. Existing training-loss graphs MUST be preserved.
- **FR-008**: Let `X` be the absolute number of completed optimizer updates. A fresh enabled run MUST validate once at `X=0` before any update; after each completed update `s>0` divisible by `val_every_n_steps`; and after the final completed update of the run. A periodic/final coincidence MUST produce one event. The next training step MUST begin only after its boundary validation finishes.
- **FR-009**: Validation MUST NOT run for an intermediate gradient-accumulation microbatch or an attempted update that did not complete. Only a completed optimizer update advances `X` and the validation schedule. An event MUST occur no more than once per absolute step within one run.
- **FR-010**: Validation MUST run without autograd or backward and MUST NOT modify model or LoRA weights, existing gradients, optimizer state, scheduler state, or training random states. It MUST NOT use optimizer evaluation hooks that alter weights or optimizer state.
- **FR-011**: Validation MUST disable model and LoRA dropout while evaluating. On both success and exception, it MUST restore every module's previous train/eval flag individually and restore Python, NumPy, PyTorch CPU, and available CUDA training random states. It MUST NOT indiscriminately set all modules to train mode afterward.
- **FR-012**: Validation MUST use the Stage 1 final level `t` directly as `sigma=t` and `timestep=1000*t`. The loss weighting MUST follow the training formula evaluated at that actual validation sigma. Reapplying a flow shift, selecting a nearby value from the training schedule, or changing the training path's precision or weighting is prohibited.
- **FR-013**: The existing one-rank and two-rank Accelerate workflows MUST evaluate each declared item once globally, without distributed padding changing either set mean. Ranks MUST cross validation boundaries in agreement without a collective deadlock, and each tag MUST be published only once per event.
- **FR-014**: An enabled training state MUST record a trustworthy absolute completed-optimizer-update count and validation-protocol metadata sufficient to check the Stage 1 frozen identity and effective controls at resume. Neither a microbatch count, a single parameter's step count, nor `accelerator.step` alone may be treated as that absolute count without establishing equivalence.
- **FR-015**: Resume from saved absolute step `s` MUST validate once at `s` before the next update; the first new training-loss point MUST be at `s+1`, with no invented training-loss point at `s`. Later periodic boundaries MUST use absolute steps, and a resumed run's final event MUST use its final absolute completed step.
- **FR-016**: Existing optimizer, scheduler, and random-state restoration MUST remain intact. `max_train_steps` MUST remain the budget for completed optimizer updates in the current run, separate from the absolute graph step; learning-rate warmup MUST NOT restart solely because validation is enabled. Exact restoration of the training data-loader position is outside scope.
- **FR-017**: Enabled validation on resume MUST reject changed Stage 1 inputs or validation protocol before accepting new results. If an older state lacks a trustworthy absolute step, enabled validation MUST fail with an explanation and correction; disabled legacy resume MUST continue to work. Loading network weights without optimizer-state resume MUST start a new run at absolute step 0.
- **FR-018**: A nonfinite check or aggregate loss, input-read error, or otherwise incomplete event MUST fail as an event; it MUST NOT publish any of that event's six tags as valid metrics, use zero as a substitute, or feed best-state selection. The error MUST identify the source or check when known and how to correct or restart it.
- **FR-019**: Acceptance MUST include controlled local tests of training loss and weighting equivalence, image-first and low/high aggregation, an unchanged subsequent training update and random state, module modes/dropout/gradients/optimizer/scheduler under success and exception, accumulation and skipped updates, fresh and resumed scheduling, the six tags, and a nonzero absolute step after real state save/load. These tests MUST NOT be described as real-model GPU verification.

### Key Entities *(include if feature involves data)*

- **Validation event**: One complete evaluation of both frozen Stage 1 sets at an absolute completed-update step, with six publishable results only after full success.
- **Check loss**: One scalar `L(x,i,j)` for an image, noise level, and realization, computed with the training objective.
- **Image and set means**: Equal-weight arithmetic means over checks within an image and then over images within a role; low and high subsets follow the same order of averaging.
- **Absolute completed-update count**: The trusted, persistent number of successful optimizer updates since the experiment's original start, distinct from the current run's update budget.
- **Validation-protocol metadata**: The saved identity and effective controls required to establish that a resumed event uses the same Stage 1 inputs and protocol.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every successful event with `F` familiar and `U` unfamiliar images accounts for exactly `(F+U)*N1*N2` checks and publishes exactly six finite scalar values, with no omitted or duplicated image occurrence.
- **SC-002**: In controlled examples with unequal image sizes and deliberately unequal check losses, all six set values match an independent image-first reference within `1e-6` absolute error, and each full mean matches the average of its low and high means within `1e-6`.
- **SC-003**: For a fresh run completing `B>0` updates at interval `E`, observed validation steps are exactly `0`, every positive multiple of `E` through `B`, and `B` if it is not already listed; there are no duplicate events. A resume at `s` starts at `s`, and its next training-loss point is `s+1`.
- **SC-004**: In controlled success and failure cases, 100% of pre-event module modes and training random states are restored, no existing gradient or optimizer/scheduler state changes during validation, and the next controlled training update matches the run without validation.
- **SC-005**: A state saved at `s>0` and resumed with `B` completed updates reaches absolute `s+B` without resetting the restored learning-rate schedule; changed protocol/input and step-unknown enabled resumes are rejected, while disabled legacy resume retains its prior behavior.
- **SC-006**: In one-rank and two-rank verification, every declared occurrence contributes exactly once to each event, six tags appear once at each scheduled step, and all ranks finish each finite validation boundary without a collective deadlock.
- **SC-007**: Every exercised nonfinite loss, read failure, and interrupted event publishes zero valid values for that event. All prescribed controlled local acceptance cases pass; their results are reported as local verification rather than real-model proof.

## Assumptions

- The [Stage 1 contract](../002-val-loss-core/contracts/validation-inputs.md) and its implemented strict preflight, fixed manifest, fingerprint, and deterministic checks are available and are not redesigned in this stage.
- The user has already prepared two nonempty role-labelled validation sets and their compatible caches. Stage 2 neither repairs those inputs nor chooses set membership.
- `max_train_steps` is a positive budget of completed updates for the current run. If a periodic boundary coincides with the run's final completed step, it is a single event.
- Existing training state restoration remains the authority for optimizer, scheduler, and random states; this stage adds only the absolute step and validation-protocol checks needed for truthful metrics.
- Best-state selection, new output structure, and retention belong to the next stage. A failed Stage 2 event cannot be treated as a valid result for any later best-state policy.
- This is a local specification. Training runs, GPU execution, model downloads, and server transfer are outside the authorized local work and are not claimed by this document.
