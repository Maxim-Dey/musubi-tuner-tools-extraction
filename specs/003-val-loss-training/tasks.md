---
description: "Dependency-ordered implementation tasks for deterministic Qwen-Image validation loss"
---

# Tasks: Deterministic Validation Loss During Training

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md), [data-model.md](data-model.md), [validation-training contract](contracts/validation-training.md), and [quickstart.md](quickstart.md).

**Scope**: Qwen-Image `model_version=original` LoRA only. Consume the completed Stage 1 manifest/check contract. Keep the existing trainer, logging backend, and Accelerate state directories. No new output hierarchy, best-state policy, second model, GPU run, or model download.

**Tests**: Required by FR-019. Write the controlled CPU tests in each story phase first and confirm the new cases fail for the expected missing behavior before implementation. Use the existing project-compatible Python 3.12 environment; do not install dependencies as part of these tasks.

**Format**: `[ID] [P?] [Story?] Action in exact file(s) — verifiable result (FR IDs)`. `[P]` is used only for independent edits in different files.

## Phase 1: Setup

**Purpose**: Reuse Stage 1 data and make later tests independent of Qwen weights.

- [X] T001 Add small CPU fixtures for a current-model forward, two-role Stage 1 manifest, controlled optimizer/scheduler, and tracker to `tests/test_qwen_image_validation_training.py`; fixtures must need no GPU, model download, or replacement trainer (FR-001, FR-003, FR-019).

## Phase 2: Foundational

**Purpose**: Keep the disabled path intact and expose the existing preflighted manifest only to the enabled branch. Complete before user-story implementation.

- [X] T002 Add a controlled absent-`val_dataset_config` regression in `tests/test_qwen_image_training_invariants.py` that compares legacy loss, update, RNG, log step, save/resume call path, and absence of validation events against the unchanged branch; confirm the new assertion fails if validation work enters the disabled path (FR-002, FR-019).
- [X] T003 In `src/musubi_tuner/qwen_image_train_network.py` and `src/musubi_tuner/training/trainer_base.py`, pass only the already preflighted Qwen `validation_manifest` into an enabled loop branch; absent config must leave the old loop path and its default calls untouched, and no Stage 1 input/noise contract may be redefined (FR-001, FR-002, FR-003).

**Checkpoint**: A prepared Qwen manifest is available to enabled training; disabled training still follows its old path.

## Phase 3: User Story 1 - Compare familiar and unfamiliar validation loss (Priority: P1) 🎯 MVP

**Goal**: Complete both fixed sets with the training objective, image-first means, and six all-or-nothing scalar tags.

**Independent Test**: Controlled check losses for unequal image buckets and repeated occurrences match an independent image-first reference within `1e-6`; off-schedule `t` uses exact weighting, and a real event file has the six contract tags once at the event step.

### Tests first

- [X] T004 [US1] Add CPU tests in `tests/test_qwen_image_validation_training.py` that compare the existing Qwen forward target `epsilon-latent` and shared loss/reduction for `none`, `sigma_sqrt`, and `cosmap`; an off-schedule `t` must use `sigma=t`, `timestep=1000*t`, without flow re-shift or nearest-schedule rounding, and training's no-override path must remain identical (FR-004, FR-012, FR-019).
- [X] T005 [US1] Add independent-reference CPU tests in `tests/test_qwen_image_validation_training.py` for exactly `N1*N2` checks per occurrence, unequal image sizes/buckets, duplicate declarations, image-first set means, low/high halves, and `mean=(low+high)/2` within `1e-6`; pixel or bucket pooling must fail these tests (FR-003, FR-005, FR-006, FR-019).
- [X] T006 [US1] Add CPU tracker/event-file tests in `tests/test_qwen_image_validation_training.py` for the six exact role-to-tag names at one absolute step and no partial publication after a late read error, nonfinite check, or nonfinite aggregate; existing training tags must remain present (FR-007, FR-018, FR-019).

### Implementation

- [X] T007 [US1] Add an optional exact-sigma input to `compute_loss_weighting_for_sd3` in `src/musubi_tuner/training/timesteps.py`; the validation branch evaluates the existing weighting formulas at actual `t`, while an absent override preserves the current scheduler lookup and training result (FR-004, FR-012).
- [X] T008 [US1] Pass the optional exact sigma through the existing `NetworkTrainer.compute_loss` in `src/musubi_tuner/training/trainer_base.py`; keep the same elementwise MSE, weighting, precision, and final reduction for training and validation, with no separate simplified reducer (FR-004, FR-012).
- [X] T009 [US1] Gate Qwen `call_dit` checkpoint-input `requires_grad` behavior by active gradient mode in `src/musubi_tuner/qwen_image_train_network.py`; validation under no-grad must not create input gradients, while training forward/target behavior is unchanged (FR-004, FR-010).
- [X] T010 [US1] Add the Qwen-only event evaluation in `src/musubi_tuner/qwen_image_train_network.py`: stream one Stage 1 item/check at a time through the existing unwrapped model, require finite scalar and exact counts, use stable image-first full/low/high sums, and retain only six finite candidate values until both roles complete (FR-003–FR-006, FR-018).
- [X] T011 [US1] Publish the complete six-value payload through the existing `accelerator.log` tracker in `src/musubi_tuner/qwen_image_train_network.py` only after success, with the exact role/tag mapping and absolute step; no partial payload, extra writer, best-state selection, or changes to old training-loss tags (FR-007, FR-018).
- [X] T012 [US1] Run the US1 cases in `tests/test_qwen_image_validation_training.py` and record their exact command/results in `specs/003-val-loss-training/validation.md`; the math, off-schedule sigma, six tags, and all-or-nothing failures must pass before US2 integration (FR-003–FR-007, FR-012, FR-018, FR-019).

**Checkpoint**: A single-rank validation event computes and logs six correct, complete scalars without introducing a new loss implementation.

## Phase 4: User Story 2 - Observe validation at predictable steps without disturbing training (Priority: P1)

**Goal**: Place events on completed-update boundaries while preserving the next training update and coordinating one/two ranks.

**Independent Test**: Fresh `B=5,E=2` events are `0,2,4,5`; accumulation/skipped attempts create none; mixed mode/RNG snapshots and the next controlled update match a no-validation control, including after an exception; two CPU ranks leave both success and error boundaries.

### Tests first

- [X] T013 [US2] Add CPU tests in `tests/test_qwen_image_validation_training.py` for mixed transformer/LoRA child modes, dropout disabled only during validation, no autograd/backward, unchanged weights/gradients/optimizer/scheduler and Python/NumPy/torch CPU/available CUDA RNG on success and exception, plus an identical next controlled training update (FR-010, FR-011, FR-019).
- [X] T014 [US2] Add controlled loop tests in `tests/test_qwen_image_validation_training.py` for fresh `B=5,E=2` → `0,2,4,5`, periodic/final deduplication, event-before-next-batch order, accumulation microbatches, skipped optimizer attempts, completed-update budget despite skips, and no invented enabled training-loss point; disabled logging/cadence stays as in T002 (FR-002, FR-008, FR-009, FR-019).
- [X] T015 [US2] Add a two-rank CPU Accelerate boundary test in `tests/test_qwen_image_validation_training.py` that proves each Stage 1 occurrence is evaluated once globally, only main publishes six tags, ranks agree on completed step, and success and injected main-rank failure both release peers without deadlock (FR-003, FR-013, FR-018, FR-019).

### Implementation

- [X] T016 [US2] Wrap the Qwen event in `src/musubi_tuner/qwen_image_train_network.py` with no-grad and per-module transformer/LoRA train/eval snapshots plus Python/NumPy/torch CPU/available CUDA RNG snapshots; restore every value in `finally` on success/error without optimizer eval/train hooks (FR-010, FR-011).
- [X] T017 [US2] Integrate enabled-only scheduling and training-log steps into the existing `_run_training_loop` in `src/musubi_tuner/training/trainer_base.py`: start event at 0, advance run/absolute counts only on synced non-skipped updates, run positive periodic/final events once after update before the next batch and sample/save hooks, continue to the current-run completed-update budget, and emit existing training tags only at completed absolute steps; keep disabled behavior unchanged (FR-002, FR-008, FR-009, FR-015, FR-016).
- [X] T018 [US2] Coordinate the enabled boundary in `src/musubi_tuner/training/trainer_base.py`: all ranks agree on completion/absolute step, main evaluates the full manifest through the unwrapped existing transformer, and a success/error status reaches peers before exit; no sharding/padding, duplicate model, duplicate log, or collective after an unbroadcast exception (FR-003, FR-013, FR-018).
- [X] T019 [US2] Run US2 and T002 cases in `tests/test_qwen_image_validation_training.py` and `tests/test_qwen_image_training_invariants.py`, recording exact commands/results in `specs/003-val-loss-training/validation.md`; require scheduling, skipped-update, state-isolation, next-update, disabled-path, and two-rank success/error checks to pass (FR-002, FR-008–FR-011, FR-013, FR-019).

**Checkpoint**: Events occur only at true optimizer boundaries and cannot alter the subsequent training path.

## Phase 5: User Story 3 - Continue a run with a trustworthy absolute timeline (Priority: P2)

**Goal**: Persist and recover the completed-update axis and frozen validation identity with real all-rank Accelerate state saves.

**Independent Test**: A real small save/load at `s>0` restores each rank's RNG, optimizer/scheduler and sidecar; resume `s=5,B=3,E=2` validates at `5,6,8` with first new train-loss at `6`, while missing/mismatched metadata fails and disabled legacy resume still works.

### Tests first

- [X] T020 [US3] Add real one-rank and two-rank CPU Accelerate save/load tests in `tests/test_qwen_image_validation_resume.py` for existing state files plus `val_loss_state.json` with version `qwen-image-val-loss-state-v1`, `model_version=original`, nonnegative exact integer `absolute_completed_step` (reject bool), 64-hex Stage 1 fingerprint and four exact integer controls; require `random_states_0.pkl` and `random_states_1.pkl` with restored rank-local RNG on two ranks (FR-014, FR-016, FR-017, FR-019).
- [X] T021 [US3] Add CPU resume tests in `tests/test_qwen_image_validation_resume.py` for `s=5,B=3,E=2` events `5,6,8`, first new train-loss at `6`, final count `8`, unchanged restored optimizer/scheduler and warmup, changed fingerprint/control, missing/malformed/unknown-step sidecar, disabled legacy resume, and network-weights-only new run at `0` (FR-014–FR-017, FR-019).

### Implementation

- [X] T022 [US3] Extend the existing step/epoch/final state helpers in `src/musubi_tuner/utils/train_utils.py` for enabled saves: every rank calls `accelerator.save_state` for the same existing directory, then main atomically writes `val_loss_state.json` with the contract fields only after all ranks finish; main alone uploads/prunes, broadcasts post-save failure, and the disabled main-only path remains unchanged (FR-002, FR-013, FR-014, FR-016).
- [X] T023 [US3] In `src/musubi_tuner/training/trainer_base.py`, validate the sidecar from the actual local/downloaded directory used by `accelerator.load_state`, agree on it across ranks, reject missing/invalid model/count/fingerprint/controls with corrective errors, and leave disabled legacy resume and network-weights-only loading on their previous paths (FR-014, FR-016, FR-017).
- [X] T024 [US3] Initialize enabled `resume_start_step=s` from validated metadata in `src/musubi_tuner/training/trainer_base.py`; run the start event before the first resumed batch, keep `max_train_steps` as this run's update budget, label enabled logs and step-named save/sample/retention with absolute `s+r`, invoke enabled step/epoch/final state helpers on all ranks, and preserve the restored scheduler/warmup and run-local training-hook step (FR-008, FR-013–FR-017).
- [X] T025 [US3] Run the US3 real one/two-rank round-trip and resume cases in `tests/test_qwen_image_validation_resume.py`, recording exact commands/results and inspected sidecar/rank-RNG names in `specs/003-val-loss-training/validation.md`; all rejection and compatibility cases must pass (FR-013–FR-017, FR-019).

**Checkpoint**: Resumed metrics and saves use a trusted absolute axis without changing the current run's optimizer-update budget.

## Phase 6: Polish and Cross-Cutting Checks

- [X] T026 [P] Review and update `README.ru.md` with the enabled six tags, image-first meaning, 0/periodic/final schedule, absolute resume step, and unchanged disabled behavior; do not document Stage 3 output/best states as implemented (FR-001, FR-002, FR-007, FR-008, FR-015, FR-017).
- [X] T027 [P] Review and update `docs/qwen_image.md` with the same Qwen-only validation and resume contract, including the requirement for a trusted state sidecar and Stage 1 inputs; keep new output hierarchy/best-state behavior for the next stage (FR-001, FR-007, FR-014, FR-017).
- [X] T028 Run the complete local CPU acceptance from `specs/003-val-loss-training/quickstart.md` plus affected `tests/test_qwen_image_config.py` and `tests/test_qwen_image_dataset_cache.py`; record passed/failed/not-run commands and the no-GPU/model limitation in `specs/003-val-loss-training/validation.md`, and verify `src/musubi_tuner/training/trainer_base.py`, `src/musubi_tuner/qwen_image_train_network.py`, and `src/musubi_tuner/utils/train_utils.py` add neither Stage 3 output hierarchy nor best-state selection (FR-001, FR-002, FR-019).

## Dependencies and Execution Order

- T001 → T002 → T003; foundational completion blocks US1–US3.
- US1: T004–T006 are test-first; T007 → T008 → T009 → T010 → T011 → T012. Complete US1 before enabling its event in the loop.
- US2: T013–T015 are test-first; T016 → T017 → T018 → T019. US2 depends on US1's finite six-value event.
- US3: T020–T021 are test-first; T022 → T023 → T024 → T025. US3 depends on the US2 completed-update axis and coordinated boundaries.
- Polish: T026 and T027 may run in parallel after US3; T028 follows them and all story checks.
- Within each story, author the listed tests before source edits and observe expected failures; mark a task complete only after its stated result is verified. Controlled CPU checks are not real-model or GPU verification.

## Parallel Opportunities

- T026 (`README.ru.md`) and T027 (`docs/qwen_image.md`) can run in parallel after US3 because they edit different files and depend on the same completed contract.
- US1, US2, and US3 source/test tasks are intentionally serial: they share the existing trainer and one focused test module, and later stories consume earlier story behavior. There is no safe same-story parallel code edit to mark `[P]`.

## Implementation Strategy

1. Establish the CPU fixtures and disabled baseline, then reuse the Stage 1 manifest.
2. Deliver US1 as the smallest meaningful increment: a complete single-rank, six-value validation event with independent arithmetic tests.
3. Add US2 event placement, state isolation, and one/two-rank boundary agreement without changing the disabled loop.
4. Add US3 versioned state metadata and real save/load resume checks.
5. Review documentation and run the prescribed local regressions; report actual coverage and preserve Stage 3 work for its separate feature.

## Phase 7: Convergence

- [X] T029 In `src/musubi_tuner/training/trainer_base.py`, include the resulting absolute completed step `X=resume_start_step+global_step+int(completed_update)` in the enabled all-rank exchange at every inner-loop boundary, including accumulation and skipped attempts; fail on any rank disagreement before a conditional event, and add a two-rank CPU mismatch regression in `tests/test_qwen_image_validation_training.py` per FR-013 and `plan.md` one-rank/two-rank coordination (partial).
