---
description: "Dependency-ordered implementation tasks for portable Qwen-Image experiments and complete training states"
---

# Tasks: Portable Experiment and Complete Training States

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md), [data-model.md](data-model.md), [CLI contract](contracts/experiment-cli.md), [state contract](contracts/experiment-states.md), and [quickstart.md](quickstart.md).

**Scope**: Explicit `experiment_dir` mode for Qwen-Image `model_version=original` LoRA training and its two cache commands. Reuse Stage 1 input identity and Stage 2 validation/absolute-step state. Do not add another trainer, a second LoRA export, a database, future similarity/quality metrics, GPU runs, or model downloads.

**Tests**: Required by FR-018. In each story phase, write the focused CPU tests first and observe an expected failure before changing production code. Use the existing Python 3.12 environment; no dependency installation is part of these tasks. A green test of mocked save calls does not substitute for real one- and two-rank Accelerate round trips.

**Format**: `[ID] [P?] [Story?] Action in exact file(s); observable result (FR IDs)`. `[P]` appears only for edits to different files with no unfinished dependency between them.

## Phase 1: Setup

**Purpose**: Build small fixtures for path and state tests without Qwen weights.

- [X] T001 Create temporary experiment-root builders with `train.toml`, unroled `train-dataset.toml`, two-role `val-dataset.toml`, PNG/.txt sources, JSONL source, and existing Qwen cache pairs in `tests/test_qwen_image_experiment_paths.py`; no fixture downloads or model loads (FR-002, FR-004, FR-005, FR-018).
- [X] T002 Create a small LoRA-compatible FP32 network, real CPU optimizer/scheduler, rank-local RNG snapshots, and per-step file inspection helpers in `tests/test_qwen_image_experiment_states.py`; keep these fixtures independent of the full Qwen model (FR-006, FR-007, FR-015, FR-018).

## Phase 2: Foundational Legacy Gate

**Purpose**: Record behavior that an absent `experiment_dir` must preserve before opt-in changes begin.

- [X] T003 Add or select absent-`experiment_dir` assertions in `tests/test_qwen_image_training_invariants.py`, `tests/test_qwen_image_config.py`, and `tests/test_qwen_image_dataset_cache.py` for CWD-relative paths, separate legacy adapter/state outputs, sampling destination, and legacy resume; record the baseline command/result in `specs/004-experiment-training-states/validation.md` (FR-001, FR-017, FR-018).

**Checkpoint**: Temporary fixtures and a measured legacy baseline exist; new behavior remains opt-in.

## Phase 3: User Story 1 - Move and run one experiment folder (Priority: P1) 🎯 MVP

**Goal**: Resolve all effective training, validation, cache, model, prompt, log, output, and resume paths from a portable root while retaining legacy path rules when disabled.

**Independent Test**: Rename/move one temporary experiment, invoke trainer and both cache parser/preflight paths from another CWD, and obtain the same Stage 1 fingerprint and correct root-relative files without loading weights.

### Tests first

- [X] T004 [US1] Add red trainer path tests in `tests/test_qwen_image_experiment_paths.py` for defaults → selected train TOML → explicit CLI, relative and absolute roots with the selected `<root>/train.toml`, rejection without `--config_file` or enabled two-role validation, root-relative and absolute model/dataset/prompt/log/local-resume paths, no process `chdir`, conflicting output/logging paths, invalid `output_name`, lower-precision `save_precision`, negative `save_last_n_steps`, and explicit `save_last_n_steps_state`/`save_last_n_epochs`/`save_last_n_epochs_state`/`save_state_to_huggingface` rejected before DiT load with source/correction (FR-001–FR-004, FR-007, FR-011, FR-018).
- [X] T005 [US1] Add red dataset/cache path tests in `tests/test_qwen_image_experiment_paths.py` for root-relative `image_directory`, `image_jsonl_file`, JSONL `image_path`, and `cache_directory` in both dataset TOMLs; verify relocated Stage 1 fingerprint and frozen-item recheck, both roles from one val TOML, cache-source errors before VAE/text load, and unchanged legacy CWD semantics (FR-001, FR-002, FR-004, FR-005, FR-017, FR-018).

### Implementation

- [X] T006 [US1] Preserve the selected train TOML's resolved absolute path and the defaults → TOML → explicit CLI merge in `src/musubi_tuner/training/parser_common.py`; a relative `experiment_dir` must have that anchor, while the disabled parser keeps its current CWD semantics (FR-001–FR-003).
- [X] T007 [US1] Add Qwen-only `experiment_dir` parsing and effective preflight in `src/musubi_tuner/qwen_image_train_network.py`: require selected `<root>/train.toml` and enabled train/two-role val dataset configs, resolve the path-valued settings in `contracts/experiment-cli.md`, require `<root>/output` and `<root>/output/tensorboard`, reject invalid basename, `fp16`/`bf16` save precision, negative `save_last_n_steps`, and explicit conflicting legacy save/retention/upload controls before weights (FR-001–FR-004, FR-007, FR-011).
- [X] T008 [US1] Resolve the two dataset TOMLs' relative image/JSONL/cache paths against the opt-in root in `src/musubi_tuner/dataset/config_utils.py` and, only if needed for JSONL record paths, `src/musubi_tuner/dataset/image_video_dataset.py`; use the same root on Stage 1 frozen-item recheck in `src/musubi_tuner/training/validation_inputs.py` so moving an unchanged root preserves identity and the legacy reader is untouched (FR-001, FR-002, FR-004, FR-005, FR-017).
- [X] T009 [P] [US1] Add `--train_config` and `--experiment_dir` to `src/musubi_tuner/qwen_image_cache_latents.py`; keep `--dataset_config` as the selected train or two-role val TOML, resolve relative `--vae` from the effective root, and run role/source preflight before model loading without changing no-root behavior (FR-001, FR-002, FR-005).
- [X] T010 [P] [US1] Add the same cache-root selection contract to `src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py`; resolve `--text_encoder`, accept both val roles in one file, and reject source errors before loading without changing no-root behavior (FR-001, FR-002, FR-005).
- [X] T011 [US1] Run `tests/test_qwen_image_experiment_paths.py` and affected config/cache regressions, record exact commands, results, resolved-path examples, and Stage 1 fingerprint comparison in `specs/004-experiment-training-states/validation.md`; all new path cases and the T003 legacy gate must pass (FR-001–FR-005, FR-007, FR-011, FR-017, FR-018).

**Checkpoint**: The portable folder is usable by trainer and both cache commands without altering legacy invocations.

## Phase 4: User Story 2 - Keep one complete state per saved step (Priority: P1)

**Goal**: Save one exact FP32 adapter inside a complete, verifiable, resumable package and restore it through the existing Accelerate workflow on one or two CPU ranks.

**Independent Test**: A small real Accelerate package saves and loads from both current and best locations; its only model file loads through the project's LoRA adapter loader and restores the exact next controlled update.

### Tests first

- [X] T012 [US2] Add red real one-rank CPU tests in `tests/test_qwen_image_experiment_states.py` for one unwrapped FP32 `model.safetensors` usable by both the Qwen LoRA adapter loader and Accelerate resume; compare parameter tensors, optimizer/scheduler/RNG, absolute step, and the next update, with no frozen DiT or standalone second adapter (FR-006, FR-007, FR-015, FR-018).
- [X] T013 [US2] Add red finite-time two-rank CPU DDP save/load tests in `tests/test_qwen_image_experiment_states.py` that inspect `random_states_0.pkl` and `random_states_1.pkl`, verify both ranks' restored RNG values and canonical adapter tensor keys without a `module.` prefix, and reject a missing/corrupt rank file before load (FR-006, FR-007, FR-015, FR-018).
- [X] T014 [US2] Add red package-integrity tests in `tests/test_qwen_image_experiment_states.py` with sampling disabled for exact `<output_name>-step-<X>` naming, required Accelerate files and matching Stage 2/experiment sidecars, `metrics_at_step=null` without an event, incomplete state/save failure staying in staging, no published partial package, and current/best location loadability; sample failures are covered in US4 (FR-006–FR-008, FR-014–FR-016, FR-018).

### Implementation

- [X] T015 [US2] Change the opt-in Accelerate save/load hooks in `src/musubi_tuner/training/trainer_base.py` to retain only the unwrapped LoRA `state_dict` in Accelerate's single FP32 `model.safetensors` slot, excluding frozen DiT and DDP `module.` keys; keep legacy hooks/save calls on the no-root branch (FR-001, FR-006, FR-007, FR-015, FR-017).
- [X] T016 [US2] Implement the narrow complete-package staging, exact name/ownership checks, two matching versioned sidecars, required-file and rank-RNG verification, and single current-location publication in `src/musubi_tuner/training/experiment_states.py`; incomplete staging must not become resumable, best, or retention eligible (FR-006–FR-008, FR-014, FR-015).
- [X] T017 [US2] Integrate one all-rank Accelerate save per package and main-only common-file verification/publication in the opt-in branch of `src/musubi_tuner/training/trainer_base.py`; broadcast failures before peers continue, bypass legacy standalone adapter and `*-state` saves, and retain Stage 2's exact `val_loss_state.json` schema (FR-001, FR-006–FR-008, FR-014, FR-015, FR-017).
- [X] T018 [US2] Preflight the actual selected current/best package and every required rank RNG file/key, then load its canonical adapter and full optimizer/scheduler/RNG/absolute-step state on every rank in `src/musubi_tuner/training/trainer_base.py` and `src/musubi_tuner/training/experiment_states.py`; reject incomplete or mismatched ownership/protocol before `accelerator.load_state` (FR-006, FR-007, FR-014–FR-016).
- [X] T019 [US2] Run the real one/two-rank tests with sampling disabled in `tests/test_qwen_image_experiment_states.py`, inspect the only adapter file and both rank RNG files, and record exact commands/results plus current/best round-trip evidence in `specs/004-experiment-training-states/validation.md`; mocked saves alone do not satisfy this gate, while sample integrity closes in US4 (FR-006, FR-007, FR-014, FR-015, FR-018).

**Checkpoint**: A single complete FP32 state package restores a controlled training state and adapter without duplicate weight files.

## Phase 5: User Story 3 - Preserve best validation state and whole-package retention (Priority: P1)

**Goal**: Select one best by strict finite `val_loss_mean`, move whole packages between best/current, and prune only owned current packages outside one inclusive step window.

**Independent Test**: Controlled metrics and temporary folders prove strict improvement/tie/failure, a protected older best, inclusive `X=12,N=8` retention (`s=4` kept, `s=3` removed), and an older-current resume that retains a later best score.

### Tests first

- [X] T020 [US3] Add red tests in `tests/test_qwen_image_experiment_states.py` for first valid step-0 best, strict lower unfamiliar `val_loss_mean`, tie, nonfinite/partial event, successful current↔best directory moves without file copies, and candidate write/move failure preserving the previous best; a two-rank CPU new-best boundary must receive one identical decision or error without deadlock (FR-009, FR-010, FR-014, FR-015, FR-018).
- [X] T021 [US3] Add red whole-package retention tests in `tests/test_qwen_image_experiment_states.py` for unset limit, inclusive `X=12,N=8` boundary, protected best outside the window, complete former-best move/delete, and preservation of foreign files, incomplete folders, symlinks, datasets and caches (FR-010, FR-011, FR-014, FR-018).
- [X] T022 [US3] Add red resume and attribution tests in `tests/test_qwen_image_experiment_states.py` for an older current package with a later best, comparison against the later best score, loaded package's own metrics/step, no overwrite of an already published conflicting step, and changed Stage 1 fingerprint/control rejection (FR-009, FR-010, FR-014–FR-016, FR-018).

### Implementation

- [X] T023 [US3] In `src/musubi_tuner/training/experiment_states.py`, select best only from complete finite `val_loss_mean`, retain the earlier best on equality, fully stage the candidate before a reversible old-best move, publish at most one best and restore the previous one on handled failure; record only metrics measured at each package's own step (FR-009, FR-010, FR-014, FR-016).
- [X] T024 [US3] In `src/musubi_tuner/training/experiment_states.py`, prune only owned complete current packages with `s < X-save_last_n_steps` as whole directories, keep equality/unset-limit/best, prevent symlink or foreign-path deletion, and reject same-step published collisions (FR-010, FR-011, FR-014).
- [X] T025 [US3] Return the completed six-value event payload through the existing Qwen validation boundary in `src/musubi_tuner/qwen_image_train_network.py` and `src/musubi_tuner/training/trainer_base.py`; let main compare it once with the independently discovered best, including after older-current resume, and broadcast the finite payload/new-best decision or error to all ranks before collective save without changing Stage 2 math/logging or the no-root loop (FR-001, FR-009, FR-015–FR-017).
- [X] T026 [US3] Run the best, failure, resume-attribution and retention cases in `tests/test_qwen_image_experiment_states.py`; record exact commands/results and a file inventory showing one best, one location per step, and no unrelated deletions in `specs/004-experiment-training-states/validation.md` (FR-009–FR-011, FR-014, FR-016, FR-018).

**Checkpoint**: One trustworthy best survives failures and retention; current packages remain only within the requested completed-step window.

## Phase 6: User Story 4 - Keep requested samples with their weights (Priority: P2)

**Goal**: Put each requested PNG set inside the one package for its current weights and coalesce every save/sample reason by absolute completed step.

**Independent Test**: A sample-only step adds one explicit complete package; a step-0 best with `sample_at_first=false` has samples but no duplicate package; overlapping periodic/epoch/final/new-best/sample reasons perform one save and one sample generation without changing the next update.

### Tests first

- [X] T027 [US4] Add red controlled call-count and filesystem tests in `tests/test_qwen_image_experiment_states.py` for sample-only, step-0 best with `sample_at_first=false`, disabled sampling, and overlapping periodic/epoch/final/new-best/sample triggers; require one package, one complete PNG set and one save/generation at each absolute step, reject a missing PNG or `image is None`, and leave no `output/sample` or separate final/adapter/state artifacts (FR-007–FR-009, FR-012–FR-014, FR-018).
- [X] T028 [US4] Add red controlled sample-isolation tests in `tests/test_qwen_image_experiment_states.py` for mixed module modes and Python/NumPy/torch CPU/all available CUDA RNG restored after `prepare_sampling`, successful sampling, and injected failure in on-before/inference/after hooks; verify VAE placement is restored after error, optimizer/scheduler/gradients and the next controlled update match an unsampled reference, and failure publishes no package or best replacement (FR-012–FR-014, FR-017, FR-018).

### Implementation

- [X] T029 [US4] Add a narrow opt-in `sample_images(save_dir, force)` seam in `src/musubi_tuner/training/trainer_base.py` so package PNGs are written once to staging `samples/` even on best/sample-only steps, bypass the old shared path, and fail rather than log-and-return on a missing image; snapshot modes and Python/NumPy/torch CPU/all available CUDA RNG before opt-in `prepare_sampling` and before sample hooks/swap, restoring them and VAE placement in `finally` on success/error while leaving no-root sampling untouched (FR-001, FR-007, FR-012–FR-014, FR-017).
- [X] T030 [US4] In the opt-in branch of `src/musubi_tuner/training/trainer_base.py`, combine initial/completed/epoch/final validation, periodic save, strict new best and sample triggers into one absolute-step package decision; broadcast identical save/sample reasons and errors from main to all ranks before all-rank sampling or state save, then order validation → reasons → sample → all-rank state → verify → publish → retention, reuse a compatible already published resume-step package, and never regenerate or resave at a coincident step (FR-008–FR-014, FR-015–FR-017).
- [X] T031 [US4] Run the sample/coalescing/isolation tests in `tests/test_qwen_image_experiment_states.py` and affected `tests/test_qwen_image_validation_training.py`; record exact commands/results plus one-package/one-PNG inventory in `specs/004-experiment-training-states/validation.md` (FR-007–FR-014, FR-017, FR-018).

**Checkpoint**: Every enabled sample is tied to one complete state of the same weights, including sample-only and step-0 best cases.

## Phase 7: Polish and Cross-Cutting Checks

- [X] T032 [P] Review and update `README.ru.md` with the real trainer/cache options, root-relative path rules, complete FP32 package, sample-only and step-0 behavior, strict best, unified retention, current/best resume, and legacy opt-out; include an experiment-mode example with `save_precision=fp32` and `save_last_n_steps=1000` without altering the existing legacy config (FR-001–FR-018).
- [X] T033 [P] Review and update `docs/qwen_image.md` with the same verified opt-in CLI/path/state contract and cache invocation examples; distinguish the example's `save_last_n_steps=1000` from an omitted unlimited limit and explain two-rank RNG state (FR-001–FR-018).
- [X] T034 Run `specs/004-experiment-training-states/quickstart.md` focused CPU tests and affected existing `tests/test_qwen_image_validation_resume.py`, `tests/test_qwen_image_training_invariants.py`, `tests/test_qwen_image_config.py`, and `tests/test_qwen_image_dataset_cache.py`; inspect help for all three entrypoints and document exact PASS/FAIL/NOT RUN, baseline differences, README review and the no-GPU/model limit in `specs/004-experiment-training-states/validation.md` (FR-001, FR-017, FR-018).

## Dependencies and Execution Order

- T001–T002 provide fixtures; T003 establishes the legacy gate before opt-in edits. No production task starts before its story's red tests.
- US1: T004–T005 tests → T006 → T007 → T008 → T009/T010 → T011. The two cache entrypoints may be edited in parallel only after the shared root/dataset resolver is stable.
- US2: T012–T014 tests → T015 → T016 → T017 → T018 → T019. US2 depends on US1 effective paths; the real one/two-rank round trips are required before calling this package layer complete.
- US3: T020–T022 tests → T023 → T024 → T025 → T026. US3 depends on US2 complete package publication and resume.
- US4: T027–T028 tests → T029 → T030 → T031. US4 depends on the US2 package writer and US3 best decision; it closes the sample-enabled parts of US2 and US3 acceptance.
- Polish: T032/T033 may run in parallel after US4; T034 follows all stories and documentation.
- Mark tests complete only after the expected red phase and later green verification. CPU evidence is not a real-model or GPU claim.

## Parallel Opportunities

- T009 and T010 edit different Qwen cache entrypoints after T008 and can be implemented in parallel.
- T032 and T033 edit separate documentation files after the code and tests establish actual behavior.
- Other production tasks share the parser, dataset reader, trainer loop, or package module and are intentionally serial. The two-rank test must have a finite process timeout and cleanup; do not run another distributed test concurrently against the same port.

## Implementation Strategy

1. Establish portable CPU fixtures and an unchanged legacy baseline.
2. Complete US1 as the smallest independently useful increment: consistent trainer/cache paths and Stage 1 identity after moving the root.
3. Complete US2 with one exact FP32 adapter and real one/two-rank save/load before adding best policy.
4. Complete US3 strict best and whole-package retention, then US4 sample events and coalesced package decisions.
5. Review user docs and run the prescribed local regression set. Record unavailable model/GPU checks once, without treating them as passed.

## Phase 8: Convergence

- [X] T035 HIGH: In the opt-in loop, ensure validation, PNG generation, and the single adapter package use the same LoRA weights when an optimizer's eval/train callbacks alter parameters; keep the legacy path and optimizer updates unchanged, add a controlled CPU regression, and verify best attribution per FR-009, FR-012, SC-006, and Constitution IV (partial).
- [X] T036 HIGH: Reject deserializable but unusable rank RNG payloads before `accelerator.load_state`, including invalid Python and NumPy states; prove a malformed payload cannot produce an apparently successful resume and preserve exact two-rank restoration per FR-014 and FR-015 (partial).
- [X] T037 HIGH: Require the selected `<root>/train-dataset.toml` and `<root>/val-dataset.toml` in trainer experiment mode and require cache `--train_config` to select `<root>/train.toml` with `--dataset_config` choosing one of those two files; diagnose conflicting absolute or relative selections before weights while legacy stays unchanged per FR-004 and FR-005 (partial).

## Phase 9: Convergence

- [X] T038 HIGH: Reject a CUDA rank RNG payload whose torch_cuda_manual_seed is empty, has a mismatched visible-device count, or contains malformed states before accelerator.load_state; add a CPU test with a simulated CUDA accelerator so a CUDA resume cannot silently skip RNG restoration, per FR-014 and FR-015 (partial).
