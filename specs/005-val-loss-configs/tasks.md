# Tasks: Qwen-Image Validation Example and Run Guide

**Input**: `specs/005-val-loss-configs/spec.md`, `plan.md`, `research.md`, `data-model.md`, `contracts/example-cli.md`, and `quickstart.md`.

**Scope**: Physical example, existing readers/CLI, controlled CPU acceptance, and documentation. Do not change training mathematics, add a framework, load external weights, or run a GPU.

## Phase 1: Setup

- [X] T001 Record the existing Stage 1–3 focused CPU baseline and inventory any pre-existing `qwen_image_lora_val_example/` files in `specs/005-val-loss-configs/validation.md`; preserve user files with matching names (FR-001, FR-013).

## Phase 2: Foundation

- [X] T002 Add one temporary, small-file fixture in `tests/test_qwen_image_val_example.py` that reuses the existing image/cache and tiny-adapter helpers in `tests/test_qwen_image_experiment_paths.py`, `tests/test_qwen_image_validation_training.py`, and `tests/test_qwen_image_experiment_states.py`; provide captioned train, familiar, and unfamiliar sources, valid latent/text caches, and two validation image sizes without external weights (FR-012).

**Checkpoint**: The temporary fixture can exercise the real readers without changing production code.

## Phase 3: User Story 1 — Prepare a portable experiment (P1)

**Goal**: A separate, movable example with four accepted files and empty user-data locations.

**Independent test**: Copy the example into the temporary fixture, replace model paths and `TOK`, and preflight it from another working directory; confirm exactly one unroled train source and two distinct validation roles.

- [X] T003 [US1] Write failing physical-file and reader assertions in `tests/test_qwen_image_val_example.py`: four required files, exact `train.toml` effective values from `spec.md`, integer `lr_warmup_steps=200`, two exact prompt lines, no unsupported keys, no `save_last_n_steps_state`, and no fabricated images/caches (FR-001–FR-004, FR-007, FR-012).
- [X] T004 [P] [US1] Create `qwen_image_lora_val_example/train.toml` with all exact Configuration Values, `experiment_dir="."`, root-relative internal paths, clearly replaceable absolute DiT/VAE/text paths, BF16 computation, and FP32 sole adapter saving (FR-002–FR-004).
- [X] T005 [P] [US1] Create `qwen_image_lora_val_example/train-dataset.toml` and `qwen_image_lora_val_example/val-dataset.toml` with the exact 1024 bucket/caption/batch/repeat settings, one unroled train source, exactly the two validation roles and distinct caches, and no random transforms or validation source in training (FR-005, FR-006).
- [X] T006 [P] [US1] Create the two exact `TOK` lines in `qwen_image_lora_val_example/sample_prompts.txt` and empty `.gitkeep` placeholders in `qwen_image_lora_val_example/dataset/train/`, `dataset/val_familiar/`, `dataset/val_unfamiliar/`, `cache/train/`, `cache/val_familiar/`, `cache/val_unfamiliar/`, and `output/`; do not add example images or cache tensors (FR-001, FR-007).
- [X] T007 [US1] Extend `tests/test_qwen_image_val_example.py` to run the real Qwen train parser, dataset/validation readers, and prompt reader against the copied example: verify effective paths from another CWD, separate source/cache membership, unchanged existing user files, and actionable rejection of unresolved placeholders, missing caption/cache, or conflicting/unknown settings before model loading (FR-001–FR-003, FR-005–FR-007, FR-012).

**Checkpoint**: US1 is usable after the user supplies real data, model paths, and caches; the empty example is never presented as training-ready.

## Phase 4: User Story 2 — Follow one complete run (P1)

**Goal**: Four real cache invocations and a documented train/monitor path backed by one controlled config-to-state CPU chain.

**Independent test**: Real parsers accept all four cache selections; the short CPU chain reads fixed inputs, emits six metrics, saves one complete package per due step, and resumes it.

- [X] T008 [P] [US2] Add cache-command checks in `tests/test_qwen_image_val_example.py` for latent and text caching of each canonical dataset TOML from another CWD: use actual `--train_config`, `--dataset_config`, `--vae`/`--text_encoder` flags; confirm each validation command sees both roles and stops before model loading (FR-008, FR-012).
- [X] T009 [P] [US2] Test the existing validation boundary in `tests/test_qwen_image_validation_training.py` with the example's `val_every_n_steps=200`: prove exactly `0,200,...,1600`, one final event at 1600, and no 1,600-update CPU loop (FR-010, SC-004).
- [X] T010 [US2] Add a short composed CPU test in `tests/test_qwen_image_val_example.py` using the parsed temporary config, real fixed input/noise reader (SHA-256 seed, 10×1 checks, exact unshifted levels), existing Qwen evaluation/common loss, image-first aggregation and exactly six finite metric tags for two differently sized validation sources; keep training `shift/2.2` in the parsed config (FR-003, FR-010, FR-012).
- [X] T011 [US2] Continue T010's parsed-config CPU chain in `tests/test_qwen_image_val_example.py` through one real Accelerate save at a nonzero completed step: the evaluated six metrics appear in one complete package containing one FP32 `model.safetensors`, optimizer/scheduler/rank RNG, and both sidecars; expose that package to T014 for resume. Rely on the existing Stage 3 tests rerun by T017 for step-0 samples, overlapping reasons, strict best, retention, and duplicate-output protection (FR-004, FR-010, FR-012, SC-005).
- [X] T012 [US2] Write the four exact cache commands, `accelerate launch`, TensorBoard command, replacement/data-preparation steps, fixed six tags and due-step grid, BF16 source DiT versus FP32 adapter explanation, sample-disable pair, step-0 best sampling, and later H200-only verification in `docs/qwen_image.md` using `specs/005-val-loss-configs/contracts/example-cli.md` (FR-002, FR-004, FR-007–FR-010, FR-012).

**Checkpoint**: US2 commands and observed CPU outputs agree; no local result is described as a full-model or GPU run.

## Phase 5: User Story 3 — Move and resume (P2)

**Goal**: Root relocation and current/best resume preserve fixed inputs and absolute step meaning.

**Independent test**: Move the prepared temporary root, parse commands from two other CWDs, then resume published current and best packages with a short new update budget.

- [X] T013 [US3] Test root rename and other-CWD invocation in `tests/test_qwen_image_val_example.py`: train/validation TOMLs, image/cache/prompt/output/log paths and local `--resume` rebase within the moved root; validation fingerprint and cache identity remain compatible (FR-008, FR-009, SC-003).
- [X] T014 [US3] Resume T011's saved package with the same parsed temporary config and fixed inputs in `tests/test_qwen_image_val_example.py`: at nonzero `s`, initial validation emits the six metrics once, first new training-loss log is `s+1`, and a short completed-update budget `B` ends at `s+B`. Rerun existing Stage 3 current/best and exact-next-update tests in T017 instead of duplicating them here (FR-010–FR-012).
- [X] T015 [US3] Document exact current/best `--resume` commands, whole-root relocation, retained fixed inputs, `max_train_steps` as this run's completed-update budget, and the lack of exact data-loader-position restoration in `docs/qwen_image.md` and `specs/005-val-loss-configs/quickstart.md` (FR-008, FR-009, FR-011).

**Checkpoint**: Both package trees resume from their saved absolute step after the folder moves.

## Phase 6: Polish and evidence

- [X] T016 Update `README.ru.md` with the separate example and verified `docs/qwen_image.md` entrypoint, required user data/model substitutions, and unchanged legacy usage; check links and avoid claiming CPU tests prove H200 performance (FR-001, FR-009, FR-012, FR-013).
- [X] T017 Run focused example, Stage 1–3 and legacy CPU tests plus lint/diff checks; record exact commands, results, unavailable server checks, and an FR-001–FR-013/SC-001–SC-006 evidence matrix in `specs/005-val-loss-configs/validation.md`, including explicit checks for zero `val_unfamiliar` training images and duplicate state/sample outputs (FR-012, FR-013).

## Dependencies and execution order

- T001 → T002 → T003. T004–T006 may run in parallel after T003, then T007 completes US1.
- US2 requires the physical US1 example. T008 and T009 may run in parallel; T010 precedes T011. T012 can proceed alongside CPU checks after the real CLI flags are confirmed.
- US3 relocation T013 uses the US1 fixture independently; its config-fed resume T014 follows T011's single complete package. T015 follows verified resume/path behavior.
- T016 follows the guide; T017 closes the stage after all three stories.

## Parallel examples

- US1: implement the independent `train.toml` (T004), dataset TOMLs (T005), and prompt/empty-directory layout (T006) concurrently after the failing contract test.
- US2: exercise cache CLI selection (T008) and the scheduling grid (T009) concurrently; both feed the composed acceptance and guide.

## Implementation strategy

Deliver US1 as the MVP physical example and reader gate. Then prove US2's command path and short CPU chain, add US3 relocation/resume, and finish with README and an evidence matrix. Reuse existing Stage 1–3 fixtures and production seams; do not add a new training path.

## Phase 7: Convergence

- [X] T018 LOW: In docs/qwen_image.md, scope the Japanese instruction to run from repository root to legacy commands without experiment_dir, and state that the portable example's absolute REPO/ROOT commands can run from another CWD; preserve the supported command contract per FR-008 and FR-009 (partial).
- [X] T019 MEDIUM: Extend the existing parsed-example CPU test in tests/test_qwen_image_val_example.py to prove its real six-metric event selects exactly one complete best package and its configured save_last_n_steps=1000 retains a complete current package at the inclusive X-1000 boundary while removing an older owned current package. Reuse Stage 3 package/retention seams without a new harness or extra training updates, per FR-004, FR-010, FR-012 and SC-005 (partial evidence).
