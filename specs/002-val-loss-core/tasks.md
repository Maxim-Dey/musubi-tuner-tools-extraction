---
description: "Task list for deterministic Qwen-Image validation inputs and noise"
---

# Tasks: Deterministic Validation Inputs and Noise

**Input**: Design documents in `specs/002-val-loss-core/`: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/validation-inputs.md](contracts/validation-inputs.md), and [quickstart.md](quickstart.md).

**Tests**: Substantive CPU tests are required by FR-013. Use real temporary images and safetensors, independent expected values, and existing pytest coverage. Do not run training, download weights, or claim real-model verification.

**Organization**: Tasks follow the three spec user stories. A task is complete only when its stated behavior and corresponding local check are demonstrated; a placeholder or file-presence check is insufficient.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: May proceed in parallel with other marked tasks once their shared prerequisites are complete, because files differ.
- **[US1] / [US2] / [US3]**: Spec user story served by the task.
- Every task names exact files, requirement IDs, and a verifiable result.

## Phase 1: Setup

**Purpose**: Prepare small, realistic CPU fixtures in the existing test layout; do not add a test framework.

- [X] T001 Add reusable temporary RGB image, UTF-8 caption, and Qwen `_qi.safetensors`/`_qi_te.safetensors` fixture builders in `tests/test_qwen_image_validation_inputs.py` (FR-004, FR-013); fixture output must represent both roles without model weights and preserve existing cache tensor names and shapes.

---

## Phase 2: Foundational Configuration

**Purpose**: Make the role-bearing dataset and effective training options expressible before implementing any story.

- [X] T002 [P] Extend the strict dataset declaration and blueprint validation in `src/musubi_tuner/dataset/config_utils.py` for optional dataset-level `role`, limited to exactly one nonempty `val_familiar` and one nonempty `val_unfamiliar` when roles are used; reject role in `[general]`, missing/duplicate/unknown roles, and effective `batch_size` or `num_repeats` other than 1, while keeping unroled training declarations valid (FR-002, FR-003, FR-011).
- [X] T003 [P] Register the five public controls only in Qwen's parser in `src/musubi_tuner/qwen_image_train_network.py`: absent `val_dataset_config`, `val_every_n_steps=200` as exact integer `>=1`, `val_seed_noise=42` as exact signed integer, `val_level_noise_n=10` as exact even integer `>=2`, and `val_seed_noise_n=1` as exact integer `>=1`; use the existing effective-value validation in `src/musubi_tuner/training/parser_common.py` after defaults → TOML → explicit CLI, add only Qwen-specific range/evenness checks, reject booleans/unknown keys, and leave disabled runs without validation resources (FR-001, FR-006, FR-011).
- [X] T004 Add effective-configuration CPU cases in `tests/test_qwen_image_config.py` for default/TOML/CLI precedence, all invalid types and boundaries, unknown keys, and absent `val_dataset_config`; assert errors precede model loading and the disabled configuration preserves legacy arguments (FR-006, FR-011, FR-013). Depends on T002–T003.

**Checkpoint**: Both configuration forms are accepted only under their declared rules; legacy unroled datasets and disabled training remain valid.

---

## Phase 3: User Story 1 — Two Fixed Validation Sets (Priority: P1) MVP

**Goal**: Prepare every declared captioned source once under its explicit role, verify exact source-to-cache associations, and fail before weight loading on incomplete or ambiguous input.

**Independent Test**: CPU fixtures with both roles produce one record per declared occurrence; missing captions/caches, stale or corrupt caches, duplicate roles, and name collisions fail with source-labelled errors before any model loader is called.

### Tests for User Story 1

- [X] T005 [P] [US1] Add cache-command contract cases to `tests/test_qwen_image_dataset_cache.py` for a role-bearing `val-dataset.toml`, raw directory images without captions, empty sources, source/cache filename collisions, and preserved legacy unroled caching; patch model loaders so valid preflight stops before weights (FR-002–FR-005, FR-011, FR-013). Depends on T001–T004.
- [X] T006 [P] [US1] Add strict-manifest CPU cases to `tests/test_qwen_image_validation_inputs.py` using real temporary image/caption/safetensors pairs; assert one record per declared occurrence, content SHA-256 `image_id`, equal IDs across roles, exact latent/text associations and naming, and rejection of missing/corrupt/stale/mismatched caches without skipping (FR-002–FR-005, FR-011, FR-013). Depends on T001–T004.

### Implementation for User Story 1

- [X] T007 [US1] Extend role-aware source preflight in `src/musubi_tuner/dataset/config_utils.py` using the existing readers in `src/musubi_tuner/dataset/datasources.py` and bucket selection in `src/musubi_tuner/dataset/bucket.py`; enumerate raw directory images before caption filtering, validate JSONL captions and readable images, and reject empty sets or missing captions with source, cause, and correction (FR-002–FR-004, FR-011). Depends on T002 and T005.
- [X] T008 [US1] Check each declared source's exact cache paths in `src/musubi_tuner/dataset/image_video_dataset.py` and the role-aware preflight in `src/musubi_tuner/dataset/config_utils.py`; retain `<stem>_<source-width:04d>x<source-height:04d>_qi.safetensors` and `<stem>_qi_te.safetensors`, reject conflicting source/caption aliases before cache writes, and retain exact duplicate declarations as separate occurrences (FR-003–FR-005, FR-011). Depends on T007.
- [X] T009 [P] [US1] Add `source_image_sha256` metadata for role-bearing Qwen latent and text caches in `src/musubi_tuner/dataset/cache_io.py` without changing filenames or tensor keys; unroled cache output and legacy reads must remain compatible (FR-004, FR-005, FR-011). Depends on T002 and T006.
- [X] T010 [US1] Wire the shared role-aware preflight and source binding through `src/musubi_tuner/qwen_image_cache_latents.py` and `src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py`; both existing `--dataset_config` entrypoints must accept the same validation TOML, reject invalid sources/collisions before VAE or text-encoder loading, and continue accepting legacy TOML (FR-002–FR-005, FR-011). Depends on T007–T009.
- [X] T011 [US1] Build the strict validation manifest in `src/musubi_tuner/training/validation_inputs.py` from declared sources, not `ImageDataset.prepare_for_training` or cache globs; compute original-byte SHA-256 IDs, record source/caption/original dimensions/resolution/bucket and exact cache paths, verify readable Qwen latent/text tensors and metadata including source binding and effective caption, and reject every missing or mismatched item before weights load (FR-003–FR-005, FR-011). Depends on T008–T010.
- [X] T012 [US1] Connect validation preflight in `src/musubi_tuner/qwen_image_train_network.py` and the before-weight seam in `src/musubi_tuner/training/trainer_base.py`; a role-bearing file used as the training dataset must fail, a valid `val_dataset_config` must prepare the manifest before `_load_dit_and_swap`, and an absent path must leave training data and model setup unchanged (FR-001, FR-003, FR-006, FR-011). Depends on T003 and T011.

**Checkpoint**: US1 is independently testable by T005–T006 and existing dataset/cache tests. Cache creation remains an initial preprocessing operation; validation itself never recaches or changes declared membership.

---

## Phase 4: User Story 2 — Reproducible Checks (Priority: P2)

**Goal**: Enumerate the exact midpoint grid and content-derived noise checks while leaving every training RNG stream unchanged.

**Independent Test**: Repeated CPU enumeration of the same image gives the independently calculated SHA-256 seeds and epsilon, `N1*N2` checks and balanced halves; RNG snapshots remain identical through manifest, loader creation/iteration, and noise generation.

### Tests for User Story 2

- [X] T013 [US2] Add independent reference cases in `tests/test_qwen_image_validation_inputs.py` for 1-based `(i,j)`, `N1=2` levels `0.275/0.725`, larger even grids, `(F+U)*N1*N2` checks, low/high halves, exact `1000*t`, and `noisy_latent=(1-t)*latent+t*epsilon`; compute expected SHA-256 seed bytes without calling the code under test (FR-007–FR-009, FR-013). Depends on US1.
- [X] T014 [US2] Add RNG-isolation cases in `tests/test_qwen_image_validation_inputs.py` snapshotting Python, NumPy, PyTorch CPU and available CUDA states before/after validation construction, loader creation and iteration, and epsilon generation; assert equal noise for equal image/check across roles, reorderings, and events, and no eager all-item epsilon allocation (FR-008, FR-010, FR-013). Depends on US1.

### Implementation for User Story 2

- [X] T015 [US2] Implement the versioned newline-delimited ASCII serialization and SHA-256 seed derivation in `src/musubi_tuner/training/validation_inputs.py`: decimal signed `val_seed_noise`, lowercase 64-hex image SHA, decimal 1-based `i` and `j`, each newline-terminated after `qwen-image-val-noise-v1`; use the first eight digest bytes as big-endian, masked to a nonnegative 63-bit seed, excluding role/path/order/step/rank (FR-005, FR-008). Depends on T013.
- [X] T016 [US2] Implement sequential check enumeration in `src/musubi_tuner/training/validation_inputs.py`: `t_i=0.05+(i-0.5)*0.90/N1`, `N1/2` levels per half, `N2` realizations per level, one private CPU-generated float32 epsilon at a time, final mixing `(1-t_i)*latent+t_i*epsilon`, and trainer-facing `1000*t_i` without training sampling, shift, or rounding (FR-007–FR-010). Depends on T015.
- [X] T017 [US2] Remove validation-path global RNG consumption in `src/musubi_tuner/dataset/config_utils.py` and `src/musubi_tuner/training/validation_inputs.py`, including the dataset-group seed draw and loader iterator worker seeding; use private generators and `shuffle=False`, batch size 1, one read/check at a time, without altering the training reader's seed/shuffle behavior (FR-003, FR-010). Depends on T014 and T016.

**Checkpoint**: US2 is independently verified by T013–T014; no model forward, loss, or TensorBoard event is introduced.

---

## Phase 5: User Story 3 — Portable Frozen Input (Priority: P3)

**Goal**: Keep a location-independent validation identity and detect changes before resume or any later item read.

**Independent Test**: Relocation and declaration reordering preserve identity; changing any source/caption/geometry/cache/protocol field, removing a file, or altering occurrence count raises instead of refreshing the frozen set.

### Tests for User Story 3

- [X] T018 [US3] Add canonical-identity CPU cases in `tests/test_qwen_image_validation_inputs.py` for relocated directories and reordered entries, same-content IDs across roles, retained duplicate occurrences, and changed image, caption, resolution/bucket, latent/text cache bytes, membership or effective controls; compare against an independent canonical record expectation (FR-004, FR-005, FR-012, FR-013). Depends on US1 and US2.
- [X] T019 [US3] Add later-read and expected-identity cases in `tests/test_qwen_image_validation_inputs.py` for changed or removed files after initial preparation and mismatch on resume comparison; assert source-labelled failure rather than rescan, substitution, or new snapshot (FR-004, FR-011–FR-013). Depends on US1 and US2.

### Implementation for User Story 3

- [X] T020 [US3] Add immutable, versioned validation identity in `src/musubi_tuner/training/validation_inputs.py`: role-labelled sorted multisets with duplicates retained, image and caption fingerprints, original dimensions, configured resolution/bucket and selected bucket, full latent/text cache digests, and effective numeric controls; canonical UTF-8 JSON must exclude paths/names and hash to the same value after relocation (FR-005, FR-012). Depends on T018 and US1.
- [X] T021 [US3] Add expected-identity comparison and per-item recheck before later reads in `src/musubi_tuner/training/validation_inputs.py`; compare current bytes and properties against the frozen manifest, raise an actionable change error, and never update the snapshot or silently accept a new item (FR-004, FR-011, FR-012). Depends on T019–T020.

**Checkpoint**: US3 exposes identity and comparison for later resume integration; this stage does not persist full training state or schedule validation events.

---

## Phase 6: Polish & Cross-Cutting Checks

**Purpose**: Finish compatibility, documentation, and local evidence without expanding the feature.

- [X] T022 [P] Review and update `README.ru.md` to explain manual familiar/unfamiliar sets, one role-bearing TOML accepted by both cache commands, required caches, deterministic checks, and the stage boundary that loss logging/training integration is still pending (FR-001–FR-003, FR-013). Depends on US1–US3.
- [X] T023 [P] Review and update `docs/qwen_image.md` against the implemented parser/cache behavior, including `val_dataset_config` disabling, public control names, missing-input errors, and the absence of a connected validation loop (FR-001, FR-006, FR-011, FR-013). Depends on US1–US3.
- [X] T024 Run the CPU acceptance in `specs/002-val-loss-core/quickstart.md` against `tests/test_qwen_image_validation_inputs.py`, `tests/test_qwen_image_dataset_cache.py`, and `tests/test_qwen_image_config.py`; inspect failures against FR-001–FR-013, rerun affected local checks after fixes, and record unavailable checks accurately without launching training, GPU, weights, or server work (FR-001, FR-013). Depends on T001–T023.
- [X] T025 Review `README.ru.md`, `docs/qwen_image.md`, and `src/musubi_tuner/qwen_image_train_network.py` for compatibility with unroled train/cache configs and absence of a validation-loop, loss, or TensorBoard claim; confirm no new model copy or altered training noise/timestep path (FR-001, FR-003, FR-006, FR-010, FR-013). Depends on T024.

---

## Dependencies & Execution Order

### Phase Dependencies

1. **Setup**: T001 prepares CPU fixtures.
2. **Foundational configuration**: T002 and T003 can run in parallel after T001; T004 checks their combined effective behavior.
3. **US1**: Starts after T004. T005 and T006 can run in parallel; T007→T008, T009, T010→T011→T012 complete strict input preparation.
4. **US2**: Starts after US1; T013 and T014 can be developed in parallel, followed by T015→T016→T017.
5. **US3**: Starts after US1/US2; T018 and T019 can be developed in parallel, followed by T020→T021.
6. **Polish**: T022 and T023 can run in parallel after the story phases; T024→T025 close local verification.

### Story Boundaries

- **US1 (P1, MVP)**: Delivers strict two-role preparation and early errors without noise enumeration.
- **US2 (P2)**: Reuses US1 items to deliver deterministic checks; its grid and seed logic is independently testable with CPU tensors.
- **US3 (P3)**: Reuses prepared items to deliver identity and change detection; no complete trainer resume is asserted.
- Tests are written before their corresponding implementation tasks. A check counts as passed only after its behavior is observed, not because a test file exists.

### Parallel Examples

- **US1**: T005 (`tests/test_qwen_image_dataset_cache.py`) and T006 (`tests/test_qwen_image_validation_inputs.py`) can proceed concurrently after configuration work. T009 (`dataset/cache_io.py`) can proceed while T008 develops collision checks in the dataset code.
- **US2 / US3**: Their test tasks share `tests/test_qwen_image_validation_inputs.py`; edit them serially to avoid file conflicts. The computation and identity work follow their stated dependencies.

## Implementation Strategy

1. Complete setup and effective configuration validation.
2. Deliver US1 and verify its strict preflight as the MVP before adding check generation.
3. Deliver US2, preserving training RNG and the exact final mixing/timestep protocol.
4. Deliver US3, then review documentation and run the finite local CPU acceptance.
5. Stop after the stage-1 component is verified locally. Training-loop metrics, TensorBoard, full resume persistence, and operational GPU verification require later stages.

## Phase 7: Convergence

- [X] T026 Reject an unroled `val_dataset_config` in `src/musubi_tuner/training/validation_inputs.py` with a source-labelled correction before manifest creation; add a CPU case in `tests/test_qwen_image_validation_inputs.py` that proves the error precedes model loading (FR-003, FR-011; partial).
- [X] T027 Reject a shared Qwen latent cache path with differing selected buckets or effective preprocessing settings in `src/musubi_tuner/dataset/config_utils.py` before either cache command loads weights; add a two-role shared-source/cache CPU case in `tests/test_qwen_image_dataset_cache.py` (FR-004, FR-005, FR-011; partial).
- [X] T028 Reject nonfinite latent and text cache tensors during strict validation preflight in `src/musubi_tuner/training/validation_inputs.py`; add real safetensors NaN/Inf CPU cases without changing legacy cache writers (FR-004, FR-011; partial).
- [X] T029 Normalize equivalent source/cache path spellings in the later-read effective-config comparison in `src/musubi_tuner/training/validation_inputs.py`, and test that `data/familiar` and `data/./familiar` do not report a changed frozen input; retain detection of changed effective settings (Constitution III minimal compatible behavior; unrequested strictness).
