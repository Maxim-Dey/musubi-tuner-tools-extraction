# Feature Specification: Deterministic Validation Inputs and Noise

**Feature Branch**: `001-scope-qwen-image-lora` (current branch; feature directory is independent)

**Created**: 2026-09-23

**Status**: Draft

**Input**: Stage 1.1 of `config_for_qwen_image_lora/val_loss_speckit_commands.md`: establish fixed validation inputs and deterministic noise checks for the existing Qwen-Image original LoRA trainer.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Define two fixed validation sets (Priority: P1)

The user manually supplies captioned images in two explicitly named sets. `val_familiar` contains training images; `val_unfamiliar` contains images excluded from training. The user can inspect exactly which source image, caption, resolution or bucket, and latent and text caches belong to each declared item before model weights load.

**Why this priority**: A changing, incomplete, or mislabelled input set makes every later validation result unreliable.

**Independent Test**: With small captioned image collections and their expected caches, prepare the validation input without loading model weights; verify both roles, every declared item, associations, and failures for invalid or missing input.

**Acceptance Scenarios**:

1. **Given** nonempty, valid collections for both roles, **when** the validation input is prepared, **then** every declared image is represented once under its explicit role with its own caption and required caches.
2. **Given** the two collections are listed in a different order or their directories are renamed without changing their contents, **when** the input is prepared again, **then** roles remain explicit and each unchanged image keeps its image ID.
3. **Given** either role is empty, a role is missing or invalid, a caption or required cache is absent or corrupt, or a source-to-cache association is ambiguous, **when** preflight runs, **then** it fails before model weights load and identifies the offending input and correction.
4. **Given** an image is represented in both roles, **when** its content is identified, **then** both occurrences have the same image ID; neither collection is automatically edited or deduplicated.

---

### User Story 2 - Reproduce validation checks (Priority: P2)

For every image the user gets the same defined noise levels and noise realizations at each validation event, without changing training randomness.

**Why this priority**: Identical inputs and noise are needed to compare validation results across steps and after resume.

**Independent Test**: With a fixed image and settings, enumerate checks twice in changed input order and verify the count, levels, seeds, image identity, and unchanged training random states on CPU.

**Acceptance Scenarios**:

1. **Given** `N1=2` and `N2=1`, **when** checks are enumerated for one image, **then** there are exactly two levels, `0.275` and `0.725`, with one low-noise and one high-noise check.
2. **Given** valid `N1` and `N2`, **when** checks are enumerated for an image, **then** exactly `N1*N2` checks use 1-based level and realization indices and each noise half contains `N1/2` levels.
3. **Given** the same image bytes and settings, **when** checks are generated again at another training step, rank, path, order, or role, **then** corresponding checks have the same seed and noise; different image bytes receive a different content ID.
4. **Given** an existing training random state, **when** validation input, its loader, and noise checks are prepared, **then** Python, NumPy, PyTorch CPU, and PyTorch CUDA training random states remain unchanged.

---

### User Story 3 - Resume with unchanged validation input (Priority: P3)

The user can relocate an unchanged experiment and continue with the same validation protocol. If any validation source or derived input changes, preparation or subsequent reading stops with an error instead of silently accepting a new set.

**Why this priority**: Comparisons across a resumed run are valid only when its underlying validation input is unchanged.

**Independent Test**: Record a validation input identity, relocate the unchanged files, then change a caption, source image, resolution or bucket, or required cache and attempt to use the recorded identity.

**Acceptance Scenarios**:

1. **Given** an unchanged validation input moved to another parent directory, **when** its identity is checked, **then** the protocol identity, image IDs, and seeds remain unchanged.
2. **Given** a previously fixed input whose image, caption, resolution or bucket, or required cache changes before resume or a later read, **when** it is checked, **then** use stops with an error identifying the change; the input is not refreshed automatically.

### Edge Cases

- An image has no required caption, its caption is unreadable, or a cache is missing, unreadable, corrupt, or associated with another image: fail rather than skip the item.
- Distinct source paths or names collide when mapped to captions or caches: fail on ambiguous association rather than substitute another item's data.
- Identical image bytes occur more than once: preserve each declared occurrence without automatic deduplication; content IDs and corresponding seeds remain equal. The user is responsible for the intended set membership.
- An explicit role is misspelled, absent, duplicated instead of providing both roles, or assigned by directory name or entry order: reject the invalid declaration.
- A validation parameter is a boolean in place of an integer, outside its allowed range, or an unknown field: reject the effective configuration before model loading.
- `N1=2` and `N2=1` are the minimum valid grid; odd `N1`, `N1<2`, and `N2<1` are invalid.
- A repeated read detects a changed input or a removed file: stop; do not replace, rescan, or silently shrink the set.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The feature MUST apply only to the existing Qwen-Image original LoRA workflow. This stage MUST provide fixed validation inputs and deterministic checks ready for later connection; it MUST NOT activate an unfinished validation loop, TensorBoard integration, other models, other validation modes, or a general metrics system.
- **FR-002**: The user MUST manually provide two nonempty captioned image sets: `val_familiar` containing training images and `val_unfamiliar` containing images not used for training. The feature MUST NOT automatically split, populate, transfer, replace, repair, or deduplicate set members. Exact content overlap MAY be checked with SHA-256; semantic similarity detection is outside scope.
- **FR-003**: Validation MUST use a separate `val-dataset.toml` whose declarations explicitly identify the `val_familiar` and `val_unfamiliar` roles. Roles MUST NOT be inferred from entry order or directory name. `train-dataset.toml` MUST continue to describe training only. Each declared validation item MUST be visited once per validation event, with validation `batch_size=1` and `num_repeats=1`; training shuffle, repeats, caption dropout, and random transformations MUST NOT be inherited.
- **FR-004**: The prepared validation input MUST fix set membership, captions, resolution or bucket assignments, and the latent and text caches used for every image. Existing deterministic preprocessing during initial caching is permitted. Validation MUST NOT randomly crop, flip, augment, alter captions, or regenerate caches during evaluation. Missing, damaged, or ambiguous input MUST cause an error rather than silently omitting or substituting an item.
- **FR-005**: `image_id` MUST equal the SHA-256 digest of the original image bytes. It MUST be independent of process hash, path, name, input order, batch, rank, and device. Identical bytes MUST yield the same ID in either set. Every source image, caption, latent cache, and text cache association MUST be unambiguous; filename collisions MUST NOT silently change an association.
- **FR-006**: The public controls MUST retain these exact names and meanings: `val_dataset_config` is the path to `val-dataset.toml`, and its absence disables validation; `val_every_n_steps` is an integer at least 1; `val_seed_noise` is one common integer seed for both sets; `val_level_noise_n` is `N1`, an even integer at least 2; `val_seed_noise_n` is `N2`, an integer at least 1 and the number of independent noise realizations at each level. Boolean values MUST NOT count as integers. The effective configuration MUST be checked after defaults, TOML, and explicit CLI overrides; unknown fields MUST be rejected. When validation is disabled, existing training behavior and configurations MUST remain unchanged.
- **FR-007**: Each image MUST have exactly `N1*N2` checks. For 1-based `i=1..N1`, the mixing level MUST be `t_i = 0.05 + (i - 0.5) * 0.90 / N1`, the midpoint of an equal interval in `[0.05, 0.95]`. Levels with `0.05 <= t_i < 0.5` are low noise; levels with `0.5 <= t_i <= 0.95` are high noise. Each group MUST contain `N1/2` levels. For `N1=2`, levels MUST be `0.275` and `0.725`.
- **FR-008**: For 1-based `j=1..N2`, the noise seed MUST be `stable_hash(val_seed_noise, image_sha256, i, j)`. `stable_hash` MUST use SHA-256 with one fixed, unambiguous serialization and convert its digest to a valid seed. It MUST exclude validation role, training step, rank, filename, path, and item order. The same image bytes and settings MUST reproduce the same noise for corresponding checks across validation events and roles.
- **FR-009**: Validation MUST use an epsilon generator separate from training randomness, and each check MUST use `noisy_latent = (1 - t_i) * latent + t_i * epsilon`. `t_i` is the final mixing coefficient. The trainer's 0–1000 scale MUST receive `timestep = 1000*t_i`. The validation level MUST NOT be shifted again, rounded to a training timestep, or randomly reselected; training `timestep_sampling` and `discrete_flow_shift` MUST NOT alter this grid.
- **FR-010**: Preparing the validation dataset and loader and generating validation noise MUST leave Python, NumPy, PyTorch CPU, and PyTorch CUDA training RNG states unchanged. Checks MUST be processed sequentially without precomputing all dataset noise, and validation MUST NOT load a second copy of model weights.
- **FR-011**: All invalid parameters, empty or invalid roles, missing required captions or caches, and other input problems detectable without model weights MUST fail before weight loading. Errors MUST identify the source, cause, and correction. No validation item may be silently skipped.
- **FR-012**: The fixed validation input MUST have a recorded identity sufficient to detect changes at resume and on subsequent reads. A changed source image, caption, resolution or bucket assignment, required cache, or set membership MUST cause an error, not automatic refresh. Relocating an unchanged experiment MUST preserve its image IDs, noise seeds, and protocol identity.
- **FR-013**: Acceptance MUST include CPU checks of the grid, SHA-256-derived seeds, 1-based indices, check counts, renamed and reordered inputs, unchanged training RNG states including loader preparation, empty sets, invalid types and values, missing captions or caches, and incorrect roles. These checks establish the input/noise component only; they MUST NOT be described as a completed validation loop or real-model verification.

### Key Entities *(include if feature involves data)*

- **Validation set**: A user-declared, nonempty collection with one explicit role, `val_familiar` or `val_unfamiliar`, and stable membership.
- **Validation item**: One declared original image with caption, resolution or bucket assignment, required latent and text caches, and a content-derived `image_id`. Identical contents can appear in multiple declared items without automatic deduplication.
- **Validation input identity**: A location-independent record of both set memberships and every input needed to recognize changes across reads and resume.
- **Noise check**: One validation item, 1-based level index `i`, 1-based realization index `j`, final mixing level `t_i`, and deterministic seed.
- **Effective validation configuration**: The public controls after defaults, TOML, and explicit CLI overrides, including the disabled state when no validation dataset is configured.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For any valid input with `F` familiar images and `U` unfamiliar images, preparation accounts for all `F+U` declared items, visits each once per event, and exposes exactly `(F+U)*N1*N2` checks, with no silent omissions.
- **SC-002**: For every valid even `N1`, exactly `N1/2` levels are low and `N1/2` are high; the `N1=2` boundary produces `0.275` and `0.725` and all indices begin at 1.
- **SC-003**: Repeated preparation of unchanged content across reordered entries, renamed paths, another role, and another training step preserves 100% of corresponding image IDs, seeds, and validation input identity where set membership is unchanged.
- **SC-004**: Every exercised empty-set, invalid-role, invalid-type/value, unknown-field, missing-caption/cache, ambiguous-association, and changed-input case fails before weights load or at the later read that discovers a change; zero such cases silently skip or substitute an item.
- **SC-005**: Before and after validation input and noise preparation, all observed Python, NumPy, PyTorch CPU, and available CUDA training RNG states are equal, and the disabled configuration preserves the existing training path.
- **SC-006**: All prescribed CPU acceptance checks pass for the input/noise component. Documentation and results state clearly that training-loop, TensorBoard, GPU, and real-model verification belong to later stages.

## Assumptions

- The constitution at `.specify/memory/constitution.md` governs this feature unchanged. This is a local specification stage; no training, GPU run, model download, or server transfer is authorized by it.
- The user prepares captions, membership, and required caches. A declared validation item represents one source occurrence; content identity does not authorize automatic deduplication or role changes.
- Existing applicable Qwen-Image original image, caption, and cache formats remain the compatibility baseline. A concrete file schema and reader integration belong to planning; the separate validation TOML and public field names are required user contracts.
- Validation input/noise reproducibility is distinct from bitwise equality of model calculations across different hardware or backends. This stage promises the former and does not claim the latter.
- The current shared dataset reader does not yet recognize a validation role, while training reads cache-derived records and may skip missing text caches. Both behaviors require strict validation-specific handling in later implementation; neither is treated as evidence that the requirement already works.
- The prescribed sequence continues with a separate plan, tasks, analysis, implementation, and convergence. This invocation creates only the feature specification, its standard quality checklist, and the Spec Kit feature pointer.
