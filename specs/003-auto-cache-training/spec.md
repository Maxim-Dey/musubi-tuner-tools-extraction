# Feature Specification: Automatic cache preparation before Qwen training

**Feature Branch**: codex/auto-cache-training
**Created**: 2026-10-06
**Status**: Complete (local acceptance)
**Input**: Automatically create missing caches when launching Qwen LoRA training; a wrapper is optional. Follow SpecKit, commit the completed validation feature first, and do not conduct extensive GPU testing.

## User Scenarios & Testing

### User Story 1 — Start with uncached images (Priority: P1)

The owner starts the existing training command. Required image and text caches are prepared for train and optional independent val images before training begins.
**Why this priority**: Removes the four manual preparation commands.
**Independent Test**: Local integration uses real images, configuration readers and cache files with expensive encoder/model execution substituted.
**Acceptance Scenarios**:
1. With absent train/val caches, one launch prepares both cache types for both sets before training model initialization.
2. With train only, only its caches are prepared.
3. Invalid sources or a missing encoder needed by any planned stage stop launch before preparation or training.

### User Story 2 — Reuse complete or partial caches (Priority: P1)

The owner restarts without regenerating existing valid files. Interrupted preparation can finish its remaining files.
**Why this priority**: Avoids repeated encoding and preserves validation identity.
**Independent Test**: Compare existing cache bytes and count preparation launches.
**Acceptance Scenarios**:
1. Complete valid caches cause zero encoder launches and add no unused preparation-only encoder requirement.
2. Partial caches cause only missing files to be created, preserving existing and unrelated files.
3. Present corrupt, incompatible or detectably stale files cause an error; they are not silently replaced or accepted as a partial training population.
4. Resuming retains existing validation fingerprint checks after preparation.

### User Story 3 — Control the existing launch (Priority: P2)

The owner controls automatic preparation in the training configuration or CLI and receives clear progress and errors.
**Why this priority**: Preserves manual workflows and makes supported scope explicit.
**Independent Test**: Real parser/configuration checks, another working directory, failure and disabled-mode integration.
**Acceptance Scenarios**:
1. The supplied main profile enables preparation. Other profiles keep manual behavior unless enabled. Explicit CLI disable overrides the file.
2. Experiment-relative paths keep their meaning, including paths with spaces and launch from another working directory.
3. Automatic preparation outside experiment mode or with multiple training processes fails before writes. Disabling it preserves existing workflows.
4. Failed preparation names the dataset and stage, prevents training and preserves completed files for retry.

### Edge Cases

Missing one file; empty sources or captions; duplicate cache names; overlapping train/val contents or cache directories; directory/broken link at a cache path; corrupt tensors/provenance; unexpected training caches; successful child leaving missing output; val disabled; CLI override; resume identity; multiple ranks; paths with spaces; RNG and user-setting preservation.

## Requirements

### Functional Requirements

- **FR-001**: Provide opt-in automatic preparation through the existing Qwen original training launch/configuration. Enable it in the supplied main experiment profile; preserve all user training values and prompts. Disabled mode retains manual behavior.
- **FR-002**: Discover every required cache from effective source images/captions, not just existing cached items. Validate source declarations, cache destinations and train/val isolation before preparation.
- **FR-003**: Create only missing latent/text files. Reuse supported valid files unchanged and skip fully complete stages. Never delete unrelated files automatically.
- **FR-004**: Present invalid, corrupt or detectably stale files remain actionable errors. Preserve existing provenance/geometry/caption checks and resumed validation identity. Distinguish absent files from invalid existing paths; never silently train on partial populations.
- **FR-005**: Finish preparation before training model initialization or updates. Stages run sequentially with each encoder released before the next stage/training; preserve parent RNG and numerical training settings.
- **FR-006**: Validate resources needed by all planned preparation stages before executing any stage. Complete caches add no unused preparation-only encoder requirement; existing training/sampling resource requirements remain unchanged.
- **FR-007**: Any preparation failure or incomplete output stops before training with dataset/stage/correction context. Preserve completed files for retry.
- **FR-008**: Respect effective configuration, CLI priority, strict booleans and existing experiment-relative paths. Preparation uses the selected training models/datasets; val preparation remains deterministic.
- **FR-009**: Automatic preparation supports one training process in experiment mode; reject unsupported automatic-mode combinations before cache writes. Existing manual/distributed launches remain available when disabled.
- **FR-010**: Keep independent cache commands operational. Document automatic launch, reuse, disable, supported scope and manual recovery in the existing Russian README and Qwen guide.
- **FR-011**: Complete specify→plan→tasks→analyze→implement→converge with English artifacts, independent review and suitable CPU integration/regressions. No model downloads, remote runs, packaging or GPU training are required; never claim them performed.
- **FR-012**: Commit completed validation separately before new implementation, preserving unrelated user edits. Give the new feature its own specification/branch and retain historical validation evidence.

### Key Entities

- Source dataset: effective image/caption population and cache destinations.
- Cache inventory: required latent/text paths classified as missing, supported valid or invalid.
- Preparation stage: one dataset role and cache type, selected only when files are missing.
- Training launch: effective settings, optional preparation, existing validation checks, native training.

## Success Criteria

### Measurable Outcomes

- **SC-001**: One uncached train/val launch reaches training only after all four preparation stages succeed.
- **SC-002**: Complete-cache relaunch performs zero encoder stages and changes zero cache bytes; partial preparation creates only missing files.
- **SC-003**: All tested invalid-input, failed-stage and incomplete-output cases stop before training with actionable context and no deletion of user files.
- **SC-004**: Local checks prove launch ordering, configuration/path compatibility, preservation and failure behavior; final convergence has zero confirmed gaps. CPU tests do not establish GPU performance or output quality.

## Assumptions

The current user workflow is Qwen original, experiment mode, one GPU. Opt-in elsewhere preserves manual behavior. Validity uses existing source/cache contracts; encoder-checkpoint identity is not newly tracked. Cache formats and resume semantics remain authoritative. No new training engine, model downloads, concurrent-run locking, distributed cache coordinator or performance benchmark is included.
