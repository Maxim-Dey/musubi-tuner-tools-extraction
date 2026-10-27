# Feature Specification: Krea2 Raw LoRA Only

**Feature Branch**: `codex/krea2-raw-lora-only`

**Created**: 2026-10-26

**Status**: Local implementation and verification complete

**Input**: Execute the ordered commands in `config_for_krea2_raw_lora/speckit_prompts.md`: retain only Krea2 Raw LoRA training and its required components, supply a coherent starter configuration, and verify locally. Other models and independent workflows may be removed incompatibly. Work in this feature and branch only.

## User Scenarios & Testing

### User Story 1 - Preserve the Raw training workflow (Priority: P1)

A trainer prepares captioned images, caches their latents and text, trains a Raw LoRA with the existing applicable settings, reviews training samples and logs, saves adapters and training state, and resumes later.

**Why this priority**: Removing other workflows must not damage the remaining workflow or change what its settings mean.

**Independent Test**: Exercise real imports, parsers, cache round trips and small CPU regression fixtures; inspect the training, sampling, checkpoint and resume call paths against the baseline.

**Acceptance Scenarios**:

1. **Given** a supported image dataset, **when** both cache commands are configured, **then** they preserve the existing Raw cache formats and the training loader consumes them.
2. **Given** applicable Raw training settings, **when** the trainer is configured, **then** its loss, noise, precision, gradients, optimizer, scheduler and RNG semantics remain unchanged, including optional optimizations.
3. **Given** sampling, logging and saving settings, **when** their existing training hooks run, **then** Raw control images, logs, LoRA weights and resumable state remain available without changing subsequent training.

### User Story 2 - Remove unsupported workflows (Priority: P1)

A maintainer has a repository containing only the remaining workflow and components it actually uses.

**Why this priority**: A smaller set of entry points alone does not meet the requested extraction.

**Independent Test**: Inventory the remaining code, dependencies, tests and documentation; import every retained module without removed modules present.

**Acceptance Scenarios**:

1. **Given** the extracted repository, **when** its executable code is inspected, **then** training other models, full fine-tuning, Turbo, video, image editing and standalone generation implementations and entry points are absent.
2. **Given** a necessary shared component such as the Qwen-Image VAE or Qwen3-VL encoder, **when** it remains, **then** only its required use survives; it does not retain a separate Qwen Image training workflow.
3. **Given** an unknown parameter, excluded mode or incompatible setting combination from CLI or TOML, **when** a command starts, **then** it reports a clear source, cause and correction before loading model weights.

### User Story 3 - Start from consistent examples (Priority: P2)

A trainer replaces clearly identified personal paths and captions in three starter files and follows the README from the repository root.

**Why this priority**: The retained workflow needs a usable entry point grounded in actual supported settings.

**Independent Test**: Parse all starter files through real consumers, verify their cross-references and README commands without launching a model.

**Acceptance Scenarios**:

1. **Given** `train.toml`, `dataset.toml` and `sample_prompts.txt` under `config_for_krea2_raw_lora/`, **when** personal values are replaced, **then** the documented cache and training commands share compatible paths and settings.
2. **Given** a default, a TOML value and an explicit CLI value for one setting, **when** configuration is resolved, **then** CLI overrides TOML and TOML overrides the default; the effective configuration is validated.
3. **Given** README instructions, **when** followed, **then** they describe dataset preparation, both caches, training, Raw control images, logging, LoRA/state saving and resume, without implying unknown hardware compatibility.

### Edge Cases

- Unknown TOML keys, invalid types/choices and removed-mode keys must fail even when the starter files are otherwise valid.
- CLI overrides must work even when the explicit CLI value equals the original default.
- Shared component names containing another architecture do not establish that the component is removable.
- Cached image datasets, repeats, buckets, captions and optional settings retain their applicable semantics; example resolution, rank and duration are not program limits.
- Sampling and resume must preserve training RNG/state; retaining their filenames alone is insufficient.
- Optional runtime dependencies must remain available for applicable optimizations even when the starter config does not enable them.

## Requirements

### Functional Requirements

- **FR-001**: Preserve image-and-caption datasets, latent caching, text caching and Krea2 Raw LoRA training, with their existing supported formats.
- **FR-002**: Preserve the meaning of applicable training settings and optimizations, including settings disabled in the example; preserve training algorithm, loss/noise, precision, gradient, optimizer, scheduler and RNG behavior.
- **FR-003**: Preserve Raw control-image generation within training, logging, LoRA saving, training-state saving and resume without auxiliary work affecting training.
- **FR-004**: Remove implementations and entry points for other model training, full fine-tuning, Turbo, video, image editing, standalone generation and other independent workflows outside FR-001–003.
- **FR-005**: Retain necessary shared components only to the extent used by FR-001–003, including Qwen-Image VAE and Qwen3-VL. Retained commands must not depend on deleted modules, including dynamic network loading.
- **FR-006**: Remove configuration, documentation, tests and dependencies used solely by excluded workflows; inspect lock files if present.
- **FR-007**: Provide `config_for_krea2_raw_lora/train.toml`, `dataset.toml` and `sample_prompts.txt`, consistent with actual parsers and consumers. Mark user substitutions; align relative paths with execution from the repository root.
- **FR-008**: Resolve defaults → TOML → CLI, validate the effective configuration, and reject unknown parameters, excluded modes and detectable incompatibilities before model loading with actionable errors identifying their source.
- **FR-009**: README must describe only supported workflows and valid commands covering dataset preparation, both caches, training, sampling, logging, saving and resume. Example choices must not constrain the program or promise compatibility with unknown hardware.
- **FR-010**: Preserve Git, licenses, user data, Spec Kit infrastructure, the prompt file, other features and other branches/worktrees. Do not modify the constitution.
- **FR-011**: Record baseline behavior before substantial logic changes; run available meaningful local checks with real imports/parsers/configs and regressions. Do not launch training, GPU runs, weight downloads, package builds or remote verification, or describe local checks as real-training verification.
- **FR-012**: Use existing components without a new framework, unnecessary abstractions, unrelated features or benchmarks. Complete ordered Spec Kit stages, independent multi-agent verification and at most two recorded post-implementation correction rounds across the whole feature.

### Key Entities

- **Image dataset**: Image paths, captions, resolution/buckets/repeats and cache location.
- **Cached representations**: Existing Raw latent and text tensor names, shapes, dtypes, metadata and file associations.
- **Training configuration**: Resolved supported settings, model/data paths and outputs, including optional memory and training optimizations.
- **Training artifacts**: Raw sample images, logs, adapter weights and state required for resumption.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Every stage of the retained workflow has a verified local code path and all prescribed available local regression checks pass.
- **SC-002**: No excluded standalone workflow implementation, entry point or exclusive dependency remains, and all retained command imports succeed independently of deleted modules.
- **SC-003**: All three starter files are accepted by their actual parsers and their references match the README commands after documented personal substitutions.
- **SC-004**: Checks cover unknown CLI/TOML parameters, excluded modes, incompatible combinations and CLI precedence, all before model loading.
- **SC-005**: README covers all retained user actions, the constitution and protected material remain intact, and any unperformed verification is reported accurately.

## Assumptions

- The user will supply local model files, captioned images and compatible runtime hardware; this local task does not obtain them.
- Explicit authorization to remove other models and modes supersedes compatibility obligations for those removed workflows only.
- Existing applicable Raw settings and artifact contracts are the reference behavior; no performance claim is requested.
- The prompt file is the user-selected task specification. Its scope does not authorize changes to unrelated features or external services.
