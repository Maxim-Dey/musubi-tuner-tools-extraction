# Feature Specification: Scope to Qwen-Image Original LoRA

**Feature Branch**: `001-scope-qwen-image-lora` (existing branch; do not create or switch branches)

**Feature Directory**: `specs/001-scope-qwen-image-lora`

**Created**: 2026-09-22

**Status**: Draft — analysis findings incorporated; implementation not started

**Input**: Narrow this repository to LoRA training for Qwen-Image original and the components required by that workflow. Preserve image/caption dataset preparation, latent and text-embedding caching, LoRA training, sample images during training, logging, LoRA and training-state saving, and resume. Preserve existing applicable settings and optimizations, including those disabled in the example. Remove implementations and entry points for other models, full finetuning, video training, Qwen-Image Edit/Layered, and unrelated standalone workflows, together with their exclusive configurations, documentation, tests, and dependencies. Preserve compatibility with the three supplied templates, subject only to corrected internal references and explicitly approved semantic corrections. Do not change the training algorithm, add features or frameworks, or perform real-model runs during the local stage. This invocation produces this specification, its standard quality checklist, and the Spec Kit feature pointer, plus the single scheduler correction separately authorized by the user during specification; it does not start later phases.

**User clarification — 2026-09-22**: Extract a standalone Qwen-Image original LoRA trainer from the shared codebase while preserving the original algorithms and the existing training process. Independence means the retained workflow and its required components no longer depend on excluded model implementations; it does not require a replacement training engine. This clarification authorizes the document corrections identified by analysis, not implementation or a change to training mathematics.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Prepare image datasets and reusable caches (Priority: P1)

As a Qwen-Image LoRA user, I can prepare images with captions, configure one or more image datasets, and create the latent and text-embedding caches consumed by training.

**Why this priority**: Training is unusable if narrowing the repository breaks dataset preparation or either cache stage.

**Independent Test**: Locally inspect and check dataset configuration, image/caption association, bucket and repeat settings, and cache producer/consumer contracts using small fixtures without weights. This establishes local compatibility evidence, not successful real-model encoding.

**Acceptance Scenarios**:

1. **Given** the supplied dataset template with user-selected image and cache paths, **When** dataset preparation reads it, **Then** images are associated with their captions and the configured resolution, bucketing, no-upscale behavior, batch size, and repeats retain their existing meaning.
2. **Given** multiple supported image datasets, **When** preparation and caching process them, **Then** each dataset retains its own source, cache location, and applicable overrides without introducing video or control-image workflows.
3. **Given** the same supported Qwen-Image original inputs and options as before narrowing, **When** latent and text-embedding caching are used, **Then** both stages remain available and their outputs retain the existing format, naming, and training-consumption semantics, including applicable cache reuse options.

### User Story 2 - Train, inspect, save, and resume LoRA (Priority: P1)

As a Qwen-Image LoRA user, I can use the retained training command, inspect sample images and logs, save LoRA checkpoints and training state, and resume an interrupted run.

**Why this priority**: These capabilities form the requested training workflow and must survive the removal of shared code for other models.

**Independent Test**: Review the retained training path and run focused local checks of changed sampling, logging, saving, and resume behavior without loading model weights or launching training. Existing relevant regression checks are reused where available.

**Acceptance Scenarios**:

1. **Given** a valid configuration for Qwen-Image original LoRA and compatible caches, **When** the retained workflow is selected, **Then** it preserves the existing LoRA training behavior, applicable settings, and frozen base-model behavior.
2. **Given** the supplied sample prompt format and a configured sample schedule, **When** a sample event is due, **Then** image generation remains available with the requested prompt, dimensions, seed, step count, guidance, and flow shift, even though the standalone inference command is removed.
3. **Given** sampling and logging are enabled, **When** those operations occur, **Then** they preserve their existing outputs and do not alter training loss, gradients, optimizer or scheduler progression, random-number state, or subsequent training behavior.
4. **Given** checkpoint and training-state saving are enabled, **When** their configured save boundaries are reached, **Then** LoRA files, metadata, save precision, and separate checkpoint/state retention rules retain their existing meaning, including retention boundaries.
5. **Given** a compatible saved training state, **When** resume is requested, **Then** the existing supported restoration of LoRA, optimizer, scheduler, and random-number state is preserved. The trainer's local epoch/global-step counters still restart and previously consumed batches are not skipped; exact counter or data-cursor continuation is not promised by this feature.
6. **Given** an existing applicable optimization is disabled or omitted in the sample configuration, **When** a user enables a previously valid combination, **Then** the option remains available with its existing semantics and prerequisites.

### User Story 3 - Use compatible templates and receive early errors (Priority: P1)

As a user, I can adapt the supplied templates to my files and settings, and receive actionable configuration errors before any model weights are loaded.

**Why this priority**: Template compatibility and early rejection are explicit acceptance conditions; merely recognizing a parameter name is insufficient.

**Independent Test**: Check the real configuration and prompt readers against all supplied entries, supported overrides, and representative invalid inputs. Verify that invalid configurations fail before any model loader is reached.

**Acceptance Scenarios**:

1. **Given** all three contract templates, the approved scheduler correction, corrected internal references, and user-supplied external paths, **When** their settings are resolved, **Then** every entry retains its supported name, type, and meaning, including `model_version="original"` and `network_module="networks.lora_qwen_image"`.
2. **Given** different valid paths, trigger words, scene descriptions, output locations, or supported hyperparameter values, **When** the templates are adapted, **Then** the workflow accepts them without requiring the example drive, folder names, GPU model, trigger token, or numeric values.
3. **Given** defaults, a training configuration, and command-line overrides, **When** settings are resolved, **Then** the existing precedence is preserved and the final effective configuration is validated before model loading.
4. **Given** an unknown key, wrong type, invalid combination, or unsupported mode in a command, configuration, dataset declaration, or sample prompt option, **When** that input is processed, **Then** the workflow rejects it before model loading and identifies the source, cause, and correction rather than ignoring it or silently substituting a value.
5. **Given** `lr_scheduler="constant"` with `lr_warmup_steps=200`, **When** configuration is validated, **Then** the incompatible combination is rejected early; the agreed example instead uses `constant_with_warmup` with 200 warmup steps.
6. **Given** an excluded mode selected through a legacy flag, configuration value, dataset kind, or network-module selection, **When** the user invokes a remaining command, **Then** that selection cannot activate an excluded training path, including when another input also requests `original`.

### User Story 4 - Maintain and follow a repository with one supported workflow (Priority: P2)

As a maintainer or user, I can understand and use a repository whose executable code, dependencies, examples, and documentation describe only Qwen-Image original LoRA training and its required supporting operations.

**Why this priority**: A superficial entry-point cleanup would leave the complexity and unsupported implementations that this feature is intended to remove.

**Independent Test**: Inspect remaining executable paths, shared components, dependency references, and documentation; perform applicable local import and command-contract checks without weights or execution of training.

**Acceptance Scenarios**:

1. **Given** the narrowed repository, **When** remaining executable code is inspected, **Then** excluded model and mode implementations are absent from entry points, nested modules, shared branches, and alternate dispatch paths; disabling them at the command line alone does not satisfy this condition.
2. **Given** shared code previously used by Qwen-Image and excluded workflows, **When** removal is complete, **Then** the Qwen-Image workflow retains the portions it needs and no remaining code requires a deleted module or a dependency used only by excluded workflows.
3. **Given** a user following README from the repository root, **When** they prepare data, cache latents and embeddings, train with sample images and logging, save, and resume, **Then** the documented sequence and examples match the retained commands, parsers, and template locations.
4. **Given** cleanup affects tracked project files, **When** its scope is reviewed, **Then** Spec Kit infrastructure, the unchanged constitution, Git metadata/history, licenses and required notices, and user data remain protected.

### Edge Cases

- The current `train.toml` refers to `qwen_image_lora/dataset.toml` and `qwen_image_lora/sample_prompts.txt`, but the supplied files are under `config_for_qwen_image_lora/`. These are fixable internal references, not reasons to duplicate the templates or change training behavior.
- Output, log, and cache directories need not exist before use. Their example locations are user choices, not broken references to shipped input files.
- External example weight and image paths may not exist locally. Syntax/contract inspection must distinguish replaceable placeholders from actual missing inputs at invocation time and must not load or download weights to validate them.
- A key may be recognized by a parser while its value, type, or combination is invalid. The current constant-scheduler/warmup conflict must not be hidden by dropping the warmup key or changing scheduler behavior.
- Unknown sample prompt options are currently warned about and ignored. The retained workflow must reject them early, as required for other unknown parameters.
- Video, Edit, Layered, and full-finetuning selections can appear outside top-level command names. Dataset declarations, prompt options, legacy aliases, network selection, and shared execution branches are also in scope for removal or rejection.
- A shared helper's historical model name, a single-frame latent dimension, or a VAE operation originally written for video does not by itself make that helper an excluded workflow. Retain computation required for Qwen-Image original images without retaining independent video capabilities.
- Deleting standalone inference, conversion, extraction, merging, captioning, or GUI tools must not delete helpers still required by the supported training workflow. Automated standalone caption generation is not a replacement for the required image/caption dataset preparation.
- A disabled example option, or an existing applicable numerical method with another model name in its label, is not sufficient reason to remove it. Applicability to Qwen-Image original training determines retention.
- Checkpoints and resumable training state are different artifacts. LoRA initialization alone must not be presented as full training-state resume, and their retention windows must not be conflated.
- Missing local dependencies or hardware capabilities may prevent a check. Such a check must be recorded as unavailable, not passed or worked around by disabling supported functionality.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The repository MUST support only Qwen-Image original LoRA training and operations necessary for the retained workflow. Removing other models and modes is the user's explicit incompatible-change decision under constitution principle III; compatibility remains required for the retained workflow.
- **FR-002**: Image/caption dataset preparation MUST retain existing applicable image-source and caption formats, multiple datasets, resolution, bucketing, no-upscale behavior, batching, repeats, and cache-location settings. Video, control-image editing, and layered-image datasets MUST be excluded.
- **FR-003**: Latent and text-embedding caching MUST remain available for Qwen-Image original. Existing compatible cache formats, data associations, consumption by training, and applicable reuse/retention settings MUST be preserved.
- **FR-004**: LoRA training MUST preserve the existing training algorithm and applicable defaults: loss and weighting, noise and timestep behavior, data processing, precision, gradients and accumulation, optimizer, scheduler, random-number behavior, and resume semantics. No numerical changes are authorized except the explicitly approved template correction below; the correction MUST use existing scheduler behavior.
- **FR-005**: Sample image generation during training MUST remain supported with existing applicable prompt formats, options, sampling schedules, and image outputs. Removing the standalone inference command MUST NOT remove this capability. Sampling MUST NOT update training weights or affect subsequent training behavior.
- **FR-006**: Existing applicable logging, metrics, and training/artifact metadata MUST remain available with their supported settings and output semantics, without altering training calculations or state.
- **FR-007**: LoRA saving, configured save precision, save schedules, naming/formats, training-state saving, separate retention rules, and resume MUST preserve existing Qwen-Image workflow compatibility. A resumable state MUST preserve the existing restoration of optimizer, scheduler, and random-number state in addition to LoRA weights. The existing trainer-local epoch/global-step reset and absence of data-cursor restoration MUST be preserved and documented; exact progress continuation is outside this extraction and requires a separate explicit user decision.
- **FR-008**: Existing settings and optimizations applicable to Qwen-Image original LoRA MUST remain supported even if absent, false, or zero in the example. This includes applicable precision and FP8 options, block swapping/offloading, attention choices, gradient checkpointing and CPU offload, compilation settings, data-loader settings, LoRA configuration, optimizer/scheduler choices, and sampling controls. This does not authorize new combinations, new optimizations, or preservation of settings used exclusively by excluded workflows.
- **FR-009**: The three files in the compatibility contract MUST remain the authoritative user templates. All supported keys/options MUST retain their names, accepted types, and meaning. The example values MUST NOT become hard-coded program constraints. Hyperparameters, keys, and enabled features MUST NOT be changed merely to make a check pass.
- **FR-010**: `model_version="original"` and the existing `network_module="networks.lora_qwen_image"` selection MUST continue to resolve to the retained workflow. Existing applicable commands, defaults, input/output formats, and override precedence MUST remain compatible, except for the expressly requested early rejection of invalid or excluded inputs. Existing attention-backend precedence MUST be preserved; multiple supported backend flags MUST NOT be rejected solely because more than one is set.
- **FR-011**: Every retained workflow command MUST validate its effective configuration after its applicable defaults, file settings, and command-line overrides are applied. Unknown keys/options, invalid types or values/combinations, and excluded modes detectable without the model MUST fail before any model weights are loaded, including in dataset and sample-prompt inputs.
- **FR-012**: Configuration errors MUST identify the input source, offending key/option or selection, cause, and a concrete correction. Invalid inputs MUST NOT be ignored, silently repaired, or deferred until after model loading. A newly discovered semantic defect in a contract template MUST be presented for the user's concrete decision before changing its meaning.
- **FR-013**: Implementations and entry points for all other model families, full finetuning, video/audio workflows, every Qwen-Image Edit variant, Qwen-Image Layered, and unrelated standalone scenarios MUST be removed throughout working code. Root-script deletion or mode rejection while leaving those implementations in place is insufficient. There MUST be no remaining executable training route for excluded models or modes.
- **FR-014**: Shared components MUST retain only the functionality needed by the supported workflow. Required image sampling, dataset, cache, model, and training helpers MUST remain usable without imports, dynamic references, or execution dependencies on removed modules. Required original-model tensor operations MUST NOT be reclassified as excluded video functionality solely by naming or shape.
- **FR-015**: Configurations, documentation, tests, dependencies, and optional dependency declarations used exclusively by excluded workflows MUST be removed. Shared material and dependencies MUST retain the portions needed by the supported workflow and its existing applicable options. Relevant regression protections MUST NOT be removed simply because their current location or imports also involve an excluded model.
- **FR-016**: README and retained workflow documentation MUST describe only the supported workflow with parser-consistent examples for data preparation, both cache stages, training, sample prompts, logging, saving, and resume. Examples MUST use the supplied template locations and clearly distinguish user-replaceable external paths from repository-relative references. Instructions MUST NOT depend on deleted scripts or documents.
- **FR-017**: Work MUST preserve Spec Kit infrastructure, the constitution, Git metadata/history, licenses and required notices, and user data, including existing user datasets, weights, caches, outputs, and unrelated user-authored files. Removal of implementation/configuration support for a workflow does not authorize deleting that workflow's user data.
- **FR-018**: The change MUST use existing mechanisms and remain limited to the requested narrowing and necessary compatibility/error handling. It MUST NOT introduce a framework, a general architecture for future models, new user features, benchmarks, performance budgets, or unnecessary weight/tensor duplication.
- **FR-019**: Verification MUST use source analysis and available local checks, reusing suitable existing tests and adding only checks for meaningful changed behavior or regressions. Required evidence MUST cover the compatibility templates and precedence, rejection before loading, image/cache contracts, removal and dependency integrity, training invariants affected by extraction, sample generation boundaries, saving/retention/resume, and README consistency. Checks must target behavior rather than coverage counts or incidental file layout.
- **FR-020**: The local stage MUST NOT launch training, GPU runs, weight downloads, package builds, server transfer, or server verification. Local checks without weights MUST NOT be described as proof of real-model training, image generation quality, or performance. Unperformed/unavailable checks MUST be recorded accurately, and the absence of external runs MUST be recorded once in the stage's final results.
- **FR-021**: This specification phase MUST create or update only this feature's specification and standard requirements checklist, plus its normal Spec Kit feature pointer, on the current branch, with the sole additional change being the scheduler correction separately authorized below. Planning, tasks, implementation, and later workflow stages MUST await a separate user instruction. Any real constitutional conflict MUST identify the exact clause and proposed minimal amendment and await the user's decision; the constitution MUST NOT be edited by this feature instruction.

### Compatibility Contract and Approved Decisions

The existing contract files are:

- `config_for_qwen_image_lora/train.toml`
- `config_for_qwen_image_lora/dataset.toml`
- `config_for_qwen_image_lora/sample_prompts.txt`

The current training template contains 40 keys. All are part of the contract; the following groups describe their current value types without narrowing the existing supported input domain.

| Group | Keys and current template types | Required meaning |
| --- | --- | --- |
| Model | `model_version`, `dit`, `vae`, `text_encoder`: strings | Original model selection and user-selected weight paths. |
| Dataset/loading | `dataset_config`: string; `max_data_loader_n_workers`: integer; `persistent_data_loader_workers`: boolean | Dataset reference and existing loader behavior. |
| LoRA | `network_module`: string; `network_dim`, `network_alpha`: integers | Existing Qwen-Image LoRA selection, rank, and alpha semantics; other already-supported numeric forms remain valid. |
| Precision/memory | `mixed_precision`: string; `fp8_base`, `fp8_scaled`, `fp8_vl`, `sdpa`, `gradient_checkpointing`: booleans; `blocks_to_swap`: integer | Existing precision, attention, checkpointing, and memory options, including their disabled forms. |
| Duration/randomness | `max_train_steps`, `gradient_accumulation_steps`, `seed`: integers | Optimizer-update count, accumulation, and seed semantics. |
| Optimization | `optimizer_type`, `lr_scheduler`: strings; `learning_rate`, `max_grad_norm`: floats; `lr_warmup_steps`: integer | Existing optimizer and schedule semantics, with only the approved example correction below. |
| Noise/loss | `timestep_sampling`, `weighting_scheme`: strings; `discrete_flow_shift`: float | Existing timestep selection, flow shift, and loss weighting. |
| Saving | `output_dir`, `output_name`, `save_precision`: strings; `save_every_n_steps`, `save_last_n_steps`, `save_last_n_steps_state`: integers; `save_state`: boolean | Existing checkpoint/state locations, formats, precision, schedule, and independent retention. |
| Logging | `logging_dir`, `log_with`: strings | Existing log destination and backend selection. |
| Samples | `sample_prompts`: string; `sample_every_n_steps`: integer; `sample_at_first`: boolean | Prompt-file reference, interval, and initial sampling behavior. |

The dataset template contains `[general]` keys `resolution` (two-integer array), `enable_bucket` and `bucket_no_upscale` (booleans), `caption_extension` (string), and `batch_size` and `num_repeats` (integers); each `[[datasets]]` entry includes `image_directory` and `cache_directory` (strings). Existing valid image-dataset variations remain supported; this example is not an exhaustive whitelist.

The text sample template currently contains two editable prompts using `--w`, `--h`, `--d`, and `--s` with integer values, and `--l` and `--fs` with numeric values. These retain dimensions, seed, sampling steps, classifier-free guidance, and flow-shift semantics. Existing applicable prompt features remain supported. `TOK`, scene text, prompt count, and example values are replaceable user content.

**Approved semantic correction — 2026-09-22**: The user selected changing `lr_scheduler` from `"constant"` to `"constant_with_warmup"` while retaining `lr_warmup_steps=200`, then explicitly authorized applying the correction immediately so it would not be lost. This one-value correction has been applied to `config_for_qwen_image_lora/train.toml` during specification. Preserve every other numerical setting and existing scheduler behavior. The incompatible `constant`/200 combination must still fail early rather than being reinterpreted. This approval does not start implementation of the feature.

**Authorized reference correction**: During implementation, correct `dataset_config` to `"config_for_qwen_image_lora/dataset.toml"` and `sample_prompts` to `"config_for_qwen_image_lora/sample_prompts.txt"` for invocation from the repository root. This does not authorize changing `output_dir`, `logging_dir`, `cache_directory`, or user-supplied external paths merely for naming consistency.

### Key Entities *(include if feature involves data)*

- **Effective workflow configuration**: The supported settings resolved from applicable defaults, user configuration files, and command-line overrides, validated before model loading.
- **Image/caption dataset**: One or more image collections with associated captions, preparation settings, and cache destinations.
- **Training caches**: Latent and text-embedding artifacts associated with dataset items and consumed by Qwen-Image original training.
- **Sample prompt set**: Editable scene descriptions and supported image-generation options used at configured training checkpoints.
- **LoRA checkpoint**: The saved adapter weights and metadata, with the configured save precision and retention rules.
- **Resumable training state**: The saved training context required by the existing resume mechanism, distinct from a LoRA checkpoint alone.
- **Training observations**: Existing logs, metrics, and sample images used to inspect progress without changing training behavior.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All eight retained workflow capabilities—image/caption preparation, latent caching, text-embedding caching, LoRA training, sample images during training, logging, saving, and resume—have traceable acceptance evidence from source review and applicable local checks, with no confirmed in-scope regression left unresolved. Evidence distinguishes checked contracts from unverified real-model execution.
- **SC-002**: All three contract templates, including all 40 training keys, every dataset entry, and every sample prompt option, agree with the retained readers and documented invocation paths after the approved corrections. There are zero unexplained key removals, numerical changes, or disabled functions.
- **SC-003**: Every exercised unknown-key, invalid-configuration, and excluded-mode case is rejected with an actionable error before model loading; none is silently ignored or converted into a supported mode.
- **SC-004**: Review finds zero retained implementations or executable training routes for excluded models and modes, and zero dependencies from the retained workflow on deleted modules or exclusively excluded functionality.
- **SC-005**: Every README command example matches a retained command and accepted arguments, and every referenced shipped configuration or document resolves. Users can follow one coherent sequence for the complete retained workflow.
- **SC-006**: All prescribed checks that can run locally pass; unavailable or unperformed checks are explicitly identified and never counted as passes. Local completion is not claimed with unresolved confirmed discrepancies or missing required local evidence, and is never represented as real-model operational verification.
- **SC-007**: Review confirms zero changes to the constitution or protected user data, no new framework or out-of-scope capability, and no unauthorized training-algorithm or numerical changes.

## Assumptions

- The constitution at `.specify/memory/constitution.md`, version 1.0.0, governs this feature unchanged. The explicit decision to remove other workflows satisfies principle III's incompatible-change approval requirement; no constitutional amendment is needed for that scope.
- Existing behavior and the supplied templates are the compatibility baseline. A setting is retained when it already applies to Qwen-Image original LoRA; parser exposure alone is not proof that it is supported for this model. The template is not the complete set of supported settings.
- Users provide compatible original-model weights and captioned images for eventual operational use. Example Windows paths and the H200 comment are not restrictions on supported paths or hardware.
- Relevant original-model computations may currently live in components shared with excluded models. Their required behavior is preserved while independent excluded capabilities are removed; this specification does not prescribe a replacement architecture.
- Local source inspection found all 40 training-template keys declared by existing parsers, two broken shipped-input references, and the scheduler/warmup incompatibility resolved by the user's decision above. This is declaration and source evidence, not a successful parser-to-training run.
- Local inspection also found that shared training configuration can retain unknown file keys and sample prompt parsing currently ignores unknown options after a warning. Strict early validation is required behavior to implement, not an existing property to claim as verified.
- No other confirmed semantic conflict in the supplied templates was found during specification. A later confirmed conflict remains subject to FR-012 rather than silent changes to the contract.
- The standard `specify -> plan -> tasks -> analyze -> implement -> converge` workflow and the constitution's approval rules and two-round limit for post-implementation corrections remain in force. Apart from the separately approved one-value configuration correction, this invocation stops after specification and its quality checklist.
