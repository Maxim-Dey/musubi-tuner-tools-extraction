# Feature Specification: Deterministic Krea2 Raw validation loss with noise bands

**Feature Branch**: `codex/004-krea2-deterministic-val`
**Created**: 2026-10-27
**Status**: Ready for planning
**Input**: `config_for_krea2_raw_lora/speckit_val_loss_commands.md`, commands 1–10, with workflow commands 11–12. That complete source is normative; the requirements below index its acceptance obligations, rather than replacing detailed prescribed configuration values or diagnostics.

## User Scenarios & Testing

### User Story 1 - Prepare a fixed independent experiment (Priority: P1)
The operator starts an experiment from any working directory, with distinct training and validation data and compatible reusable caches.
**Why this priority**: A comparison is meaningful only if its inputs are independent and fixed.
**Independent Test**: Synthetic files exercise effective configuration, content identity, leakage, cache reuse and manifest rejection before any model loads.
**Acceptance Scenarios**:
1. Given a complete explicit validation configuration, when preflight succeeds, then a nonempty unique independent validation set and its fixed inputs are established.
2. Given partial settings, empty data, leakage or incompatible cache, when starting or resuming, then an actionable error identifies source, cause and correction before model loading.
3. Given absent compatible caches and automatic caching enabled, when preparing, then only missing caches are generated before DiT loading and compatible caches are reused.

### User Story 2 - Compare reproducible noise-band losses (Priority: P1)
The operator compares states using one mean and two noise-band means on the same validation set, without changing training.
**Why this priority**: These are the feature's primary measurements.
**Independent Test**: Known tensors and a separate reference calculation prove exact levels/seeds, weighting, equal image contribution and isolation; real Krea2 checks follow in group F.
**Acceptance Scenarios**:
1. Given identical weights and fixed inputs, when validation repeats or input order changes, then seeds/noise match exactly and the three forward metrics match within declared dtype-based tolerances.
2. Given different image sizes or uneven process partitions, when reducing, then each unique image contributes once with equal weight, including an empty rank.
3. Given successful or failing validation, when training continues, then its RNG, modes, gradients, optimizer and scheduler are preserved.

### User Story 3 - Continue one coherent training trajectory (Priority: P1)
The operator sees validation at completed optimizer steps and resumes complete experiment states with one adapter and correctly associated samples.
**Why this priority**: Comparisons and continuation require consistent state.
**Independent Test**: CPU fixtures verify schedules, accumulation, save hooks, retention and exact resume position; actual GPU continuous/resumed trajectories are compared in F.
**Acceptance Scenarios**:
1. Given period 2 and final step 5, when training, then validation occurs at 0,2,4,5 once each, regardless of sampling schedule.
2. Given a completed checkpoint at step 2, when resuming to 5, then step 2 is not relogged and continuation uses the restored optimizer, scheduler, RNG and data position.
3. Given experiment state saving, when saving a step, then its complete folder contains one FP32 independently loadable adapter and samples only from that state; retention removes complete obsolete folders.

### User Story 4 - Run the supplied experiment (Priority: P2)
The operator receives actual full-data configurations and a remotely verified environment, caches, diagnostics and TensorBoard, then launches full training independently.
**Why this priority**: Local unit checks cannot establish real-model readiness.
**Independent Test**: Actual parsers and inventory locally; supplied Pod logs, adapter tensors, event files and HTTP responses remotely.
**Acceptance Scenarios**:
1. Given the existing supplied split and prompts, when producing configurations, then all original samples/captions and ten subjects remain, with prescribed settings and separate train/val caches.
2. Given local acceptance, when performing F, then the implemented working version, models, environment and complete data are prepared and real short training/resume diagnostics produce evidence.
3. Given unavailable infrastructure or multi-GPU capacity, when reporting, then corresponding criteria remain explicitly unverified, never passed.

### Edge Cases
Partial validation groups; boolean numeric values; duplicate bytes under different names; conflicting duplicate captions/preprocessing; equal stems; relocated experiments; legacy caches without provenance; changed active validation inputs; empty distributed rank; exception during block swap; accumulation boundaries; matching final/periodic saves; older checkpoint with later events; interrupted checkpoint write; unavailable Pod/model/GPU/external TensorBoard address.

## Requirements

### Functional Requirements
- **FR-001 (A)**: Preserve existing commands/defaults/paths/formats and training mathematics when new modes are disabled. Scope is Krea2 Raw image-caption LoRA, Qwen-Image VAE, Qwen3-VL-4B-Instruct and networks.lora_krea2 only; excluded modes and parameters in source command 1 MUST NOT be introduced.
- **FR-002 (A)**: Validate effective defaults/TOML/CLI while rejecting unknown keys. experiment_mode and auto_cache default false. Validation is disabled only if all five val settings are absent; otherwise require every setting explicitly. Period is integer >=1, seeds integers, N1 even integer >=2, N2 integer >=1; booleans fail integer validation.
- **FR-003 (A)**: Experiment main paths resolve from main TOML, dataset paths from their respective TOMLs; preserve absolute paths and repository IDs, URLs and module names. Arbitrary directory names and other cwd work; ordinary mode retains old rules.
- **FR-004 (A)**: Before models, check configs, nonempty train/val, required captions, validation batch_size=1 and num_repeats=1, accessible cache metadata and content leakage. SHA-256 original image bytes define identity. Stable compatible val copies count once; conflicting captions/preprocessing fail. No random validation augmentations/caption changes. Train batch/repeats remain independent.
- **FR-005 (B)**: Use separate compatible Krea2 train/val caches and existing cache tools. auto_cache generates missing files only, synchronizes distributed writers and frees VAE/encoder before DiT. Missing caches with auto_cache=false and incompatible existing caches fail. Preserve source command 6 filename, architecture, tensor, normalization, shape and dtype contracts and detect stem collisions before writes.
- **FR-006 (B)**: Fix validation images, captions, preprocessing, buckets, caches, noise parameters/generation representation and model identities in a relocation-independent manifest. Reject drift before resume and on each validation. Unknown legacy provenance is insufficient compatibility evidence; never silently replace an incompatible manifest/cache.
- **FR-007 (C)**: For each unique image evaluate N1*N2 combinations with t_i=0.05+(i-0.5)*0.90/N1, i=1..N1, j=1..N2. Low t<0.5, high t>=0.5. SHA-256 stable unambiguous seed serialization includes only general seed, image ID, i,j; document digest conversion and fixed test vector. Generate standard normal noise using an isolated generator with fixed dtype/device across resume.
- **FR-008 (C)**: Use x_t=(1-t)*latents+t*noise, target=noise-latents and Krea2 call_dit with timesteps=1000*t, cached normalized latents and Krea2 Qwen3-VL context/positions/masks. No repeated normalization, train sampler, shift, rounding, +1, min/max restriction, CFG or image sampling. Apply existing weighting_scheme meaning at fixed t without changing training loss.
- **FR-009 (C)**: Average MSE over latent elements, combinations per image and then equally over unique images. Reduce sums/counts across unpadded partitions, including empty ranks; publish one global mean/low/high triple on main process. Equal bands imply mean=(low+high)/2 within declared tolerance.
- **FR-010 (C)**: Validation uses eval without dropout/autograd or gradient-checkpoint requires_grad. Preserve weights, existing gradients, optimizer, scheduler, Python/NumPy/Torch CPU/CUDA RNG, module modes, gradient checkpointing and block swap, including exceptions. Do not duplicate weights or retain all GPU results.
- **FR-011 (D)**: Validate before first update at step 0, each completed optimizer-step period and unduplicated final step. Resume restores S and validates only if that exact state lacks persisted validation; sampling flags do not control validation. Log val_loss_mean, val_loss_low_noise, val_loss_high_noise with optimizer-step X axis, main process only; avoid incompatible trajectories when resuming an older state with later events.
- **FR-012 (D)**: Resume restores adapter, optimizer, scheduler, per-process RNG, completed optimizer steps, exact train position, manifest and completed validation/metrics. accelerator.step is not global_step. Order validation then samples/save then consistent metadata; network_weights is adapter loading only.
- **FR-013 (D)**: Only experiment_mode with save_state uses <output_name>-stepNNNNNN folders, one FP32 model.safetensors, native optimizer/scheduler/RNG state, position/validation/manifest metadata and same-step samples. Reuse Accelerate hooks without duplicate LoRA state. Explicit incompatible save_precision fails. Separately loading the adapter and full resume both work. Final actual step saves once; incomplete writes never appear complete.
- **FR-014 (D)**: Retention is an optimizer-step window: save_last_n_steps_state takes precedence over save_last_n_steps; neither means retain all. Preserve existing boundary semantics and remove only complete obsolete current-experiment folders including samples. Unequal sample/save intervals and sample_at_first use separate explicitly numbered sample output; never attach to another state.
- **FR-015 (E)**: Generate actual train.toml, train-dataset.toml, val-dataset.toml, ten preserved sample prompts and updated README/Krea2 docs using every exact target value and prohibition in source command 9. Preserve legacy dataset.toml. Confirm actual expected 107 train/20 val paired images without altering split/names/captions; diagnostics are separate. Parse schema and paths through real tools including other cwd. Keep tensorboard available using the project's dependency mechanism.
- **FR-016 (E/F)**: Main configs retain all requested settings and no active REPLACE_WITH_*, diagnostic limits or diagnostic resume. Only F substitutes verified real model paths; unavailable model facts are recorded, never invented. Memory-driven main changes require short diagnostic evidence and explicit effective-batch reporting.
- **FR-017 (F)**: After local acceptance use the supplied SSH endpoints and key locally, preserve user data and Pod, inspect existing workspace/models/processes/GPU, transfer actual working-tree implementation and intact complete split, prepare compatible environment/models/tokenizer through current hf and primary sources, and full distinct caches. Record source/version/path and integrity; serialize GPU/environment ownership, never expose credentials or rename source model files.
- **FR-018 (F)**: Execute source command 10's real GPU diagnostics: batch1/accumulation2/N1=2/N2=1/period2/final5; separate N1=10/N2=2; repeated val, reference aggregation, continuous vs controlled checkpoint2 resume with identical target5, val/no-val trajectory, FP32 single adapter/reload, retention/samples, gradient checkpointing/block swap and actually used quantization. Read actual TensorBoard events. Multi-GPU checks run only if available; unavailable capacity remains unverified.
- **FR-019 (F)**: Preserve full prepared environment/data/caches/main config, stop only created diagnostics, leave GPU free, keep TensorBoard on 0.0.0.0:6006 after SSH closes, verify local HTTP and actual external address. Record exact cwd, activation, new-training/resume/TensorBoard commands; absent user state must be reported instead of selecting diagnostic state. Do not launch full training or stop Pod.
- **FR-020 (workflow)**: One feature directory; English artifacts/Russian dialogue; preserve user changes/constitution. A→B→C→D→E→F; independent evidence review, exclusive file ownership, sequential trainer integration. No Pod/models/training before F. Analyze read-only, converge append-only. All initial A–F remain tracked; after initial implementation globally at most two confirmed-defect fix→verify rounds in verification.md. Never mark unavailable or failed checks complete or require human review.

### Key Entities
Experiment configuration; unique image/content ID; cache provenance; fixed validation manifest; noise combination; per-image/global metrics; completed optimizer step and train cursor; complete unified checkpoint; verification evidence and correction round.

## Success Criteria
### Measurable Outcomes
- **SC-001**: All prescribed invalid configurations/data/cache mutations fail before model loading; compatible duplicate validation copies count exactly once.
- **SC-002**: Exact expected seeds/noise and midpoint levels pass reference checks, all three image-balanced means match the independent reference and repeatability tolerance.
- **SC-003**: Validation leaves training state unchanged on success and exception, and continuous/resumed and val/no-val trajectories agree within predeclared justified tolerances.
- **SC-004**: Period2/final5 produces exactly steps 0,2,4,5; resuming completed step2 adds only 4,5; every saved state has one independently loadable FP32 adapter with correct retention/samples.
- **SC-005**: Main configurations parse from another cwd with the complete supplied split and all ten prompts, using exact requested settings.
- **SC-006**: Required actual GPU evidence, ready full-data experiment and working externally verified monitoring are recorded; outstanding external criteria remain explicitly incomplete.

## Assumptions
- The source document and supplied configurations define the detailed authoritative targets, including operational diagnostics; no new model family or general experiment framework is needed.
- Remote availability and actual model file paths are facts to establish in F, not assumptions of success. Real multi-GPU verification depends on supplied capacity and cannot be replaced by CPU aggregation evidence.
