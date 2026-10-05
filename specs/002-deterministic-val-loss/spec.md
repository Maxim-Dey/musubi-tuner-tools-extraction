# Feature Specification: Deterministic validation loss and noise-level breakdown

**Feature Branch**: `codex/deterministic-val-loss`
**Created**: 2026-10-05
**Status**: Complete for the user-authorized scope on 2026-10-06; exact GPU isolation verified in the recorded deterministic environment
**Input**: Commands 0–12 in `config_for_qwen_image_lora/speckit_val_loss_codex_prompts.md`, explicitly requested as an autonomous multi-agent implementation including RunPod acceptance.

## User Scenarios & Testing

### A — Fixed validation inputs and loss (Priority: P1)

The trainer owner supplies independent train and val images and measures comparable validation losses without changing training.
**Independent test**: Small CPU tensors, real readers and separate processes establish deterministic identity/noise, fixed timesteps, equal image weighting and strict input errors.
**Acceptance scenarios**:
1. Given nonempty disjoint inputs and complete immutable caches, every image receives exactly N1*N2 forwards. N1=2 produces 0.275,0.725; N1=10 produces 0.095,0.185,0.275,0.365,0.455,0.545,0.635,0.725,0.815,0.905.
2. Given renamed/reordered/moved inputs, identical source bytes retain seeds and noise; changes of level/realization produce distinct seeds.
3. Given unequal image resolutions and multiple datasets, the three metrics weight images equally and mean=(low+high)/2.
4. Empty/overlapping inputs, wrong settings, unknown keys and missing/corrupt/stale caches fail before loading the model.

### B — Schedule, isolation and resumed plots (Priority: P1)

The owner receives validation at actual completed updates and continues the same measurement series after resuming.
**Independent test**: Execute the real trainer loop with small CPU model boundaries, real optimizer/Accelerate and actual TensorBoard events.
**Acceptance scenarios**:
1. Target=5, interval=2 yields 0,2,4,5; target=4 yields 0,2,4 once. Accumulation=2 does not change the optimizer-step schedule; overflow skips do not advance it.
2. Resume at S=3 to target=5 yields baseline 3 if missing, then 4,5. Existing measurements at loaded weights/inputs are not duplicated, and events beyond a rollback checkpoint are discarded from the continued trajectory.
3. Resume S>=target performs zero additional updates. Train loss has no artificial point at S.
4. With identical initial states/batches, enabling val leaves RNG, gradients, weights, optimizer/scheduler and the next training update unchanged, including dropout and evaluation exceptions.
5. Multiple processes count every image/level/realization once, synchronize without hanging and use one event writer.

### C — One experiment and one checkpoint (Priority: P1)

The owner moves an experiment folder or invokes commands from another working directory and can resume from one self-contained step directory.
**Independent test**: Actual CPU Accelerate save/load, tensor artifact inventory, path resolution and retention boundaries.
**Acceptance scenarios**:
1. All three entry points resolve explicit experiment paths consistently. Legacy paths retain their old semantics.
2. A complete step directory has one usable FP32 LoRA copy plus actual supported optimizer/scheduler/RNG files and trainer metadata, with no base model tensors; load restores these and subsequent training works.
3. Periodic/final saves at the same step deduplicate. Samples use their actual generation step; samples-only or incomplete directories cannot resume.
4. At current=100, interval=10, weight window=40, state window=20, weights 60..100 and states 80..100 remain. With state window=60 weights 40..100 remain. Missing steps are never invented. Other output names are unaffected.

### D — Working configuration and local acceptance (Priority: P1)

The owner receives supported TOML files and verified commands without losing prompts or data.
**Independent test**: Parse examples with real entry points, compare effective settings, run CPU integration tests and review documentation.
**Acceptance scenarios**: Preserve every numeric/model value in command 9; provide four cache commands, train from two working directories, resume and TensorBoard. Separate short GPU configurations from the 5000-step profile. Missing local `/workspace` resources are explicitly unverified.

### E — Real-model remote acceptance (Priority: P1)

The owner receives evidence from the supplied RunPod after the local gate passes.
**Independent test**: One GPU-process owner transfers the tested tree, uses independent real val data and runs cache/train/val/save/resume and paired isolation scenarios.
**Acceptance scenarios**: Record finite real-model losses, a complete N1=10,N2=2 pass, repeated evaluation, exact event steps/tags, real checkpoint inventory/load/continued training, samples/retention and effective user config. Multi-GPU execution and the batch16/1024 profile run are excluded by the user on 2026-10-06; they are not completion blockers. Unavailable SSH/models/val are blockers, never passes.

### Edge Cases

Boolean numeric settings; odd/zero level counts; empty effective cache datasets; duplicate names with shared cache destinations; duplicate source contents; changed captions/caches after resume; random VAE encoding; accumulation overflow; resumed step at/above target; an event written without updated checkpoint metadata or vice versa; rollback past later events; state window larger than weight window; final checkpoint between periods; validation exceptions; nonzero block swap and gradient checkpointing; unavailable server or independent val data.

## Requirements

### Functional Requirements

- **FR-001**: Preserve Qwen-Image original LoRA training and its loss. `val_dataset_config` enables val; `experiment_mode=true` independently opts into experiment layout (default false). Both off preserve existing commands/defaults/layout/state loading. No new trainer, other architectures or automatic train/val split.
- **FR-002**: Reject unknown parameters and explicitly supplied val settings without val data. Validate effective defaults/TOML/CLI before loading weights; explicit CLI wins. Errors name the source, setting, cause and correction.
- **FR-003**: Require nonempty effective train and val sets, disjoint by SHA-256 of source image bytes. Train contributes no validation metric. Multiple val datasets form one image population.
- **FR-004**: Fix val images, captions, geometry, latent and text caches. Require batch_size=num_repeats=1; reject random augmentation, caption dropout/shuffle and random repeated encoding. Deterministic fixed resolution/bucketing is allowed. Never silently coerce or skip inputs.
- **FR-005**: Image identity is source-byte SHA-256, independent of paths, workers, ranks and order. Record a sufficient fingerprint of sources, captions, configuration and both caches; changed or missing/corrupt caches cannot silently resume the same measurement series or regenerate during training. Detect cache-name collisions.
- **FR-006**: `val_every_n_steps` is integer >=1, `val_seed_noise` integer, `val_level_noise_n=N1` even integer >=2, `val_seed_noise_n=N2` integer >=1; bool is not integer. Defaults when val is enabled are 50,42,10,2 respectively.
- **FR-007**: For each image, evaluate exactly N1*N2 pairs, i=1..N1, j=1..N2. Set t_i=0.05+(i-0.5)*0.90/N1. Low is 0.05<=t<0.5, high 0.5<=t<=0.95, each N1/2 levels.
- **FR-008**: t is final mixing coefficient: x_t=(1-t)*latent+t*noise; trainer receives exactly 1000*t. Never apply training sampling/shift, scheduler grid rounding or +1. Weighting uses this same t as sigma.
- **FR-009**: Seed is SHA-256 of compact ASCII JSON `[val_seed_noise,lowercase_image_sha256_hex,i,j]`, no whitespace; first eight digest bytes unsigned big-endian modulo 2**63. No step/epoch/path/rank/order input. Fix the byte contract and known vectors in plan/tests.
- **FR-010**: Use a separate noise generator. Isolate the complete validation path, loader/setup/evaluation included, preserving Python/NumPy/torch CPU and all used CUDA RNG states.
- **FR-011**: Share the actual training latent transform, forward, target and weighted loss/reduction. Define L[k,i,j] as scalar image loss. mean=(1/M)*sum_k((1/(N1*N2))*sum_i,j L[k,i,j]); low/high restrict i and divide by (N1/2)*N2. Equal image weight regardless of pixels/bucket/rank; mean=(low+high)/2 within numerical tolerance.
- **FR-012**: Evaluate before first update at 0, after each completed update divisible by interval and at final completed step if not already measured. Accumulation microbatches/overflow skips do not count. Evaluate before consuming the next train batch.
- **FR-013**: Evaluate without autograd/backward/optimizer or scheduler update; disable base-model and adapter dropout. Restore original per-module modes, RNG and temporary settings in finally, also on error. Parameters, gradients and subsequent train RNG/batches/updates remain unchanged.
- **FR-014**: Reuse loaded model/adapter, stream one noise at a time, retain no graphs or duplicate DiT. Multi-process execution counts each image/level/realization once and writes metrics on one process without padded duplicate samples or deadlocks.
- **FR-015**: Add exactly `val_loss_mean`, `val_loss_low_noise`, `val_loss_high_noise`. Preserve existing train-loss names and values. New-mode train/val x-axis is completed optimizer updates.
- **FR-016**: Explicitly persist and restore `optimizer_global_step`, independent of Accelerator.step and filenames. At resume S evaluate S only if same weights/inputs have no persisted event; never create synthetic train loss. Next periodic is S+(interval-S%interval).
- **FR-017**: With val or experiment mode, max_train_steps is target total update count; S>=target makes zero updates. Continue same TensorBoard run, reconcile durable events with checkpoint metadata and suppress abandoned post-checkpoint trajectory on rollback.
- **FR-018**: Restore LoRA/optimizer/scheduler/RNG. Legacy states remain loadable in legacy mode; new modes reject unknown resume step with explanation. Exact train-loader position beyond existing capability is excluded; do not claim unverified bitwise training resume.
- **FR-019**: In experiment mode root is main train.toml directory; resolve its relative paths (including resume) there and dataset paths against each dataset TOML. Preserve absolute paths. Both cache commands explicitly support the same semantics; legacy paths remain cwd-relative.
- **FR-020**: Supply train.toml, train-dataset.toml, val-dataset.toml, sample_prompts.txt, dataset/train,val, cache/train,val and output/tensorboard. Separate train and val caches. Step output is output/<output_name>-step<optimizer_step>/ with samples/ and canonical model.safetensors.
- **FR-021**: One model.safetensors is usable as adapter and exact trainable-weight resume via supported Accelerate hooks. State saves require FP32 lossless adapter weights and reject lossy configurations early. Save actual optimizer/scheduler/RNG files plus metadata; never duplicate LoRA or base DiT tensors.
- **FR-022**: Save final/periodic step only once. Samples retain true step with mismatched sample/save intervals. Only complete checkpoints are resumable; samples-only/partial directories are not states. Save RNG for all ranks.
- **FR-023**: Retention uses completed-step windows, not file counts. State window absent/zero inherits weight window; absent weight window is unlimited. Preserve LoRA as long as either state or weight retention requires it. Remove only matching output-name artifacts in this experiment; preserve samples as appropriate and never invent missing steps.
- **FR-024**: Preserve user prompts/data/caches/results and command 9 profile exactly, adding required supported settings; use existing README.ru.md and repair relevant links without restoring deleted README.md. Document four real cache commands, training from root/other cwd, full-checkpoint resume and TensorBoard.
- **FR-025**: Supply separate short acceptance configs with accumulation>1 and dropout>0; verify real readers/schema/CLI and CPU regressions. Record requirement→implementation→check→result in validation.md; distinguish baseline failures and absent local model resources.
- **FR-026**: After local A–D acceptance transfer precisely the implemented working tree to provided RunPod using existing SSH key without logging secrets, isolated checkout/environment/output, dependencies compatible with pyproject and actual GPU. Preserve remote source resources; do not create/shutdown Pods or split train data. One agent controls all GPU work sequentially.
- **FR-027**: Remote acceptance covers both cache stages for train/val, real forward and finite losses, full N1=10,N2=2 pass, repeatability with predeclared tolerance, paired val on/off next-update isolation, canonical FP32 artifacts/resume, programmatically read event tags/steps/deduplication, samples/retention, user effective config. Multi-GPU execution and batch16/1024 execution are excluded by the user; retain existing configuration and CPU distributed coverage. Record environment/code identity/config/commands/expected and actual events/artifacts in gpu-validation.md.
- **FR-028**: Complete commands in order with independent reviewers, English artifacts and Russian user messages. Converge is append-only tasks; analyze is read-only. After initial implementation allow at most two recorded fix→verify rounds total across local/remote stages; stop autonomous corrections at the limit and report outstanding defects. The explicit 2026-10-06 user continuation authorizes T032 correction and the remaining single-GPU acceptance beyond that exhausted allowance; record it as round3 without resetting prior rounds. Never mark blocked/not-run checks passed.

### Key Entities

- Validation image: immutable source identity, caption, geometry and corresponding latent/text cache fingerprints.
- Validation measurement: image/level/realization scalar loss, aggregated into exactly three metrics at an optimizer step.
- Experiment: input/config/cache/output root plus opt-in path semantics.
- Checkpoint: canonical adapter, supported optimizer/scheduler/RNG state, explicit completed-step and val/run metadata, completeness marker.
- Acceptance evidence: requirements, tested code/environment and factual outcomes, including correction-round counter.

## Success Criteria

- **SC-001**: Every val image contributes exactly N1*N2 equally weighted measurements and the specified mean/low/high identity holds.
- **SC-002**: Repeating measurements in the same environment preserves noise exactly and losses within a predeclared numerical tolerance, while the next identical training update remains unaffected.
- **SC-003**: New and resumed runs have exactly the expected measurement steps and one unambiguous continued history.
- **SC-004**: A movable experiment can resume losslessly from a complete step directory containing one adapter copy, with correct retention boundaries.
- **SC-005**: Configurations and documented commands pass local checks and required real-model remote scenarios, with user-excluded multi-GPU and batch16/1024 execution explicitly separated from the required acceptance.

## Assumptions

- User explicitly authorizes the complete command series, remote work and autonomous corrections within the original two-round cap. On 2026-10-06 the user explicitly requests completion of T032, single-GPU repeatability/isolation and final evidence, superseding the prior stop for this continuation. The constitution file remains unchanged; this feature-specific user authorization is recorded openly.
- The supplied command 9 profile is mandatory. Technical details outside the user contract are decided from repository evidence in plan.md.
- Numerical portability across GPU models, framework versions and attention kernels is not promised. Independent val data and model availability must be observed remotely.

## User-authorized continuation — 2026-10-06

The user explicitly requests points1,2,4: fix T032, prove GPU repeatability/isolation and finish evidence/tasks within SpecKit. Multi-GPU and batch16/1024 runs are excluded; the user will test the latter separately. This is an explicit continuation after the original two autonomous correction rounds, not a reset of their counter. Preserve all numerical contracts, tolerances, production compatibility and prior failed evidence.
