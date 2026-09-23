# Research: Validation Loss Integration

## Reuse the current Qwen objective

**Decision**: Run each Stage 1 check through `QwenImageNetworkTrainer.call_dit` and `NetworkTrainer.compute_loss`; add an optional exact-sigma override only for validation.

**Evidence**: `qwen_image_train_network.py:267-355` packs the noisy latent, sends `timesteps/1000` to the existing transformer under Accelerate autocast, and returns target `noise-latents`. `trainer_base.py:1160-1220` calls that forward and computes elementwise MSE, optional SD3 weighting, then `.mean()`. `training/timesteps.py:58-94` looks up scheduler sigmas and rounds timesteps that are absent from the schedule. Its weighting is sigma^-2 for `sigma_sqrt`, `2/(pi*(1-2*sigma+2*sigma^2))` for `cosmap`, and absent otherwise.

**Rationale**: The fixed validation level is already the actual mixing sigma. Passing `1000*t` to `call_dit` and `t` directly as the optional weighting sigma preserves the training target/reduction while avoiding a shifted/rounded training schedule. Keep the default `compute_loss` and weighting call unchanged when the override is absent; in the override branch only the sigma source changes. Gate `call_dit`'s checkpointing `requires_grad_` at lines 310-313 by `torch.is_grad_enabled()`.

**Alternatives considered**: Calling `process_batch` would draw training timesteps/noise; calling `get_sigmas` would round the fixed grid; a separate validation MSE would drift from weighted training loss. Each violates FR-004 or FR-012.

## Stream one image and use image-first arithmetic

**Decision**: Use the frozen Stage 1 item reader/check iterator and maintain per-image full/low/high scalar accumulators, then role-level accumulators. Use `math.fsum`, explicit counts, and finite checks at every level.

**Evidence**: `training/validation_inputs.py:207-295` preflights both roles and fingerprints their multisets; `:319-416` supplies an unshuffled batch-one loader, verifies frozen files on read, and draws each epsilon with a private CPU generator. The latent cache is `[C,1,H,W]`, so Qwen forward receives `[1,C,1,H,W]` and a single text embedding in `batch['vl_embed']`.

**Rationale**: Bucket/pixel pooling changes image weights. Scalar accumulation gives each declared occurrence one share and releases check tensors immediately. Stage 1 levels contain equal low/high halves, so the overall mean must match their average within tolerance. Values remain unpublished until all checks in both roles pass.

**Alternatives considered**: A prepared/distributed validation DataLoader can shard or pad; pooling all pixels or bucket means gives unequal image weights; buffering all check tensors wastes memory.

## Isolate model modes and training RNG

**Decision**: At each event snapshot all individual module `.training` flags and Python, NumPy, torch CPU and available CUDA RNG states, run main-rank forward in `torch.no_grad()`, and restore snapshots in `finally`. Do not call optimizer evaluation hooks.

**Evidence**: `networks/lora.py:139-161` uses `.training` for LoRA dropout; `:685-711` patches the transformer's forward and registers LoRA modules on the separate network. `trainer_base.py:2047-2051,2143-2165,2197-2213` uses `optimizer_eval_fn`/`optimizer_train_fn` for sampling and saving. `qwen_image_model.py:1098` tests `torch.is_grad_enabled()` for checkpointing.

**Rationale**: The transformer and LoRA tree can start with different flags, and schedule-free optimizer hooks may change weights/state. Root-level `.train()` restoration alone would overwrite mixed child modes. The Stage 1 local generator is isolated, but a forward may still consume global RNG; snapshots make success and exceptions equivalent to no event for subsequent training.

**Alternatives considered**: Only setting the transformer to eval leaves LoRA dropout active; indiscriminate `.train()` on exit loses prior flags; optimizer hooks are not state-neutral.

## Count completed updates separately from run budget

**Decision**: Preserve run-local `global_step` for `process_batch` and existing hooks. Enabled mode maintains `absolute_completed_step = resume_start_step + run_completed_steps`; only `accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped` advances the completed count. Use the absolute count for validation and enabled step logs/state identity. Extend the existing epoch iteration in enabled mode until the requested completed-update budget is met despite skipped attempts.

**Evidence**: `trainer_base.py:1964-1970` initializes `global_step=0` even after `accelerator.load_state`; `:2082-2141` calls optimizer/scheduler under accumulation and increments on `sync_gradients`, which can still represent an AMP-skipped update. Accelerate 1.6.0 exposes `optimizer_step_was_skipped` in `accelerator.py:3747-3755`. `trainer_base.py:2167-2183` logs on every microbatch with the current `global_step`.

**Rationale**: The restored Accelerate `step` counts microbatch progress, not proven successful optimizer updates. A separate persisted counter prevents a false resume start of zero. In enabled mode, publish existing step-loss tags only after a completed update at its absolute step; publish `loss/epoch` at that absolute step only if the epoch completed an update. Otherwise a resumed accumulation microbatch or skipped-only epoch would create a training loss point before any new update. Use the absolute count consistently for enabled step-triggered output names/cadence/retention; leave disabled calls and counters in their present branch.

**Alternatives considered**: Replacing `global_step` passed into training hooks with an absolute count could alter step-dependent training algorithms; inferring count from optimizer parameters/scheduler would be untrustworthy; logging each microbatch at `s` violates FR-015.

## Coordinate one-rank and two-rank evaluation

**Decision**: All ranks agree on the boundary and enter it; main rank evaluates the full manifest using `accelerator.unwrap_model(transformer)` and logs once. Broadcast a success/error status before ranks leave the boundary.

**Evidence**: `trainer_base.py:1759-1776` prepares the transformer and network; the prepared transformer can be DDP wrapped. The Qwen transformer forward at `qwen_image_model.py:1019-1140` and the LoRA forward at `networks/lora.py:134-195` contain no explicit distributed collective. Stage 1 validation loader is deliberately not prepared/sharded.

**Rationale**: Bypassing DDP forward avoids its buffer broadcast when other ranks wait. A rank-wide status exchange prevents a main-rank exception from stranding peers at a later barrier. A two-rank CPU test must confirm normal and failing boundaries, because compile/block-swap wrappers and backends need operational verification.

**Alternatives considered**: Evaluating every rank duplicates items; sharding may pad unequal partitions and change counts; using wrapped DDP on main alone risks a hang.

## Persist a trustworthy resume timeline

**Decision**: Put versioned `val_loss_state.json` in each enabled Accelerate state directory with an explicit nonnegative absolute completed count, model version, Stage 1 fingerprint, and effective validation controls. Write atomically after all ranks finish saving, before upload/retention; validate collectively from the actual local/downloaded directory before enabled resume results.

**Evidence**: `_register_hooks_and_resume` at `trainer_base.py:1780-1809` registers load hooks before `accelerator.load_state`; local and Hugging Face resume use `accelerator.load_state` at `:476-528`. Current step/epoch/final save calls at `:2154-2159,2200-2210,2223-2227` are main-only, while `utils/train_utils.py:119-180` delegates to `accelerator.save_state`. Accelerate 1.6.0 `checkpointing.py:150-168` saves `random_states_{process_index}.pkl` for each calling rank, including an internal microbatch `step` that is not the required absolute optimizer count.

**Rationale**: All-rank state saves are needed for rank-local RNG restoration. The main rank performs sidecar write, upload and retention only after a barrier, preventing a remote/incomplete state from masquerading as complete. A missing or changed fingerprint fails enabled resume; disabled legacy resume remains untouched. Loading network weights alone does not invoke `load_state` and begins at zero.

**Alternatives considered**: Reusing `accelerator.step`, a scheduler index or one parameter's optimizer step cannot prove completed updates; a separate output database is unnecessary; a pre-save sidecar could remain after an interrupted state write.

## Tracker integration and verification limit

**Decision**: Continue using the existing Accelerate trackers and `accelerator.log`, with a complete six-key dict and the absolute event step. Verify actual TensorBoard event tags/steps with CPU fixtures.

**Evidence**: `training/accelerator_setup.py:49-117` builds the Accelerator logging setup; `trainer_base.py:1951-1960` initializes trackers on main. `trainer_base.py:2167-2183` uses `generate_step_logs` and `accelerator.log` for training. `accelerator.end_training()` occurs at `:2223`, so the final validation event must precede it.

**Rationale**: A second writer can duplicate or misalign events. Controlled CPU tests can verify mathematics, scheduling and state isolation; they cannot prove GPU Qwen throughput or backend numerics, which belong to later server verification.

**Alternatives considered**: New TensorBoard writer or real-model local training expands scope and breaks the project's local-stage rule.
