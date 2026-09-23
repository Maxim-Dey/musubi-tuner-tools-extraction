# Implementation Plan: Deterministic Validation Loss During Training

**Branch**: `001-scope-qwen-image-lora` (current checkout; feature directory `003-val-loss-training`) | **Date**: 2026-09-23 | **Spec**: [spec.md](spec.md)

**Input**: Stage 2.1 specification and the completed [Stage 1 input contract](../002-val-loss-core/contracts/validation-inputs.md).

## Summary

Connect the frozen Stage 1 validation manifest to the existing Qwen-Image original LoRA trainer. Stream one image and one fixed noise check at a time through the current transformer and loss method, aggregate image-first means, and publish six values only after both sets finish. Schedule events by a persisted count of completed optimizer updates. Keep the disabled training path unchanged.

## Technical Context

**Language/Version**: Python `>=3.10,<3.13`; local CPU checks use Python 3.12

**Primary Dependencies**: Existing PyTorch, Accelerate 1.6.0, NumPy, safetensors, TensorBoard tracker, and Stage 1 `validation_inputs`

**Storage**: Existing Accelerate state directories plus one versioned JSON sidecar per enabled state; existing event logs and model checkpoints

**Testing**: pytest with small CPU models and fixtures, TensorBoard event reads, one-rank and two-rank CPU Accelerate state round trips; no model downloads or GPU execution locally

**Target Platform**: Existing Windows/Linux Qwen-Image original LoRA CLI and one-rank/two-rank Accelerate workflows

**Project Type**: Python CLI training application

**Performance Goals**: No new benchmark target; one existing model, one image/check in memory at a time, no padding or duplicate validation work

**Constraints**: Preserve training forward, objective, precision, update semantics, and disabled behavior; validation cannot consume training RNG or mutate training state; exact `sigma=t` must bypass training scheduler lookup

**Scale/Scope**: Two Stage 1 roles, `N1*N2` checks per declared occurrence, six scalars per completed event; no best-state policy or new output hierarchy

## Constitution Check

*Gate evaluated before research and again after Phase 1 design.*

| Principle | Design check | Status |
| --- | --- | --- |
| I. Language | Feature artifacts are in English; user updates remain Russian. | Pass |
| II. Task Fidelity | Design covers FR-001–FR-019 and leaves best-state/output hierarchy for Stage 3. | Pass |
| III. Minimal, Compatible Changes | Reuse the current loop, `call_dit`, `compute_loss`, Stage 1 checks, Accelerate state/logging; enabled-only changes have concrete needs. | Pass |
| IV. Training Invariants | No validation backward/update or optimizer evaluation hook; exact mode/RNG restoration and unchanged default training path. | Pass |
| V. Memory and Performance | Stream checks and Python scalar accumulators; no model copy or retained per-check tensors. | Pass |
| VI. Configuration and Errors | Stage 1 preflight still precedes weights; resume identity, sidecar schema, nonfinite and read errors fail with correction. | Pass |
| VII. Verification and Documentation | Controlled CPU, tracker, and two-rank checks; review `README.ru.md` and `docs/qwen_image.md` during implementation. | Pass |
| Local Stage / Workflow | Planning changes documentation only; no local training, GPU run, transfer, or model download. | Pass |

No constitution exception or unresolved clarification is needed. The post-design check also passes: the contract uses the existing training objective and state directories, and disabled mode retains its current calls and log timing.

## Design

### Existing forward and exact weighted loss

Stage 1 already preflights a `ValidationManifest` before `_load_dit_and_swap` and yields exact cache pairs and deterministic checks. In an enabled event, read one item, move its latent and text embedding to the existing accelerator device, and iterate its checks without retaining outputs. Form a batch of one (`latents`, one `vl_embed`, `epsilon`, `noisy_latent`, and a one-element `1000*t` timestep) and call `QwenImageNetworkTrainer.call_dit` on the **unwrapped existing transformer**. This retains packing, autocast, target `epsilon-latent`, and model timestep division by 1000. Guard its checkpointing input `requires_grad_` branch with `torch.is_grad_enabled()` so validation under `torch.no_grad()` cannot enable input gradients; training behavior is unchanged.

Call the inherited `NetworkTrainer.compute_loss` for the scalar. Add an optional exact-sigma argument to that method and to `compute_loss_weighting_for_sd3`: absent means its existing `get_sigmas` path and arithmetic are unchanged; present uses a `[1,1,1,1,1]` tensor containing the actual Stage 1 `t` and the same `sigma_sqrt`/`cosmap` formulas. Do not invoke the training timestep sampler, `get_sigmas` for validation, flow shift, or a second loss reducer. Require each detached scalar to be finite. Details are in [research.md](research.md) and [contracts/validation-training.md](contracts/validation-training.md).

### Event and state isolation

Introduce a small validation event method on the Qwen trainer, called by the existing loop at enabled boundaries. At entry snapshot Python, NumPy, torch CPU and available CUDA RNG states; snapshot every individual `.training` flag in the unwrapped transformer and LoRA module trees, deduplicating modules by identity. Within `torch.no_grad()`, put both trees in eval mode without touching weights or optimizer hooks. In `finally`, restore every original module flag individually and all RNG snapshots, including after a cache read, forward, or reduction error. The existing `optimizer_eval_fn`/`optimizer_train_fn` remain reserved for current sample/save paths. There is no second model and no validation backward.

For each image compute full, low and high means from its check scalars with `math.fsum` and exact counts; then compute each role mean from its image means, giving repeated declarations their declared multiplicity. The equal-sized Stage 1 halves imply `full=(low+high)/2` within tolerance. Keep six candidate values local until both sets and all finite checks succeed, then call the existing `accelerator.log` once on main with exactly the six contract tags at the absolute step. A failed event publishes none of its values.

### Completed-update schedule and loss logs

Keep `global_step` as the **run-local** training counter passed to `process_batch` and existing training hooks. Add `absolute_completed_step = resume_start_step + run_completed_steps` for validation scheduling, enabled training log steps, enabled state metadata and checkpoint step names. The current run's `max_train_steps` remains its number of completed updates; `resume_start_step` does not enter optimizer/scheduler creation or training computation. In enabled mode, advance both counters only when `accelerator.sync_gradients` is true **and** `accelerator.optimizer_step_was_skipped` is false, after `optimizer.step` and post-step weight normalization. An accumulation microbatch or skipped mixed-precision update cannot advance the budget, log a new training point, or trigger validation. If a skipped update consumes an expected epoch slot, continue the existing outer epoch/inner batch loop until the completed-update budget is reached; the disabled loop retains its current range and behavior.

Run the initial event after tracker setup and existing initial sampling/setup, immediately before the first training batch, at absolute 0 or restored `s`. After each completed update, run at positive absolute multiples of `val_every_n_steps` and at the run's final completed update. Use one event decision so periodic/final overlap is emitted once. Place boundary validation before any next training batch and before current sample/save optimizer evaluation hooks. Enabled-only training loss logging keeps existing `generate_step_logs` tags and moving-average computation but publishes only after a completed update at its absolute step. Enabled epoch-loss logging keeps its tag but uses the current absolute step and skips an epoch with no completed update; the initial empty tracker log uses `s`. This avoids any resumed microbatch or skipped update creating a training-loss point at `s`; disabled per-microbatch/epoch logs remain unchanged. Enabled step-triggered save/sample cadence, numbered output and state retention use the absolute count consistently; epoch-triggered cadence retains its epoch basis. See the event table in the contract.

### One-rank/two-rank coordination

The Qwen forward and LoRA patched forward contain no explicit distributed collectives; the prepared transformer wrapper can perform DDP synchronization. In enabled multi-rank mode, all ranks exchange the post-attempt completed flag and resulting absolute count at each inner-loop boundary; any disagreement fails on all ranks before a conditional event. They wait before main evaluates the full unsharded Stage 1 manifest through `accelerator.unwrap_model(transformer)`. Other ranks do no validation forward or item partitioning. Main catches any event error and broadcasts a small success/error status to every rank; all ranks then raise the same diagnostic or proceed. Do not place a barrier after a possible main exception without that status exchange. Only main logs the six values. This keeps each declared occurrence exactly once globally; a two-process CPU test must exercise a successful and failing boundary.

### Resume and all-rank state saves

With validation enabled, save a versioned `val_loss_state.json` inside each existing Accelerate state directory. It contains a nonnegative exact integer absolute completed-update count, `model_version=original`, the frozen Stage 1 fingerprint, and the four effective numeric validation controls. The fingerprint already includes the controls; recording them explicitly supports diagnostics. A main-rank atomic write must complete **after** every rank has finished `accelerator.save_state` and before optional upload/retention, so an incomplete save never looks like a valid enabled state. Adapt the three existing `train_utils` step/epoch/final state-save helpers so enabled mode calls `accelerator.save_state` on **all ranks** at the same directory and then synchronizes; only main writes the sidecar, uploads and removes old directories. Broadcast a main-side post-save error before peers proceed. Their disabled main-only path stays unchanged. Accelerate writes `random_states_{process_index}.pkl` for each rank that calls save; calling it only on main would not preserve the second rank's RNG.

At `_register_hooks_and_resume`, the existing `accelerator.load_state` still restores model/optimizer/scheduler/RNG on every rank. Add a load pre-hook or equivalent check of the actual resolved state directory (local or downloaded) for the sidecar; collectively agree on schema, absolute count, controls and fingerprint against the freshly prepared Stage 1 manifest before accepting a result. Missing, malformed or mismatched metadata rejects enabled resume; no inference from `accelerator.step`, scheduler state or a parameter's step count. Store the loaded `s` separately and initialize the loop's absolute counter from it; network-weights-only loading (without `args.resume`) starts at 0. Disabled legacy resume ignores this sidecar and keeps prior behavior. Do not reset restored optimizer/scheduler state or warmup based on the new absolute graph axis.

### Verification boundary

Add focused CPU tests for objective/weighting equivalence (`none`, `sigma_sqrt`, `cosmap`), exact non-schedule sigma, image-first means with unequal sizes and repeats, finite/failing all-or-nothing logs, mode/RNG/gradient/optimizer/scheduler preservation and the next controlled update, accumulation/skipped updates, fresh/resumed boundaries, TensorBoard tag steps, and a real small Accelerate save/load round trip at `s>0`. Include a two-rank CPU success and error case and verify per-rank RNG state files. Existing Stage 1 tests and relevant training invariants remain regression checks. These tests are not real-model GPU verification. Implementation also reviews and updates `README.ru.md` and `docs/qwen_image.md` for the enabled workflow.

## Project Structure

### Documentation (this feature)

```text
specs/003-val-loss-training/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── contracts/
│   └── validation-training.md
└── quickstart.md
```

### Source Code (repository root)

```text
src/musubi_tuner/
├── qwen_image_train_network.py          # existing Qwen forward and narrow event hook
├── training/
│   ├── trainer_base.py                   # existing loop, shared loss, state/load boundaries
│   ├── timesteps.py                      # optional exact sigma for existing weighting
│   ├── validation_inputs.py             # existing Stage 1 manifest/check stream
│   └── accelerator_setup.py             # existing tracker setup, no new writer
└── utils/train_utils.py                 # existing state-save helpers, enabled all-rank path
tests/
├── test_qwen_image_validation_training.py # focused CPU and tracker tests
├── test_qwen_image_validation_inputs.py   # Stage 1 regression
└── test_qwen_image_training_invariants.py # existing forward/loss regression
README.ru.md
docs/qwen_image.md
```

**Structure Decision**: Extend the single Python trainer and its existing state and logging mechanisms. Add no second training loop, model, validation framework, cache format or output hierarchy.

## Complexity Tracking

No constitution violations require an exception.
