# Data Model: Validation Loss Events and Resume State

Stage 1 owns `ValidationManifest`, role-labelled `ValidationItem` occurrences, `ValidationCheck`, source/cache verification and deterministic noise. This stage consumes them unchanged.

## ValidationCheckLoss

Fields: role, item occurrence, one-based level `i`, realization `j`, exact `t`, low/high group, and one finite scalar `L(x,i,j)`. The forward receives the Stage 1 mixed latent, `epsilon`, and `1000*t`; the shared loss receives exact `sigma=t`. The scalar is produced by current Qwen `call_dit` and inherited weighted-MSE `compute_loss`, then detached. An item has exactly `N1*N2` losses, with `N1/2*N2` in each group. A nonfinite scalar is an error, not a zero or omitted check.

## ValidationImageResult

Fields: item occurrence, `checks_seen`, `low_checks_seen`, `high_checks_seen`, finite `mean`, `low_mean`, `high_mean`. `mean` averages every scalar for that item; `low_mean` and `high_mean` average their own subsets. Counts must match the effective Stage 1 controls before this result is accepted. An occurrence remains distinct even if its bytes or paths duplicate another occurrence.

## ValidationSetResult

Fields: `role`, `images_seen`, `mean`, `low_mean`, `high_mean`. Each field is the arithmetic mean of the corresponding **image** means, not all check tensors, pixels, buckets or repeat counts pooled together. `images_seen` equals the number of declared occurrences for that role; both roles are nonempty. All six values must be finite. The equal low/high check counts make `mean=(low_mean+high_mean)/2` within numerical tolerance.

## ValidationEvent

Fields: `absolute_completed_step`, Stage 1 `fingerprint`, expected item/check counts, two set results, and a six-key scalar payload. States: `scheduled` → `running` → `complete` → `published`; any read, model, nonfinite, count or aggregation failure sends `running` → `failed`. Only `complete` may publish, once per absolute step within the current run. A failed event has no valid published subset and supplies no result to later best-state selection. Role-to-tag mapping is in [contracts/validation-training.md](contracts/validation-training.md).

## ValidationTimeline

Fields: `resume_start_step=s` (nonnegative exact integer, zero for a fresh run), `run_completed_steps=r` (nonnegative exact integer, bounded by the current `max_train_steps=B`), `absolute_completed_step=s+r`, positive interval `E=val_every_n_steps`, and a set of event steps already emitted in this run. Training `global_step` remains run-local for calls into `process_batch` and hooks. An accumulation microbatch or skipped optimizer attempt leaves `r` and the absolute step unchanged. A completed optimizer update advances `r` once. The initial event is at `s`; positive periodic events occur when the new absolute step is divisible by `E`; the final event is at `s+B`, coalesced if periodic. Resume at a divisible `s` still gets one start event.

`max_train_steps` is the new run's completed-update budget, not an absolute stop step. The restored optimizer and scheduler state, including their own step counters, remains owned by Accelerate. The validation timeline does not seed or reset them. Disabled training has no new timeline object or scheduling side effects.

## ValidationStateMetadata

File: `val_loss_state.json` in an existing Accelerate state directory. Required fields: `version` (`qwen-image-val-loss-state-v1`), `model_version` (`original`), `absolute_completed_step` (nonnegative JSON integer, boolean rejected), `validation_fingerprint` (lowercase 64-hex Stage 1 digest), and `controls` (exact effective integers for `val_every_n_steps`, `val_seed_noise`, `val_level_noise_n`, `val_seed_noise_n`). The Stage 1 fingerprint is portable and already binds those controls plus both role-labelled input multisets. `val_dataset_config` path is excluded because relocation of unchanged inputs is permitted.

State transition: after an all-rank successful Accelerate save, main atomically writes metadata and then allows upload/retention. On enabled resume, every rank reads the metadata from the state directory used by `accelerator.load_state`, checks schema/types, count, controls, and fingerprint against its freshly prepared manifest, and agrees on the same `s` before validation starts. Missing, malformed, inconsistent or changed metadata is an error. Disabled legacy resume ignores the sidecar. A network-weights-only load is not a state resume and starts at `s=0`.

## TrainingStateSnapshot

Transient fields for one event: every unique model/LoRA module's original `.training` flag; Python `random`, NumPy, torch CPU and available CUDA RNG states. The event uses `torch.no_grad()` and restores the snapshot in `finally`, both on success and failure. Existing parameter gradients, weights, optimizer and scheduler states have no validation write path; controlled tests compare them before/after, including the next training update. No snapshot is persisted in the validation sidecar because Accelerate already owns training-state checkpointing.
