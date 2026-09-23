# Validation Training and Resume Contract (Stage 2)

This contract applies only when the Qwen-Image original LoRA trainer has `val_dataset_config`. It consumes the existing [Stage 1 validation-input contract](../../002-val-loss-core/contracts/validation-inputs.md). No validation mode is activated when that option is absent.

## One check and one event

For each declared image occurrence and each Stage 1 pair `(i,j)`, use its frozen latent/text cache, fixed `t`, seeded epsilon, and mixed latent. Call the **same current Qwen transformer and LoRA weights** through `QwenImageNetworkTrainer.call_dit`, with a batch-one `timestep=1000*t`. Its model receives `t`; its flow target is `epsilon-latent`. Call the inherited `NetworkTrainer.compute_loss` with exact `sigma=t`, using the same elementwise MSE, optional `args.weighting_scheme`, precision and final `.mean()` as training. The exact-sigma override applies only to validation; absent override keeps training's existing scheduler-sigma path.

The check produces one finite scalar. For every occurrence, calculate `mean_x = sum(all checks)/(N1*N2)`, `low_x = sum(t<0.5 checks)/((N1/2)*N2)`, and `high_x` likewise. For each role, average the corresponding **image** means over its declared occurrence count. Each occurrence contributes once even if it shares image bytes with another. Counts must be exact, each set nonempty, and `role_mean≈(role_low+role_high)/2` within numerical tolerance. Use stable finite scalar summation; no pixel, bucket or tensor pooling across images.

Only after both roles finish, publish one scalar payload via the existing Accelerate tracker on main rank:

| Role | Full mean | Low group | High group |
| --- | --- | --- | --- |
| `val_familiar` | `train_eval_loss_mean` | `train_eval_loss_low_noise` | `train_eval_loss_high_noise` |
| `val_unfamiliar` | `val_loss_mean` | `val_loss_low_noise` | `val_loss_high_noise` |

The payload contains exactly these six finite numbers at one absolute completed-update step. A nonfinite scalar/aggregate, frozen-input mismatch, read failure or interrupted event publishes **none** of that event's six values and reports the role, item/check/source when known, cause and correction. It cannot be used as a best result.

## Step and event schedule

`s` is the saved absolute completed-update count, or 0 for a fresh run. `r` is the number of **completed** updates in this invocation; `B=max_train_steps` is the requested completed-update budget for this invocation. The event axis is `X=s+r`, independent of the run-local `global_step` passed into training computation/hooks. A completed update requires both `accelerator.sync_gradients` and `not accelerator.optimizer_step_was_skipped`. In enabled multi-rank mode all ranks exchange their post-attempt completed flag and resulting `X` at every inner-loop boundary; a disagreement fails on all ranks before any rank enters a conditional event.

| Boundary | `r`/`X` change | Validation | Enabled training-loss log |
| --- | --- | --- | --- |
| Fresh or resumed start, before first training batch | `r=0`, `X=s` | Once at `s`, even if `s` is divisible by interval | None |
| Accumulation microbatch | No change | None | None |
| Synced but skipped optimizer attempt | No change | None | None |
| Completed update, positive `X` divisible by `E=val_every_n_steps` | `r+=1`, `X=s+r` | Once after update, before next batch | Existing training tags at `X` |
| Completed update with `r=B` | `r+=1`, `X=s+B` | Once even if not divisible; coalesce if also periodic | Existing training tags at `X` |
| Other completed update | `r+=1`, `X=s+r` | None | Existing training tags at `X` |

Enabled training logs keep the current `generate_step_logs` tag names and loss-recorder calculation, but are emitted only at a completed update. Enabled `loss/epoch` remains present and uses the current absolute step, provided that epoch completed at least one update; an epoch of only skipped attempts emits no training-loss point. The initial empty tracker log uses `s`. Thus a resume at `s>0` has no fabricated training loss at `s`; the first new point is `s+1`. With validation disabled, the original per-microbatch/epoch training logs, step handling, sampling and saving remain unchanged. In enabled mode the run-local counter remains the argument to `process_batch` and training hooks; the absolute count labels validation, logs, state and step-named output so resume cannot overwrite an earlier numbered step. Step-triggered save/sample cadence and retention use the same absolute count; epoch-triggered cadence retains its epoch basis. `B` does not become `s+B` in optimizer/scheduler setup. If a skipped attempt prevents reaching `B` within the initially estimated epochs, the same training loop continues through more epochs until `B` completed updates occur.

## State isolation and distributed boundary

All ranks arrive at an enabled validation boundary and wait. Main rank reads/evaluates the entire unsharded Stage 1 manifest on `accelerator.unwrap_model(transformer)`; the LoRA network remains attached to that same transformer's forward. The prepared DDP wrapper is not called for main-only validation. Other ranks do no validation forwards. Main captures failures and broadcasts a success/error status before ranks leave the boundary; every rank raises the same failure instead of waiting forever at a later barrier. Only main publishes metrics.

The event runs under `torch.no_grad()`, without backward, `optimizer.step`, scheduler step, zero-grad, or optimizer eval/train hooks. Set transformer and LoRA trees to eval for the event. Snapshot and restore **each** prior module mode individually and Python, NumPy, torch CPU and available CUDA RNG states in `finally`, including a failed event. Existing weights, gradients, optimizer/scheduler state and the next controlled update must match a run without an intervening event.

## Accelerate state sidecar

Each enabled step, epoch or final state save uses its existing directory name and includes the following JSON sidecar, written atomically as `val_loss_state.json` after every rank finishes `accelerator.save_state` and before optional upload/retention:

```json
{
  "version": "qwen-image-val-loss-state-v1",
  "model_version": "original",
  "absolute_completed_step": 123,
  "validation_fingerprint": "<64 lowercase hexadecimal SHA-256 characters>",
  "controls": {
    "val_every_n_steps": 200,
    "val_seed_noise": 42,
    "val_level_noise_n": 10,
    "val_seed_noise_n": 1
  }
}
```

The example count and controls are illustrative; save their actual effective values. JSON numbers for counts/controls must be exact integers, not booleans. The fingerprint is the Stage 1 portable identity, which binds both role-labelled multisets and controls; the path string is not identity. The sidecar is an additional file, not a replacement for Accelerate optimizer, scheduler, model or per-rank RNG files. In enabled mode all ranks call `accelerator.save_state` for the same directory; only main writes the sidecar, uploads and prunes state directories, then broadcasts its post-save status before peers proceed. Normal one-rank and two-rank saves must produce each rank's `random_states_{process_index}.pkl`.

On `args.resume`, validate this file from the **actual state directory loaded** (local or downloaded) against the freshly prepared Stage 1 manifest and effective controls, and obtain `s` from `absolute_completed_step`. All ranks agree on the result before continuing. A missing/invalid sidecar, nontrustworthy count, model mismatch or changed fingerprint/controls rejects enabled resume with a corrective message; never guess from `accelerator.step`, scheduler or parameter state. Accelerate then restores its normal training state; no warmup restart is introduced by the validation axis. Without `args.resume`, loading network weights alone starts `s=0`. Without `val_dataset_config`, old state directories without this file remain resumable and the legacy path is unchanged.
