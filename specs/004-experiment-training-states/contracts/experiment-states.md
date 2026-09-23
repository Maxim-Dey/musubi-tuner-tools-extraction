# Complete Training State Contract

Applies only to Qwen-Image original LoRA training with effective `experiment_dir`, selected `<root>/train.toml` through `--config_file`, and enabled Stage 2 validation using both `<root>/train-dataset.toml` and `<root>/val-dataset.toml`. An absolute root does not require a relative anchor, but it does not permit CLI-only trainer mode. Cache commands may use an absolute root without `--train_config`. These requirements and experiment-mode rejection of explicit `save_last_n_steps_state`, `save_last_n_epochs`, `save_last_n_epochs_state`, and `save_state_to_huggingface` are preflight checks before model loading. If supplied, `save_last_n_steps` must be nonnegative. The [Stage 2 validation and absolute-step contract](../../003-val-loss-training/contracts/validation-training.md) remains authoritative for evaluation mathematics, tracker tags, completed updates and `val_loss_state.json`.

## Layout and naming

```text
<experiment_dir>/output/
├── tensorboard/
├── current_training_states/
│   └── <output_name>-step-<absolute_step>/
│       ├── model.safetensors
│       ├── optimizer.bin
│       ├── scheduler.bin
│       ├── random_states_<rank>.pkl
│       ├── val_loss_state.json
│       ├── experiment_state.json
│       └── samples/                 # PNGs when sampling is enabled
└── val_training_states/
    └── val-loss/
        └── <output_name>-step-<absolute_step>/   # at most one best package
```

Accelerate may write a scaler or other legitimate non-model state file. The exact installed-version optimizer/scheduler filenames must be inspected and tested, not renamed. Exactly one `model.safetensors` is present and no other model weight file is written in or outside the package by the trainer. `output/sample`, legacy standalone `*.safetensors`, and separate `*-state` directories are absent from experiment-mode output. Existing unrelated user files are never deletion targets.

An owned package must have the final `<output_name>-step-<X>` name in the selected output tree and valid package/Stage 2 sidecars with matching output name, step, model version, fingerprint and controls. Confirm its resolved path is within the selected output tree and do not follow a symlink as a retention target. Unknown or incomplete folders are unrelated and cannot be moved or pruned.

## Required package metadata

`val_loss_state.json` retains the exact Stage 2 schema and validation behavior. A second sidecar, `experiment_state.json`, is written after all state and sample files succeed:

```json
{
  "version": "qwen-image-experiment-state-v1",
  "output_name": "<configured output_name>",
  "model_version": "original",
  "absolute_completed_step": 123,
  "validation_fingerprint": "<Stage 1 64-hex fingerprint>",
  "controls": {
    "val_every_n_steps": 200,
    "val_seed_noise": 42,
    "val_level_noise_n": 10,
    "val_seed_noise_n": 1
  },
  "save_reasons": ["periodic", "new_best"],
  "metrics_at_step": {
    "train_eval_loss_mean": 0.1,
    "train_eval_loss_low_noise": 0.1,
    "train_eval_loss_high_noise": 0.1,
    "val_loss_mean": 0.2,
    "val_loss_low_noise": 0.2,
    "val_loss_high_noise": 0.2
  },
  "sample_count": 2
}
```

Values shown are illustrative. `absolute_completed_step`, controls, and `sample_count` are exact integers, not booleans; metrics are finite JSON numbers. `metrics_at_step` is null when no validation event completed at that step. `sample_count` is zero when sampling is disabled and otherwise equals the exact number of valid PNG files in `samples/`. Validate this count before publishing, loading, best selection, or retention. The single complete package physically under `val-loss/` is the authoritative best, and its own `metrics_at_step.val_loss_mean` is the selected score. Cross-check both sidecars' step, model version, fingerprint and controls. No old best metric is assigned to a different loaded model.

A package is complete only when all files above, including every participating rank's `random_states_<rank>.pkl`, and its required sample PNGs are verified. Missing/corrupt sidecars or rank files reject resume before `accelerator.load_state`. Validate device-appropriate RNG keys; then the real CPU save/load test must confirm actual RNG restoration because Accelerate 1.6 logs rather than propagates RNG-load failures.

## FP32 adapter and DDP hook

The single `model.safetensors` contains canonical **unwrapped** LoRA `state_dict` tensor names and FP32 trainable values. Accelerate's save pre-hook excludes frozen DiT model weights and keeps only this network; its load pre-hook presents the unwrapped network so DDP `module.` prefixes cannot enter the adapter file. All ranks save to one staging directory, but Accelerate writes common model/optimizer/scheduler files from main and one RNG file per rank. The file must load through both the project's LoRA adapter loader and Accelerate state restore. Lower-precision `save_precision` is rejected in experiment preflight.

## Event and package decision

`X=s+r`, where `s` is the Stage 2 loaded absolute completed step and `r` counts completed optimizer updates in this invocation. Accumulation microbatches and skipped updates do not advance `X` or trigger step saves. At each initial/completed-update boundary, compute one union of package reasons:

| Reason | Boundary |
| --- | --- |
| New best | A complete finite `val_unfamiliar` result with `val_loss_mean <` the current best; the first valid result qualifies, including at step 0. |
| Periodic | Positive `X` divisible by configured `save_every_n_steps`. |
| Final | Last completed update in this invocation, even off cadence. |
| Epoch | Configured epoch save at an epoch that completed an update; if this step already has a package, reuse it. |
| Sample | Configured sample trigger; a sample-only step explicitly creates a full package. |

Main receives the complete six-value validation payload (or no event), reads the current best, and determines best eligibility plus the combined save/sample reasons. Main broadcasts one decision or corrective error to all ranks before any rank enters sampling or `save_state`. A failed validation, best read or package decision aborts collectively at this boundary. If multiple reasons coincide, there is one state-save call, package and sample set. If an epoch boundary adds reasons to an already published package at the same completed step, update only its sidecar with the reason union; do not save state or generate samples again. A loaded, already published package at the initial resume step may be reused or moved to best if compatible; it is never saved a second time. On promotion, its own package sidecar receives that step's validation metrics and `new_best` reason, while the weight and sample files are moved without copying; a handled move failure restores the prior best and original sidecar. Incompatible published step collisions fail rather than overwrite previous weights. This also applies when replaying from an older current package while later states already exist.

When sampling is enabled, **every** new package gets one set of PNGs from that step's weights, even if the sample cadence alone was not due. `sample_at_first=false` suppresses only a standalone initial sample request; it does not remove PNGs from a required step-0 best package. With no prompts or no configured trigger, sampling is disabled: do not prepare sample resources or perform inference, and an empty `samples/` directory is allowed. A full package remains required and resumable without samples.

## Publish and replace

1. Main receives the completed validation payload, determines best eligibility and all package/sample reasons, and broadcasts either the decision or an error to every rank before any all-rank sample or save call. No optimizer update follows until this package decision finishes.
2. Only when sampling is enabled, all ranks participate in the existing sample path directed into a new staging package. Snapshot and restore RNG and module modes in `finally`; collectively report any failure before a rank enters save. A sample failure leaves the previous best and published packages untouched.
3. All ranks enter one `accelerator.save_state(staging_dir, safe_serialization=True)`. Main writes `val_loss_state.json` and `experiment_state.json` only after rank saves finish, then verifies the required files and expected samples. Collectively report errors.
4. Main renames a non-best staging directory directly to its sole current location. For a new best, first keep the fully staged candidate, move the old best to a reversible temporary location (or to current if retained), then rename the candidate into `val-loss/`. If candidate publication fails, restore the old best before reporting failure. No two published best packages coexist. Broadcast publication outcome.
5. After each successful publication or best replacement, perform mandatory retention: remove all owned complete current packages with `step < X-save_last_n_steps`, including a former best if outside the inclusive window; leave the former best in current when still within it. If the limit is unset, retain all current packages. Never prune current best, unknown folders or experiment inputs. Broadcast a retention error rather than treating failed cleanup as success.

Temporary directories have names outside the final package pattern and are not resumable or best candidates. Directory moves avoid physical weight/sample duplication; this protocol does not claim power-loss atomicity for a multi-directory exchange.

## Resume

For `--resume` to a current or best package, all ranks validate experiment ownership, package completeness, both sidecars, rank count/files and Stage 2 protocol against freshly preflighted inputs. Then every rank calls the existing `accelerator.load_state` on the **actual published directory** and obtains `s` from the Stage 2 sidecar. The run's `max_train_steps` remains its new update budget; logging/save/validation labels use `s+r`, while training computation keeps its run-local counter.

The current best score/step is read separately from the one published best package, even when the loaded state is older or is in current. A tie leaves that best unchanged. A newly saved package contains only metrics from its own validation event or null, not metrics from the existing best. Changed validation identity/controls remain rejected by Stage 2. Resume is a full state load, not `--network_weights`; dataloader position is not promised unless already supported by the existing state.

## Focused verification gates

US2 must prove a real one/two-rank Accelerate package is complete and loadable with sampling disabled: no sample resources are prepared, no inference runs, and an empty `samples/` directory is acceptable. A two-rank event test must also verify that every rank receives the same main-decided reasons or main error before entering any all-rank sampling/save operation. US4 separately injects a sample-generation failure and verifies that no partial package is published and the prior best remains intact, alongside successful sample-only and step-0-best cases.
