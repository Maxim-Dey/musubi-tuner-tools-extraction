# Implementation Plan: Portable Experiment and Complete Training States

**Branch**: `001-scope-qwen-image-lora` (current checkout; feature directory `004-experiment-training-states`) | **Date**: 2026-09-23 | **Spec**: [spec.md](spec.md)

**Input**: Stage 3 specification, its sampling clarifications, and the [Stage 1](../002-val-loss-core/contracts/validation-inputs.md) and [Stage 2](../003-val-loss-training/contracts/validation-training.md) contracts.

## Summary

Add an explicit `experiment_dir` mode to the existing Qwen-Image original LoRA trainer and both Qwen cache commands. Resolve inputs from one portable root, then coalesce validation, sampling, and saving by absolute completed optimizer step. Each saved step is one complete Accelerate state directory with one FP32 LoRA `model.safetensors`; one best directory and whole-package retention govern its location. The legacy branch remains unchanged when `experiment_dir` is absent.

## Technical Context

**Language/Version**: Python `>=3.10,<3.13`; local CPU validation uses Python 3.12

**Primary Dependencies**: Existing PyTorch, Accelerate 1.6.0, safetensors, TOML parser, Stage 1 frozen inputs, and Stage 2 validation state

**Storage**: Local experiment directory with Accelerate state files, the existing Stage 2 JSON sidecar, one versioned package JSON sidecar, and PNG samples; no new database or cache format

**Testing**: pytest path/preflight fixtures, real small CPU Accelerate save/load (one and two ranks), adapter load, sample/save counters, RNG and next-update comparison, legacy regressions

**Target Platform**: Existing Windows/Linux Qwen-Image original LoRA CLI and one/two-rank Accelerate workflows

**Project Type**: Python CLI training and cache commands

**Performance Goals**: No benchmark target; no second persistent LoRA copy or model instance

**Constraints**: FP32 trainable LoRA weights; one physical model file per step; no base DiT serialization; all-rank RNG save; no change to training mathematics or disabled-mode calls; no local GPU run/download

**Scale/Scope**: One portable root, train plus two validation datasets, one best metric (`val_loss_mean`), one retention window, and complete packages for periodic/final/epoch/best/sample events

## Constitution Check

*Gate evaluated before research and after Phase 1 design.*

| Principle | Design check | Status |
| --- | --- | --- |
| I. Language | All Stage 3 artifacts are English; user updates remain Russian. | Pass |
| II. Task Fidelity | FR-001–FR-018 and the two sampling clarifications are preserved. | Pass |
| III. Minimal, Compatible Changes | Reuse the parser, dataset readers, loop, sample method, and Accelerate hooks; opt-in branches protect legacy behavior. | Pass |
| IV. Training Invariants | Run-local computation and Stage 2 completed-update boundaries remain; package work never updates weights. | Pass |
| V. Memory and Performance | One prepared transformer/network; transient Accelerate state references are not another persisted model file. | Pass |
| VI. Configuration and Errors | Effective TOML/CLI settings, hierarchy, precision, retention and paths are checked before model loading. | Pass |
| VII. Verification and Documentation | Real CPU state/adapter checks, two-rank RNG, failure/retention cases, legacy tests, and README review are planned. | Pass |
| Local Stage / Workflow | Planning changes documentation only; no training, GPU, download or transfer here. | Pass |

No constitution exception or unresolved clarification remains. The post-design check passes: FP32 is a supported save precision, the one-file state preserves exact resume, and the opt-in path leaves prior output behavior available.

## Design

### Effective configuration and portable paths

Add `experiment_dir` to the Qwen training parser and `--experiment_dir` plus `--train_config` to each Qwen cache parser. Experiment-mode **training always requires** the existing `--config_file` selecting the resolved `<root>/train.toml`, plus enabled Stage 2 validation with `<root>/train-dataset.toml` and `<root>/val-dataset.toml`; an absolute root removes only the need for a *relative-root anchor*, not the selected training file. Resolve the selected file from invocation CWD once; anchor a relative effective `experiment_dir` to its parent and verify the selected file is the resulting root's `train.toml`. Cache `--train_config` serves the same anchoring role and may supply `experiment_dir` from its TOML, with an explicit cache CLI root taking precedence. A cache command may omit `--train_config` only when its `--experiment_dir` is absolute. Merge parser defaults, train TOML, then explicit CLI before path resolution and preflight. Do not `chdir`.

In experiment mode only, resolve relative `dataset_config`, `val_dataset_config`, `sample_prompts`, `dit`, `vae`, `text_encoder`, `network_weights`, each `base_weights` entry, `log_tracker_config`, and local `resume` from the root. Preserve absolute values and remote Hugging Face identifiers. Resolve `image_directory`, `image_jsonl_file`, `cache_directory` and relative JSONL `image_path` values under both dataset TOMLs from that root. Use the same dataset resolver in Stage 1's later frozen-item recheck; its portable fingerprint still excludes paths. Apply the resolver to both cache CLIs before role-aware preflight and model loading. Without opt-in, retain CWD semantics.

Use `<root>/output` and `<root>/output/tensorboard` as effective output/logging directories. Fill omitted values and reject explicit conflicting values with source/correction. Require the selected train/validation dataset TOMLs and Stage 2 validation controls before model loading; there is no CLI-only trainer experiment mode. Reject explicit `save_last_n_steps_state`, `save_last_n_epochs`, `save_last_n_epochs_state`, and `save_state_to_huggingface`; require nonnegative `save_last_n_steps` when supplied and reject effective `save_precision` below FP32. Cache CLIs retain `--dataset_config`; they do not assume trainer `--config_file`. See [contracts/experiment-cli.md](contracts/experiment-cli.md).

### One canonical adapter and complete Accelerate state

Continue to prepare the existing transformer and LoRA network once. In opt-in mode, all ranks call `accelerator.save_state(staging_dir, safe_serialization=True)`. Accelerate gathers state dictionaries for registered models before its pre-hook; replace the weight list with one canonical `accelerator.unwrap_model(network).state_dict()`, excluding the frozen transformer. Avoid the current `isinstance(prepared_model, type(unwrapped_network))` test: DDP wrappers do not satisfy it. On load, pass the unwrapped LoRA network to Accelerate so its canonical file loads on every rank. Its tensor keys match `network.save_weights`/`network.load_weights`; verify with the actual small LoRA loader and two-rank DDP. Keep the exact existing `val_loss_state.json` schema for Stage 2 resume and a separate `experiment_state.json` for ownership and metrics measured at this step; Accelerate's safetensors metadata remains `format=pt`. The project's Qwen adapter loader consumes tensor keys, not legacy export metadata.

The LoRA network trains in FP32 even with BF16 compute. Accept omitted, `float` or `fp32` `save_precision`; reject `bf16`/`fp16` in preflight because casting the sole file would lose exact FP32 weights. Suppress legacy standalone step/epoch/last `save_model` calls and legacy `*-state` helpers only in this mode. Accelerate keeps its real `optimizer.bin`, `scheduler.bin`, scaler if present, and per-rank `random_states_<rank>.pkl`. All ranks enter save; main checks exactly one model file, optimizer/scheduler and every rank RNG file after the barrier. On resume, check sidecar and each rank's required RNG file/keys before `load_state`; Accelerate catches RNG-load exceptions, so a successful call alone is insufficient. Preserve the existing Stage 2 validation identity check within each package.

### One event and one package per absolute step

Use Stage 2's `X=resume_start_step+run_completed_steps`; keep run-local `global_step` passed to training computation/hooks. At initial and completed-update boundaries, evaluate validation once where due. The existing Qwen evaluator already returns its complete six-value payload; narrowly return that payload from `_run_validation_boundary` on main as well. Main reads the current best, decides strict improvement and combines all reasons for a package at `X`: periodic `save_every_n_steps`, epoch, final completed update, strict new best, or enabled sample trigger. Main broadcasts either an error or the agreed metric/best/save/sample decision to every rank **before** any rank enters sampling or `save_state`; a failed validation or best lookup cannot strand peers at a later collective. Do not evaluate twice or expose partial values. Only a complete finite event can be a best candidate. A sample-only reason creates a full package. A valid step-0 best receives samples when sampling is enabled even if `sample_at_first=false`; that flag only suppresses a standalone initial sample request. Sampling enabled requires prompts and a trigger.

Once every rank receives the decision, create one staging directory if a package is due. When sampling is enabled, run existing sampling once into `staging_dir/samples/` from the current weights, then collectively report sample failures **before** any rank enters Accelerate save. With no prompts or trigger, skip sample resource preparation/inference and permit an empty `samples/` folder. Save Accelerate state once on all ranks and write both sidecars after all ranks finish. Preserve/restore used RNG streams and per-module modes around sampling even on errors. Add a narrow destination/force option to existing `sample_images` and Qwen inference so best/sample-only steps avoid shared `output/sample` and the old trigger gate. Order: validation → broadcast decision/error → optional sample → broadcast sample status → all-rank save → verify → publish → mandatory retention. No optimizer update occurs between validation, samples and saved weights. Epoch/final overlap is handled by the same event coordinator.

### Publication, best, resume and retention

Write a new package in a temporary directory under the experiment output volume. Main writes a versioned sidecar only after all-rank save and samples finish, checks required files and finite metrics, and publishes by same-volume directory rename. Temporary/incomplete folders are neither resumable nor eligible for best or retention. Broadcast failures so peers do not wait on a failed main. The sidecar records `output_name`, absolute step, Stage 2 fingerprint/controls and metrics measured **at this step**. Valid final filename, location and both sidecars establish package ownership; no separate root registry is needed. A package at another step does not inherit the old best's metrics. Resuming an older current package must discover/validate the existing best folder and read its own score instead of resetting the comparison.

Each saved step physically resides either at `<root>/output/current_training_states/<output_name>-step-<X>/` or `<root>/output/val_training_states/val-loss/<output_name>-step-<X>/`. Fully stage a new best before touching the old. Move the old best reversibly to current if retained, otherwise to a temporary holding path; publish the new best, then remove the old holding path only after success. On a handled replacement error, restore the previous best and report failure. Never expose two published best packages or claim crash-atomic exchange across directories. Reject an incompatible existing package at the same step instead of overwriting it, especially when resuming an older current step while a later best exists.

After **every successful** save/replacement, remove all owned complete current packages with `step < X-save_last_n_steps`; retention is a required publication phase, not optional cleanup. Equality stays; an unset limit keeps all; best is protected. Report and broadcast a retention failure rather than silently leaving the state inconsistent. Never call legacy independent state retention in opt-in mode. Current/best locations are moves, not copies/links. A resumed run uses the loaded package's own absolute step; best comparison comes independently from the existing best package. See [contracts/experiment-states.md](contracts/experiment-states.md).

### Verification boundary

Use relocated temporary roots and real parsers/readers to prove paths, precedence, mandatory selected train TOML/validation, source preflight and legacy CWD behavior before model construction. Use a small actual LoRA-compatible network with Accelerate to save/load current and best packages and compare adapter tensors, optimizer, scheduler, rank RNG and the next controlled update. Exercise two-rank CPU DDP hook selection, missing/corrupt rank RNG rejection, main-side decision/error broadcast before sample/save, and lower-precision preflight. The US2 package test uses a no-sampling configuration and asserts no sample preparation/inference while the full state remains loadable. Count save/sample calls for overlapping periodic/final/epoch/new-best/sample triggers, sample-only and step-0 best in US4; inject a sample failure there and confirm no published package or best replacement. Test strict improvement/tie/nonfinite results, mandatory inclusive retention after success, old-best movement and no deletion outside owned packages. Review `README.ru.md` and `docs/qwen_image.md` against implemented CLI. These are local checks, not real-model GPU verification.

## Project Structure

### Documentation (this feature)

```text
specs/004-experiment-training-states/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── contracts/
│   ├── experiment-cli.md
│   └── experiment-states.md
└── quickstart.md
```

### Source Code (repository root)

```text
src/musubi_tuner/
├── qwen_image_train_network.py                # Qwen-only opt-in args and preflight
├── qwen_image_cache_latents.py                 # experiment-aware latent cache entrypoint
├── qwen_image_cache_text_encoder_outputs.py   # experiment-aware text cache entrypoint
├── dataset/
│   ├── config_utils.py                        # shared dataset path resolver
│   └── image_video_dataset.py                 # JSONL source root only if needed
├── training/
│   ├── parser_common.py                       # existing TOML/CLI precedence
│   ├── trainer_base.py                         # existing loop, hooks and sampling seams
│   ├── validation_inputs.py                   # same portable resolver on frozen recheck
│   └── experiment_states.py                   # narrow package publication/best/retention logic
└── utils/train_utils.py                       # reuse state helpers; opt-in bypass
tests/
├── test_qwen_image_experiment_paths.py
├── test_qwen_image_experiment_states.py
├── test_qwen_image_validation_resume.py
└── test_qwen_image_training_invariants.py
README.ru.md
docs/qwen_image.md
```

**Structure Decision**: Extend the existing Python trainer and Qwen cache entrypoints. Keep package lifecycle in one small module so the training loop passes step, completed validation metrics and sample resources; do not build another trainer or general transaction framework.

## Complexity Tracking

No constitution violations require an exception.
