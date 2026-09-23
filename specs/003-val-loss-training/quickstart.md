# Quickstart: Validate Stage 2 Locally

These acceptance commands use CPU fixtures, not Qwen model weights or a GPU training run. Run them from the repository root in a Python environment with the project and test dependencies installed.

## Prerequisites

- Use Python `>=3.10,<3.13`, PyTorch, Accelerate 1.6.0, pytest and TensorBoard as declared by the project.
- Keep Stage 1 source images, captions and both role-bound caches fixed. The [Stage 1 contract](../002-val-loss-core/contracts/validation-inputs.md) describes their preparation and fingerprint.
- Set `PYTHONPATH` to the repository's `src` directory if the package is not installed.
- Disable GPU and external logging/downloads for the controlled local tests.

PowerShell setup:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:WANDB_MODE = 'disabled'
$env:PYTHONDONTWRITEBYTECODE = '1'
```

## Controlled acceptance

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_qwen_image_validation_training.py tests/test_qwen_image_validation_resume.py tests/test_qwen_image_validation_inputs.py tests/test_qwen_image_training_invariants.py
python -B -m pytest -p no:cacheprovider -q tests/test_qwen_image_config.py tests/test_qwen_image_dataset_cache.py
```

The Stage 2 tests should verify the following scenarios with a small CPU forward and independently calculated expected values:

| Scenario | Expected observation |
| --- | --- |
| Exact sigma with `none`, `sigma_sqrt`, `cosmap` | Validation scalar matches the existing training loss formula evaluated at the fixed `t`; an off-schedule `t` is not rounded or shifted. |
| Unequal image sizes/buckets and repeated declarations | Each occurrence has one equal share of its role; six values match an independent image-first reference within `1e-6`; full mean matches low/high average. |
| Fresh `B=5`, interval `E=2` | Validation event steps are exactly `0, 2, 4, 5`; the final/periodic overlap is never duplicated. |
| Resume from `s=5` for `B=3`, unchanged `E=2` | Start event at `5`, then events at `6, 8`; first new training-loss point at `6`, final absolute count `8`; optimizer/scheduler continue from loaded state. |
| Accumulation and skipped optimizer attempt | Neither creates an event, advances the absolute completed count, or publishes a training-loss point. |
| Mixed module modes, dropout and RNG; success and raised error | Each individual mode and Python/NumPy/torch CPU/available CUDA RNG state is restored; gradients, weights, optimizer and scheduler are unchanged by validation; the next controlled update matches no-validation control. |
| Nonfinite check or frozen input read failure | No partial six-tag payload; role/item/check/source appears in the diagnostic when known. |
| Actual TensorBoard event file | Exactly the six [contract tags](contracts/validation-training.md) appear once at each completed event's absolute step; existing training tag names remain present. |
| Two-rank CPU Accelerate boundary and state round trip | Every item is evaluated once globally, both ranks leave success/error boundaries, and the state has each rank's `random_states_{process_index}.pkl` plus a valid sidecar. |

Record the actual pass, failure or not-run result for each command in [validation.md](validation.md). Existing Stage 1, config, cache and Qwen training-invariant modules are regression checks. A full suite run does not replace the specified acceptance cases.

## Resume rejection checks

Exercise a real small Accelerate save/load at `s>0`. Confirm [`val_loss_state.json`](contracts/validation-training.md) records the explicit count, original model version, effective controls and Stage 1 fingerprint. Mutate a caption/cache or one control, or remove/corrupt the sidecar: enabled resume must fail before publishing validation values. A legacy state without that file must still resume when validation is disabled. Loading network weights without `--resume` must begin a fresh validation timeline at 0.

## Operational boundary

The user-facing Qwen entrypoint remains `python -m musubi_tuner.qwen_image_train_network --config_file <training.toml>`, with `val_dataset_config` pointing to the role-bearing validation TOML. Model weights, cache generation, GPU execution, multi-GPU production behavior and numerical backend comparison require the separate server verification stage; local CPU results must be reported as local checks only. The new output hierarchy and best-state selection are not part of Stage 2.
