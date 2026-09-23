# Local Validation: Training Metrics and Resume

Date: 2026-09-23

## US1 checkpoint

With Python 3.12.14 from the existing virtual environment and `src` on `PYTHONPATH`:

```text
pytest -q tests/test_qwen_image_validation_training.py tests/test_qwen_image_training_invariants.py
46 passed
```

This checkpoint covers the current Qwen forward/target, exact off-schedule `sigma=t` through the shared loss for `none`, `sigma_sqrt`, and `cosmap`, image-first low/high aggregation, real TensorBoard scalar tags, and rejection of partial/nonfinite events. The disabled-path invariant test also passes. Scheduling, distributed coordination, state persistence, and resume are not yet verified at this checkpoint.

No model weights, training run, GPU execution, download, or server work was used for these local checks.

## US2 checkpoint

With the same Python 3.12.14 virtual environment and `PYTHONPATH=src`:

```text
python -B -m pytest -p no:cacheprovider -q tests/test_qwen_image_validation_training.py tests/test_qwen_image_training_invariants.py --tb=short
53 passed in 24.50s
```

These cases cover mixed module modes, Python/NumPy/torch random-state restoration on success and failure, unchanged optimizer/scheduler/gradients and next update, accumulation and one skipped update, completed-update event ordering and logging, the disabled path, and a two-rank CPU success/error boundary with one global evaluation and no deadlock.

## US3 checkpoint

From the repository root, with the existing Python 3.12.14 virtual environment:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:WANDB_MODE = 'disabled'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:/Users/inbox/Desktop/musubi-tuner-flux2dev-lora/.venv/Scripts/python.exe' -B -m pytest -p no:cacheprovider -q --tb=short tests/test_qwen_image_validation_resume.py
```

Result: **14 passed in 15.09s**. A real one-rank Accelerate round trip checked step, epoch and final state directories, their versioned `val_loss_state.json`, and `random_states_0.pkl`. A real two-rank CPU round trip checked the shared sidecar plus both `random_states_0.pkl` and `random_states_1.pkl`, with restored rank-local Python, NumPy and torch RNG and optimizer/scheduler state. A rank-specific fingerprint mismatch and an injected main-rank sidecar-write failure reached both ranks without a deadlock. Missing, malformed, unknown-step, boolean-step, changed fingerprint/control and wrong model/version metadata were rejected; disabled legacy resume and network-weights-only fresh timeline remained available.

The controlled resumed loop loaded `s=5`, completed a new `B=3` update budget after one skipped attempt, evaluated exactly at `5, 6, 8`, and logged step training loss at `6, 7, 8`. Its initial empty tracker/sample used step 5; the skipped-only epoch added no training-loss point or duplicate sample. Numbered model/state saves used absolute 6 and 8, the final state recorded 8, and the restored scheduler continued to step 8 without restarting warmup. This is local CPU verification, not a real-model or GPU run.

The unchanged save/resume branch was also checked with:

```powershell
& 'C:/Users/inbox/Desktop/musubi-tuner-flux2dev-lora/.venv/Scripts/python.exe' -B -m pytest -p no:cacheprovider -q --tb=short tests/test_qwen_image_training_invariants.py -k 'actual_accelerator_hooks_restore_adapter_optimizer_scheduler_rng or names_and_independent_retention_windows or absent_validation_config_preserves_legacy_training_path'
```

Result: **3 passed, 27 deselected in 6.88s** under the same environment variables above.

## Final local acceptance

Using the PowerShell offline environment from [quickstart.md](quickstart.md) and the existing Python 3.12.14 virtual environment:

```text
& 'C:/Users/inbox/Desktop/musubi-tuner-flux2dev-lora/.venv/Scripts/python.exe' -B -m pytest -p no:cacheprovider -q tests/test_qwen_image_validation_training.py tests/test_qwen_image_validation_resume.py tests/test_qwen_image_validation_inputs.py tests/test_qwen_image_training_invariants.py --tb=short
107 passed in 35.85s

& 'C:/Users/inbox/Desktop/musubi-tuner-flux2dev-lora/.venv/Scripts/python.exe' -B -m pytest -p no:cacheprovider -q tests/test_qwen_image_config.py tests/test_qwen_image_dataset_cache.py --tb=short
1 failed, 199 passed: test_prompt_templates_and_all_readers expects 2 sample prompts, while tracked config_for_qwen_image_lora/sample_prompts.txt already contains 10. This unrelated baseline mismatch predates validation.

& 'C:/Users/inbox/Desktop/musubi-tuner-flux2dev-lora/.venv/Scripts/python.exe' -B -m pytest -p no:cacheprovider -q tests/test_qwen_image_config.py tests/test_qwen_image_dataset_cache.py -k 'not test_prompt_templates_and_all_readers' --tb=short
199 passed, 1 deselected in 15.00s
```

`ruff check` passed on the changed Stage 2 source and test files. Inspection of the Stage 2 trainer, Qwen event and state helpers found no Stage 3 experiment hierarchy or best-state selection. Real Qwen weights, model training, GPU execution, downloads and server verification were not run in this local stage.

## Convergence fix round 1

The independent converge found one confirmed gap: enabled ranks exchanged completion flags on every microbatch but did not compare the resulting absolute count until a due validation event. A two-rank CPU test first failed with the missing agreement seam. The loop now exchanges `sync_gradients`, `completed_update`, and candidate absolute step on every enabled inner-loop boundary. Both ranks reject different candidate steps before any conditional event. The targeted mismatch test passed, and the complete validation/input/invariant acceptance then passed with **108 tests** in 42.36s using the first final-acceptance command above. The unrelated prompt-template baseline result is unchanged.
