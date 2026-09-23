# Stage 3 Local Validation

## T003: Absent-`experiment_dir` legacy baseline (2026-09-23)

The selected CPU tests cover CWD-relative configuration and dataset paths, the separate legacy adapter and state outputs, the shared `output/sample` destination, and local state resume. The focused CWD and artifact assertions were added to `tests/test_qwen_image_training_invariants.py`. No full Qwen model weights, GPU training, download, or user prompt edit was involved.

PowerShell setup used for both commands:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:WANDB_MODE = 'disabled'
$env:PYTHONDONTWRITEBYTECODE = '1'
```

Full baseline command:

```powershell
& 'C:/Users/inbox/Desktop/musubi-tuner-flux2dev-lora/.venv/Scripts/python.exe' -B -m pytest -p no:cacheprovider -q --tb=short tests/test_qwen_image_training_invariants.py tests/test_qwen_image_config.py tests/test_qwen_image_dataset_cache.py
```

**FAIL (exit 1): 230 passed, 1 failed in 16.30s.** The only failure is the known pre-existing `tests/test_qwen_image_config.py::test_prompt_templates_and_all_readers`: it asserts two supplied prompts, while the unchanged user `sample_prompts.txt` currently contains ten. This failure was not hidden or repaired by changing prompts.

Focused legacy gate with only that unrelated mismatch deselected:

```powershell
& 'C:/Users/inbox/Desktop/musubi-tuner-flux2dev-lora/.venv/Scripts/python.exe' -B -m pytest -p no:cacheprovider -q --tb=short tests/test_qwen_image_training_invariants.py tests/test_qwen_image_config.py tests/test_qwen_image_dataset_cache.py -k 'not test_prompt_templates_and_all_readers'
```

**PASS (exit 0): 230 passed, 1 deselected in 15.93s.** This includes the new CWD check, the legacy training-loop adapter/state and resume assertions, and the existing sample-destination check. These are local CPU results, not real-model or GPU verification.

## T011: Portable experiment paths and cache preflight (2026-09-23)

The new path tests were written first. Their initial result was **26 failed, 4 passed**, because the experiment-root parser and cache options did not yet exist. After implementing the opt-in path rules, the same command was:

```powershell
$env:PYTHONPATH = 'src'
& 'C:/Users/inbox/Desktop/musubi-tuner-flux2dev-lora/.venv/Scripts/python.exe' -B -m pytest -p no:cacheprovider -q tests/test_qwen_image_experiment_paths.py --tb=line
```

**PASS (exit 0): 30 passed in 8.22s.** Tests call the trainer and both Qwen cache entrypoints from another CWD, stopping before model load. For a selected `<root>/train.toml` containing `experiment_dir = "."`, effective `dataset_config`, `val_dataset_config`, DiT, VAE, text encoder, and JSONL image paths resolve under `<root>`; output and logs resolve to `<root>/output` and `<root>/output/tensorboard`. Renaming the unchanged root preserves the Stage 1 fingerprint and its later frozen-item cache read. Relative paths in legacy calls remain CWD-relative.

The affected legacy gate was repeated after these edits:

```powershell
$env:PYTHONPATH = 'src'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:WANDB_MODE = 'disabled'
& 'C:/Users/inbox/Desktop/musubi-tuner-flux2dev-lora/.venv/Scripts/python.exe' -B -m pytest -p no:cacheprovider -q --tb=short tests/test_qwen_image_training_invariants.py tests/test_qwen_image_config.py tests/test_qwen_image_dataset_cache.py -k 'not test_prompt_templates_and_all_readers'
```

**PASS (exit 0): 230 passed, 1 deselected in 16.77s.** The deselected test is the same baseline prompt-count mismatch described above. No GPU, full model, or network download was used.

## T019/T026/T031: Complete states, best selection and samples (2026-09-23)

The state tests were written before the package module; initial US2 result was **1 passed, 9 failed** because `experiment_states` did not exist. US3 then recorded **8 expected failures** for missing strict-best, retention and distributed-decision operations. US4 recorded failures for missing package-sample seams; three later integrity tests proved that removed PNGs were incorrectly accepted before `sample_count` was added. The skipped-final-attempt epoch test and metadata-union test each reproduced a separate scheduling gap before repair.

```powershell
$env:PYTHONPATH = 'src'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:WANDB_MODE = 'disabled'
& 'C:/Users/inbox/Desktop/musubi-tuner-flux2dev-lora/.venv/Scripts/python.exe' -B -m pytest -p no:cacheprovider -q --tb=short tests/test_qwen_image_experiment_paths.py tests/test_qwen_image_experiment_states.py tests/test_qwen_image_validation_resume.py tests/test_qwen_image_validation_training.py tests/test_qwen_image_training_invariants.py tests/test_qwen_image_config.py tests/test_qwen_image_dataset_cache.py -k 'not test_prompt_templates_and_all_readers'
```

**PASS (exit 0): 348 passed, 1 skipped, 1 deselected in 95.87s.** The skip is a Windows symlink-privilege case; retention's foreign, incomplete and input-preservation cases pass without it. The deselection is the unchanged prompt-count mismatch from the baseline. One PyTorch warning comes from the controlled test's replaced optimizer step, not product execution.

The real tiny Qwen-compatible FP32 LoRA and CPU Accelerate tests save one canonical `model.safetensors` plus `optimizer.bin`, `scheduler.bin`, both sidecars and each rank's `random_states_<rank>.pkl`; no frozen DiT or second adapter appears. The Qwen LoRA loader accepts that file. One-rank and two-rank DDP load checks restore the absolute completed step, parameters, optimizer, scheduler and rank-local Python/NumPy/torch RNG; the next controlled update matches an uninterrupted reference. Both current and best locations load. Missing or corrupt rank files, changed validation fingerprint/controls, missing PNGs, and incomplete packages fail before load.

The best tests show strict finite `val_unfamiliar.val_loss_mean` improvement, ties retained, one best folder, directory moves rather than weight copies, and rollback on publication failure. At `X=12,N=8`, a complete owned current step 4 remains and step 3 is removed as a whole; best and unrelated files remain. Resuming an older current package reads the later best independently; an improved initial resume event promotes the existing current package without another state save. A package's sidecar stores only its own event metrics.

The sample tests show one full package and one PNG set for sample-only, step-0 best with `sample_at_first=false`, and overlapping periodic/epoch/final/new-best/sample reasons. Controlled preparation and sampling preserve Python/NumPy/torch RNG, module modes, VAE placement, gradients and the next update on success and injected hook/inference failures. A missing image or PNG prevents publication and leaves the prior best intact. When the epoch's final optimizer attempt is skipped, its last completed step receives the epoch/sample reasons without another state or PNG save.

## T034: Command surface and scope

All three `python -m musubi_tuner.qwen_image_* --help` entrypoints were inspected. The trainer exposes `--experiment_dir` beside `--config_file` and `--val_dataset_config`; both cache commands expose `--train_config`, `--experiment_dir`, and their existing `--dataset_config`. On Windows, the trainer's multilingual help requires `PYTHONIOENCODING=utf-8` under the current cp1252 console; with that environment variable the help exits 0. `README.ru.md` and `docs/qwen_image.md` were reviewed against the implemented opt-in paths, package contents and legacy branch; 31 local Markdown links resolve. These checks use temporary files and a tiny CPU adapter only. Full Qwen weights, GPU training, downloads and remote transfer were **NOT RUN** in Stage 3.

## Convergence correction round 1 (2026-09-23)

The three appended Phase 8 tasks T035–T037 are complete. A controlled mode-switching optimizer test first reproduced differing weights between validation and samples; the opt-in experiment event now keeps validation, PNG generation and state publication on the same training weights. A malformed but deserializable per-rank RNG payload first reproduced a late resume failure; package preflight now validates Python, NumPy and torch rank RNG payloads without changing process RNG. Canonical-path tests first reproduced acceptance of renamed train/validation dataset TOMLs; the experiment trainer and cache commands now require the specified root files while legacy paths remain unchanged.

The combined affected CPU suite including the new tests passed: **369 passed, 1 skipped, 1 deselected, 1 warning in 95.70s**. It covered experiment paths, packages, rank RNG, validation training/resume, and the existing training/config/cache invariants. The skip, deselection and warning have the same causes described above. Ruff on affected code/tests and git diff --check passed. No model weights, GPU training or network transfer was used.

## Convergence correction round 2: T038 CUDA RNG preflight

The CPU test simulates a CUDA accelerator and two visible devices without calling CUDA. Before the fix, all six malformed payloads (empty states, wrong device count, dtype, shape, byte length, and Philox offset) reached `accelerator.load_state`: **6 failed, 4 passed**. Preflight now requires one usable CPU byte tensor per visible CUDA device, accepting the 8-byte legacy or 16-byte current PyTorch state format and a valid 16-byte offset. The same test confirms a valid two-device payload reaches load without executing CUDA.

```powershell
$env:PYTHONPATH = 'src'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:WANDB_MODE = 'disabled'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:/Users/inbox/Desktop/musubi-tuner-flux2dev-lora/.venv/Scripts/python.exe' -B -m pytest -p no:cacheprovider -q --tb=short tests/test_qwen_image_experiment_rng_integrity.py
& 'C:/Users/inbox/Desktop/musubi-tuner-flux2dev-lora/.venv/Scripts/python.exe' -B -m pytest -p no:cacheprovider -q --tb=short tests/test_qwen_image_experiment_states.py -k two_rank_ddp_package
```

**PASS:** RNG integrity 11 passed in 5.31s (including valid 8-byte legacy and 16-byte states); real CPU two-rank DDP packages 3 passed, 40 deselected in 29.29s. No CUDA/GPU execution or model weights were used.
