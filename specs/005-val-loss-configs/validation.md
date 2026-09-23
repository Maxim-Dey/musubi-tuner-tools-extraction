# Stage 4 Local Validation

## T001: Existing CPU baseline (2026-09-23)

Before example creation, qwen_image_lora_val_example/ did not exist; no user file at that path was overwritten. The current Stage 1–3 CPU baseline used the existing Python 3.12 environment, PYTHONPATH=src, CUDA_VISIBLE_DEVICES=-1, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, WANDB_MODE=disabled, and PYTHONDONTWRITEBYTECODE=1.

Command: python -B -m pytest -p no:cacheprovider -q --tb=short tests/test_qwen_image_validation_inputs.py tests/test_qwen_image_experiment_paths.py tests/test_qwen_image_experiment_states.py tests/test_qwen_image_experiment_rng_integrity.py tests/test_qwen_image_validation_training.py tests/test_qwen_image_validation_resume.py tests/test_qwen_image_training_invariants.py tests/test_qwen_image_config.py tests/test_qwen_image_dataset_cache.py -k 'not test_prompt_templates_and_all_readers'

PASS: 417 passed, 1 skipped, 1 deselected, 1 warning in 99.95s. The Windows symlink-permission case is skipped. The deselected existing prompt-count test expects two prompts while the unchanged user file has ten. The PyTorch warning is from a controlled test's replaced optimizer step. These are CPU checks; no external model weights or GPU training ran.

## T002–T003: Fixture and expected red example contract

`tests/test_qwen_image_val_example.py` reuses the existing path, validation-item, and tiny-adapter helpers. Its temporary train source and two validation roles have real captioned images, source-bound latent/text caches, and 64×64/32×64 validation buckets. The physical-file contract checks all four required files, exact TOML values and supported keys, the two prompt lines, distinct train/validation declarations, and absence of fabricated images or caches.

```powershell
$env:PYTHONPATH="$PWD/src"
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:WANDB_MODE='disabled'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/inbox/Desktop/musubi-tuner-flux2dev-lora/.venv/Scripts/python.exe' -B -m pytest -p no:cacheprovider -q --tb=short tests/test_qwen_image_val_example.py
```

Expected RED before example creation: **1 passed, 1 failed in 7.43s**. The fixture/real-cache test passed. `test_example_files_and_values` failed at its first assertion because none of `train.toml`, `train-dataset.toml`, `val-dataset.toml`, or `sample_prompts.txt` exists yet. Ruff on the new test file passed. No example files, production code, or user data were changed in this step.

## T007: Real readers and preflight from another CWD

The shipped four files were copied to a temporary root with the existing captioned image/cache fixture. Only test inputs were substituted: small 64-pixel bucket resolution, three existing temporary model paths, CPU-compatible `AdamW`, and `CPU_TRIGGER` in prompts. From another working directory, the real Qwen training parser and experiment preflight resolved the root and every internal path. The training reader found one unroled train image and its exact cache pair; validation found two familiar images at 64×64 and 32×64 plus one unfamiliar image with distinct source-bound caches. The prompt reader accepted both lines. Existing `config_for_qwen_image_lora/train.toml` and `sample_prompts.txt` bytes remained unchanged.

The same preflight rejected an unresolved DiT placeholder, a missing familiar caption, a missing unfamiliar text cache, a conflicting output path, and an unknown training key before model setup. With the T002–T003 test file, the targeted CPU command above passed: **8 passed in 8.96s**. Ruff on `tests/test_qwen_image_val_example.py` passed. No production files or example values were changed.

## T008: Four cache CLI selections

The copied physical example was run from another CWD through each real cache command's parser with `--train_config <root>/train.toml`, explicit `--dataset_config` and `--vae` or `--text_encoder`, and `--model_version original`. The existing test seam stopped at the model-load boundary. Both train calls selected only the one unroled train source/cache. Both validation calls selected `val_familiar` (two images) and `val_unfamiliar` (one image) with their separate caches. All four selected canonical TOMLs inside the temporary root. The targeted CPU command above passed: **12 passed in 8.44s**; Ruff on the test file passed. No VAE/text-encoder model was loaded.

## T010–T011: Config-fed fixed evaluation and complete package

The copied physical `train.toml` was parsed and preflighted with only small CPU substitutions (the same temporary inputs as T007). Its validation manifest contained two familiar images in 64×64 and 32×64 buckets and one unfamiliar image. For every image, the real cache reader and noise generator produced the fixed SHA-256-based 10×1 checks with one-based indices, exact midpoint levels 0.095–0.905, and the same noise on repetition, while the parsed training controls remained `timestep_sampling="shift"` and `discrete_flow_shift=2.2`.

The existing Qwen forward and common loss produced exactly six finite TensorBoard metric series at nonzero completed step 200. A real CPU Accelerate save then published one complete `qwen_image_lora-step-200` current package with those exact metrics in its sidecar, one canonical FP32 floating-point LoRA `model.safetensors` (integer alpha tensors stay integer), optimizer, scheduler, rank-0 RNG, and matching validation/experiment sidecars. The existing Qwen LoRA loader accepted the adapter. There were no duplicate adapter/model files or PNGs. This short chain did not run 200 optimizer updates; Stage 2/3 tests cover event scheduling and sample/best/retention cases separately.

The targeted CPU command above passed: **13 passed in 9.44s**. Ruff on `tests/test_qwen_image_val_example.py` passed. No full Qwen weights or GPU execution were used.

## T013–T014: Relocation and nonzero resume

The temporary experiment was preflighted from one external CWD, then its whole root was renamed and preflighted from a second external CWD without changing the internal TOMLs. The three dummy model paths were absolute files outside the movable root, matching the server-path contract. Train/validation TOMLs, prompt, output, logging, and local resume paths rebased to the new root; the validation fingerprint, source image hashes, cache hashes, bucket sizes, and cache reads remained identical.

The same config-fed CPU chain's complete step-200 package was loaded through the real opt-in Accelerate hook. A short `B=2` controlled trainer loop used the real validation event and emitted its six metrics once at initial absolute step `s=200`, training loss at 201 and 202, and a final complete package at `s+B=202`. The new test reused the Stage 2 loop seam and Stage 3 package loader, without a second model/training harness. Current/best and exact-next-update variants remain in the existing Stage 3 tests for T017. The targeted test command above passed: **15 passed in 10.39s**; Ruff passed. No GPU or external weights were used.

## T004–T006: Physical example and green file contract

Created the four separate example files and seven empty `.gitkeep` directories. `train.toml` marks the three absolute server paths as replaceable, specifies the original BF16 DiT, and explains `save_precision="fp32"` for the sole resumable adapter. The two dataset TOMLs have one unroled train source and exactly two separate validation roles; no images or caches were added.

```powershell
$env:PYTHONPATH='src'
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:WANDB_MODE='disabled'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/inbox/Desktop/musubi-tuner-flux2dev-lora/.venv/Scripts/python.exe' -B -m pytest -p no:cacheprovider -q --tb=short tests/test_qwen_image_val_example.py
```

PASS: **2 passed in 8.64s**. The physical-file test checked every exact TOML value and type through the real training parser, both dataset declarations, both prompt lines through the real reader, all seven directories, and absence of fabricated dataset/cache files. This is a CPU fixture check; no model weights or GPU training were used.

## T009: 1,600-step validation schedule without training

The existing Qwen validation boundary was exercised once at each candidate absolute step 0..1600 with a controlled evaluator and val_every_n_steps=200. The observed events were exactly 0,200,400,600,800,1000,1200,1400,1600, with one event at the coincident periodic/final step 1600. This check does not perform optimizer updates.

Targeted pytest: **1 passed, 24 deselected in 7.80s**. Ruff on the edited test module passed.

## T012: Example run guide

Updated `docs/qwen_image.md` to link the physical example and document the four canonical cache invocations from another CWD, explicit cache model flags, trainer and TensorBoard commands, required server paths/data/`TOK`, six tags and the 0..1600 validation grid. The guide states BF16 compute versus the sole FP32 resumable adapter, removes both prompt and cadence keys to disable sampling, explains step-0 best samples, and reserves full H200 operation for later private-server verification. Existing generic and legacy instructions remain in place.

Read-only CLI verification with the existing Python 3.12 environment, `PYTHONPATH=src`, `CUDA_VISIBLE_DEVICES=-1`, offline environment variables, and `PYTHONIOENCODING=utf-8`:

```powershell
& $python -B -m musubi_tuner.qwen_image_train_network --help
& $python -B -m musubi_tuner.qwen_image_cache_latents --help
& $python -B -m musubi_tuner.qwen_image_cache_text_encoder_outputs --help
```

All three help commands exited 0. The trainer exposes `--config_file`, `--resume`, `--experiment_dir`, and `--val_dataset_config`; the latent cache exposes `--train_config`, `--dataset_config`, `--vae`, and `--model_version {original}`; the text cache exposes `--train_config`, `--dataset_config`, required `--text_encoder`, and `--model_version {original}`. No model or GPU was run. The first trainer help attempt without UTF-8 console encoding failed while printing Unicode help text; setting `PYTHONIOENCODING=utf-8` resolved it.

## T015: Move and resume guide

`docs/qwen_image.md` and the Stage 4 quickstart now show the actual current/best published package paths with `--resume`, run from another CWD after updating only the moved root path. Both state locations and the trainer's `--resume` flag match the Stage 3 CLI/state contracts and the real trainer help checked in T012. The guide requires unchanged validation images, captions, source-bound caches, and noise controls; it distinguishes saved absolute step `s` from a new `max_train_steps=B` update budget, gives first log `s+1` and final step `s+B`, and does not promise data-loader position restoration. These are documentation checks; the T013/T014 controlled CPU relocation/resume acceptance is tracked separately.

## T016: README entrypoint

The Russian README now links the shipped example and exact Qwen command guide, names the three model-path and TOK replacements and user-owned image/caption/cache preparation, and preserves the opt-in boundary. Both local link targets exist and the guide heading anchor matches. git diff --check for README passed. This is documentation only; no H200 result is claimed.

## T017: Combined CPU acceptance and requirement evidence

With PYTHONPATH=src, CUDA_VISIBLE_DEVICES=-1, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, WANDB_MODE=disabled and PYTHONDONTWRITEBYTECODE=1, the existing Python 3.12 environment ran:

python -B -m pytest -p no:cacheprovider -q --tb=short tests/test_qwen_image_val_example.py tests/test_qwen_image_validation_inputs.py tests/test_qwen_image_experiment_paths.py tests/test_qwen_image_experiment_states.py tests/test_qwen_image_experiment_rng_integrity.py tests/test_qwen_image_validation_training.py tests/test_qwen_image_validation_resume.py tests/test_qwen_image_training_invariants.py tests/test_qwen_image_config.py tests/test_qwen_image_dataset_cache.py -k 'not test_prompt_templates_and_all_readers'

PASS: **433 passed, 1 skipped, 1 deselected, 1 warning in 106.63s**. The skip, deselection and controlled PyTorch warning are the same baseline cases recorded under T001. Ruff on the two Stage 4 test modules passed, as did git diff --check. The three Qwen entrypoint --help commands exited 0 (Windows trainer help required PYTHONIOENCODING=utf-8). No full Qwen weights, GPU training, download or server transfer was used.

| Requirement | Local evidence |
| --- | --- |
| FR-001 / SC-001 | Physical-file test reads all four example files and seven empty directory markers; no fabricated image/cache and no original user file rewrite. |
| FR-002 | Exact trainer TOML values and three explicit absolute replaceable original-model paths parsed by the real Qwen parser/preflight. |
| FR-003 | Exact 45-key value/type contract, integer warmup and training shift; fixed noise grid test separately proves validation levels are unshifted. |
| FR-004 | FP32 sole adapter and one complete state in the composed save; Stage 3 retention and no-duplicate package tests rerun. |
| FR-005 / SC-002 | Training TOML has exactly one unroled train image/cache source; zero val_unfamiliar images occur in its real reader. |
| FR-006 | Validation TOML has exactly familiar/unfamiliar roles and separate source/cache paths; two image sizes read in CPU fixture. |
| FR-007 | Both exact TOK prompt lines pass the real prompt reader; documentation gives replacement and removal of both sample controls. |
| FR-008 | Four real cache CLI selections from another CWD reach both validation roles; trainer/cache --help and guide commands checked. |
| FR-009 / SC-003 | Whole-root move from one CWD to another retains internal path rebasing, fingerprint, image/cache hashes and cache reads. |
| FR-010 / SC-004 | Controlled due-grid gives exactly nine 0..1600 events; real evaluator writes exactly six finite TensorBoard tags. Existing Stage 3 best/coalescing tests pass. |
| FR-011 | Config-fed complete package at s=200 resumes with validation once at 200, first new training loss 201, B=2 final package 202. |
| FR-012 / SC-006 | Real readers, source-bound caches, fixed SHA-256 noise, Qwen forward/loss, TensorBoard metrics, Accelerate save/load and controlled resume form one short CPU chain. Full H200 verification remains a separate stage. |
| FR-013 | All affected pre-existing Stage 1–3 and legacy tests pass under the combined gate; no production training algorithm was changed in Stage 4. |
| SC-005 | One canonical model.safetensors and complete sidecars in the composed package; Stage 3 strict-best, samples, inclusive retention, current/best resume and duplicate-output tests pass in the same combined suite. |

## T018: Japanese CWD wording

The Japanese closing paragraph in `docs/qwen_image.md` now limits repository-root execution advice to legacy commands without `experiment_dir` and explicitly permits the portable example's absolute `REPO`/`ROOT` commands from another CWD. The English command blocks and CLI contract were unchanged. This is a documentation-only correction; no model or GPU run was needed.

## T019: Config-fed best and inclusive retention

The existing parsed-example CPU chain produced six real TensorBoard metrics at step 200 and a complete current package. The new focused test used the production `decide_best_event` and `promote_published_current_to_best` seams to select exactly one complete best package with those same metrics, then verified it through `read_best_package` and `load_package`. With the parsed `save_last_n_steps=1000`, complete owned current packages at steps 199 and 201 were saved without training updates. `prune_current_packages(..., X=1201)` removed the whole step-199 package, retained the complete inclusive-boundary step-201 package (`X-1000`), and preserved the sole best package.

The focused test passed on its first run because the production behavior was already implemented: **1 passed in 8.28s**. The full example test file passed **16 tests in 10.36s**; Ruff passed. No production code, GPU, or model weights were changed or used.

