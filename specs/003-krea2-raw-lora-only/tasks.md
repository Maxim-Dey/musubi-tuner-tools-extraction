# Tasks: Krea2 Raw LoRA Only

Follow spec.md, plan.md and constitution 2.0.0. Only local CPU/no-weight/no-network checks. Coordinator owns this file and the shared correction counter in verification.md. All paths below are repository-relative.

## Phase 1: Baseline

- [X] T001 Record existing environment, protected files, lock/dependency inventory, real Krea2 imports/parser and relevant pytest baseline in `specs/003-krea2-raw-lora-only/verification.md` before code edits (FR-010–012); distinguish fails/skips from passes.

## Phase 2: Preserve necessary shared components

- [X] T002 [P] Connect `src/musubi_tuner/krea2_train_network.py` directly to `training/{trainer_base,accelerator_setup,parser_common,sampling_prompts}.py`; preserve image-save helper in sampling_prompts before removing Hunyuan dependencies (FR-001–005); actual imports must succeed after deletion.
- [X] T003 [P] Retain VAE conversion/load functions in `src/musubi_tuner/qwen_image/qwen_image_utils.py` and necessary VAE/activation internals; remove foreign imports while preserving normalization and key mapping (FR-001,005); compare functions and import actual loader.
- [X] T004 [P] Adapt `src/musubi_tuner/dataset/{architectures,bucket,cache_io,config_utils,datasources,image_video_dataset,media_utils}.py` and src `cache_latents.py`/`cache_text_encoder_outputs.py` to actual Raw image flow, preserving directory/JSONL captions, repeats, buckets, cache keys/metadata/varlen semantics (FR-001,004–005); verify actual image-to-cache-to-loader round trip.

## Phase 3: US1 — Raw training preservation

Goal: preserve complete training workflow and optimizations. Independent check: tiny CPU regressions plus real imports and reviewed hooks.

- [X] T005 [US1] Remove Turbo code from `src/musubi_tuner/krea2_train_network.py` and `krea2/krea2_utils.py` while preserving Raw samples and cleanup/RNG restoration (FR-002–004); existing timestep/checkpointing/ConvRot tests and targeted sample checks pass.
- [X] T006 [US1] Retain loss/noise/gradient/precision/optimizer/scheduler, logging, adapter save, Accelerate state/resume and applicable optimization paths in `src/musubi_tuner/training/trainer_base.py`; remove video/audio/control branches only, fix demonstrated dim_from_weights/default-optimizer/unsafe-merge defects without dropping options (FR-002–003); check relevant CPU tests and save/resume hooks.
- [X] T007 [US1] Add meaningful regressions in `tests/test_krea2_training_config.py` and `tests/test_krea2_cache_contract.py` (or equivalently focused files) for real parser/import/cache and changed behavior; use existing `test_krea2_timesteps.py`, `test_krea2_gradient_checkpointing.py`, `test_krea2_convrot_int8.py`, `test_grad_metrics.py`, `test_save_precision.py`, `test_lora_dtype_bridging.py` (FR-001–003,011); no blanket stubs.

## Phase 4: US2 — Excluded workflow removal and early errors

Goal: retain no excluded implementations or dependency leaks. Independent check: real module/entry-point imports and effective-config rejection tests.

- [X] T008 [P] [US2] Remove excluded root/src wrappers and model directories (`flux`, `flux_2`, `frame_pack`, `hidream_o1`, `hunyuan_model`, `hunyuan_video_1_5`, `ideogram4`, `kandinsky5`, `minimax_h3`, `wan`, `zimage`, GUI), Qwen model/commands, dataset/audio and training/audio helpers after T002–004; preserve Git/licenses/user data/Spec Kit/prompt (FR-004–006,010); retained module inventory has no broken imports.
- [X] T009 [P] [US2] Narrow `src/musubi_tuner/networks/lora.py`, keep `lora_krea2.py`, remove other adapter factories/network modules and unused standalone functions in `krea2/krea2_sampling.py`; narrow `utils/sai_model_spec.py` to Krea2, remove unused modules/utils after usage check (FR-004–005); tiny LoRA/metadata tests preserve contracts.
- [X] T010 [US2] Validate defaults → TOML → explicit CLI in `src/musubi_tuner/training/parser_common.py` and Krea2 trainer before sampling/model loading; unknown keys/type/choice/duplicate flattened keys, foreign networks/Turbo, incompatible precision/attention/quantization/swap/optimizer-scheduler settings fail with source/cause/correction, while dynamic applicable optimizer/scheduler support and valid settings remain (FR-008); actual parser tests include CLI values equal to defaults.
- [X] T011 [US2] Remove irrelevant flags and validate actual arguments/dataset/sample inputs in src Krea2 cache modules, `dataset/config_utils.py`, `dataset/datasources.py`, `training/sampling_prompts.py` (FR-004,008); reject video/audio/control/edit/unknown fields before weights, retain valid image sample formats.
- [X] T012 [P] [US2] Remove excluded-only tests/docs/examples and dependencies in `pyproject.toml`; inspect absent lock files, adapt `tests/test_top_level_entrypoints.py`/`test_sai_model_spec.py` and preserve relevant test assertions (FR-006,010–011); all remaining tests target retained code and required license notices remain.

## Phase 5: US3 — Starter configuration and documentation

Goal: a coherent root-invoked workflow. Independent check: actual parsers and file/command references.

- [X] T013 [US3] Write `config_for_krea2_raw_lora/{train.toml,dataset.toml,sample_prompts.txt}` using actual supported fields, marked user paths/caption, compatible optimizer/scheduler/precision, Raw CFG and state saving (FR-007–009); parse each through real consumers and check cross-references.
- [X] T014 [US3] Rewrite `README.md` and relevant `docs/{krea2,dataset_config,advanced_config,sampling_during_training,block_swap,torch_compile}.md`, align feature `quickstart.md`, and remove stale README variants while preserving licensing (FR-009–010); document dataset, both caches, train, Raw sampling/logging, save/resume and optional optimizations with valid root commands, without hardware promises.

## Phase 6: Final checks and convergence

- [X] T015 Run all retained tests, all retained module imports and three root commands' --help, targeted bad CLI/TOML/removed-mode tests, starter consumers, syntax/lint/import/dependency and README link/command checks; record exact results in `specs/003-krea2-raw-lora-only/verification.md` (FR-001–012, SC-001–005).
- [X] T016 Independently assess every requirement and plan decision against actual code via speckit-converge, append only proven nonduplicate gaps to this `tasks.md`, record results separately and respect the shared two-round limit (FR-011–012); do not declare incomplete checks passed.

## Dependencies and parallel execution

T001 precedes source edits. T002/T003/T004 run in distinct owned files. T005–007 preserve US1 before deleting dependencies. T008/009/012 follow preservation; T010/011 run with their assigned owners. T013 follows stable parsers; T014 follows agreed configuration. T015/016 close all stories. Within US1 trainer and data/cache tests are independent files; within US2 coordinator cleanup and agent parser changes are independent. Only coordinator updates this file. Initial implementation must cover all stories, not stop at a partial MVP.
