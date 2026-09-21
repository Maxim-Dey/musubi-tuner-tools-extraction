# Implementation Plan: Scope the Repository to FLUX.2 Dev LoRA

**Branch**: `main` | **Feature identifier**: `001-scope-flux2-dev-lora` | **Date**: 2026-09-20 | **Spec**: [spec.md](spec.md)

Setup reports the feature identifier as `BRANCH`; Git remains on `main`. This invocation produces design documents only and stops after Phase 1.

## Summary

Keep the existing image/caption -> AE cache -> Mistral cache -> Dev LoRA workflow, including training samples, TensorBoard, weight/state saving and resume. Rewire common imports before deleting excluded implementations. Narrow existing parsers/factories/mixed modules and validate consumed sources and effective configuration before weights. Preserve useful Dev options outside the templates, numerical behavior and artifact formats.

No framework, universal registry, trainer rewrite, optimization or download is needed. Only the existing image-grid saver needs relocation; trainer/parser helpers already have common owners.

## Technical Context

**Language/Version**: Existing Python `>=3.10,<3.13`; `.python-version` is 3.10. Available planning interpreter: 3.12.14. No version migration.

**Primary Dependencies**: PyTorch/torchvision, Accelerate, bitsandbytes, Transformers/Mistral 3, Diffusers utilities, safetensors, Hugging Face Hub, einops, NumPy, Pillow/OpenCV, TOML, Voluptuous and TensorBoard.

**Storage**: Existing images/captions, TOML/JSON/JSONL, safetensors caches/adapters, Accelerate state directories and tracker events. No format migration.

**Testing**: Existing pytest, production parsers/CLI and focused CPU fixtures. Dependency-dependent baseline checks are currently blocked: [baseline.md](baseline.md).

**Target Platform**: Existing Windows/Linux Python setup; later suitable accelerator runtime with external resources. Local source/CPU checks only.

**Project Type**: One `src/musubi_tuner` package with root wrappers.

**Performance Goals**: Preserve behavior; no benchmarks, additional model copies or persistent tensor retention.

**Constraints**: Constitution 1.0.0 unchanged; protect user work; no local training, GPU work, weights download, packaging or server transfer/verification.

**Scale/Scope**: Three operational commands, one training model/method. Existing image conditioning remains; standalone generation/merge/export/conversion/post-hoc EMA/caption generation and Self-Flow are excluded.

## Constitution Check

| Gate | Before research | After design |
| --- | --- | --- |
| I: Language | Pass: English documents, Russian dialogue | Pass |
| II: Task fidelity | Pass: FR-020 scoped the previous specify invocation; planning is now requested | Pass: no implementation/tasks generation |
| III: Minimal compatibility | Pass: excluded removals explicitly approved | Pass: existing owners, one helper move, no redesign |
| IV: Invariants | Pass: baseline comparison required | Pass for design: algorithm/state changes prohibited; existing resume limitation visible |
| V: Memory/performance | Pass | Pass: no added model retention/benchmark |
| VI: Configuration | Existing gaps identified | Pass for design: source and final-value validation before weights |
| VII: Verification/docs | Available baseline established | Pass for design: relevant tests/all README variants; blocked checks not passed |
| Local stage/governance | Pass | Pass: no external run/amendment; subsequent converge approval rules unchanged |

Design gates do not mean implementation acceptance. The existing resume-counter limitation in baseline prevents a claim that full progress restoration has been established. Preserve baseline behavior; report unmet acceptance requirements without silently changing the specification or repairing resume during deletion.

## Project Structure

```text
specs/001-scope-flux2-dev-lora/
  spec.md, plan.md, research.md, baseline.md
  data-model.md, quickstart.md, contracts/cli-and-config.md
  checklists/requirements.md
flux2dev_lora/
  dataset.toml, train.toml             # existing protected user files
  sample_prompts.txt                   # later implementation deliverable
flux_2_cache_latents.py
flux_2_cache_text_encoder_outputs.py
flux_2_train_network.py
src/musubi_tuner/
  flux_2_cache_latents.py, flux_2_cache_text_encoder_outputs.py, flux_2_train_network.py
  cache_latents.py, cache_text_encoder_outputs.py  # internal helpers only
  flux_2/, training/, dataset/, modules/, networks/, utils/
tests/
docs/
```

Keep existing filenames and package structure. No `tasks.md` in this phase. Protect `.specify`, `.agents`, `.cursor`, `.github`, contributor tooling, attribution, user data and unrelated edits.

## Dependency audit and component inventory

The following inventory was prepared after tracing all three wrappers/package entrypoints, static and dynamic imports, network factories, architecture/bucket registrations, config/dataset factories, cache I/O, sampling and resume. [research.md](research.md) records source evidence. Paths below are package-relative unless stated otherwise.

### Retain and change

| Components | Action and reason; removed branches |
| --- | --- |
| Three FLUX.2 cache/train modules and root wrappers | Retain names/module execution. Only `dev`; early validation. Cache commands consume dataset TOML/JSON, not training TOML. |
| `flux_2_train_network.py` imports | Replace `hv_train_network` re-exports with `training.trainer_base` (`DiTOutput`, `NetworkTrainer`), `training.accelerator_setup` (`clean_memory_on_device`), `training.sampling_prompts` (`load_prompts`), `training.parser_common` helpers. |
| `training/trainer_base.py` | Preserve loop, noise/loss/timesteps, gradients/accumulation, optimizer/scheduler, logging/save/state/resume. Remove video/audio construction and sampling branches, video frame-grid handling and excluded network plugin fallback. Preserve singleton `(B,C,1,H,W)` image output, RNG restoration and swapping. |
| `hv_generate_video.py` -> `utils/image_utils.py` | Move only `save_images_grid` unchanged before deleting owner. Preserve torchvision grid/pixel conversion, PNG naming/tracker behavior; delete video saver. |
| `cache_latents.py` | Retain common parser, `encode_datasets`, image/console preview helpers. Remove Hunyuan imports/preprocess/encode/main/parser additions, module execution and video preview/save. FLUX already owns preprocessing/encode. |
| `cache_text_encoder_outputs.py` | Retain common parser, path preparation, batching and cleanup. Remove Hunyuan imports/encode/main/parser additions/module execution. Preserve skip/keep-cache behavior. |
| `flux_2/flux2_models.py` | Keep Dev DiT/AE, attention/checkpointing/swapping/precision/state keys. Remove Klein4B/Klein9B parameter classes and Klein-only absence-of-guidance branches. |
| `flux_2/flux2_utils.py` | Keep Dev registry, loaders, Mistral, packing/coordinates, `compress_time`, control images, sampling schedule/denoise. Remove four Klein entries, Qwen imports/embedder/loader arm, zimage import, Klein CFG denoiser/unused guidance helper. Keep Mistral supporting formatting/image helpers; no zimage implementation relocation. |
| `networks/lora_flux_2.py`, `networks/lora.py` | Keep FLUX targeting and generic LoRA/dtype/dropout/parameter groups/I/O. Remove Hunyuan constant/factory wrappers from generic LoRA. Keep inference/merge internals used by existing `base_weights` training. |
| `utils/lora_utils.py` | Keep safetensors/FP8 loader and retained ordinary LoRA internals. Remove LoHa/LoKr detection/dispatch/imports and standalone-only `attach_lora_weights`/`filter_lora_state_dict`. No non-LoRA weight-selection bypass. |
| `training/parser_common.py`, `sampling_prompts.py`, `accelerator_setup.py` | Narrow existing schemas; add source/final validation and early dependency/prompt/tracker-TOML checks. Preserve arbitrary one-level training TOML grouping while validating every contained parameter. Validate consumed `log_tracker_config` readability, syntax, tracker sections and supported initialization parameters before weights; keep tracker initialization in `trainer_base.py` at its existing point. Preserve dtype/DDP/log directories/prefix/tracker names. Remove video/audio/standalone options, not generic Dev options. |
| `training/timesteps.py`, common timestep branches | Retain numeric options usable by Dev: `flux_shift`, `qwen_shift`, `krea2_shift`, `ideogram4_shift`, `qinglong_*`, SD3-named weighting and distribution/bucket controls. Names do not select models. |
| `dataset/config_utils.py` | Keep image dataclasses, Voluptuous, precedence. Remove video/audio/FramePack-only fields and video factory fallback; require image source. Keep legacy aliases that only map to retained image-control settings. |
| `dataset/{image_video_dataset,datasources,media_utils}.py` | Keep image directory/JSONL/caption/control handling, relative paths/order/repeats/workers/transforms. Remove video sources/dataset/audio and excluded architecture branches, including FramePack target sequencing. Preserve Dev reference images outside templates; no cosmetic rename. |
| `dataset/{architectures,bucket,cache_io}.py` | Keep `f2d`/`flux_2_dev`, Dev bucket behavior, Dev latent/text writers and shared I/O with exact fields/dtypes/metadata. Remove other registrations/serializers/video grid and eager MiniMax imports before deleting owner. |
| `modules/{attention,custom_offloading_utils,fp8_optimization_utils,lr_schedulers,scheduling_flow_match_discrete}.py` | Direct Dev/common dependencies; keep optional backends and generic flow scheduler. Do not run CUDA diagnostic examples locally. |
| `utils/{device_utils,model_utils,safetensors_utils,train_utils,huggingface_utils}.py` | Keep dtype/compile/weight/shard/hash/save/retention/state/Hub behavior. Internal conversion is not a standalone utility. |
| `utils/sai_model_spec.py`, package initializers | Keep Dev LoRA metadata/user overrides and package boundaries; remove other architecture branches, unused standalone merged metadata helpers and obsolete re-exports. |
| Root `flux2dev_lora/train.toml`, `dataset.toml` | Preserve all active values. Only shipped-path changes: `./flux2dev_lora/dataset.toml`, `./flux2dev_lora/sample_prompts.txt`. Add nonempty prompt in implementation; document external substitutions. |
| All README variants and common docs | Update README.md/.ja.md/.ru.md and `docs/{flux_2,dataset_config,advanced_config,sampling_during_training,block_swap,torch_compile}.md`, package dataset docs. Extract shared guidance from deleted docs; fix links and scope claims. |

### Delete after rewiring

Use exact reviewed tracked paths, not wildcard deletion of user/untracked content.

| Components | Reason/prerequisite |
| --- | --- |
| Tracked root model commands except the three retained wrappers | Excluded workflows. Root generic cache commands are Hunyuan commands; narrowed package cache helpers remain without `main`. |
| Package model commands except three retained entrypoints/two internal cache helpers | Remove excluded architectures' cache/train/generate commands and full trainers, `flux_2_generate_image.py`, `flux_2_train_network_self_flow.py`; `hv_*` only after common dependencies detached. |
| `hunyuan_model/`, `hunyuan_video_1_5/`, `wan/`, `frame_pack/`, `flux/`, `qwen_image/`, `zimage/`, `kandinsky5/`, `hidream_o1/`, `ideogram4/`, `krea2/`, `minimax_h3/` | Excluded implementations; all identified Hunyuan/Qwen/MiniMax imports handled above. |
| `gui/` | Existing Qwen/ZImage cache/train/conversion UI; no Dev GUI migration. |
| Root/package `merge_lora.py`, `convert_lora.py`, `lora_post_hoc_ema.py`, `caption_images_by_qwen_vl.py`, `qwen_extract_lora.py` | Explicit standalone exclusions, including module execution/settings such as `save_merged_model`. |
| `networks/{loha,lokr,network_arch}.py`, other architecture `lora_*.py` except `lora_flux_2.py`, converter modules | Excluded methods/targeting. Generic `lora.py` remains; architecture detector is only used by LoHa/LoKr. |
| `dataset/audio_utils.py`, `training/audio_loss.py` | No retained audio objective; detach consumers first. |
| `modules/{adafactor_fused,unet_causal_3d_blocks,comfy_quant_utils,convrot_int8_utils,convrot_int8_kernels,nvfp4_utils}.py` | Callers only excluded full trainers/VAEs/Krea/MiniMax. Ordinary Adafactor optimizer remains. |
| Other-model docs/tests/examples/assets, LoHa/LoKr/GUI docs, standalone sections in `docs/tools.md` | Remove after extracting shared Dev guidance/assertions; delete tools doc if nothing retained. Keep shared timestep figures and attribution assets; remove exclusive post-hoc EMA/kisekaeichi examples. |

### Dependency changes

| Decision | Actual use |
| --- | --- |
| Keep runtime pins | Accelerate, bitsandbytes, Diffusers, einops, Hub, Pillow, safetensors, toml, tqdm, Transformers, Voluptuous are consumed by retained paths. |
| Keep OpenCV/torchvision | `media_utils.resize_image_to_bucket` uses `cv2.INTER_AREA`; grid saver uses `torchvision.utils.make_grid`. NumPy/packaging currently arrive transitively; verify them in install smoke. |
| Keep sentencepiece | Existing tokenizer support/fallback; historical Kontext comment is not removal evidence. Preserve processor/tokenizer selection; AutoProcessor `use_fast=False` alone does not prove slow tokenization. |
| Ensure TensorBoard/pytest | Explicit pip installation in quickstart: TensorBoard only in dev group, pytest not declared. Do not disable active logging. |
| Keep tools/options | Ruff/CI/Spec Kit, ascii-magic for console image previews, matplotlib for timestep plots; optional W&B/attention/optimizer/scheduler/image-codec guidance. |
| Remove after imports clean | `av`: video/audio/debug only; `ftfy`: Wan tokenizer; `easydict`: Wan config; `prompt-toolkit`: standalone interactive generation. |
| Remove excluded extras | `gui`/Gradio, `hidream_o1`/sensecraft. Preserve CUDA extras/source config; prune Ruff exclusions only for deleted files. |

Confirm the narrowed installation without removed packages using actual imports/processor checks when equipped. No successful dependency solve is claimed here; research/quickstart distinguish metadata availability from runtime evidence.

## Minimal implementation sequence

1. Freeze baseline revision, active template values and protected files. Repeat [available checks](baseline.md); establish missing production-parser/CPU-test results before deletion once dependencies are available. Keep pre-existing failures separate.
2. Rewire common helpers, move image saver, detach Hunyuan/MiniMax cache imports. Check imports before deleting owners.
3. Prune inventoried mixed branches. Validate original sources, apply existing valid defaults/TOML/CLI precedence, then validate effective values. Preserve arbitrary one-level training groups and flattening order. Parse dataset/prompts and consumed `log_tracker_config` TOML and check selected dependencies before sampling/DiT/AE/Mistral loads. Validation starts no tracker and performs no RNG-consuming work; preserve valid initialization order.
4. Apply reviewed tracked deletions, narrow dependencies, correct paths/add prompt/update docs. No cleanup of user model/data/output directories.
5. Run focused comparisons, inspect remaining execution/factory paths, check all active template values and dependency closure. Report unavailable checks and discrepancies without claiming full acceptance.

## Local verification matrix

| Area | Required evidence |
| --- | --- |
| Existing tests | Retain/run `test_save_precision.py`, `test_lora_dtype_bridging.py`, `test_grad_metrics.py`. Preserve image/JSONL/path/order assertions in mixed `test_datasource_item_extras.py` using Dev; remove video/Qwen-only setup after recording baseline. Inspect assertions before deleting architecture-named tests. Replace excluded-command checks in `test_top_level_entrypoints.py` with retained/removal checks; cover Dev metadata when narrowing `test_sai_model_spec.py`. |
| Imports/CLI | Actual imports of retained modules/both adapter spellings; six root/module `--help` calls without weights. Inspect dynamic imports/factories/registries/`__main__`. AST supplements imports. |
| Real templates | Production training parser + `read_config_from_file` + final validator: compare all 42 original active values except two paths. Production `load_user_config`/`ConfigSanitizer`/`BlueprintGenerator.generate` with real dataset and `ARCHITECTURE_FLUX_2_DEV`; `load_prompts` on shipped prompt. No copied schema or syntax-only substitute. |
| Negative inputs | CLI/TOML/combined: four Klein values, other architectures, generic/other-model/LoHa/LoKr/LyCORIS selectors, nested excluded algorithms, `save_merged_model`, video/audio/Self-Flow, misspelled/unknown parameters and malformed structures/types. Reject unknown sections in dataset/prompt/tracker schemas; preserve arbitrary one-level training grouping. Positive cases include `[model]`/`[lora]`, unchanged flattening/CLI precedence and valid tracker TOML; unknown contained parameters still fail despite valid CLI overrides. Cache commands test CLI plus dataset, not new training-TOML support. |
| Early failure | Invoke real entrypoints with weight loaders replaced by sentinels that fail if reached. Invalid config/dataset/TXT/TOML/JSON prompts, consumed `log_tracker_config` TOML, FP8 Mistral and missing selected dependencies must fail first with source/key/value/correction. Tracker cases cover missing/unreadable files, malformed TOML, unknown tracker sections/initialization arguments and invalid structures/values; no tracker initialization or service access. No permissive validator stubs. |
| Data/cache | Tiny RGB/caption/control fixtures: compare buckets/resize/order/repeats/JSONL paths. Synthetic tensor round-trip of existing latent/control/text writers/readers, exact keys/metadata/dtype/shape. Skip/keep cleanup in temporary dirs only. |
| Math | Small CPU tensors/toy FLUX-shaped block and LoRA: fixed-RNG baseline comparisons of touched packing, `noise-latents`, noise/timesteps/weighting, dtype/dropout/gradients, accumulation and optimizer/scheduler order. Include retained non-template numeric modes. CPU AdamW does not verify CUDA AdamW8bit. No full training run. |
| Samples/logs/state | Tiny decoded tensor through moved PNG saver with unchanged pixels/name/tracker call; 0/250 triggers, update intervals, save hooks, retention/RNG restoration. Small Accelerate CPU toy-state round-trip where available; expose baseline counter limitation rather than inventing a fix. Small TensorBoard event write/read. |
| Scope/protection | Zero excluded implementations/factories/registrations, not zero historical formula names. Review README variants/links/package use/protected files. Existing Ruff on changed retained Python, unrelated findings visible. |

Use a few focused pytest tests and temporary fixtures for changed boundaries. No test framework, generalized config engine, benchmark suite or full-model substitute.

## Phase outputs and completion boundary

Phase 0: [research.md](research.md), [baseline.md](baseline.md). Phase 1: [data-model.md](data-model.md), [contracts/cli-and-config.md](contracts/cli-and-config.md), [quickstart.md](quickstart.md). Technical decisions are resolved; unavailable runtime evidence remains explicit.

Stop before tasks/implementation. Operational acceptance requires the separate real-model scenario in quickstart; local checks cannot establish real-model reproducibility, GPU compatibility or complete resume correctness.

## Complexity Tracking

No constitutional exceptions or additional abstractions. Environment/resume limitations are recorded, not waived.
