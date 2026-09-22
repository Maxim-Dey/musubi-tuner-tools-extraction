# Implementation Plan: Qwen-Image Original LoRA Scope

**Branch**: `001-scope-qwen-image-lora` | **Date**: 2026-09-22 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification and constitution 1.0.0, unchanged.

## Summary

Extract a standalone Qwen-Image original LoRA trainer using the existing algorithms and training loop. Retain Qwen-Image original adapter training, image/caption preparation, both caches, training samples, logging, checkpoint/state saving and existing resume behavior. Remove other models, video/audio, Edit/Layered, full finetuning and unrelated commands only after detaching their retained dependencies. Reuse existing modules; no new training engine, registry or configuration framework.

`hv_train_network.py` re-exports an already extracted trainer: Qwen can import the existing `training/` modules directly. Three small helpers still owned by Hunyuan/Flux must be preserved before those implementations are removed. Shared files must lose excluded branches, not shelter complete foreign implementations under neutral names.

Keep all three templates in `config_for_qwen_image_lora/`. Implementation corrects only `dataset_config` and `sample_prompts` references to this directory; output/log/cache locations remain independent. Preserve the already approved `constant_with_warmup` / 200 correction. The old `constant` / 200 combination raises during scheduler construction and must fail earlier without changing scheduler mathematics.

## Technical Context

**Language/Version**: Python `>=3.10,<3.13`; `.python-version` is 3.10. No migration.

**Primary Dependencies**: Existing PyTorch/torchvision, Accelerate 1.6.0, Transformers 4.57.6, Diffusers 0.32.1, safetensors, TOML, voluptuous, NumPy, Pillow, OpenCV, einops and Hugging Face Hub; applicable optional optimizer/attention/logging packages. See [research.md](research.md).

**Storage**: Image/caption sources, TOML/JSON dataset declarations, safetensors caches/checkpoints, Accelerate state directories, sample PNGs and logs. No new formats.

**Testing**: Source/AST inspection; real-module CPU parser/import and focused regression tests where dependencies exist. No simulated full training.

**Target Platform**: Existing supported training environments; private-server operational verification is a later stage. No H200, path, resolution, rank or run-length restriction derived from examples.

**Project Type**: Python CLI/package; three retained root commands.

**Performance Goals**: Preserve existing computation/resource behavior; no benchmarks or budgets.

**Constraints**: This invocation changes planning documents only. Later local implementation must not run training, GPU workloads, weight downloads, packaging or server transfer. Do not duplicate model tensors during extraction.

**Scale/Scope**: One original model family, existing applicable adapter/options support, multiple image datasets and the existing training loop. Research resolves design decisions; missing runtime dependencies are an explicit verification limitation.

## Constitution Check

| Gate | Before research | After design |
| --- | --- | --- |
| I: Language | Pass | English artifacts; Russian dialogue |
| II: Task fidelity | Pass | Eight retained capabilities covered; no added feature |
| III: Minimal compatible changes | Pass; removal explicitly authorized | Existing layout/loop retained; only necessary helper relocation and validation |
| IV: Training invariants | Required baseline identified | Characterize math, gradients, precision, RNG, save/resume before affected edits; preserve existing resume limitations |
| V: Memory/performance | Pass | No tensor duplication or performance work |
| VI: Configuration/errors | Existing gaps identified | Validate effective defaults → TOML → CLI and dataset/prompt inputs before loaders |
| VII: Verification/docs | Pass | Meaningful tests, explicit unavailable evidence, README audit |
| Local stage/governance | Planning only | Constitution unchanged; later stages and post-converge approval rules unchanged |

These are design gates, not claims that current code already satisfies the spec. No amendment or exception is required. Resume currently reloads Accelerator state but restarts trainer counters; preserve and describe existing supported restoration, without promising a new exact data-cursor resume algorithm.

## Project Structure

```text
specs/001-scope-qwen-image-lora/
  spec.md
  plan.md
  research.md
  data-model.md
  quickstart.md
  contracts/cli-and-config.md
  checklists/requirements.md
qwen_image_cache_latents.py
qwen_image_cache_text_encoder_outputs.py
qwen_image_train_network.py
config_for_qwen_image_lora/
  train.toml
  dataset.toml
  sample_prompts.txt
src/musubi_tuner/
  qwen_image_train_network.py
  qwen_image_cache_latents.py
  qwen_image_cache_text_encoder_outputs.py
  qwen_image/
  training/
  dataset/
  networks/
  modules/
  utils/
tests/
docs/
```

Keep current names; `tasks.md` belongs to the subsequent tasks phase and is not generated here.

## Concrete File Decisions

Package paths below are relative to `src/musubi_tuner/`. The complete reasons and excluded-family inventory are in [research.md](research.md).

| Action | Files/components | Reason |
| --- | --- | --- |
| Keep | Three root Qwen wrappers above; three templates | Existing public contract |
| Adapt | `qwen_image_train_network.py`; `training/trainer_base.py`, `parser_common.py`, `sampling_prompts.py`, `accelerator_setup.py` | Direct imports, original-only execution and early validation; preserve training/sampling/logging/state mechanics |
| Adapt minimally | `qwen_image/qwen_image_modules.py` receives exact `attention`, layout and required backend support from `hunyuan_model/attention.py`; `utils/model_utils.py` receives `flux/flux_utils.py:is_fp8`; `utils/image_utils.py` receives `hv_generate_video.py:save_images_grid` | Detach necessary helpers without replacing algorithms or keeping foreign models |
| Adapt | Both Qwen cache modules; shared `cache_latents.py`, `cache_text_encoder_outputs.py` | Keep orchestration/reuse/cleanup and image previews; remove Hunyuan implementation/imports/parser fields |
| Adapt | `qwen_image/qwen_image_model.py`, `qwen_image_utils.py`, `qwen_image_autoencoder_kl.py` | Retain original DiT/VAE/text/packing/sampling; remove Edit/Layered dispatch and implementations, retain needed single-frame causal/3D VAE math |
| Adapt | `dataset/config_utils.py`, `image_video_dataset.py`, `datasources.py`, `cache_io.py`, `bucket.py`, `architectures.py`, `media_utils.py` | Keep image sources/captions/buckets/repeats/cache contracts; remove video/audio/control/layered paths |
| Keep/adapt | `networks/lora_qwen_image.py`, `lora.py`, `loha.py`, `lokr.py`, `network_arch.py` | Preserve shipped Qwen-applicable adapters; remove Hunyuan factories and foreign architecture detection |
| Keep/adapt | `training/timesteps.py`; `modules/custom_offloading_utils.py`, `fp8_optimization_utils.py`, `lr_schedulers.py`, `scheduling_flow_match_discrete.py`; `utils/train_utils.py`, `lora_utils.py`, `device_utils.py`, `safetensors_utils.py`, `sai_model_spec.py`, `huggingface_utils.py` | Required numerical, loading, optimization, metadata and state behavior; trim only excluded functionality |
| Delete after detachment | Root/package Hunyuan trainer/generator; root generic cache wrappers; `hunyuan_model/`, `flux/` and other model packages listed in research | Qwen dependencies first redirected/extracted |
| Delete | Root/package `qwen_image_train.py`, `qwen_image_generate_image.py`, `qwen_extract_lora.py`; other model entry points/adapters; GUI and unrelated tools; `dataset/audio_utils.py`, `training/audio_loss.py`, exclusive quantization/VAE modules | Excluded workflows; trainer sampling remains available |
| Keep/adapt tests | `test_save_precision.py`, `test_lora_dtype_bridging.py`, `test_grad_metrics.py`, `test_krea2_timesteps.py`, `test_ideogram4_timesteps.py`; image cases in `test_datasource_item_extras.py`; adapt `test_sai_model_spec.py`, `test_top_level_entrypoints.py` | Preserve shared regressions regardless of historical names |
| Adapt docs/examples | `README.md`, `README.ja.md`, `README.ru.md`, `CONTRIBUTING.md`, `CONTRIBUTING.ja.md`, Qwen/shared docs enumerated in research; two template references | One coherent supported workflow, no dead references; retain attribution |
| Adapt dependencies | `pyproject.toml` runtime/optional/dev declarations and stale Ruff paths | Remove exclusive consumers only; no incidental upgrades or removal of supported CUDA options |
| Protect | `.specify/`, `.agents/`, `.cursor/`, `.git/`, constitution, notices/licenses, user-authored `speckit_prompts.md`, datasets/weights/caches/outputs | Outside removal authorization |

No lock file exists in this checkout, by tracked-file and filesystem inspection. Do not create one incidentally. Recheck before implementation: if an existing lock is then present, snapshot it and keep it consistent with the dependency change, without network resolution or unrelated version churn; report unavailable offline lock updates.

## Implementation Sequence

1. **Record the baseline before substantial logic edits.** Snapshot revision/template hashes. Run applicable existing CPU regressions and real parser/import checks when dependencies exist. Characterize effective TOML/CLI values, scheduler conflict, prompts, timestep/loss/LoRA gradients, attention, packing, cache artifacts, sampling boundaries and save/retention/resume. Keep current defects separate from required validation changes. Unavailable execution is not a pass.
2. **Detach required behavior.** Redirect Qwen imports to existing `training/` definitions. Relocate only the three helpers above, preserving interfaces and attribution. Keep inheritance and dynamic adapter factories. Verify dependency closure before deleting source owners.
3. **Narrow and validate.** Remove excluded branches from Qwen/shared code and schemas. Reuse parser actions and current readers to cover all applicable settings, not a 40-key whitelist. Validate at existing entry boundaries before sampling preparation, model loaders, tracker side effects and cache writes. Preserve defaults/precedence, including attention-backend selection when multiple flags are set, and deliberate derived behavior; reject unsupported TOML as well as CLI values. Explicitly validate names inside `network_args` against the selected retained adapter's consumed arguments, preserving its valid options. Follow [contracts/cli-and-config.md](contracts/cli-and-config.md).
4. **Remove excluded closure.** Apply the reviewed component/file inventory; recheck imports, inheritance, dynamic factories, package initialization and remaining execution paths. Remove exclusive tests/docs/assets/dependencies. No filename-mask deletion or neutral-name copy of a foreign implementation.
5. **Align references and verify.** Correct the two template references, preserving all other template values. Update README/docs using [quickstart.md](quickstart.md). Run focused new/retained checks, syntax/import/help checks, link and dependency audits. Report limitations; do not declare local implementation complete with unresolved confirmed regressions or missing required local evidence.

## Verification Design

| Requirements | Minimum evidence before/after affected edits |
| --- | --- |
| FR-009–012: templates/precedence/errors | Actual parser + TOML loader + CLI; assert effective values/defaults. Preserved backend precedence and selected-backend prerequisites; unknown/type/choice errors from TOML separately, including unknown names inside `network_args`; mixed legacy selectors; real prompt readers. Sentinel only at external loader boundary proves early failure |
| FR-002–003: image/cache | Real schema/blueprint/buckets and tiny PNG/TXT/JSONL sources; real small-tensor safetensors writer/reader round trips, variable embedding lengths, repeats, skip/keep behavior |
| FR-004,008,014: numerics/extraction | Existing dtype/LoRA/timestep tests; deterministic small-tensor attention, packing/loss and gradient comparisons; base weights stay frozen |
| FR-005–006: samples/logging | Real prompt/trigger/PNG/metrics helpers; assert parameters, gradients, optimizer/scheduler and Python/NumPy/torch RNG unchanged at exercised boundaries; label narrow generation/tracker substitutions |
| FR-007: save/resume | Real adapter serialization/metadata/precision, retention edge fixtures, small CPU Accelerator save/load of actual parameter/optimizer/scheduler/RNG state and actual filter hooks; characterize trainer counter reset separately |
| FR-013–017: removal/docs | Real imports and `--help` for all three wrappers, short/qualified adapter resolution, no excluded model dispatch/dependency; README examples and input links resolve; protected files unchanged |

Do not replace the complete tested chain with stubs or simulate a full training run. Pure helper/source characterizations do not replace required real-module/parser checks. Preserve current valid inputs; malformed values and excluded selections are deliberate validation changes. Baseline sampling exception/RNG and resume limitations are recorded in research; do not silently redesign them.

## Complexity Tracking

No deviations requiring justification. Existing mechanisms suffice; a generic model/configuration framework is not warranted.
