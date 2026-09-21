# Research: dependency-aware Dev LoRA narrowing

Date: 2026-09-20. Source baseline: `bf478828ffa8b5ef3ebdf225ee72431bdb463159` and the two user templates. All code references below are relative to the repository. The research phase inspected current source rather than relying on historical names or dependency comments.

## 1. Keep the existing training architecture

**Decision:** Keep the three FLUX.2 operational entrypoints and existing package layout; detach shared functionality before deleting excluded owners.

**Rationale:** The root wrappers delegate directly to their package `main` functions. `src/musubi_tuner/flux_2_train_network.py:11-18` imports through `hv_train_network`, whose imports at lines 20-51 re-export existing `training.trainer_base`, `training.accelerator_setup`, `training.sampling_prompts` and `training.parser_common`. No trainer implementation needs moving. However, `training/trainer_base.py:48` imports image and video savers from `hv_generate_video`. Its `save_images_grid` at lines 117-147 is a necessary image function without a suitable existing replacement; relocate it unchanged into `utils/image_utils.py`.

The cache commands also have hidden ownership: `flux_2_cache_latents.py:12` uses mixed `cache_latents`; `flux_2_cache_text_encoder_outputs.py:11` uses mixed `cache_text_encoder_outputs`. Those helpers eagerly import Hunyuan loaders even though FLUX supplies its own model encoding callbacks. Keep common loops/parser/image previews in place; remove Hunyuan imports/encoding/entrypoints. `dataset/cache_io.py:24-25` eagerly imports MiniMax packing/text code; remove the excluded serializers before deleting MiniMax.

**Alternatives considered:** Keeping Hunyuan files for a common function would preserve excluded executable support. Copying/replacing the trainer or creating a cache framework adds needless risk. Both rejected.

## 2. Prune actual model branches, preserve generic math

**Decision:** Keep Dev DiT/AE/Mistral, common LoRA, image/control preprocessing and all existing numeric Dev options. Remove excluded model classes, loaders, registrations and factories, not historical names.

**Rationale:** `flux_2/flux2_utils.py:69-111` registers four Klein variants plus Dev; its Qwen loader imports `zimage.zimage_utils.load_qwen3` at line 36 and instantiates Qwen through lines 812-816. Remove Qwen/Klein registry, embedder and model-param classes. Dev uses Mistral and the guidance-distilled denoiser; the non-distilled CFG arm belongs to Klein base. `pack_control_latent` calls `listed_prc_img`; image reconstruction calls `scatter_ids` and `compress_time`. Their time-coordinate naming is used for image references and does not imply an executable video model.

`networks/lora_flux_2.py:13,41,70` directly imports/calls generic `networks.lora`. Keep its common module/dtype/dropout/weight behavior; remove Hunyuan-specific factory wrappers at `lora.py:369,962`. `trainer_base.merge_base_weights` uses `create_arch_network_from_weights(...,for_inference=True)` and `merge_to`, so ordinary LoRA inference internals remain necessary for training. Standalone merge/export callers and LoHa/LoKr dynamic branches in `utils/lora_utils.py:207,211` are excluded. `networks/network_arch.py` serves LoHa/LoKr, not the retained adapter.

`trainer_base.sample_timesteps` directly implements numeric `qwen_shift`, `krea2_shift`, `ideogram4_shift` and Qinglong options; common SD3-named weighting also serves Dev. Removing them by name would violate the preserved non-template option scope. The same applies to ordinary Adafactor versus `modules/adafactor_fused.py`, which is exclusive to removed full trainers.

**Alternatives considered:** A Dev-only parser limited to the 42 template keys would remove valid optional behavior. Retaining every generic factory would permit other methods. Use the existing Dev factory with explicit accepted spellings and validated nested options instead.

## 3. Image datasets, cache identity and execution surfaces

**Decision:** Narrow existing image/config/cache modules and registrations in place. Retain image directory/JSONL paths, captions, controls, cache formats and cleanup semantics.

**Rationale:** `dataset/architectures.py:10-11` defines `f2d` and `flux_2_dev`; `bucket.py:57` maps all architecture resolutions and already rejects unknown architectures at lines 84-87/141-144. Reduce the map/constants to Dev. `config_utils.BlueprintGenerator.generate` currently falls back to `VideoDatasetParams`; reject missing/invalid image source after removing that branch.

Dev uses `cache_io.save_latent_cache_flux_2` and `save_text_encoder_output_cache_flux_2`, with shared metadata/writers. Keep latent/control shape-qualified keys, text dtype key, architecture metadata and existing filenames. `image_video_dataset.py` consumes these caches through its image path; retain that path rather than renaming/rebuilding the module. Remove video/audio/FramePack/other-model branches and serializers.

Audit distributed root wrappers, package `main`/`__main__`, GUI launchers, `importlib.import_module`, `network_arch`, LoHa/LoKr and weight-method dispatch, not just CLI help. Internal helper modules may remain importable but must not retain excluded operational `main` functions. Other model directories and quantization/VAEs were checked for retained consumers; prerequisites for their deletion are enumerated in plan.md.

**Alternatives considered:** File-prefix deletion before resolving imports would break common dataset imports. A new registry is unnecessary; current maps and explicit Dev validation suffice.

## 4. Validate original sources and the effective configuration

**Decision:** Extend existing parsers/loaders with small command-specific checks. Validate raw keys/structure, preserve valid merge precedence, then validate effective values before all model loaders. Keep source provenance only for diagnostics.

**Rationale:** `training/parser_common.py:787-818` flattens TOML sections into an argparse Namespace; unknown attributes survive and pre-existing Namespace values bypass CLI choices. `docs/advanced_config.md:65,107` explicitly permits arbitrary section names for grouping. Its `[optimizer]`, `[training]`, `[output]` example is not an exhaustive schema. Preserve flat keys and arbitrary one-level grouping, including `[model]` and `[lora]`, while validating every contained parameter before flattening. Reject unknown/removed parameters and unsupported deeper structures, retaining existing flattening and override order. A valid CLI override cannot hide an unknown source parameter. This strictness change does not authorize restricting valid grouping labels.

Dataset validation already uses Voluptuous; retain it and improve source-qualified errors. `sampling_prompts.line_to_prompt_dict` currently warns and ignores unknown switches, and JSON/TOML prompt dictionaries lack a strict key boundary. Parse/validate them before `process_sample_prompts` loads Mistral. Guard the selected network before `_build_network`, which currently runs after model loading. Validate recognized nested adapter arguments rather than passing arbitrary `**kwargs`. Keep existing optimizer/scheduler extensibility for standard Dev LoRA; validate their imported constructor arguments without instantiating a model or changing optimizer defaults.

`training/parser_common.py:244-247` also exposes `log_tracker_config`; `training/trainer_base.py:2042-2047` reads that TOML immediately before `accelerator.init_trackers`, after model loading. Include this existing source in the pre-weight boundary: check readability, TOML syntax/structure, recognized tracker sections and backend-supported initialization parameters/values without constructing trackers or contacting services. Preserve valid nested backend payloads and the existing tracker initialization behavior. Positive and negative temporary-file cases must exercise production validation with untouched loader sentinels.

Mistral FP8 currently raises inside its constructor after loading weights. Reject `fp8_text_encoder=true` early; accept the mandatory explicit false. Missing TensorBoard or bitsandbytes must fail early rather than change logging or the optimizer.

**Alternatives considered:** Narrower argparse `choices` alone does not cover TOML. A universal schema/registry, new config library or new validation-only public CLI is unnecessary.

## 5. Dependencies and an installation that covers sampling

**Decision:** Keep runtime pins and CUDA alternatives; remove only dependencies with exclusively excluded consumers. Explicitly include TensorBoard and pytest in the documented pip setup. Preserve tokenizer selection and sentencepiece until a real processor smoke justifies any further change.

**Rationale:** OpenCV is used by `dataset/media_utils.py:133` for image downsampling; torchvision by the moved image saver. bitsandbytes is the actual template optimizer dependency (`trainer_base.py:239-247`). `av` is used for video/audio and video previews, `ftfy` only by Wan tokenizers, `easydict` only by Wan configs, `prompt-toolkit` only by standalone interactive generation. Gradio and sensecraft extras belong to excluded GUI/HiDream. ascii-magic and matplotlib support retained optional previews/timestep plots. Keep optional W&B/attention/custom optimizer/scheduler/image-codec dependencies documented when those options are selected.

The proposed Python 3.12, torch 2.7.1/torchvision 0.22.1 CUDA 12.8 example uses a published Windows/Linux pair already permitted by the project. It does not upgrade project requirements. [PyTorch installation archive](https://pytorch.org/get-started/previous-versions/).

bitsandbytes documents Python >=3.10, torch >=2.4 and compatible Windows/Linux CUDA builds including 12.8; hardware/driver suitability remains an operational prerequisite. [bitsandbytes installation](https://huggingface.co/docs/bitsandbytes/main/en/installation).

Mistral's processor is loaded using the hardcoded ID `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (`flux2_utils.py:48,605`), independently of the checkpoint directory. Its resources must be accessible online or in the normal HF cache; no `--tokenizer_path` exists. The official resource set includes processor/preprocessor/config, tokenizer JSON/config, special tokens and chat template; do not invent a required `tokenizer.model`. The tokenizer configuration selects `LlamaTokenizerFast`, so `AutoProcessor(use_fast=False)` alone does not establish slow-tokenizer use. [Mistral resources](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/tree/main), [tokenizer configuration](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/raw/main/tokenizer_config.json).

TensorBoard is currently only in the dev group, so `pip install -e .` alone is incomplete for the template. Explicit installation covers it; no need to move it into core dependencies. NumPy/packaging and processor dependencies are checked by normal imports plus `pip check`. No package solve, processor load or installation succeeded locally because none was attempted; available evidence is source and official metadata, not runtime verification.

**Alternatives considered:** Disabling samples/logging, dropping OpenCV/sentencepiece by comments, adding Qwen caption packages, switching tokenizers or adding mistral-common/tiktoken without a consumer are rejected.

## 6. Baseline and limits

**Decision:** Preserve [baseline.md](baseline.md), repeat comparable checks before/after implementation, and leave blocked/unverified evidence explicit.

**Rationale:** No local venv exists and neither available interpreter has torch/pytest. Real parser/import attempts failed before execution; 336 tracked Python files and both TOML syntaxes passed source checks. Source-isolated prompt/trigger/retention diagnostics passed; an unknown prompt option was demonstrably ignored. Source inspection also found resume loads state and then resets local epoch/global-step counters. This existing limitation is not a cleanup regression or authorization to redesign resume.

**Alternatives considered:** Reporting missing dependencies as test passes, using copied validators, masking the resume discrepancy or treating synthetic checks as real-model verification would invalidate acceptance evidence. No unresolved technology choice remains; unavailable runtime evidence is recorded as such.
