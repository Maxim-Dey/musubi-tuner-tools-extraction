# Implementation Plan: Krea2 Raw LoRA Only

**Branch**: `codex/krea2-raw-lora-only` | **Date**: 2026-10-26 | **Spec**: [spec.md](spec.md)

## Summary

Keep the existing Raw LoRA algorithm and image/cache contracts. Connect Krea2 directly to the existing common training modules, preserve required helpers before deleting foreign implementations, validate effective configuration before loading models, and provide the three requested starter files.

## Technical Context

Python 3.10–3.12; local .venv is Python 3.12.11, PyTorch 2.7.1 CPU. Existing Accelerate, Transformers/Qwen3-VL, Diffusers scheduler/optimization, NumPy, Pillow/OpenCV, safetensors, TOML, voluptuous and optional optimizer/attention dependencies. CLI project; filesystem datasets, cache, adapters, state and logs. Tests use pytest and real local consumers with tiny CPU tensors/models. No new framework or resource/performance target. No lock file is present; do not generate one through package resolution/building.

## Constitution Check

Pre-research and post-design gates pass: faithful scope, English documents/Russian dialogue, explicitly authorized incompatible removals, preserved training invariants and protected files, no added weight copies, strict effective configuration, meaningful local evidence. No training/GPU/download/build/remote execution. Three GPT-6.1 Sol agents research/implement independent portions and verify independently. One shared maximum of two post-implementation correction rounds is recorded in verification.md.

## Project Structure and Component Decisions

| Components | Decision and reason |
|---|---|
| Root/src krea2_cache_latents.py, krea2_cache_text_encoder_outputs.py, krea2_train_network.py | Keep the three supported commands. |
| training/{trainer_base,accelerator_setup,parser_common,sampling_prompts,timesteps}.py | Keep existing training core; remove video/control/audio-only branches. Move only save_images_grid out of hv_generate_video into sampling_prompts; switch Krea2 away from hv re-exports. |
| krea2/{krea2_mmdit,krea2_encoder}.py | Keep actual Raw DiT and Qwen3-VL. |
| krea2/krea2_utils.py and krea2_train_network.py | Remove Turbo state load/swap/composition; preserve Raw sampling and applicable quantization/offloading. |
| krea2/krea2_sampling.py | Keep roundup, prepare and timesteps used during training; remove standalone sample/encode_prompts/gather_valid_text used only by deleted generation command. |
| qwen_image/{qwen_image_autoencoder_kl,qwen_image_modules}.py | Keep VAE and activation helper. Causal/3D internals are necessary for image F=1 and stay intact. |
| qwen_image/qwen_image_utils.py | Keep VAE key conversion and load_vae only; remove Qwen text encoder/scheduler/edit/generation logic and FLUX dependency. |
| dataset/{architectures,bucket,cache_io,config_utils,datasources,image_video_dataset,media_utils}.py | Preserve directory/JSONL images and captions, resize/buckets/repeats, cache metadata, loading/padding and Krea2 writers. Remove other architectures, video/audio/control/edit source/cache branches. Keep filenames to minimize churn. |
| src cache_latents.py and cache_text_encoder_outputs.py | Keep actual shared image batching/debug/cache lifecycle; remove Hunyuan model operations/main and unused options. Remove generic root wrappers. |
| networks/{lora,lora_krea2}.py | Keep generic LoRA mechanics and actual Krea2 targets; remove Hunyuan factory wrappers. LoHa/LoKr have no Krea2 detection, so do not introduce them as a new feature. |
| modules/{attention,fp8_optimization_utils,convrot_int8_utils,convrot_int8_kernels,comfy_quant_utils,custom_offloading_utils,adafactor_fused,lr_schedulers,scheduling_flow_match_discrete}.py | Keep actual transitive consumers, including optional optimizations. Remove unused nvfp4_utils and unet_causal_3d_blocks. |
| utils/* | Retain actual checkpoint/device/optimizer/logging/model utilities; limit sai_model_spec to Krea2 metadata. Remove unused image preprocessing if no retained consumer exists. |
| Other model dirs and src/root commands, GUI, LoRA conversion/merge/EMA/caption commands | Remove full implementations after preserving required helpers. Preserve license files and required notices. |
| tests, docs, README variants, pyproject.toml | Adapt relevant regressions/docs and delete excluded-only content/dependencies. Preserve user/image data, Git, prompts, Spec Kit and other specs. |

Mathematical timestep distributions named after other models remain because Krea2 uses them without those models. Preserve applicable FP8, ConvRot int8, block swap/pinned transfer, gradient checkpointing/CPU offload, attention, optimizer/scheduler, precision, accumulation, metrics, save and resume options. Names alone are not a deletion criterion.

## Configuration

Use actual argparse definitions for known keys, types and choices. Apply defaults → TOML (existing section flattening) → explicit CLI, including CLI values equal to defaults. Reject unknown keys, duplicate flattened keys and invalid values with their source/key/correction. Validate effective required paths and incompatibilities before prepare_sampling, which loads the text encoder before the DiT. Dataset sanitizer rejects excluded/unknown fields; cache commands validate arguments/datasets before loaders.

Cover Krea2 network aliases, sampling dependencies, FP8/ConvRot exclusions, precision, attention conflicts, block-swap/checkpointing constraints and optimizer/scheduler kwargs. Preserve supported dynamic optimizer/scheduler paths. Fix confirmed applicable-option defects: dim_from_weights must read network_weights; empty optimizer alias must resolve to existing documented AdamW default. Quantized base-LoRA merging must use safe existing load-time merge or an explicit valid incompatibility error, never silent corruption. Remove unused inherited VAE tiling/guidance flags, not functioning optimizations.

Starter files: network_module='networks.lora_krea2', AdamW/constant, bf16, SDPA, gradient checkpointing, rank/alpha 32, 1024 image buckets, batch 1, 1000 example steps, seed 42. Optional FP8/ConvRot/swap remain off in starter. REPLACE_WITH_ identifies user paths/captions. Relative dataset/sample/output paths resolve from repository root. Raw prompt uses --l and a nonempty --n. State saving is enabled. These choices are examples, not limits/hardware claims.

## Execution and Verification

1. Record baseline tests/imports/parsers before code changes in verification.md.
2. Adapt training and data/cache independently; preserve VAE/LoRA helpers before dependency deletion.
3. Remove excluded implementations, wrappers, tests, docs, dependencies; audit all static/dynamic imports.
4. Validate effective configuration; write starter files and README/quickstart commands using PYTHONPATH=src.
5. Run retained pytest suite, real imports/--help, bad CLI/TOML, real cache round trips and tiny CPU model/LoRA checks; inspect sampling RNG and save/resume call paths; check syntax/lint/docs.
6. Independently converge against all requirements. Append confirmed gaps only; fix separately, maximum two rounds total.

Ownership: trainer agent owns training/* except removed audio_loss.py, Krea2 trainer/utils and new trainer tests. Data agent owns dataset/*, four src cache modules, three starter files and new cache tests. Coordinator owns other source removal/adaptation, docs/dependencies and feature records. Verification agent records baseline, analyzes artifacts and independently verifies completed code. Only coordinator updates task/checklist status from evidence.
