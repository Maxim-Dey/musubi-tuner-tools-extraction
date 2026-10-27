# Research

## Existing common trainer

Decision: Krea2 imports directly from `training` modules. `hv_train_network.py:20–48` already re-exports the common components. `trainer_base.py` still imports `save_images_grid`/video saving from Hunyuan generation; relocate only image saving. Copying or retaining the full Hunyuan implementation is unnecessary.

## Image and cache extraction

Decision: Narrow existing dataset files in place. Preserve directory/JSONL captions, image resizing/buckets/repeats, cache metadata and varlen behavior. Both shared cache modules import Hunyuan; `cache_io.py` imports MiniMax. Actual image imports currently fail through `audio_utils.py` on missing `av`. Remove these unused chains instead of installing video dependencies or mocking imports.

Latents use `{basename}_{width:04d}x{height:04d}_kr2.safetensors`, `latents_1xHxW_{dtype}`, `(16,1,H,W)`, architecture `krea2` and format `1.0.1`. Text uses `{basename}_kr2_te.safetensors` and `varlen_krea2_vl_embed_{dtype}`, `(valid_len,12,2560)`. Preserve these contracts and caption metadata.

## Required models and dependencies

Decision: Keep Qwen3-VL through Transformers, Qwen VAE/activation plus VAE key conversion/load functions. VAE causal/3D operations remain required for image frame-axis F=1. Remove Qwen text/edit/generation logic. Keep actual generic LoRA mechanics and Krea2 targets. `krea2_sampling` standalone sample/encode/gather functions have no training consumer; keep only roundup/prepare/timesteps. Mathematical timestep distributions remain usable even when their names reference removed models.

Keep accelerate, bitsandbytes, diffusers, einops, huggingface-hub, opencv-python, pillow, safetensors, toml, tqdm, transformers, voluptuous, torch/torchvision/CUDA extras. Keep ruff, tensorboard, ascii-magic and matplotlib for actual optional consumers. Remove av, ftfy, easydict, sentencepiece, gui/gradio and hidream/sensecraft extras, and prompt-toolkit after deleting their consumers. No lock file exists. Preserve license notices embedded in retained VAE/ConvRot code and the README license section.

## Configuration and starter choices

Decision: Validate TOML using real argparse actions and validate the final configuration before any model load. Baseline confirms unknown keys, invalid choices and numeric strings bypass validation. Existing CLI precedence works and must stay intact, including overrides equal to defaults. Preserve section flattening and reject collisions.

The starter explicitly chooses AdamW because the existing empty optimizer default is not normalized by its consumer. Use constant scheduler with warmup 0, bf16/SDPA, checkpointing, rank/alpha 32, 1024 resolution and 1000 example steps. Paths/captions are marked REPLACE_WITH_. Raw CFG uses --l and nonempty --n; --g is unused by Krea2. Cache VAE tiling/dtype flags inherited from Hunyuan are ignored/rejected and must not be advertised.

Confirmed code issues to address within retention: dim_from_weights loads the flag instead of network_weights; attention/block-swap checks can happen after model load; base-LoRA merge after quantization is unsafe although the existing Krea2 loader offers load-time merge. Preserve supported dynamic optimizers/schedulers and applicable precision choices. Do not identify Raw/Turbo by filename heuristics because weights share architecture.

## Verification

Use local `.venv`, `PYTHONPATH=src`, offline flags and tiny CPU fixtures. Existing baseline is 59 passed, 7 skipped, 4 failed on missing av, with three additional collection errors on the same import. Save exact evidence in verification.md. No real model execution, installation, downloads, network or builds. All artifact analysis/convergence remains read-only except converge's allowed task append; record results separately.
