# CLI and Configuration Contract

This is the target contract for the narrowing, based on the existing readers and consumers. The gaps described in [research.md](../research.md) are not already implemented fixes.

## Supported Commands and Modes

| Command from repository root | Inputs | Outputs |
| --- | --- | --- |
| `python qwen_image_cache_latents.py` | Dataset declaration, `--vae`, original mode and applicable cache CLI | Original image latent safetensors |
| `python qwen_image_cache_text_encoder_outputs.py` | Dataset declaration, required `--text_encoder`, original mode and applicable cache CLI | Caption embedding safetensors |
| `accelerate launch qwen_image_train_network.py` | Training TOML/CLI, dataset/caches, original DiT, sample resources when enabled | Adapter checkpoints, optional training state, logs and sample PNGs |

Preserve the three matching `musubi_tuner` module entry points. There is no new command, dry-run framework, standalone inference or full-finetuning mode.

`model_version` is `original` (also the existing default). Reject all Edit versions, Layered, video/audio/control/multiple-target declarations and legacy aliases that request them, even if another source selects original. Shipped applicable network modules are `networks.lora_qwen_image`, `networks.loha`, `networks.lokr`, with their existing `musubi_tuner.`-qualified equivalents. Resolve these to their existing factories. Reject generic Hunyuan `networks.lora`, other-model adapters and unsupported network selections before import/loading can activate an excluded route.

Keep the original-applicable option surface, not just the example's 40 keys: LoRA dimension/alpha/dropout/patterns and weights, supported alternative adapters, data-loader settings, durations/accumulation, precision/FP8, checkpoint/CPU offload/block swap, attention, compilation, timestep/loss/optimizer/scheduler choices, sample/save schedules, logs/metrics/metadata, and existing optional Hub integration. No value is restricted to the supplied paths, device model, resolution, rank or duration.

Supported training attention is SDPA, FlashAttention or xformers under the existing backend/split-attention constraints and installed prerequisites. Preserve unconditional rejection of `sage_attn=true`. For the other flags, preserve SDPA > FlashAttention > xformers > FlashAttention 3 selection precedence after TOML/CLI resolution; multiple flags alone are not an error. For example, `sdpa=true` with `xformers=true` selects SDPA. Validate prerequisites and split constraints only for the selected backend, so unused flags do not require their packages. Reject no selection or selected FlashAttention 3 before model loading because the Qwen attention implementation lacks Flash3; a higher-priority selected backend continues to mask `flash3`. This preserves the original selection algorithm and adds no attention implementation.

## Training Resolution: Defaults → TOML → Explicit CLI

1. Build the actual common + Qwen parser and read CLI to locate optional `config_file`.
2. Read training TOML with the current optional `.toml` suffix behavior. Preserve one-level section flattening and last-value behavior for duplicate flattened supported keys; retain enough source information to identify the winning field in errors.
3. Apply parser defaults for missing fields, TOML values, then explicitly present CLI options. Omitted CLI must not reset TOML. Preserve `store_true` semantics; no invented `--no-*` arguments.
4. Resolve existing original-model/default-derived values, including fixed bf16 DiT behavior, default VAE dtype, Accelerator precision fallback and dataset-derived epoch/step counts, at the earliest existing stage where the required information is available. Validate parser values and then relevant derived values before weights. Do not let derived values conceal an explicitly excluded selector.
5. Validate the final effective types, choices, ranges and combinations, and raw unknown/excluded fields from every source. Type/choice checks must apply to TOML Namespace fields, not only argparse tokens. A valid explicit CLI override can replace an ordinary supported TOML value; an unknown key or prohibited mode/module declaration cannot be hidden by a later original selector.

Current CLI `dataset_config` converts to Path whereas TOML supplies a string; both are valid path inputs to the existing reader. Accept integers in real-valued TOML fields such as `network_alpha=16`; reject booleans masquerading as numbers, wrong list shapes and stringified numeric values where the TOML contract requires a number. Do not normalize away integer-versus-ratio scheduler semantics.

All paths retain process-CWD semantics. The authoritative shipped inputs are `config_for_qwen_image_lora/train.toml`, `dataset.toml` and `sample_prompts.txt`. Their two cross-references must use that directory. Output/log/cache directories may be separate and need not exist yet. External example paths are replaceable placeholders in static template checks, but required actual inputs must exist when invoked.

### Scheduler

For the supplied ordinary AdamW8bit path, `constant` with nonzero effective warmup is invalid. Existing scheduler construction raises; the new validation must give the same semantic rejection before model loading. The approved template uses `constant_with_warmup` and integer 200. Zero warmup with constant remains valid.

Keep existing integer step counts, floating ratios, CLI percentage syntax and schedule-free/custom scheduler handling. `200.0` from TOML is a floating ratio under current code, not integer 200; do not silently repair it. Validate the chosen scheduler's actually used settings and preserve applicable custom class/argument mechanisms. Do not impose the example scheduler or 200-step limit on other valid configurations.

## Dataset and Cache Inputs

Keep existing dataset TOML and JSON readers, multiple original-image datasets and image directory/JSONL sources. General fields: `resolution`, `enable_bucket`, `bucket_no_upscale`, `caption_extension`, `batch_size`, `num_repeats`. Per-dataset fields may override these and specify `image_directory` or `image_jsonl_file`, with `cache_directory`. Preserve compatible image/caption formats and existing fallback behavior; JSONL requires a usable explicit cache location when no directory fallback exists.

Dataset value precedence remains dataset → general → applicable argparse → runtime → dataclass default (first non-None). Do not replace it with training-TOML precedence. Caption files share the image basename; JSONL items carry `image_path` and `caption`. Validate source content as well as top-level dataset keys. Video/audio/control paths, multiple image targets, `multiple_target`, FramePack window/one-frame fields, control resolution/no-resize fields and deprecated editing aliases are rejected.

Both caches use their own defaults + CLI, and load the dataset file. They **do not read train.toml or accept `--config_file`**. Preserve `--device`, `--batch_size`, `--num_workers`, `--skip_existing`, `--keep_cache`. Cache batch size limits encoding chunks, not the training batch declaration.

- Latent cache: preserve `--vae`, `--disable_cudnn_backend`, `--debug_mode image|console` and console options. Explicit `--vae_dtype` is currently unsupported for Qwen and must still fail clearly. Remove Hunyuan `vae_tiling`, `vae_chunk_size`, `vae_spatial_tile_sample_min_size` and video-debug selections; parser exposure without Qwen consumption is not support.
- Text cache: preserve required `--text_encoder` and optional `--fp8_vl`.
- `skip_existing` retains its existence-only semantics. `keep_cache` retains otherwise removed stale caches. Do not introduce fingerprints or silently change deletion/overwrite policy.

Use the existing Qwen cache names, tensors, metadata, dtype-suffix replacement and reader behavior in [data-model.md](../data-model.md). Validate inputs before cache writes or stale-file cleanup. Existing missing-text-cache warning/skip remains a characterized behavior; an empty effective training dataset fails before models.

## Sample Prompt Inputs

Preserve `.txt`, `.toml`, `.json` readers and their existing structures, comments/default merging/enumeration. Supported original fields are prompt, width, height, seed, sample_steps, cfg_scale, discrete_flow_shift and negative_prompt, plus reader-generated enumeration. Text options include `--w`, `--h`, `--d`, `--s`, `--l`, `--fs`, `--n`. `--l` is Qwen CFG. Missing negative prompt retains Qwen's existing space-string default.

Strict parsing consumes the complete option value; unknown options, trailing junk, invalid counts/nonfinite numeric values or malformed input structures fail with file/line or record context. Preserve legitimate dimension normalization and Qwen latent packing constraints; require positive usable dimensions and steps without freezing them to 1024/30. Do not accept zero steps by silently clamping. Reject `--g`/guidance embedding, frame counts, image/video/control/reference/audio/layered inputs and standalone-only output options that do not drive original training samples.

Step-zero sampling uses `sample_at_first`. Later step and epoch schedules combine by the existing OR rule. Prompt validation occurs before sample text-encoder/VAE loading, even though sampling preparation precedes DiT loading.

## Early Error Contract

Errors detectable from configuration, files, source declarations and installed prerequisites must occur before **any** DiT/VAE/text/tokenizer loader, tracker initialization, cache write or training start. Reuse existing functions with small validation calls; no general schema/provenance framework is needed.

| Input / example | Required result |
| --- | --- |
| Unknown CLI or TOML key, including nested section | Source + key + unsupported parameter + remove/correct spelling |
| TOML `mixed_precision="bad"`, invalid weighting/timestep, `max_train_steps="1600"`, integer boolean, malformed argument list | Effective value/type/choice error, not silent Namespace acceptance |
| TOML Edit/Layered model, `edit=true` plus CLI original, other-model network | Unsupported original-only selection, naming the offending source and valid alternative |
| Video/control/multiple-target TOML or JSONL, unsupported prompt field | Dataset/item or prompt file/line and correction to an original image/caption input |
| `constant` + 200 warmup on the ordinary scheduler path | Explain incompatibility; use zero warmup or the existing warmup scheduler, without silently choosing either |
| `fp8_scaled` without `fp8_base`; Sage enabled; absent or unsupported selected attention backend; missing selected-backend prerequisite | Explain the actual prerequisite/selection failure. Preserve backend precedence; multiple flags alone are not an error |
| Persistent loader workers with effective zero workers | Explain nonzero-worker prerequisite or disabling persistence |
| Nonpositive update/accumulation/sample/save intervals; invalid ranks/ranges; negative/excessive swaps | Validate against actual consumer constraints and configured block count, not template values; preserve documented zero-disables fields |
| Missing required used input; invalid dataset/prompt structure; empty effective dataset | Identify path/source and how to supply/correct it; new output directories are allowed |
| Invalid `network_args`, `optimizer_args`, `lr_scheduler_args` syntax/known unsupported arguments; unknown consumed-option names such as `network_args=["rank_dropuot=0.1"]` | Reject before loaders and identify source, key, cause and correction. Validate adapter argument names against the selected retained factory and its shared consumers; preserve valid `rank_dropout`, pattern, LoKr `factor` and other existing class-specific arguments |

Example target diagnostic: `config_for_qwen_image_lora/train.toml: lr_warmup_steps=200 is incompatible with lr_scheduler='constant'; set warmup to 0 or select constant_with_warmup.` Test meaning/source/correction, not an incidental exact sentence.

## Saving and Resume

Preserve adapter safetensors, configured save dtype, metadata, step/epoch/final naming and independent checkpoint/state retention. `save_state=true` enables state at supported save boundaries and final completion; `save_state_on_train_end` remains an applicable separate option. `network_weights` initializes adapters; `base_weights` merges base adapters; `resume` selects an Accelerate state directory (or existing configured Hub state mechanism).

Preserve actual supported saved adapter/optimizer/scheduler/RNG restoration and the current loop counter reset documented in research. Do not claim exact old batch position or global-step continuation. A correction to that baseline is not authorized as a silent part of narrowing.

## Acceptance Evidence

Exercise actual parsers/readers on all three templates with corrected input references and temporary user-input fixtures. Assert effective values after CLI overrides and unchanged selection for simultaneous attention flags, including selected-backend-only prerequisites. Separately exercise unsupported TOML values, raw mixed selectors, unknown names inside `network_args`, and prompt/dataset rejection before loader sentinels. Test applicable non-template options/values and both dynamic module aliases. Keep full imports, help output, tiny real cache/adapter serialization and selected regressions distinct from source-only checks; unavailable dependencies are never reported as passes.
