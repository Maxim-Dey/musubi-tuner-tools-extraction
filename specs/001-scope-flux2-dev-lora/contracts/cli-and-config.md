# CLI and configuration contract

## Operational entrypoints

| Root script and corresponding `python -m musubi_tuner.<name>` | Consumed inputs |
| --- | --- |
| `flux_2_cache_latents.py` | Existing cache CLI including `--model_version dev`, `--dataset_config`, `--vae`, `--vae_dtype`; dataset TOML/JSON |
| `flux_2_cache_text_encoder_outputs.py` | Existing text-cache CLI including `--model_version dev`, `--dataset_config`, `--text_encoder`; dataset TOML/JSON |
| `flux_2_train_network.py` | Existing training CLI, optional `--config_file` TOML, dataset TOML/JSON, optional sample TXT/TOML/JSON and `--log_tracker_config` TOML; ordinary LoRA/state inputs |

Keep root and package execution; all three support `--help` without weights. Common cache modules retain importable helpers, not Hunyuan `main` or module execution. No standalone generation, merge/export/conversion, post-hoc EMA, caption-generation or alternate training command remains. No new validation-only public command or train-TOML support for cache commands is required.

Model is `dev` (existing default). Accepted network spellings are `networks.lora_flux_2` and `musubi_tuner.networks.lora_flux_2`; preserve supplied spelling in existing metadata. Both resolve to the same supported implementation. Generic `networks.lora`, other architecture modules, LoHa/LoKr and arbitrary LyCORIS/network plugin modules are not selectors for this workflow. Internal generic LoRA functions remain importable for the Dev wrapper.

## Validation order

1. Parse CLI structure and read each consumed source. Reject unknown/removed parameters and malformed section/value shapes with file/CLI location. Training TOML retains flat keys and arbitrary one-level grouping names, as documented by the existing loader; `[optimizer]`, `[training]`, `[output]`, `[model]` and `[lora]` are examples, not a whitelist. Group labels do not select behavior. Validate every contained key against retained parser destinations before flattening, preserving existing flattening order. Reject deeper grouping and unsupported value structures; a valid group label cannot hide an unknown parameter.
2. Preserve existing defaults -> TOML -> explicit CLI precedence for valid training values, including grouped-key flattening order. Preserve dataset precedence from data-model.md. A consumed source's unknown key remains an error even if a later input would override it.
3. Validate the effective Namespace: types, choices and model/method compatibility, required paths, coupled flags and selected dependencies. Recheck after model-specific defaults such as FP32 VAE are applied. Do not rely on `argparse.choices` to validate TOML-provided Namespace attributes. Invalid known values overridden by valid CLI follow the existing precedence; unknown parameters and invalid structures never disappear this way. Dataset, prompt and tracker section names follow their own schemas; training grouping labels remain unrestricted.
4. Parse and validate the referenced dataset and configured sample prompts before model initialization. Validate image sources and controls, source paths, writable destinations and active sampler resources without scanning/loading model tensors. Confirm Mistral's companion shards and processor resource availability where model-independent; no checkpoint-format guess or new tokenizer option.
5. Resolve the allowed network and check nested options and selected optimizer/scheduler/backend/tracker availability before model loading. Read and validate any consumed `log_tracker_config` TOML at this boundary, using the tracker contract below. No tensor allocation, optimizer substitution, tracker disabling, tracker initialization, RNG consumption or data shuffling in validation. Preserve RNG-consuming initialization at its original point.

Use small checks in existing parser/config/sampling owners; a lightweight source-key map for errors is sufficient. Do not serialize a second configuration format or introduce a registry.

## Nested selections

For the retained adapter, validate actual consumed arguments: `conv_dim`, `conv_alpha`, `rank_dropout`, `module_dropout`, `verbose`, `exclude_patterns`, `include_patterns`, `loraplus_lr_ratio`. Preserve existing literal/regex and valid-value meanings. Reject unknown names, malformed `key=value`, excluded `algo`/method/module injection, invalid regex/literals and duplicate ambiguous nested options before constructing the network. Top-level rank/alpha/dropout remain supported independently. Do not expose internal `module_class`/`module_kwargs` as a route to another method.

Keep the existing generic optimizer/scheduler dispatch for standard Dev LoRA. Validate names/import availability and keyword signatures of selected classes/functions before constructing model weights, while preserving valid constructor defaults and custom selections. For constructors accepting `**kwargs`, validate against their actual supported parameter contract rather than silently accepting unknown inputs; do not create a generalized plugin framework. Do not instantiate an alternative optimizer to pass a check. Template AdamW8bit must actually select bitsandbytes AdamW8bit.

`fp8_text_encoder=true` is rejected for Mistral before loading any shard. False remains a recognized setting. Keep supported DiT FP8, compile, block swap, optional timestep/optimizer and image-control options. A historically named scalar timestep formula does not enable another model and must not be rejected as such.

## Dataset and prompt inputs

Dataset TOML/JSON retains existing general/image schema and valid control aliases. Reject video/audio/FramePack fields, other architecture IDs, unknown sections/keys and malformed source combinations. Preserve applicable image directory/JSONL and relative-path fallback. Do not silently fall back to VideoDataset or drop unknown fields to make the template pass.

TXT sample switches retain image settings: `--w`, `--h`, `--d`, `--s`, `--g`, `--fs`, repeated `--ci`, and the existing `--n`/`--l` compatibility behavior. Structured counterparts are `prompt`, `width`, `height`, `seed`, `sample_steps`, `guidance_scale`, `discrete_flow_shift`, `control_image_path`, `negative_prompt`, `cfg_scale`. The current Dev path encodes negative context but uses guidance-distilled denoising; cleanup must not invent a CFG behavior change. If retaining the legacy `--f`/`frame_count` image alias, only 1 is valid; no video branch remains. Reject video/audio/reference-video/standalone options, unsupported extensions/shapes, unknown switches, partially matched malformed tokens and empty unusable prompt files. A TOML `[prompt]` plus `[[prompt.subset]]` and JSON list continue to follow the existing loader with strict source-key checks.

Internal fields (`enum`, cached contexts) are added after validation. Do not accept user-provided tensors/unknown dictionaries as bypasses. Preserve current numeric normalization/defaults for valid values, including existing sample-step clamping, instead of retuning the sampler.

## Tracker configuration input

Preserve the existing optional `log_tracker_config` file and its tracker-specific initialization settings. Before the first AE/Mistral/DiT load, validate file readability, TOML syntax, mapping structure, recognized tracker sections and supported initialization parameters/values against the retained backend contracts, including supported forwarded arguments. Nested backend payload dictionaries keep their existing meaning; their user-defined payload keys are not training parameters. Unknown tracker sections, unsupported initialization parameters and malformed structures fail with file/full key path, offending value, cause and correction. Do not apply the training TOML grouping/flattening rule to this file.

Validation must not construct a tracker, create an event run, log in or contact a service. Preserve valid initialization arguments, defaults and the existing initialization point. Local cases use temporary TOML files and loader sentinels: a valid tracker configuration, a missing/unreadable file, malformed TOML, an unknown tracker section, an unknown initialization argument and an invalid structure/value.

## Compatibility and failure examples

- Both real templates remain valid after documented external substitutions and the two shipped-path corrections. All 42 active training values and all active dataset values retain their meaning; the commented `blocks_to_swap=20` stays commented.
- CLI or effective TOML `model_version=klein-4b`, `klein-base-4b`, `klein-9b`, `klein-base-9b` fails nonzero before loaders. Cache commands reject equivalent CLI selections and excluded dataset settings.
- Training groups such as `[model]` with `model_version="dev"` and `[lora]` with `network_module="networks.lora_flux_2"` remain valid with existing precedence. Unknown `train.toml:model.learnng_rate`, an unsupported dataset/prompt/tracker section, CLI `--save_merged_model`, or prompt `--unknown_option` fails with the source, offending key/value, reason and correction.
- A known TOML value overridden by valid CLI is checked in its final form. An unknown TOML key is still an error alongside `--model_version dev` or another valid CLI override.
- Missing TensorBoard with effective `log_with=tensorboard`, missing bitsandbytes for AdamW8bit, or unavailable selected optional backend fails explicitly before weights. No fallback or ignored active setting.

Negative tests must use loader sentinels to prove ordering, not merely assert any nonzero exit caused by an unavailable import. Syntax/import failures in the baseline are separately classified.

## Artifacts and resume

Keep Dev caches, LoRA keys/metadata, PNG naming/pixels, TensorBoard records, save/sample timing/retention and Accelerate state format. `--resume` takes an existing state directory; a LoRA `.safetensors` file is not full state. Loading initial adapter weights remains distinct from resume. Preserve the pre-change resume behavior and document the source counter limitation; do not claim full progress restoration or interrupted/uninterrupted equivalence without evidence.
