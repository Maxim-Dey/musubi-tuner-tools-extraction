# Research: Qwen-Image Original LoRA Scope

**Date**: 2026-09-22. **Source revision**: `3157a911d75805f7bad8c2e5747f81655407b5bd` plus existing working-tree changes. Research is based on this checkout, not an assumed upstream architecture. No implementation changes were made.

## 1. Preserve the Actual Dependency Chain Before Removal

**Decision:** Keep the current architecture and filenames. Point Qwen directly at the already extracted common components, relocate three necessary helpers, then delete excluded implementations and prune their branches from shared modules.

**Rationale:** The retained chain is:

```text
root qwen_image_train_network.py
  -> musubi_tuner.qwen_image_train_network.main
  -> setup_parser_common + qwen_image_setup_parser + read_config_from_file
  -> QwenImageNetworkTrainer(NetworkTrainer)
     -> dataset config/blueprint -> ImageDataset -> cached tensors/batches
     -> prepare_sampling -> prompt reader -> Qwen2.5-VL and VAE
     -> Qwen original DiT loader -> offload/FP8/safetensors helpers
     -> importlib.import_module(network_module) -> adapter factory/apply_to
     -> existing loss/noise/optimizer/scheduler/accumulation loop
     -> sample trigger -> Qwen do_inference -> image grid / tracker
     -> adapter save + metadata / Accelerate hooks + save_state/load_state
```

`qwen_image_train_network.py` currently imports `NetworkTrainer`, `DiTOutput`, `clean_memory_on_device`, `load_prompts`, `setup_parser_common` and `read_config_from_file` through `hv_train_network.py`. Their definitions already reside in `training/trainer_base.py`, `accelerator_setup.py`, `sampling_prompts.py` and `parser_common.py`. No new base class is needed.

| Current dependency | Preserve at | Remove after detachment |
| --- | --- | --- |
| `trainer_base.py` imports image/video savers from `hv_generate_video.py` | Move only `save_images_grid` to existing `utils/image_utils.py`; preserve torchvision grid, einops layout, pixels and PNG behavior | Complete Hunyuan generator, video saver and video sample branch |
| `qwen_image_model.py` imports `hunyuan_model.attention.attention` | Move exact `attention`, required `MEMORY_LAYOUT` and optional imports to existing `qwen_image/qwen_image_modules.py`; keep caller signature/alias | Hunyuan models and unused `get_cu_seqlens` / `parallel_attention` |
| `qwen_image_utils.py` imports `flux.flux_utils.is_fp8` | Move this four-dtype predicate unchanged to existing `utils/model_utils.py` | Full Flux utilities/model |
| Qwen cache entry modules import shared cache modules with eager Hunyuan imports | Keep their common parsers, image preview, batching, skip/keep and cleanup functions in place | Hunyuan encoders, loaders, main functions and Hunyuan-only arguments |

Do not substitute `modules/attention.py`: its `AttentionParams` API differs from the attention implementation actually used by Qwen. Preserve split/padded attention and QKV-list behavior through small-tensor before/after checks. Original VAE causal/3D operations and `[B,C,1,H,W]` latents remain necessary image computation; their names/dimensions do not justify deletion.

**Alternatives considered:** Removing every `hv`/`flux` filename immediately breaks Qwen imports; renaming complete foreign modules retains excluded implementations; rewriting a universal trainer or switching attention implementations introduces unnecessary numerical risk.

## 2. Concrete Retain / Adapt / Delete Inventory

Paths in this section are relative to `src/musubi_tuner/` unless explicitly called root paths. A directory-level decision is based on its consumers and implementations, not a filename pattern. Recheck incoming references and dynamic strings immediately before removal.

### Retain or minimally adapt

| Files | Components and disposition |
| --- | --- |
| Root and package `qwen_image_train_network.py`, `qwen_image_cache_latents.py`, `qwen_image_cache_text_encoder_outputs.py` | Keep public commands; remove Edit/Layered routes and add early validation |
| `training/trainer_base.py` | Keep `NetworkTrainer`, `DiTOutput`, weighted MSE, timestep/noise construction, optimizer/scheduler, data loading, accumulation/clipping, compile/offload hooks, metrics, samples, metadata and adapter/state hooks; remove audio/video/control and excluded model-specific branches |
| `training/parser_common.py`, `accelerator_setup.py`, `sampling_prompts.py`, `timesteps.py` | Keep applicable parsers/config precedence, collator, tracker setup, prompt formats/trigger and numerical distributions. Add strict errors; remove unsupported options |
| `qwen_image/qwen_image_model.py` | Keep original transformer, attention/checkpoint/offload/FP8. Remove `QwenEmbedLayer3DRope`, Layered additional timestep conditioning and Edit-2511 `zero_cond_t`/checkpoint dispatch |
| `qwen_image/qwen_image_utils.py` | Keep original text embeddings/loading, original VAE load and required key conversion, pack/unpack and sampling schedule. Remove Edit VL processor/control-image embedding, caption generation, layered packing and excluded version resolution |
| `qwen_image/qwen_image_autoencoder_kl.py`, `qwen_image_modules.py` | Keep original RGB VAE computation and activation helper; prune Layered-only configuration/branches without rewriting causal layers; add only extracted attention helper |
| `cache_latents.py` | Keep `encode_datasets`, common original-applicable CLI and image/console preview; delete Hunyuan `main`, encoding, VAE imports/parser and video previews |
| `cache_text_encoder_outputs.py` | Keep `prepare_cache_files_and_paths`, `process_text_encoder_batches`, `post_process_cache_files`, common parser; delete Hunyuan main/encoder/parser/imports |
| `dataset/config_utils.py` | Image schemas, loader, blueprint/fallback and dataset-group assembly; remove video/control/FramePack schema/constructors |
| `dataset/image_video_dataset.py` | Keep `ItemInfo`, `BaseDataset`, `ImageDataset`, `DatasetGroup` and required exports in the same file; remove `VideoDataset`, audio/control/layered branches |
| `dataset/datasources.py`, `media_utils.py` | Keep image directory/JSONL sources, captions, image globbing, resize/crop and item association; remove video/audio decoding, control/multiple-target sources and `av` import. `cv2` is required by image downscaling |
| `dataset/bucket.py`, `cache_io.py`, `architectures.py` | Keep Qwen bucket step 16, batch/variable-text handling, Qwen cache writers/common metadata and `qi` identity; remove other model mappings/writers/control/audio formats |
| `networks/lora_qwen_image.py`, `lora.py` | Keep Qwen block targeting and `_mod_` exclusion, generic adapter engine/dtype bridging/dropout/rank/weight IO. Delete `HUNYUAN_TARGET_REPLACE_MODULES` and generic file's two Hunyuan `create_arch_network*` factories |
| `networks/loha.py`, `lokr.py`, `network_arch.py` | Existing alternative adapters support Qwen explicitly; retain those options, Qwen detection and error fallback, remove all other architecture dispatch |
| `modules/custom_offloading_utils.py`, `fp8_optimization_utils.py`, `lr_schedulers.py`, `scheduling_flow_match_discrete.py` | Existing required offload/FP8 and scheduler mechanisms; preserve formulas/options |
| `utils/model_utils.py`, `device_utils.py`, `safetensors_utils.py`, `lora_utils.py` | Dtypes/compile/device/shards/loading and retained adapter/FP8 paths. Remove standalone-only `attach_lora_weights` and exclusive quantizer integration only after confirming no retained caller |
| `utils/train_utils.py`, `sai_model_spec.py`, `huggingface_utils.py`, `image_utils.py` | Save precision/naming/retention/state, Qwen metadata, existing optional Hub integration and image saver. Remove foreign architecture tables and Edit-only preprocessing |
| Package `__init__.py` files for retained directories | Keep required package structure; no imports/exports of deleted implementations |

Dynamic adapter resolution in `trainer_base.py:_build_network` adds the package directory to `sys.path`, imports `args.network_module`, invokes its factory and applies adapters only to the transformer. Preserve short and `musubi_tuner.`-qualified names for `networks.lora_qwen_image`, `networks.loha`, `networks.lokr`. LoHa/LoKr are already documented for Qwen and dispatch through `network_arch.py`'s `QwenImageTransformerBlock` branch. They are not foreign model implementations. Do not reinterpret `networks.lora`, whose public factories currently target Hunyuan. Other model modules and unverified arbitrary network selections must not reopen excluded training routes; no new adapter-plugin framework is planned.

`base_weights` uses the selected adapter's `create_arch_network_from_weights` and `merge_to`. `network_weights`/`dim_from_weights` initialize adapters. Neither is equivalent to `resume`, which uses Accelerate state. Preserve all three distinct existing paths.

### Excluded file sets

The following explicitly enumerated stems identify both the root `.py` wrapper and the matching package `.py` implementation where present:

| Family | Entry-point stems to remove | Implementation directories to remove |
| --- | --- | --- |
| Hunyuan | `hv_train_network`, `hv_train`, `hv_generate_video`, root `cache_latents`, root `cache_text_encoder_outputs` | `hunyuan_model/` after attention extraction; shared package cache files stay stripped as above |
| Hunyuan 1.5 | `hv_1_5_train_network`, `hv_1_5_generate_video`, `hv_1_5_cache_latents`, `hv_1_5_cache_text_encoder_outputs` | `hunyuan_video_1_5/` |
| FramePack | `fpack_train_network`, `fpack_generate_video`, `fpack_cache_latents`, `fpack_cache_text_encoder_outputs` | `frame_pack/` |
| Flux | `flux_kontext_train_network`, `flux_kontext_generate_image`, `flux_kontext_cache_latents`, `flux_kontext_cache_text_encoder_outputs` | `flux/` after `is_fp8` extraction |
| Flux 2 | `flux_2_train_network`, `flux_2_generate_image`, `flux_2_cache_latents`, `flux_2_cache_text_encoder_outputs`; package-only `flux_2_train_network_self_flow` | `flux_2/` |
| HiDream | `hidream_o1_train`, `hidream_o1_train_network`, `hidream_o1_generate_image`, `hidream_o1_cache_pixel`, `hidream_o1_cache_text_encoder_outputs` | `hidream_o1/` |
| Ideogram | `ideogram4_train_network`, `ideogram4_generate_image`, `ideogram4_cache_latents`, `ideogram4_cache_text_encoder_outputs` | `ideogram4/` |
| Kandinsky | `kandinsky5_train_network`, `kandinsky5_generate_video`, `kandinsky5_cache_latents`, `kandinsky5_cache_text_encoder_outputs` | `kandinsky5/` |
| Krea | `krea2_train_network`, `krea2_generate_image`, `krea2_cache_latents`, `krea2_cache_text_encoder_outputs` | `krea2/` |
| MiniMax | `minimax_h3_train_network`, `minimax_h3_generate_video`, `minimax_h3_cache_latents`, `minimax_h3_cache_text_encoder_outputs` | `minimax_h3/` |
| Wan | `wan_train_network`, `wan_generate_video`, `wan_cache_latents`, `wan_cache_text_encoder_outputs` | `wan/` |
| Z-Image | `zimage_train`, `zimage_train_network`, `zimage_generate_image`, `zimage_cache_latents`, `zimage_cache_text_encoder_outputs` | `zimage/` |
| Qwen standalone/full finetuning | `qwen_image_train`, `qwen_image_generate_image`, `qwen_extract_lora`, `caption_images_by_qwen_vl` | Keep `qwen_image/` original-only helpers and trainer sampling |
| Other standalone tools | `convert_lora`, `merge_lora`, `lora_post_hoc_ema` | `gui/` |

Also remove these concrete excluded components after incoming-reference checks:

- `networks/lora_flux.py`, `lora_flux_2.py`, `lora_framepack.py`, `lora_hidream_o1.py`, `lora_hv_1_5.py`, `lora_ideogram4.py`, `lora_kandinsky.py`, `lora_krea2.py`, `lora_minimax_h3.py`, `lora_wan.py`, `lora_zimage.py`; `convert_hunyuan_video_1_5_lora_to_comfy.py`, `convert_z_image_lora_to_comfy.py`.
- `dataset/audio_utils.py`, `training/audio_loss.py`; `modules/convrot_int8_utils.py`, `convrot_int8_kernels.py`, `comfy_quant_utils.py`, `nvfp4_utils.py`, `unet_causal_3d_blocks.py`, `adafactor_fused.py` and `modules/attention.py` where the final call graph confirms their existing consumers belong only to removed implementations. `trainer_base.get_optimizer` uses Transformers Adafactor, not the full-finetuning fused implementation. Do not confuse this with removing the supported Adafactor optimizer.

### Tests, documentation, assets and dependencies

**Decision:** Keep shared behavior tests even when historically named for an excluded model. Update mixed-purpose documents/tests by content; remove exclusive material.

- Keep `tests/test_save_precision.py`, `test_lora_dtype_bridging.py`, `test_grad_metrics.py`, `test_krea2_timesteps.py`, `test_ideogram4_timesteps.py`. The last two exercise shared formulas, parsers and trainer; Krea timestep tests explicitly use Qwen VAE geometry. No rename is necessary.
- Adapt `test_datasource_item_extras.py`: preserve original image association/source-index behavior; remove video/control/reference cases. Adapt `test_sai_model_spec.py` to Qwen metadata and `test_top_level_entrypoints.py` to genuine Qwen import/parser interfaces rather than exact wrapper-text assertions.
- Remove exclusive `test_audio_dataset_seam.py`, `test_convrot_int8_prequantized.py`; Ideogram autoencoder/FP8-loading/LoRA-sampling/synthetic/text-FP8 tests; Krea convrot/gather-valid-text/gradient-checkpointing/turbo tests; all currently inventoried MiniMax attention/cache-plan/cache-contract/convrot/dataset/generation/layering/model/NVFP4/packing/provenance/sampling/TE-streaming/temporal-stretch/text-encoder/training/VAE tests. This is a content classification of current files, not permission to delete future matching names. Shared timestep and dtype tests above are explicit exceptions.
- Adapt root `README.md`, `README.ja.md`, `README.ru.md`, `CONTRIBUTING.md`, `CONTRIBUTING.ja.md`: supported workflow, actual options, links and required license/attribution text. Preserve retained-source provenance even when its original model directory disappears.
- Keep/adapt `docs/qwen_image.md`, `dataset_config.md`, `sampling_during_training.md`, `advanced_config.md`, `block_swap.md`, `torch_compile.md`, `loha_lokr.md` and `src/musubi_tuner/dataset/dataset_config.md`. Remove Edit/Layered/video/foreign examples and generic `networks.lora` examples; preserve original-applicable optimizations and mathematical distributions. Step and epoch sample triggers are OR, not an epoch-priority override.
- Remove exclusive `docs/hunyuan_video.md`, `hunyuan_video_1_5.md`, `framepack.md`, `framepack_1f.md`, `flux_kontext.md`, `flux_2.md`, `hidream_o1.md`, `ideogram4.md`, `kandinsky5.md`, `krea2.md`, `minimax_h3.md`, `minimax_h3_1f.md`, `minimax_h3_advanced.md`, `wan.md`, `wan_1f.md`, `zimage.md`, `tools.md`. Remove `docs/kisekaeichi_start.png`, `kisekaeichi_result.png`, `kisekaeichi_ref_mask.png`, `kisekaeichi_ref.png`, `kisekaeichi_start_mask.png` only once their excluded Edit/tool documentation has gone. Remove `docs/betas_for_sigma_rel.png` with `docs/tools.md`: its only reference is the excluded post-hoc EMA guide, so it is not a shared training-distribution figure. Retain shared distribution figures and `images/logo_aihub.png` when still referenced.
- Preserve `.github/FUNDING.yml` and generic Ruff workflow unless a deleted reference needs correction. Do not turn this task into CI redesign. Preserve `.specify/`, `.agents/`, `.cursor/`, `.git/`, user-authored `config_for_qwen_image_lora/speckit_prompts.md`, notices and user data.

| Dependency decision in `pyproject.toml` | Consumer/reason |
| --- | --- |
| Keep current versions of accelerate, diffusers, transformers, safetensors, toml, voluptuous, einops, huggingface-hub, tqdm, Pillow | Retained model, parser, dataset and cache chain |
| Keep bitsandbytes and existing optional optimizer/backend prerequisites | Template AdamW8bit and other applicable existing optimizer choices |
| Keep OpenCV | `dataset/media_utils.py:resize_image_to_bucket` uses real image downsampling |
| Keep torch/torchvision CUDA extras/index choices | Training and extracted `save_images_grid`; no hardware narrowing |
| Keep NumPy/packaging requirements supplied by the existing environment | Direct retained imports; do not remove transitive dependencies merely because excluded packages also use them |
| Keep dev ascii-magic, matplotlib, tensorboard, Ruff | Original image console preview, timestep plot option, logging and lint |
| Remove av after video imports are removed | No retained image use |
| Remove ftfy, easydict and sentencepiece | Wan text/config and removed T5/Flux consumers; original Qwen uses Qwen2Tokenizer |
| Remove gui and hidream_o1 extras, dev prompt-toolkit | Removed GUI/HiDream/interactive standalone generation |
| Keep conditional wandb, attention backends and optimizer imports/documentation | Existing selectable functions; only required when selected, never activated by local validation |
| Prune Ruff excludes/per-file rules for removed files | Manifest/reference consistency, no unrelated dependency upgrade |

`git ls-files` and a filesystem `.lock` scan found **no existing dependency lock file**; there is no `uv.lock` or requirements lock to edit now. Recheck at implementation start and preserve/update any then-existing lock consistently, offline where available. Do not manufacture a lock or silently re-resolve versions.

**Alternatives considered:** A template-only option/dependency whitelist would lose supported optimizations. Keeping all old dependencies/tests/docs would leave unsupported workflows and imports behind.

## 3. Three-Template Audit Against Declarations and Consumers

**Decision:** Keep names/types/meanings and the authoritative `config_for_qwen_image_lora/` directory. Correct shipped references during implementation, not by moving/duplicating templates or rebasing all paths.

`P` = `training/parser_common.py`; `T` = `training/trainer_base.py`; `Q` = `qwen_image_train_network.py`; `U` = `qwen_image/qwen_image_utils.py`, under `src/musubi_tuner/`. This is an all-key source/declaration audit, not a passing execution of the complete parser.

| Training key (40 total) | Actual parser type | Actual consumer / meaning |
| --- | --- | --- |
| `model_version` | U: string | U resolver → Q architecture, original only |
| `dit` | P: string | T `_load_dit_and_swap` → Q loader |
| `vae` | P: string | Q sample VAE; cache command takes its own argument |
| `text_encoder` | Q: string | Q prompt embedding → U loader |
| `dataset_config` | P: Path for CLI; TOML currently string | T `_build_dataset` → dataset config loader |
| `max_data_loader_n_workers` | P: integer | T DataLoader and collator; zero is valid |
| `persistent_data_loader_workers` | P: boolean flag | DataLoader persistence; invalid with zero workers |
| `network_module` | P: string | T dynamic import and adapter factory |
| `network_dim` | P: integer | Adapter rank; omitted generic LoRA rank defaults to 4 |
| `network_alpha` | P: float | Adapter scaling; TOML integer 16 is valid numeric input |
| `mixed_precision` | P: no/fp16/bf16 | Accelerator/network dtype; omission inherits Accelerator |
| `fp8_base` | P: boolean | T DiT weight dtype |
| `fp8_scaled` | Q: boolean | Q scaled-FP8 loader; requires `fp8_base` |
| `fp8_vl` | Q: boolean | Sample text encoder precision; cache has separate flag |
| `blocks_to_swap` | P: integer | Q offloader; zero disables, limit depends on configured blocks |
| `sdpa` | P: boolean | T selects Qwen torch attention |
| `gradient_checkpointing` | P: boolean | Transformer checkpointing and Q input gradients |
| `max_train_steps` | P: integer | Loop and scheduler; epoch limit can derive effective steps |
| `gradient_accumulation_steps` | P: integer | Accelerator accumulation and update counts |
| `seed` | P: integer | Session seeding; omitted seed remains existing randomized behavior |
| `optimizer_type` | P: string | T optimizer factory; adamw8bit is case-insensitive |
| `learning_rate` | P: float | Optimizer and scheduler |
| `lr_scheduler` | P: string | T scheduler factory |
| `lr_warmup_steps` | P: `_int_or_float` | Integer steps or floating ratio of total scheduler steps |
| `max_grad_norm` | P: float | Gradient clipping; zero disables |
| `timestep_sampling` | P: choice string | T / timesteps numerical sampling |
| `discrete_flow_shift` | P: float | Training flow distribution and sample default |
| `weighting_scheme` | P: choice string | Density/loss weighting: logit_normal, mode, cosmap, sigma_sqrt, none |
| `output_dir` | P: string | Checkpoints/state and `sample/` |
| `output_name` | P: string | Artifact naming/metadata |
| `save_precision` | P: float/fp32/fp16/bf16 string | Save dtype resolver; independent from base storage dtype |
| `save_every_n_steps` | P: integer | Step checkpoint/state schedule |
| `save_last_n_steps` | P: integer | Adapter checkpoint retention window |
| `save_state` | P: boolean | Accelerate state save calls |
| `save_last_n_steps_state` | P: integer | Separate state retention/fallback |
| `logging_dir` | P: string | Accelerator timestamped log directory |
| `log_with` | P: tensorboard/wandb/all | Tracker selection and conditional prerequisites |
| `sample_prompts` | P: string | Q → real prompt loader |
| `sample_every_n_steps` | P: integer | `should_sample_images` trigger |
| `sample_at_first` | P: boolean | Step-zero trigger |

| Dataset entry | Actual reader/consumer |
| --- | --- |
| `general.resolution` | ConfigSanitizer accepts scalar integer or two-integer pair; BucketSelector resolution/crop |
| `general.enable_bucket` | Aspect bucket selection |
| `general.bucket_no_upscale` | BucketSelector; BaseDataset currently disables it when bucketing is disabled |
| `general.caption_extension` | ImageDirectoryDatasource basename-associated UTF-8 caption, stripped |
| `general.batch_size` | Cache batching and training BucketBatchManager |
| `general.num_repeats` | Training cache-item references repeated, not repeated encoding |
| `datasets[].image_directory` | ImageDirectoryDatasource, user path |
| `datasets[].cache_directory` | Cache naming/writers/training enumeration |

Keep multiple datasets, image JSONL (`image_path`, `caption`), dataset-level overrides, dataset JSON/TOML support and explicit JSONL cache locations. Actual dataset fallback is dataset field → general field → argparse field → runtime field → dataclass default, first non-None. Cache CLI `batch_size` limits processing chunks; it does not change training's dataset batch size.

| Sample entry | Parsed field and consumer |
| --- | --- |
| Scene text / `TOK` | `prompt` → Qwen text embedding; arbitrary user content |
| `--w`, `--h` | Integer `width`, `height` → sample dimensions/latent preparation |
| `--d` | Integer `seed` → sample generator |
| `--s` | Integer `sample_steps` → denoising schedule |
| `--l` | Numeric `cfg_scale` → Qwen CFG, **not** Hunyuan guidance embedding |
| `--fs` | Numeric `discrete_flow_shift` → Qwen sample scheduler |

Both supplied lines resolve to 1024×1024, 30 steps, CFG 4.0, flow shift 2.2, seeds 42/43. Retain text/TOML/JSON prompt files and applicable `--n` / `negative_prompt`; Qwen inserts a space negative prompt when absent. `--g`/guidance_scale is not consumed by original Qwen (model guidance is None), and video/control/layered/audio/standalone output fields are unsupported. Preserve existing valid dimension normalization/packing rules, without making 1024 compulsory; reject malformed tokens and invalid dimensions/counts instead of hiding them.

**Path decision:** Correct `dataset_config` to `config_for_qwen_image_lora/dataset.toml` and `sample_prompts` to `config_for_qwen_image_lora/sample_prompts.txt`. All relative paths remain relative to process CWD. Keep `qwen_image_lora/output`, `/logs`, `/cache/train`; new output directories need not exist. External `D:/AI/...` files are replaceable examples.

**Scheduler decision:** The working template already has `constant_with_warmup` with 200 steps, separately approved in spec. `T.get_lr_scheduler` builds the constant scheduler and then raises for nonzero warmup; it does not ignore warmup. Keep the correction and move this error before loading. Preserve integer versus floating-ratio semantics (`max_train_steps × process_count`), existing custom scheduler and schedule-free branches. Do not replace them with template-specific checks or silently coerce TOML `200.0` into integer 200.

**Alternatives considered:** Moving configs to `qwen_image_lora/` or changing output paths adds churn; dropping warmup or altering scheduler mathematics violates the recorded decision.

## 4. Effective Configuration and Pre-Load Validation

**Decision:** Implement small checks at existing parser/reader boundaries, retaining current precedence. See the [interface contract](contracts/cli-and-config.md).

Current training config loading first parses CLI, loads TOML (optional extension), flattens one section level in insertion order, builds `argparse.Namespace`, then reparses actual `sys.argv` over it. Missing values receive defaults; explicit CLI wins. Duplicate flattened keys currently use the last value. Store-true flags cannot turn a TOML true into false; do not introduce new boolean CLI semantics.

Unknown TOML keys survive as Namespace attributes; TOML values bypass argparse choices and conversion. Unsupported strings, integer booleans, wrong list shapes and string step counts therefore need final validation. Accept valid integer TOML values for real-valued fields, but not booleans as integers. Validate raw unknown/excluded declarations too: current `model_version=original` masks `edit=true`/`edit_plus=true`. Excluded selectors must error even when another input chooses original.

Validation must precede `prepare_sampling`, which can load the sample text encoder and VAE before the DiT. Both cache commands also validate their actual dataset contents before loaders and cache writes. Caches take defaults + CLI and a separate dataset config, **not** training TOML or `--config_file`.

Retain original-applicable options beyond the templates: rank/alpha/dropout/patterns, adapter initialization/base weights, precision/FP8, block swap/CPU offload/checkpointing, compilation, loader settings, optimizers/schedulers, epoch/step save/sample settings, logging/metrics/metadata and optional Hub state/artifact paths. Keep shared numerical distributions with historical names (`flux_shift`, `flux2_shift`, `krea2_shift`, `ideogram4_shift`, `qinglong_flux`, `qinglong_qwen`, etc.). They are formulas, not foreign model routes.

Unknown names inside `network_args` must also fail before loaders: the current `key=value` split and adapter `kwargs.get` calls silently ignore a typo such as `rank_dropuot=0.1`. Derive accepted argument names and value rules from the selected retained Qwen LoRA/LoHa/LoKr factory and its shared consumers; preserve valid options such as `rank_dropout`, patterns and LoKr's `factor`. Record the current behavior as a validation gap and test the early diagnostic, without creating a generic argument-validation framework.

Qwen training's actual attention supports SDPA, FlashAttention and xformers, including applicable split attention. Preserve the existing unconditional rejection of `sage_attn=true`. For the other flags, preserve current selection precedence: SDPA, then FlashAttention, then xformers, then FlashAttention 3. Multiple flags alone are not an error: SDPA plus xformers selects SDPA, and FlashAttention plus xformers selects FlashAttention. Validate backend-specific prerequisites and split constraints for the selected backend only. Common `flash3` is exposed but absent from the called attention helper's layout table: reject it early when selected, while preserving its existing lack of effect when a higher-priority backend wins. Reject absent selection before weights. Do not replace the attention implementation or introduce mutual exclusivity.

**Alternatives considered:** Validating only initial CLI misses TOML defects; validating only the DiT boundary is too late; a new schema library or general plugin mechanism is unnecessary.

## 5. Cache, Sampling, Save and Resume Baseline

Latent flow: real dataset blueprint → RGB image buckets → `cache_latents.encode_datasets` → Qwen `/127.5-1` normalization and `[B,C,1,H,W]` VAE input → `save_latent_cache_qwen_image`.

Text flow: real caption batches → shared cache orchestration → `get_qwen_prompt_embeds` → trim by valid mask length → `save_text_encoder_output_cache_qwen_image`.

| Artifact / behavior | Existing contract to preserve |
| --- | --- |
| Latent file | `<basename>_<originalW:04d>x<originalH:04d>_qi.safetensors`; `latents_1x<H>x<W>_<dtype>` with `[C,1,H,W]` |
| Text file | `<basename>_qi_te.safetensors`; `varlen_vl_embed_<dtype>` with `[text_length,hidden_dim]` |
| Metadata | `architecture=qwen_image`, `format_version=1.0.1`, original image dimensions or `caption1` as applicable |
| Training read | Match latent/text caches; current missing text cache warns/skips; bucket reader strips format suffixes, stacks latents and keeps variable-length text list |
| Cache reuse | `skip_existing` checks existence only, not fingerprints; `keep_cache` prevents default removal of stale cache files in the configured cache dataset |
| Text write | Existing shared writer merges keys and replaces logical keys across dtype changes; do not silently switch to total overwrite |
| Original forward/loss | Pack latents, pad embeddings/mask nonsplit batches, timestep divided by 1000, unpack prediction, target `noise-latents`; shared weighting/MSE unchanged |
| Sampling | Trigger by initial/step/epoch settings; no-grad, eval/train and block-swap switching; original denoising and PNG saver; torch CPU/CUDA RNG restored on successful completion |
| Sampling boundary limitation | Restoration is not in finally and does not explicitly snapshot Python/NumPy RNG. Characterize actual normal/error boundaries; do not claim universal exception recovery or silently redesign it |
| Save precision | Explicit precision; otherwise full-fp16/full-bf16 flags; otherwise fp32; preserve network metadata/hash behavior |
| Checkpoint names | `name-step00000200.safetensors`, epoch `name-000001.safetensors`, final `name.safetensors` |
| State names | `name-step00000200-state`, epoch `name-000001-state`, final `name-state` |
| Retention | Step removal: `step-last_n_steps-1`, rounded down to save boundary. State-specific window uses existing truthy fallback to checkpoint window; keep exact separate boundary checks |
| Resume | Actual adapter-only Accelerate hooks + `load_state` restore saved model/optimizer/scheduler/RNG state; `_run_training_loop` then sets epoch/global_step to zero and does not skip consumed batches |

Resume's last row is an existing limitation, not exact progress/data-cursor continuation. Preserve it in this narrowing; do not turn initialization from a safetensors adapter into a claim of full resume. If later work proposes changing counter/cursor semantics, present that concrete numerical/behavioral change separately rather than hiding it in extraction.

## 6. Evidence Recorded Now and Required Before Implementation Edits

| Check performed during planning | Result / scope |
| --- | --- |
| Setup script and feature branch | Correct feature directory; branch `001-scope-qwen-image-lora`; no branch switch |
| Syntax parse of root, src and test Python | 330 files parsed with AST; zero syntax errors. Does not establish imports |
| Three templates | Both TOMLs parsed with stdlib tomllib; all 40 train keys matched to parser declarations and consumers; dataset/prompt entries traced above |
| Exact source-only prompt function | Both prompts parsed; unknown option warns/ignores; `--w 512junk --s 0` becomes width 512, steps 1. These are characterized defects to reject, not target behavior |
| Exact source-only mode resolver | Explicit original masks true edit/edit_plus flags; target must reject |
| Exact source-only retention/name helpers | Window 1000, interval 200: steps 1000/1200/1400/1600 select removal None/0/200/400; step 200 filename is `qwen_image_lora-step00000200.safetensors` |
| Real lightweight module import | `musubi_tuner.dataset.architectures` imports; Qwen identity is `qi` |
| Actual training + both cache imports | Bundled Python 3.12.14 fails at missing torch; parser_common/sampling_prompts imports fail at missing toml. No complete parser/import pass |
| Local environment | No project `.venv`; system Python 3.13.7 is outside declared range and lacks dependencies; bundled 3.12 has NumPy/Pillow but lacks torch/accelerate/transformers/diffusers/toml/voluptuous/safetensors/pytest |
| Dependency locks | None present; no lock update performed |

Source-only characterization executes exact isolated function AST with standard-library inputs; it does not replace or fake missing third-party modules and is not reported as real parser/training execution.

Before changing substantial behavior, capture actual available CPU baseline results in the implementation evidence: real parser→TOML→CLI effective Namespace (including non-template values), unsupported TOML and mixed selectors, real prompt txt/TOML/JSON readers, dataset/safetensors round trips, original attention/packing/loss/LoRA gradients, optimizer/scheduler progression, PNG/metrics nonmutation, retention and small real Accelerate state round trip. Reuse existing regression files listed above. Use at most a narrow weight-loader sentinel, tracker sink or encoding callback for the boundary under test, never stubs for the entire chain. Required dependency-bound checks remain unperformed until a compatible existing environment is available; source checks do not discharge them.

Baseline SHA256 values for checking that planning did not modify protected inputs:

| File | SHA256 |
| --- | --- |
| `.specify/memory/constitution.md` | `651A05349939B3156DB15E7EB278368B20CF7F3E46DCA5F5027D29C5061CEDD8` |
| `config_for_qwen_image_lora/train.toml` | `1F199A8E56375433DB4CF72ACC56798E4AC061EC6BB28DC948126651242338B1` |
| `config_for_qwen_image_lora/dataset.toml` | `832516BC8788930BCB23D195F768420612F920869B86E1BDAF88A8B0BA49FE15` |
| `config_for_qwen_image_lora/sample_prompts.txt` | `C32DDF8BB7F28309E5F0AE5236534A8C3A6D1FC584B9A0B4F7CFF0B663440DB0` |
| `specs/001-scope-qwen-image-lora/spec.md` | `0F6B8E998A76013022B7B7C26B1CBAC60EF97C9D60D79DB811B25A21B3CF3A4A` |

No external run was performed: no training, GPU execution, weight download, package build, transfer or server verification. Planning resolves the design; it does not claim model operational verification.
