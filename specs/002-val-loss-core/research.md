# Research: Deterministic Validation Inputs and Noise

## 1. Shared declaration and cache readers

**Decision**: Add a validation role to the existing dataset declaration schema and retain `ImageDataset`, `ImageDirectoryDatasource`, `ImageJsonlDatasource`, `BucketSelector`, and the original Qwen cache names and tensor formats. Distinguish role-bearing validation declarations from legacy training declarations at the caller.

**Rationale**: Both Qwen cache commands already call `BlueprintGenerator(ConfigSanitizer())`, `generate_dataset_group_by_blueprint`, and `validate_dataset_sources` before loading a VAE or text encoder. A `role` added only to an example would currently fail the shared strict schema; this is the smallest path for one TOML to work in both cache commands and the future trainer validation reader. `ImageDataset.get_latent_cache_path` builds `<stem>_<width:04d>x<height:04d>_qi.safetensors`; `get_text_encoder_output_cache_path` builds `<stem>_qi_te.safetensors`. Qwen latent caches contain `[C,1,H,W]` under a dtype-suffixed `latents_...` key; text caches contain variable-length `vl_embed` under a dtype-suffixed key. Safetensors metadata carries architecture and dimensions or caption.

**Alternatives considered**: A separate validation-only TOML parser would duplicate schema and drift from the cache commands. Renaming caches by SHA-256 would break existing formats. Neither is needed.

## 2. Declared membership and strict preflight

**Decision**: Derive validation membership from declared source records and require all referenced captions/caches. Perform validation-specific source and path checks in both cache commands before model loading, add source SHA-256 metadata to role-bearing caches, then require the cache contents and source binding when preparing the trainer-side validation manifest.

**Rationale**: `ImageDataset.prepare_for_training` uses a latent-cache glob, logs a warning and skips a missing text cache, then applies repeats. It cannot prove that every declared validation image is present. `glob_images` removes directory images without matching captions before `ImageDirectoryDatasource` sees them; ordinary `validate_dataset_sources` cannot detect those omitted files. Cache writers use the filename stem, so distinct images with the same stem may overwrite a shared text cache even when image extensions or paths differ. A role-aware raw-directory preflight and cache-path collision check must occur before either cache command loads a model. The trainer's strict preflight then checks exact files and metadata, not just existence.

**Alternatives considered**: Reusing the training cache glob would silently omit incomplete inputs. Changing the training reader's skip behavior would alter legacy training semantics. Checking a cache filename alone cannot prove that a stale file came from the current source image. These options are rejected. Additional safetensors metadata keeps existing tensor formats and filenames while binding newly cached validation inputs.

## 3. Portable input identity

**Decision**: Use SHA-256 of each original image file for `image_id`. Record a versioned, canonical digest of the effective protocol and a sorted multiset of role-labelled occurrence records. Each record includes image digest, caption representation, original dimensions, effective resolution/bucket, and full-byte digests of both caches. Paths and filenames are diagnostic fields only.

**Rationale**: Sorting and retaining duplicate records makes declaration order irrelevant without deleting repeated occurrences. Excluding paths makes relocation safe; including caption, geometry and cache digests catches changes that an image-only hash misses. Use the same snapshot for the first preparation and later reads; compare rather than refreshing. For directory captions, include both raw file bytes and effective stripped text; for JSONL, include the exact caption string in UTF-8 plus effective text. Hashing complete cache bytes also detects altered tensor or metadata payloads without a new database or cache format.

**Alternatives considered**: Modification times and file sizes are insufficient and path-dependent. A manifest keyed only by `image_id` would collapse multiple occurrences and miss caption/cache changes.

## 4. Seed serialization and tensor RNG

**Decision**: Serialize `b"qwen-image-val-noise-v1\n"` followed by ASCII decimal `val_seed_noise`, lowercase 64-hex `image_id`, ASCII decimal 1-based `i`, and ASCII decimal 1-based `j`, each terminated by `\n`; hash with SHA-256. The torch seed is `int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)`. Draw each epsilon using a newly seeded local CPU `torch.Generator` in float32, then transfer/cast the single tensor to the latent's device/dtype for the forward path.

**Rationale**: Field alphabets exclude newlines, so serialization is unambiguous and accepts arbitrary signed integer base seeds without an extra range limit. The 63-bit result is nonnegative and accepted by PyTorch generator seeding. CPU generation decouples the random stream from CUDA global state and accelerator/rank. Validation preflight, `DataLoader` iteration and check creation must have explicit local generators or no random calls. The existing dataset group factory currently calls global `random.randint` even when not training; direct use would violate FR-010. A bare `DataLoader` iterator may consume global torch RNG to seed workers, even when `num_workers=0`, so provide an isolated CPU generator.

**Alternatives considered**: Python `hash()` is process-dependent. `torch.manual_seed`/`torch.cuda.manual_seed` would perturb training. Device-local epsilon generators could produce different samples across CPU and CUDA and complicate RNG isolation.

## 5. Fixed grid and trainer interface

**Decision**: Calculate `t_i = 0.05 + (i - 0.5)*0.90/N1` once for each 1-based level. `t_i < 0.5` is low and `t_i >= 0.5` high. For each `j`, pass `noisy_latent=(1-t_i)*latent+t_i*epsilon` and `timestep=1000*t_i` to the existing Qwen model-call interface.

**Rationale**: `QwenImageNetworkTrainer.call_dit` divides its timestep input by 1000 and forms target `noise-latents`. The general training path samples timesteps and may apply scheduler shift; validation must bypass it. `N1=2` yields `0.275` and `0.725`, with one level in each half. Validation input/noise reproducibility does not imply bitwise equal model output across different GPU kernels or backends.

**Alternatives considered**: Reusing training `get_noisy_model_input_and_timesteps` would resample or transform the fixed grid. Precomputing all noise would waste memory and defeat sequential processing.

## 6. Effective options and disabled behavior

**Decision**: Register `val_dataset_config`, `val_every_n_steps`, `val_seed_noise`, `val_level_noise_n`, and `val_seed_noise_n` in the existing training parser. Validate their effective types/values after default → TOML → explicit CLI merge. If `val_dataset_config` is absent, create no validation dataset, loader or model work and keep the existing training behavior.

**Rationale**: `read_config_from_file` already rejects unknown training TOML keys, merges CLI over TOML and calls `validate_parser_values`, which uses exact `int` checks rather than accepting bool. A focused range/evenness check and early validation preflight complete the contract. Example values `200/42/10/1` are defaults for this proposed interface, not claims of optimal hyperparameters.

**Alternatives considered**: Parsing validation options in a second CLI system would risk precedence differences. Allowing inactive validation options to trigger cache reads would change disabled runs.

## 7. Test and documentation seams

**Decision**: Add CPU tests alongside the existing Qwen dataset/config tests, using real temporary images and safetensors and independent expected seed/grid values. Review `README.ru.md` and `docs/qwen_image.md`; stage-1 documentation must state that the validation loop is not connected yet.

**Rationale**: Existing tests already cover the legacy skip/repeat/cache naming behavior and pre-model error boundaries. New tests can assert isolation and strict validation behavior without a GPU or model download. `README.md` is absent in this checkout; the Russian README and Qwen guide are the relevant maintained entry points.

**Alternatives considered**: Mock-only cache tests would not establish real tensor and metadata reading. Running training would violate the local-stage boundary.
