# Implementation Plan: Deterministic Validation Inputs and Noise

**Branch**: `001-scope-qwen-image-lora` (current checkout; feature directory `002-val-loss-core`) | **Date**: 2026-09-23 | **Spec**: [spec.md](spec.md)

**Input**: Stage 1.1 specification for fixed Qwen-Image original LoRA validation data and deterministic noise. Stage 1 stops at a usable input/noise component; the training loop and metrics are later work.

## Summary

Add an explicit two-role validation dataset declaration that both existing Qwen cache commands can read. Build a strict validation manifest from declared source images rather than the training cache glob, verify captions, bucket assignments and exact cache files before model loading, and record a location-independent fingerprint. Expose sequential deterministic checks using a fixed midpoint grid, SHA-256-derived seeds and an isolated CPU generator. Keep the existing training dataset and disabled training path intact.

## Technical Context

**Language/Version**: Python `>=3.10,<3.13` (project constraint)

**Primary Dependencies**: Existing PyTorch, safetensors, Pillow, toml, voluptuous, NumPy and Qwen-Image dataset modules; standard-library hashlib/json/random

**Storage**: Source images and captions, TOML/JSONL declarations, existing `_qi.safetensors` and `_qi_te.safetensors` caches; canonical manifest/fingerprint data for later resume state

**Testing**: Existing pytest suite plus CPU fixture images and safetensors; no model weights

**Target Platform**: Local CPU validation on Windows and Linux-compatible Python; later GPU execution is out of scope

**Project Type**: Python CLI trainer and two cache commands

**Performance Goals**: No new benchmark target. Hash one input at a time; never precompute all epsilon tensors or load a second model.

**Constraints**: Preserve the existing training/cache file format and behavior for old configurations; fail validation input errors before DiT loading; avoid consuming Python/NumPy/PyTorch CPU/CUDA training RNG; no training or GPU run in this stage

**Scale/Scope**: Exactly two user-managed validation roles for Qwen-Image original; one occurrence per declared source entry; `N1*N2` checks per occurrence

## Constitution Check

*Gate evaluated before research and again after Phase 1 design.*

| Principle | Design check | Status |
| --- | --- | --- |
| I. Language | Feature documents are in English; user communication remains Russian. | Pass |
| II. Task Fidelity | Design implements FR-001–FR-013 without activating metrics or widening model scope. | Pass |
| III. Minimal, Compatible Changes | Reuse `ConfigSanitizer`, `ImageDataset`, `BucketSelector`, existing cache naming/readers; add validation strictness only when roles are declared. | Pass |
| IV. Training Invariants | Separate noise and loader generators; validation does not call training timestep sampling, shuffle or backward/update paths. | Pass |
| V. Memory and Performance | Sequential per-item/per-check work, one existing model at later integration, no bulk epsilon cache. | Pass |
| VI. Configuration and Errors | Effective defaults/TOML/CLI validation and strict preflight precede weight loading. | Pass |
| VII. Verification and Documentation | Substantive CPU regressions and review of `README.ru.md`/`docs/qwen_image.md`; no real-model claim. | Pass |
| Local Stage / Workflow | Only design is produced now; no server or GPU run. | Pass |

No constitution exception or unresolved clarification is needed. The post-design check reaches the same result: the contracts below use existing cache formats, retain legacy behavior when validation is absent, and keep model execution out of this stage.

## Design

### Configuration and source readers

Extend the shared dataset schema with an optional dataset-level `role`, limited to `val_familiar` and `val_unfamiliar`. When any role is present, require exactly one declaration of each role and effective `batch_size=1`/`num_repeats=1`; reject an absent, duplicate or unknown role. Keep `role` out of `[general]` so it cannot be inherited. Both Qwen cache commands accept this mode via their existing `--dataset_config` and validate sources and cache path collisions before loading encoders. The training dataset path rejects a role-bearing config, leaving its existing cache-glob, repeats and shuffle behavior unchanged. The validation path requires roles; no directory name or block order determines them. Details are in [contracts/validation-inputs.md](contracts/validation-inputs.md).

The existing `ImageDirectoryDatasource` and `ImageJsonlDatasource` enumerate declared sources; `BucketSelector` supplies deterministic buckets. For role-bearing directory declarations, scan image filenames separately *before* creating the filtered datasource: `glob_images(..., caption_extension)` silently omits images without captions. The same check must run in both cache commands before model loading. Build a record for each declared occurrence, including source path, caption, original dimensions, bucket, and the two paths generated by `ImageDataset`'s naming methods. Detect aliases and filename-stem collisions before any cache command can overwrite one item's file. Exact duplicate declarations with the same source/caption/cache association remain two occurrences; conflicting associations fail. Validation never gets membership from cache globs.

### Strict cache manifest and immutable input

For role-bearing datasets, cache writers add `source_image_sha256` metadata to both existing safetensors formats without changing filenames or tensor keys. After caches have been created, validation preflight checks every declared occurrence and opens both exact files. Check readability, architecture and source binding metadata, original dimensions, effective caption against text metadata, and required Qwen latent/text tensor keys and shapes. Old caches lacking the binding must be rebuilt for validation but remain usable by legacy training. Reuse the existing `BucketBatchManager` key normalization or a narrowly extracted helper to read one known pair; never use `ImageDataset.prepare_for_training`, which glob-loads and skips missing text caches. A failed source, caption, bucket or cache produces a source-labelled exception; no item is silently skipped.

Compute `image_id` from SHA-256 of original image bytes. Fingerprint each source image, raw caption file bytes or exact JSONL caption value, effective caption, resolution and bucket, and complete latent/text cache bytes. Canonicalize records by role and content fields as a sorted multiset with duplicates retained; exclude absolute paths and filenames from identity. Hash a versioned canonical serialization containing the effective validation protocol. Hold this immutable manifest/fingerprint for later state persistence. On resume comparison or every later item read, recheck current bytes and properties against the manifest before use; a change raises rather than refreshing. Stage 1 exposes this comparison; persistence in full training state is Stage 2/3.

### Noise and loader isolation

Use 1-based `i` and `j`. The fixed level is `0.05 + (i - 0.5)*0.90/N1`; expose low/high halves and `1000*t` as the trainer-facing timestep. Hash a version-tagged, newline-delimited ASCII tuple `(val_seed_noise, lowercase image SHA-256, i, j)` with SHA-256; derive a nonnegative 63-bit seed from the first eight digest bytes in big-endian order. This serialization excludes path, role, step, rank and device. For each check, construct a dedicated `torch.Generator(device="cpu")`, seed it, draw one float32 epsilon matching that item's latent shape, then move/cast only that tensor as needed. Neither `torch.manual_seed` nor CUDA global RNG is touched. Compute `(1-t)*latent+t*epsilon` directly and pass `1000*t`; do not call the training timestep sampler or scheduler shift. Bitwise identical model results across different backends are not promised.

Validation construction must bypass the current `generate_dataset_group_by_blueprint` global `random.randint` side effect, either by supplying an explicit local seed to that factory or constructing the validation group from the blueprint without a global draw. A validation `DataLoader` uses `shuffle=False`, batch size 1, and an explicitly seeded private CPU generator even with zero workers because iterator creation may otherwise consume global torch RNG. Iterate one record and one check at a time, with stable role/bucket/content ordering; changing declaration order does not change image IDs, seeds or fingerprint.

### Integration and verification boundary

Register the five exact public control names and strict integer checks in the existing trainer parser; its default → TOML → explicit CLI merge supplies effective values. Prepare and verify the validation component after effective argument checks and before `_load_dit_and_swap` or other model loading. This stage does not invoke validation during training, publish a loss, or alter resume mechanics; the component only provides an input fingerprint and check iterator to later stages. Review and update `README.ru.md` and `docs/qwen_image.md` to describe preparation and say the validation loop is not yet connected. See [quickstart.md](quickstart.md) for CPU acceptance scenarios.

## Project Structure

### Documentation (this feature)

```text
specs/002-val-loss-core/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── contracts/
│   └── validation-inputs.md
└── quickstart.md
```

### Source Code (repository root)

```text
src/musubi_tuner/
├── dataset/
│   ├── config_utils.py                 # shared schema, role-aware checks
│   ├── datasources.py                  # existing image/caption readers
│   ├── image_video_dataset.py          # existing cache path construction
│   ├── bucket.py                       # existing bucket choice and cache tensor reader
│   └── cache_io.py                     # unchanged serialized cache format
├── qwen_image_cache_latents.py         # role-bearing TOML preflight before VAE
├── qwen_image_cache_text_encoder_outputs.py # same preflight before encoder
├── qwen_image_train_network.py         # Qwen validation preflight hook
└── training/
    ├── parser_common.py                # public controls and effective-value validation
    ├── trainer_base.py                 # before-weight preparation seam
    └── validation_inputs.py            # proposed small Qwen manifest/check component
tests/
├── test_qwen_image_dataset_cache.py     # compatibility and real cache fixtures
├── test_qwen_image_config.py            # effective config and early failures
└── test_qwen_image_validation_inputs.py # proposed CPU input/noise/RNG tests
README.ru.md
docs/qwen_image.md
```

**Structure Decision**: Keep the single Python package and its two existing cache entrypoints. Place the small validation component near training, not in a generic metrics framework. No new cache format or database is introduced.

## Complexity Tracking

No constitution violations require an exception.
