# Validation Input and Noise Contract (Stage 1)

This contract applies only to Qwen-Image `model_version=original` validation preparation. It describes a component ready for later training-loop integration; Stage 1 does not publish validation loss.

## Dataset TOML accepted by both Qwen cache commands and validation preparation

```toml
[general]
resolution = [1024, 1024]
enable_bucket = true
bucket_no_upscale = true
caption_extension = ".txt"
batch_size = 1
num_repeats = 1

[[datasets]]
role = "val_familiar"
image_directory = "data/val_familiar"
cache_directory = "cache/val_familiar"

[[datasets]]
role = "val_unfamiliar"
image_directory = "data/val_unfamiliar"
cache_directory = "cache/val_unfamiliar"
```

Each `[[datasets]]` selects exactly one `image_directory` or `image_jsonl_file`. JSONL records keep the existing `{ "image_path": "...", "caption": "..." }` schema and require `cache_directory`; image paths follow the current working-directory convention. Directory sources require a nonempty `caption_extension` and one readable UTF-8 caption file per discovered image. Role is mandatory on both validation entries, appears only at dataset level, and must be exactly one `val_familiar` plus one `val_unfamiliar`. A role-bearing file with zero, one or more than two entries, duplicate/unknown roles, or empty sets fails. The role is independent of entry order and directory names. Unknown fields fail under the shared strict schema.

For validation, effective `batch_size` and `num_repeats` must both equal 1, including values inherited from `[general]`; setting another value is an error. Existing unroled training dataset TOML retains its prior multi-dataset, repeat and cache semantics. The training dataset reader rejects role-bearing TOML so a validation file cannot become training data by accident. The cache commands accept both legacy unroled TOML and this role-bearing TOML through their existing `--dataset_config`. They validate roles, all directory images/captions, source readability, bucket usability and filename collisions before loading VAE or text encoder. An image without a caption is an error, not a filtered omission. Cache commands still perform initial deterministic preprocessing and write the existing Qwen safetensors names. Validation itself never writes caches.

## Training CLI and TOML fields

| Field | Enabled default | Validation |
| --- | --- | --- |
| `val_dataset_config` | absent (`None`) | Path to role-bearing TOML. Absence disables the feature. |
| `val_every_n_steps` | `200` | Exact integer, `>=1`; used for later scheduling, not Stage-1 loop execution. |
| `val_seed_noise` | `42` | Any signed exact integer; common to both roles. |
| `val_level_noise_n` | `10` | Exact even integer, `>=2` (`N1`). |
| `val_seed_noise_n` | `1` | Exact integer, `>=1` (`N2`, realizations per level). |

The numeric defaults are initial examples, not optimized settings. Effective values follow existing training precedence: parser defaults, then training TOML, then explicitly provided CLI arguments. Unknown training TOML keys and unsupported validation dataset keys fail. Boolean is not an integer; even if a TOML bool enters an integer field, it fails after merging. Validate supplied numeric values even if the dataset path is absent, but in the disabled state do not build a validation dataset, loader or noise and preserve the legacy run. Cache commands continue to use `--dataset_config` and do not read training TOML.

## Image, caption and cache binding

For every declared occurrence, compute `image_id = sha256(original_image_file_bytes).hexdigest()` in lowercase. Record the effective caption, raw caption-file digest (or exact JSONL caption string), original dimensions and selected bucket. Use existing cache filenames exactly:

- latent: `<image-stem>_<source-width:04d>x<source-height:04d>_qi.safetensors`
- text: `<image-stem>_qi_te.safetensors`

These are inside that dataset's `cache_directory`. The selected bucket determines the latent tensor shape; the source dimensions in the filename and cache metadata remain the original dimensions. Distinct images that map to the same cache path, or the same source with conflicting captions mapped to one text cache, fail before caching or validation. Exact repeated declarations of the same source/caption association remain repeated occurrences. Different image paths with identical bytes retain separate occurrences and share an `image_id`; role does not enter that ID.

For newly cached role-bearing datasets, write the source `image_id` as an additional safetensors metadata string on each cache; existing tensor keys, shapes and filenames remain unchanged. Validation requires this binding and checks it against current source bytes. Existing unroled cache files remain compatible with legacy training; role-bearing validation with an old cache lacking the binding fails with an instruction to regenerate it through the role-aware cache commands. The latent cache must expose one usable Qwen `[C,1,H,W]` tensor and matching architecture/source dimensions. The text cache must expose one usable variable-length `[L,D]` embedding, matching architecture, source binding and `caption1` equal to the effective caption. Missing, corrupt, stale, mismatched or ambiguous files fail before weight loading; validation never silently skips them.

## Portable frozen identity and later reads

The component exports an immutable canonical identity with a version tag, effective numeric validation controls and both role-labelled multisets of occurrence records. Each record includes `image_id`, caption fingerprint/value, original dimensions, configured resolution/bucket options, selected bucket, and SHA-256 of both complete cache files. Sort complete records within each role and keep duplicates. Serialize as canonical UTF-8 JSON with sorted keys and compact separators, then SHA-256 the bytes. Paths and filenames are retained for diagnostics but excluded from the identity. Relocating unchanged data/cache files preserves identity; reordering declarations also preserves it. Changing membership, caption, source bytes, geometry, cache bytes or protocol controls changes identity.

The component accepts an expected identity for a future resume comparison and verifies each item against its frozen record at each later read. It raises on a mismatch; it must not update the snapshot or substitute another cache. Stage 1 provides the identity and comparison interface. Full state persistence and validation scheduling belong to later stages.

## Deterministic check contract

For each item, visit every 1-based level `i=1..N1` and realization `j=1..N2` exactly once. `t_i = 0.05 + (i - 0.5)*0.90/N1`. Low levels satisfy `t_i<0.5`; high levels satisfy `t_i>=0.5`; each half contains `N1/2` levels. With `N1=2`, `t_1=0.275` and `t_2=0.725`.

Serialize the seed input as the ASCII bytes of `qwen-image-val-noise-v1\n`, decimal `val_seed_noise` plus `\n`, lowercase 64-hex `image_id` plus `\n`, decimal `i` plus `\n`, and decimal `j` plus `\n`. SHA-256 those bytes. `seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)`. This is independent of role, step, rank, path, name and order. A dedicated local CPU torch generator seeded with this value draws one float32 epsilon of the latent's shape for each check. Move/cast only this transient tensor as required by the existing model path; no global Python, NumPy, torch CPU or CUDA RNG is changed. The exact final mixing is `(1-t_i)*latent+t_i*epsilon`, and the existing Qwen call receives `timestep=1000*t_i`. No training timestep sampling, discretization or flow shift is applied.

## Failure diagnostics

Errors identify the config or source path and role/item when known, explain the offending value or file, and give a correction. Examples: `val_dataset_config: missing val_unfamiliar role; add exactly one dataset with that role`, `val_familiar item 3: image has no .txt caption; create the matching caption`, `val_unfamiliar item 2: ..._qi_te.safetensors is missing; rebuild the text cache`, and `val input changed at item ...; restore the frozen input or start a new experiment`. No error is converted to a warning or zero loss.
