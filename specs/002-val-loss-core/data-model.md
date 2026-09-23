# Data Model: Validation Input and Noise Protocol

## EffectiveValidationConfig

Fields: `val_dataset_config` (path or absent), `val_every_n_steps` (integer >=1), `val_seed_noise` (signed integer), `val_level_noise_n` (`N1`, even integer >=2), `val_seed_noise_n` (`N2`, integer >=1). All integers reject booleans. The path is required only for the enabled state. Defaults, training TOML and explicit CLI values are merged in that order before validation. Unknown options fail. The identity records the effective values but excludes the path string; moving unchanged files does not change the protocol.

State: `disabled` when `val_dataset_config` is absent; otherwise `enabled` after option validation and two-role declaration checks. Disabled state has no validation manifest, loader or checks.

## ValidationSet

Fields: explicit `role` (`val_familiar` or `val_unfamiliar`), one source (`image_directory` or `image_jsonl_file`), cache directory, effective resolution and bucket flags, and a nonempty ordered collection of `ValidationItem` occurrences. Exactly one set of each role exists. The role is stored from the TOML field, never inferred from order or directory name. `batch_size=1` and `num_repeats=1` are enforced in this mode. The user owns set membership; no automatic split, repair or deduplication occurs.

## ValidationItem

Fields: role, source path for diagnostics, caption source, effective caption string, caption fingerprint, original `(width,height)`, configured resolution, selected bucket `(width,height)`, exact latent-cache path, exact text-cache path, SHA-256 of each cache file, and `image_id` (lowercase SHA-256 of original image bytes). Source paths and cache paths locate files for reading but are not identity fields. The same image bytes in either set have the same `image_id`; repeated declarations remain separate occurrences and contribute separate checks.

Relationships: one set contains one or more items. One item refers to exactly one source image, one caption value, one Qwen latent cache and one Qwen text cache. A path collision is allowed only for exact duplicate declarations with the same source and effective caption; conflicting source/caption associations fail. Cache names retain `<stem>_<width:04d>x<height:04d>_qi.safetensors` and `<stem>_qi_te.safetensors`.

Validation: the source is readable and decodable; a directory image lacking its required caption is an error even though the ordinary datasource would filter it. The selected bucket is usable. Required caches are present, readable safetensors with Qwen architecture and `source_image_sha256` binding metadata, matching source dimensions/caption metadata and expected latent/text tensor keys and dimensions. Role-bearing caches add this metadata without changing tensor keys or filenames. No cache is regenerated during validation.

## ValidationInputIdentity

Fields: schema/version marker, effective protocol fields, two role-labelled sorted lists of item identity records, and SHA-256 fingerprint of a canonical UTF-8 JSON serialization (`sort_keys=True`, compact separators, no path fields). Each item record contains `image_id`, caption fingerprint/effective value, original dimensions, configured resolution/bucket settings and selected bucket, plus latent and text cache SHA-256 digests. Lists are sorted by all identity fields and retain duplicates, so input order is immaterial but multiplicity matters.

State transition: `declared` → `preflighted` after all source/caption/cache checks → `frozen` with immutable identity. `frozen` → `verified` only when an immediate later read reproduces its item and protocol fields. Any mismatch transitions to `error`; no automatic rescan or replacement is allowed. Later stages persist the identity in resume state and compare it at resume. Relocation changes only diagnostic paths, not identity.

## NoiseCheck

Fields: item occurrence, 1-based level index `i`, 1-based realization index `j`, `t_i`, `low`/`high` label, derived 63-bit seed, `epsilon` drawn on demand, mixed `noisy_latent`, and trainer-facing `timestep=1000*t_i`. For every item, there are exactly `N1*N2` checks; exactly `N1/2` levels are low and `N1/2` high. `epsilon` is transient and never part of the input manifest. It is generated with a private CPU torch generator and released after the check. Role, path, step and rank do not enter the seed.
