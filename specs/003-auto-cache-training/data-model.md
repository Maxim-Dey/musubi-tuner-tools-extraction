# Data Model: Automatic cache preparation

No persistent schema or cache format is added.

## CacheInventory
- dataset_config: absolute effective dataset TOML path.
- validation: boolean role; true only for val.
- missing_latents / missing_text: immutable tuples of absent source-required cache paths.
- Existing supported valid files produce no job; invalid files raise before preparation.

## Ephemeral stage
Role plus latent/text kind selects the existing module and required VAE/text_encoder model. Contains an argv list only; execution is sequential, no queue or global state.

## State transitions
Disabled→native training unchanged.
Enabled→scope/config/source/cache validation→all resource checks→missing stages→complete inventory check→existing validation/resume checks→native training.
Any failure stops before training; completed cache files remain for retry.

## Invariants
Only genuinely absent files are creation targets. All present required files retain bytes. Source/cache isolation is validated before encoding. Parent RNG and user configuration are unchanged.
