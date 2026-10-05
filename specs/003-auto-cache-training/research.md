# Research: Automatic cache preparation

## Launch boundary
Decision: use QwenImageNetworkTrainer.validate_training_inputs after native validate_training_args.
Rationale: called before training dataset/seed/Accelerator/models, shares actual parser and initialization.
Alternatives: external wrapper duplicates launch parsing or requires a new training interface; embedding in training loop loads models too early.

## Discovery and validation
Decision: reuse _source_records/_validate_cache and factor shared source-only pairing checks.
Rationale: prepare_for_training globs existing latents and skips absent text, so it cannot inventory missing sources. Existing validators already know names, geometry, provenance and overlap.
Alternatives: check cache-directory existence or catch every validation error as missing would hide incomplete/stale data.

## Encoding
Decision: sequential native cache subprocesses, only for missing stages, with skip_existing+keep_cache.
Rationale: native mains always load encoders; processes release memory and Accelerator state. Their shared loops already implement existence skipping and safe preservation when keep_cache is set.
Alternatives: in-process main substitutions risk model/RNG/Accelerator leakage; new encoder implementation duplicates accepted code.

## Supported scope
Decision: auto_cache is opt-in, one-process experiment mode; enable in current user profile.
Rationale: experiment mode already carries source provenance and consistent paths. Legacy train+val without it lacks independent train-cache provenance; avoid inventing a new flag or silently changing legacy paths. Disabled mode keeps old behavior.
Alternatives: distributed writer coordination and legacy provenance expansion add unrequested mechanisms. Users can still manually prepare caches for those modes.

## Verification
Decision: CPU source/cache/entrypoint integration, targeted and full regression; no GPU/download/remote run.
Rationale: changed behavior is orchestration around already exercised encoders; the user excludes broad GPU work. Preserve boundaries of what local tests prove.
