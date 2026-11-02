# Validation and experiment contracts

## Input boundary (A/B owner → C/D owner)
`training/krea2_validation_inputs.py` exposes:
- `preflight_validation_inputs(args) -> ValidationInputs | None`, before any models.
- `prepare_validation_caches(args, inputs, accelerator)`, rank0 writes then all ranks synchronize, before DiT/dataset/sample model loading.
- inputs.records is a list of records with image_sha256, latent_cache_path, text_cache_path.
- inputs.manifest is JSON-compatible; inputs.validate_unchanged() checks fixed inputs/caches; inputs.load_batch(record) returns existing Krea2 batch representation.
- Object also exists for experiment_mode/auto_cache without val, with records=[]; disabled legacy path returns None.
- Persist output/val_manifest.json and checkpoint/val_manifest.json; resume lacking or contradicting manifest fails before loading models. Checkpoint owner copies frozen manifest rather than regenerating it.

## Cache files
Latent `<stem>_<original_width:04d>x<original_height:04d>_kr2.safetensors`; text `<stem>_kr2_te.safetensors`; metadata architecture krea2; text key `varlen_krea2_vl_embed_<dtype>`. Existing tensor format unchanged; additional provenance mandatory in new modes. Do not silently recache incompatible files. Detect equal stems before write.

## Seed and loss
Use canonical compact JSON UTF-8 serialization of `[val_seed_noise,image_sha256,i,j]`, SHA-256 digest first 8 bytes big-endian masked with `(1<<63)-1`. i/j one-based. Fixed known test vector must be asserted. Dedicated generator uses frozen generation device/dtype and normalized latent shape. t midpoint formula; x_t direct interpolation; call_dit receives 1000*t. Mean latent-element MSE times analytic fixed-t weighting, then equal-combination per-image and equal-image global averaging. No train noise/sampler, lookup shift or CFG.

## Lifecycle/checkpoint
Call val at initial restored optimizer state only if not already validated, after each successful completed optimizer update if due, and unduplicated final state. Persist completed validation with state. Maintain optimizer count separately from Accelerator accumulation counter. New state save hooks write sole model.safetensors FP32 and remove framework model copies; ordinary path unchanged. Samples for an unsaved state are explicitly step-numbered separately. Readiness marker is last after all-rank RNG/state completion.

## Retention
At completed step S with effective window W and save interval I, preserve current existing boundary semantics: compute removable step floor((S-W-1)/I)*I (negative means none), and remove whole current-experiment complete folders at or below eligible boundary. Verify exact boundary and nonmatching window/interval in fixtures. W=save_last_n_steps_state if supplied else save_last_n_steps; absent both means unlimited.
