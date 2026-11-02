# Research decisions

## Effective configuration and paths
Decision: extend existing strict read_config_from_file merge and _config_sources; use opt-in dataset resolve_paths flag. Rationale: current parser already orders defaults/TOML/CLI and rejects unknown settings. Avoid a second parser. Inventory must independently enumerate supported image files because existing glob_images silently excludes missing captions.

## Existing cache mechanism and provenance
Decision: reuse main(args=None) cache entrypoints, batching, skip_existing/keep_cache and additional_metadata seam. Existing metadata proves only architecture/dimensions/caption, insufficient source/model/processing origin: new modes reject legacy caches unless explicitly recached by operator. Latent normalization already occurs in Qwen VAE pipeline, never repeat it. Bucket resize/crop is deterministic. Fingerprint actual local VAE/encoder weights and tokenizer assets/config, selected layers (2,5,8,11,14,17,20,23,26,29,32,35), max length and prompt representation. TE representation seq×12×2560; latent 16×1×bucket_h/8×bucket_w/8. Actual cache_io path is resolved by owner before editing.

## Numerical forward
Decision: call existing Krea2 call_dit with direct fixed x_t and 1000*t. Guard gradient-checkpoint requires_grad with torch.is_grad_enabled(). Existing training get_sigmas does nearest schedule lookup, unsuitable for exact fixed validation t. Use analytic sigma=t: sigma_sqrt=t^-2; cosmap=2/(pi*(1-2*t+2*t*t)); other current schemes give unit loss weight. Keep training helper unchanged. No CFG/shift/+1.

## Isolation/distribution
Decision: snapshot individual module training flags and GC/activation offload/block-swap state, restore in finally; dedicated noise generator plus full RNG guard. Unwrapped DiT avoids uneven DDP forward collectives. Reduce sums and image counts, including empty ranks. Alternative equal-rank mean is wrong for uneven shards.

## State and lifecycle
Decision: explicit optimizer count/cursor, not accelerator.step. Current resume loads Accelerate state but loop initializes step/epoch at zero and has no exact skipping. Current on_post_optimizer_step runs each microbatch. Dataset in-place bucket shuffles require replaying epochs, not one final shuffle. New save hooks remove all Accelerate model weights and write exactly one FP32 adapter with existing network save/load methods. Native optimizer/scheduler/per-rank RNG remain Accelerate responsibility.

## Configuration and operations
Decision: preserve main exact command9 settings and legacy dataset.toml; diagnostics separate and use original data read-only. TensorBoard already appears in dev dependency group; document/install that group remotely rather than add redundant dependency. Remote model paths/environment/HTTP endpoint remain facts to discover only in F.

## Evidence
Read full source commands1–12, constitution2.0.0, skills specify/plan/tasks/analyze, current templates and user diff. Independent A/B and C/D research reports provided exact insertion points and inventory. No unresolved technical clarification; infrastructure availability is pending F.
