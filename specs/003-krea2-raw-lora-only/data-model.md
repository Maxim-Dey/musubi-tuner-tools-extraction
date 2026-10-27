# Data Model

- Image sources retain directory/sidecar captions and image JSONL, resolution/buckets/repeats/batch size and cache directory. Video/audio/control/edit fields are rejected. Architecture is Krea2.
- Latent caches retain `_kr2.safetensors`, `latents_1xHxW_{dtype}`, `(16,1,H,W)`, original size/caption metadata and format version `1.0.1`; no migration of user data.
- Text caches retain `_kr2_te.safetensors`, `varlen_krea2_vl_embed_{dtype}`, variable valid-token stacks and existing collation/padding.
- Effective arguments retain applicable destinations/meaning with defaults → TOML → CLI, strict type/choice/key checks and early cross-field validation.
- LoRA retains Krea2 targets, `lora_unet` names, rank/alpha/dropout/include/exclude/LoRA+ semantics and safetensors metadata.
- Accelerate state retains optimizer/scheduler/step/epoch/RNG resume behavior. Samples restore training RNG; naming/save retention stays consistent.
