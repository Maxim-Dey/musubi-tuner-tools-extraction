# Dev image dataset internals

The user-facing contract is [docs/dataset_config.md](../../../docs/dataset_config.md). `ConfigSanitizer` validates the original TOML/JSON structure and supported aliases; `BlueprintGenerator` resolves dataset -> general -> applicable CLI -> runtime -> dataclass defaults. Entrypoints validate paths before weights, without shuffling or allocating model tensors.

Only `ImageDataset` with `f2d` / `flux_2_dev` remains. Directory and JSONL sources share item/index/caption/control and cache contracts. JSONL paths use cwd precedence followed by the list directory. DatasetGroup stamps dataset indices; extras retain source indices and JSONL origin labels. Other item metadata is preserved without enabling another model.

Latent cache names derive from the item basename and original dimensions: `<stem>_<width:04d>x<height:04d>_f2d.safetensors`; text caches use `<stem>_f2d_te.safetensors`. Writers retain architecture/format metadata, logical dtype suffixes, control order and NaN handling. Mistral per-item `[512,15360]` contexts collate to `[B,512,15360]`. Cache callbacks retain skip-existing, custom currentness predicates and keep/stale cleanup. Training consumes both caches; image controls split buckets by count and shape. No migration is part of this cleanup.

Keep helper tests in the existing focused workflow/scope files; no alternative registry, model-specific cache format, video/audio fallback or replacement trainer is required.
