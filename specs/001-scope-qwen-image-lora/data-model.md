# Data Model: Retained Original-Image Workflow

No new persisted entities or formats are introduced. This describes existing objects/artifacts and the required input-validation boundaries.

| Entity | Fields / relationship | Validation and lifecycle |
| --- | --- | --- |
| Effective training configuration | Parser Namespace resolved from defaults, flattened training TOML and explicit CLI; dataset/prompt/weights/output references | Validate supported names/types/values after resolution and existing derived defaults, before model loading; reject raw excluded selectors even if masked by original |
| Dataset declaration / blueprint | General image defaults and one or more image datasets; source, caption settings, resolution/buckets, batch/repeats, cache location | Existing dataset→general→CLI→runtime→default fallback; original image directory or JSONL only, no control/video/audio/layered fields |
| Image/caption item | `ItemInfo`, item key/source index, RGB image, original dimensions, caption and cache paths | Associate basename caption or JSONL caption; retain source identity, resizing/bucketing and batching. Preparation does not need model weights |
| Latent cache | One Qwen original safetensors file per image/bucket identity | Existing writer/reader format below; encode once, reuse for training repeats |
| Text embedding cache | One variable-length Qwen embedding associated with item caption | Trim padding by mask length; preserve logical-key replacement and dtype handling |
| Training batch | Stacked `latents`, list of variable-length `vl_embed`, existing timestep-bucket data | Reader removes cache suffixes; Qwen pads embeddings/masks as needed and packs latents without changing numerical behavior |
| Sample prompt set | Prompt, dimensions, seed, step count, CFG, flow shift, negative prompt; reader enumeration | Text/TOML/JSON readers; strict applicable fields; sample events use initial/step/epoch trigger semantics |
| Adapter checkpoint | LoRA/retained adapter tensors, dtype, training/model metadata and existing hashes | Existing adapter save/load, names and independent retention; not a resumable optimizer state |
| Resumable state | Accelerate state directory containing registered adapter/model, optimizer, scheduler and RNG state | Existing save/load hooks filter frozen base weights; restoration does not currently restore the trainer's local epoch/global-step counter or data cursor |
| Observations | Scalar metrics/tracker events, logs and sample PNGs | Existing names/backend options; no training-weight updates or changes to subsequent normal training behavior |

## Cache Serialization

| Property | Original-image format |
| --- | --- |
| Architecture code / metadata | `qi` / `qwen_image`; `format_version=1.0.1` |
| Latent filename | `<basename>_<originalW:04d>x<originalH:04d>_qi.safetensors` |
| Latent key / shape | `latents_1x<latentH>x<latentW>_<dtype>` / `[C,1,H,W]` |
| Text filename | `<basename>_qi_te.safetensors` |
| Text key / shape | `varlen_vl_embed_<dtype>` / `[text_length,hidden_dim]` |
| Other metadata | Existing original dimensions for latents; `caption1` for text |
| Batch form | Latents `[B,C,1,H,W]`; text list padded in Qwen trainer; dtype/shape/varlen prefixes removed by reader |

Preserve image bucket step 16 and original packing/normalization. A singleton temporal axis is part of the image format. It does not permit a video dataset or video-training path. Cache round-trip tests use tiny real tensors and safetensors, without claiming successful VAE/text encoding.

## State Transitions

```text
image/caption declarations -> validated blueprint -> image/caption items
  -> latent cache + text cache -> matched/repeated/bucketed training batches

defaults + TOML + CLI -> validated original configuration
  -> existing resources/adapter/optimizer/scheduler -> normal training updates
  -> optional observations + adapter checkpoints + separate Accelerate states

saved adapter -> network_weights initialization
saved base adapters -> base_weights merge
saved Accelerate state -> resume restoration -> existing loop restart semantics
```

`skip_existing` checks cache existence; it does not detect changed captions, models or settings. Default stale-cache cleanup and `keep_cache` retain existing behavior within the selected dataset cache directory. Missing text cache currently warns/skips that item; zero usable training items is an early error. No new cache invalidation system is introduced.

Checkpoint step names use eight digits, epoch names six: `name-step00000200.safetensors`, `name-000001.safetensors`, final `name.safetensors`; state directories append `-state`, with final `name-state`. Keep configured checkpoint and state retention separate, including their existing fallback and boundary semantics. Required pre-change evidence is specified in [plan.md](plan.md) and [research.md](research.md).
