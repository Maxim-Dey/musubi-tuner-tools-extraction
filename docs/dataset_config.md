# Captioned image datasets

The declaration is TOML or JSON with `general` and a nonempty `datasets` list. Each dataset selects exactly one `image_directory` or `image_jsonl_file`. General fields are `resolution`, `enable_bucket`, `bucket_no_upscale`, `caption_extension`, `batch_size` and `num_repeats`; datasets can override them and add `cache_directory`.

```toml
[general]
resolution = [512, 768]
caption_extension = ".txt"
enable_bucket = true
bucket_no_upscale = false
batch_size = 1
num_repeats = 1

[[datasets]]
image_directory = "/srv/datasets/subject"
cache_directory = "/srv/cache/subject"
```

Image captions use the same basename, for example `portrait.png` and `portrait.txt`. The existing image reader supports common PIL formats and optional JPEG XL when its plugin is installed. RGB inputs are used for the original VAE; alpha is discarded during original latent preprocessing. Captions are UTF-8.

Alternatively, set `image_jsonl_file` and an explicit `cache_directory`:

```json
{"image_path": "/srv/datasets/subject/portrait.png", "caption": "TOK, a portrait in daylight"}
```

Use one object per line. Only `image_path` and `caption` are accepted. Paths use process CWD, not the JSONL directory. Images and caption content are checked before model loading or cache writes/cleanup. Unsupported keys, numbered targets, control/video/audio declarations and malformed records are errors.

Value fallback remains dataset → general → applicable CLI → runtime → dataclass default, choosing the first non-None value. Multiple datasets must use distinct cache directories. Directory sources default to their image directory for caching; JSONL requires an explicit cache directory. Output/cache directories may be new.

Resolution accepts a positive integer (square) or two positive integers. The original bucket step is 16 pixels. Without buckets both dimensions must be divisible by 16; configurations must produce nonzero usable buckets. `batch_size` and `num_repeats` must be positive integers. Cache CLI `batch_size` only limits encoding chunks; it does not replace an explicit dataset batch declaration.

Cache names, metadata and dtype suffixes remain compatible. A latent cache has the image basename, original dimensions and `_qi.safetensors`; its paired caption cache ends in `_qi_te.safetensors`. Latent tensors use `latents_1xHxW_dtype`; text tensors are stored as `varlen_vl_embed_<dtype>`, with unpadded length. The training reader exposes them as the `vl_embed` batch key. `skip_existing` is existence-only. `keep_cache` preserves stale files; otherwise cache commands remove stale cache files in the selected cache directory. Missing text caches during training are warned/skipped, and zero effective items is an early error.

## 日本語

画像ごとの同名テキストキャプション、または`image_path`と`caption`を持つJSONLを使用します。各データセットのキャッシュ先は重複させないでください。設定値はデータセット、general、CLI、実行時値、既定値の順に解決されます。相対パスは実行時の作業ディレクトリ基準です。`skip_existing`は内容の更新を検証しません。

See the [Qwen workflow](qwen_image.md) and [shipped dataset](../config_for_qwen_image_lora/dataset.toml).
