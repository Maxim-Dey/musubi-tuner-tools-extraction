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

Use one object per line. Only `image_path` and `caption` are accepted. Relative image paths use the JSONL directory in experiment mode and process CWD in legacy mode. Images and caption content are checked before model loading or cache writes/cleanup. Unsupported keys, numbered targets, control/video/audio declarations and malformed records are errors.

Value fallback remains dataset → general → applicable CLI → runtime → dataclass default, choosing the first non-None value. Multiple datasets must use distinct cache directories. Directory sources default to their image directory for caching; JSONL requires an explicit cache directory. Output/cache directories may be new.

Resolution accepts a positive integer (square) or two positive integers. The original bucket step is 16 pixels. Without buckets both dimensions must be divisible by 16; configurations must produce nonzero usable buckets. `batch_size` and `num_repeats` must be positive integers. Cache CLI `batch_size` only limits encoding chunks; it does not replace an explicit dataset batch declaration.

Cache names, metadata and dtype suffixes remain compatible. A latent cache has the image basename, original dimensions and `_qi.safetensors`; its paired caption cache ends in `_qi_te.safetensors`. Latent tensors use `latents_1xHxW_dtype`; text tensors are stored as `varlen_vl_embed_<dtype>`, with unpadded length. The training reader exposes them as the `vl_embed` batch key. In legacy operation, `skip_existing` checks existence and missing training text caches are warned/skipped. Zero effective items is always an early error. `keep_cache` preserves stale files; otherwise cache commands remove stale cache files in the selected cache directory.

## Fixed validation inputs and experiment paths

With `experiment_mode=true` in training, dataset TOML paths resolve against the main training TOML; source/cache paths inside each dataset TOML resolve against that dataset TOML; relative `image_path` entries resolve against their JSONL directory. Both cache commands select the same behavior with `--experiment_mode`. Absolute paths remain absolute. With this mode disabled, relative paths retain legacy CWD semantics.

`val_dataset_config` selects the independent validation declaration; no `role` field is accepted. Supply nonempty train and val sets yourself; no automatic split occurs. Source image SHA-256 values must not overlap. Val requires `batch_size=1`, `num_repeats=1`, fixed captions/geometry, and no stochastic augmentation or caption dropout/shuffle. Missing val captions, duplicate content or cache-name collisions are errors. Multiple val datasets form one equally weighted image population.

Use separate, nonoverlapping cache directories, such as `cache/train` and `cache/val`. Prepare all four caches using the [documented commands](../README.ru.md). `--experiment_mode` writes source/caption/preparation provenance; val cache commands additionally require `--validation`. With `--validation`, existing caches selected by `--skip_existing` must have current valid provenance. The validation preflight checks both cache contents, tensor shapes, source association and provenance, including effective training cache associations; it never silently skips invalid val items or regenerates caches during training. Val fingerprinting excludes paths but includes sources, captions, preprocessing and both caches. Changing these inputs requires a new measurement series.

## 日本語

画像ごとの同名テキストキャプション、または`image_path`と`caption`を持つJSONLを使用します。各データセットのキャッシュ先は重複させないでください。設定値はデータセット、general、CLI、実行時値、既定値の順に解決されます。experiment modeでは相対パスを各設定ファイル基準、JSONL内の画像パスをJSONL基準で解決します。従来モードは作業ディレクトリ基準です。`--validation`付きの`--skip_existing`はキャッシュの出所と設定を検証します。

See the [Qwen workflow](qwen_image.md), [train dataset](../config_for_qwen_image_lora/train-dataset.toml) and [val dataset](../config_for_qwen_image_lora/val-dataset.toml).
