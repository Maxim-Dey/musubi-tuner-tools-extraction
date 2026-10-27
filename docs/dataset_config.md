# Captioned image datasets

Use one dataset TOML for both cache commands and training. Resolve all relative paths from the repository root. The [starter dataset](../config_for_krea2_raw_lora/dataset.toml) uses directory images with same-basename UTF-8 `.txt` captions.

```toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
num_repeats = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "REPLACE_WITH_IMAGE_DIRECTORY"
cache_directory = "cache/krea2_raw_lora"
```

Each `[[datasets]]` entry defines one image source. Specify exactly one of `image_directory` or `image_jsonl_file`. Shared settings can be inherited from `[general]` and overridden per dataset. `resolution` accepts a scalar or `[width, height]`; resolutions, batch size and repeat count are user choices rather than fixed program limits.

`enable_bucket` groups images by compatible aspect ratios. `bucket_no_upscale` avoids increasing smaller images. `num_repeats` controls repeated training items; `batch_size` applies within buckets. Different datasets must use separate cache directories to avoid filename collisions.

## Directory captions

```text
images/
  photo01.png
  photo01.txt
  photo02.jpg
  photo02.txt
```

The caption sidecar must use `caption_extension`. Keep captions associated with the intended images; replace the starter path before caching. Images without the expected sidecar cannot provide the required captioned-image input.

## JSONL captions

Replace `image_directory` with `image_jsonl_file = "REPLACE_WITH_IMAGES.jsonl"`. Each nonempty line contains one image and its caption:

```json
{"image_path": "images/photo01.png", "caption": "a description of the training image"}
```

The existing single-image alias `image_path_0` is also accepted. Do not provide both keys in one record. JSONL image paths resolve from the current working directory, so root-relative or absolute image paths avoid ambiguity. Captions are strings; image files must exist.

## Cache lifecycle

Both cache commands use the same dataset and preserve the Krea2 cache format:

- Latents: `{basename}_{width:04d}x{height:04d}_kr2.safetensors`, with `latents_1xHxW_{dtype}` tensors.
- Text: `{basename}_kr2_te.safetensors`, with `varlen_krea2_vl_embed_{dtype}` selected-layer stacks.

Training reads caches instead of running the VAE/text encoder for each training batch. Re-cache when images, captions, resolutions or the corresponding encoder/VAE representations change. Reusing a cache solely because a file exists does not establish that it matches updated data.

Only image-and-caption fields are supported. Unknown fields and video/audio/control/edit settings are rejected before model loading. Cache batching, worker/device settings and diagnostic image display are command options; inspect each cache command's `--help`.
