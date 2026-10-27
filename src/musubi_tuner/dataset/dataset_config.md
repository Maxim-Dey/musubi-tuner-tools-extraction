# Image dataset configuration

Run commands from the repository root. Relative paths resolve from the current working directory.

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

Each image uses a UTF-8 caption file with the same basename. Images without a matching caption are excluded. Supported image extensions are PNG, JPEG, WebP, BMP and AVIF, plus JPEG-XL when its optional Pillow plugin is installed.

`resolution` accepts one integer or `[width, height]`; dimensions must be positive integers. Without bucketing they must be multiples of 16. Bucketing constructs aligned dimensions from the requested area and preserves aspect ratios by choosing the nearest existing bucket and using the existing resize/center-crop behavior. `bucket_no_upscale` is effective only with bucketing. `batch_size` and `num_repeats` are positive integers. General settings may be overridden in each dataset.

Alternatively, use `image_jsonl_file` instead of `image_directory` and explicitly set `cache_directory`. Each JSONL record contains a single `image_path` (the existing `image_path_0` alias is also accepted) and a string `caption`. Image paths retain their existing current-working-directory resolution.

```json
{"image_path": "images/example.png", "caption": "a caption"}
```

Each dataset needs its own cache directory. Directory datasets default their cache directory to the image directory. Cache both latents and text embeddings with the same dataset configuration before training. Latent filenames retain the original image dimensions and the `_kr2.safetensors` suffix; text embeddings use `_kr2_te.safetensors` and variable token lengths.

Only captioned Krea2 Raw image datasets are supported. Unknown configuration or JSONL fields, including video, audio, controls, edits and multiple targets, produce an error before model loading. See the repository README for commands and the starter configuration.
