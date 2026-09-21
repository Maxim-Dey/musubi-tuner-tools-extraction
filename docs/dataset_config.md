# Image dataset configuration

Use [the supplied dataset](../flux2dev_lora/dataset.toml) for the directory/caption workflow. TOML and equivalent JSON are accepted. Paths are relative to the working directory (normally repository root). Each dataset must specify exactly one nonempty `image_directory` or `image_jsonl_file`. Video/audio/FramePack settings and unknown fields are rejected before weights.

```toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "./data/flux2_dev_style/images"
cache_directory = "./data/flux2_dev_style/cache"
num_repeats = 1
```

Common options are `resolution` (positive integer or two positive integers), `caption_extension`, positive `batch_size`/`num_repeats`, `enable_bucket`, and `bucket_no_upscale`. Per-dataset settings override general settings, then applicable CLI/runtime values and dataclass defaults. General sections do not accept arbitrary training options. With directory sources, provide same-stem UTF-8 caption files; the existing directory glob selects captioned images when a caption extension is set and sorts filenames. Prepared captions are consumed as-is; no caption-generation command is provided.

## JSONL and controls

For a metadata list, replace `image_directory` with `image_jsonl_file` and set a dedicated `cache_directory`. Each JSONL line is an object:

```json
{"image_path":"cup.png","caption":"a ceramic cup","control_path":"reference.png"}
```

An existing path relative to the working directory takes precedence; otherwise the image reader checks the JSONL directory. Missing paths are reported during entrypoint validation. Absolute paths retain their meaning. `image_path_0` is accepted as the target alias. `control_path`, or numbered `control_path_0`, `control_path_1`, etc., represent image controls, normalized in numeric order. Arbitrary item extras remain metadata and do not enable another model/objective.

A directory dataset may use `control_directory`: matching target basenames with optional numeric suffixes select ordered controls. JSONL controls belong in each record, not a `control_directory`. Retained per-dataset options `no_resize_control` and `control_resolution` control resizing. Historical aliases `flux_kontext_no_resize_control`, `qwen_image_edit_no_resize_control` and `qwen_image_edit_control_resolution` remain accepted with deprecation messages; they do not select another architecture.

Each dataset needs a distinct writable cache directory; for directory datasets it defaults to the image directory. Image/control buckets retain Dev's resolution grid, repeat and batching behavior. Control count and shapes distinguish training buckets. The singleton frame axis in sampling is an internal image shape, not video support. See [both caches and training](flux_2.md).
