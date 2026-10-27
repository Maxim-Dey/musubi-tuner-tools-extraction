# Raw sample images during training

Set `sample_prompts` and a sampling interval in the training configuration. Samples use the current Krea2 Raw DiT plus the trained adapter, with the same Qwen-Image VAE. `text_encoder` is required to prepare sample prompt representations before training.

```toml
sample_prompts = "config_for_krea2_raw_lora/sample_prompts.txt"
sample_every_n_steps = 200
# sample_every_n_epochs = 1
# sample_at_first = true
```

The default starter uses step intervals. `sample_at_first` also requests images at step zero. Epoch intervals are an alternative; sampling restores the training RNG state afterwards. The sample seed controls its initial noise separately.

## Text prompts

Put one prompt per nonempty line; lines beginning with `#` are comments:

```text
REPLACE_WITH_TRAINING_CAPTION --n blurry, low quality --w 1024 --h 1024 --s 28 --l 5.5 --d 42
```

| Option | Meaning |
|---|---|
| `--n` | Negative prompt; a nonempty value enables the unconditional CFG branch |
| `--w`, `--h` | Requested image dimensions, aligned to the VAE/DiT spatial grid |
| `--s` | Sampling steps |
| `--l` | CFG scale; scales above 1 combine unconditional and conditional predictions |
| `--d` | Sample seed |

Krea2 Raw uses its resolution-aware timestep schedule. The inherited embedded-guidance option is not part of Raw sampling. Unknown prompt flags and video/control/edit parameters are errors.

## Structured prompts

JSON accepts a list of prompt dictionaries:

```json
[{"prompt": "REPLACE_WITH_TRAINING_CAPTION", "negative_prompt": "blurry", "width": 1024, "height": 1024, "sample_steps": 28, "cfg_scale": 5.5, "seed": 42}]
```

TOML uses shared `[prompt]` fields and `[[prompt.subset]]` entries:

```toml
[prompt]
negative_prompt = "blurry"
width = 1024
height = 1024
sample_steps = 28
cfg_scale = 5.5
seed = 42

[[prompt.subset]]
prompt = "REPLACE_WITH_TRAINING_CAPTION"
```

Samples are saved to `sample/` under `output_dir`. Console output records progress; an active W&B tracker can also receive sample images. Adjust intervals, dimensions, steps and captions to your run.
