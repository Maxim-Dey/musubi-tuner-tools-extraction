# Sample images during Raw training

Set `sample_prompts` and a sampling interval in the training configuration. Samples use the current Krea2 Raw DiT plus trained adapter by default. With `turbo_dit`, sampling temporarily swaps to the distilled Turbo base with that same adapter, then restores Raw; the Qwen-Image VAE is unchanged. `text_encoder` is required to prepare sample prompt representations before training.

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
REPLACE_WITH_TRAINING_CAPTION --w 1024 --h 1024 --s 28 --l 5.5 --d 42
```

| Option | Meaning |
|---|---|
| `--n` | Optional negative prompt; when omitted, CFG uses an encoded empty string |
| `--w`, `--h` | Requested image dimensions, aligned to the VAE/DiT spatial grid |
| `--s` | Sampling steps |
| `--l` | CFG scale; scales above 1 combine unconditional and conditional predictions |
| `--d` | Sample seed |

CFG is active when `--l` is greater than 1, including when `--n` is omitted. Set `--l 1` to disable CFG. The default CFG scale is 5.5.

Krea2 Raw uses its resolution-aware timestep schedule. The inherited embedded-guidance option is not part of Raw sampling. Unknown prompt flags and video/control/edit parameters are errors.

## Structured prompts

JSON accepts a list of prompt dictionaries:

```json
[{"prompt": "REPLACE_WITH_TRAINING_CAPTION", "width": 1024, "height": 1024, "sample_steps": 28, "cfg_scale": 5.5, "seed": 42}]
```

TOML uses shared `[prompt]` fields and `[[prompt.subset]]` entries:

```toml
[prompt]
width = 1024
height = 1024
sample_steps = 28
cfg_scale = 5.5
seed = 42

[[prompt.subset]]
prompt = "REPLACE_WITH_TRAINING_CAPTION"
```

Samples are saved to `sample/` under `output_dir`. Console output records progress; an active W&B tracker can also receive sample images. Adjust intervals, dimensions, steps and captions to your run.

## Experiment-mode sample ownership

In the supplied experiment configuration, `sample_prompts="sample_prompts.txt"` resolves from the main TOML directory. The ten supplied subjects use `--w 1024 --h 1024 --d 42 --s 8 --l 1`, with no `--n`. The unsupported `--fs` flag must not be supplied. CFG is independent of validation loss and does not participate in its calculation.

With experiment_mode and state saving, samples whose actual optimizer step is saved belong in that exact checkpoint's `samples/` folder. If save/sample intervals differ, or sample_at_first has no checkpoint, samples are saved separately with an explicit step number. They are never attached to the nearest different checkpoint. Sample schedules do not control initial, periodic or final validation. Ordinary mode retains its previous `sample/` output.


## Optional Turbo base

Set `turbo_dit` to the full Turbo safetensors path; add `turbo_dit_cache=true` only when sufficient CPU RAM is available. The current main config enables this mode; native BF16 resident/streaming GPU acceptance and post-test cleanup passed. Turbo uses fixed mu1.15 and defaults to8 steps/CFG1 only when those prompt fields are absent; explicit values are honored. The ten main prompts explicitly use8/1. No negative prompt is added and no prompt content, seed or resolution is changed.

Raw training and Raw val-loss are unchanged. Successful or exceptional sampling restores the Raw base and module/RNG/checkpointing state before further work. No flags means prior Raw behavior. Block swap and ConvRot INT8 are incompatible with full Turbo swapping; cached mode requires turbo_dit. See [memory accounting](krea2.md#turbo-samples-during-raw-training) before enabling the CPU cache. The trained adapter is applied directly on the Turbo base, without a second adapter or an irreversible merge.
