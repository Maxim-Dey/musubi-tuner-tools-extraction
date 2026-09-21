# PNG samples during Dev LoRA training

The template supplies [one prompt](../flux2dev_lora/sample_prompts.txt), enables `sample_at_first` and samples every 250 optimizer updates. Mistral and AE resources are required for samples even when both training caches exist. Sample outputs use `<output_dir>/sample`; names retain the output prefix, update/epoch, prompt index, timestamp and optional seed.

TXT lines contain a prompt followed by options separated with ` --`. Blank lines and comment lines are ignored. Available switches are `--w`, `--h`, `--d` (seed), `--s` (steps), `--g` (guidance), `--fs` (flow shift), repeated `--ci` (control image), `--n` (negative prompt), and `--l` (legacy CFG scale). Optional `--f` accepts only 1. Unknown/partial switches and malformed numeric tokens fail. TXT step counts retain the existing clamp to 1–1000.

```text
A ceramic cup on a wooden table.
A ceramic cup --w 512 --h 512 --d 42 --s 20 --g 4
```

No options means 256×256, 20 steps and guidance 4.0. This does not change training resolution or hyperparameters. Width/height are normalized to the existing multiple-of-eight image dimensions. Dev uses guidance-distilled denoising: negative context is encoded for compatibility, but `--n`/`--l` do not introduce a different CFG algorithm.

JSON is a list of strings or records. TOML uses `[prompt]` defaults plus nonempty `[[prompt.subset]]` entries. Defaults and a subset cannot duplicate a key (preserving the original merge contract). Structured fields are `prompt`, `width`, `height`, `seed`, `sample_steps`, `guidance_scale`, `discrete_flow_shift`, `control_image_path` (string or list), `negative_prompt`, `cfg_scale`, and optional `frame_count=1`. Internal `enum` and encoded contexts are rejected from user input. Empty unusable files and unsupported formats fail before weights.

```toml
[prompt]
width = 512
height = 512
[[prompt.subset]]
prompt = "A ceramic cup"
seed = 42
```

Sampling preserves train/eval/offload transitions, global RNG restoration, PNG pixel conversion and tracker image events. `(B,C,1,H,W)` is an internal image representation. No standalone generation or video/audio sampling command remains. See [resources and artifacts](flux_2.md).
