# Sample images during Qwen training

`sample_prompts` selects UTF-8 `.txt`, `.toml` or `.json` prompts. Qwen caches the positive and negative text embeddings by prompt text. An omitted negative prompt becomes a single space, preserving the original CFG path.

At step zero, only `sample_at_first` triggers sampling. Later, step and epoch schedules combine by OR: use positive `sample_every_n_steps` and/or `sample_every_n_epochs`. Omit all scheduling fields to disable samples. A supplied prompt file is still prepared by the existing trainer and needs VAE/text resources.

```text
TOK, a portrait --w 1024 --h 1024 --d 42 --s 30 --l 4.0 --fs 2.2 --n blurry
```

`w/h` are dimensions, `d` is seed, `s` is denoising steps, `l` is CFG scale, `fs` is sampling flow shift and `n` is the negative prompt. `#` comment lines and blank lines are ignored. Values must be complete and finite. Steps are positive; dimensions must be usable (at least 16). The existing consumer rounds dimensions down to multiples of 8, then latent packing accounts for 16-pixel boundaries. Use multiples of 16 for an exact requested output size. There is no frame/control/reference/output-name or guidance-embedding option.

JSON accepts a list of prompt objects or text lines. Objects use `prompt`, `negative_prompt`, `width`, `height`, `seed`, `sample_steps`, `cfg_scale`, `discrete_flow_shift`. The reader supplies `enum` for output naming. TOML shares defaults and overrides them per subset:

```toml
[prompt]
width = 512
height = 512
sample_steps = 20
[[prompt.subset]]
prompt = "TOK, daylight"
seed = 42
[[prompt.subset]]
prompt = "TOK, evening"
width = 768
```

PNG grids appear in `<output_dir>/sample`, using output name, step/epoch, prompt index, timestamp and optional seed. An active wandb tracker receives the PNGs. Normal sampling uses eval/no-grad, switches block swapping for inference and back, restores the previous training mode and torch RNG state, and does not update training weights or optimizer/scheduler state. Exceptional-exit restoration is not guaranteed by the existing implementation.

See [shipped prompts](../config_for_qwen_image_lora/sample_prompts.txt) and [Qwen training](qwen_image.md).

## 日本語

初回は`sample_at_first`、以降はステップ条件またはエポック条件でPNGを生成します。負のプロンプトを省略すると半角スペースが使われます。通常終了時は学習モードとtorch RNGを復元します。例外終了時の復元は保証しません。
