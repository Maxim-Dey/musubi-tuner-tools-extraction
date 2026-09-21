# Advanced Dev LoRA configuration

Training accepts CLI and optional TOML. Defaults are overridden by TOML, then explicit CLI arguments. Flat keys and arbitrary **one-level** grouping labels remain valid; labels such as `[model]`, `[lora]`, `[optimizer]` are examples, not a whitelist. Every contained key is checked before flattening. Unknown keys fail even if the CLI provides a valid override; known keys use their final effective value. Deeper grouping and malformed structures fail with source/key/value/correction.

```toml
[model]
model_version = "dev"
[lora]
network_module = "musubi_tuner.networks.lora_flux_2"
network_dim = 32
network_alpha = 32
```

Only `networks.lora_flux_2` and `musubi_tuner.networks.lora_flux_2` select the standard factory; the supplied spelling remains in metadata. `network_args` accepts unique `name=value` entries for `conv_dim`, `conv_alpha`, `rank_dropout`, `module_dropout`, `verbose`, `exclude_patterns`, `include_patterns`, `loraplus_lr_ratio`. Regex lists use Python literals; `verbose` uses `True`/`False`. Unknown/malformed/duplicate entries, invalid regexes and method/class injection fail. Default exclusions include modulation/norm layers. LoRA+ changes the up-matrix learning-rate ratio only when explicitly selected; the template does not enable it.

Optimizer/scheduler dispatch retains AdamW, bitsandbytes AdamW8bit, ordinary Adafactor, supported torch classes and installed dotted custom classes. Nested args are Python literals checked against the selected callable before construction. Forwarded kwargs need an identifiable supported constructor contract. Missing selected packages produce errors without substitution. Relative-step Adafactor keeps its existing scheduler/LR handling. Schedule-free optimizers use the existing dummy scheduler. `lr_scheduler_type` selects custom scheduler classes; `rex` remains available.

## Precision, memory and numeric options

DiT `fp8_base` and `fp8_scaled` remain optional; scaled FP8 requires base FP8. `fp8_text_encoder=true` is unsupported for Mistral. See [block swap](block_swap.md) and [torch.compile](torch_compile.md); these do not change the supplied settings.

Retained scalar timestep modes include `flux2_shift`, `qwen_shift`, `krea2_shift`, `ideogram4_shift`, `qinglong_flux`, `qinglong_qwen`, `logsnr` and the other choices shown by training `--help`. `logsnr2` is an internal component of the Qinglong mixture, not a selectable mode. Historical names denote numeric formulas, not enabled model implementations. Existing `min_timestep`/`max_timestep`, bucketing, logit and shift parameters remain; `num_timestep_buckets` <=1 retains disabled behavior. SD3 loss weighting options remain available. Do not change those values merely to pass a local test.

Shared sampler attribution: Style-Friendly SNR sampling and Qing Long's hybrid sampler are retained from upstream. [Distribution](qinglong_distribution.png). Common FP8 optimization credits the [HunyuanVideo implementation](https://github.com/Tencent/HunyuanVideo/blob/7df4a45c7e424a3f6cd7d653a7ff1f60cddc1eb1/hyvideo/modules/fp8_optimization.py) and [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe). These are attribution links, not supported commands.

## Logging

`logging_dir` enables TensorBoard by default; `log_with` explicitly accepts `tensorboard`, `wandb` or `all`. TensorBoard needs its installed package and logging directory. WandB needs its optional package and operational account setup; `wandb_run_name` and `wandb_api_key` retain their existing meanings. `log_prefix`, `log_tracker_name`, `log_config` and `log_grad_metrics` remain supported. Gradient metrics are pre-clipping; optional config logging omits credentials and internal validation fields.

`log_tracker_config` points to a separate TOML of initialization kwargs. It is read before weights, without constructing trackers, creating runs or contacting services. Its sections are backend names, not training groups:

```toml
[tensorboard]
flush_secs = 30
max_queue = 10
filename_suffix = ".experiment"
```

TensorBoard kwargs follow `SummaryWriter`; log directory is already supplied by Accelerate. WandB kwargs follow `wandb.init`; project is already supplied by the tracker name. Supported nested WandB `config` payloads retain user-defined keys and `settings` follows the installed backend's settings schema. Unknown sections/arguments, invalid types and unreadable/malformed files fail with their full source path. Initialization still occurs at the original training point.

See [workflow and resume compatibility](flux_2.md). Standalone merge/export, alternative adapters/objectives and full fine-tuning are absent.
