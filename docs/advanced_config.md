# Advanced Qwen adapter settings

All options below use `qwen_image_train_network.py`; combine them with the [training template](../config_for_qwen_image_lora/train.toml). Training TOML supports one-level section flattening and explicit CLI overrides. Inspect the actual command's `--help` for the complete current interface.

## Adapters and optimization

`network_dim`, `network_alpha`, `network_dropout`, `network_weights` and `dim_from_weights` keep their existing meanings. LoRA scaling is alpha/rank; zero alpha retains the engine's existing rank fallback. `network_args` accepts consumed factory keys: `rank_dropout`, `module_dropout`, `verbose`, `include_patterns`, `exclude_patterns`, `loraplus_lr_ratio`, `conv_dim`, `conv_alpha`; LoKr additionally consumes `factor`. Convolution settings are retained shared-engine arguments; Qwen targets are linear projections. Pattern strings are Python lists of regex strings. Filters match original dotted module names; include patterns can override exclusions. The existing `_mod_` pattern is preserved literally and does not match every name containing `mod` after adapter-name flattening.

```toml
network_module = "networks.lora_qwen_image"
network_dim = 8
network_alpha = 8
network_args = ["rank_dropout=0.1", "loraplus_lr_ratio=8", "include_patterns=['.*attn.*']"]
optimizer_type = "AdamW"
optimizer_args = ["betas=(0.9, 0.99)", "weight_decay=0.01"]
learning_rate = 0.0001
gradient_accumulation_steps = 2
max_grad_norm = 1.0
```

Transformers Adafactor, AdamW8bit (bitsandbytes) and custom optimizer classes remain available. Optimizer/scheduler arguments are `key=Python_literal`. Adafactor's existing `relative_step`/`warmup_init` behavior can derive its own schedule; no algorithm is replaced. A class name ending in `schedulefree` retains the existing dummy scheduler and optimizer train/eval switching, requiring its own installed package.

`lr_scheduler` supports the existing Transformers schedules, `piecewise_constant` and `rex`; `lr_scheduler_type` selects a custom class such as `StepLR`, with `lr_scheduler_args=["step_size=100"]`. Integer warmup/decay values are steps; floats are ratios of total scheduler steps. CLI `20%` means 0.2. TOML `200.0` is a ratio and is not repaired to integer 200. Ordinary `constant` requires zero warmup; the supplied template uses `constant_with_warmup` with integer 100. Unused warmup fields do not restrict custom/schedule-free schedules. `max_train_epochs` derives update count using dataset length, process count and accumulation; it overrides `max_train_steps` as before.

## Timestep and loss formulas

Original flow training forms `x_t = (1-t) * latents + t * noise`, predicts `noise - latents`, and applies the existing weighted elementwise MSE then mean. DiT timesteps are divided by 1000. `weighting_scheme` remains `none`, `sigma_sqrt`, `cosmap`, `logit_normal` or `mode`; the last two also configure the existing density path when `timestep_sampling="sigma"`.

`uniform` draws uniform t; `sigmoid` uses `sigmoid(sigmoid_scale * normal)`. `shift` then uses `t * shift / (1 + (shift - 1) * t)`, with `discrete_flow_shift`. Resolution-dependent `qwen_shift`, `flux_shift`, `flux2_shift`, `krea2_shift` and `ideogram4_shift` are retained mathematical sampling methods, not extra model implementations. Qwen's mu interpolates `(256,0.5)` to `(8192,0.9)` using packed latent token count; Krea's uses `(256,0.5)` to `(6400,1.15)`. Flux variants retain their original token-count definitions and interpolation. Their formulas remain in the [shared timestep code](../src/musubi_tuner/training/timesteps.py) and [trainer](../src/musubi_tuner/training/trainer_base.py).

`logsnr` samples a normal log-SNR with `logit_mean`/`logit_std`, then `t=sigmoid(-logsnr/2)`, based on [Style-Friendly SNR sampling](https://arxiv.org/abs/2411.14793v3). `qinglong_qwen` uses 95% Qwen shift and 5% log-SNR normal(5.36,1). `qinglong_flux` preserves 79% shift, 11% configured log-SNR and 10% normal(5.36,1). `min_timestep`/`max_timestep` restrict the 0..1000 range; `preserve_distribution_shape` retains the existing rejection-sampling behavior instead of affine range scaling. `num_timestep_buckets` retains its existing bucketed sampling and shuffling.

![Log-SNR distribution](logsnr_distribution.png)
![Qinglong distribution](qinglong_distribution.png)
![Shift distribution](shift_3.png)
![Restricted range](shift_3_500_1000.png)
![Preserved distribution](shift_3_500_1000_preserve.png)

## Memory, logging and artifacts

SDPA, FlashAttention and xformers use priority SDPA → FlashAttention → xformers → Flash3. Unused flags do not require packages. Selected Flash3 and any Sage selection fail; padded multi-item text batches require split attention for selected FlashAttention/xformers. The selected optional backend must be installed and compatible with the runtime.

`fp8_base` uses the existing FP8 base-weight path; `fp8_scaled` requires it and retains dynamic scaled quantization. `fp8_vl` controls sample text encoding. These options do not change trainable adapter precision or replace the loss. `gradient_checkpointing`, `gradient_checkpointing_cpu_offload`, [block swap](block_swap.md) and [compilation](torch_compile.md) retain their existing implementations. Hardware support and performance need operational verification.

Use `logging_dir` and `log_with="tensorboard"`, `"wandb"` or `"all"`; corresponding packages are required. With no explicit `log_with`, a logging directory selects TensorBoard. `log_config` records filtered configuration; `log_grad_metrics` adds pre-clipping gradient metrics. `wandb_run_name`, `log_tracker_name` and `log_tracker_config` retain their roles. Optional wandb login/Hub tokens are only used in an authorized actual run.

Adapter saving uses `save_precision`, `save_every_n_steps`/`save_every_n_epochs`; `save_state` or `save_state_on_train_end` stores compatible Accelerate state. Metadata title/author/description/license/tags/resolution/timesteps and `no_metadata` remain supported. With both new modes disabled, the existing names, retention and Hub upload/resume paths remain available. Opt-in resume requires a local complete state; download remote state before selecting it.

`experiment_mode=true` uses `output/<output_name>-step<optimizer_step>/model.safetensors`, actual Accelerate optimizer/scheduler/per-rank RNG files, `trainer_state.json`, a completion manifest, and optional `samples/`. The sole adapter must be FP32 for any state save; lossy state precision is rejected early. Weights-only exports honor `save_precision`, but cannot resume. Periodic/final saves at the same step share one checkpoint. Samples-only or incomplete directories cannot resume.

`save_last_n_steps` and `save_last_n_steps_state` use inclusive completed-step windows. Missing/zero state window inherits the weight window; missing weight window is unlimited. At step100 with windows40/20, weights60..100 and states80..100 remain. With windows40/60, weights40..100 remain to support retained states. The shipped weight window is1000; add `save_last_n_steps_state=200` for a shorter state window. Cleanup preserves samples and unrelated output names. Legacy epoch retention flags retain their legacy meaning outside experiment mode; new-layout retention uses the step windows.

With validation or experiment mode enabled, `resume` restores the explicit completed optimizer step and `max_train_steps` is an absolute target. It continues the same TensorBoard run, purges abandoned later events and reconciles durable validation points by input/adapter/computation identity. Changed base-model contents, merged adapters, loss or relevant precision/forward settings cannot silently reuse prior metrics. With both modes off, legacy local counters still restart. Exact dataloader position is not guaranteed in either mode. `network_weights` initializes only an adapter; `base_weights` and multipliers merge into the frozen base. See the [README](../README.ru.md) for commands and artifact names.

## Deterministic validation

`val_dataset_config` enables validation independently of `experiment_mode`. Numerical defaults are interval50, seed42, N1=10, N2=2; explicit `val_*` numerical fields without a val dataset are errors. N1 must be even and at least2; interval/N2 must be positive integers; bool is not an integer setting.

Each image uses N1×N2 forwards with final mixing coefficient `t_i=0.05+(i-0.5)*0.90/N1`, independent of training shift/sampling. The trainer receives `1000*t_i` and loss weighting receives the same t. Seeds hash compact ASCII JSON `[val_seed_noise,lowercase_image_sha256,i,j]` using SHA-256; the first8 digest bytes are unsigned big-endian modulo2**63. Indices start at1. Dedicated CPU FP32 noise is cast to latent dtype/device. The same training forward/target/weighted loss is reused, then images receive equal weight.

Tags are exactly `val_loss_mean`, `val_loss_low_noise`, `val_loss_high_noise`; mean equals half the sum of low/high. Validation preserves Python/NumPy/CPU/CUDA RNG, per-module training flags and training updates, including evaluation exceptions. It runs at0, after complete periodic optimizer updates, and once at the final step. Microbatches and overflow-skipped updates do not advance this scale. Noise is repeatable in the same environment; identical results across different GPU/framework/kernel versions are not promised.
