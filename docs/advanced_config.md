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

`lr_scheduler` supports the existing Transformers schedules, `piecewise_constant` and `rex`; `lr_scheduler_type` selects a custom class such as `StepLR`, with `lr_scheduler_args=["step_size=100"]`. Integer warmup/decay values are steps; floats are ratios of total scheduler steps. CLI `20%` means 0.2. TOML `200.0` is a ratio and is not repaired to integer 200. Ordinary `constant` requires zero warmup; the template uses `constant_with_warmup` with integer 200. Unused warmup fields do not restrict custom/schedule-free schedules. `max_train_epochs` derives update count using dataset length, process count and accumulation; it overrides `max_train_steps` as before.

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

Adapter saving uses `save_precision`, `save_every_n_steps`/`save_every_n_epochs`, `save_last_n_steps`/`save_last_n_epochs`; `save_state` or `save_state_on_train_end` stores compatible Accelerate state. State retention has separate `save_last_n_steps_state`/`save_last_n_epochs_state` fields; a zero/None state window falls back to the checkpoint window. Step retention is an elapsed-step window, not a count. Metadata title/author/description/license/tags/resolution/timesteps and `no_metadata` remain supported. Optional Hub upload/resume uses the existing `huggingface_*`, `save_state_to_huggingface`, `resume_from_huggingface` and `async_upload` settings.

`network_weights` initializes only an adapter; `base_weights` and optional multipliers merge adapters into the frozen base. `resume` loads an Accelerate state directory and restores saved adapter/optimizer/scheduler/RNG state. It does not restore the local epoch/global-step counters or skip consumed batches; use suitable output locations and do not assume exact continuation. See the [README](../README.md) for commands and artifact names.
