# Advanced Krea2 Raw LoRA configuration

The [starter](../config_for_krea2_raw_lora/train.toml) is an example. Inspect `python krea2_train_network.py --help` for all retained destinations and defaults. Run commands with the repository-root environment described in [README](../README.md).

## Configuration precedence

Settings resolve as parser defaults → training TOML → explicit CLI. CLI overrides work even when their value equals a parser default. TOML sections are flattened as before; duplicate flattened destinations are errors. Unknown keys, invalid types/choices and detectable incompatible combinations fail before model loading, identifying the input that needs correction.

For example, append `--learning_rate 2e-4` to the README training command to override its TOML learning rate. Paths are relative to the repository root when launched there.

## Adapters and optimizer settings

Use `network_module = "networks.lora_krea2"`. `network_dim`, `network_alpha` and `network_dropout` select rank, alpha and neuron dropout. Existing `network_args` use strings of `key=value`:

```toml
network_args = ["loraplus_lr_ratio=4", "rank_dropout=0.1", "module_dropout=0.1"]
```

To narrow targets, use Python-literal pattern lists inside the string values:

```toml
network_args = ["exclude_patterns=['.*']", "include_patterns=['.*attn.*']"]
```

The default wraps all DiT Linear layers. Pattern changes affect adapter compatibility; preserve compatible network settings when resuming. `--network_weights` initializes an existing adapter; `--dim_from_weights` derives its dimensions from those weights.

AdamW, AdamW8bit and Adafactor retain their existing handling. Dynamic optimizer classes remain available through `optimizer_type`, for example `torch.optim.AdamW`. Additional `optimizer_args` are `key=value` strings. Optional optimizer packages must be installed separately.

The constant starter scheduler uses zero warmup. Other retained scheduler choices, warmup/decay/cycle settings and custom `lr_scheduler_type`/`lr_scheduler_args` remain configurable. Adafactor's relative-step mode requires its matching scheduler configuration. Invalid keyword arguments and known incompatible optimizer/scheduler combinations fail before model weights load.

Gradient accumulation (`gradient_accumulation_steps`), clipping (`max_grad_norm`), precision, timestep distributions and loss weighting remain independent supported settings. `max_train_epochs`, when supplied, overrides the step limit. Example values are not program restrictions.

## Memory and attention

See [Krea2 options](krea2.md), [block swap](block_swap.md) and [compilation](torch_compile.md). Use scaled FP8 as a pair (`fp8_base` and `fp8_scaled`) or ConvRot INT8, rather than combining the two. Activation CPU offload requires gradient checkpointing. Alternative attention backends require compatible optional packages.

## Logging

Console progress is available without a tracker. Enable TensorBoard with `--logging_dir logs/krea2_raw_lora --log_with tensorboard`; `log_prefix` prefixes its run directory. View recorded logs with `tensorboard --logdir logs/krea2_raw_lora` in an environment with TensorBoard installed.

W&B uses `--log_with wandb`, or `all` for both trackers; install/configure W&B separately. `log_tracker_name`, `wandb_run_name` and `log_tracker_config` customize the tracker. `--log_grad_metrics` adds pre-clipping gradient norm, mean norm and maximum diagnostics to an active tracker.

## Saving and resume

`output_dir` and `output_name` control adapter names/location. Configure `save_every_n_steps` or `save_every_n_epochs` and their retention settings. `save_precision` controls adapter serialization independently from runtime mixed precision. Safetensors metadata and Krea2 adapter naming remain unchanged.

`save_state` saves training state at configured checkpoint intervals; `save_state_on_train_end` saves final state. Keep state directories when optimizer/scheduler/RNG continuation is needed. Resume by adding `--resume REPLACE_WITH_SAVED_STATE_DIRECTORY` to the original training command. `network_weights` alone does not resume a training run.

For images generated during training, see [sample configuration](sampling_during_training.md). They use the trained Raw adapter and do not replace the training loss.
