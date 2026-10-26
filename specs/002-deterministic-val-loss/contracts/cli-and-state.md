# CLI and state contracts

Training: val_dataset_config; val_every_n_steps/val_seed_noise/val_level_noise_n/val_seed_noise_n (defaults50/42/10/2 only with val); experiment_mode (false). Bool is not integer. Explicit CLI beats TOML. Unknown fields fail before models.

Both cache commands: explicit --experiment_mode anchors paths at dataset TOML; --validation requests strict deterministic val cache provenance. No --config_file is invented. Training paths including resume anchor at primary train.toml; dataset paths at dataset TOML. Absolute/legacy paths unchanged. No role field. Val batch/repeats=1; train/val cache dirs separate.

Seed/math contract is FR-007–011 and plan, including known vectors. Scalar tags: val_loss_mean, val_loss_low_noise, val_loss_high_noise.

New-layout checkpoint: output/<output_name>-step<completed_step>/model.safetensors, actual Accelerate state files, trainer metadata, complete manifest, optional samples/. One FP32 adapter serves export/load. Invalid/incomplete checkpoint or changed inputs rejected; legacy state stays loadable with both features off.
