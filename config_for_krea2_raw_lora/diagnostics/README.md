# Isolated Krea2 diagnostics

`short/` and `main-noise/` contain isolated diagnostic configurations with separate caches/output. Stage F supplies verified model paths and may select a fixed original-file subset in these configurations only. Do not use the user's `output/` directory for diagnostic logs, manifests or checkpoints.

Use a fixed nonoverlapping subset of the supplied train/val files without moving, renaming or modifying them. Each diagnostic owns its own dataset TOMLs, caches/manifest, output and TensorBoard directories. Main train.toml remains the full-data1000-step experiment.

Required short scenario: train batch_size=1, gradient_accumulation_steps=2, val_level_noise_n=2, val_seed_noise_n=1, val/save period2, max_train_steps=5. Keep max_train_steps=5 in both continuous and controlled-stop-after-checkpoint2/resume runs. Verify val0,2,4,5; resume adds4,5 only. A separate scenario uses the main N1=10,N2=2. Compare val/no-val trajectories with identical sampling conditions. Actual paths, commands, tolerances and results are recorded in specs/004-krea2-deterministic-val/verification.md.

No runnable model filenames are invented locally. Diagnostic execution and any subset selection happen during F; the absence of a real GPU run is not a pass.
