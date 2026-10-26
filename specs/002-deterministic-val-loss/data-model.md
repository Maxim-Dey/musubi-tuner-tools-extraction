# Data model

- ValidationInputs: SHA-sorted records, manifest, fingerprint; verify_unchanged(), load_batch(index). Each record links source/caption/preprocessing to latent/text cache hashes. Paths/order excluded from identity; seed/N1/N2 included. Each image contributes once.
- ValidationResult: completed step, input/weight identity, mean/low/high, image and forward counts. Exactly three scalar tags; counts remain metadata.
- TrainerState: schema, optimizer_global_step, run location, validation fingerprint/latest result. Fresh → baseline → completed update → logged/evaluated → flushed → complete checkpoint. Resume validates metadata/input identity before loading.
- CheckpointManifest: completed step, canonical adapter digest and actual state-file inventory/rank count. Written after all state files. State expiry removes completeness; sample-only folders never become resumable.
- Experiment: main TOML root and dataset-relative resolved source/cache paths, output name, independent inclusive retention windows. Legacy paths unchanged.
- Evidence: requirement, code point, command/scenario/result and artifacts; local/GPU status separate; correction_rounds_used starts at zero.
