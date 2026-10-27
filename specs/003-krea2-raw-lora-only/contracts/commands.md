# Command Contracts

Root wrappers: `krea2_cache_latents.py`, `krea2_cache_text_encoder_outputs.py`, `krea2_train_network.py`; equivalent package-module calls remain possible with `src` on Python's path.

Latent cache consumes `--dataset_config`, `--vae` and applicable batching/worker/device/cache/debug options. Text cache consumes the same dataset, `--text_encoder` and supported precision/batching/cache options. Both preserve Krea2 cache contracts.

Training consumes `--config_file config_for_krea2_raw_lora/train.toml`; explicit CLI overrides TOML, including values equal to defaults. Krea2 network aliases resolve to `musubi_tuner.networks.lora_krea2`. Unknown CLI/TOML fields, Turbo/foreign networks and invalid combinations fail before model loading with source/key/correction. Existing dynamic applicable optimizer/scheduler paths remain supported.

Dataset TOML retains `[general]`/`[[datasets]]` image settings; excluded fields fail. Image sample text/JSON/TOML formats retain prompt, negative text, dimensions, steps, CFG and seed. Text flags are `--n`, `--w`, `--h`, `--s`, `--l`, `--d`. Excluded video/control/edit sample options fail. No new public API is introduced.
