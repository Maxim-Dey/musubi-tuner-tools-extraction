# Quickstart

Run from repository root with an existing compatible environment, local Raw DiT, Qwen-Image VAE and Qwen3-VL-4B safetensors. The existing encoder resolves its tokenizer by repository ID; it must be cached or obtained by the user for an actual run. Commands below document operational use only.

1. Prepare images with matching `.txt` captions. Replace REPLACE_WITH_ model/data paths in the three starter files and replace the sample training caption.
2. PowerShell: `$env:PYTHONPATH = (Join-Path (Get-Location) 'src')`. POSIX: `export PYTHONPATH="$PWD/src"`.
3. Cache latents:

   ```text
   python krea2_cache_latents.py --dataset_config config_for_krea2_raw_lora/dataset.toml --vae REPLACE_WITH_QWEN_IMAGE_VAE.safetensors
   ```

4. Cache text with the same dataset:

   ```text
   python krea2_cache_text_encoder_outputs.py --dataset_config config_for_krea2_raw_lora/dataset.toml --text_encoder REPLACE_WITH_QWEN3_VL_4B.safetensors --batch_size 1
   ```

5. Train using the same VAE/encoder paths as caching:

   ```text
   accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 krea2_train_network.py --config_file config_for_krea2_raw_lora/train.toml
   ```

The starter requests periodic Raw samples, console logging and LoRA/state saves under its output directory. Sample CFG uses --l with nonempty --n. Optional logging and memory optimizations remain configurable; refer to README and --help. Example resolution, rank and duration do not imply hardware fit.

Resume a saved state by appending `--resume REPLACE_WITH_SAVED_STATE_DIRECTORY` to the training command and keeping compatible model/data/network settings. `--network_weights` loads adapters only and does not restore optimizer/scheduler/RNG state.

Local checks use `.venv/Scripts/python.exe -m pytest -q tests` with PYTHONPATH=src/offline flags, real imports and the three commands' --help. Exact results and unperformed verification are recorded separately.
