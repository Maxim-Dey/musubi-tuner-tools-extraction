# Validation guide

Use supported Python >=3.10,<3.13 and pyproject-compatible packages plus TensorBoard. Activate the installed environment. For an uninstalled source checkout set `PYTHONPATH` to its absolute `src` directory. Run `python -m pytest -q tests` for local evidence; actual outcomes belong in validation.md. On Windows set `PYTHONIOENCODING=utf-8` when reading CLI help through a pipe.

Prepare independent nonempty dataset/train and dataset/val with captions under config_for_qwen_image_lora. Never split automatically. The following arguments were checked against the actual entrypoint help and readers; actual model execution is a separate remote acceptance gate. Run from the repository root:

```sh
python qwen_image_cache_latents.py --dataset_config config_for_qwen_image_lora/train-dataset.toml --vae /workspace/models/qwen_image_vae.safetensors --model_version original --experiment_mode
python qwen_image_cache_latents.py --dataset_config config_for_qwen_image_lora/val-dataset.toml --vae /workspace/models/qwen_image_vae.safetensors --model_version original --experiment_mode --validation
python qwen_image_cache_text_encoder_outputs.py --dataset_config config_for_qwen_image_lora/train-dataset.toml --text_encoder /workspace/models/qwen_2.5_vl_7b.safetensors --model_version original --experiment_mode
python qwen_image_cache_text_encoder_outputs.py --dataset_config config_for_qwen_image_lora/val-dataset.toml --text_encoder /workspace/models/qwen_2.5_vl_7b.safetensors --model_version original --experiment_mode --validation
accelerate launch qwen_image_train_network.py --config_file config_for_qwen_image_lora/train.toml
accelerate launch qwen_image_train_network.py --config_file config_for_qwen_image_lora/train.toml --resume output/qwen_image_lora_val_example-step50
tensorboard --logdir config_for_qwen_image_lora/output/tensorboard
```

From another working directory, use the installed package or absolute PYTHONPATH and absolute entrypoint/config paths. For example, with the repository at `/workspace/musubi-val`:

```sh
export PYTHONPATH=/workspace/musubi-val/src
cd /tmp
accelerate launch /workspace/musubi-val/qwen_image_train_network.py --config_file /workspace/musubi-val/config_for_qwen_image_lora/train.toml
```

`experiment_mode=true` anchors training-relative paths (including resume) to the training TOML. The two cache commands use explicit `--experiment_mode` and resolve paths against the dataset TOML. Dataset image/cache paths resolve against their own TOML. `--validation` requests fixed validation preparation and strict provenance; it does not implicitly change paths. Opt-in JSONL image paths resolve against their JSONL file. Absolute model paths remain unchanged. Legacy mode retains working-directory-relative semantics.

Validation training requires both train and val caches generated with provenance through the above opted-in commands. Missing or stale caches fail; training does not repair them. Val batch size and repeats must both be 1. The dataset files have no `role` field; `val_dataset_config` selects val.

A complete state checkpoint is `output/<output_name>-step<S>/`, containing one `model.safetensors`, native optimizer/scheduler/all-rank RNG files, `trainer_state.json` and `checkpoint_manifest.json`. Resume takes this entire directory. Sample PNGs are under the directory for their actual optimizer step, inside `samples/`; a samples-only directory cannot resume. FP32 adapter export is mandatory for resumable state. A weights-only export may use its requested export precision.

The user profile retains weights for an inclusive 1000-step window. Omitted or zero `save_last_n_steps_state` inherits 1000. To retain fewer resumable states, set e.g. `save_last_n_steps_state = 200`; adapter weights remain while either retention window needs them. An absent weight window is unlimited.

The TensorBoard tags are exactly `val_loss_mean`, `val_loss_low_noise`, `val_loss_high_noise`. Train and val use completed optimizer updates. Resume continues the same run and total `max_train_steps` target; resuming at or above the target performs no updates. Rollback suppresses the abandoned later trajectory. A durable result for the loaded adapter, inputs and loss computation is reused without a duplicate event.

Local `/workspace` models are not assumed available. The 5000-step, 1024-resolution, train-batch16 user profile is preserved. Separate configs under `config_for_qwen_image_lora/acceptance` use shorter targets, reduced resolution/batch, accumulation>1 and dropout>0; their real source/model paths must be verified remotely. They use separate caches and outputs and do not establish that the user batch16 fits the GPU. Record real GPU results only after the local gate.

Stop at3 then resume to5 preserves final3: combined events0,2,3,4,5; baseline3 is deduplicated if durable. Uninterrupted target5 gives0,2,4,5. A separate target4 gives0,2,4. These are distinct scenarios.
