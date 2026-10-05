# Short real-model acceptance configurations

These configurations are separate from the parent 5000-step profile. They use 256-pixel buckets, batch size 1, accumulation 2 and LoRA dropout 0.05. The sample keeps one existing prompt text with reduced dimensions and two inference steps. The parent sample prompts are unchanged.

Model paths are the parent's `/workspace/models/...` declarations. They are **not verified locally**. Before running, inspect the supplied server, confirm the actual model files, and place links or copies of independently prepared real train and val images with captions into `acceptance/dataset/train` and `acceptance/dataset/val`. Do not split train automatically or reuse train images as val. No images or caches are shipped here.

Run from the repository root in the installed environment:

```sh
python qwen_image_cache_latents.py --dataset_config config_for_qwen_image_lora/acceptance/train-dataset.toml --vae /workspace/models/qwen_image_vae.safetensors --model_version original --experiment_mode
python qwen_image_cache_latents.py --dataset_config config_for_qwen_image_lora/acceptance/val-dataset.toml --vae /workspace/models/qwen_image_vae.safetensors --model_version original --experiment_mode --validation
python qwen_image_cache_text_encoder_outputs.py --dataset_config config_for_qwen_image_lora/acceptance/train-dataset.toml --text_encoder /workspace/models/qwen_2.5_vl_7b.safetensors --model_version original --experiment_mode
python qwen_image_cache_text_encoder_outputs.py --dataset_config config_for_qwen_image_lora/acceptance/val-dataset.toml --text_encoder /workspace/models/qwen_2.5_vl_7b.safetensors --model_version original --experiment_mode --validation
accelerate launch qwen_image_train_network.py --config_file config_for_qwen_image_lora/acceptance/train.toml
accelerate launch qwen_image_train_network.py --config_file config_for_qwen_image_lora/acceptance/resume.toml
accelerate launch qwen_image_train_network.py --config_file config_for_qwen_image_lora/acceptance/periodic-final.toml
```

Use the inspected model locations consistently in cache commands and training TOML if the server paths differ. From another working directory, pass absolute wrapper and configuration paths. Relative data/cache/output/resume locations remain anchored to their TOML files in experiment mode.

`train.toml` stops at optimizer step 3 and measures val at 0,2,3 with N1=10,N2=2. `resume.toml` loads `output-resume/qwen_image_val_acceptance-step3` and continues the same run to step 5; the combined val history is 0,2,3,4,5. `periodic-final.toml` runs independently under `output-periodic`, uses N1=2,N2=1 and measures 0,2,4 without repeating the final point. Samples are requested every two completed optimizer updates. Retention windows are four steps for weights and two for full states.

Actual GPU results, resource paths and any server-specific overrides belong in `specs/002-deterministic-val-loss/gpu-validation.md`. These small configurations do not establish that the parent batch size 16 at 1024 pixels fits or has been tested.
