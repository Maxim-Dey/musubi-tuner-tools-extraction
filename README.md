# Musubi Tuner — обучение LoRA для Qwen-Image

Инструмент обучает адаптеры для **Qwen-Image original**: изображения с подписями → кэш латентов и текста → обучение. Есть валидация, TensorBoard, контрольные изображения и сохранение состояния для продолжения обучения.

## Быстрый старт

1. Создайте свою папку эксперимента из шаблона `qwen_image_lora_val_example`. В [train.toml](qwen_image_lora_val_example/train.toml) укажите пути к моделям и параметры обучения, в `train-dataset.toml` и `val-dataset.toml` — изображения с `.txt`-подписями и каталоги кэша.
2. Из корня репозитория соберите кэши и запустите обучение. Подставьте свои пути: кэш-скрипты не берут VAE и текстовый энкодер из `train.toml`.

```bash
EXP=/path/to/experiment
VAE=/path/to/vae.safetensors
TE=/path/to/text_encoder.safetensors
export PYTHONPATH=src

for ds in train-dataset.toml val-dataset.toml; do
  python qwen_image_cache_latents.py --train_config "$EXP/train.toml" --dataset_config "$ds" --vae "$VAE"
  python qwen_image_cache_text_encoder_outputs.py --train_config "$EXP/train.toml" --dataset_config "$ds" --text_encoder "$TE"
done

python qwen_image_train_network.py --config_file "$EXP/train.toml"
```

Число шагов, оптимизатор, валидация и сохранение задаются в `train.toml`.
Пути и настройки датасетов задаются в `train-dataset.toml` и `val-dataset.toml`