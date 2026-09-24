# Musubi Tuner — обучение LoRA для Qwen-Image

Инструмент обучает адаптеры для **Qwen-Image original**: изображения с подписями → кэш латентов и текста → обучение. Есть валидация, TensorBoard, контрольные изображения и сохранение состояния для продолжения обучения.

## Быстрый старт

1. В папке `qwen_image_lora_val_example` настройте [train.toml](qwen_image_lora_val_example/train.toml): модели, число шагов и другие параметры обучения. В `train-dataset.toml` и `val-dataset.toml` укажите датасеты и каталоги кэша.
2. Запустите обучение:

```bash
python qwen_image_train_network.py --config_file /path/to/experiment/train.toml
```

Перед обучением скрипт проверит кэши всех указанных датасетов, создаст отсутствующие и пересчитает устаревшие кэши валидации. Параметры берутся из конфигов.
