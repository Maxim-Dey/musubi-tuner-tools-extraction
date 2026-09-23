# Musubi Tuner — обучение LoRA для Qwen-Image

Инструмент обучает адаптеры для **Qwen-Image original**: изображения с подписями → кэш латентов и текста → обучение. Есть валидация, TensorBoard, контрольные изображения и сохранение состояния для продолжения обучения.

## Быстрый старт

1. Создайте папку эксперимента из шаблона `qwen_image_lora_val_example`. В [train.toml](qwen_image_lora_val_example/train.toml) настройте модели и обучение, в `train-dataset.toml` и `val-dataset.toml` — изображения, подписи и каталоги кэша.
2. Запустите обучение:

```bash
python qwen_image_train_network.py --config_file /path/to/experiment/train.toml
```

Перед обучением скрипт проверит кэши всех указанных датасетов, создаст отсутствующие и пересчитает устаревшие кэши валидации. Параметры берутся из конфигов.
