# Musubi Tuner — обучение LoRA для Qwen-Image

Инструмент обучает адаптеры для **Qwen-Image original**: изображения с подписями → кэш латентов и текста → обучение. Есть валидация, TensorBoard, контрольные изображения и сохранение состояния для продолжения обучения.

## Быстрый старт

1. В папке `config_for_qwen_image_lora` настройте [train.toml](config_for_qwen_image_lora/train.toml): модели, число шагов и другие параметры обучения. В [train-dataset.toml](config_for_qwen_image_lora/train-dataset.toml) и [val-dataset.toml](config_for_qwen_image_lora/val-dataset.toml) укажите датасеты и каталоги кэша.
2. Запустите обучение:

```bash
python qwen_image_train_network.py --config_file config_for_qwen_image_lora/train.toml
```

В готовом конфиге автокэширование включено: перед обучением скрипт проверит кэши указанных датасетов и создаст отсутствующие. Если кэши устарели или повреждены, запуск остановится с объяснением причины. Параметры берутся из конфигов.
