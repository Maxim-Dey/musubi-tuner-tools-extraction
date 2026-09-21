# Musubi Tuner — FLUX.2 Dev LoRA

[English](README.md) · [日本語](README.ja.md)

Форк поддерживает только **LoRA для изображений FLUX.2 Dev**: кэш латентов VAE, кэш текстовых выходов Mistral и обучение. Три корневых скрипта также запускаются как `python -m musubi_tuner.<имя_скрипта_без_py>`. Удалены другие модели, включая Klein, полное обучение модели, Self-Flow, LoHa/LoKr/LyCORIS, GUI, отдельная генерация, создание подписей, конвертация, merge/export и post-hoc EMA. Сохранены PNG-примеры во время обучения и внутренние операции с обычными LoRA.

## Подготовка

Нужны Python 3.10–3.12, совместимые PyTorch/torchvision и оборудование для BF16/AdamW8bit. Ниже пример установки для отдельного рабочего окружения с CUDA 12.8, а не часть локальных проверок. Другие сохранённые CUDA-варианты указаны в [pyproject.toml](pyproject.toml).

```sh
python -m venv .venv
# Activate .venv before the following commands.
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -m pip install tensorboard pytest "ruff>=0.12.10,<0.16"
python -m pip check
```

В Windows активация: `.\.venv\Scripts\Activate.ps1`; в POSIX: `source .venv/bin/activate`. Для консольного просмотра изображений и графиков timestep сохранены `ascii-magic==2.3.0` и `matplotlib==3.10.0`. Дополнительные выбранные backend/трекеры устанавливаются отдельно. Отсутствие bitsandbytes, TensorBoard или выбранного backend вызывает ошибку до весов, без замены настроек.

Подготовьте оригинальные файлы Dev DiT и AE, **все десять шардов Mistral** рядом с первым `model-00001-of-00010.safetensors` и ресурсы processor/tokenizer/chat template для `mistralai/Mistral-Small-3.1-24B-Instruct-2503` в обычном кэше Hugging Face. Processor проверяется без сети до весов и не выбирается путём `text_encoder`. Параметра `tokenizer_path` нет. [Пути и ресурсы](docs/flux_2.md).

## Порядок работы

Запускайте из корня репозитория. Согласованные файлы: [train.toml](flux2dev_lora/train.toml), [dataset.toml](flux2dev_lora/dataset.toml), [sample_prompts.txt](flux2dev_lora/sample_prompts.txt). Измените внешние пути под свой компьютер. В папке изображений нужны подготовленные пользователем UTF-8 подписи `.txt` с теми же именами; папки кэша, результатов и логов должны быть доступны для записи.

Эти команды **загружают реальные веса**. Сначала создайте оба кэша:

```sh
python flux_2_cache_latents.py --model_version dev --dataset_config ./flux2dev_lora/dataset.toml --vae ./models/flux2-dev/ae.safetensors --vae_dtype float32
python flux_2_cache_text_encoder_outputs.py --model_version dev --dataset_config ./flux2dev_lora/dataset.toml --text_encoder ./models/flux2-dev/text_encoder/model-00001-of-00010.safetensors
accelerate launch --num_processes 1 --mixed_precision bf16 flux_2_train_network.py --config_file ./flux2dev_lora/train.toml
```

Настройки шаблона сохранены: rank=alpha=32, dropout .05, BF16 и FP32 AE, SDPA/checkpointing, FP8 выключен, `flux2_shift`, AdamW8bit 1e-4, warmup 100, 2000 обновлений, batch 1 и accumulation 4, seed 42, два постоянных worker. `blocks_to_swap=20` остаётся комментарием. Полезные дополнительные Dev-настройки и оба имени модуля LoRA поддерживаются. Неизвестные и исключённые параметры отклоняются до весов.

LoRA и полное состояние сохраняются каждые 250 обновлений, окна хранения — по 1000 обновлений. PNG-примеры создаются в начале и каждые 250 обновлений в `<output_dir>/sample`; готовый prompt использует прежние defaults 256×256, 20 шагов, guidance 4. TensorBoard сохраняет исходные prefix/имя трекера в `./logs/flux2_dev_style`; просмотр: `tensorboard --logdir ./logs/flux2_dev_style`.

Файл LoRA `.safetensors` и папка полного состояния Accelerate — разные артефакты. `--resume` восстанавливает из новых состояний эпоху, число обновлений и позицию в данных, продолжая оставшиеся шаги с исходной конфигурацией. Старые состояния без метаданных прогресса загружаются с явным предупреждением: счётчики и данные начинают отсчёт заново, как раньше. `--dim_from_weights --network_weights <файл>` восстанавливает rank, alpha и веса адаптера. [Подробности](docs/flux_2.md).

## Документация и локальные проверки

- [Полный цикл Dev](docs/flux_2.md)
- [Датасет и controls](docs/dataset_config.md)
- [Дополнительные настройки и трекеры](docs/advanced_config.md)
- [Примеры во время обучения](docs/sampling_during_training.md)
- [Block swap](docs/block_swap.md), [torch.compile](docs/torch_compile.md)

Для локальных проверок используйте подготовленное CPU-окружение и переменные `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, вывод UTF-8:

```sh
python flux_2_cache_latents.py --help
python flux_2_cache_text_encoder_outputs.py --help
python flux_2_train_network.py --help
python -m pytest -p no:cacheprovider -q tests
```

Тесты включают шесть вариантов root/module `--help`, небольшие CPU-тензоры и временные файлы. Локальный этап не включает реальное обучение, GPU, скачивание ресурсов/весов, сборку пакетов или серверную проверку. Фактические результаты и недоступные проверки записаны в [baseline.md](specs/001-scope-flux2-dev-lora/baseline.md).

## Разработка и авторство

Сохранены [инструкции для разработчиков](CONTRIBUTING.md), [японская версия](CONTRIBUTING.ja.md), настройки Ruff, инструкции `.agents/skills`, личные файлы агентов, Spec Kit и CI. Упомянутой в прежнем README папки `.ai` в этом checkout нет; она не нужна для этих команд. Общие численные тесты не удаляются из-за исторических имён моделей.

Musubi Tuner by [kohya-ss](https://github.com/kohya-ss/musubi-tuner). Retained FLUX code derives from [Black Forest Labs](https://github.com/black-forest-labs/flux); some common code is copied/modified from Diffusers. Retained code uses Apache-2.0 notices in its sources; model/resource licenses are separate. Shared FP8/offloading credits remain in the [advanced](docs/advanced_config.md) and [block-swap](docs/block_swap.md) guides. This fork is unofficial and is not affiliated with model authors.

Upstream sponsor: [AiHUB](https://aihub.co.jp/top-en).

[![AiHUB](images/logo_aihub.png)](https://aihub.co.jp/top-en)

[Support upstream development](https://github.com/sponsors/kohya-ss/).
