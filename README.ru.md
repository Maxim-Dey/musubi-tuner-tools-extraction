# Musubi Tuner — адаптеры Qwen-Image original

[English](README.md) · [日本語](README.ja.md)

Сохранён один процесс: изображения с подписями → кэш latent → кэш текстовых эмбеддингов → обучение LoRA/LoHa/LoKr исходной Qwen-Image. Работают контрольные PNG во время обучения, логирование, сохранение адаптеров/состояния и resume. Существующий алгоритм обучения сохранён. Другие архитектуры, Edit/Layered, видео/аудио/control, полное обучение модели, отдельная генерация/подписи/конвертация и GUI удалены.

## Среда и подготовка

Нужны Python >=3.10,<3.13 и зависимости из [pyproject.toml](pyproject.toml), включая совместимые PyTorch/torchvision. В манифесте сохранены CUDA-варианты `cu124`, `cu128`, `cu130`, `cu132`. AdamW8bit требует bitsandbytes; TensorBoard, wandb, сторонние оптимизаторы, FlashAttention/xformers и Triton для Inductor — соответствующие установленные пакеты. SDPA использует PyTorch. Конкретная модель GPU не зафиксирована.

Подготовьте исходные DiT и RGB VAE Qwen-Image, веса Qwen2.5-VL и ресурсы токенизатора `Qwen/Qwen-Image`, подпапка `tokenizer`. Вычисления DiT используют bf16; смешанная точность и точность сохранения адаптера задаются отдельно.

Работайте из корня репозитория с установленным пакетом либо задайте `PYTHONPATH`: в PowerShell `$env:PYTHONPATH = (Join-Path (Get-Location) 'src')`, в POSIX — `export PYTHONPATH="$PWD/src"`. Все относительные пути, включая TOML/JSONL, считаются от текущей рабочей папки. Каталоги вывода, логов и кэша могут быть новыми.

1. Подготовьте пары `portrait.png` + `portrait.txt` с UTF-8-подписями либо JSONL с `image_path` и `caption`.
2. В [dataset.toml](config_for_qwen_image_lora/dataset.toml) замените внешний путь к изображениям и выберите кэш. Поддерживаются несколько датасетов и переопределения; кэши должны быть раздельными. Для JSONL каталог кэша обязателен. [Правила датасета](docs/dataset_config.md).
3. В [train.toml](config_for_qwen_image_lora/train.toml) замените пути DiT/VAE/text и настройте вывод/логи. Две ссылки на поставляемые dataset/prompts исправлены. Комментарий H200, rank 16 и 1600 шагов — примеры. Scheduler шаблона — `constant_with_warmup`, warmup — целое 200.
4. Замените TOK и сцены в [sample_prompts.txt](config_for_qwen_image_lora/sample_prompts.txt).

Приоритет обучения: значения по умолчанию → TOML → явно указанный CLI. Неуказанные флаги сохраняют TOML; store-true не позволяет выключить истинный параметр через отрицательный флаг. Неизвестные поля, неверные типы и запрещённые режимы отклоняются заранее. `model_version` — только `original`.

## Оба кэша и обучение

Замените `/srv/...` реальными путями и настройте Accelerate под рабочую среду. Точность launcher должна согласовываться с TOML.

```bash
python qwen_image_cache_latents.py --dataset_config config_for_qwen_image_lora/dataset.toml --vae /srv/models/qwen_image_vae.safetensors --model_version original
python qwen_image_cache_text_encoder_outputs.py --dataset_config config_for_qwen_image_lora/dataset.toml --text_encoder /srv/models/qwen_2.5_vl_7b.safetensors --model_version original
accelerate launch --mixed_precision bf16 qwen_image_train_network.py --config_file config_for_qwen_image_lora/train.toml
```

Сохранены также три запуска через `python -m musubi_tuner.<имя_команды>`. Кэш-команды не читают `train.toml` и не принимают `--config_file`. Общие параметры: `--device`, положительный `--num_workers`, `--batch_size`, `--skip_existing`, `--keep_cache`. Batch кэширования ограничивает порцию кодирования; batch обучения задаёт датасет.

`skip_existing` проверяет только наличие файлов: после изменения исходных данных/весов пересоздайте затронутые кэши. Обычная очистка удаляет устаревшие кэш-файлы выбранного датасета; `keep_cache` сохраняет их. Latent debug поддерживает `image`/`console`, текстовый кэш — `--fp8_vl`; явный `--vae_dtype` для latent-кэша не поддерживается. Отсутствующие текстовые кэши вызывают предупреждение и пропуск; пустой эффективный датасет — раннюю ошибку.

## Контрольные изображения и настройки

Начальный сэмпл зависит от `sample_at_first`. Дальше `sample_every_n_steps` и `sample_every_n_epochs` объединяются по OR. Интервалы положительные. Для отключения удалите prompt/schedule-параметры; указанный prompt-файл подготавливается даже без активного расписания. TXT-флаги: `--w/--h/--d/--s/--l/--fs/--n` — размеры, seed, шаги, CFG, flow shift, negative prompt. Есть TOML/JSON; отсутствующий negative prompt заменяется пробелом. PNG сохраняются в `<output_dir>/sample`. [Подробности](docs/sampling_during_training.md).

`log_with`: `tensorboard`, `wandb`, `all`. TensorBoard требует `logging_dir`; один logging_dir без явно выбранного backend также включает TensorBoard. `log_grad_metrics` добавляет метрики градиентов. Сохранены metadata и настройки Hub.

Имена адаптеров: `networks.lora_qwen_image`, `networks.loha`, `networks.lokr` и варианты с `musubi_tuner.`. Поддерживаются rank/alpha/dropout/patterns, начальные/базовые адаптеры, длительность, accumulation/clipping, workers, precision/FP8, checkpointing/CPU offload, block swap, compile/Dynamo, timestep/loss, оптимизаторы/scheduler, расписания, логи и Hub. См. [расширенные параметры](docs/advanced_config.md), [block swap](docs/block_swap.md), [compile](docs/torch_compile.md), [LoHa/LoKr](docs/loha_lokr.md) и `--help`.

Приоритет attention: SDPA → FlashAttention → xformers → Flash3. Неиспользованные флаги не требуют пакетов; выбранный Flash3 и любой Sage отклоняются. Для padded text batch больше одного FlashAttention/xformers требуют split attention. `fp8_scaled` требует `fp8_base`, persistent workers — ненулевой workers. Целый warmup означает шаги, дробный — долю; TOML `200.0` не превращается в 200 шагов. Custom/schedule-free сохраняют прежнее поведение.

## Сохранение и resume

Имена адаптеров: `qwen_image_lora-step00000200.safetensors`, `qwen_image_lora-000001.safetensors`, финальный `qwen_image_lora.safetensors`. Точность задаёт `save_precision`. `save_state` сохраняет Accelerate state на предусмотренных границах и в конце; `save_state_on_train_end` отдельно включает финальное состояние. State-каталоги имеют суффикс `-state`, финальный — `qwen_image_lora-state`.

`save_last_n_steps`/`save_last_n_epochs` и отдельные `save_last_n_steps_state`/`save_last_n_epochs_state` управляют хранением. Шаговое окно — число прошедших шагов, не количество файлов. Нулевое/неуказанное окно state использует окно checkpoint; исходное поведение на границах сохранено.

```bash
accelerate launch --mixed_precision bf16 qwen_image_train_network.py --config_file config_for_qwen_image_lora/train.toml --resume qwen_image_lora/output/qwen_image_lora-step00000200-state
```

Восстанавливаются адаптер, optimizer, scheduler и RNG. Локальные счётчики epoch/global-step запускаются заново, уже прочитанные батчи не пропускаются. Точное продолжение позиции данных не гарантируется; учитывайте это при выборе вывода и дополнительных шагов. `--network_weights /srv/adapters/initial.safetensors` только инициализирует адаптер; `--base_weights /srv/adapters/base.safetensors` сливает базовые адаптеры до обучения. Это не resume оптимизатора.

## Проверки

[Локальные результаты](specs/001-scope-qwen-image-lora/validation.md) и [quickstart](specs/001-scope-qwen-image-lora/quickstart.md) отделяют CPU-проверки импортов, reader, малых тензоров и Accelerate от эксплуатационной проверки реальной модели. GPU-производительность и полное обучение ими не подтверждаются. [Участие в разработке](CONTRIBUTING.md).

## Attribution and licenses

Derived from [Musubi Tuner by kohya_ss](https://github.com/kohya-ss/musubi-tuner). The project uses Apache License 2.0 except where retained third-party notices specify otherwise. Some code is copied and modified from [Diffusers](https://github.com/huggingface/diffusers). Original Qwen model/VAE copyright and Apache notices credit the Qwen-Image, Wan and HuggingFace teams and remain in their source files.

The extracted attention helper derives from [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), whose applicable original license remains relevant; the PNG helper comes from Musubi Tuner's Hunyuan sampling code. Block swap is based on 2kpr's implementation. LoHa/LoKr are based on [LyCORIS by KohakuBlueleaf](https://github.com/KohakuBlueleaf/LyCORIS).

Historical upstream notices are retained for provenance: [HunyuanVideo 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) followed its own license; [Wan2.1](https://github.com/Wan-Video/Wan2.1), [FramePack](https://github.com/lllyasviel/FramePack) and [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) code was Apache 2.0, with the latter also crediting dxqb/OneTrainer and ComfyUI-Flux2-INT8. Those standalone architecture/quantizer implementations are not part of this extraction.

Upstream sponsor: [AiHUB Inc.](https://aihub.co.jp/top-en)

![AiHUB Inc.](images/logo_aihub.png)
