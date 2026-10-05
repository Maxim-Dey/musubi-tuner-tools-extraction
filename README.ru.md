# Musubi Tuner — адаптеры Qwen-Image original

Сохранён один процесс: изображения с подписями → кэш latent → кэш текстовых эмбеддингов → обучение LoRA/LoHa/LoKr исходной Qwen-Image. Детерминированная валидация измеряет общий loss и отдельно низкий/высокий шум, сохраняя алгоритм обучения. Поддерживаются PNG-сэмплы, TensorBoard, единый checkpoint и resume. Другие архитектуры, Edit/Layered, видео/аудио/control, полное обучение модели и GUI не входят в этот репозиторий.

## Среда и подготовка

Нужны Python >=3.10,<3.13 и зависимости из [pyproject.toml](pyproject.toml), включая совместимые PyTorch/torchvision. В манифесте сохранены CUDA-варианты `cu124`, `cu128`, `cu130`, `cu132`. AdamW8bit требует bitsandbytes; TensorBoard, wandb, сторонние оптимизаторы, FlashAttention/xformers и Triton для Inductor — соответствующие установленные пакеты. SDPA использует PyTorch. Конкретная модель GPU не зафиксирована.

Подготовьте исходные DiT и RGB VAE Qwen-Image, веса Qwen2.5-VL и ресурсы токенизатора `Qwen/Qwen-Image`, подпапка `tokenizer`. Вычисления DiT используют bf16; смешанная точность и точность сохранения адаптера задаются отдельно.

Работайте с установленным пакетом либо задайте `PYTHONPATH` из корня репозитория: в PowerShell `$env:PYTHONPATH = (Join-Path (Get-Location) 'src')`, в POSIX — `export PYTHONPATH="$PWD/src"`. Приведённые ниже команды предназначены для сервера с моделями по указанным `/workspace/models/...` путям; при другом расположении замените их в TOML и cache-командах.

1. Подготовьте непустые независимые `config_for_qwen_image_lora/dataset/train` и `dataset/val`: изображения и одноимённые UTF-8 `.txt`. Изображения val не должны встречаться в train; содержимое сравнивается по SHA-256. Автоматического разделения нет.
2. Проверьте [train-dataset.toml](config_for_qwen_image_lora/train-dataset.toml) и [val-dataset.toml](config_for_qwen_image_lora/val-dataset.toml): 1024×1024, buckets, без увеличения маленьких изображений; train batch=16, val batch=1, repeats=1. Кэши раздельны: `cache/train`, `cache/val`. Поле `role` не используется. [Правила датасета](docs/dataset_config.md).
3. В [train.toml](config_for_qwen_image_lora/train.toml) проверьте три файла модели. Профиль: 5000 суммарных обновлений, LoRA rank/alpha=32, dropout=0.05, AdamW8bit, lr=1e-4, warmup=100, bf16, `save_precision="fp32"`. Val, samples и сохранения выполняются каждые 50 шагов.
4. Сохранённые пользовательские [sample_prompts.txt](config_for_qwen_image_lora/sample_prompts.txt) содержат 10 сцен. Их наличие не заменяет отдельный val-набор.

Приоритет обучения: значения по умолчанию → TOML → явно указанный CLI. Неуказанные флаги сохраняют TOML; store-true не позволяет выключить истинный параметр через отрицательный флаг. Неизвестные поля, неверные типы и запрещённые режимы отклоняются заранее. `model_version` — только `original`.

## Пути и четыре прохода кэширования

В примере включён `experiment_mode=true`: корень эксперимента — каталог основного `train.toml`. Относительные пути этого файла, включая `resume`, считаются от него; `image_directory`, `image_jsonl_file`, `cache_directory` внутри dataset TOML — от соответствующего TOML, а `image_path` внутри JSONL — от каталога JSONL. Абсолютные пути сохраняются. Cache-команды включают эту семантику явным `--experiment_mode`; относительные пути их моделей также считаются от dataset TOML. Без этого флага остаются прежние пути от рабочей папки.

Из корня репозитория:

```bash
export PYTHONPATH="$PWD/src"
python qwen_image_cache_latents.py --dataset_config config_for_qwen_image_lora/train-dataset.toml --vae /workspace/models/qwen_image_vae.safetensors --model_version original --experiment_mode
python qwen_image_cache_latents.py --dataset_config config_for_qwen_image_lora/val-dataset.toml --vae /workspace/models/qwen_image_vae.safetensors --model_version original --experiment_mode --validation
python qwen_image_cache_text_encoder_outputs.py --dataset_config config_for_qwen_image_lora/train-dataset.toml --text_encoder /workspace/models/qwen_2.5_vl_7b.safetensors --model_version original --experiment_mode
python qwen_image_cache_text_encoder_outputs.py --dataset_config config_for_qwen_image_lora/val-dataset.toml --text_encoder /workspace/models/qwen_2.5_vl_7b.safetensors --model_version original --experiment_mode --validation
```

Сохранены три запуска через `python -m musubi_tuner.<имя_команды>`. Cache-команды не читают `train.toml` и не принимают `--config_file`. `--validation` включает строгую подготовку val независимо от семантики путей: batch/repeats=1, неизменные подписи, детерминированное кодирование и сведения об источнике в кэшах. Случайные аугментации, caption dropout/shuffle и пропуск val-изображений недопустимы.

Суффиксы существующего формата не меняются: latent — `*_qi.safetensors`, текст — `*_qi_te.safetensors`. При включённой валидации отсутствие, повреждение или устаревание обязательных кэшей вызывает ошибку до загрузки модели; обучение их не пересоздаёт. После сознательного изменения входов подготовьте кэши заново и начните новую серию измерений. С `--validation` флаг `--skip_existing` проверяет происхождение существующих кэшей; без `--validation` сохраняется прежняя проверка наличия. `--keep_cache` отключает обычную очистку устаревших файлов выбранного cache-каталога.

## Обучение и val-loss

Настройте Accelerate под рабочую среду. Запуск профиля из корня:

```bash
accelerate launch --mixed_precision bf16 qwen_image_train_network.py --config_file config_for_qwen_image_lora/train.toml
```

Запуск из другой рабочей папки на POSIX-сервере (первые две строки выполните в корне репозитория):

```bash
REPO="$PWD"
export PYTHONPATH="$REPO/src"
cd /tmp
accelerate launch --mixed_precision bf16 "$REPO/qwen_image_train_network.py" --config_file "$REPO/config_for_qwen_image_lora/train.toml"
```

`val_dataset_config` включает валидацию независимо от `experiment_mode`. Без val-набора явно заданные `val_*` параметры отклоняются. Значения по умолчанию при включённой val: `val_every_n_steps=50`, `val_seed_noise=42`, `val_level_noise_n=10`, `val_seed_noise_n=2`. Все — целые, bool не принимается; interval/N2 положительны, N1 чётное и не меньше 2.

Каждое изображение измеряется ровно N1×N2 раз. Уровни `t_i=0.05+(i-0.5)*0.90/N1`, i=1..N1; низкий шум — t<0.5, высокий — t≥0.5. Используются та же формула loss и веса, что в train, с фактическим фиксированным t. Изображения имеют равный вес независимо от разрешения. Шумы зависят от seed, SHA-256 изображения и индексов уровня/реализации, поэтому перенос или перестановка файлов их не меняет. Train-набор в эти метрики не входит.

TensorBoard получает ровно три новых scalar tags: `val_loss_mean`, `val_loss_low_noise`, `val_loss_high_noise`; общий loss равен полусумме двух групп. Сохраняются прежние train-loss tags. В новом режиме ось шагов отражает завершённые обновления optimizer: накопление градиентов и пропущенный overflow-update её не увеличивают. Val выполняется на 0, после заданных интервалов и в конце без повторной точки. Для target=5/interval=2 это 0,2,4,5; для остановки на 3 и resume до 5 — 0,2,3,4,5.

```bash
tensorboard --logdir config_for_qwen_image_lora/output/tensorboard
```

## Контрольные изображения и настройки

Начальный сэмпл зависит от `sample_at_first`. Дальше `sample_every_n_steps` и `sample_every_n_epochs` объединяются по OR. TXT-флаги: `--w/--h/--d/--s/--l/--fs/--n` — размеры, seed, шаги, CFG, flow shift, negative prompt. В experiment mode PNG сохраняются в `output/<output_name>-step<фактический шаг>/samples/`, даже если на этом шаге checkpoint не запланирован; legacy использует `<output_dir>/sample`. [Подробности](docs/sampling_during_training.md).

`log_with`: `tensorboard`, `wandb`, `all`. TensorBoard требует `logging_dir`; один logging_dir без явно выбранного backend также включает TensorBoard. `log_grad_metrics` добавляет метрики градиентов. Сохраняется metadata адаптера; прежние операции Hub относятся к legacy layout.

Имена адаптеров: `networks.lora_qwen_image`, `networks.loha`, `networks.lokr` и варианты с `musubi_tuner.`. Поддерживаются rank/alpha/dropout/patterns, начальные/базовые адаптеры, длительность, accumulation/clipping, workers, precision/FP8, checkpointing/CPU offload, block swap, compile/Dynamo, timestep/loss, оптимизаторы/scheduler, расписания, логи и Hub. См. [расширенные параметры](docs/advanced_config.md), [block swap](docs/block_swap.md), [compile](docs/torch_compile.md), [LoHa/LoKr](docs/loha_lokr.md) и `--help`.

Приоритет attention: SDPA → FlashAttention → xformers → Flash3. Неиспользованные флаги не требуют пакетов; выбранный Flash3 и любой Sage отклоняются. Для padded text batch больше одного FlashAttention/xformers требуют split attention. `fp8_scaled` требует `fp8_base`, persistent workers — ненулевой workers. Целый warmup означает шаги, дробный — долю; TOML `200.0` не превращается в 200 шагов. Custom/schedule-free сохраняют прежнее поведение.

## Сохранение и resume

В `experiment_mode` файлы одного шага находятся вместе:

```text
config_for_qwen_image_lora/
  train.toml, train-dataset.toml, val-dataset.toml, sample_prompts.txt
  dataset/train/, dataset/val/
  cache/train/, cache/val/
  output/tensorboard/
  output/qwen_image_lora_val_example-step50/
    model.safetensors
    optimizer.bin, scheduler.bin, random_states_0.pkl
    trainer_state.json, checkpoint_manifest.json
    samples/
```

Имена дополнительных файлов определяет Accelerate; каждый процесс сохраняет свой RNG. Единственный `model.safetensors` одновременно служит адаптером и весами для resume; DiT не сохраняется. Для `save_state=true` и `save_state_on_train_end=true` требуется FP32. Сохранение только весов учитывает `save_precision`, но такой каталог не resumable. Каталог только с samples или незавершённым manifest также не является state. Совпадение periodic/final не создаёт второй checkpoint.

`save_last_n_steps` и `save_last_n_steps_state` — окна в завершённых optimizer steps с включённой нижней границей. В профиле окно весов — 1000; отсутствующее или нулевое окно state наследует 1000. Для меньшего окна добавьте `save_last_n_steps_state=200`. Без окна весов ограничения нет; если state живёт дольше весов, необходимые для resume веса сохраняются вместе с ним. Очистка затрагивает только принадлежащие checkpoint файлы выбранного `output_name`, сохраняя samples.

Resume из полного checkpoint (из корня репозитория):

```bash
accelerate launch --mixed_precision bf16 qwen_image_train_network.py --config_file config_for_qwen_image_lora/train.toml --resume output/qwen_image_lora_val_example-step50
```

При val или experiment mode `max_train_steps` означает целевой суммарный шаг: resume с 50 при цели 5000 продолжает до 5000; если цель уже достигнута, новых обновлений нет. Восстанавливаются адаптер, optimizer, scheduler, RNG и явный optimizer step. TensorBoard продолжает тот же run, скрывает отброшенные события после checkpoint и не дублирует подтверждённую val-точку. Изменение val/кэшей, базовых весов, loss или существенных precision/forward-настроек прерывает прежнюю серию с понятной ошибкой. Новый режим требует локальный полный state; state из Hub предварительно скачайте целиком.

Точное восстановление позиции train dataloader не гарантируется. `network_weights` только инициализирует адаптер; `base_weights` сливает адаптеры в базовую модель до обучения. При выключенных val и experiment mode сохраняются старые имена `.safetensors`/`-state`, пути и legacy resume; неизвестный старый optimizer step нельзя использовать для новой шкалы.

## Проверки

[Проверки val-loss](specs/002-deterministic-val-loss/validation.md) и [quickstart](specs/002-deterministic-val-loss/quickstart.md) содержат локальные результаты и команды. CPU-проверки не подтверждают наличие локальных `/workspace` ресурсов, работоспособность исходного GPU batch=16 или полный профиль 5000 шагов. Короткие GPU acceptance-конфиги отделены от пользовательского профиля; фактическая проверка модели выполняется отдельно на сервере. [Предыдущие CPU-проверки](specs/001-scope-qwen-image-lora/validation.md), [участие в разработке](CONTRIBUTING.md).

## Attribution and licenses

Derived from [Musubi Tuner by kohya_ss](https://github.com/kohya-ss/musubi-tuner). The project uses Apache License 2.0 except where retained third-party notices specify otherwise. Some code is copied and modified from [Diffusers](https://github.com/huggingface/diffusers). Original Qwen model/VAE copyright and Apache notices credit the Qwen-Image, Wan and HuggingFace teams and remain in their source files.

The extracted attention helper derives from [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), whose applicable original license remains relevant; the PNG helper comes from Musubi Tuner's Hunyuan sampling code. Block swap is based on 2kpr's implementation. LoHa/LoKr are based on [LyCORIS by KohakuBlueleaf](https://github.com/KohakuBlueleaf/LyCORIS).

Historical upstream notices are retained for provenance: [HunyuanVideo 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) followed its own license; [Wan2.1](https://github.com/Wan-Video/Wan2.1), [FramePack](https://github.com/lllyasviel/FramePack) and [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) code was Apache 2.0, with the latter also crediting dxqb/OneTrainer and ComfyUI-Flux2-INT8. Those standalone architecture/quantizer implementations are not part of this extraction.

Upstream sponsor: [AiHUB Inc.](https://aihub.co.jp/top-en)

![AiHUB Inc.](images/logo_aihub.png)
