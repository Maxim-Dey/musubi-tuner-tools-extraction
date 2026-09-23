# Musubi Tuner — адаптеры Qwen-Image original

Сохранён один процесс: изображения с подписями → кэш latent → кэш текстовых эмбеддингов → обучение LoRA/LoHa/LoKr исходной Qwen-Image. Работают контрольные PNG во время обучения, логирование, сохранение адаптеров/состояния и resume. Существующий алгоритм обучения сохранён. Другие архитектуры, Edit/Layered, видео/аудио/control, полное обучение модели, отдельная генерация/подписи/конвертация и GUI удалены.

## Среда и подготовка

Нужны Python >=3.10,<3.13 и зависимости из [pyproject.toml](pyproject.toml), включая совместимые PyTorch/torchvision. В манифесте сохранены CUDA-варианты `cu124`, `cu128`, `cu130`, `cu132`. AdamW8bit требует bitsandbytes; TensorBoard, wandb, сторонние оптимизаторы, FlashAttention/xformers и Triton для Inductor — соответствующие установленные пакеты. SDPA использует PyTorch. Конкретная модель GPU не зафиксирована.

Подготовьте исходные DiT и RGB VAE Qwen-Image, веса Qwen2.5-VL и ресурсы токенизатора `Qwen/Qwen-Image`, подпапка `tokenizer`. Вычисления DiT используют bf16; смешанная точность и точность сохранения адаптера задаются отдельно.

Работайте из корня репозитория с установленным пакетом либо задайте `PYTHONPATH`: в PowerShell `$env:PYTHONPATH = (Join-Path (Get-Location) 'src')`, в POSIX — `export PYTHONPATH="$PWD/src"`. Без `experiment_dir` все относительные пути, включая TOML/JSONL, считаются от текущей рабочей папки. Каталоги вывода, логов и кэша могут быть новыми.

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

Сохранены также три запуска через `python -m musubi_tuner.<имя_команды>`. Без `--train_config` кэш-команды не читают `train.toml`; `--config_file` они не принимают. Общие параметры: `--device`, положительный `--num_workers`, `--batch_size`, `--skip_existing`, `--keep_cache`. Batch кэширования ограничивает порцию кодирования; batch обучения задаёт датасет.

`skip_existing` проверяет только наличие файлов: после изменения исходных данных/весов пересоздайте затронутые кэши. Обычная очистка удаляет устаревшие кэш-файлы выбранного датасета; `keep_cache` сохраняет их. Latent debug поддерживает `image`/`console`, текстовый кэш — `--fp8_vl`; явный `--vae_dtype` для latent-кэша не поддерживается. Отсутствующие текстовые кэши вызывают предупреждение и пропуск; пустой эффективный датасет — раннюю ошибку.

## Подготовка валидации

Для Qwen-Image original вручную подготовьте два непустых набора изображений с подписями: `val_familiar` из обучающих изображений и `val_unfamiliar` из изображений вне обучения. Укажите их в отдельном `val-dataset.toml`: по одному `[[datasets]]` с `role = "val_familiar"` и `role = "val_unfamiliar"`, `image_directory` или `image_jsonl_file` и отдельным `cache_directory`. Эффективные `batch_size` и `num_repeats` должны равняться 1. Один и тот же файл TOML передаётся через `--dataset_config` обеим Qwen cache-командам; для этих наборов пересоздавайте оба кэша без `--skip_existing` после изменения источников.

Создайте `val-dataset.toml` в корне репозитория и подготовьте оба кэша для этих наборов теми же командами, что и для обучения:

```bash
python qwen_image_cache_latents.py --dataset_config val-dataset.toml --vae /srv/models/qwen_image_vae.safetensors --model_version original
python qwen_image_cache_text_encoder_outputs.py --dataset_config val-dataset.toml --text_encoder /srv/models/qwen_2.5_vl_7b.safetensors --model_version original
accelerate launch --mixed_precision bf16 qwen_image_train_network.py --config_file config_for_qwen_image_lora/train.toml --val_dataset_config val-dataset.toml
```

`val_dataset_config` можно указать также в training TOML. Без него обучение сохраняет прежнее поведение. При указанном пути тренер до загрузки весов проверяет подписи, изображения и оба кэша, затем использует фиксированные уровни и шум. Параметры: `val_every_n_steps` (по умолчанию 200), `val_seed_noise` (42), `val_level_noise_n` (10, чётное число от 2) и `val_seed_noise_n` (1, число от 1). [Правила входных данных](specs/002-val-loss-core/contracts/validation-inputs.md).

Validation loss рассчитывается с текущими весами LoRA: сначала среднее по всем проверкам одного изображения, затем среднее по изображениям набора. Размер изображения, бакет и число повторов обучения не меняют его вес. Для каждого набора записываются полное среднее, среднее при `t<0.5` и при `t>=0.5`:

| Набор | TensorBoard tags |
| --- | --- |
| `val_familiar` | `train_eval_loss_mean`, `train_eval_loss_low_noise`, `train_eval_loss_high_noise` |
| `val_unfamiliar` | `val_loss_mean`, `val_loss_low_noise`, `val_loss_high_noise` |

Валидация проходит до первого обновления на шаге 0, затем после завершённых обновлений с абсолютным номером, кратным `val_every_n_steps`, и на последнем завершённом шаге запуска. Совпавшие периодическая и финальная проверки выполняются один раз. Промежуточные microbatches и пропущенные обновления шаг не увеличивают. Шесть значений одного события публикуются только после успешного расчёта обоих наборов; прежние графики training loss остаются. [Контракт метрик и расписания](specs/003-val-loss-training/contracts/validation-training.md).

## Переносимый эксперимент Qwen-Image LoRA

Режим включается только явным `experiment_dir` для Qwen-Image `original` LoRA. Возьмите [отдельный пример](qwen_image_lora_val_example/) или создайте собственный `<root>/train.toml`, `<root>/train-dataset.toml` без ролей и `<root>/val-dataset.toml` с обеими ролями. В примере замените три абсолютных пути к серверным моделям и `TOK`, добавьте свои изображения с `.txt`-подписями и создайте оба типа кэша для train и двух validation-наборов. Пустые каталоги не являются готовыми данными. Точные четыре команды кэширования, запуск, TensorBoard и возобновление приведены в [инструкции Qwen-Image](docs/qwen_image.md#portable-qwen-image-lora-experiment). Прежний шаблон и поведение без `experiment_dir` сохранены. Тренер всегда требует `--config_file <root>/train.toml` и включённую валидацию. Например, добавьте в свой полный `train.toml`:

```toml
experiment_dir = "."
dataset_config = "train-dataset.toml"
val_dataset_config = "val-dataset.toml"
save_precision = "fp32"
save_last_n_steps = 1000
```

Относительный `experiment_dir` отсчитывается от выбранного `train.toml`, даже при запуске из другой папки; явный CLI переопределяет TOML. От корня эксперимента считаются относительные пути к моделям, конфигурациям, изображениям и кэшам внутри обоих dataset TOML, `image_path` в JSONL, prompts, tracker config и локальному `resume`. Абсолютные пути сохраняются, рабочая папка процесса не меняется. Вывод и TensorBoard фиксированы в `<root>/output` и `<root>/output/tensorboard`.

Для каждого из двух dataset TOML запустите обе Qwen cache-команды, явно выбирая его через `--dataset_config`. Например, для валидации (замените пути к весам своими):

```bash
python qwen_image_cache_latents.py --train_config /work/portrait/train.toml --dataset_config val-dataset.toml --vae models/vae.safetensors
python qwen_image_cache_text_encoder_outputs.py --train_config /work/portrait/train.toml --dataset_config val-dataset.toml --text_encoder models/text.safetensors
accelerate launch --mixed_precision bf16 qwen_image_train_network.py --config_file /work/portrait/train.toml
```

`--train_config` передаёт кэшу только корень из `train.toml`; `--experiment_dir` у cache CLI переопределяет его. Абсолютный корень можно указать без `--train_config`, относительный требует этот файл. Проверка источников и конфликтов кэша выполняется до загрузки VAE/text encoder.

На каждом нужном шаге сохраняется один полный пакет `<output_name>-step-<X>`: `output/current_training_states/` либо единственный лучший в `output/val_training_states/val-loss/`. Внутри — один FP32 `model.safetensors` LoRA, optimizer, scheduler, RNG каждого rank, `val_loss_state.json`, `experiment_state.json` и `samples/` с PNG при включённом сэмплинге. `save_precision` должен быть `fp32`/`float` или не задан; отдельные файлы адаптера и `*-state` в этом режиме не создаются.

Лучший пакет выбирает только строго меньшее конечное `val_loss_mean` набора `val_unfamiliar`; равенство сохраняет прежний лучший. Первый корректный результат может стать лучшим на шаге 0. Совпавшие периодический, эпохальный, финальный, лучший и sample-поводы дают один пакет и один набор PNG. Отдельный sample-повод также создаёт полный пакет; лучший шаг 0 получает PNG при включённом сэмплинге даже с `sample_at_first = false`. Сэмплинг включается prompt-файлом и расписанием; без них ресурсы генерации не подготавливаются.

`save_last_n_steps = 1000` оставляет целые текущие пакеты с шагом `s >= X-1000` после сохранения на шаге `X`; неуказанный предел хранит все. Лучший защищён и может быть старше окна; прежний лучший перемещается в текущие, если попадает в окно. `--resume` принимает опубликованный каталог как из `current_training_states`, так и из `val-loss`; возобновление с более старого текущего пакета сохраняет оценку уже существующего лучшего. Отдельные `save_last_n_steps_state`, `save_last_n_epochs`, `save_last_n_epochs_state` и `save_state_to_huggingface` в режиме эксперимента отклоняются. [Контракт пакета](specs/004-experiment-training-states/contracts/experiment-states.md).

## Контрольные изображения и настройки

Без `experiment_dir` начальный сэмпл зависит от `sample_at_first`. Дальше `sample_every_n_steps` и `sample_every_n_epochs` объединяются по OR. Интервалы положительные. Для отключения удалите prompt/schedule-параметры; указанный prompt-файл подготавливается даже без активного расписания. TXT-флаги: `--w/--h/--d/--s/--l/--fs/--n` — размеры, seed, шаги, CFG, flow shift, negative prompt. Есть TOML/JSON; отсутствующий negative prompt заменяется пробелом. PNG сохраняются в `<output_dir>/sample`. [Подробности](docs/sampling_during_training.md).

`log_with`: `tensorboard`, `wandb`, `all`. TensorBoard требует `logging_dir`; один logging_dir без явно выбранного backend также включает TensorBoard. `log_grad_metrics` добавляет метрики градиентов. Сохранены metadata и настройки Hub.

Имена адаптеров: `networks.lora_qwen_image`, `networks.loha`, `networks.lokr` и варианты с `musubi_tuner.`. Поддерживаются rank/alpha/dropout/patterns, начальные/базовые адаптеры, длительность, accumulation/clipping, workers, precision/FP8, checkpointing/CPU offload, block swap, compile/Dynamo, timestep/loss, оптимизаторы/scheduler, расписания, логи и Hub. См. [расширенные параметры](docs/advanced_config.md), [block swap](docs/block_swap.md), [compile](docs/torch_compile.md), [LoHa/LoKr](docs/loha_lokr.md) и `--help`.

Приоритет attention: SDPA → FlashAttention → xformers → Flash3. Неиспользованные флаги не требуют пакетов; выбранный Flash3 и любой Sage отклоняются. Для padded text batch больше одного FlashAttention/xformers требуют split attention. `fp8_scaled` требует `fp8_base`, persistent workers — ненулевой workers. Целый warmup означает шаги, дробный — долю; TOML `200.0` не превращается в 200 шагов. Custom/schedule-free сохраняют прежнее поведение.

## Сохранение и resume

Без `experiment_dir` имена адаптеров: `qwen_image_lora-step00000200.safetensors`, `qwen_image_lora-000001.safetensors`, финальный `qwen_image_lora.safetensors`. Точность задаёт `save_precision`. `save_state` сохраняет Accelerate state на предусмотренных границах и в конце; `save_state_on_train_end` отдельно включает финальное состояние. State-каталоги имеют суффикс `-state`, финальный — `qwen_image_lora-state`.

В этом обычном режиме `save_last_n_steps`/`save_last_n_epochs` и отдельные `save_last_n_steps_state`/`save_last_n_epochs_state` управляют хранением. Шаговое окно — число прошедших шагов, не количество файлов. Нулевое/неуказанное окно state использует окно checkpoint; исходное поведение на границах сохранено.

```bash
accelerate launch --mixed_precision bf16 qwen_image_train_network.py --config_file config_for_qwen_image_lora/train.toml --resume qwen_image_lora/output/qwen_image_lora-step00000200-state
```

Восстанавливаются адаптер, optimizer, scheduler и RNG. При включённой валидации каждый Accelerate state содержит `val_loss_state.json` с абсолютным числом завершённых обновлений, отпечатком входных данных и параметрами протокола. Resume проверяет этот файл и неизменность обоих наборов; старый state без достоверного шага можно возобновить только без валидации. Если сохранён шаг `s`, стартовая проверка записывается на `s`, первое новое значение training loss — на `s+1`; `max_train_steps` остаётся бюджетом обновлений текущего запуска. Локальные счётчики epoch/global-step запускаются заново, уже прочитанные батчи не пропускаются. Точное продолжение позиции данных не гарантируется. `--network_weights /srv/adapters/initial.safetensors` только инициализирует адаптер и начинает новую шкалу валидации с 0; `--base_weights /srv/adapters/base.safetensors` сливает базовые адаптеры до обучения. Это не resume оптимизатора.

## Проверки

[Локальные результаты](specs/001-scope-qwen-image-lora/validation.md) и [quickstart](specs/001-scope-qwen-image-lora/quickstart.md) отделяют CPU-проверки импортов, reader, малых тензоров и Accelerate от эксплуатационной проверки реальной модели. [Команды проверки валидации](specs/003-val-loss-training/quickstart.md) охватывают метрики, расписание и resume на малых CPU-примерах. GPU-производительность и полное обучение ими не подтверждаются. [Участие в разработке](CONTRIBUTING.md).

[Короткая проверка на RunPod L40S](specs/006-runpod-verification/verification.md): подготовка данных и кэшей прошла, 228 контролируемых CPU-тестов прошли, начальная валидация записала шесть метрик. Генерация обязательного PNG на шаге 0 дважды завершилась CUDA OOM; ни одного обновления и полного пакета состояния не получено. Проверка resume и длительного обучения на этой машине не выполнена.

[Повторная короткая проверка на RunPod RTX PRO 6000](specs/006-runpod-verification/runs/20260923T0705Z-b8ed90/verification.md): исходные настройки BF16 и 1024×1024 выполнили 38 ограниченных шагов без CUDA OOM. G01/G02 и проверка повторяемости валидации прошли; парное сравнение обучения и один сценарий возобновления выявили расхождения. Полная матрица не прошла; подробности и доказательства в отчёте.

[Дополнительное парное сравнение без валидации](specs/006-runpod-verification/runs/20260923T075021Z-offoff-f6ce8c/verification.md) воспроизвело численное расхождение при одинаковых входах. Поэтому по одному сравнению G05 нельзя заключить, что его вызвала валидация.

## Attribution and licenses

Derived from [Musubi Tuner by kohya_ss](https://github.com/kohya-ss/musubi-tuner). The project uses Apache License 2.0 except where retained third-party notices specify otherwise. Some code is copied and modified from [Diffusers](https://github.com/huggingface/diffusers). Original Qwen model/VAE copyright and Apache notices credit the Qwen-Image, Wan and HuggingFace teams and remain in their source files.

The extracted attention helper derives from [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), whose applicable original license remains relevant; the PNG helper comes from Musubi Tuner's Hunyuan sampling code. Block swap is based on 2kpr's implementation. LoHa/LoKr are based on [LyCORIS by KohakuBlueleaf](https://github.com/KohakuBlueleaf/LyCORIS).

Historical upstream notices are retained for provenance: [HunyuanVideo 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) followed its own license; [Wan2.1](https://github.com/Wan-Video/Wan2.1), [FramePack](https://github.com/lllyasviel/FramePack) and [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) code was Apache 2.0, with the latter also crediting dxqb/OneTrainer and ComfyUI-Flux2-INT8. Those standalone architecture/quantizer implementations are not part of this extraction.

Upstream sponsor: [AiHUB Inc.](https://aihub.co.jp/top-en)

![AiHUB Inc.](images/logo_aihub.png)
