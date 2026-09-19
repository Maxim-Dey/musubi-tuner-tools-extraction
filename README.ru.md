# Musubi Tuner

[English](./README.md) | [日本語](./README.ja.md) | [Русский](./README.ru.md)

## Содержание

<details>
<summary>Нажмите, чтобы развернуть</summary>

- [Musubi Tuner](#musubi-tuner)
  - [Содержание](#содержание)
  - [Введение](#введение)
    - [Спонсоры](#спонсоры)
    - [Поддержать проект](#поддержать-проект)
    - [Недавние обновления](#недавние-обновления)
    - [Релизы](#релизы)
    - [Для разработчиков, использующих ИИ-агенты](#для-разработчиков-использующих-ии-агенты)
  - [Обзор](#обзор)
    - [Требования к оборудованию](#требования-к-оборудованию)
    - [Возможности](#возможности)
    - [Документация](#документация)
  - [Установка](#установка)
    - [Установка через pip](#установка-через-pip)
    - [Установка через uv](#установка-через-uv-экспериментально)
    - [Linux/MacOS](#linuxmacos)
    - [Windows](#windows)
  - [Загрузка моделей](#загрузка-моделей)
  - [Использование](#использование)
    - [Настройка датасета](#настройка-датасета)
    - [Предварительное кэширование и обучение](#предварительное-кэширование)
    - [Настройка Accelerate](#настройка-accelerate)
    - [Обучение и инференс](#обучение-и-инференс)
  - [Прочее](#прочее)
    - [Установка SageAttention](#установка-sageattention)
    - [Версия PyTorch](#версия-pytorch)
  - [Отказ от ответственности](#отказ-от-ответственности)
  - [Участие в разработке](#участие-в-разработке)
  - [Лицензия](#лицензия)

</details>

## Введение

Этот репозиторий содержит скрипты для обучения моделей LoRA (Low-Rank Adaptation) с архитектурами HunyuanVideo, Wan2.1/2.2, FramePack, FLUX.1 Kontext, FLUX.2 dev/klein, Qwen-Image, Z-Image и MiniMax-H3.

Репозиторий неофициальный и не связан с официальными репозиториями этих архитектур.

*Репозиторий находится в разработке.*

### Спонсоры

Мы благодарны следующим компаниям за щедрую поддержку:

<a href="https://aihub.co.jp/top-en">
  <img src="./images/logo_aihub.png" alt="AiHUB Inc." title="AiHUB Inc." height="100px">
</a>

### Поддержать проект

Если проект оказался полезен, рассмотрите возможность поддержать его развитие через [GitHub Sponsors](https://github.com/sponsors/kohya-ss/). Мы очень ценим вашу поддержку!

### Недавние обновления

Включены GitHub Discussions: мы открыли GitHub Discussions для вопросов сообщества, обмена знаниями и технической информацией. Issues используйте для сообщений об ошибках и запросов функций, Discussions — для вопросов и обмена опытом. [Присоединиться к обсуждению →](https://github.com/kohya-ss/musubi-tuner/discussions)

- 16 сентября 2026
    - Добавлена экспериментальная поддержка MiniMax-H3 (обучение LoRA и совместная генерация видео/аудио). Большое спасибо sdbds за исходный [PR #1018](https://github.com/kohya-ss/musubi-tuner/pull/1018) и последующие правки.
        - Подробности — в [документации](./docs/minimax_h3.md) и в [документации по обучению на одном кадре (изображение)](./docs/minimax_h3_1f.md). Список уже слитых возможностей и оставшейся работы ведётся в [roadmap поддержки MiniMax-H3](https://github.com/kohya-ss/musubi-tuner/issues/1029).
    - Добавлена int8-квантизация ConvRot замороженных базовых весов DiT для обучения LoRA Krea 2 (`--convrot_int8`) как альтернатива `--fp8_base --fp8_scaled`. См. [PR #1008](https://github.com/kohya-ss/musubi-tuner/pull/1008).
        - VRAM под веса уменьшается вдвое, как и при fp8. Основное преимущество — скорость на GPU без поддержки fp8 (серия RTX 30 и старше). Для fused-ядер требуется `triton`. Подробности — в [документации Krea 2](./docs/krea2.md#convrot-int8--convrot-int8).
    - Изменения в конфигурации датасета для файлов метаданных JSONL. Подробности — в [документации по настройке датасета](./docs/dataset_config.md).
        - Относительные пути в JSONL теперь также разрешаются относительно каталога с JSONL-файлом, если они не найдены относительно рабочей директории. [PR #1023](https://github.com/kohya-ss/musubi-tuner/pull/1023)
        - Записи видео могут содержать необязательное поле `audio_path` для архитектур с поддержкой аудио (сейчас MiniMax-H3); если поле опущено, используется sidecar-файл аудио с тем же именем или встроенная аудиодорожка. [PR #1020](https://github.com/kohya-ss/musubi-tuner/pull/1020), [PR #1021](https://github.com/kohya-ss/musubi-tuner/pull/1021)
        - Ключи вне общей схемы передаются в architecture-specific скрипты кэширования как дополнительные поля элемента. [PR #1094](https://github.com/kohya-ss/musubi-tuner/pull/1094)
    - Исправлена ошибка `--attn_mode sdpa` в общих attention-бэкендах; теперь это псевдоним `torch`. Спасибо rossnot [PR #1092](https://github.com/kohya-ss/musubi-tuner/pull/1092).
    - Исправлено игнорирование `enable_bucket` и `bucket_no_upscale` видеодатасетами при кэшировании латентов; путь кэширования видео всегда применял бакетинг независимо от настройки. Спасибо christopher5106 [PR #1100](https://github.com/kohya-ss/musubi-tuner/pull/1100).
        - **Изменение поведения:** видеодатасеты без `enable_bucket = true` теперь кэшируются в единственном заданном `resolution` (ресайз и центральный кроп), как это всегда было для изображений. Если вы полагались на бакетинг, не задавая его, добавьте `enable_bucket = true` в датасет. Иначе повторно запустите кэширование латентов (и кэширование выхода текстового энкодера для MiniMax-H3 `fl2va` / `ref2va`, чьи кэши содержат изменённые control-изображения), чтобы кэши соответствовали заданному разрешению.
    - Скрипты обучения теперь останавливаются при старте, если отсутствуют `--output_dir` или `--output_name`, вместо падения при первом сохранении. Спасибо rossnot [PR #1070](https://github.com/kohya-ss/musubi-tuner/pull/1070).
    - Krea 2: теперь учитывается `--gradient_checkpointing_cpu_offload` (offload активаций на CPU при gradient checkpointing). Спасибо rockerBOO [PR #1101](https://github.com/kohya-ss/musubi-tuner/pull/1101).
    - Krea 2: добавлен `--turbo_lora` для наложения Turbo LoRA поверх RAW-модели при генерации сэмплов во время обучения, как альтернатива `--turbo_dit`. Можно сочетать с block swap, fp8 и ConvRot int8. Подробности — в [документации Krea 2](./docs/krea2.md#sample-image-generation-during-training--学習中のサンプル画像生成). Спасибо rockerBOO [PR #1103](https://github.com/kohya-ss/musubi-tuner/pull/1103).

- 14 июля 2026
    - Добавлена опция `--log_grad_metrics` для логирования диагностики нормы градиента (`grad/norm`, `grad/mean_norm`, `grad/max`, измеряются до gradient clipping) в трекер. Спасибо rockerBOO [PR #988](https://github.com/kohya-ss/musubi-tuner/pull/988).
        - Полезно для диагностики взрыва / затухания градиента и выбора подходящего `--max_grad_norm`. По умолчанию выключено. Подробности — в [документации по расширенной конфигурации](./docs/advanced_config.md#log-gradient-metrics--勾配メトリクスのログ出力).

### Релизы

Мы благодарны всем, кто вносит вклад в экосистему Musubi Tuner через документацию и сторонние инструменты. Чтобы поддержать эти вклады, рекомендуем опираться на наши [релизы](https://github.com/kohya-ss/musubi-tuner/releases) как на стабильные точки отсчёта: проект активно развивается, возможны ломающие изменения.

Последний релиз и историю версий можно найти на [странице релизов](https://github.com/kohya-ss/musubi-tuner/releases).

### Для разработчиков, использующих ИИ-агенты

Репозиторий содержит рекомендуемые инструкции, которые помогают ИИ-агентам вроде Claude и Gemini понять контекст проекта и стандарты кода.

Чтобы ими пользоваться, нужно явно включить их, создав свой конфигурационный файл в корне проекта.

**Быстрая настройка:**

1.  Создайте файл `CLAUDE.md`, `GEMINI.md` и/или `AGENTS.md` в корне проекта.
2.  Добавьте в `CLAUDE.md` следующую строку, чтобы подключить рекомендуемый промпт репозитория (сейчас они почти одинаковые):

    ```markdown
    @./.ai/claude.prompt.md
    ```

    или для Gemini:

    ```markdown
    @./.ai/gemini.prompt.md
    ```

    Промпт также можно импортировать в кастомный файл агента, например `AGENTS.md`.

3.  Ниже строки импорта можно добавить свои личные инструкции (например: `Always include a short summary of the change before diving into details.`).

Так вы полностью контролируете инструкции агента и при этом используете общий контекст проекта. Файлы `CLAUDE.md`, `GEMINI.md` и `AGENTS.md` (а также `.mcp.json` Claude) уже указаны в `.gitignore`, поэтому они не попадут в репозиторий.

## Обзор

### Требования к оборудованию

- VRAM: рекомендуется 12 ГБ и больше для обучения на изображениях, 24 ГБ и больше для обучения на видео
    - *Фактические требования зависят от разрешения и настроек обучения.* Для 12 ГБ используйте разрешение 960x544 или ниже и опции экономии памяти: `--blocks_to_swap`, `--fp8_llm` и т.д.
- Оперативная память: рекомендуется 64 ГБ и больше, возможно 32 ГБ + swap

### Возможности

- Реализация с экономией памяти
- Совместимость с Windows подтверждена (совместимость с Linux подтверждена сообществом)
- Обучение на нескольких GPU (через [Accelerate](https://huggingface.co/docs/accelerate/index)), документация будет добавлена позже

### Документация

Подробности по конкретным архитектурам, конфигурациям и расширенным возможностям — в документации ниже.

**По архитектурам:**
- [HunyuanVideo](./docs/hunyuan_video.md)
- [Wan2.1/2.2](./docs/wan.md)
- [Wan2.1/2.2 (один кадр)](./docs/wan_1f.md)
- [FramePack](./docs/framepack.md)
- [FramePack (один кадр)](./docs/framepack_1f.md)
- [FLUX.1 Kontext](./docs/flux_kontext.md)
- [Qwen-Image](./docs/qwen_image.md)
- [Z-Image](./docs/zimage.md)
- [HiDream-O1-Image](./docs/hidream_o1.md)
- [HunyuanVideo 1.5](./docs/hunyuan_video_1_5.md)
- [Kandinsky 5](./docs/kandinsky5.md)
- [FLUX.2](./docs/flux_2.md)
- [MiniMax-H3](./docs/minimax_h3.md)
- [MiniMax-H3 (один кадр)](./docs/minimax_h3_1f.md)

**Общая конфигурация и использование:**
- [Настройка датасета](./docs/dataset_config.md)
- [Расширенная конфигурация](./docs/advanced_config.md)
- [Сэмплирование во время обучения](./docs/sampling_during_training.md)
- [Block Swap (CPU offloading для экономии памяти)](./docs/block_swap.md)
- [Инструменты и утилиты](./docs/tools.md)
- [Использование torch.compile](./docs/torch_compile.md)

## Установка

### Установка через pip

Требуется Python 3.10 или новее (проверено на 3.10).

Создайте виртуальное окружение и установите PyTorch и torchvision, соответствующие вашей версии CUDA.

Требуется PyTorch 2.5.1 или новее (см. [примечание](#версия-pytorch)).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Установите зависимости следующей командой.

```bash
pip install -e .
```

Опционально можно использовать FlashAttention и SageAttention (**только для инференса**; инструкции — в разделе [Установка SageAttention](#установка-sageattention)).

Опциональные зависимости для дополнительных функций:
- `ascii-magic`: проверка датасета
- `matplotlib`: визуализация timestep
- `tensorboard`: логирование хода обучения
- `prompt-toolkit`: интерактивное редактирование промптов в скриптах инференса Wan2.1 и FramePack. Если установлен, автоматически используется в интерактивном режиме. Особенно удобен в Linux.

```bash
pip install ascii-magic matplotlib tensorboard prompt-toolkit
```

### Установка через uv (экспериментально)

Можно установить через uv, но эта установка экспериментальная. Обратная связь приветствуется.

1. Установите uv (если его ещё нет в системе).

#### Linux/MacOS

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Следуйте инструкциям, чтобы вручную добавить путь к uv, пока не перезапустите сессию...

#### Windows

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Следуйте инструкциям, чтобы вручную добавить путь к uv, пока не перезагрузите систему... или просто перезагрузите систему на этом этапе.

## Загрузка моделей

Процедура загрузки моделей зависит от архитектуры. Инструкции — в документах по архитектурам в разделе [Документация](#документация).

## Использование


### Настройка датасета

См. [здесь](./docs/dataset_config.md).

### Предварительное кэширование

Процедура предварительного кэширования зависит от архитектуры. Инструкции — в документах по архитектурам в разделе [Документация](#документация).

### Настройка Accelerate

Запустите `accelerate config`, чтобы настроить Accelerate. На каждый вопрос выберите подходящие значения для вашего окружения (введите значение напрямую или выбирайте стрелками и Enter; значение в верхнем регистре — по умолчанию, если оно подходит, просто нажмите Enter). Для обучения на одной GPU отвечайте так:

```txt
- In which compute environment are you running?: This machine
- Which type of machine are you using?: No distributed training
- Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)?[yes/NO]: NO
- Do you wish to optimize your script with torch dynamo?[yes/NO]: NO
- Do you want to use DeepSpeed? [yes/NO]: NO
- What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]: all
- Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]: NO
- Do you wish to use mixed precision?: bf16
```

*Примечание*: в некоторых случаях может возникнуть ошибка `ValueError: fp16 mixed precision requires a GPU`. Тогда на шестой вопрос (`What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:`) ответьте `0`. Будет использована только первая GPU (id `0`).

### Обучение и инференс

Процедуры обучения и инференса сильно зависят от архитектуры. Подробные инструкции — в документах по архитектурам в разделе [Документация](#документация) и в документах по конфигурации.

## Прочее

### Установка SageAttention

sdbsd предоставил совместимую с Windows реализацию SageAttention и готовые wheels здесь: https://github.com/sdbds/SageAttention-for-windows. После установки triton, если версии Python, PyTorch и CUDA совпадают, можно скачать и установить готовый wheel со страницы [Releases](https://github.com/sdbds/SageAttention-for-windows/releases). Спасибо sdbsd за этот вклад.

Для справки ниже инструкции по сборке и установке. Может потребоваться обновить Microsoft Visual C++ Redistributable до последней версии.

1. Скачайте и установите wheel triton 3.1.0 под вашу версию Python [здесь](https://github.com/woct0rdho/triton-windows/releases/tag/v3.1.0-windows.post5).

2. Установите Microsoft Visual Studio 2022 или Build Tools for Visual Studio 2022 с поддержкой сборки C++.

3. Клонируйте репозиторий SageAttention в удобный каталог:
    ```shell
    git clone https://github.com/thu-ml/SageAttention.git
    ```

4. Откройте `x64 Native Tools Command Prompt for VS 2022` из меню Пуск в разделе Visual Studio 2022.

5. Активируйте venv, перейдите в папку SageAttention и выполните команду ниже. Если появится ошибка, что DISTUTILS не настроен, выполните `set DISTUTILS_USE_SDK=1` и повторите:
    ```shell
    python setup.py install
    ```

На этом установка SageAttention завершена.

### Версия PyTorch

Если для `--attn_mode` указано `torch`, используйте PyTorch 2.5.1 или новее (в более ранних версиях видео может получаться чёрным).

Если используете более раннюю версию, применяйте xformers или SageAttention.

## Отказ от ответственности

Этот репозиторий неофициальный и не связан с официальными репозиториями поддерживаемых архитектур.

Репозиторий экспериментальный и активно развивается. Использование сообществом и обратная связь приветствуются, но учтите:

- Не предназначен для промышленного использования
- Возможности и API могут меняться без предупреждения
- Некоторые функции всё ещё экспериментальные и могут работать не так, как ожидается
- Функции обучения на видео всё ещё в разработке

Если вы столкнулись с проблемами или ошибками, создайте Issue в этом репозитории и укажите:
- Подробное описание проблемы
- Шаги для воспроизведения
- Сведения об окружении (ОС, GPU, VRAM, версия Python и т.д.)
- Соответствующие сообщения об ошибках или логи

## Участие в разработке

Мы приветствуем вклад в проект! Подробности — в [CONTRIBUTING.md](./CONTRIBUTING.md).

## Лицензия

Код в каталоге `hunyuan_model` изменён на основе [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) и следует их лицензии.

Код в каталоге `hunyuan_video_1_5` изменён на основе [HunyuanVideo 1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) и следует их лицензии.

Код в каталоге `wan` изменён на основе [Wan2.1](https://github.com/Wan-Video/Wan2.1). Лицензия — Apache License 2.0.

Код в каталоге `frame_pack` изменён на основе [FramePack](https://github.com/lllyasviel/FramePack). Лицензия — Apache License 2.0.

Код в `modules/convrot_int8_kernels.py` изменён на основе [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) (в свою очередь на основе dxqb/OneTrainer и ComfyUI-Flux2-INT8). Лицензия — Apache License 2.0.

Остальной код — под Apache License 2.0. Часть кода скопирована и изменена из Diffusers.
