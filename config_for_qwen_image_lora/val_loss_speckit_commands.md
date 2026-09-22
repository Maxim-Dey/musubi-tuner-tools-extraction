# Команды Spec Kit: детерминированный val-loss

Отправляйте каждый блок отдельным сообщением ИИ-агенту в этом репозитории. Используется установленный синтаксис Codex **$speckit-…**; это команды агенту, не терминалу. Конституция уже принята.

Последовательность соответствует [процессу Spec Kit](https://github.github.com/spec-kit/quickstart.html) и Конституции проекта: specify → plan → tasks → analyze → implement → converge. Следующий этап начинайте после завершения предыдущего. Источниками требований станут новые формальные спецификации; обращаться к исходным черновикам не требуется.

| Этап | Результат |
| --- | --- |
| 1 | Фиксированные val-датасеты, идентификаторы, уровни шума и seeds |
| 2 | Расчёт метрик в тренере, TensorBoard, расписание и resume |
| 3 | Единая папка эксперимента, полные states и один лучший checkpoint |
| 4 | Физически созданные конфиги и проверенная инструкция запуска |
| 5 | Конечный набор проверок на RunPod с реальной моделью, GPU и отчётом |

Документы Spec Kit пишутся на английском, общение ведётся на русском. Этапы 1–4 остаются локальными: код, конфиги и CPU-проверки без обучения, GPU-прогонов и переноса на сервер. Этап 5 выполняется отдельно после их завершения и разрешает подготовку RunPod и ограниченные по числу шагов GPU-тесты. Это предусмотренный Конституцией отдельный серверный этап; запрет удалённых запусков во время локального этапа сохраняется. Повторное создание Конституции не требуется.

Отсутствие пользовательского лимита времени не означает бесконечные прогоны: этап 5 завершается после проверки заданных сценариев и подготовки отчёта. Дообучение ради качества, подбор гиперпараметров и ожидание снижения loss в него не входят. Pod выключает пользователь.

После analyze сначала исправляются подтверждённые противоречия документов. После converge исправления требуют отдельного согласования конкретного списка: это правило раздела Development Workflow в [.specify/memory/constitution.md](../.specify/memory/constitution.md). Допускается не более двух согласованных раундов исправлений после первоначальной реализации. Условные команды для них приведены в конце.

Принятые уточнения:

- По выбору пользователя сохраняется один лучший state по минимальному val_loss_mean на val_unfamiliar.
- В новом режиме один срок хранения полного комплекта: save_last_n_steps=1000. Веса, optimizer, scheduler, RNG, metadata и samples хранятся/удаляются вместе. Отдельный save_last_n_steps_state в новом конфиге не используется; его прежнее поведение сохраняется только в legacy-режиме.
- При resume стартовая валидация относится к восстановленному шагу s, первая новая обучающая loss — к s+1. Искусственный train-loss на s не создаётся.
- Сохраняются реальные имена кэшей, включая текстовый суффикс _qi_te.safetensors. Счётчик optimizer steps сохраняется явно, без предположения о его наличии в random_states_*.pkl.
- Значения val_every_n_steps=200, val_seed_noise=42, val_level_noise_n=10, val_seed_noise_n=1 — стартовый пример, а не оценка оптимальных настроек.

## Этап 1. Фиксированные данные и детерминированный шум

### 1.1. Спецификация

```
$speckit-specify SPECIFY_FEATURE_DIRECTORY=specs/002-val-loss-core

Создай спецификацию основы фичи «Детерминированный val-loss + разбивка по уровням шума» для существующего Qwen-Image original LoRA trainer. Следуй действующей Конституции без изменений. Документы Spec Kit — на английском, общение — на русском.

Этот этап реализует входы валидации и построение детерминированных проверок. Подключение к training loop и TensorBoard будет отдельным этапом. Не добавляй другие модели, виды валидации или универсальную систему метрик.

Требования:
1. Пользователь вручную готовит два непустых набора изображений с подписями: val_familiar содержит изображения из обучения; val_unfamiliar содержит изображения, не участвующие в обучении. Автоматически разбивать, пополнять, подменять наборы или переносить изображения между ними запрещено. Фактическое совпадение файлов можно проверять SHA-256; семантический поиск похожих изображений не входит в scope.

2. Предусмотри отдельный val-dataset.toml с явно обозначенными ролями val_familiar и val_unfamiliar. Роль не должна зависеть от порядка записей или имени папки. train-dataset.toml описывает только обучение. Validation batch_size=1 и num_repeats=1. Не наследуй training shuffle, повторы, caption dropout и случайные преобразования.

3. Зафиксированы состав наборов, подписи, разрешения/бакеты и используемые latent/text caches. Существующая детерминированная обработка при предварительном кэшировании допустима; во время обучения validation-вход больше не меняется. Запрещены случайные crop/flip/augmentation, изменение подписей и пересоздание кэшей во время валидации. Отсутствующий/повреждённый вход вызывает ошибку, а не молчаливый пропуск. Не исправляй и не дедуплицируй данные автоматически.

4. Постоянный image_id — SHA-256 байтов исходного изображения. Он не зависит от Python hash(), пути, имени, порядка загрузки, батча, rank или устройства. Одинаковое содержимое получает одинаковый ID в обоих наборах. Связь исходника, подписи и кэшей должна быть однозначной; коллизии имён не должны подменять данные.

5. Публичные параметры:
   val_dataset_config — путь к val-dataset.toml; отсутствие отключает валидацию;
   val_every_n_steps — целое >=1;
   val_seed_noise — общий целочисленный seed для обоих наборов;
   val_level_noise_n — N1, чётное целое >=2;
   val_seed_noise_n — N2, целое >=1, именно число реализаций шума на уровне.
   Не переименовывай эти поля. Boolean не считается integer. Проверяй эффективную конфигурацию после defaults → TOML → явный CLI. Неизвестные поля не игнорируются.

6. На каждое изображение приходится ровно N1*N2 проверок. Для i=1..N1:
   t_i = 0.05 + (i - 0.5) * 0.90 / N1.
   Это середины N1 равных интервалов [0.05,0.95].
   Low noise: 0.05 <= t < 0.5.
   High noise: 0.5 <= t <= 0.95.
   Каждая группа содержит N1/2 уровней. Для N1=2 ожидаются 0.275 и 0.725.

7. Для j=1..N2:
   seed = stable_hash(val_seed_noise, image_sha256, i, j).
   stable_hash основан на hashlib.sha256 с однозначной фиксированной сериализацией и преобразованием digest в допустимый seed. Не включай в seed роль набора, training step, rank, имя или порядок файла. Одинаковое изображение при одинаковых параметрах получает одинаковый шум на каждом вызове валидации, в том числе между наборами.

8. Используй отдельный генератор epsilon:
   noisy_latent = (1 - t_i) * latent + t_i * epsilon.
   t_i — конечный коэффициент смешивания. Интерфейсу тренера со шкалой 0–1000 передаётся timestep=1000*t_i. Повторный shift, округление к training timestep и случайный повторный выбор t запрещены. Training timestep_sampling и discrete_flow_shift эту сетку не меняют.

9. Подготовка validation dataset, loader и генерация шума не меняют Python/NumPy/PyTorch CPU/CUDA RNG обучения. Обрабатывай проверки последовательно, не материализуя шум для всего датасета и всех N1*N2 заранее. Вторую копию модели не загружай.

10. Ошибки параметров, пустые наборы, отсутствие обязательных подписей/кэшей и другие проблемы, обнаружимые без модели, сообщай до загрузки весов. Зафиксируй сведения, позволяющие обнаруживать изменение validation-входа при resume. Повторное чтение во время обучения не должно незаметно принять изменённые файлы; обнаруженное изменение требует ошибки, а не автоматического обновления набора. Перенос неизменного эксперимента в другую папку не должен менять ID, seeds или идентичность протокола.

Приёмка: CPU-проверки сетки, SHA-256 seeds, индексов 1-based, количества проверок, переименования/перестановки входов, неизменности RNG, пустых наборов, ошибочных типов/значений, пропавших подписей/кэшей и неправильных ролей. Результат — пригодный для подключения компонент; незавершённый validation loop пока не включать.
```

### 1.2. План

```
$speckit-plan SPECIFY_FEATURE_DIRECTORY=specs/002-val-loss-core

Спланируй минимальные изменения в текущем Python/PyTorch-коде. Проверь dataset/config_utils.py, image_video_dataset.py, datasources.py, cache_io.py, bucket.py и обе qwen_image_cache_* команды в src/musubi_tuner. Используй существующие readers и форматы, не переписывай training dataset.

В contracts зафиксируй конкретную схему val-dataset.toml, обязательность/defaults новых полей, правила отключения, связь image→caption→latent→text-cache и диагностику. Один val-dataset.toml впоследствии должен приниматься cache-командами и тренером: нельзя добавить role только в пример TOML, не поддержав его readers.

Определи сериализацию stable_hash, диапазон seed, индексы 1-based, dtype/device генератора и устойчивый обход данных. Раздели повторяемость входов/шума и численную повторяемость вычислений в фиксированной среде; не обещай побитовую идентичность разных GPU/backend и не меняй глобальные training settings.

Учти: нынешний training reader собирает записи по cache files, пропускает отсутствующий text cache и применяет repeats/shuffle. Validation должен учитывать все заявленные элементы ровно один раз. Предусмотри минимальную фиксацию входов и проверку их неизменности, без базы данных и фонового сервиса.

Проверки: временные изображения и safetensors, существующий pytest, изоляция RNG включая создание loader, совместимость прежних train/cache configs. Включи README. Создай research.md, plan.md, data-model.md, contracts и quickstart.md. Не запускай обучение и не скачивай веса.
```

### 1.3. Задачи

```
$speckit-tasks SPECIFY_FEATURE_DIRECTORY=specs/002-val-loss-core

Создай зависимые задачи: схема/валидация конфигов; связь исходников и кэшей; фиксированный набор и fingerprint; сетка t и stable_hash; отдельный RNG; содержательные CPU-проверки; совместимость и README. Для каждой задачи укажи требование, файлы и проверяемый результат. Training loop, TensorBoard и best states относятся к следующим этапам. Заглушки не считать реализацией.
```

### 1.4. Анализ

```
$speckit-analyze SPECIFY_FEATURE_DIRECTORY=specs/002-val-loss-core

Проверь spec/plan/tasks без правок. Особое внимание: фиксированный вход, ID по содержимому, отсутствие роли/step/rank в seed, индексы 1-based, точный t без shift, batch=1/repeats=1, отсутствие пропусков, RNG и согласованность readers. Покажи подтверждённые противоречия и непокрытые требования с конкретными исправлениями. Не запускай implement автоматически.
```

### 1.5. Реализация

```
$speckit-implement SPECIFY_FEATURE_DIRECTORY=specs/002-val-loss-core

Реализуй все задачи этапа в рабочем коде и проведи согласованные локальные проверки. Сохрани поведение прежних train/cache configs. Не запускай полную модель или обучение. Обнови README, не объявляя ещё не подключённый validation loop готовым. Отчитайся о проверках и оставшихся несоответствиях.
```

### 1.6. Проверка реализации

```
$speckit-converge SPECIFY_FEATURE_DIRECTORY=specs/002-val-loss-core

Сопоставь фактический код со spec/plan/tasks. Проверь численные и конфигурационные требования по реализации и результатам проверок. Не меняй код/spec/plan. Добавляй только подтверждённые задачи Convergence и представь список конкретных исправлений для отдельного согласования. Не запускай implement автоматически; при нуле расхождений не добавляй пустую фазу.
```

## Этап 2. Метрики, расписание, TensorBoard и resume

### 2.1. Спецификация

```
$speckit-specify SPECIFY_FEATURE_DIRECTORY=specs/003-val-loss-training

Создай спецификацию подключения детерминированной валидации к Qwen-Image original LoRA trainer. Используй реализованные контракты specs/002-val-loss-core без переопределения входов, сетки и seeds. Конституцию не изменяй. Документы — на английском, общение — на русском.

1. val_dataset_config включает validation. Без него прежние loss, данные, noise/timestep sampling, precision, gradients, optimizer, scheduler, RNG, сохранения и resume сохраняют прежнее поведение.

2. Validation event полностью оценивает оба набора: batch=1, каждое изображение на всех N1*N2 проверках. Используются текущие LoRA-веса, тот же forward/target и та же математическая loss с теми же weighting/reduction, что при training. Для текущего Qwen flow matching target=epsilon-latent. Не создавай независимую «упрощённую MSE», которая сможет разойтись с обучением.

3. Пусть L(x,i,j) — scalar loss отдельной проверки. Сначала усредни N1*N2 значений одного изображения, затем усредни по изображениям набора. Каждое изображение имеет одинаковый вес независимо от разрешения, числа пикселей, бакета и training repeats. Low/high усредняются тем же способом по соответствующей половине уровней. Нельзя давать одинаковый вес бакетам разного размера или усреднять все пиксели набора сразу. Контрольное равенство: loss_mean=(loss_low_noise+loss_high_noise)/2 с учётом численной погрешности.

4. Добавь через существующий logging backend шесть точных scalar tags, сохранив прежние training loss graphs:
   train_eval_loss_mean — val_familiar, все уровни;
   train_eval_loss_low_noise — val_familiar, t<0.5;
   train_eval_loss_high_noise — val_familiar, t>=0.5;
   val_loss_mean — val_unfamiliar, все уровни;
   val_loss_low_noise — val_unfamiliar, t<0.5;
   val_loss_high_noise — val_unfamiliar, t>=0.5.

5. X — число завершённых optimizer updates. Свежий запуск: validation на 0 до первого update; далее после каждого завершённого update s, кратного val_every_n_steps; обязательно после последнего шага. Совпавшие periodic/final события выполняются один раз. Следующий training step начинается после validation. Промежуточные microbatches gradient accumulation не запускают validation; пропущенный update не считается завершённым.

6. Validation работает без autograd/backward, обновления model/optimizer/scheduler и изменения существующих gradients. Выключается dropout модели и LoRA. После успеха или исключения восстанавливаются все прежние train/eval flags и RNG. Нельзя безусловно переводить все модули в train(), если часть ранее была в eval(). Не используй optimizer eval hooks, которые меняют веса/optimizer state.

7. Используй точный timestep=1000*t и sigma=t. Для loss weighting сохраняется training-формула, вычисленная на фактическом validation sigma. Повторный shift и незаметный выбор ближайшего sigma из training schedule запрещены. Сам training path и его precision/weighting не меняются.

8. Сохрани Accelerate workflow для одного и двух ranks/GPU: все элементы учитываются ровно один раз, distributed padding не искажает среднее, tags пишутся однократно. Ranks согласованно проходят validation boundary без collective deadlock. Второй экземпляр модели не создаётся.

9. Сохраняй в training state явный абсолютный счётчик завершённых optimizer updates и metadata validation-протокола. На resume из шага s: стартовая validation на s до следующего update; первый новый train-loss на s+1; дальнейшее расписание использует абсолютные шаги. Не сбрасывай графики в 0 и не выдумывай train-loss на s. Один validation event на шаг в пределах запуска.

10. Optimizer/scheduler/RNG восстанавливаются штатно. Не отождествляй автоматически accelerator.step, microbatch count или счётчик отдельного параметра с global optimizer step. Сохрани нынешнюю семантику max_train_steps как бюджета шагов данного запуска; абсолютный счётчик графиков отдели от этого бюджета. LR/warmup не сбрасывать. Точное восстановление позиции training dataloader в scope не входит.

11. При resume проверяй неизменность validation-входов и протокола. Если старый state не содержит достоверного шага, включённая новая validation должна дать понятную ошибку, а не step=0 или догадку. Legacy resume при выключенной фиче сохранить. network_weights без optimizer resume — новый запуск со шкалой от 0.

12. NaN/Inf, ошибка чтения и неполный event не публикуются как корректная validation. Не подменять их нулём и не использовать для выбора best.

Приёмка: содержательные CPU-тесты loss/weighting и усреднений; включение validation не меняет training RNG и следующий controlled update; проверены modes/dropout/gradients/optimizer/scheduler, исключения, accumulation, 0/periodic/final, шесть tags, ненулевой step после реального save/load. Новая output-структура и best states — следующий этап.
```

### 2.2. План

```
$speckit-plan SPECIFY_FEATURE_DIRECTORY=specs/003-val-loss-training

Проверь QwenImageNetworkTrainer.call_dit, trainer_base.compute_loss, _run_training_loop, _register_hooks_and_resume, training/timesteps.py и logging setup. Используй существующий trainer, не дублируй training loop.

Учти конкретные особенности кода: call_dit делит timestep на 1000 и при gradient_checkpointing выставляет requires_grad; get_sigmas может округлять timestep по training schedule; global_step сейчас обнуляется после загрузки Accelerate state. План должен решить эти проблемы для новой фичи без изменения прежнего training path.

Определи минимальное переиспользование общей loss, передачу точного sigma, restoration через try/finally и устойчивое суммирование. Выбери простой корректный Accelerate workflow: например, main-rank evaluation существующей модели при ожидании остальных ranks, если одиночный forward не содержит collectives. Проверь это по используемым вызовам.

Зафиксируй step/event/resume contracts и формат явных metadata. Отдели абсолютную ось от бюджета steps текущего запуска. Сохрани старый режим, когда validation выключена.

План проверок: малые CPU-модели; weighting=none и существующие weighted схемы; неравные размеры изображений/бакетов; accumulation; RNG/modes при исключении; реальные TensorBoard tags/steps; небольшой Accelerate round-trip. GPU/Qwen-проверки описать отдельно без запуска. Обновить README.
```

### 2.3. Задачи

```
$speckit-tasks SPECIFY_FEATURE_DIRECTORY=specs/003-val-loss-training

Создай задачи: config/core integration; общий loss/точный sigma; безопасный evaluation context; агрегация; optimizer-boundary scheduling; logging; явный step и protocol metadata в save/load; Accelerate coordination; CPU-регрессии и README. Явно покрой 0/periodic/final без дублей, accumulation, неравные бакеты, resume и unchanged training при выключении. Новую output-структуру и best checkpoint пока не реализовывать.
```

### 2.4. Анализ

```
$speckit-analyze SPECIFY_FEATURE_DIRECTORY=specs/003-val-loss-training

Проверь spec/plan/tasks и совместимость с specs/002-val-loss-core. Ищи изменение training algorithm, повторный shift/округление sigma, неверное усреднение, microbatch вместо optimizer step, повтор final event, сброс step при resume, нарушение RNG/dropout и collective deadlocks. Проверь наличие реальных logging/state round-trip задач. Только анализ без правок.
```

### 2.5. Реализация

```
$speckit-implement SPECIFY_FEATURE_DIRECTORY=specs/003-val-loss-training

Реализуй все задачи в действующем тренере. Выполни согласованные CPU-проверки и затронутые существующие test_qwen_image_training_invariants/config/cache tests. Один успешный forward не доказывает невлияние на обучение. Обнови README с точным смыслом шагов/resume. Не запускай обучение/GPU/скачивание весов. Результат — работающие шесть метрик в training loop, а не только вспомогательные функции.
```

### 2.6. Проверка реализации

```
$speckit-converge SPECIFY_FEATURE_DIRECTORY=specs/003-val-loss-training

Проверь требования этапа и совместимость с core. Особое внимание: optimizer update → validation → следующий training step, restoration при исключении, точный sigma, логирование и step после реального save/load. Добавь только подтверждённые задачи Convergence, покажи исправления для отдельного согласования. Не меняй spec/plan/код и не запускай implement автоматически.
```

## Этап 3. Папка эксперимента и полные training states

### 3.1. Спецификация

```
$speckit-specify SPECIFY_FEATURE_DIRECTORY=specs/004-experiment-training-states

Создай спецификацию единой папки эксперимента и сохранений Qwen-Image original LoRA с валидацией из specs/002-val-loss-core и specs/003-val-loss-training. Конституцию не изменяй. Документы — на английском, общение — на русском. Новый режим включается явно; прежние configs без него сохраняют прежнее поведение.

1. Пользователь задаёт корень эксперимента через experiment_dir; имя папки не зашито в код. В новом режиме относительные пути datasets, caches, configs, prompts, output, logging и resume разрешаются от этого корня; абсолютные пути сохраняются. Относительное значение самого experiment_dir разрешается от папки выбранного train.toml. Это позволяет переносить папку и запускать из другого cwd. Без experiment_dir сохраняются прежние правила путей. Сохрани defaults → TOML → явный CLI; конфликтующие назначения новой иерархии объясняй, а не игнорируй.

2. Структура нового режима:
   <experiment_dir>/train.toml
   <experiment_dir>/train-dataset.toml
   <experiment_dir>/val-dataset.toml
   <experiment_dir>/sample_prompts.txt
   <experiment_dir>/dataset/train/
   <experiment_dir>/dataset/val_familiar/
   <experiment_dir>/dataset/val_unfamiliar/
   <experiment_dir>/cache/train/
   <experiment_dir>/cache/val_familiar/
   <experiment_dir>/cache/val_unfamiliar/
   <experiment_dir>/output/tensorboard/
   <experiment_dir>/output/current_training_states/
   <experiment_dir>/output/val_training_states/val-loss/
   В dataset — изображения и .txt-подписи. В cache — существующие Qwen latent/text embeddings. Формат и имена, включая _qi_te.safetensors, не менять ради дерева папок.

3. Тренер и обе cache-команды должны согласованно понимать experiment_dir и train/val configs. Не предполагай, что нынешние cache entrypoints уже поддерживают train.toml/--config_file: реализуй минимальный конкретный интерфейс и документируй его. Один val-dataset.toml позволяет кэшировать оба val-набора, не добавляя val_unfamiliar в обучение.

4. Единица сохранения — папка полного training state конкретного абсолютного optimizer step. Имя содержит output_name и step. Состав:
   model.safetensors — один файл LoRA, пригодный для использования адаптера и восстановления training model;
   optimizer.bin и scheduler.bin либо реальные эквиваленты текущей версии Accelerate;
   random_states_<rank>.pkl для каждого rank;
   явная metadata завершённого step, validation-протокола и необходимых сведений о метриках/best;
   samples/ — PNG тех же весов, если семплирование включено.
   Не сохраняй замороженный DiT и не создавай второй файл тех же LoRA-весов внутри Accelerate state. Не переименовывай служебные файлы произвольно, ломая load_state. Совместимость единственного model.safetensors с export/resume и точностью обучения должна быть проверена по коду, а не предположена.

5. Периодическое сохранение определяется save_every_n_steps; финальный state обязателен после последнего шага. Совпавшие periodic/epoch/final/new-best причины на одном шаге объединяются: checkpoint не записывается повторно, одинаковые samples не генерируются дважды. На один абсолютный optimizer step приходится не более одного полного пакета и одного комплекта samples. Отдельные LoRA-веса вне папки state, второй файл тех же весов внутри неё, дублирующий *-state, отдельный финальный дубль и копии samples в общем output/sample запрещены в новом режиме.

6. В output/val_training_states/val-loss/ хранится один лучший полный state по минимальному val_loss_mean на val_unfamiliar. Улучшение строгое (<), при равенстве остаётся предыдущий best. Первый корректный результат, включая step 0, становится best. NaN/Inf и неполный event не допускаются. Новый best сохраняется сразу, даже вне periodic save. Остальные пять метрик диагностические и не создают собственных best states.

7. Один пакет не должен физически дублироваться в current и best. Предпочтительное минимальное решение: лучшая папка находится только в val-loss, остальные — в current_training_states. При смене best предыдущая папка возвращается в current, если ещё попадает в окно хранения; иначе удаляется после успешного сохранения нового best. Перемещение не перегенерирует samples. Не строй content-addressed storage или сложную систему ссылок.

8. Единственный срок хранения в новом режиме задаёт save_last_n_steps; он применяется ко всей папке training state, включая веса, optimizer, scheduler, RNG, metadata и samples. В поставляемом конфиге save_last_n_steps=1000. Срок измеряется прошедшими optimizer steps, не числом файлов; граница включительна: сохраняется пакет шага s при s >= current_step-save_last_n_steps. Неуказанный лимит означает хранение всех пакетов. Текущий best защищён от очистки независимо от возраста. Отдельного срока для optimizer state нет: save_last_n_steps_state остаётся только legacy-параметром. Если пользователь явно укажет его вместе с experiment_dir, сообщи до загрузки модели, что его нужно удалить и использовать единый save_last_n_steps; не запускай две независимые очистки. Прежние конфиги без experiment_dir сохраняют прежнее поведение. Нельзя удалять отдельные части остающегося пакета.

9. При включённых samples каждый сохраняемый полный пакет должен содержать семплы своих весов, включая best между periodic saves. Совпавшие sample triggers объединяются по шагу. Для sample-only шагов и sample_at_first=false при best на step 0 согласуй однозначное поведение до реализации: нельзя молча терять заданные события, создавать скрытые checkpoints или нарушать иерархию. При выключенных samples генерация не выполняется; пустая samples/ допустима. Семплирование сохраняет training RNG и режимы.

10. Пакет доступен для resume и заменяет best только после успешной записи обязательных файлов. Ошибка записи не удаляет предыдущий корректный best. Retention не затрагивает datasets, caches, конфиги и чужие файлы; удаляется только принадлежащий этому эксперименту пакет.

11. Все ranks участвуют в сохранении необходимых rank-local RNG states; main rank координирует общие файлы. Для двух ranks нужны оба random_states_*.pkl. Resume из current и best восстанавливает model/optimizer/scheduler/RNG/absolute step. Продолжай сравнение с ранее сохранённым best, не сбрасывай его в infinity. При загрузке другого шага не приписывай новым весам метрики старого best.

12. NN-search/SSCD, duplicates, CSD и VQAScore — будущая работа. Не реализовывай их, не добавляй зависимости, параметры или пустые подсистемы.

Приёмка: пути при другом cwd/имени папки; один save и один комплект samples при совпадении periodic/final/new-best; единственный файл весов внутри единственного пакета данного шага, без копий в других каталогах; перемещение current↔best без копирования; один best, ties и ошибка его записи; включительная граница единого retention с удалением пакета целиком; защита best; реальный малый Accelerate save/load из current и best; legacy output без experiment_dir остаётся рабочим.
```

### 3.2. Уточнение редких сочетаний настроек

```
$speckit-clarify SPECIFY_FEATURE_DIRECTORY=specs/004-experiment-training-states

Уточни только оставшиеся неоднозначности семплирования и внеси ответы в spec.md до plan. Уже решено: один best по минимуму val_loss_mean на val_unfamiliar и один срок save_last_n_steps для всей папки state. Отдельного срока для optimizer нет; повторно эти вопросы не задавай.

Что делать на шаге, когда пора генерировать samples, но не пора сохранять полный state? Как сочетается обязательный комплект samples с sample_at_first=false и первым best на step 0? Предложи минимальные варианты и объясни, меняют ли они количество полных checkpoints. Любой вариант должен сохранять запрет повторных пакетов и повторной генерации одних samples на одном шаге.

Не расширяй scope. Вопросы о внутренних классах и библиотеках решай по коду. Если решение уже явно принято в доступной формальной спецификации, используй его и не спрашивай повторно.
```

### 3.3. План

```
$speckit-plan SPECIFY_FEATURE_DIRECTORY=specs/004-experiment-training-states

Спланируй минимальные изменения parser/config readers, cache entrypoints, trainer_base, utils/train_utils.py, sample_images и Accelerate save/load hooks. Второй training framework не нужен.

Пропиши разрешение каждого path-valued параметра в TOML/CLI, включая model paths, dataset_config, val_dataset_config, image_directory, cache_directory и resume. Standalone cache-команды должны получать тот же корень; не меняй legacy cwd semantics. Перенос папки не меняет validation fingerprint.

Определи порядок validation/sample/save на одном шаге так, чтобы model, optimizer, метрики и PNG принадлежали одним весам. Зафиксируй ответы clarify и sample-only поведение. Для retention используй только save_last_n_steps и удаляй пакет целиком; legacy save_last_n_steps_state не должен создавать второй срок или вторую очистку в новом режиме.

Проверь фактическую совместимость network.save_weights и Accelerate network.state_dict, dtype trainable weights и save_precision. Единственный model.safetensors должен реально загружаться как LoRA и при resume. Нельзя отбросить float32 master weights ради bf16 export и объявить точный resume. Если требования одного файла и точности нельзя совместить текущим форматом, до реализации представь конкретный конфликт и минимальные варианты для решения пользователя; не ослабляй требования самостоятельно.

Проверь, какие save/load вызовы нужны на каждом rank для RNG. Определи минимальную запись во временную папку и завершение записи до смены best, без универсального transaction framework. Зафиксируй metadata best score/step/protocol и поведение resume старого current state при существующем более позднем best.

Проверки: реальные малые Accelerate save/load, точность восстановления, временные папки, счётчики save/sample, retention boundary, ошибка записи и legacy regressions. Включи README/contracts/quickstart. Полную модель/GPU не запускать.
```

### 3.4. Задачи

```
$speckit-tasks SPECIFY_FEATURE_DIRECTORY=specs/004-experiment-training-states

Создай задачи по согласованному plan: opt-in/path resolution; cache CLI; единый state и resume hooks; координация periodic/epoch/final/best/sample; замена/защита best; retention полных пакетов; RNG всех ranks; реальные CPU round-trip и проверки файлов; legacy regressions; README. Отдельно проверь precision единственного model.safetensors. Одних конфигов вместо реализации нового режима недостаточно.
```

### 3.5. Анализ

```
$speckit-analyze SPECIFY_FEATURE_DIRECTORY=specs/004-experiment-training-states

Проверь spec/plan/tasks и контракты предыдущих этапов. Ищи несовместимые cwd bases, несуществующие cache CLI flags, duplicate adapter/state/final/best, потерю precision при resume, отсутствие RNG второго rank, необработанные sample-only шаги, частичные states, удаление best, сброс step/best. Ответы clarify должны быть отражены во всех документах. Только анализ без правок.
```

### 3.6. Реализация

```
$speckit-implement SPECIFY_FEATURE_DIRECTORY=specs/004-experiment-training-states

Реализуй все согласованные задачи experiment_dir и полных states. Проверь реальные save/load малой CPU-модели, точность, единичность save/sample, retention и сохранность best при ошибке. Проверка существования файлов или mock save/load недостаточна. Выполни затронутые legacy tests и обнови README. Обучение/GPU/скачивание/перенос на сервер не запускать.
```

### 3.7. Проверка реализации

```
$speckit-converge SPECIFY_FEATURE_DIRECTORY=specs/004-experiment-training-states

Сверь код с experiment_dir/state/best/samples/retention/resume требованиями. Проверь реальный формат одного файла LoRA и работающий load_state, не только дерево папок. Добавь подтверждённые задачи Convergence и представь конкретные исправления для отдельного согласования. Код/spec/plan не менять; implement автоматически не запускать.
```

## Этап 4. Конфиги и сквозная проверка

### 4.1. Спецификация

```
$speckit-specify SPECIFY_FEATURE_DIRECTORY=specs/005-val-loss-configs

Создай спецификацию завершения фич из specs/002-val-loss-core, specs/003-val-loss-training и specs/004-experiment-training-states. Результат включает физически созданные конфиги, принимаемые изменённым инструментом, и точную инструкцию. Одного Markdown-описания недостаточно. Конституцию не изменяй. Spec Kit документы — на английском, общение — на русском.

Создай отдельный пример qwen_image_lora_val_example/ с train.toml, train-dataset.toml, val-dataset.toml, sample_prompts.txt и согласованной иерархией dataset/cache/output. Имя примера не зашивать в код. Не перезаписывай пользовательские configs/prompts и не заполняй datasets выдуманными обучающими картинками.

train.toml должен содержать:
model_version = "original"
dit/vae/text_encoder — явно помеченные заменяемые абсолютные пути сервера; исходный DiT BF16, не готовый FP8
experiment_dir = "."
dataset_config = "train-dataset.toml"
val_dataset_config = "val-dataset.toml"
network_module = "networks.lora_qwen_image"
network_dim = 16
network_alpha = 16
mixed_precision = "bf16"
fp8_base = false
fp8_scaled = false
fp8_vl = false
blocks_to_swap = 0
sdpa = true
gradient_checkpointing = true
max_train_steps = 1600
gradient_accumulation_steps = 1
seed = 42
optimizer_type = "adamw8bit"
learning_rate = 5e-5
lr_scheduler = "constant_with_warmup"
lr_warmup_steps = 200
max_grad_norm = 1.0
timestep_sampling = "shift"
discrete_flow_shift = 2.2
weighting_scheme = "none"
max_data_loader_n_workers = 0
persistent_data_loader_workers = false
val_every_n_steps = 200
val_seed_noise = 42
val_level_noise_n = 10
val_seed_noise_n = 1
output_name = "qwen_image_lora"
save_precision = "bf16"
save_every_n_steps = 200
save_last_n_steps = 1000
save_state = true
log_with = "tensorboard"
logging_dir = "output/tensorboard"
sample_prompts = "sample_prompts.txt"
sample_every_n_steps = 200
sample_at_first = false

lr_warmup_steps должен оставаться integer. Shift относится к training, не validation. save_last_n_steps=1000 управляет всей папкой state; save_last_n_steps_state в новый конфиг не добавлять. output_dir и остальные необходимые поля задать строго по реализованному контракту; output расположен внутри эксперимента. Не добавляй несуществующие ключи. Если совместимость единственного state-файла потребовала ранее согласованного изменения save_precision, примени это решение и явно объясни его в конфиге/инструкции; не допускай скрытой потери precision resume.

train-dataset.toml: resolution=[1024,1024], enable_bucket=true, bucket_no_upscale=true, caption_extension=".txt", batch_size=1, num_repeats=1; один пример с image_directory="dataset/train", cache_directory="cache/train".

val-dataset.toml: реальная поддерживаемая схема с двумя явными ролями. val_familiar использует dataset/val_familiar и cache/val_familiar; val_unfamiliar — dataset/val_unfamiliar и cache/val_unfamiliar. Те же resolution/buckets, batch_size=1, num_repeats=1, без случайных преобразований. val_unfamiliar не попадает в train config.

sample_prompts.txt: минимум две корректные строки текущего parser с явным заменяемым триггером TOK:
TOK, a gold coin on a plain background. --w 1024 --h 1024 --d 42 --s 30 --l 4.0 --fs 2.2
TOK, a copper teapot on a table. --w 1024 --h 1024 --d 42 --s 30 --l 4.0 --fs 2.2
Прокомментируй замену TOK и отключение samples по реализованному контракту.

Инструкция: точные команды latent/text caching для train и обоих val-наборов, accelerate launch, resume из current и best, TensorBoard. Флаги должны реально приниматься entrypoints. Покажи запуск из другого cwd и перенос/переименование корня. Пользователь подготавливает данные и кэши заранее, после чего validation-входы фиксированы.

Ожидаемые validation steps свежего запуска: 0,200,400,...,1600, без повторной проверки на 1600; шесть точных tags; один best по val_loss_mean; полные states; включительная граница retention. Объясни абсолютный offset при resume и существующий max_train_steps как бюджет данного запуска. Не обещай восстановление позиции dataloader, если оно не реализовано.

Приёмка: реальные parsers читают все файлы; paths/flags корректны; источники/cache разделены; CPU integration проверяет config→fixed data→evaluation→metrics→save→resume. Тесты используют временные малые fixtures вместо внешних весов/данных; боевые конфиги остаются честными шаблонами с явно заменяемыми путями. Полный H200 run описывается как отдельная серверная проверка без локального запуска.
```

### 4.2. План

```
$speckit-plan SPECIFY_FEATURE_DIRECTORY=specs/005-val-loss-configs

Спланируй создание четырёх файлов примера и согласование README/quickstart с фактическими контрактами этапов 1–3. Проверь публичные parsers и path resolution. Подготовь краткую таблицу «поле/файл → reader → значение/путь» для проверки.

Минимальная CPU integration: реальные readers, temporary images/cache, малая модель, два val-набора разного размера, точная сетка, шесть tags, periodic/final dedup, один best, полный state и ненулевой resume step. Используй существующую test infrastructure, не создавай новый framework.

Отдельно проверь cache CLI для experiment_dir и val-dataset.toml: набор флагов может отличаться от trainer. Инструкцию запуска составь из реально поддержанных entrypoints. Укажи необходимые замены путей/TOK и заполнение datasets; пустые папки не считать готовыми данными.

Проверь max_train_steps/resume, precision state и согласованную политику sample_at_first/best step 0. Противоречия утверждённым требованиям сообщай явно, не скрывай их удобным примером. H200-производительность не измерять.
```

### 4.3. Задачи

```
$speckit-tasks SPECIFY_FEATURE_DIRECTORY=specs/005-val-loss-configs

Создай задачи: четыре физических файла; необходимые каталоги; реальные parser/path/CLI checks; CPU integration цепочки; README и серверный quickstart; итоговая матрица требований и выполненных проверок. Явно проверь отсутствие val_unfamiliar в training и duplicate state/sample outputs. Не заканчивай этап одной генерацией Markdown.
```

### 4.4. Анализ

```
$speckit-analyze SPECIFY_FEATURE_DIRECTORY=specs/005-val-loss-configs

Проверь spec/plan/tasks и контракты предыдущих этапов. Все параметры, роли и команды должны приниматься реализованными readers. Проверь paths, BF16/AdamW8bit, integer warmup, validation независимо от shift, единственный best, samples и resume. Только анализ; дефект реализации нельзя скрыть ослабленным конфигом.
```

### 4.5. Реализация

```
$speckit-implement SPECIFY_FEATURE_DIRECTORY=specs/005-val-loss-configs

Создай готовые файлы примера. Проверь их реальными readers, CLI/path checks и согласованной CPU integration. Обнови README/quickstart по фактическому поведению. Не изменяй пользовательские configs/prompts и не запускай обучение, GPU, скачивание или перенос на сервер. Перечисли созданные файлы, необходимые пользовательские замены и выполненные проверки; отсутствие серверного запуска отметь один раз.
```

### 4.6. Финальная проверка

```
$speckit-converge SPECIFY_FEATURE_DIRECTORY=specs/005-val-loss-configs

Проверь четыре итоговых файла и всю согласованную цепочку этапов 1–4: fixed inputs, SHA-256 seeds, N1*N2, точный t без shift, общая loss, mean/low/high, шесть tags, 0/periodic/final, невлияние на training, resume step, единая папка, полный единственный best, отсутствие дублей, retention и реальные команды.

Для подтверждённых расхождений укажи требование, доказательство и конкретное исправление; добавь соответствующие задачи Convergence. Код/spec/plan не менять. Исправления согласуются отдельным списком по Конституции. При отсутствии расхождений сообщи завершение локального этапа; успешного GPU/H200 запуска не заявляй.
```

## Этап 5. Проверки на RunPod

Этот этап добавляет тестирование реальной модели после локальной реализации. В первой implement-команде готовятся сервер и средства проверки, во второй выполняется конечная матрица GPU-тестов.

Предоставленная SSH-команда использует Basic SSH. Он не поддерживает SCP/SFTP; клонирование Git можно выполнить внутри удалённого терминала. Прямой SSH over exposed TCP потребуется, если возможностей Basic SSH окажется недостаточно для надёжного управления тестами или получения артефактов. [Документация RunPod](https://docs.runpod.io/pods/configuration/use-ssh).

### 5.1. Спецификация серверной проверки

```
$speckit-specify SPECIFY_FEATURE_DIRECTORY=specs/006-runpod-verification

Создай спецификацию отдельного серверного этапа проверки реализованных фич specs/002-val-loss-core, specs/003-val-loss-training, specs/004-experiment-training-states и specs/005-val-loss-configs. Этапы 1–4 должны быть завершены локально. Конституцию не изменяй: её ограничения Local Stage продолжают действовать для локального этапа; данный этап — отдельная remote operational verification.

Цель — подтвердить корректность инструмента на реальной Qwen-Image original и GPU. Обновления LoRA разрешены только как часть конечных тестовых сценариев. Не запускай обычное обучение на 1600 шагов, не подбирай learning rate/rank/seeds, не обучай до снижения loss или улучшения картинок. Отсутствие лимита времени не разрешает бесконечные повторы. Pod останавливает пользователь; агент не останавливает, не удаляет, не перезапускает и не заменяет Pod.

Доступ:
ssh yxd2b6cm3e8gfa-64412491@ssh.runpod.io -i ~/.ssh/id_ed25519
На текущем Windows-компьютере путь к этому ключу разрешается как C:\Users\inbox\.ssh\id_ed25519. Используй ключ локально, не выводи его содержимое и не копируй его на сервер. Модели и пользовательские датасеты уже находятся в /workspace. Репозиторий: https://github.com/Maxim-Dey/musubi-tuner-tools-extraction.

Подготовка:
1. Проверь SSH, доступность /workspace, GPU/VRAM, driver, Python, CUDA/PyTorch, свободное место и существующие модели/данные. Найди реальные пути и проверь совместимость форматов. Не объявляй подключение или наличие зависимостей проверенными без результата команды. Если выбранный Pod недоступен, запроси актуальные реквизиты; новый Pod не создавай.
2. Выдели отдельный каталог /workspace/musubi-val-loss-tests/<run_id>/ для кода, окружения, тестовых конфигов, новых кэшей и результатов. Не изменяй исходные модели, пользовательские datasets и прежние результаты. Повторный запуск не должен перезаписывать доказательства предыдущего.
3. Разверни именно код с завершённой реализацией. Если нужный commit доступен на remote, clone/fetch и checkout конкретного commit. Если необходимые изменения только локальны, используй проверяемый перенос исходников/patch доступным способом, зафиксировав base commit и hash перенесённых изменений. Не тестируй молча старую ветку GitHub и не публикуй изменения/credentials ради развёртывания. При недоступности надёжного переноса запроси подходящее SSH-подключение, а не запускай неверную версию.
4. Создай отдельное окружение согласно pyproject.toml и существующему lock, если он есть. Выбери совместимые Python, torch/torchvision/CUDA, установи необходимые зависимости тренера, AdamW8bit и TensorBoard. Не обновляй глобальное окружение пользователя и не устанавливай альтернативные attention/quantization режимы без тестовой необходимости. Запиши фактические версии.
5. Используй существующие роли train/val_familiar/val_unfamiliar. Для ограниченных тестовых fixtures допустимы явно перечисленные маленькие поднаборы этих уже назначенных пользователем ролей: целевой пример — 3 train, 2 familiar, 3 unfamiliar. Все familiar fixtures входят в train fixtures, unfamiliar с ними не пересекаются по содержимому. Это отдельный тестовый эксперимент; рабочие val-наборы не меняются. Состав fixtures и SHA-256 фиксируются до прогонов. Если роли исходных данных неоднозначны, задай один конкретный вопрос; не придумывай train/val split. Не дополняй недостающие данные скрыто или синтетическими изображениями для видимости успешного real-model теста.
6. Подготовь необходимые latent/text caches штатными изменёнными cache-командами. Используй существующие корректные кэши только после проверки соответствия fixtures/настройкам; иначе создай отдельные тестовые. После подготовки validation-входы фиксированы. Модели из /workspace повторно не скачивай.

Тестовые конфиги отделены от рабочего примера на 1600 шагов. Сохрани Qwen original, LoRA rank/alpha=16, BF16, SDPA, AdamW8bit, learning_rate=5e-5, gradient_checkpointing и отсутствие FP8/block swap. Целевое training resolution=1024 с существующими bucket-правилами. Для коротких прогонов warmup=2 optimizer steps, val_seed_noise=42, N1=4, N2=2. Отдельная validation-only проверка покрывает пример N1=10,N2=1. Все отличия от рабочего конфига перечисли. Samples проверяются на одном фиксированном prompt с фиксированным seed и четырьмя inference steps; качество изображения не оценивается.

Конечная матрица:

G01 — Основной полный запуск.
12 завершённых optimizer updates, gradient_accumulation_steps=2, val_every_n_steps=4, save_every_n_steps=4, sample_every_n_steps=4, save_last_n_steps=8. Включены samples и реальные state saves. Ожидаемые validation steps: 0,4,8,12. На шаге 12 periodic/final/new-best, если он произошёл, не создают повторных записей. Проверить реальную работу GPU, конечные loss/gradients и изменение LoRA после ненулевых updates. Нулевой первый LR warmup не трактовать автоматически как дефект. Счётчик не должен увеличиваться на каждом microbatch.

G02 — Финал вне периодического интервала.
10 optimizer updates, accumulation=1, val_every_n_steps=4, save_every_n_steps=8, sample_every_n_steps=8, retention=8. Ожидаемые validation steps: 0,4,8,10; полный финальный state относится к 10. Проверить, что validation расписание не зависит от save расписания, и применить уже согласованную политику samples/best вне periodic save.

G03 — Детерминированность без обучения.
На одном неизменном checkpoint и фиксированных входах выполнить две validation подряд и одну после загрузки того же checkpoint в новом процессе. Дополнительных optimizer updates нет. Image IDs, t, seeds и шумы должны совпадать точно при фиксированном backend/device. Метрики сравниваются с заранее заданными и обоснованными допусками для данной precision, а не с допуском, подогнанным после результата. Перестановка перечисления/переименование копий fixtures не меняют протокол и результаты. Отдельно проверить N1=10,N2=1 на тех же весах без обучения: это новый validation-only эксперимент с загрузкой весов, а не resume прежнего протокола N1=4,N2=2. Его метрики не добавляются в прежнюю историю и не сравниваются с прежним best; проверку несовместимого протокола при настоящем resume не обходить. Не обещать битовую одинаковость разных GPU/backend.

G04 — Независимая проверка формул.
Для малой выборки реальных model forwards сохранить достаточные scalar/debug evidence, чтобы независимо проверить t_i, смешивание, timestep=1000*t, target, элементарную loss, per-image reduction и средние по каждому набору/low/high. Expected values не должны вычисляться той же проверяемой aggregation-функцией. Используй разные размеры наборов и доступные разные разрешения/бакеты, если исходные данные это позволяют. Подтверди mean=(low+high)/2 в пределах допуска. Проверь отсутствие повторного shift; weighted sigma path дополнительно покрывается направленной проверкой без отдельного обучающего прогона.

G05 — Validation не меняет обучение.
Вокруг реального evaluation сохрани и сравни model/LoRA modes, RNG всех используемых генераторов, existing gradients, веса LoRA, optimizer и scheduler state. Для этого сравнения не допускается изменение соответствующего состояния. Затем два контрольных запуска по 4 optimizer updates из одинаковых начальных LoRA-весов, с одинаковыми batches/seeds/прочими настройками: validation включена/выключена, samples выключены в обоих. Сравни training noise/timesteps, losses и изменения LoRA с заранее заданной точностью. Обеспечь одинаковый порядок данных, не приписывай расхождение shuffle влиянию validation. Не загружай вторую полную базовую модель одновременно ради сравнения.

G06 — Resume из current и из best.
Используй реальные states предыдущих тестов. Для каждого из двух вариантов создай отдельный тестовый эксперимент, сохранив исходный state как неизменный вход. Прочитай сохранённый шаг s из metadata, выполни ещё 4 optimizer updates. Начальная val относится к s, новая training loss начинается с s+1, итоговый step=s+4. Проверь загруженные веса/optimizer/scheduler/RNG и непрерывность LR/счётчиков, сохранение validation-протокола и best score/history. Не проверяй только факт отсутствия ошибки load_state. Не требуй идентичности с непрерывным обучением при иной позиции dataloader: точное восстановление этой позиции не входит в заявленный контракт. Копия state между изолированными тестовыми экспериментами не считается дублем внутри одного эксперимента.

G07 — Файлы, TensorBoard и выбор best.
Прочитай реальные event files и filesystem результатов G01/G02/G06. Проверь шесть точных scalar tags, ожидаемые absolute steps, finite values, единственность каждой пары tag/step внутри соответствующего запуска и сохранение training graphs. Новый процесс resume может иметь свою стартовую val на том же s — это предусмотренное отдельное событие нового запуска.
По журналу всех реальных validation events вычисли ожидаемый минимум val_loss_mean, правило строгого улучшения и текущий best. Сопоставь его step, metrics, model и samples с единственным полным пакетом в val-loss. На один step в output одного эксперимента — один пакет, один model.safetensors, один комплект samples; standalone/финальных копий вне пакета нет. Заявленные неизменные входы resume из другого эксперимента и отдельные архивы доказательств не считать выходными дублями тренера. Retention удаляет пакет целиком, включает граничный step и сохраняет best. Для G01 граница на step 12 равна 4. Проверка отсутствия дублей основана на событиях/metadata/расположении, а не только совпадении hashes разных optimizer states.

G08 — Пограничные ветви и ошибки.
Не продлевай обучение, ожидая случайного улучшения best, равной loss, NaN или ошибки записи. Такие ветви проверь конечными controlled tests реальных функций выбора/сохранения на малой модели: строгий best/tie, совпадение триггеров, best вне periodic save, включительная граница и защита старого best, прерывание записи, restoration после ошибки validation, ранние ошибки конфигов/пустых наборов/изменённых входов. Синтетические метрики допустимы только здесь, с явной пометкой controlled test; не подмешивай их в графики или выводы реальных GPU-прогонов.

G09 — Два GPU, если доступны.
Если в данном Pod есть два подходящих GPU, выполнить короткий двухпроцессный Accelerate запуск: 2 optimizer updates, accumulation=2, val_every_n_steps=1, N1=2,N2=1. Проверить отсутствие зависания, правильные counts/reduction, однократные tags и RNG-файлы обоих ranks, а также загрузку сохранённого state в двухпроцессном режиме без дополнительных updates. Если двух GPU нет, отметить NOT RUN с причиной; не арендовать второй Pod и не объявлять multi-GPU подтверждённым.

G10 — Итоговый отчёт.
Для каждой проверки: PASS/FAIL/NOT RUN, требование, условия, ожидание, измеренный результат и ссылка на evidence. Сохранить commit/patch hash, версии окружения/GPU, manifest fixtures и модели, эффективные конфиги, команды, exit codes, stdout/stderr, TensorBoard events, state metadata и измеренные отклонения. Не выдавать наличие PNG за хорошее качество модели, снижение loss за доказательство корректности, короткий запуск за проверку длительной стабильности или тест другого GPU за тест H200. Проверить README и добавить ссылку на фактический серверный отчёт с границами проверки; статус непрошедших или невыполненных тестов не скрывать.

В начальной матрице не более 40 optimizer updates реальной модели суммарно: 12+10+4+4+4+4 и, при наличии двух GPU, ещё 2. Validation-only, cache и sample inference не обновляют LoRA. Количество событий каждого сценария фиксируется в плане; successful tests не повторяются без изменения кода или конкретного неразрешённого вопроса. Повтор после потери соединения сначала проверяет живой процесс и его результаты, не запускает второй экземпляр.

После выполнения конечной матрицы заверши тестовые процессы и подготовь отчёт; Pod и пользовательские процессы оставь работающими. Ошибка функциональности оформляется как подтверждённое расхождение, а не повод обучать дольше. Недоступный обязательный тест означает неполную проверку. Исправления продуктового кода после локальной реализации — только по отдельному согласованному списку и в пределах правил двух раундов Конституции; не исправляй их скрыто только на сервере.
```

### 5.2. План проверок и развёртывания

```
$speckit-plan SPECIFY_FEATURE_DIRECTORY=specs/006-runpod-verification

Составь конкретный план выполнения G01–G10 по завершённым спецификациям этапов 1–4. Допускается read-only SSH inventory для определения среды и путей; на этом шаге не устанавливай зависимости, не создавай кэши и не запускай модель. Если доступ не работает, зафиксируй блокирующее условие, не угадывай среду.

Подтверди подходящую версию кода до запуска. Зафиксируй способ clone/checkout или точного переноса локальных изменений, включая проверку совпадения source hashes. Basic SSH не гарантирует SCP/SFTP и произвольный noninteractive exec; проверь доступный способ управления терминалом. При необходимости запроси SSH over exposed TCP. Не отключай проверку SSH host key для удобства. Отделяй transport/environment failure от ошибки фичи.

Определи каталоги, изолированное окружение, зависимости по pyproject/lock и реальные paths моделей/fixtures. Укажи необходимые изменения только тестовых configs: короткий step budget, warmup, N1/N2, sample settings, retention. До первого optimizer update проверь эффективный max_train_steps каждого запуска; наследование 1600 недопустимо. Не уменьшай precision/resolution и не включай FP8 для обхода OOM молча; несоответствие среды целевой конфигурации должно попасть в отчёт.

Создай verification-plan.md с матрицей G01–G10: requirement → setup → точная команда → максимальные optimizer steps/validation calls → ожидаемые события → критерий PASS → файлы evidence. Задай численные rtol/atol заранее, обоснуй precision/backend; seeds/t/RNG и дискретные counts сравнивай точно. Численное расхождение нельзя скрыть поздним ослаблением допуска.

Спланируй минимальные test scripts/wrappers над реальными entrypoints и функциями. Не создавай второй trainer, плагинную систему тестов или платформу управления Pod. Для независимого reference aggregation не переиспользуй проверяемый reducer. Используй уже имеющиеся CPU-тесты для controlled ветвей вместо долгого ожидания их на GPU.

Обеспечь фиксированный run_id, command/PID/exit-code/log files, сохранение результатов при разрыве SSH и проверку уже запущенного процесса перед повтором. Используй имеющиеся средства удалённой оболочки; бесконечный мониторинг/автоматические retries не нужны. Отсутствие общего лимита времени не отменяет обработки потерянного соединения, аварийного процесса и подтверждённого зависания.

Определи, как сохранить отчёт в specs/006-runpod-verification/verification.md, компактные результаты в verification.json, а сырые logs/events — в отдельной папке результатов. Получи читаемые evidence локально доступным способом; не копируй базовые модели, пользовательские datasets и все тяжёлые states ради отчёта. Если артефакт остался только на Pod, явно укажи его местонахождение и статус получения.

Включи подготовку сервера, скриптов и configs отдельной фазой Preparation; GPU-матрицу и отчёт — фазами Verification и Report. Серверные дефекты существующей фичи фиксируются для согласования исправлений, не разрешают скрытую переделку требований. Проверки считаются завершёнными после конечного списка, без условия «дождаться хорошего loss».
```

### 5.3. Задачи

```
$speckit-tasks SPECIFY_FEATURE_DIRECTORY=specs/006-runpod-verification

Сформируй задачи по фазам Preparation, Verification, Report.
Preparation: SSH/environment preflight; точное развёртывание проверяемого кода; отдельное окружение; inventory и manifest фиксированных inputs; реальные cache-команды; отдельные короткие configs; минимальные проверочные scripts; статическая проверка step budgets, expected events и критериев PASS.
Verification: G01–G09 в порядке зависимостей с переиспользованием неизменных checkpoints; early stop при провале prerequisite; условный multi-GPU без дополнительной аренды.
Report: G10, сбор evidence, PASS/FAIL/NOT RUN, подтверждённые дефекты, сверка README и ссылка на отчёт, проверка отсутствия собственных незавершённых тестовых процессов.

Укажи requirement/test ID, файлы, входы, критерий завершения каждой задачи. Полное обучение на 1600 шагов, подбор качества и повторные успешные прогоны в задачи не включать. Общая начальная матрица реальной модели — максимум 40 optimizer updates, включая оба сравнения и оба resume-сценария. Задачи NOT RUN не помечать как успешно проверенные.
```

### 5.4. Анализ готовности тестов

```
$speckit-analyze SPECIFY_FEATURE_DIRECTORY=specs/006-runpod-verification

Проверь spec/plan/tasks и verification-plan против требований этапов 1–4. Только анализ без изменений.

Проверь: точная версия кода; доступ и transfer method; поддерживаемое окружение; разделение пользовательских данных и test fixtures; реальные cache/CLI; конечные step budgets; отсутствие наследования 1600; независимые reference values; заранее заданные допуски; validation-only без updates; одинаковые inputs A/B; правильный resume budget s+4; один срок хранения; доказуемое отсутствие дублей; optional multi-GPU с честным NOT RUN; сбор exit codes/evidence и восстановление контроля после разрыва SSH.

Выяви ситуации, в которых тест может пройти по mock, наличию файла или снижению loss без доказательства требования. Подтверди, что test harness не меняет production algorithm и не подменяет реальные GPU-метрики. Обнаруженные противоречия устранить в документах до implement; GPU-тесты автоматически не запускать.
```

### 5.5. Подготовка Pod и проверочных сценариев

```
$speckit-implement SPECIFY_FEATURE_DIRECTORY=specs/006-runpod-verification

Выполни только фазу Preparation из tasks.md. Это отдельный серверный этап: разрешены подключение к указанному Pod, развёртывание точной версии musubi-tuner, создание изолированного окружения, установка зависимостей, подготовка тестовых configs/scripts и необходимых caches из моделей/данных /workspace. Optimizer updates пока не запускай.

Работай в выделенной папке, исходные модели/данные и чужие окружения не изменяй. Зафиксируй commit/patch hashes, GPU/versions, manifest fixtures, команды caching и все эффективные test budgets. Если обнаружен дефект уже реализованного инструмента, сохрани доказательство для согласования; не правь его только на сервере.

Покажи готовность каждой предпосылки и точные сценарии следующей фазы. Выполненная подготовка не является PASS GPU-проверок. Pod не выключай. Следующая команда отдельно запускает Verification и Report.
```

### 5.6. Выполнение конечной матрицы тестов

```
$speckit-implement SPECIFY_FEATURE_DIRECTORY=specs/006-runpod-verification

Выполни фазы Verification и Report по подготовленным tasks.md и verification-plan.md. Пользователь разрешил эти короткие GPU-прогоны для тестирования; повторное разрешение на каждый предусмотренный сценарий не запрашивай. Лимита wall-clock времени нет, но начальная матрица ограничена 40 optimizer updates и заранее перечисленными validation/inference событиями. Pod отключает пользователь.

Перед каждым процессом проверь эффективный config и его конечный step budget. Запускай реальные entrypoints, собирай stdout/stderr, exit codes, TensorBoard events и state metadata. Сначала проверяй prerequisites, затем зависимые сценарии. При потере SSH сначала выясни состояние уже запущенного теста. Не продолжай обучение ради снижения loss, нового best или визуального качества; controlled ветви проверяй предназначенными для них тестами.

На functional failure останови зависимые сценарии и сохрани evidence; независимые безопасные проверки можно закончить. Отделяй проблему доступа/окружения от ошибки инструмента. Не ослабляй assertions/tolerances/конфиги ради PASS. Не исправляй продуктовый код скрыто: подтверждённые дефекты идут в отчёт и на отдельное согласование по Конституции.

Сохрани verification.md и verification.json с PASS/FAIL/NOT RUN по G01–G10, source/environment/config/input идентификаторами, ожидаемыми/фактическими steps/counts, численными отклонениями и ссылками на доказательства. Получи локальную копию отчёта и доступных компактных evidence. После конечной матрицы заверши работу, убедись, что собственные тестовые процессы не продолжают updates. Pod и пользовательские процессы не останавливай.
```

### 5.7. Проверка полноты серверного тестирования

```
$speckit-converge SPECIFY_FEATURE_DIRECTORY=specs/006-runpod-verification

Сверь серверные test scripts/configs, результаты verification.md/json и evidence со spec/plan/tasks и матрицей G01–G10. На этом шаге новые GPU-прогоны не запускай; код/spec/plan и результаты тестов не переписывай.

Проверь, что PASS подкреплён реальным запуском именно указанной версии и независимой проверкой результата; что отсутствие ошибок или снижение loss не заменило критерии. Сверь количество optimizer updates, корректность временных шагов и усреднений, точные RNG snapshots, численные допуски, resume states, event files, best/retention и отсутствие физических дублей внутри эксперимента.

Различай отсутствующий тест, проваленный тест, недоступную среду и подтверждённый дефект реализации. NOT RUN на двух GPU при единственном доступном GPU допустим как ограничение отчёта, но multi-GPU не объявляется проверенным. Другие непроверенные обязательные требования не позволяют объявить серверную проверку полной.

Добавляй только подтверждённые оставшиеся задачи с указанием исходного требования и evidence. Покажи конкретный список необходимых исправлений для отдельного согласования и учитывай уже использованные раунды по Конституции. Не начинай исправления/повторные прогоны автоматически. При отсутствии обязательных расхождений сообщи, что перечисленные серверные проверки прошли, с точными границами покрытия. Pod не выключай.
```

## Если analyze или converge выявили несоответствия

Эти команды запускаются только по конкретному отчёту. Замените <каталог текущей фичи> и <ID> фактическими значениями. Каждое сообщение согласует только перечисленные изменения.

После analyze — исправление документов:

```
Согласовываю исправления документов по пунктам <ID отчёта analyze> для SPECIFY_FEATURE_DIRECTORY=<каталог текущей фичи>. Внеси именно эти исправления в spec/plan/tasks, сохрани смысл требований и Конституцию. Код не реализовывай. После этого я отдельно запущу соответствующую speckit-analyze повторно.
```

После converge — один конкретный раунд исправления кода:

```
$speckit-implement SPECIFY_FEATURE_DIRECTORY=<каталог текущей фичи>

Согласовываю раунд исправлений №<1 или 2> только по подтверждённым пунктам <ID отчёта converge> и связанным задачам <T...>. Выполни их, проведи необходимые проверки, обнови README при необходимости. Не ослабляй требования, не расширяй scope и не начинай следующий раунд автоматически. Учти уже использованные раунды для исправляемой реализации; переход к серверному этапу не обнуляет этот лимит.

Для локальных этапов 1–4 эта команда не разрешает GPU-запуск или перенос на сервер. Если согласованные исправления относятся к серверному этапу 5, сначала внеси их в основную локальную рабочую копию, выполни затронутые локальные проверки, затем перенеси проверяемую версию в отдельный test run. Разрешён только повтор затронутых ранее определённых сценариев с теми же конечными step budgets; общие training runs и повтор всех уже прошедших тестов без причины не разрешены. Сохрани новый отчёт с source hash, оставив прежние результаты доступными. Pod не выключай.
```

После исправлений отдельно повторите converge соответствующего этапа. После второго раунда оставшиеся подтверждённые проблемы перечисляются; этап с такими проблемами не объявляется завершённым.
