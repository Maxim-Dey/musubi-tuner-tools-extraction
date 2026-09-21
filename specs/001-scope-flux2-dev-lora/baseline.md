# Local baseline before narrowing

Date: 2026-09-20. Source: `bf478828ffa8b5ef3ebdf225ee72431bdb463159`, Git branch `main`, plus the two untracked user templates. No application files were changed to obtain these results.

## Environment and protection record

The repository has no `.venv`. System Python is 3.13.7, outside `pyproject.toml`'s `>=3.10,<3.13` range. The bundled Python 3.12.14 satisfies that range and was used for the attempts below. Both interpreters lack pytest, torch, toml, voluptuous, accelerate, safetensors, transformers, diffusers, tensorboard and bitsandbytes. No dependency installation was performed.

Commands ran from the repository root with `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES` empty and `HF_HUB_OFFLINE=1`.

| Protected file | SHA-256 before planning |
| --- | --- |
| `flux2dev_lora/train.toml` | `3e2c5c671fa8da72668917e8e466099216409236c99a41524729a5cda7a493b5` |
| `flux2dev_lora/dataset.toml` | `13eef91db4f6747d8fbd30f164e6042cee2431661d6ef3dd669a4664eae2ee2f` |
| `.specify/memory/constitution.md` | `6725e90c675a7ce188d6d218096155fa4a3d465c87f1fac3be8190cdb2b6fbcb` |

Initial `git status --short --branch`: `main...origin/main`, untracked `flux2dev_lora/` and `specs/`. Untracked content is not a deletion list. Spec Kit's setup script reports feature identifier `001-scope-flux2-dev-lora` as `BRANCH`; it does not change the actual Git branch.

## Executed checks

| Check | Observed result | Interpretation |
| --- | --- | --- |
| `python -m pytest -p no:cacheprovider -q tests/test_save_precision.py tests/test_lora_dtype_bridging.py tests/test_grad_metrics.py tests/test_datasource_item_extras.py` | Exit 1: `No module named pytest` | Blocked before collection; no test passes or failures established. |
| Normal training-template path: import `flux2_setup_parser`, `setup_parser_common`, `read_config_from_file`; set argv to `--config_file flux2dev_lora/train.toml`; parse and merge | Exit 1 on import: `No module named 'torch'` | Production parsing unavailable. |
| Normal dataset-template path: `load_user_config`, `ConfigSanitizer`, `BlueprintGenerator.generate` | Exit 1 on import: `No module named 'toml'` | Production schema/blueprint validation unavailable. The attempted probe used an architecture placeholder and never reached generation; future probes must use the real constant `ARCHITECTURE_FLUX_2_DEV` (`f2d`). |
| Import `musubi_tuner.networks.lora_flux_2` followed by sampling helpers | Exit 1: missing torch | Dynamic adapter dependency chain not executed. |
| `python flux_2_cache_latents.py --help` and `python -m musubi_tuner.flux_2_cache_latents --help` | Both exit 1: missing torch | Existing environment failure, before argument parsing. |
| Corresponding root/module `--help` for `flux_2_cache_text_encoder_outputs` | Both exit 1: missing torch | Same limitation. |
| Corresponding root/module `--help` for `flux_2_train_network` | Both exit 1: missing torch | Same limitation. |
| `ast.parse` of every tracked Python file returned by `git ls-files '*.py'` | 336 files parsed; zero syntax errors | Source syntax only; does not resolve imports. |
| `tomllib.loads` on both real templates | Both valid TOML; training template has 42 active top-level keys | Syntax only; not a substitute for the project's `toml`/argparse/Voluptuous path. |
| Resolve current shipped references in training template | Both `./configs/flux2_dev_style/dataset.toml` and `./configs/flux2_dev_style/sample_prompts.txt` absent | Pre-existing example-path defects, explicitly in implementation scope. |

Three additional source-isolated diagnostics executed the original function definitions selected by AST, with only their standard-library dependencies supplied. They did not import the production modules and are not integration tests:

- `line_to_prompt_dict`: `a ceramic cup on a wooden table --w 1024 --h 1024 --d 42 --s 20 --g 4` produced the expected prompt, dimensions, seed, steps and guidance dictionary.
- `should_sample_images` with the template intervals returned true at 0, 250 and 500; false at 1, 249 and 251.
- `get_remove_step_no` with interval 250 and retention 1000 returned `None, None, 0, 250, 750` at steps `250, 1000, 1250, 1500, 2000` respectively. Preserve the callers' handling of these values rather than replacing the retention rule with a file count.

The same isolated prompt parser accepted `test --unknown_option 1` with a warning and returned `{'prompt': 'test'}`. This confirms a pre-existing strict-validation gap covered by FR-011/FR-012.

## Source findings to distinguish from deletion regressions

- `training/parser_common.py:787-818` flattens arbitrary TOML section names into a Namespace. Unknown keys remain attached. Values already present in a Namespace do not obtain normal CLI `choices` validation. Validation of source keys and the final configuration is required, not just narrower CLI choices.
- `Flux2NetworkTrainer.process_sample_prompts` loads the text encoder after parsing prompts, but currently unknown prompt options are only warned about. Dynamic network and optimizer setup occur after sampling resources and DiT loading. Move model-independent validation ahead of all those loads without moving RNG-consuming training initialization.
- `Mistral3Embedder` rejects FP8 inside its constructor after weight loading; fail early for `fp8_text_encoder=true`, retaining the template's explicit false value.
- `NetworkTrainer._register_hooks_and_resume` registers adapter-only hooks and calls `accelerator.load_state`; subsequently `_run_training_loop` sets `epoch_to_start=0` and `global_step=0` (`trainer_base.py:2050-2055`, with an existing skip-step TODO). This is a confirmed source limitation, not a regression caused by narrowing. Runtime restoration and subsequent event numbering remain unverified. Characterize the existing behavior before any touched resume logic; do not silently repair counters, alter state formats, or claim interrupted/uninterrupted equivalence. If acceptance requires behavior absent from this baseline, report the discrepancy without weakening the specification or expanding this cleanup into a resume redesign.

## Comparison rule

Repeat available checks under the same interpreter, dependencies and fixtures before and after implementation. Once a suitable environment is available, establish the missing production-parser and CPU-test baseline before deleting owners. Record each result as passed, failed, blocked or not run. An import failure, mock-only diagnostic or unchanged source is not a passing runtime check. A pre-existing failure remains visible and must not be relabeled as a deletion regression or silently waived for completion.

## Implementation preparation: T001 (2026-09-20)

Source revision: `bf478828ffa8b5ef3ebdf225ee72431bdb463159`. No tracked application changes at entry.

Analyze gate: C1 (arbitrary one-level training groups) and G1 (early consumed tracker TOML validation) were approved and corrected in task ANALYZE (`01a0bb06-77d3-7c70-acc0-9ec274ff2dde`). Current plan/tasks/research/data-model/contract/quickstart contain both corrections. No unresolved document blocker remains from that analyze report. The runtime baseline and existing resume concern remain separate open gates.

Checklist `requirements.md`: 16 total, 16 checked, 0 unchecked; unchanged. No `.specify/extensions.yml` is present.

The user subsequently authorized installing necessary dependencies in this implementation task. This overrides the tasks execution-rule prohibition on installation only; builds, training, GPU execution, weights downloads and server work remain prohibited. Use a repository-local Python 3.12 environment and prebuilt CPU wheels.

Initial Git status:

```text
## main...origin/main
?? flux2dev_lora/
?? specs/
```

Initial untracked paths (all protected from cleanup):

```text
flux2dev_lora/dataset.toml
flux2dev_lora/speckit_commands.md
flux2dev_lora/train.toml
specs/001-scope-flux2-dev-lora/baseline.md
specs/001-scope-flux2-dev-lora/checklists/requirements.md
specs/001-scope-flux2-dev-lora/contracts/cli-and-config.md
specs/001-scope-flux2-dev-lora/data-model.md
specs/001-scope-flux2-dev-lora/plan.md
specs/001-scope-flux2-dev-lora/quickstart.md
specs/001-scope-flux2-dev-lora/research.md
specs/001-scope-flux2-dev-lora/spec.md
specs/001-scope-flux2-dev-lora/tasks.md
```

| Protected file | SHA-256 at implementation entry |
| --- | --- |
| `flux2dev_lora/train.toml` | `3e2c5c671fa8da72668917e8e466099216409236c99a41524729a5cda7a493b5` |
| `flux2dev_lora/dataset.toml` | `13eef91db4f6747d8fbd30f164e6042cee2431661d6ef3dd669a4664eae2ee2f` |
| `.specify/memory/constitution.md` | `6725e90c675a7ce188d6d218096155fa4a3d465c87f1fac3be8190cdb2b6fbcb` |
| `flux2dev_lora/speckit_commands.md` | `04eed9a22e09a97d4d3818827fa9e48133249140aabfeda644e92b02d10d52ad` |

The extra untracked `flux2dev_lora/speckit_commands.md` is unrelated user work, not a deletion candidate. Protected resource paths are recorded without traversing or reading their data:

- `models/`: absent at entry; preserve.
- `data/`: absent at entry; preserve.
- `output/`: absent at entry; preserve.
- `logs/`: absent at entry; preserve.
- `.specify/`: present; preserve.
- `.agents/`: present; preserve.
- `.cursor/`: present; preserve.
- `.github/`: present; preserve.
- `CONTRIBUTING.md`: present; preserve.
- `CONTRIBUTING.ja.md`: present; preserve.
- `images/logo_aihub.png`: present; preserve.

### Literal original templates

The snapshots below preserve every active key/value, all original comments, and the disabled swap example. Hashes above additionally protect exact bytes/line endings. Training has 42 active top-level settings; dataset has five general settings and three settings in its single dataset.

`flux2dev_lora/train.toml`:

```toml
# FLUX.2 Dev — обучение стилевой LoRA.
# Запуск из корня musubi-tuner. Пути к моделям и датасету измените под себя.

# === 1. Модели и датасет ===
model_version = "dev"
dit = "./models/flux2-dev/flux2-dev.safetensors"
vae = "./models/flux2-dev/ae.safetensors"
# Источник всех моделей: https://huggingface.co/black-forest-labs/FLUX.2-dev
# Все 10 оригинальных шардов Mistral 3 должны лежать рядом с первым.
text_encoder = "./models/flux2-dev/text_encoder/model-00001-of-00010.safetensors"
dataset_config = "./configs/flux2_dev_style/dataset.toml"

# === 2. Архитектура LoRA ===
network_module = "networks.lora_flux_2"
network_dim = 32
network_alpha = 32
network_dropout = 0.05

# === 3. Точность весов ===
# DiT и текстовый энкодер — BF16; VAE сохраняет исходную точность FP32.
mixed_precision = "bf16"
vae_dtype = "float32"
fp8_base = false
fp8_scaled = false
# FP8 для Mistral 3 в этой версии не поддерживается.
fp8_text_encoder = false

# === 4. Attention и экономия памяти ===
sdpa = true
gradient_checkpointing = true
# Необязательный перенос блоков между GPU и RAM; сейчас выключен.
# blocks_to_swap = 20

# === 5. Выбор уровня шума и взвешивание ошибки ===
timestep_sampling = "flux2_shift"
weighting_scheme = "none"

# === 6. Оптимизатор и learning rate ===
optimizer_type = "adamw8bit"
learning_rate = 1e-4
lr_scheduler = "constant_with_warmup"
lr_warmup_steps = 100
max_grad_norm = 1.0

# === 7. Длительность обучения и накопление градиентов ===
max_train_steps = 2000
# При batch_size = 1 и одной GPU эффективный batch равен 4.
gradient_accumulation_steps = 4
seed = 42

# === 8. Загрузка данных ===
max_data_loader_n_workers = 2
persistent_data_loader_workers = true

# === 9. Сохранение LoRA и полного состояния обучения ===
output_dir = "./output/flux2_dev_style"
# Основа имён LoRA, папок состояния и изображений-примеров.
output_name = "flux2_dev_style"
# Интервал в обновлениях оптимизатора.
save_every_n_steps = 250
# Полное состояние: LoRA, оптимизатор, планировщик LR и RNG.
save_state = true
save_state_on_train_end = true
# Окна хранения в шагах, не количество файлов. Старые сохранения удаляются.
# Чтобы хранить всё, закомментируйте оба save_last_n_steps*.
save_last_n_steps = 1000
# Отдельное окно для состояний; если не задано, наследует save_last_n_steps.
save_last_n_steps_state = 1000

# === 10. Генерация примеров ===
sample_prompts = "./configs/flux2_dev_style/sample_prompts.txt"
# Интервал в обновлениях оптимизатора.
sample_every_n_steps = 250
sample_at_first = true

# === 11. Логи TensorBoard ===
log_with = "tensorboard"
logging_dir = "./logs/flux2_dev_style"
# Префикс папки запуска перед автоматически добавляемой датой.
log_prefix = "flux2_dev_style_"
# Имя подпапки трекера TensorBoard; названия метрик не меняет.
log_tracker_name = "flux2_style"
```

`flux2dev_lora/dataset.toml`:

```toml
# Paths are relative to the musubi-tuner repository root.
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "./data/flux2_dev_style/images"
cache_directory = "./data/flux2_dev_style/cache"
num_repeats = 1
```


## Reviewed cleanup inventory: T002 (2026-09-20)

Source remains `bf478828ffa8b5ef3ebdf225ee72431bdb463159`. AST import tracing covered all tracked Python sources; direct/dynamic factory, GUI and registration review supplemented it. All exact paths in tasks.md D1-D4 exist and are tracked, with no duplicates; no untracked path belongs to a deletion set. No deletions have occurred.

| Set | Exact paths | SHA-256 of sorted LF-delimited paths | Reviewed disposition and prerequisite |
| --- | ---: | --- | --- |
| D1 | 115 | `baebee8de4d8555c8d9f6fd2b702c45b1e81a52f965057db7a265e0f463057f1` | 115 excluded root/package commands; delete only after T005-T011 detach retained callers (T012). Keep the three root/module cache/train pairs, package initializer and two internal cache helper modules. |
| D2 | 134 | `76251ec096244bcef800de659ae4bc153b4422cc42e16f5fba51951438782ed6` | 134 excluded model/GUI/adapter/audio/quantization implementation paths; delete in T013 after D1 and retained import closure. No excluded architecture is to be moved into a common owner. |
| D3 | 32 | `9dafa7e1a0d4bba381692f9adab82feea70721583c760c9646aad617c355cab0` | 32 candidates reviewed by imports and assertions, not names. Preserve the common cases listed below; T027 removes only exclusive cases after earlier adaptations. |
| D4 | 27 | `ef466bd872f4b734430b1df53ecafabb6049e973f0e8f200c1a752e0340188c2` | 27 excluded docs/assets; T028 must extract shared guidance/attribution before T029. Existing timestep figures and images/logo_aihub.png remain. |

Exact membership is the D1-D4 tables in tasks.md, protected by the fingerprints above. Revalidate these lists against the current index immediately before deletion.

### Retained importers of excluded owners

| Retained importer | Excluded owner (still present) |
| --- | --- |
| `src/musubi_tuner/cache_latents.py:16` | `src/musubi_tuner/hunyuan_model/vae.py` |
| `src/musubi_tuner/cache_latents.py:17` | `src/musubi_tuner/hunyuan_model/autoencoder_kl_causal_3d.py` |
| `src/musubi_tuner/cache_text_encoder_outputs.py:18` | `src/musubi_tuner/hunyuan_model/text_encoder.py` |
| `src/musubi_tuner/cache_text_encoder_outputs.py:19` | `src/musubi_tuner/hunyuan_model/text_encoder.py` |
| `src/musubi_tuner/dataset/cache_io.py:24` | `src/musubi_tuner/minimax_h3/packing.py` |
| `src/musubi_tuner/dataset/cache_io.py:25` | `src/musubi_tuner/minimax_h3/text_encoder.py` |
| `src/musubi_tuner/dataset/config_utils.py:17` | `src/musubi_tuner/dataset/audio_utils.py` |
| `src/musubi_tuner/dataset/datasources.py:11` | `src/musubi_tuner/dataset/audio_utils.py` |
| `src/musubi_tuner/dataset/image_video_dataset.py:42` | `src/musubi_tuner/dataset/audio_utils.py` |
| `src/musubi_tuner/flux_2/flux2_utils.py:36` | `src/musubi_tuner/zimage/zimage_utils.py` |
| `src/musubi_tuner/flux_2_train_network.py:11` | `src/musubi_tuner/hv_train_network.py` |
| `src/musubi_tuner/training/trainer_base.py:48` | `src/musubi_tuner/hv_generate_video.py` |
| `src/musubi_tuner/utils/lora_utils.py:207` | `src/musubi_tuner/networks/loha.py` |
| `src/musubi_tuner/utils/lora_utils.py:211` | `src/musubi_tuner/networks/lokr.py` |

Detachment assignments: Dev trainer re-exports/image saver -> T005; Hunyuan cache encoder/parser/main paths -> T006; Qwen/zimage and Klein classes/registries -> T009; video saving/plugin factory/LoHa/LoKr dispatch -> T010; MiniMax cache imports/audio datasets/constants -> T011. The graph is not yet detached.

Execution review: root wrappers delegate into package main; package main/__main__ include all model cache/train/generate tools, Self-Flow, captioning and weight utilities. GUI launches Qwen/ZImage workflows. `NetworkTrainer._build_network` dynamically imports `network_module` after inserting the package directory and has generic plugin fallback; `lora.py` exposes Hunyuan factory wrappers; `lora_utils` dynamically selects LoHa/LoKr. `flux2_utils` registers Dev and four Klein selections and loads Qwen through zimage. Dataset architecture/bucket/cache maps and VideoDataset fallback also require narrowing. `pyproject.toml` has no additional project.scripts entrypoints. Ordinary dynamic optimizer/scheduler imports remain supported and need early argument checks, not removal.

### Assertion dispositions before D3 removal

- Keep all assertions in `test_save_precision.py`, `test_lora_dtype_bridging.py`, `test_grad_metrics.py`; historical Ideogram labels in dtype tests characterize shared LoRAModule behavior.
- Retain `test_ideogram4_timesteps.py` and `test_krea2_timesteps.py`: every assertion targets common parser/timestep/trainer math required for optional Dev configurations; these files are not exclusive model tests.
- `test_datasource_item_extras.py`: preserve image JSONL extras filtering, base directory/line labels, empty directory extras, datasource indices and cache-batch association; change Qwen fixture architecture to Dev in T011. Remove only video assertions.
- `test_sai_model_spec.py`: current two tests are Ideogram/MiniMax metadata only. T011 replaces excluded-specific cases with production Dev metadata plus shared override checks without weakening the retained metadata contract. `test_top_level_entrypoints.py` requires the T012 retained/removal adaptation.
- `test_minimax_h3_te_streaming.py`: retain common default selector, mixed-dtype flat layout, forward hook ordering/output, and invalid selector attribute cases. The MiniMax selector/CUDA scenario is exclusive; shared fixtures must not keep that selector dependency.
- `test_minimax_h3_cache_plan.py`: preserve common cache callback skip/current/keep behavior and DatasetGroup index propagation using image fixtures in the focused workflow tests; remove H3 plan/provenance-specific semantics.
- `test_minimax_h3_generation_request.py`: preserve literal prompt text handling (`d 12 monkeys --w 64`). Unknown-switch acceptance is an explicitly authorized behavior change (FR-011/012), so replace that expectation with strict rejection, not a hidden skip.
- `test_audio_dataset_seam.py`: video/audio APIs are excluded; carry the applicable cwd-first/JSONL-relative/missing-path expectations into image JSONL checks. No audio owner needs retaining.
- `test_ideogram4_fp8_loading.py` and `test_ideogram4_te_fp8_loading.py`: checkpoint remapping/markers are exclusive, while the common FP8 linear patch arithmetic and existing tolerance need retained tiny-tensor coverage. Do not preserve the Ideogram loader for that coverage.
- `test_minimax_h3_training.py`: joint AV/guidance/teacher objectives, formats and H3 targeting are exclusive. Preserve applicable base image saver and common LoRAModule enable/bypass/restore behavior in workflow fixtures; no teacher model or objective port.
- Other D3 assertions target the inventoried excluded models, quantization schemes, video/audio/one-frame timelines, caption presentation, generation, checkpoint mapping or model-specific checkpoint/offload integration. Generic helper use inside these scenarios does not make their model executable support necessary; Dev/helper regressions cover retained behavior. `test_minimax_h3_vae.py` shard/prefix helpers belong to the excluded checkpoint module, not the retained `load_split_weights`.

### Dependency consumers and setup

Keep torch/torchvision (Dev/trainer and image grid), accelerate (trainer/loaders), bitsandbytes (AdamW8bit), diffusers (common scheduler/weighting), einops (packing), Hub/transformers/sentencepiece (Mistral/processor and artifact I/O), numpy/Pillow/OpenCV (image transforms), safetensors (caches/adapters), toml/voluptuous (parsers), tqdm (loops), TensorBoard (active tracker), ascii-magic (console image preview), matplotlib (timestep plots), packaging (version handling), Ruff/pytest (development). `av` consumers are video/audio/preview paths, `ftfy`/`easydict` are Wan-only, Gradio/sensecraft are GUI/HiDream-only and prompt-toolkit is standalone generation-only. Remove those declarations only in T026 after import closure; keep their installed baseline packages until then. Optional retained backends remain conditional.

Git setup verified. The existing .gitignore lacked bytecode/build/egg-info and ordinary temporary/OS artifacts; only those patterns were added. No Docker, Node publication, ESLint, Prettier, Terraform or Helm setup was detected, so no unrelated ignore files were added.

## Production baseline: T003, equipped attempt (2026-09-20)

Interpreter: `C:\Users\inbox\Desktop\musubi-tuner-flux2dev-lora\.venv\Scripts\python.exe` (`3.12.14`), source unchanged at `bf478828ffa8b5ef3ebdf225ee72431bdb463159`. Environment: `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`. User-approved install used only wheels via `uv pip install --only-binary :all:`; no project build/editable installation. PyTorch pair: 2.7.1+cpu / torchvision 0.22.1+cpu.

The initial sandbox test attempt was blocked by a PermissionError reading pytest installed through the uv cache. The same test command outside the sandbox completed: **29 passed, 1 warning**, exit 0, 29.95s. Files: `test_save_precision.py`, `test_lora_dtype_bridging.py`, `test_grad_metrics.py`, `test_datasource_item_extras.py`. Pytest initially emitted Windows access-violation diagnostics during torch import, but the process continued and completed all tests; retain this observation. A separate ordinary torch import and two-element CPU addition passed, reporting no CUDA runtime. `python -m pip check`: passed, no broken requirements. Third-party einops invalid-escape warning remains visible.

- **passed**: `python -B flux_2_cache_latents.py --help`.
- **passed**: `python -B -m musubi_tuner.flux_2_cache_latents --help`.
- **passed**: `python -B flux_2_cache_text_encoder_outputs.py --help`.
- **passed**: `python -B -m musubi_tuner.flux_2_cache_text_encoder_outputs --help`.
- **failed**: `python -B flux_2_train_network.py --help`.

```text
Trying to import sageattention
Failed to import sageattention
C:\Users\inbox\Desktop\musubi-tuner-flux2dev-lora\.venv\Lib\site-packages\einops\einops.py:847: SyntaxWarning: invalid escape sequence '\s'
  \sum_{c, d, g} x[a, b, c] * y[c, b, d] * z[a, g, k]
Traceback (most recent call last):
  File "C:\Users\inbox\Desktop\musubi-tuner-flux2dev-lora\flux_2_train_network.py", line 4, in <module>
    main()
  File "C:\Users\inbox\Desktop\musubi-tuner-flux2dev-lora\src\musubi_tuner\flux_2_train_network.py", line 353, in main
    args = parser.parse_args()
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\argparse.py", line 1904, in parse_args
    args, argv = self.parse_known_args(args, namespace)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\argparse.py", line 1914, in parse_known_args
    return self._parse_known_args2(args, namespace, intermixed=False)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\argparse.py", line 1943, in _parse_known_args2
    namespace, args = self._parse_known_args(args, namespace, intermixed)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\argparse.py", line 2184, in _parse_known_args
    start_index = consume_optional(start_index)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\argparse.py", line 2113, in consume_optional
    take_action(action, args, option_string)
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\argparse.py", line 2018, in take_action
    action(self, namespace, argument_values, option_string)
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\argparse.py", line 1148, in __call__
    parser.print_help()
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\argparse.py", line 2621, in print_help
    self._print_message(self.format_help(), file)
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\argparse.py", line 2627, in _print_message
    file.write(message)
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode characters in position 8113-8128: character maps to <undefined>

```
- **failed**: `python -B -m musubi_tuner.flux_2_train_network --help`.

```text
Trying to import sageattention
Failed to import sageattention
C:\Users\inbox\Desktop\musubi-tuner-flux2dev-lora\.venv\Lib\site-packages\einops\einops.py:847: SyntaxWarning: invalid escape sequence '\s'
  \sum_{c, d, g} x[a, b, c] * y[c, b, d] * z[a, g, k]
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\inbox\Desktop\musubi-tuner-flux2dev-lora\src\musubi_tuner\flux_2_train_network.py", line 365, in <module>
    main()
  File "C:\Users\inbox\Desktop\musubi-tuner-flux2dev-lora\src\musubi_tuner\flux_2_train_network.py", line 353, in main
    args = parser.parse_args()
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\argparse.py", line 1904, in parse_args
    args, argv = self.parse_known_args(args, namespace)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\argparse.py", line 1914, in parse_known_args
    return self._parse_known_args2(args, namespace, intermixed=False)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\argparse.py", line 1943, in _parse_known_args2
    namespace, args = self._parse_known_args(args, namespace, intermixed)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\argparse.py", line 2184, in _parse_known_args
    start_index = consume_optional(start_index)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\argparse.py", line 2113, in consume_optional
    take_action(action, args, option_string)
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\argparse.py", line 2018, in take_action
    action(self, namespace, argument_values, option_string)
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\argparse.py", line 1148, in __call__
    parser.print_help()
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\argparse.py", line 2621, in print_help
    self._print_message(self.format_help(), file)
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\argparse.py", line 2627, in _print_message
    file.write(message)
  File "C:\Users\inbox\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode characters in position 8113-8128: character maps to <undefined>

```
- **passed**: `python -B -c "import sys; from musubi_tuner.training.parser_common import setup_parser_common,read_config_from_file; from musubi_tuner.flux_2_train_network import flux2_setup_parser; sys.argv=['check','--config_file','./flux2dev_lora/train.toml']; p=flux2_setup_parser(setup_parser_common()); a=read_config_from_file(p.parse_args(),p); assert a.model_version=='dev'; assert a.network_module=='networks.lora_flux_2'; print('training TOML parsed')"`.
- **passed**: `python -B -c "import argparse; from musubi_tuner.dataset.config_utils import ConfigSanitizer,BlueprintGenerator,load_user_config; from musubi_tuner.dataset.architectures import ARCHITECTURE_FLUX_2_DEV; b=BlueprintGenerator(ConfigSanitizer()).generate(load_user_config('./flux2dev_lora/dataset.toml'),argparse.Namespace(),architecture=ARCHITECTURE_FLUX_2_DEV); assert b.dataset_group.datasets; print('Dev blueprint parsed')"`.
- **passed**: `python -B -c "import sys; from pathlib import Path; import musubi_tuner; from musubi_tuner.networks import lora_flux_2; sys.path.insert(0,str(Path(musubi_tuner.__file__).parent)); import networks.lora_flux_2; print('both adapter spellings imported')"`.
- **passed**: `python -B -c "from pathlib import Path; import tempfile; from musubi_tuner.training.sampling_prompts import load_prompts; t=tempfile.TemporaryDirectory(); p=Path(t.name)/'sample.txt'; p.write_text('A ceramic cup on a wooden table.\n',encoding='utf-8'); result=load_prompts(str(p)); assert result and result[0]['prompt']=='A ceramic cup on a wooden table.'; print('production prompt loader parsed temporary fixture'); t.cleanup()"`.

- **passed**: stdlib AST parsing of 336 tracked Python files and TOML syntax of both original templates (42 active training keys). These are supplementary source checks.
- **not run**: shipped prompt loading, because the prompt is not supplied until T020. The temporary prompt above checks the actual production loader, not the shipped-reference deliverable. Both old configs/flux2_dev_style references remain absent, as recorded in the original baseline.

Installed distributions for reproducible comparison:

```text
Jinja2==3.1.6
Markdown==3.10.3
MarkupSafe==3.0.3
PyYAML==6.0.3
Pygments==2.21.0
Werkzeug==3.1.8
absl-py==2.5.0
accelerate==1.6.0
ascii-magic==2.3.0
av==14.0.1
bitsandbytes==0.50.2
certifi==2026.7.22
charset-normalizer==3.5.1
colorama==0.4.6
contourpy==1.4.0
cycler==0.12.1
diffusers==0.32.1
easydict==1.13
einops==0.7.0
filelock==3.32.3
fonttools==4.65.0
fsspec==2026.7.0
ftfy==6.3.1
grpcio==1.84.0
huggingface-hub==0.34.3
idna==3.20
importlib_metadata==9.0.1
iniconfig==2.3.0
kiwisolver==1.5.1
matplotlib==3.10.0
mpmath==1.3.0
networkx==3.6.1
numpy==2.5.2
opencv-python==4.10.0.84
packaging==26.3
pillow==12.3.0
pip==25.0.1
pluggy==1.6.0
prompt_toolkit==3.0.51
protobuf==7.36.2
psutil==7.2.2
pyparsing==3.3.2
pytest==9.1.1
python-dateutil==2.9.0.post0
regex==2026.9.10
requests==2.34.2
ruff==0.15.22
safetensors==0.4.5
sentencepiece==0.2.1
setuptools==78.1.0
six==1.17.0
sympy==1.14.0
tensorboard-data-server==0.7.2
tensorboard==2.21.0
tokenizers==0.22.2
toml==0.10.2
torch==2.7.1+cpu
torchvision==0.22.1+cpu
tqdm==4.67.1
transformers==4.57.6
typing_extensions==4.16.0
urllib3==2.8.0
voluptuous==0.15.2
wcwidth==0.8.4
zipp==4.1.0
```
# T003 UTF-8 help retry

Both training help variants (`flux_2_train_network.py --help` and
`python -m musubi_tuner.flux_2_train_network --help`) passed with exit 0 and
the expected model option after setting `PYTHONIOENCODING=utf-8` in the same
Python 3.12.14 CPU environment. No application code changed. The earlier
cp1252 output failures remain recorded above. Together with the four cache
help calls, all six retained help invocations now have a passing baseline.

## T004 original CPU contract fixtures

Source is still bf478828; no application source edits or deletions preceded
these fixtures. `.venv/Scripts/python.exe -m pytest -p no:cacheprovider -q
--tb=short tests/test_flux2_workflow_contracts.py` used the T003 CPU/offline/UTF-8
environment. First draft: 21 passed, 1 failed (test reference used Python-double
normalization instead of the original float32 result; no production change).
Corrected original float32 value and nested control fixture; added sample
mode/default/name/tracker evidence: **23 passed**, 1 existing einops warning.
The Windows import-time access-violation diagnostic appeared again, but the
process continued and exited 0; it is not hidden or counted as a process crash.

Contracts cover RGB/RGBA preprocessing, exact filenames/cache keys/metadata,
NaN handling, text dtype replacement and collation, cache callbacks/batching/
skip/current/keep/cleanup, reference coordinates, flow target and scalar MSE,
PNG pixels/names, sample/update/retention boundaries, swap/mode and RNG return,
local tracker payloads, TensorBoard scalar readback and Accelerate CPU toy
state files/weights/optimizer LR and RNG. All tensors and files are tiny
fixtures; no trainer loop, model weights, GPU or external service is used.

Resume limitation remains: `_register_hooks_and_resume` loads state before
`_run_training_loop`, whose original lines 2054 onward reset `epoch_to_start`
and `global_step` to zero. This is source evidence, not a test of source
spelling or execution of the prohibited training loop. The toy state round
trip does not establish restoration of training progress. Existing precision,
LoRA dtype and gradient assertions are reused, not duplicated.

## T005–T007 shared-owner checkpoint

`save_images_grid` moved byte-for-byte to the existing image-utils owner;
the old owner temporarily re-exports it. Dev imports now target the existing
trainer/parser/sampling/accelerator owners. Common cache helpers retain
callback loops and image/console previews; Hunyuan encoders, parser additions,
main blocks and video previews were removed. No D1/D2 owner file is deleted.
The unchanged 23 workflow assertions pass (30.06 s, exit 0, same einops/import
diagnostics) using the T004 command/environment. AST parsing of touched
entrypoints/helpers passed. Remaining excluded dependencies are the video
saver in trainer_base (T010), MiniMax cache imports and dataset architecture
branches (T011), and Qwen/adapter factories (T009–T010).

## T008 initial rejection matrix

Production entrypoint/parser tests use tiny image/cache/TOML/JSON fixtures and
raising sentinels at all three real weight loaders. No trainer loop executes.
Initial test-authoring syntax/helper-name errors were corrected before using
results as evidence (earlier attempts: collection error; 68 failed/17 passed;
87 failed/18 passed). Final pre-narrowing matrix: **84 failed, 21 passed** in
25.48 s, exit 1 (`pytest ... --tb=no tests/test_flux2_scope.py`, T003 environment).
These failures expose acceptance/diagnostic gaps, not successful rejections.
All three valid entries reach only the guarded loader, arbitrary one-level
groups and CLI precedence pass, valid prompt formats pass, and a valid tracker
file reaches the loader without initializing a writer. Invalid model/method,
raw/grouped/effective values, dataset TOML/JSON, prompts, tracker files and
selected-dependency cases remain open for T009–T017. None is satisfied by an
ImportError. Existing unknown CLI cases already reject before loaders.

## T009–T010 Dev/model/adapter checkpoint

Klein parameter classes/registry entries, Qwen embedder and zimage dependency,
non-distilled CFG, video/audio sample construction/saving, Hunyuan adapter
wrappers, arbitrary network factory dispatch and LoHa/LoKr merge dispatch
are removed. Standard LoRA and internal base-weight merge/inference remain.
The standard Dev guidance-embedding computation is unchanged; the unused
Hunyuan guidance CLI selector is removed, preserving the Dev metadata default
`ss_guidance_scale=1.0`. Numeric timestep formulas remain. No D1/D2 files deleted.
The T004 workflow plus existing save precision, dtype bridging, grad metrics,
Ideogram/Krea numeric timestep tests: **55 passed**, 1 existing einops warning,
18.70 s, exit 0 in the same CPU environment.

## T011 image/data/cache checkpoint

Removed other architecture identifiers/bucket registrations, video/audio and
FramePack dataset/source branches, layered multiple-target sequencing, and
exclusive cache writers. Dev serializer bodies, control bucketing, fallback
order, paths, repeats and metadata defaults are retained. Mixed datasource
assertions now use Dev; only video assertions were removed. Metadata tests
cover Dev defaults and ordinary custom metadata overrides.

The first T011 comparison hung in the image batching test and was interrupted
(exit 1, not a pass): an over-broad conditional splice removed adjacent common
batch append/future removal code. Restored that code directly from HEAD and
removed only complete excluded architecture chains; no algorithm adjustment.
Repeat of workflow/datasource/metadata tests: **33 passed**, 1 existing einops
warning, 30.14 s, exit 0. `ruff check --select F821` on touched owner groups
passed. Static retained-source import audit found zero imports into D1/D2.
The original unrelated user data/templates/constitution are untouched.

## T012 command-removal checkpoint

Removed exactly the 115 tracked D1 files after the retained-caller audit.
Root commands are exactly the three Dev commands; package top-level files are
those three, __init__, and the two import-only cache helpers. Tests now execute
commands rather than comparing wrapper text. Six actual root/module --help
calls plus the command-surface assertion: **7 passed**, 166.60 s, exit 0,
using the same Python/CPU/offline/UTF-8 environment; no weights loaded.
Internal config/FP8 developer diagnostics are not operational model workflows;
no excluded model is selected through them, and they were not executed.

## T013 implementation-removal checkpoint

Removed exactly the 134 tracked D2 files after static/dynamic caller review.
No aliases or relocated foreign implementations were introduced. Existing
attribution/license files and retained FLUX source notices remain intact.
Post-removal production imports and workflow/datasource/metadata comparison:
**33 passed**, 1 existing einops warning, 21.69 s, exit 0. Dynamic retained
imports now select only optimizer/scheduler classes (validated in T016),
not network/model implementations. D3 tests await assertion extraction and
T027; their now-excluded collection is deliberately not reported as passing.

## T014 raw/effective training configuration

Training grouping labels remain unrestricted at one level. Each contained
key is checked before flattening; unknown/deeper values cannot disappear via
CLI override. Effective types/choices/coupled values are checked after the
FP32 VAE default, with CLI/file/full-key source locations. Existing flattening
and explicit CLI precedence remain. Focused model/CLI/group/type cases:
**33 passed, 72 deselected**, 14.51 s, exit 0. This is not a claim that the
remaining adapter/dataset/prompt/tracker boundary cases already pass.

### T015 — strict dataset and prompt inputs

Passed: supported Python 3.12 CPU/offline environment, `pytest -p no:cacheprovider -q --tb=short tests/test_flux2_scope.py -k "dataset or prompt_source or valid_prompt"`: 43 passed, 62 deselected; `tests/test_datasource_item_extras.py`: 3 passed. Dataset TOML/JSON reject excluded/unknown keys and malformed sources with full source paths. TXT/TOML/JSON prompts retain image defaults, repeated controls, negative/CFG compatibility and TXT step clamping; unsupported/internal fields and malformed tokens fail. No checkpoint opened. Existing einops escape warning remains.

### T016 — selected nested contracts and trackers

Initial focused run: 30 passed, 3 failed. Two failures exposed escaped Windows path diagnostics for missing/malformed tracker files; messages now include the actual path. The valid tracker test replaced `SummaryWriter.__init__` without preserving its signature; its fail-on-construction sentinel now uses `functools.wraps` so production signature inspection observes the real contract. Assertions and the prohibition on starting a tracker are unchanged. Re-run: `pytest -p no:cacheprovider -q --tb=short tests/test_flux2_scope.py -k "training_rejects or tracker or dependency or valid_entry"`: **33 passed, 72 deselected** (Python 3.12, CPU/offline). No optimizer, tracker or weights constructed by validation. Ordinary optimizer/scheduler dispatch remains in place; keyword contracts are checked on the selected callable, including inherited forwarding contracts. Template settings were not changed. `num_timestep_buckets=0` remains accepted because the original numeric implementation treats values <=1 as disabled.

### T017–T018 — pre-weight boundary and US2 checkpoint

Passed: first full source matrix 105 cases; extended matrix 117 cases after adding valid nested/custom optimizer/scheduler, missing checkpoint/shard/processor and no-RNG preflight cases. Combined US2/metadata/entrypoint run initially had **130 passed, 1 failed**: signature-default inference incorrectly treated SGD's `momentum=0` default as integer-only. Fixed validation to respect explicit integer annotations while allowing numeric values for unannotated numeric defaults; no optimizer code/value changed. Re-run of the changed source matrix: **117 passed**. The previous combined run already passed all **7 metadata and 7 entrypoint checks**, including all six root/module help calls; no help source changed afterward.

Preflight reads original sources and checks images/captions/controls, distinct writable caches/destinations, local resume directories, optional initial/base adapters, companion shard names and the fixed cached Mistral processor before any weight boundary. Path checks create no files; processor lookup uses `local_files_only=True`. Separate RNG test verifies Python/NumPy/torch state remains unchanged. Positive entry fixtures use empty checkpoint path placeholders plus explicit AE/DiT/Mistral loader sentinels. Processor availability is isolated by an explicit test double checking the exact processor ID and offline arguments; this is **not** evidence of a real cached processor. T032 retains that actual resource gate.

Plan/source discrepancy identified before the JSONL adjustment: cwd-first then JSONL-directory fallback existed in the original video reader, not the original Dev image reader. The plan explicitly requires it for images. Explained the concrete difference to the user; the shipped directory-based template is unaffected. Implemented the agreed plan's fallback in the existing image reader, preserving cwd precedence and missing paths for actionable validation; no new data format or training math.

Execution audit: root has exactly three retained commands; package exposes those three with the two internal cache helpers. Architecture/model maps contain only `f2d`/`flux_2_dev`/`dev`; canonical LoRA factory is fixed, supplied metadata spelling preserved. Remaining dynamic imports select ordinary optimizer/scheduler/dependency implementations, not model/adapter plugins. D1's 115 and D2's 134 excluded files are absent. Internal dataset-config/FP8 developer examples are not excluded model workflows and were not run. No GUI implementation or generation/full-training/converter/alternate-objective path remains.

Additional pre-existing issue observed, not repaired: `--dim_from_weights` is boolean but the old trainer passes it to `load_file`, then unpacks a factory return as a pair although the Dev factory returns one network. This optional route was already broken at baseline; do not claim successful use or silently redesign it in cleanup.

### T019–T020 — template contract before path correction

New production template tests compare every active training/dataset value and the complete original training text/comments against the literal T001 snapshot, with only the two allowed replacements. Before the change: **2 failed, 23 deselected**; the comparison reports exactly two incorrect references and 40 identical remaining settings. The missing prompt/reference defects were already recorded in T003. Changed only those two ASCII path byte sequences (retaining original line endings) and supplied `sample_prompts.txt` with the specified ceramic-cup sentence. Dataset bytes untouched; no optimizer, precision, cadence or hyperparameter substitution.

### T021 — effective template result

Passed: both complete supplied-template cases (2 passed, 23 deselected), production TOML merge/effective type checks, Dev dataset sanitizer/blueprint, strict prompt loader, both adapter spellings, actual bitsandbytes optimizer-class signature and TensorBoard backend validation. Temporary copies substitute only external resources/destinations; rank/alpha/dropout/precision/optimizer/LR/accumulation/events remain exactly as supplied. Processor boundary is explicitly isolated and real checkpoint loaders fail the test if called. This proves parsing/preflight, not training or AdamW8bit kernel execution. T018 additionally passed ordinary Adafactor, SGD/StepLR, dotted AdamW, extra LoRA options and Dev memory/numeric settings.

### T022 — data and cache comparisons

Existing workflow/image cases plus new gaps: initial 30 passed, 1 fixture failure because two control entries referenced the same tensor storage, which safetensors correctly rejects. Corrected only fixture allocation to independent tensors; all three new cases passed on repeat. Tests cover sorted source/caption lookup, dataset/source indices, repeats and control-count bucket separation, cwd-first/JSONL-directory/missing image paths, and Mistral production forward selection `[10,20,30]` with padding 512/output 15360 using synthetic hidden states (no model weights). Existing exact cache filenames/keys/metadata/dtype/NaN/merge/skip/keep comparisons passed. Thread completion order is not treated as a source-order guarantee; item/index associations are compared independently.

### T023 — numerical/adapter comparison

Passed: **62 CPU tests** across workflow, original precision/dtype/gradient suites and retained Ideogram/Krea numeric timestep suites. Added missing Dev-factory rank=alpha=32/dropout=.05 forward/RNG/gradient/group-LR and enable/bypass/restore checks using a tiny linear block; one scalar accumulated SGD update verifies gradient 15, parameter -0.5, then scheduler LR .05. These CPU optimizer checks do not substitute for or validate CUDA AdamW8bit. Existing T004 fixed-input packing/noise-target/loss/sample assertions and tolerances were preserved.

Source comparison supplementary evidence: all 14 train_utils functions and all four timestep functions remain AST-identical; 60 common trainer functions and 27 common LoRA functions unchanged. Changed optimizer/scheduler bodies only use split('=',1) instead of split('=') for validated nested literal arguments. Training-loop changes are limited to the retained metadata guidance default, nested-argument splitting and early-validated tracker init kwargs. Backward/gradient clipping/optimizer/scheduler/update order remains unchanged. This audit is not an invented runtime test or a GPU equivalence claim.

### T024–T025 — artifacts and nine-capability assessment

Artifact run: 31 passed, 2 failed because the new reload assertion assumed the in-memory `alpha` buffer was float; original construction with integer alpha=32 creates an int64 buffer. The tests now explicitly require int64 alpha=32 after reload and exact FP32 parameter values converted from the stored precision. Production dtype/serialization unchanged. Re-run of the four touched artifact/log cases: **4 passed**. Expanded TensorBoard coverage writes the actual `generate_step_logs` output and checks its scalar names; added private validation/provenance fields are excluded from optional config logging so internal fields do not alter user-facing logged configuration.

| Retained capability | Local result and practical limit |
| --- | --- |
| Image/caption input | Passed production blueprint/datasource/bucket/order/index/repeat/control and JSONL fallback fixtures; temporary data only. |
| VAE latent caching | Passed original preprocessing, exact cache files/keys/metadata/dtypes/NaN and callbacks; no actual AE encoding. |
| Mistral output caching | Passed writer/merge/collation/skip/keep and synthetic production-forward layer/512/15360 contract; real cached processor probe pending T032, no Mistral weights. |
| LoRA training | Passed real template/nested selection and tiny CPU math/dropout/dtype/gradient/accumulation/scheduler fixtures; no trainer loop or CUDA AdamW8bit evidence. |
| LoRA saving | Passed exact adapter keys, FP32/BF16 serialization, metadata, hashes and reload; existing save-precision assertions retained. |
| Full-state saving | Passed Accelerate CPU toy state/hook contents and state names/retention/train-end helpers. |
| Resume | Passed existing toy state restoration including optimizer/LR/RNG. **Acceptance concern remains open**: `_run_training_loop` resets epoch/global_step after state load; no full progress restoration/equivalence claim. |
| Training samples | Passed baseline PNG pixels/path/defaults/first+250 triggers/tracker calls, train/eval/swap and RNG restoration. |
| Logging | Passed actual scalar generation and real TensorBoard event round-trip; optional WandB service operation not run. |

All original precision/dtype/gradient tests retain their meaningful assertions. Source review found no retuned hyperparameters, objective, shared timestep math, backward/update order or cache migration. The known resume discrepancy and optional pre-existing dim_from_weights defect are not cleanup regressions and are not repaired by this initial scope. Overall acceptance remains open despite completed local evidence tasks.

### T026–T027 — dependency/test cleanup

Removed only exclusive declared av/ftfy/easydict/prompt-toolkit and GUI/HiDream extras after zero retained imports were found (rg exit 1 means no matches). Preserved runtime pins, CUDA extras/index definitions, build setup and shared dev packages. Removed Ruff entries only for deleted paths. No user-environment packages uninstalled.

Exact D3 disposition: **29 tracked files deleted**, after validating each absolute path was inside this repository and present in the tracked inventory. Kept all common numeric tests in `test_ideogram4_timesteps.py` and `test_krea2_timesteps.py`; narrowed `test_minimax_h3_te_streaming.py` to four unchanged common CPU assertions and their fixtures. Dataset index/skip/keep, literal prompt, relative paths, common FP8 arithmetic (exact TE equality and existing 2e-2 DiT tolerance), and LoRA enable/bypass/restore were preserved in focused workflow tests. No excluded model loader/teacher objective was ported. Original save/dtype/gradient assertions intact. **38 focused tests passed**; full remaining suite collection: **204 collected**, zero missing/deleted imports. Collection is not a test pass.

### T028 — retained guidance

Reconciled all seven retained docs with the Dev image path, strict input sources, processor/shards, both caches, unchanged templates, samples/logging/save-state/retention and known resume limitations. Preserved shared block-swap attribution to 2kpr, common FP8 attribution to HunyuanVideo/diffusion-pipe, sampler attribution and shared distribution figure. D4's standalone converter/caption/merge/EMA/LoHa/LoKr and other-model instructions have no retained operational consumer. Common image controls, base-LoRA initialization, offloading and numeric guidance now live in the retained docs. README setup/translations follow in T030. No operational command was executed.

### T029–T030 — docs/assets and README

Deleted exact D4's **27 tracked files**, with resolved in-repository path and tracked-file checks before removal. Shared figures and AiHUB logo remain. Replaced all three README variants with matching Dev-only scope, setup prerequisites, repository-root workflow, unchanged template values, cache/sample/log/state distinctions and explicit known limitations. Operational examples are labelled separately from CPU local checks. Attribution for retained FLUX/Diffusers/common helpers, upstream sponsor and contributor/agent instructions remains; contribution files, CI and developer tooling are unchanged. Final link/option audit follows in T033.

### T031 — final local suite and style

Final command (Python 3.12.14, `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `PYTHONIOENCODING=utf-8`): `.venv/Scripts/python.exe -m pytest -p no:cacheprovider -q --tb=short tests`. **208 passed**, one existing einops invalid-escape SyntaxWarning, exit 0, 104.02 seconds. Includes all six root/package `--help` invocations and all retained tests; zero skips/xfails. No active template option was changed to pass.

Initial Ruff reported 12 unused imports left by removal and 25 files needing formatting. Removed those unused imports and applied existing Ruff formatting to changed retained Python only. Final `ruff check` and `ruff format --check`: **passed**, all 30 changed/new Python files formatted. AST of all 54 remaining source/test/root Python files passed; project/template TOML syntax passed. Final contract review added early checks for an absent attention backend, selected-backend precedence, existing H2D checkpointing/ring requirements and compile backend lookup (no compilation), and rejects rank_dropout=1 before its division-by-zero path. Those four additional invalid cases passed with zero weight-loader calls. No backend/optimizer was substituted.

### T032 — actual dependency closure and cached processor probe

Python 3.12.14 CPU/offline probe imported actual torch/torchvision, accelerate, bitsandbytes, diffusers, einops, huggingface_hub, cv2, PIL, NumPy, packaging, safetensors, toml, tqdm, transformers, voluptuous, sentencepiece, TensorBoard, Mistral3Config/AutoProcessor and Mistral3ForConditional Generation/SummaryWriter and all three retained package entrypoints: **passed**. To verify closure in the prepared environment without uninstalling user packages, a deny-only import finder made av/ftfy/easydict/prompt_toolkit/gradio/sensecraft unavailable; availability queries returned None and attempted imports would raise ModuleNotFoundError. No retained import was replaced with a permissive stub. All mandatory imports still passed. This proves import closure with those modules inaccessible, not a new installation/build. `python -m pip check`: **passed**, No broken requirements found.

Actual `AutoProcessor.from_pretrained('mistralai/Mistral-Small-3.1-24B-Instruct-2503', use_fast=False, local_files_only=True)` with `HF_HUB_OFFLINE=1`: **blocked**, OSError reporting the files were not found in the cache. Probe exited 3 deliberately; tokenization did not run and is not counted as passed. No online retry/resource download. Optional WandB is not installed; its service and actual positive runtime initialization contract are **not run**, not inferred from TensorBoard checks.

### T033 — final requirements/code/README reconciliation

Reviewed all three README variants, seven retained docs and supplied configuration against final parser/resource/factory/sampling/state code and FR-005–008/014/017. **83 internal file links across 12 README/doc/contributor files resolve**. Supplied dataset/prompt references resolve from root. All README variants distinguish external checkpoint/shard/processor/image prerequisites, both caches, unchanged mandatory settings, samples/logging, LoRA vs state and resume limitations. Removed a stale inherited `.ai` setup claim after confirming that directory is absent in this checkout; existing `.agents/skills`, personal instructions and contributor/CI files are untouched. Historical quickstart text describes the planning snapshot; current READMEs and this evidence log explicitly reflect implemented preflight and available runtime.

### T034 — protection audit performed; acceptance remains open

HEAD remains `bf478828ffa8b5ef3ebdf225ee72431bdb463159`; no commit/reset/branch operation. Exact dispositions: D1 **115 absent**, D2 **134 absent**, D3 **29 removed + 3 common candidates retained**, D4 **27 absent**. No untracked user file was deleted. Actual retained imports have no missing local owner. Final root/package commands are exactly the three Dev operations; shared helper/developer modules do not expose an excluded model workflow.

Protection hashes match T001: dataset `13eef91db4f6747d8fbd30f164e6042cee2431661d6ef3dd669a4664eae2ee2f`, constitution `6725e90c675a7ce188d6d218096155fa4a3d465c87f1fac3be8190cdb2b6fbcb`, unrelated user command file `04eed9a22e09a97d4d3818827fa9e48133249140aabfeda644e92b02d10d52ad`. Current training template hash `7efcbd7c8e1a761c0850f6e21e35e43313c0c550d001b8c4758f6f18c8f191bf`; reversing exactly the two authorized path byte replacements restores the original T001 hash, proving all other bytes/comments/values unchanged. Model/data/output/log directories remain absent as at entry; no user resource was traversed or edited. Constitution/Spec Kit/agent/cursor/contributor/CI tracked files have zero changes. AiHUB logo and shared timestep figures remain; retained source copyright notices preserved.

| Acceptance area | Final local status |
| --- | --- |
| SC-001 removal/scope, FR-001–003/021 | Passed execution/factory/map/import and exact inventory audit. |
| SC-002 template compatibility, FR-005–008 | Passed all 42 active values, dataset values, byte protection and supplied references; external resources remain external. |
| SC-003 complete workflow, FR-004/010 | Partial: all nine local helper contracts recorded in T025; real processor/tokenization blocked and complete resume acceptance unresolved. |
| SC-004 early sources, FR-011/012 | Passed local production entry/source rejection and loader-sentinel matrix, including actual selected dependency failures; optional unavailable backend evidence is not claimed. |
| SC-005 preservation, FR-009/010/016/017 | Passed available CPU comparisons and diff review; known pre-existing resume and optional dim_from_weights defects remain visible, without equivalence claims. |
| SC-006 focused repository, FR-013/014 | Passed deletion, docs/links and actual mandatory dependency import closure. |
| SC-007 protection, FR-015/019 | Passed hash, exact deletion and protected-tooling audit. |
| FR-020 historical specify restriction | No spec/plan/constitution rewrite; implementation is the separately authorized invocation. |

**Final local-stage limits (FR-018):** no real training/full caching run, GPU execution, weight/resource download, package build, server transfer or server verification was performed. Only the separately user-authorized dependency installation used prebuilt CPU wheels. Full-model numerical/quality/hardware acceptance is not established. No convergence workflow or new correction round was launched. `.specify/extensions.yml` is absent; no after_implement hooks are registered to dispatch.

T001–T033 execution/evidence work is recorded, including blocked/not-run subchecks. **T034 is intentionally unchecked** because the required final acceptance is not closed: missing actual processor resources and known pre-existing resume acceptance gap cannot be resolved within this local cleanup scope. They are not waived by documentation or by the 208 passing local tests.

## Approved correction round 1 — 2026-09-21

Authorization: T035, T036, T037, T038, T039 and T040, plus the user's explicit `dim_from_weights` and progress-preserving `resume` repairs. No other task, second correction round or converge is authorized. Earlier prohibitions on silently repairing these two baseline defects remain historical context; this explicit request authorizes those repairs. Requirements, design documents and Constitution 1.0.0 were not rewritten.

Source HEAD remains `bf478828ffa8b5ef3ebdf225ee72431bdb463159`, with the prior implementation's uncommitted working tree preserved. Interpreter: `C:\Users\inbox\Desktop\musubi-tuner-flux2dev-lora\.venv\Scripts\python.exe`, Python 3.12.14 / torch 2.7.1+cpu / Accelerate 1.6.0. All pytest commands use `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `PYTHONIOENCODING=utf-8`, `-p no:cacheprovider -q --tb=short`.

### Changes within the approval

- **T035:** Existing preflight rejects constant-scheduler warmup, invalid SGD Nesterov coupling (also the dotted torch selector), impossible Dev block swapping and out-of-range timestep indices. Diagnostics retain source, key/value, cause and correction; invalid cases never reach a weight loader. Valid boundaries remain accepted, including 29 swapped blocks, the documented timestep endpoints, fixed timesteps and fractional warmup that rounds to zero. Float warmup with an epoch-derived target retains the existing later check because the target is not yet known at this boundary.
- **T036:** Optimizer, scheduler, network and network-metadata consumers now use the same existing nested-argument parser as validation. Whitespace around assignment names no longer changes or silently drops the effective option. Actual tiny AdamW/StepLR/Dev-adapter consumers verify canonical names and literal values.
- **T037:** Removed Conv3d discovery, creation, 5D rank-dropout broadcasting and 5D merge/materialization from the common adapter. Linear/Conv2d behavior, retained LoRA precision/gradient checks and singleton sample frame axes remain.
- **T038:** Structured JSON/TOML control strings normalize to a one-element list. Production sampling reaches control preprocessing with complete paths, preserves list order and torch RNG, and touches no real loader. Existing repeated TXT controls remain covered.
- **T039:** Removed only unused `build_merged_from`, its exclusive metadata reader and unused imports; ordinary metadata/hash/save functions remain covered.
- **T040:** The advanced guide identifies `logsnr2` as an internal Qinglong component, not a selectable CLI mode. README links and retained choices were reconciled without adding a sampling mode.
- **`dim_from_weights`:** `_build_network` reads `network_weights`, accepts the FLUX.2 factory's single network return, then loads adapter tensors through the existing loader. A real tiny safetensors fixture with rank 3 / alpha 6 verifies rank, scale and every tensor despite different configured dimensions; the no-flag configured construction remains covered.
- **`resume`:** Existing save/load hooks add/read `training_progress.pt`; the existing Accelerate weight/optimizer/scheduler/RNG files remain unchanged. Progress records epoch, next batch, global update, loader settings/target, epoch/sampler RNG, deterministic dataset shuffle seeds/epochs, timestep-bucket pool and moving loss. Existing Accelerate batch skipping recreates the saved permutation without reading completed caches, and the returned loader retains persistent workers for later epochs. Resuming an epoch boundary preserves the RNG draws needed for the next iterator. State saving follows the corresponding log/sample events, while optimizer train/eval transition counts and optimization math remain unchanged. New tracker initialization is isolated from restored training RNG; initial samples are not repeated. Final-step states finish pending epoch events once; finished states perform no additional updates. Incompatible counters/loader settings or malformed progress fail explicitly.

**Legacy compatibility:** An absent sidecar preserves the former state-loading behavior and emits an explicit warning that epoch/global_step and data restart at zero while restored optimizer/scheduler/RNG continue from the old state. No position is invented from checkpoint names; exact continuation is not claimed for that legacy format. A malformed sidecar is an error, not legacy fallback. Existing LoRA and Accelerate artifacts require no migration.

### Test evidence and intermediate failures

1. Before T035 fixes, `tests/test_flux2_scope.py -k round1`: **8 failed, 5 passed**. All eight invalid cases reached the real loader sentinel, reproducing the missing early checks; no real weight was loaded.
2. During staged edits, the new focused workflow subset reported **1 failed, 8 passed**: the Conv3d exclusion still discovered the volume adapter. This was an intermediate run, not a pristine before-fix baseline. The first six-file aggregate then reported **55 failed, 154 passed** because the new swap guard did not handle the existing `None` default. After correcting that guard and completing T037, the same affected files passed: **209 passed**, 20.20 s. These intermediate failures are retained here rather than replaced with the success result.
3. The first CPU continuation fixture failed because its tiny trainer lacked the Dev model-info field; the fixture now supplies the actual Dev info record. Subsequent checks exposed Accelerate 1.6's empty skipped-shard iteration bug and a boundary RNG restoration error. The helper now primes an empty base loader without its shard prefetch, and does not undo legitimate new-epoch RNG consumption. After these fixes, the continuation/legacy subset passed: **4 passed**, 84.94 s. No numerical tolerances were relaxed.
4. Final full retained command: `.venv/Scripts/python.exe -m pytest -p no:cacheprovider -q --tb=short tests`: **234 passed**, 175.83 s, exit 0. This includes the strengthened tracker-startup RNG test, all six root/package `--help` calls and the original precision/gradient/metadata/cache tests. One pre-existing third-party einops invalid-escape SyntaxWarning remains; no skips or xfails.
5. Final T035 compatibility review preserved the existing fixed-timestep case and fractional-warmup rounding. After those last narrow changes, `.venv/Scripts/python.exe -m pytest -p no:cacheprovider -q --tb=short tests/test_flux2_scope.py -k round1`: **15 passed, 121 deselected**, 13.85 s, exit 0. This is a targeted rerun after the full-suite result, not a claim that the full suite was rerun again.

The continuation regression uses five tiny cached items in two production image datasets, rank-32 toy Linear LoRA with dropout, six CPU AdamW updates, accumulation 2 and a linear scheduler. It exercises 0 workers, 1 ordinary worker and 2 persistent workers. For each, resume is compared from a mid-epoch state, an accumulation-remainder/end-of-epoch step state, a second-epoch state, an epoch state, the final-step state and a finished state. Exact equality (`rtol=atol=0`) is required for parameters, optimizer, scheduler, Python/NumPy/torch RNG, moving losses, timestep pool and counters. Remaining batch records, sample/log/save event sequences and no duplicate initial sample are checked. These are explicitly requested CPU unit tests through the production loop with a toy forward, not operational training of FLUX.2.

### Final reconciliation and stop boundary

Passed: Ruff check and format-check on the 9 touched Python files, AST parsing of all 54 retained source/test/root Python files, project and both template TOML parses, and all **83 internal file links in 12 README/documentation/contributor files**. The three READMEs and `docs/flux_2.md` explain new-state continuation, explicit legacy behavior, initial adapter loading and CPU evidence limits. `docs/advanced_config.md` matches the CLI choices. Existing ignore patterns cover the Python environment/artifacts; no ignore-file change was necessary for this round.

Protection hashes match the round entry and prior evidence: constitution `6725e90c675a7ce188d6d218096155fa4a3d465c87f1fac3be8190cdb2b6fbcb`; training template `7efcbd7c8e1a761c0850f6e21e35e43313c0c550d001b8c4758f6f18c8f191bf`; dataset template `13eef91db4f6747d8fbd30f164e6042cee2431661d6ef3dd669a4664eae2ee2f`; supplied prompt `b6403749e91f000c6c84f81b3bfb41c4c98d39c42d47f3ea52b4b64de7db5129`. No training hyperparameter/template edit, dependency change, file deletion, commit or protected-tooling edit was made in this round.

**Round limits:** No real-model training/full caching, GPU run, resource/weight download, installation, packaging, server transfer or server verification was performed. Real-model/GPU continuation is not established by the CPU results. T034 remains unchecked and outside this approval; its prior actual offline Mistral processor probe is still **blocked** by missing cached resources and was not reclassified as passed or retried in this round. `.specify/extensions.yml` is absent, so no before/after implement hooks exist. T035-T040 and the two explicitly added repairs are complete; no second round or converge was started.
