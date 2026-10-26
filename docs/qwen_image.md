# Qwen-Image original adapter training

This repository trains original Qwen-Image LoRA/LoHa/LoKr adapters with the existing Musubi Tuner engine. The workflow is captioned train/val images → latent caches → caption embedding caches → adapter training with deterministic val-loss and optional sample PNGs. Use the [README](../README.ru.md), [training profile](../config_for_qwen_image_lora/train.toml), [train dataset](../config_for_qwen_image_lora/train-dataset.toml) and [val dataset](../config_for_qwen_image_lora/val-dataset.toml).

Prepare independent, nonempty train and val folders with image/text pairs. There is no automatic split; image contents must not overlap. The shipped profile uses 1024×1024 buckets, train batch=16, val batch/repeats=1, LoRA rank/alpha=32, 5000 total updates and integer warmup=100. These are configurable values. CPU checks do not establish that the supplied resources exist or batch=16 fits a GPU.

The supplied `train.toml` enables `auto_cache=true`: the normal training launch prepares missing train/val latent and text caches before loading the training model. Complete valid caches skip encoder/tokenizer preparation entirely; partial caches only create missing files. Encoder stages run sequentially in separate processes. Needed encoder files and the native Qwen tokenizer are checked before the first stage. Sampling-model requirements are unchanged.

Automatic preparation requires `experiment_mode=true` and one training process. It defaults to false in other profiles. Use `--no_auto_cache` or `auto_cache=false` for manual/distributed workflows; `--auto_cache` enables it explicitly. Both CLI flags override TOML and are mutually exclusive. Existing files must satisfy source/provenance/geometry/caption checks; invalid or unprovenanced files are reported rather than overwritten. Unrelated files are retained, and unexpected train latent caches are rejected. Failed preparation stops before training and keeps completed files for retry. Encoder-checkpoint identity is not newly tracked by this feature.

Run the normal automatic launch from the repository root, using your model paths in `train.toml` and an installed package or `src` in `PYTHONPATH`:

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 qwen_image_train_network.py --config_file config_for_qwen_image_lora/train.toml
```

For manual preparation or deliberate regeneration after correcting an invalid cache, the four independent commands remain available. Replace the model paths with your files:

```bash
export PYTHONPATH="$PWD/src"
python qwen_image_cache_latents.py --dataset_config config_for_qwen_image_lora/train-dataset.toml --vae /workspace/models/qwen_image_vae.safetensors --model_version original --experiment_mode
python qwen_image_cache_latents.py --dataset_config config_for_qwen_image_lora/val-dataset.toml --vae /workspace/models/qwen_image_vae.safetensors --model_version original --experiment_mode --validation
python qwen_image_cache_text_encoder_outputs.py --dataset_config config_for_qwen_image_lora/train-dataset.toml --text_encoder /workspace/models/qwen_2.5_vl_7b.safetensors --model_version original --experiment_mode
python qwen_image_cache_text_encoder_outputs.py --dataset_config config_for_qwen_image_lora/val-dataset.toml --text_encoder /workspace/models/qwen_2.5_vl_7b.safetensors --model_version original --experiment_mode --validation
accelerate launch --mixed_precision bf16 qwen_image_train_network.py --config_file config_for_qwen_image_lora/train.toml --no_auto_cache
```

The three entrypoints also work as `python -m musubi_tuner.<command_name>`. Cache commands accept their own CLI settings, including `--experiment_mode` and `--validation`; they do not read training TOML or accept `--config_file`. With experiment mode, training paths resolve against `train.toml`, dataset source/cache paths against their dataset TOML, and JSONL image paths against their JSONL file. Relative cache-command model paths resolve against the dataset TOML. Absolute paths remain unchanged; without experiment mode, relative paths retain process-CWD semantics. The [README](../README.ru.md) includes a launch from another working directory.

Use the original DiT, RGB VAE and Qwen2.5-VL weights. The text loader obtains tokenizer assets from `Qwen/Qwen-Image`, subfolder `tokenizer`; prepare them in the operational environment. DiT execution uses bf16. The training VAE dtype default is bf16; the Qwen VAE loader retains its existing loading behavior. `num_layers` defaults to 60 and must match the checkpoint. Model version is `original` only.

Defaults are overridden by training TOML and then explicit CLI. Omitted CLI flags preserve TOML; ordinary store-true flags cannot disable a true TOML setting, while `auto_cache` has its explicit `--no_auto_cache` counterpart. Unknown fields/types, excluded selectors and invalid effective values fail early. [Alternative adapters](loha_lokr.md) retain the Qwen model engine.

Cache CLI `batch_size` caps encoding chunks; the dataset declaration controls training batch size. Explicit `num_workers` must be positive. Without `--validation`, `skip_existing` checks existence; with it, existing caches must have current valid source/preparation provenance. `keep_cache` preserves stale files that normal cleanup removes. Latent debug supports `image` and `console`; Qwen latent caching does not accept explicit `vae_dtype` or tiling/chunk options.

Latent caches preserve `[C,1,H,W]`; the singleton axis is required by the original VAE. Text caches keep variable-length `[L,D]` embeddings. Training pads text and preserves masks/split attention. Legacy manual training warns and skips missing text caches. Automatic preparation inventories every source image before cached training buckets exist and rejects missing captions in both roles. Enabled validation still requires complete, valid prepared caches and checks associations before model loading; missing files can now be prepared by the preceding automatic stage. Validation itself never skips images or regenerates caches. The strict resume fingerprint check still rejects changed caches. Empty effective datasets are errors.

`val_dataset_config` enables validation independently of experiment mode. Default interval/seed/N1/N2 are 50/42/10/2. Fixed noise and timestep evaluation reuses the train loss and reports `val_loss_mean`, `val_loss_low_noise`, `val_loss_high_noise` at step zero, configured completed optimizer steps and the final step. It preserves training RNG and module modes. N1 must be even and at least 2; see [advanced options](advanced_config.md) for the precise math.

In experiment mode, each checkpoint is `output/<output_name>-step<actual_optimizer_step>/`: one `model.safetensors`, optional complete Accelerate state plus `trainer_state.json` and a completion manifest, and `samples/`. Resumable state requires FP32 adapter saving. Weights-only export honors `save_precision` but cannot resume. Sample-only folders cannot resume either. Periodic and final saves at the same step share the checkpoint.

Resume with `--resume output/<output_name>-step<step>` relative to `train.toml`. With validation or experiment mode, `max_train_steps` is the total target; the explicit optimizer step and the same TensorBoard run continue. The profile keeps an inclusive 1000-step weights window; an omitted/zero state window inherits it. Set `save_last_n_steps_state=200` for a shorter state window. Retention preserves samples. With both modes disabled, legacy checkpoint names, path semantics and resume behavior remain.

See [datasets](dataset_config.md), [sampling](sampling_during_training.md), [advanced options](advanced_config.md), [block swap](block_swap.md) and [compilation](torch_compile.md). No standalone inference, full finetuning, editing, layered, audio or video workflow is shipped.

## 日本語

元のQwen-Imageのアダプター学習のみを対象とします。独立したtrain/val画像とキャプションを用意してください。付属設定の`auto_cache=true`は、experiment modeの単一プロセス起動時に不足するキャッシュだけを自動生成します。手動の場合は`--no_auto_cache`と上記4回の処理を使用できます。experiment modeでは各設定ファイルとJSONLを基準に相対パスを解決し、従来モードでは作業ディレクトリを基準にします。手動のval準備では`--validation`を指定します。再開可能なstateにはFP32のアダプターが必要です。実モデルと元のbatch=16でのGPU確認はCPUテストとは別です。
