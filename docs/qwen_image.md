# Qwen-Image original adapter training

This repository trains original Qwen-Image adapters with the existing Musubi Tuner engine. The supported workflow is captioned images → latent cache → caption embedding cache → adapter training with optional sample PNGs. Use the [README](../README.md) and the three [templates](../config_for_qwen_image_lora/train.toml).

Run from the repository root. Replace all `/srv/...` paths with your inputs; paths are resolved from process CWD, including paths inside TOML and JSONL.

```bash
python qwen_image_cache_latents.py --dataset_config config_for_qwen_image_lora/dataset.toml --vae /srv/models/qwen_image_vae.safetensors --model_version original
python qwen_image_cache_text_encoder_outputs.py --dataset_config config_for_qwen_image_lora/dataset.toml --text_encoder /srv/models/qwen_2.5_vl_7b.safetensors --model_version original
accelerate launch --mixed_precision bf16 qwen_image_train_network.py --config_file config_for_qwen_image_lora/train.toml
```

The same commands are available as `python -m musubi_tuner.qwen_image_cache_latents`, `python -m musubi_tuner.qwen_image_cache_text_encoder_outputs`, and `python -m musubi_tuner.qwen_image_train_network`. The package must be installed or `src` included in `PYTHONPATH`.

Use the original DiT, RGB VAE and Qwen2.5-VL weights. The text loader obtains tokenizer assets from `Qwen/Qwen-Image`, subfolder `tokenizer`; prepare them in the operational environment. DiT execution uses bf16. The training VAE dtype default is bf16; the Qwen VAE loader retains its existing loading behavior. `num_layers` defaults to 60 and must match the checkpoint.

The supplied paths, H200 comment, rank 16, 1600 updates and batch/resolution values are examples, not fixed requirements. Defaults are overridden by training TOML and then explicit CLI. Omitted CLI flags preserve TOML; store-true flags cannot disable a true TOML setting. Unknown fields/types, excluded selectors and invalid effective values fail early. The only model version is `original`. [Alternative adapters](loha_lokr.md) retain the Qwen model engine.

Both cache commands have independent CLI settings and never read training TOML. `batch_size` caps encoding chunks; the dataset declaration controls training batch size. `num_workers` must be positive when explicitly set. `skip_existing` checks existence only; rebuild affected caches after changing images, captions, weights or relevant settings. `keep_cache` preserves stale files that normal cache cleanup removes. Latent cache debug supports `image` and `console`; Qwen latent caching does not accept an explicit `vae_dtype` or tiling/chunk options.

Latent caches preserve `[C,1,H,W]`; the singleton axis is required by the original VAE. Text caches keep variable-length `[L,D]` embeddings. Training pads text and preserves masks/split attention. Missing text caches are warned and skipped; an empty effective training dataset fails before models.

See [datasets](dataset_config.md), [sampling](sampling_during_training.md), [advanced options](advanced_config.md), [block swap](block_swap.md) and [compilation](torch_compile.md). No standalone inference, full finetuning, editing, layered, audio or video workflow is shipped.

## 日本語

元のQwen-Imageのアダプター学習のみを対象とします。画像とキャプションを用意し、latentとテキスト埋め込みの両方をキャッシュしてから学習します。上記コマンドはリポジトリのルートで実行し、外部パスを置き換えてください。相対パスの基準は設定ファイルの場所ではなく、実行時の作業ディレクトリです。実モデルでの動作確認は別の運用段階です。
