# Musubi Tuner — FLUX.2 Dev LoRA

[English](README.md) · [Русский](README.ru.md)

このフォークは **FLUX.2 Dev の画像LoRA** に限定しています。VAE潜在表現のキャッシュ、Mistralのキャプション出力キャッシュ、LoRA学習の3コマンドを提供します。ルートのスクリプトと `python -m musubi_tuner.<拡張子なしのスクリプト名>` は同じ処理です。Kleinを含む他モデル、フル学習、Self-Flow、LoHa/LoKr/LyCORIS、GUI、単独生成、キャプション生成、変換、merge/export、post-hoc EMAは削除されています。学習中のPNGサンプルと内部の標準LoRA処理は残っています。

## 環境と必要なファイル

Python 3.10–3.12、互換性のあるPyTorch/torchvision、およびBF16/AdamW8bitに対応する実行環境が必要です。以下は別途準備する実運用環境のCUDA 12.8設定例であり、ローカル検証で実行するコマンドではありません。ほかのCUDAオプションは [pyproject.toml](pyproject.toml) を参照してください。

```sh
python -m venv .venv
# Activate .venv before the following commands.
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -m pip install tensorboard pytest "ruff>=0.12.10,<0.16"
python -m pip check
```

Windowsでは `.\.venv\Scripts\Activate.ps1`、POSIXでは `source .venv/bin/activate` で有効化します。コンソール画像表示とtimestepグラフには既存の `ascii-magic==2.3.0`、`matplotlib==3.10.0` を使用します。追加のバックエンドやトラッカーは選択時に準備してください。bitsandbytes/TensorBoard/選択した依存関係が欠けている場合、設定を置換せず重みの読み込み前にエラーになります。

元形式のDev DiTとAEチェックポイント、最初の `model-00001-of-00010.safetensors` と同じ場所にある **全10個のMistral shard**、および `mistralai/Mistral-Small-3.1-24B-Instruct-2503` のprocessor/tokenizer/chat templateキャッシュが必要です。processorは通常のHugging Faceキャッシュからオフライン検証され、重みのパスとは独立しています。`tokenizer_path` オプションはありません。[リソースとパス](docs/flux_2.md)。

## 実行手順

リポジトリのルートから実行します。[train.toml](flux2dev_lora/train.toml)、[dataset.toml](flux2dev_lora/dataset.toml)、[sample_prompts.txt](flux2dev_lora/sample_prompts.txt) は相互に整合しています。外部リソースのパスを環境に合わせて変更し、画像と同名のUTF-8 `.txt` キャプションを用意してください。キャッシュ、出力、ログのディレクトリには書き込み権限が必要です。

以下は **実際の重みを読み込む運用コマンド** です。学習前に両方のキャッシュを作成します。

```sh
python flux_2_cache_latents.py --model_version dev --dataset_config ./flux2dev_lora/dataset.toml --vae ./models/flux2-dev/ae.safetensors --vae_dtype float32
python flux_2_cache_text_encoder_outputs.py --model_version dev --dataset_config ./flux2dev_lora/dataset.toml --text_encoder ./models/flux2-dev/text_encoder/model-00001-of-00010.safetensors
accelerate launch --num_processes 1 --mixed_precision bf16 flux_2_train_network.py --config_file ./flux2dev_lora/train.toml
```

テンプレートはrank=alpha=32、dropout .05、BF16、FP32 AE、SDPA/checkpointing、FP8無効、`flux2_shift`、AdamW8bit 1e-4、warmup 100、2000更新、batch 1/accumulation 4、seed 42、persistent worker 2のままです。`blocks_to_swap=20` はコメントのままです。追加のDev用設定と両方のLoRAモジュール名も保持されています。不明なキーや削除対象の設定は重みより前に拒否されます。

LoRAと完全な状態は250更新ごと、保持期間はそれぞれ1000更新です。PNGサンプルは開始時と250更新ごとに `<output_dir>/sample` に保存します。付属promptの既存デフォルトは256×256、20 steps、guidance 4です。TensorBoardの元のprefix/トラッカー名と `./logs/flux2_dev_style` を保持します。表示には `tensorboard --logdir ./logs/flux2_dev_style` を使用します。

LoRAの `.safetensors` とAccelerateの完全状態ディレクトリは別の成果物です。`--resume` は新しい状態からepoch、更新回数、データ位置を復元し、元の設定で残りの更新を続けます。進捗情報のない旧状態も読み込めますが、従来どおりカウンターとデータを最初から開始する旨の警告が出ます。`--dim_from_weights --network_weights <file>` はrank、alpha、アダプター重みを復元します。[詳細](docs/flux_2.md)。

## ガイドとローカル検証

- [Devの全ワークフロー](docs/flux_2.md)
- [画像データセットと制御画像](docs/dataset_config.md)
- [高度な設定とトラッカー](docs/advanced_config.md)
- [学習中のサンプル](docs/sampling_during_training.md)
- [Block swap](docs/block_swap.md)、[torch.compile](docs/torch_compile.md)

準備済みCPU環境で `PYTHONPATH=src`、`PYTHONDONTWRITEBYTECODE=1`、`CUDA_VISIBLE_DEVICES=-1`、`HF_HUB_OFFLINE=1` とUTF-8出力を設定します。

```sh
python flux_2_cache_latents.py --help
python flux_2_cache_text_encoder_outputs.py --help
python flux_2_train_network.py --help
python -m pytest -p no:cacheprovider -q tests
```

テストには6通りのroot/module `--help`、小さいCPUテンソル、一時ファイルを含みます。実学習、GPU、重み/リソースのダウンロード、パッケージビルド、サーバー検証はローカル段階では実行しません。実際の結果と利用できない検証は [baseline.md](specs/001-scope-flux2-dev-lora/baseline.md) に記録しています。

## 開発と帰属

[CONTRIBUTING.ja.md](CONTRIBUTING.ja.md) と [English guide](CONTRIBUTING.md)、Ruff設定、既存の `.agents/skills`、個人のエージェント設定、Spec Kit、CIを保持しています。以前のREADMEに記載されていた `.ai` はこのcheckoutには存在せず、これらのコマンドには不要です。過去のモデル名だけを理由に共通の数値テストを削除しません。

Musubi Tuner by [kohya-ss](https://github.com/kohya-ss/musubi-tuner). Retained FLUX code derives from [Black Forest Labs](https://github.com/black-forest-labs/flux); some common code is copied/modified from Diffusers. Retained code uses Apache-2.0 notices in its sources; model/resource licenses are separate. Shared FP8/offloading credits remain in the [advanced](docs/advanced_config.md) and [block-swap](docs/block_swap.md) guides. This fork is unofficial and is not affiliated with model authors.

Upstream sponsor: [AiHUB](https://aihub.co.jp/top-en).

[![AiHUB](images/logo_aihub.png)](https://aihub.co.jp/top-en)

[Support upstream development](https://github.com/sponsors/kohya-ss/).
