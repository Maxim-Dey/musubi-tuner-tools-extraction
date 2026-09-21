"""Common argument parser and config-file loader shared by all training scripts.

`setup_parser_common()` assembles the parser by calling category-specific
`_add_*_args` helpers. Each helper owns a cohesive slice of the CLI surface,
so adding / relocating an argument only requires editing one helper.
"""

import argparse
import logging
import os
import pathlib
import ast
import importlib
import inspect
import math
import sys

import toml
from accelerate.utils import DynamoBackend


logger = logging.getLogger(__name__)


def _int_or_float(value):
    if value.endswith("%"):
        try:
            return float(value[:-1]) / 100.0
        except ValueError:
            raise argparse.ArgumentTypeError(f"Value '{value}' is not a valid percentage")
    try:
        float_value = float(value)
        if float_value >= 1 and float_value.is_integer():
            return int(float_value)
        return float_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not an int or float")


def _add_general_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="using .toml instead of args to pass hyperparameter / ハイパーパラメータを引数ではなく.tomlファイルで渡す",
    )
    parser.add_argument(
        "--dataset_config",
        type=pathlib.Path,
        default=None,
        help="config file for dataset / データセットの設定ファイル",
    )


def _add_attention_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sdpa",
        action="store_true",
        help="use sdpa for CrossAttention (requires PyTorch 2.0) / CrossAttentionにsdpaを使う（PyTorch 2.0が必要）",
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        help="use FlashAttention for CrossAttention, requires FlashAttention / CrossAttentionにFlashAttentionを使う、FlashAttentionが必要",
    )
    parser.add_argument(
        "--sage_attn",
        action="store_true",
        help="use SageAttention. requires SageAttention / SageAttentionを使う。SageAttentionが必要",
    )
    parser.add_argument(
        "--xformers",
        action="store_true",
        help="use xformers for CrossAttention, requires xformers / CrossAttentionにxformersを使う、xformersが必要",
    )
    parser.add_argument(
        "--flash3",
        action="store_true",
        help="use FlashAttention 3 for CrossAttention; requires FlashAttention 3",
    )
    parser.add_argument(
        "--split_attn",
        action="store_true",
        help="use split attention for attention calculation (split batch size=1, affects memory usage and speed)"
        " / attentionを分割して計算する（バッチサイズ=1に分割、メモリ使用量と速度に影響）",
    )


def _add_compile_and_dynamo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (requires Triton) / torch.compileを有効にする（Tritonが必要）",
    )
    parser.add_argument(
        "--compile_backend",
        type=str,
        default="inductor",
        help="torch.compile backend (default: inductor) / torch.compileのバックエンド（デフォルト: inductor）",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",  # 学習用のデフォルト
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile mode (default: default) / torch.compileのモード（デフォルト: default）",
    )
    parser.add_argument(
        "--compile_dynamic",
        type=str,
        default=None,
        choices=["true", "false", "auto"],
        help="Dynamic shapes mode for torch.compile (default: None, same as auto)"
        " / torch.compileの動的形状モード（デフォルト: None、autoと同じ動作）",
    )
    parser.add_argument(
        "--compile_fullgraph",
        action="store_true",
        help="Enable fullgraph mode in torch.compile / torch.compileでフルグラフモードを有効にする",
    )
    parser.add_argument(
        "--compile_cache_size_limit",
        type=int,
        default=None,
        help="Set torch._dynamo.config.cache_size_limit (default: PyTorch default, typically 8-32) / torch._dynamo.config.cache_size_limitを設定（デフォルト: PyTorchのデフォルト、通常8-32）",
    )
    parser.add_argument(
        "--cuda_allow_tf32",
        action="store_true",
        help="Allow TF32 on Ampere or higher GPUs / Ampere以降のGPUでTF32を許可する",
    )
    parser.add_argument(
        "--cuda_cudnn_benchmark",
        action="store_true",
        help="Enable cudnn benchmark for possibly faster training / cudnnのベンチマークを有効にして学習の高速化を図る",
    )

    parser.add_argument(
        "--dynamo_backend",
        type=str,
        default="NO",
        choices=[e.value for e in DynamoBackend],
        help="dynamo backend type (default is None) / dynamoのbackendの種類（デフォルトは None）",
    )

    parser.add_argument(
        "--dynamo_mode",
        type=str,
        default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
        help="dynamo mode (default is default) / dynamoのモード（デフォルトは default）",
    )

    parser.add_argument(
        "--dynamo_fullgraph",
        action="store_true",
        help="use fullgraph mode for dynamo / dynamoのfullgraphモードを使う",
    )

    parser.add_argument(
        "--dynamo_dynamic",
        action="store_true",
        help="use dynamic mode for dynamo / dynamoのdynamicモードを使う",
    )


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
    parser.add_argument(
        "--max_train_epochs",
        type=int,
        default=None,
        help="training epochs (overrides max_train_steps) / 学習エポック数（max_train_stepsを上書きします）",
    )
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=8,
        help="max num workers for DataLoader (lower is less main RAM usage, faster epoch start and slower data loading) / DataLoaderの最大プロセス数（小さい値ではメインメモリの使用量が減りエポック間の待ち時間が減りますが、データ読み込みは遅くなります）",
    )
    parser.add_argument(
        "--persistent_data_loader_workers",
        action="store_true",
        help="persistent DataLoader workers (useful for reduce time gap between epoch, but may use more memory) / DataLoader のワーカーを持続させる (エポック間の時間差を少なくするのに有効だが、より多くのメモリを消費する可能性がある)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for training / 学習時の乱数のseed")
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", help="enable gradient checkpointing / gradient checkpointingを有効にする"
    )
    parser.add_argument(
        "--gradient_checkpointing_cpu_offload",
        action="store_true",
        help="enable CPU offloading of activation for gradient checkpointing / gradient checkpointing時に活性化のCPUオフロードを有効にする",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass / 学習時に逆伝播をする前に勾配を合計するステップ数",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="use mixed precision / 混合精度を使う場合、その精度",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=["float", "fp32", "fp16", "bf16"],
        help="precision for saving network weights, default: fp32 (the precision network weights are trained in)"
        " / ネットワークの重みを保存する際の精度、省略時はfp32（ネットワークの重みはfp32で学習されるため）",
    )


def _add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="enable logging and output TensorBoard log to this directory / ログ出力を有効にしてこのディレクトリにTensorBoard用のログを出力する",
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default=None,
        choices=["tensorboard", "wandb", "all"],
        help="what logging tool(s) to use (if 'all', TensorBoard and WandB are both used) / ログ出力に使用するツール (allを指定するとTensorBoardとWandBの両方が使用される)",
    )
    parser.add_argument(
        "--log_prefix", type=str, default=None, help="add prefix for each log directory / ログディレクトリ名の先頭に追加する文字列"
    )
    parser.add_argument(
        "--log_tracker_name",
        type=str,
        default=None,
        help="name of tracker to use for logging, default is script-specific default name / ログ出力に使用するtrackerの名前、省略時はスクリプトごとのデフォルト名",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="The name of the specific wandb session / wandb ログに表示される特定の実行の名前",
    )
    parser.add_argument(
        "--log_tracker_config",
        type=str,
        default=None,
        help="path to tracker config file to use for logging / ログ出力に使用するtrackerの設定ファイルのパス",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="specify WandB API key to log in before starting training (optional). / WandB APIキーを指定して学習開始前にログインする（オプション）",
    )
    parser.add_argument("--log_config", action="store_true", help="log training configuration / 学習設定をログに出力する")
    parser.add_argument(
        "--log_grad_metrics",
        action="store_true",
        help="log gradient norm metrics (grad/norm, grad/mean_norm, grad/max, pre-clipping) to the tracker."
        " Adds a small per-step GPU sync overhead"
        " / 勾配ノルムのメトリクス（grad/norm, grad/mean_norm, grad/max、クリッピング前）をトラッカーに出力する。ステップごとにわずかなGPU同期のオーバーヘッドが発生する",
    )


def _add_ddp_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ddp_timeout",
        type=int,
        default=None,
        help="DDP timeout (min, None for default of accelerate) / DDPのタイムアウト（分、Noneでaccelerateのデフォルト）",
    )
    parser.add_argument(
        "--ddp_gradient_as_bucket_view",
        action="store_true",
        help="enable gradient_as_bucket_view for DDP / DDPでgradient_as_bucket_viewを有効にする",
    )
    parser.add_argument(
        "--ddp_static_graph",
        action="store_true",
        help="enable static_graph for DDP / DDPでstatic_graphを有効にする",
    )


def _add_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sample_every_n_steps",
        type=int,
        default=None,
        help="generate sample images every N steps / 学習中のモデルで指定ステップごとにサンプル出力する",
    )
    parser.add_argument(
        "--sample_at_first", action="store_true", help="generate sample images before training / 学習前にサンプル出力する"
    )
    parser.add_argument(
        "--sample_every_n_epochs",
        type=int,
        default=None,
        help="generate sample images every N epochs (overwrites n_steps) / 学習中のモデルで指定エポックごとにサンプル出力する（ステップ数指定を上書きします）",
    )
    parser.add_argument(
        "--sample_prompts",
        type=str,
        default=None,
        help="file for prompts to generate sample images / 学習中モデルのサンプル出力用プロンプトのファイル",
    )


def _add_optimizer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="",
        help="Optimizer to use / オプティマイザの種類: AdamW (default), AdamW8bit, AdaFactor. "
        "Also, you can use any optimizer by specifying the full path to the class, like 'torch.optim.AdamW', 'bitsandbytes.optim.AdEMAMix8bit' or 'bitsandbytes.optim.PagedAdEMAMix8bit' etc. / ",
    )
    parser.add_argument(
        "--optimizer_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...") / オプティマイザの追加引数（例： "weight_decay=0.01 betas=0.9,0.999 ..."）',
    )
    parser.add_argument("--learning_rate", type=float, default=2.0e-6, help="learning rate / 学習率")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm, 0 for no clipping / 勾配正規化の最大norm、0でclippingを行わない",
    )


def _add_lr_scheduler_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="scheduler to use for learning rate / 学習率のスケジューラ: linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor, rex",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=_int_or_float,
        default=0,
        help="Int number of steps for the warmup in the lr scheduler (default is 0) or float with ratio of train steps"
        " / 学習率のスケジューラをウォームアップするステップ数（デフォルト0）、または学習ステップの比率（1未満のfloat値の場合）",
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=_int_or_float,
        default=0,
        help="Int number of steps for the decay in the lr scheduler (default is 0) or float (<1) with ratio of train steps"
        " / 学習率のスケジューラを減衰させるステップ数（デフォルト0）、または学習ステップの比率（1未満のfloat値の場合）",
    )
    parser.add_argument(
        "--lr_scheduler_num_cycles",
        type=int,
        default=1,
        help="Number of restarts for cosine scheduler with restarts / cosine with restartsスケジューラでのリスタート回数",
    )
    parser.add_argument(
        "--lr_scheduler_power",
        type=float,
        default=1,
        help="Polynomial power for polynomial scheduler / polynomialスケジューラでのpolynomial power",
    )
    parser.add_argument(
        "--lr_scheduler_timescale",
        type=int,
        default=None,
        help="Inverse sqrt timescale for inverse sqrt scheduler,defaults to `num_warmup_steps`"
        + " / 逆平方根スケジューラのタイムスケール、デフォルトは`num_warmup_steps`",
    )
    parser.add_argument(
        "--lr_scheduler_min_lr_ratio",
        type=float,
        default=None,
        help="The minimum learning rate as a ratio of the initial learning rate for cosine with min lr scheduler, warmup decay scheduler and rex scheduler"
        + " / 初期学習率の比率としての最小学習率を指定する、cosine with min lr スケジューラ、warmup decay スケジューラ、rex スケジューラ で有効",
    )
    parser.add_argument("--lr_scheduler_type", type=str, default="", help="custom scheduler module / 使用するスケジューラ")
    parser.add_argument(
        "--lr_scheduler_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for scheduler (like "T_max=100") / スケジューラの追加引数（例： "T_max100"）',
    )


def _add_memory_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fp8_base", action="store_true", help="use fp8 for base model / base modelにfp8を使う")
    # Keep the common parser default for helper callers; the Dev parser adds the explicit option.
    parser.set_defaults(fp8_scaled=False)
    # parser.add_argument("--full_fp16", action="store_true", help="fp16 training including gradients / 勾配も含めてfp16で学習する")
    # parser.add_argument("--full_bf16", action="store_true", help="bf16 training including gradients / 勾配も含めてbf16で学習する")

    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=None,
        help="number of blocks to swap in the model, max XXX / モデル内のブロックの数、最大XXX",
    )
    parser.add_argument(
        "--use_pinned_memory_for_block_swap",
        action="store_true",
        help="use pinned memory for block swapping, which may speed up data transfer between CPU and GPU but uses more shared GPU memory on Windows"
        " / ブロックスワッピングにピン留めメモリを使用する。これによりCPUとGPU間のデータ転送が高速化される可能性があるが、Windowsではより多くの共有GPUメモリを使用する。",
    )
    parser.add_argument(
        "--block_swap_h2d_only",
        action="store_true",
        help="(experimental, frozen-base / LoRA training only) use H2D-only block swap:"
        " keep a CPU master copy of streamed weights and only copy Host->Device, never back. Removes the D2H transfer."
        " / (実験的、ベース凍結＝LoRA学習専用) H2DのみのブロックスワップでD2H転送を行わない。",
    )
    parser.add_argument(
        "--block_swap_ring_size",
        type=int,
        default=2,
        help="(used with --block_swap_h2d_only) number of GPU ring buffers for streamed blocks. 2 = double buffering"
        " (one computed on, one prefetched); 1 = minimal memory but no transfer/compute overlap."
        " / (--block_swap_h2d_only用) ストリーミング用GPUリングバッファ数。2でダブルバッファ、1で最小メモリ(オーバーラップなし)。",
    )
    parser.add_argument(
        "--img_in_txt_in_offloading",
        action="store_true",
        help="offload img_in and txt_in to cpu / img_inとtxt_inをCPUにオフロードする",
    )
    parser.add_argument(
        "--disable_numpy_memmap",
        action="store_true",
        help="Disable numpy memory mapping for model loading. For FLUX.2 checkpoint loading. Increases RAM usage but speeds up model loading in some cases."
        " / モデル読み込み時のnumpyメモリマッピングを無効にします。FLUX.2で有効です。RAM使用量が増えますが、場合によってはモデルの読み込みが高速化されます。",
    )


def _add_timestep_args(parser: argparse.ArgumentParser) -> None:
    # parser.add_argument("--flow_shift", type=float, default=7.0, help="Shift factor for flow matching schedulers")
    parser.add_argument(
        "--timestep_sampling",
        choices=[
            "sigma",
            "uniform",
            "sigmoid",
            "shift",
            "flux_shift",
            "flux2_shift",
            "ideogram4_shift",
            "qwen_shift",
            "krea2_shift",
            "logsnr",
            "qinglong_flux",
            "qinglong_qwen",
        ],
        default="sigma",
        help="Method to sample timesteps: sigma-based, uniform random, sigmoid of random normal, shift of sigmoid and flux shift."
        " / タイムステップをサンプリングする方法：sigma、random uniform、random normalのsigmoid、sigmoidのシフト、flux shift。",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=1.0,
        help="Discrete flow shift for the Euler Discrete Scheduler, default is 1.0. / Euler Discrete Schedulerの離散フローシフト、デフォルトは1.0。",
    )
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help='Scale factor for sigmoid timestep sampling (only used when timestep-sampling is "sigmoid" or "shift"). / sigmoidタイムステップサンプリングの倍率（timestep-samplingが"sigmoid"または"shift"の場合のみ有効）。',
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["logit_normal", "mode", "cosmap", "sigma_sqrt", "none"],
        help="weighting scheme for timestep distribution. Default is none / タイムステップ分布の重み付けスキーム、デフォルトはnone",
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme / `'logit_normal'`重み付けスキームを使用する場合の平均",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme / `'logit_normal'`重み付けスキームを使用する場合のstd",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme` / モード重み付けスキームのスケール",
    )
    parser.add_argument(
        "--min_timestep",
        type=int,
        default=None,
        help="set minimum time step for training (0~999, default is 0) / 学習時のtime stepの最小値を設定する（0~999で指定、省略時はデフォルト値(0)） ",
    )
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=None,
        help="set maximum time step for training (1~1000, default is 1000) / 学習時のtime stepの最大値を設定する（1~1000で指定、省略時はデフォルト値(1000)）",
    )
    parser.add_argument(
        "--preserve_distribution_shape",
        action="store_true",
        help="If specified, constrains timestep sampling to [min_timestep, max_timestep] "
        "using rejection sampling, preserving the original distribution shape. "
        "By default, the [0, 1] range is scaled, which distorts the distribution. Only effective when `timestep_sampling` is not 'sigma'."
        " / 指定すると、タイムステップのサンプリングを[最小タイムステップ、最大タイムステップ]に制約し、元の分布形状を保持します。"
        "デフォルトでは、[0, 1]の範囲がスケーリングされ、分布が歪むことがあります。timestep_samplingがsigma以外で有効です。",
    )
    parser.add_argument(
        "--num_timestep_buckets",
        type=int,
        default=None,
        help=(
            "Number of buckets for timestep sampling. Default is None, which disables bucketing. "
            "Set to 2 or more to enable stratified sampling. This forces timesteps to be sampled "
            "uniformly from the [0, 1] range, which can improve training stability, especially for small datasets."
            " / timestepサンプリングのバケット数。デフォルトはNoneで、バケット化を無効にします。"
            "2以上に設定すると、層化抽出が有効になり、タイムステップが[0, 1]の範囲から均等にサンプリングされるようになります。"
            "これは、特に小規模なデータセットでの学習の安定性向上が期待できます。"
        ),
    )

    parser.add_argument(
        "--show_timesteps",
        type=str,
        default=None,
        choices=["image", "console"],
        help="show timesteps in image or console, and return to console / タイムステップを画像またはコンソールに表示し、コンソールに戻る",
    )


def _add_network_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない"
    )
    parser.add_argument(
        "--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み"
    )
    parser.add_argument(
        "--network_module", type=str, default=None, help="network module to train / 学習対象のネットワークのモジュール"
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=None,
        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )


def _add_save_load_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output_dir", type=str, default=None, help="directory to output trained model / 学習後のモデル出力先ディレクトリ"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="base name of trained model file / 学習後のモデルの拡張子を除くファイル名",
    )
    parser.add_argument("--resume", type=str, default=None, help="saved state to resume training / 学習再開するモデルのstate")

    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=None,
        help="save checkpoint every N epochs / 学習中のモデルを指定エポックごとに保存する",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=None,
        help="save checkpoint every N steps / 学習中のモデルを指定ステップごとに保存する",
    )
    parser.add_argument(
        "--save_last_n_epochs",
        type=int,
        default=None,
        help="save last N checkpoints when saving every N epochs (remove older checkpoints) / 指定エポックごとにモデルを保存するとき最大Nエポック保存する（古いチェックポイントは削除する）",
    )
    parser.add_argument(
        "--save_last_n_epochs_state",
        type=int,
        default=None,
        help="save last N checkpoints of state (overrides the value of --save_last_n_epochs)/ 最大Nエポックstateを保存する（--save_last_n_epochsの指定を上書きする）",
    )
    parser.add_argument(
        "--save_last_n_steps",
        type=int,
        default=None,
        help="save checkpoints until N steps elapsed (remove older checkpoints if N steps elapsed) / 指定ステップごとにモデルを保存するとき、このステップ数経過するまで保存する（このステップ数経過したら削除する）",
    )
    parser.add_argument(
        "--save_last_n_steps_state",
        type=int,
        default=None,
        help="save states until N steps elapsed (remove older states if N steps elapsed, overrides --save_last_n_steps) / 指定ステップごとにstateを保存するとき、このステップ数経過するまで保存する（このステップ数経過したら削除する。--save_last_n_stepsを上書きする）",
    )
    parser.add_argument(
        "--save_state",
        action="store_true",
        help="save training state additionally (including optimizer states etc.) when saving model / optimizerなど学習状態も含めたstateをモデル保存時に追加で保存する",
    )
    parser.add_argument(
        "--save_state_on_train_end",
        action="store_true",
        help="save training state (including optimizer states etc.) on train end even if --save_state is not specified"
        " / --save_stateが未指定時にもoptimizerなど学習状態も含めたstateを学習終了時に保存する",
    )


def _add_metadata_args(parser: argparse.ArgumentParser) -> None:
    # SAI Model spec
    parser.add_argument(
        "--metadata_title",
        type=str,
        default=None,
        help="title for model metadata (default is output_name) / メタデータに書き込まれるモデルタイトル、省略時はoutput_name",
    )
    parser.add_argument(
        "--metadata_author",
        type=str,
        default=None,
        help="author name for model metadata / メタデータに書き込まれるモデル作者名",
    )
    parser.add_argument(
        "--metadata_description",
        type=str,
        default=None,
        help="description for model metadata / メタデータに書き込まれるモデル説明",
    )
    parser.add_argument(
        "--metadata_license",
        type=str,
        default=None,
        help="license for model metadata / メタデータに書き込まれるモデルライセンス",
    )
    parser.add_argument(
        "--metadata_tags",
        type=str,
        default=None,
        help="tags for model metadata, separated by comma / メタデータに書き込まれるモデルタグ、カンマ区切り",
    )
    parser.add_argument(
        "--metadata_reso",
        type=str,
        default=None,
        help="resolution for model metadata (e.g., `1024,1024`) / メタデータに書き込まれるモデル解像度（例: `1024,1024`）",
    )
    parser.add_argument(
        "--metadata_arch",
        type=str,
        default=None,
        help="architecture for model metadata / メタデータに書き込まれるモデルアーキテクチャ",
    )


def _add_huggingface_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--huggingface_repo_id",
        type=str,
        default=None,
        help="huggingface repo name to upload / huggingfaceにアップロードするリポジトリ名",
    )
    parser.add_argument(
        "--huggingface_repo_type",
        type=str,
        default=None,
        help="huggingface repo type to upload / huggingfaceにアップロードするリポジトリの種類",
    )
    parser.add_argument(
        "--huggingface_path_in_repo",
        type=str,
        default=None,
        help="huggingface model path to upload files / huggingfaceにアップロードするファイルのパス",
    )
    parser.add_argument("--huggingface_token", type=str, default=None, help="huggingface token / huggingfaceのトークン")
    parser.add_argument(
        "--huggingface_repo_visibility",
        type=str,
        default=None,
        help="huggingface repository visibility ('public' for public, 'private' or None for private) / huggingfaceにアップロードするリポジトリの公開設定（'public'で公開、'private'またはNoneで非公開）",
    )
    parser.add_argument(
        "--save_state_to_huggingface", action="store_true", help="save state to huggingface / huggingfaceにstateを保存する"
    )
    parser.add_argument(
        "--resume_from_huggingface",
        action="store_true",
        help="resume from huggingface (ex: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type}) / huggingfaceから学習を再開する(例: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type})",
    )
    parser.add_argument(
        "--async_upload",
        action="store_true",
        help="upload to huggingface asynchronously / huggingfaceに非同期でアップロードする",
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dit", type=str, help="DiT checkpoint path / DiTのチェックポイントのパス")
    parser.add_argument("--vae", type=str, help="VAE checkpoint path / VAEのチェックポイントのパス")
    parser.add_argument("--vae_dtype", type=str, default=None, help="data type for VAE, default depends on model")


def setup_parser_common() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _add_general_args(parser)
    _add_attention_args(parser)
    _add_compile_and_dynamo_args(parser)
    _add_training_args(parser)
    _add_logging_args(parser)
    _add_ddp_args(parser)
    _add_sampling_args(parser)
    _add_optimizer_args(parser)
    _add_lr_scheduler_args(parser)
    _add_memory_args(parser)
    _add_timestep_args(parser)
    _add_network_args(parser)
    _add_save_load_args(parser)
    _add_metadata_args(parser)
    _add_huggingface_args(parser)
    _add_model_args(parser)
    return parser


def read_config_from_file(args: argparse.Namespace, parser: argparse.ArgumentParser):
    sources = {}
    if not args.config_file:
        args._config_sources = sources
        return args

    config_path = args.config_file + ".toml" if not args.config_file.endswith(".toml") else args.config_file

    logger.info(f"Loading settings from {config_path}...")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = toml.load(f)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"{config_path}: config_file={args.config_file!r}: {error}; provide a readable, valid TOML file"
        ) from error

    destinations = {action.dest for action in parser._actions if action.dest != "help"}

    def add_value(key, value, location):
        if key not in destinations:
            raise ValueError(f"{config_path}:{location}={value!r}: unknown or removed parameter; use a retained option from --help")
        if isinstance(value, dict) or (isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value)):
            raise ValueError(
                f"{config_path}:{location}={value!r}: nested value/group is unsupported; use a scalar or flat option list"
            )
        ignore_nesting_dict[key] = value
        sources[key] = f"{config_path}:{location}"

    # combine all sections into one
    ignore_nesting_dict = {}
    for section_name, section_dict in config_dict.items():
        # if value is not dict, save key and value as is
        if not isinstance(section_dict, dict):
            add_value(section_name, section_dict, section_name)
            continue

        # if value is dict, save all key and value into one dict
        for key, value in section_dict.items():
            add_value(key, value, f"{section_name}.{key}")

    config_args = argparse.Namespace(**ignore_nesting_dict)
    args = parser.parse_args(namespace=config_args)
    # Explicit CLI values take precedence, including their diagnostic location.
    for token in sys.argv[1:]:
        option = token.split("=", 1)[0]
        action = parser._option_string_actions.get(option)
        if action is not None:
            sources[action.dest] = f"CLI:{option}"
    args._config_sources = sources
    args.config_file = os.path.splitext(args.config_file)[0]
    logger.info(args.config_file)

    return args


def config_error(args, key, value, cause, correction):
    source = getattr(args, "_config_sources", {}).get(key, f"CLI/default:{key}")
    return ValueError(f"{source}={value!r}: {cause}; {correction}")


def parse_nested_args(args, key, *, literal=True):
    result = {}
    for assignment in getattr(args, key, None) or []:
        name, separator, value = assignment.partition("=")
        name = name.strip()
        if not separator or not name or name in result:
            raise config_error(args, key, assignment, "malformed or duplicate assignment", "use unique name=value entries")
        try:
            result[name] = ast.literal_eval(value) if literal else value
        except (ValueError, SyntaxError) as error:
            raise config_error(args, key, assignment, str(error), "use a valid Python literal after =") from error
    return result


def require_dependency(args, key, package):
    try:
        return importlib.import_module(package)
    except (ImportError, OSError) as error:
        raise config_error(
            args,
            key,
            package,
            f"selected dependency is unavailable: {error}",
            f"install a compatible {package} in the execution environment",
        ) from error


def validate_path(value, source, *, directory=False, writable=False):
    """Check filesystem prerequisites without creating output or reading tensors."""
    try:
        if not value:
            raise ValueError("a path is required")
        path = pathlib.Path(value)
        if writable:
            if path.exists() and not path.is_dir():
                raise ValueError("destination is not a directory")
            parent = path
            while not parent.exists() and parent != parent.parent:
                parent = parent.parent
            if not parent.is_dir() or not os.access(parent, os.W_OK):
                raise ValueError("destination parent is not writable")
        elif directory:
            if not path.is_dir() or not os.access(path, os.R_OK):
                raise ValueError("directory is missing or unreadable")
        else:
            if not path.is_file():
                raise ValueError("file is missing or not a regular file")
            with path.open("rb"):
                pass
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"{source}={value!r}: {error}; provide an accessible {'writable directory' if writable else 'directory' if directory else 'file'}"
        ) from error
    return path


def validate_call_kwargs(args, key, target, kwargs, positional=(), supplied=None):
    """Bind the selected callable without constructing an optimizer or tracker."""
    supplied = supplied or {}
    try:
        signature = inspect.signature(target)
        duplicate = set(kwargs) & set(supplied)
        if duplicate:
            raise TypeError(f"arguments already supplied by the trainer: {sorted(duplicate)}")
        bound = signature.bind(*positional, **supplied, **kwargs)
        # Inherited optimizer constructors may forward kwargs to a base constructor.
        # Check that contract rather than treating **kwargs as arbitrary configuration.
        parameters = dict(signature.parameters)
        variadic = next((p.name for p in parameters.values() if p.kind == p.VAR_KEYWORD), None)
        if variadic and bound.arguments.get(variadic):
            inherited = {}
            for base in getattr(target, "__mro__", ())[1:]:
                inherited.update(inspect.signature(base).parameters)
            unknown = set(bound.arguments[variadic]) - set(inherited)
            if unknown:
                raise TypeError(f"no supported forwarded parameter contract for {sorted(unknown)}")
            parameters.update(inherited)
        for name, value in kwargs.items():
            parameter = parameters.get(name)
            if parameter is None:
                continue
            default = parameter.default
            if type(default) in (bool, int, float, str):
                valid = type(value) is type(default)
                # Numeric defaults such as SGD momentum=0 are not integer-only.
                # Respect an explicit int annotation; otherwise accept either numeric type.
                if type(default) in (int, float) and parameter.annotation not in (int, "int"):
                    valid = type(value) in (int, float) and math.isfinite(value)
                if not valid:
                    raise TypeError(f"{name} expects {type(default).__name__}, got {value!r}")
    except (TypeError, ValueError) as error:
        raise config_error(
            args, key, kwargs, str(error), "use supported initialization arguments for the selected callable"
        ) from error


def validate_effective_args(args, parser):
    """Check the actual merged values; argparse does not validate Namespace defaults."""
    for action in parser._actions:
        if action.dest == "help":
            continue
        value = getattr(args, action.dest, None)
        if value is None:
            continue
        values = value
        if action.nargs in ("*", "+"):
            if not isinstance(value, list) or (action.nargs == "+" and not value):
                raise config_error(args, action.dest, value, "expected a list", "supply a flat list of option values")
        else:
            values = [value]
        for item in values:
            expected = action.type
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                valid = type(item) is bool
            elif expected is int:
                valid = type(item) is int
            elif expected in (float, _int_or_float):
                valid = type(item) in (float, int) and math.isfinite(item)
            elif expected is pathlib.Path:
                valid = isinstance(item, (str, pathlib.Path))
            elif expected is str or expected is None:
                valid = isinstance(item, str)
            else:
                valid = True
            if not valid:
                raise config_error(args, action.dest, value, "invalid value type", f"use the type documented by --{action.dest}")
            if action.choices is not None and item not in action.choices:
                raise config_error(args, action.dest, value, "unsupported value", f"choose one of {list(action.choices)!r}")

    positive = (
        "network_dim",
        "max_train_steps",
        "max_train_epochs",
        "gradient_accumulation_steps",
        "save_every_n_steps",
        "save_every_n_epochs",
        "sample_every_n_steps",
        "sample_every_n_epochs",
        "batch_size",
        "num_workers",
    )
    for key in positive:
        value = getattr(args, key, None)
        if value is not None and value <= 0:
            raise config_error(args, key, value, "must be positive", "supply a value greater than zero")
    for key in (
        "learning_rate",
        "max_grad_norm",
        "blocks_to_swap",
        "max_data_loader_n_workers",
        "lr_warmup_steps",
        "lr_decay_steps",
        "save_last_n_steps",
        "save_last_n_steps_state",
    ):
        value = getattr(args, key, None)
        if value is not None and value < 0:
            raise config_error(args, key, value, "must be nonnegative", "supply zero or a positive value")
    dropout = getattr(args, "network_dropout", None)
    if dropout is not None and not 0 <= dropout <= 1:
        raise config_error(args, "network_dropout", dropout, "outside [0, 1]", "use a dropout probability in [0, 1]")
    if getattr(args, "fp8_scaled", False) and not getattr(args, "fp8_base", False):
        raise config_error(args, "fp8_scaled", True, "requires fp8_base", "enable fp8_base or disable fp8_scaled")
    if getattr(args, "fp8_text_encoder", False):
        raise config_error(args, "fp8_text_encoder", True, "Mistral FP8 is unsupported", "set fp8_text_encoder=false")
    if getattr(args, "persistent_data_loader_workers", False) and args.max_data_loader_n_workers == 0:
        raise config_error(
            args,
            "persistent_data_loader_workers",
            True,
            "requires loader workers",
            "use a positive worker count or disable persistent workers",
        )
    if getattr(args, "vae_dtype", None) not in (None, "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"):
        raise config_error(args, "vae_dtype", args.vae_dtype, "unsupported VAE dtype", "use float32, float16 or bfloat16")
    # Dev has 8 double / 48 single blocks. Its existing swap allocation first
    # exceeds the reserved two blocks of each kind at blocks_to_swap=30.
    if (getattr(args, "blocks_to_swap", None) or 0) > 29:
        raise config_error(
            args,
            "blocks_to_swap",
            args.blocks_to_swap,
            "exceeds Dev's 8 double / 48 single block allocation",
            "use blocks_to_swap in [0, 29]",
        )
    min_t = getattr(args, "min_timestep", None)
    max_t = getattr(args, "max_timestep", None)
    min_t = 0 if min_t is None else min_t
    max_t = 1000 if max_t is None else max_t
    if not 0 <= min_t < 1000:
        raise config_error(args, "min_timestep", min_t, "outside the 1000-step sampling range", "use min_timestep in [0, 999]")
    if not 0 < max_t <= 1000:
        raise config_error(args, "max_timestep", max_t, "outside the 1000-step sampling range", "use max_timestep in [1, 1000]")
