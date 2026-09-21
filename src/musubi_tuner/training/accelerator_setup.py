"""Accelerator/device setup utilities shared across training scripts."""

from datetime import timedelta
import argparse
import gc
import os
import time
import inspect
import toml

import torch
from packaging.version import Version
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs
from accelerate.utils import TorchDynamoPlugin, DynamoBackend


def validate_tracker_config(args):
    from musubi_tuner.training.parser_common import config_error, require_dependency, validate_call_kwargs

    backend = args.log_with or ("tensorboard" if args.logging_dir is not None else None)
    selected = {"tensorboard", "wandb"} if backend == "all" else ({backend} if backend else set())
    if "tensorboard" in selected and args.logging_dir is None:
        raise config_error(args, "logging_dir", None, "TensorBoard requires a logging directory", "set logging_dir")
    for name in selected:
        require_dependency(args, "log_with", name)
    config = {}
    path = args.log_tracker_config
    if path:
        try:
            config = toml.load(path)
        except (OSError, ValueError) as error:
            raise ValueError(f"{path}:log_tracker_config={path!r}: {error}; provide a readable valid tracker TOML file") from error
    for name, kwargs in config.items():
        source_key = f"{path}:{name}"
        if name not in ("tensorboard", "wandb") or not isinstance(kwargs, dict):
            raise ValueError(
                f"{source_key}={kwargs!r}: unknown tracker or invalid mapping; use [tensorboard] or [wandb] initialization arguments"
            )
        # Validate configured backends even if this run does not select them.
        if name == "tensorboard":
            require_dependency(args, "log_tracker_config", "tensorboard")
            from torch.utils.tensorboard import SummaryWriter

            target, positional, supplied = SummaryWriter, ("validation-only",), {}
        else:
            wandb = require_dependency(args, "log_tracker_config", "wandb")
            target, positional, supplied = wandb.init, (), {"project": "validation-only"}
        parameters = inspect.signature(target).parameters
        for key, value in kwargs.items():
            location = f"{source_key}.{key}"
            if key not in parameters or key in supplied or (name == "tensorboard" and key == "log_dir"):
                raise ValueError(
                    f"{location}={value!r}: unsupported or already supplied initialization parameter; use the selected backend's supported arguments"
                )
            valid = True
            if name == "tensorboard":
                if key in ("flush_secs", "max_queue"):
                    valid = type(value) is int and value > 0
                elif key == "purge_step":
                    valid = type(value) is int and value >= 0
                elif key in ("comment", "filename_suffix"):
                    valid = isinstance(value, str)
            elif key in ("config", "settings"):
                valid = isinstance(value, dict) or (key == "config" and isinstance(value, str))
                if key == "settings" and isinstance(value, dict):
                    # Settings performs local field validation, without starting a run.
                    try:
                        wandb.Settings(**value)
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"{location}={value!r}: {error}; use supported WandB settings") from error
            elif name == "wandb" and key in ("tags", "config_exclude_keys", "config_include_keys"):
                valid = isinstance(value, list) and all(isinstance(v, str) for v in value)
            elif name == "wandb" and key in ("allow_val_change", "force", "anonymous", "resume", "reinit", "mode"):
                allowed = {
                    "allow_val_change": (True, False),
                    "force": (True, False),
                    "anonymous": ("never", "allow", "must"),
                    "resume": (True, False, "allow", "never", "must", "auto"),
                    "reinit": (True, False, "default", "return_previous", "finish_previous", "create_new"),
                    "mode": ("online", "offline", "disabled", "shared"),
                }
                valid = value in allowed[key]
            elif name == "wandb" and key not in ("config", "settings"):
                valid = isinstance(value, str)
            if not valid:
                raise ValueError(f"{location}={value!r}: invalid initialization value; use the backend's documented type and range")
        try:
            validate_call_kwargs(args, "log_tracker_config", target, kwargs, positional, supplied)
        except ValueError as error:
            raise ValueError(f"{source_key}: {error}") from error
    args._tracker_init_kwargs = config


def validate_attention_dependencies(args):
    from musubi_tuner.training.parser_common import config_error, require_dependency

    if args.sage_attn:
        raise config_error(
            args,
            "sage_attn",
            True,
            "SageAttention training is unsupported",
            "select sdpa, xformers or another supported training backend",
        )
    # Match the loader's precedence when users leave multiple historical flags set.
    for option, package in (
        ("sdpa", None),
        ("flash_attn", "flash_attn"),
        ("xformers", "xformers.ops"),
        ("flash3", "flash_attn_interface"),
    ):
        if getattr(args, option, False):
            if package:
                require_dependency(args, option, package)
            break
    else:
        raise config_error(args, "sdpa", False, "no attention backend selected", "select sdpa, flash_attn, xformers or flash3")
    if args.blocks_to_swap:
        from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig

        try:
            BlockSwapConfig.from_args(args, torch.device("cpu"), supports_backward=True)
        except ValueError as error:
            raise config_error(
                args,
                "block_swap_h2d_only",
                args.block_swap_h2d_only,
                str(error),
                "enable gradient_checkpointing and use a positive block_swap_ring_size",
            ) from error
    if args.compile:
        from torch._dynamo.backends.registry import lookup_backend

        try:
            lookup_backend(args.compile_backend)  # registration/import only; never compile or allocate a model
        except (ImportError, KeyError, RuntimeError, ValueError) as error:
            raise config_error(
                args, "compile_backend", args.compile_backend, str(error), "select an installed torch.compile backend"
            ) from error


def clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


# for collate_fn: epoch and step is multiprocessing.Value
class collator_class:
    def __init__(self, epoch, dataset):
        self.current_epoch = epoch
        self.dataset = dataset  # not used if worker_info is not None, in case of multiprocessing

    def __call__(self, examples):
        worker_info = torch.utils.data.get_worker_info()
        # worker_info is None in the main process
        if worker_info is not None:
            dataset = worker_info.dataset
        else:
            dataset = self.dataset

        # set epoch for validation
        dataset.set_current_epoch(self.current_epoch.value)
        return examples[0]  # batch size is always 1, so we unwrap it here


def prepare_accelerator(args: argparse.Namespace) -> Accelerator:
    """
    DeepSpeed is not supported in this script currently.
    """
    if args.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if args.log_prefix is None else args.log_prefix
        logging_dir = args.logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    if args.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        log_with = args.log_with
        if log_with in ["tensorboard", "all"]:
            if logging_dir is None:
                raise ValueError(
                    "logging_dir is required when log_with is tensorboard / Tensorboardを使う場合、logging_dirを指定してください"
                )
        if log_with in ["wandb", "all"]:
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb / wandb がインストールされていないようです")
            if logging_dir is not None:
                os.makedirs(logging_dir, exist_ok=True)
                os.environ["WANDB_DIR"] = logging_dir
            if args.wandb_api_key is not None:
                wandb.login(key=args.wandb_api_key)

    kwargs_handlers = [
        (
            InitProcessGroupKwargs(
                backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
                init_method=(
                    "env://?use_libuv=False" if os.name == "nt" and Version(torch.__version__) >= Version("2.4.0") else None
                ),
                timeout=timedelta(minutes=args.ddp_timeout) if args.ddp_timeout else None,
            )
            if torch.cuda.device_count() > 1
            else None
        ),
        (
            DistributedDataParallelKwargs(
                gradient_as_bucket_view=args.ddp_gradient_as_bucket_view, static_graph=args.ddp_static_graph
            )
            if args.ddp_gradient_as_bucket_view or args.ddp_static_graph
            else None
        ),
    ]
    kwargs_handlers = [i for i in kwargs_handlers if i is not None]

    dynamo_plugin = None
    if args.dynamo_backend.upper() != "NO":
        dynamo_plugin = TorchDynamoPlugin(
            backend=DynamoBackend(args.dynamo_backend.upper()),
            mode=args.dynamo_mode,
            fullgraph=args.dynamo_fullgraph,
            dynamic=args.dynamo_dynamic,
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision else None,
        log_with=log_with,
        project_dir=logging_dir,
        dynamo_plugin=dynamo_plugin,
        kwargs_handlers=kwargs_handlers,
    )
    print("accelerator device:", accelerator.device)
    return accelerator
