"""Opt-in configuration for the retained Qwen validation/experiment workflow."""

from pathlib import Path, PurePosixPath, PureWindowsPath


VALIDATION_DEFAULTS = {
    "val_every_n_steps": 50,
    "val_seed_noise": 42,
    "val_level_noise_n": 10,
    "val_seed_noise_n": 2,
}


def add_training_arguments(parser):
    parser.add_argument(
        "--experiment_mode", action="store_true", help="Resolve paths relative to the training TOML and use step directories"
    )
    parser.add_argument("--val_dataset_config", type=Path, default=None, help="Independent fixed validation dataset TOML")
    for key in VALIDATION_DEFAULTS:
        parser.add_argument("--" + key, type=int, default=None)


def add_cache_arguments(parser):
    parser.add_argument(
        "--experiment_mode", action="store_true", help="Resolve dataset and model paths relative to the dataset TOML"
    )
    parser.add_argument("--validation", action="store_true", help="Prepare fixed validation caches with source provenance")


def _resolve(value, root):
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute() or PurePosixPath(str(value)).is_absolute() or PureWindowsPath(str(value)).is_absolute():
        return str(value)
    return str((root / path).resolve())


def configure_training_args(args):
    source = getattr(args, "_config_source", "CLI")
    enabled = bool(getattr(args, "val_dataset_config", None))
    for key, default in VALIDATION_DEFAULTS.items():
        value = getattr(args, key, None)
        if not enabled and value is not None:
            raise ValueError(
                f"{source}: {key}: requires val_dataset_config; provide independent validation data or omit this parameter"
            )
        if enabled:
            value = default if value is None else value
            if type(value) is not int:
                raise ValueError(f"{source}: {key}={value!r}: expected an integer, not a boolean; correct this parameter")
            if key != "val_seed_noise" and value < (2 if key == "val_level_noise_n" else 1):
                raise ValueError(f"{source}: {key}={value}: value is too small; use a positive count (at least 2 for levels)")
            if key == "val_level_noise_n" and value % 2:
                raise ValueError(f"{source}: {key}={value}: expected an even level count; correct this parameter")
            setattr(args, key, value)
    if getattr(args, "experiment_mode", False):
        if source == "CLI":
            raise ValueError("CLI: experiment_mode requires --config_file; place train.toml in the experiment directory")
        root = Path(source).resolve().parent
        args._experiment_root = str(root)
        for key in (
            "dataset_config",
            "val_dataset_config",
            "dit",
            "vae",
            "text_encoder",
            "network_weights",
            "sample_prompts",
            "output_dir",
            "logging_dir",
            "log_tracker_config",
            "resume",
        ):
            if hasattr(args, key):
                setattr(args, key, _resolve(getattr(args, key), root))
        if getattr(args, "base_weights", None):
            args.base_weights = [_resolve(value, root) for value in args.base_weights]
        if getattr(args, "save_state", False) or getattr(args, "save_state_on_train_end", False):
            if getattr(args, "save_precision", None) not in (None, "float", "fp32"):
                raise ValueError(f"{source}: save_precision: state checkpoints require FP32 for their sole adapter copy; use fp32")
            if getattr(args, "full_fp16", False) or getattr(args, "full_bf16", False):
                raise ValueError(
                    f"{source}: full_fp16/full_bf16: state checkpoints require FP32 trainable weights; disable full precision conversion"
                )
    return args


def configure_cache_args(args):
    if getattr(args, "experiment_mode", False):
        args.dataset_config = str(Path(args.dataset_config).resolve())
        root = Path(args.dataset_config).parent
        for key in ("vae", "text_encoder"):
            if hasattr(args, key):
                setattr(args, key, _resolve(getattr(args, key), root))
    if getattr(args, "validation", False) and getattr(args, "batch_size", None) not in (None, 1):
        raise ValueError("CLI: validation batch_size must be 1; omit the override or use --batch_size 1")
    return args


def resolve_dataset_paths(config, source):
    """Copy only path-bearing tables; never rewrite a user's TOML."""
    root = Path(source).resolve().parent
    result = dict(config)
    result["datasets"] = [dict(dataset) for dataset in config.get("datasets", [])]
    for dataset in result["datasets"]:
        for key in ("image_directory", "image_jsonl_file", "cache_directory"):
            if key in dataset:
                dataset[key] = _resolve(dataset[key], root)
    return result
