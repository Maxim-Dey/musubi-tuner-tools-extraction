"""Image dataset configuration and existing blueprint construction."""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import random
from typing import Optional, Sequence, Tuple
import toml
import voluptuous as vol
from musubi_tuner.dataset.architectures import ARCHITECTURE_KREA2
from musubi_tuner.dataset.image_video_dataset import DatasetGroup, ImageDataset

logger = logging.getLogger(__name__)


@dataclass
class BaseDatasetParams:
    resolution: Tuple[int, int] = (960, 544)
    enable_bucket: bool = False
    bucket_no_upscale: bool = False
    caption_extension: Optional[str] = None
    batch_size: int = 1
    num_repeats: int = 1
    cache_directory: Optional[str] = None
    debug_dataset: bool = False
    architecture: str = ARCHITECTURE_KREA2


@dataclass
class ImageDatasetParams(BaseDatasetParams):
    image_directory: Optional[str] = None
    image_jsonl_file: Optional[str] = None


@dataclass
class DatasetBlueprint:
    is_image_dataset: bool
    params: ImageDatasetParams


@dataclass
class DatasetGroupBlueprint:
    datasets: Sequence[DatasetBlueprint]


@dataclass
class Blueprint:
    dataset_group: DatasetGroupBlueprint


def _positive_int(value):
    if type(value) is not int or value < 1:
        raise vol.Invalid("must be a positive integer")
    return value


def _resolution(value):
    values = (value, value) if type(value) is int else value
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise vol.Invalid("resolution must be an integer or [width, height]")
    if any(type(v) is not int or v < 1 for v in values):
        raise vol.Invalid("resolution dimensions must be positive integers")
    return tuple(values)


class ConfigSanitizer:
    DATASET_ASCENDABLE_SCHEMA = {
        "caption_extension": str,
        "batch_size": _positive_int,
        "num_repeats": _positive_int,
        "resolution": _resolution,
        "enable_bucket": bool,
        "bucket_no_upscale": bool,
    }
    IMAGE_DATASET_DISTINCT_SCHEMA = {"image_directory": str, "image_jsonl_file": str, "cache_directory": str}

    def __init__(self):
        self.image_dataset_schema = {**self.DATASET_ASCENDABLE_SCHEMA, **self.IMAGE_DATASET_DISTINCT_SCHEMA}
        self.user_config_validator = vol.Schema(
            {
                "general": self.DATASET_ASCENDABLE_SCHEMA,
                vol.Required("datasets"): vol.All([self.image_dataset_schema], vol.Length(min=1)),
            }
        )

    def sanitize_user_config(self, user_config):
        try:
            result = self.user_config_validator(user_config)
            for index, dataset in enumerate(result["datasets"]):
                sources = [key for key in ("image_directory", "image_jsonl_file") if dataset.get(key)]
                if len(sources) != 1:
                    raise ValueError(f"datasets[{index}]: specify exactly one image_directory or image_jsonl_file")
                if "image_jsonl_file" in sources and not dataset.get("cache_directory"):
                    raise ValueError(
                        f"datasets[{index}].cache_directory: required for image_jsonl_file; set a unique cache directory"
                    )
            return result
        except vol.Invalid as error:
            raise ValueError(f"Dataset config: {error}; use only supported image dataset fields and values") from error

    def sanitize_argparse_namespace(self, namespace):
        if hasattr(namespace, "debug_dataset") and type(namespace.debug_dataset) is not bool:
            raise ValueError("CLI debug_dataset: expected a boolean")
        return namespace


class BlueprintGenerator:
    BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME = {}

    def __init__(self, sanitizer):
        self.sanitizer = sanitizer

    def generate(self, user_config, argparse_namespace, **runtime_params):
        config = self.sanitizer.sanitize_user_config(user_config)
        args = vars(self.sanitizer.sanitize_argparse_namespace(argparse_namespace))
        datasets = []
        for index, dataset in enumerate(config["datasets"]):
            params = self.generate_params_by_fallbacks(
                ImageDatasetParams, [dataset, config.get("general", {}), args, runtime_params]
            )
            if params.architecture != ARCHITECTURE_KREA2:
                raise ValueError(f"datasets[{index}].architecture: only Krea2 Raw image datasets are supported")
            if not params.enable_bucket and any(value % 16 for value in params.resolution):
                raise ValueError(
                    f"datasets[{index}].resolution: fixed dimensions must be multiples of 16; enable buckets or align dimensions"
                )
            if params.enable_bucket and params.resolution[0] * params.resolution[1] < 32 * 32:
                raise ValueError(f"datasets[{index}].resolution: bucket area is too small; increase resolution or disable buckets")
            if params.image_directory and not params.caption_extension:
                raise ValueError(f"datasets[{index}].caption_extension: set the caption sidecar extension, e.g. '.txt'")
            datasets.append(DatasetBlueprint(True, params))
        return Blueprint(DatasetGroupBlueprint(datasets))

    @staticmethod
    def generate_params_by_fallbacks(param_klass, fallbacks):
        defaults = asdict(param_klass())
        names = BlueprintGenerator.BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME
        return param_klass(
            **{name: BlueprintGenerator.search_value(names.get(name, name), fallbacks, value) for name, value in defaults.items()}
        )

    @staticmethod
    def search_value(key, fallbacks, default_value=None):
        for candidate in fallbacks:
            if candidate.get(key) is not None:
                return candidate[key]
        return default_value


def generate_dataset_group_by_blueprint(dataset_group_blueprint, training=False, num_timestep_buckets=None, shared_epoch=None):
    datasets = [ImageDataset(**asdict(blueprint.params)) for blueprint in dataset_group_blueprint.datasets]
    cache_directories = [str(Path(dataset.cache_directory).resolve()) for dataset in datasets]
    if len(set(cache_directories)) != len(cache_directories):
        raise ValueError("Dataset cache_directory: use a unique directory for each dataset")
    seed = random.randint(0, 2**31)
    for index, dataset in enumerate(datasets):
        logger.info("Dataset %s: %s", index, dataset.get_metadata())
        dataset.set_seed(seed, shared_epoch)
        if training:
            dataset.prepare_for_training(num_timestep_buckets=num_timestep_buckets)
    return DatasetGroup(datasets)


def load_user_config(file, resolve_paths=False):
    path = Path(file)
    if not path.is_file():
        raise ValueError(f"Dataset config {path}: file not found; set an existing dataset config path")
    try:
        with path.open(encoding="utf-8") as stream:
            if path.suffix.lower() == ".json":
                config = json.load(stream)
            elif path.suffix.lower() == ".toml":
                config = toml.load(stream)
            else:
                raise ValueError("unsupported format; use .toml or .json")
        config = ConfigSanitizer().sanitize_user_config(config)
        if resolve_paths:
            for dataset in config["datasets"]:
                for key in ("image_directory", "image_jsonl_file", "cache_directory"):
                    if dataset.get(key) and not Path(dataset[key]).is_absolute():
                        dataset[key] = str(path.resolve().parent / dataset[key])
        return config
    except ValueError as error:
        raise ValueError(f"Dataset config {path}: {error}; correct the indicated field or file syntax") from error
