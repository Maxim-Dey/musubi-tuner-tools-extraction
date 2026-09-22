import argparse
from dataclasses import (
    asdict,
    dataclass,
)
import functools
import random
from textwrap import dedent
import json
from pathlib import Path

from typing import List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized


SharedEpoch = Optional["Synchronized[int]"]


import toml
import voluptuous
from voluptuous import Any, ExactSequence, MultipleInvalid, Object, Schema

from musubi_tuner.dataset.image_video_dataset import DatasetGroup, ImageDataset
from musubi_tuner.dataset.bucket import BucketSelector

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def positive_int(value):
    if type(value) is not int or value <= 0:
        raise voluptuous.Invalid("expected a positive integer (not a boolean)")
    return value


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
    architecture: str = "no_default"  # original Qwen short name is "qi"


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


class ConfigSanitizer:
    # @curry
    @staticmethod
    def __validate_and_convert_twodim(klass, value: Sequence) -> Tuple:
        Schema(ExactSequence([klass, klass]))(value)
        return tuple(value)

    # @curry
    @staticmethod
    def __validate_and_convert_scalar_or_twodim(klass, value: Union[float, Sequence]) -> Tuple:
        Schema(Any(klass, ExactSequence([klass, klass])))(value)
        try:
            Schema(klass)(value)
            return (value, value)
        except:
            return ConfigSanitizer.__validate_and_convert_twodim(klass, value)

    # datasets schema
    DATASET_ASCENDABLE_SCHEMA = {
        "caption_extension": str,
        "batch_size": positive_int,
        "num_repeats": positive_int,
        "resolution": functools.partial(__validate_and_convert_scalar_or_twodim.__func__, positive_int),
        "enable_bucket": bool,
        "bucket_no_upscale": bool,
    }
    IMAGE_DATASET_DISTINCT_SCHEMA = {
        "image_directory": str,
        "image_jsonl_file": str,
        "cache_directory": str,
    }

    # options handled by argparse but not handled by user config
    ARGPARSE_SPECIFIC_SCHEMA = {
        "debug_dataset": bool,
    }

    def __init__(self) -> None:
        self.image_dataset_schema = self.__merge_dict(
            self.DATASET_ASCENDABLE_SCHEMA,
            self.IMAGE_DATASET_DISTINCT_SCHEMA,
        )
        self.dataset_schema = Schema(self.image_dataset_schema)

        self.general_schema = self.__merge_dict(
            self.DATASET_ASCENDABLE_SCHEMA,
        )
        self.user_config_validator = Schema(
            {
                "general": self.general_schema,
                "datasets": [self.dataset_schema],
            }
        )
        self.argparse_schema = self.__merge_dict(
            self.ARGPARSE_SPECIFIC_SCHEMA,
        )
        self.argparse_config_validator = Schema(Object(self.argparse_schema), extra=voluptuous.ALLOW_EXTRA)

    def sanitize_user_config(self, user_config: dict) -> dict:
        try:
            return self.user_config_validator(user_config)
        except MultipleInvalid:
            # TODO: clarify the error message
            logger.error("Invalid user config / ユーザ設定の形式が正しくないようです")
            raise

    # NOTE: In nature, argument parser result is not needed to be sanitize
    #   However this will help us to detect program bug
    def sanitize_argparse_namespace(self, argparse_namespace: argparse.Namespace) -> argparse.Namespace:
        try:
            return self.argparse_config_validator(argparse_namespace)
        except MultipleInvalid:
            # XXX: this should be a bug
            logger.error(
                "Invalid cmdline parsed arguments. This should be a bug. / コマンドラインのパース結果が正しくないようです。プログラムのバグの可能性が高いです。"
            )
            raise

    # NOTE: value would be overwritten by latter dict if there is already the same key
    @staticmethod
    def __merge_dict(*dict_list: dict) -> dict:
        merged = {}
        for schema in dict_list:
            # merged |= schema
            for k, v in schema.items():
                merged[k] = v
        return merged


class BlueprintGenerator:
    BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME = {}

    def __init__(self, sanitizer: ConfigSanitizer):
        self.sanitizer = sanitizer

    # runtime_params is for parameters which is only configurable on runtime, such as tokenizer
    def generate(self, user_config: dict, argparse_namespace: argparse.Namespace, **runtime_params) -> Blueprint:
        source = getattr(argparse_namespace, "dataset_config", None) or "dataset configuration"
        try:
            sanitized_user_config = self.sanitizer.sanitize_user_config(user_config)
        except (voluptuous.Invalid, TypeError) as error:
            raise ValueError(f"{source}: {error}; use supported original image dataset fields and types") from error
        if not sanitized_user_config.get("datasets"):
            raise ValueError(f"{source}: datasets is empty; provide at least one image dataset")
        sanitized_argparse_namespace = self.sanitizer.sanitize_argparse_namespace(argparse_namespace)

        argparse_config = {k: v for k, v in vars(sanitized_argparse_namespace).items() if v is not None}
        general_config = sanitized_user_config.get("general", {})

        dataset_blueprints = []
        for dataset_config in sanitized_user_config.get("datasets", []):
            is_image_dataset = True
            source_keys = [key for key in ("image_directory", "image_jsonl_file") if dataset_config.get(key)]
            if len(source_keys) != 1:
                raise ValueError(
                    f"{source}: datasets[{len(dataset_blueprints)}] requires exactly one image_directory or image_jsonl_file; provide one image source"
                )
            params = self.generate_params_by_fallbacks(
                ImageDatasetParams, [dataset_config, general_config, argparse_config, runtime_params]
            )
            if params.image_jsonl_file and not params.cache_directory:
                raise ValueError(
                    f"{source}: datasets[{len(dataset_blueprints)}].cache_directory is required for image_jsonl_file; set a cache location"
                )
            dataset_blueprints.append(DatasetBlueprint(is_image_dataset, params))

        dataset_group_blueprint = DatasetGroupBlueprint(dataset_blueprints)

        return Blueprint(dataset_group_blueprint)

    @staticmethod
    def generate_params_by_fallbacks(param_klass, fallbacks: Sequence[dict]):
        name_map = BlueprintGenerator.BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME
        search_value = BlueprintGenerator.search_value
        default_params = asdict(param_klass())
        param_names = default_params.keys()

        params = {name: search_value(name_map.get(name, name), fallbacks, default_params.get(name)) for name in param_names}

        return param_klass(**params)

    @staticmethod
    def search_value(key: str, fallbacks: Sequence[dict], default_value=None):
        for cand in fallbacks:
            value = cand.get(key)
            if value is not None:
                return value

        return default_value


# if training is True, it will return a dataset group for training, otherwise for caching
def generate_dataset_group_by_blueprint(
    dataset_group_blueprint: DatasetGroupBlueprint,
    training: bool = False,
    num_timestep_buckets: Optional[int] = None,
    shared_epoch: SharedEpoch = None,
) -> DatasetGroup:
    datasets: List[ImageDataset] = []

    for dataset_blueprint in dataset_group_blueprint.datasets:
        dataset_params = asdict(dataset_blueprint.params)
        dataset = ImageDataset(**dataset_params)
        datasets.append(dataset)

    # assertion
    cache_directories = [dataset.cache_directory for dataset in datasets]
    num_of_unique_cache_directories = len(set(cache_directories))
    if num_of_unique_cache_directories != len(cache_directories):
        raise ValueError(
            "cache directory should be unique for each dataset (note that cache directory is image directory if not specified)"
            + " / cache directory は各データセットごとに異なる必要があります（指定されていない場合はimage directoryが使われるので注意）"
        )

    # print info
    info = ""
    for i, dataset in enumerate(datasets):
        is_image_dataset = isinstance(dataset, ImageDataset)
        info += dedent(
            f"""\
      [Dataset {i}]
        is_image_dataset: {is_image_dataset}
        resolution: {dataset.resolution}
        batch_size: {dataset.batch_size}
        num_repeats: {dataset.num_repeats}
        caption_extension: "{dataset.caption_extension}"
        enable_bucket: {dataset.enable_bucket}
        bucket_no_upscale: {dataset.bucket_no_upscale}
        cache_directory: "{dataset.cache_directory}"
        debug_dataset: {dataset.debug_dataset}
    """
        )

        info += f"    image_directory: {dataset.image_directory}\n    image_jsonl_file: {dataset.image_jsonl_file}\n"
    logger.info(f"{info}")

    # make buckets first because it determines the length of dataset
    # and set the same seed for all datasets
    seed = random.randint(0, 2**31)  # actual seed is seed + epoch_no
    for i, dataset in enumerate(datasets):
        # logger.info(f"[Dataset {i}]")
        dataset.set_seed(seed, shared_epoch)
        if training:
            dataset.prepare_for_training(num_timestep_buckets=num_timestep_buckets)

    return DatasetGroup(datasets)


def load_user_config(file: str) -> dict:
    file: Path = Path(file)
    if not file.is_file():
        raise ValueError(f"file not found / ファイルが見つかりません: {file}")

    if file.name.lower().endswith(".json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            logger.error(
                f"Error on parsing JSON config file. Please check the format. / JSON 形式の設定ファイルの読み込みに失敗しました。文法が正しいか確認してください。: {file}"
            )
            raise
    elif file.name.lower().endswith(".toml"):
        try:
            config = toml.load(file)
        except Exception:
            logger.error(
                f"Error on parsing TOML config file. Please check the format. / TOML 形式の設定ファイルの読み込みに失敗しました。文法が正しいか確認してください。: {file}"
            )
            raise
    else:
        raise ValueError(f"not supported config file format / 対応していない設定ファイルの形式です: {file}")

    if not isinstance(config, dict):
        raise ValueError(f"{file}: dataset configuration must be a table/object; provide general and datasets")

    return config


# for config test
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_config")
    config_args, remain = parser.parse_known_args()

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_dataset", action="store_true")
    argparse_namespace = parser.parse_args(remain)

    logger.info("[argparse_namespace]")
    logger.info(f"{vars(argparse_namespace)}")

    user_config = load_user_config(config_args.dataset_config)

    logger.info("")
    logger.info("[user_config]")
    logger.info(f"{user_config}")

    sanitizer = ConfigSanitizer()
    sanitized_user_config = sanitizer.sanitize_user_config(user_config)

    logger.info("")
    logger.info("[sanitized_user_config]")
    logger.info(f"{sanitized_user_config}")

    blueprint = BlueprintGenerator(sanitizer).generate(user_config, argparse_namespace)

    logger.info("")
    logger.info("[blueprint]")
    logger.info(f"{blueprint}")

    dataset_group = generate_dataset_group_by_blueprint(blueprint.dataset_group)


def validate_dataset_sources(group, source):
    """Read source declarations and image headers before models or cache output side effects."""
    from PIL import Image

    for index, dataset in enumerate(group.datasets):
        label = f"{source}: dataset {index + 1}"
        if any(size < 16 for size in dataset.resolution):
            raise ValueError(f"{label}: resolution must be usable for image packing (at least 16); increase it")
        if dataset.enable_bucket and min(dataset.resolution) < 32 and dataset.resolution[0] * dataset.resolution[1] < 1024:
            raise ValueError(f"{label}: resolution area is too small for buckets; increase it or disable buckets")
        if not dataset.enable_bucket and any(size % 16 for size in dataset.resolution):
            raise ValueError(f"{label}: resolution must be divisible by 16 without buckets; correct the dimensions")
        cache = Path(dataset.cache_directory)
        for path in (cache, *cache.parents):
            if path.exists() and not path.is_dir():
                raise ValueError(
                    f"{label}: cache_directory={cache}: {path} is a file; "
                    "supply a directory path whose existing parents are directories"
                )
        datasource = dataset.datasource
        if len(datasource) == 0:
            raise ValueError(f"{label}: no images with captions; supply original-image sources and matching captions")
        buckets = BucketSelector(dataset.resolution, dataset.enable_bucket, dataset.bucket_no_upscale, dataset.architecture)
        for item_index in range(len(datasource)):
            try:
                image_path, caption = datasource.get_caption(item_index)
                with Image.open(image_path) as image:
                    bucket = buckets.get_bucket_resolution(image.size)
                    image.verify()
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"{label}: item {item_index + 1}: invalid image/caption source: {error}; correct the source file"
                ) from error
            if any(size < 16 for size in bucket):
                raise ValueError(
                    f"{label}: item {item_index + 1}: source {image_path!r} selects unusable bucket {bucket}; "
                    "use images at least 16 pixels in each dimension or disable bucket_no_upscale"
                )


def validate_cache_args(args):
    for key in ("batch_size", "num_workers", "console_width", "console_num_images"):
        value = getattr(args, key, None)
        if value is not None and value <= 0:
            raise ValueError(f"CLI: {key} must be positive; omit it for the existing default")
    checkpoint_key = "text_encoder" if hasattr(args, "text_encoder") else "vae"
    checkpoint = getattr(args, checkpoint_key, None)
    if not getattr(args, "debug_mode", None) and (not checkpoint or not Path(checkpoint).is_file()):
        raise ValueError(f"CLI: {checkpoint_key}={checkpoint!r} is not an input file; supply an existing checkpoint")
