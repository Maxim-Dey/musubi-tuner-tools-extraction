import argparse
from dataclasses import (
    asdict,
    dataclass,
)
import functools
import random
from textwrap import dedent, indent
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

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _positive_integer(value):
    if type(value) is not int or value <= 0:
        raise voluptuous.Invalid("expected a positive integer")
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
    architecture: str = "no_default"  # supplied by the Dev entrypoint


@dataclass
class ImageDatasetParams(BaseDatasetParams):
    image_directory: Optional[str] = None
    image_jsonl_file: Optional[str] = None
    control_directory: Optional[str] = None

    no_resize_control: Optional[bool] = False  # if True, control images are not resized to target resolution
    control_resolution: Optional[Tuple[int, int]] = None  # if set, control images are resized to this resolution


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
        if not isinstance(value, (list, tuple)) or any(type(v) is not klass or v <= 0 for v in value):
            raise voluptuous.Invalid("expected positive integer dimensions")
        Schema(ExactSequence([klass, klass]))(value)
        return tuple(value)

    # @curry
    @staticmethod
    def __validate_and_convert_scalar_or_twodim(klass, value: Union[float, Sequence]) -> Tuple:
        if isinstance(value, (list, tuple)):
            return ConfigSanitizer.__validate_and_convert_twodim(klass, value)
        if type(value) is not klass or value <= 0:
            raise voluptuous.Invalid("expected a positive integer resolution or two positive dimensions")
        Schema(Any(klass, ExactSequence([klass, klass])))(value)
        try:
            Schema(klass)(value)
            return (value, value)
        except:
            return ConfigSanitizer.__validate_and_convert_twodim(klass, value)

    # datasets schema
    DATASET_ASCENDABLE_SCHEMA = {
        "caption_extension": str,
        "batch_size": _positive_integer,
        "num_repeats": _positive_integer,
        "resolution": functools.partial(__validate_and_convert_scalar_or_twodim.__func__, int),
        "enable_bucket": bool,
        "bucket_no_upscale": bool,
    }
    IMAGE_DATASET_DISTINCT_SCHEMA = {
        "image_directory": str,
        "image_jsonl_file": str,
        "cache_directory": str,
        "control_directory": str,
        "no_resize_control": bool,
        "control_resolution": functools.partial(__validate_and_convert_scalar_or_twodim.__func__, int),
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

    def sanitize_user_config(self, user_config: dict, source="dataset") -> dict:
        try:
            sanitized = self.user_config_validator(user_config)
        except voluptuous.Invalid as error:
            path = ".".join(str(part) for part in error.path) or "root"
            value = user_config
            for part in error.path:
                try:
                    value = value[part]
                except (KeyError, IndexError, TypeError):
                    break
            raise ValueError(f"{source}:{path}={value!r}: {error.msg}; use the general/image dataset schema") from error
        if not isinstance(sanitized, dict) or not sanitized.get("datasets"):
            raise ValueError(f"{source}:datasets={user_config!r}: no image datasets; provide a nonempty datasets list")
        for index, dataset in enumerate(sanitized["datasets"]):
            sources = [key for key in ("image_directory", "image_jsonl_file") if dataset.get(key)]
            if len(sources) != 1:
                raise ValueError(
                    f"{source}:datasets.{index}={dataset!r}: ambiguous or missing image source; provide exactly one of image_directory and image_jsonl_file"
                )
            if "image_jsonl_file" in dataset and dataset.get("control_directory"):
                raise ValueError(
                    f"{source}:datasets.{index}.control_directory={dataset['control_directory']!r}: JSONL uses per-record controls; put control_path in JSONL records"
                )
        return sanitized

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
        sanitized_user_config = self.sanitizer.sanitize_user_config(user_config)
        sanitized_argparse_namespace = self.sanitizer.sanitize_argparse_namespace(argparse_namespace)

        argparse_config = {k: v for k, v in vars(sanitized_argparse_namespace).items() if v is not None}
        general_config = sanitized_user_config.get("general", {})

        dataset_blueprints = []
        for dataset_config in sanitized_user_config.get("datasets", []):
            if not ("image_directory" in dataset_config or "image_jsonl_file" in dataset_config):
                raise ValueError("dataset requires image_directory or image_jsonl_file")
            is_image_dataset = True
            dataset_params_klass = ImageDatasetParams

            params = self.generate_params_by_fallbacks(
                dataset_params_klass, [dataset_config, general_config, argparse_config, runtime_params]
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
def validate_dataset_paths(blueprint, source):
    from musubi_tuner.training.parser_common import validate_path
    from musubi_tuner.dataset.datasources import ImageDirectoryDatasource, ImageJsonlDatasource

    caches = set()
    for index, dataset in enumerate(blueprint.dataset_group.datasets):
        params = dataset.params
        location = f"{source}:datasets.{index}"
        if params.image_directory is not None:
            validate_path(params.image_directory, f"{location}.image_directory", directory=True)
            if params.control_directory is not None:
                validate_path(params.control_directory, f"{location}.control_directory", directory=True)
            datasource = ImageDirectoryDatasource(params.image_directory, params.caption_extension, params.control_directory)
            for image in datasource.image_paths:
                validate_path(image, f"{location}.image_directory")
                caption = str(Path(image).with_suffix(params.caption_extension)) if params.caption_extension else None
                validate_path(caption, f"{location}.caption_extension")
        else:
            validate_path(params.image_jsonl_file, f"{location}.image_jsonl_file")
            datasource = ImageJsonlDatasource(params.image_jsonl_file)
            for row, record in enumerate(datasource.data, 1):
                image = record.get("image_path", record.get("image_path_0"))
                validate_path(image, f"{params.image_jsonl_file}:{row}.image_path")
                if not isinstance(record.get("caption"), str):
                    raise ValueError(
                        f"{params.image_jsonl_file}:{row}.caption={record.get('caption')!r}: expected text; provide a caption string"
                    )
        if len(datasource) == 0:
            raise ValueError(f"{location}: no images; provide a nonempty image dataset")
        for paths in datasource.get_control_paths().values():
            for path in paths:
                validate_path(path, f"{location}.control_path")
        cache = validate_path(params.cache_directory or params.image_directory, f"{location}.cache_directory", writable=True)
        identity = cache.resolve()
        if identity in caches:
            raise ValueError(
                f"{location}.cache_directory={str(cache)!r}: duplicate cache directory; use a separate directory per dataset"
            )
        caches.add(identity)


def generate_dataset_group_by_blueprint(
    dataset_group_blueprint: DatasetGroupBlueprint,
    training: bool = False,
    num_timestep_buckets: Optional[int] = None,
    shared_epoch: SharedEpoch = None,
) -> DatasetGroup:
    datasets: List[ImageDataset] = []

    for dataset_blueprint in dataset_group_blueprint.datasets:
        dataset_klass = ImageDataset

        dataset_params = asdict(dataset_blueprint.params)
        dataset = dataset_klass(**dataset_params)
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

        info += indent(
            dedent(
                f"""\
    image_directory: "{dataset.image_directory}"
    image_jsonl_file: "{dataset.image_jsonl_file}"
    control_directory: "{dataset.control_directory}"
    no_resize_control: {dataset.no_resize_control}
    control_resolution: {dataset.control_resolution}
\n"""
            ),
            "    ",
        )
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
    if file is None:
        raise ValueError("CLI:dataset_config=None: dataset configuration is required; provide a readable TOML/JSON file")
    file: Path = Path(file)
    if not file.is_file():
        raise ValueError(f"{file}:dataset_config={str(file)!r}: file not found; provide a readable dataset TOML/JSON file")

    if file.name.lower().endswith(".json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, ValueError) as error:
            logger.error(
                f"Error on parsing JSON config file. Please check the format. / JSON 形式の設定ファイルの読み込みに失敗しました。文法が正しいか確認してください。: {file}"
            )
            raise ValueError(f"{file}:dataset_config: {error}; correct the JSON syntax and file permissions") from error
    elif file.name.lower().endswith(".toml"):
        try:
            config = toml.load(file)
        except (OSError, ValueError) as error:
            logger.error(
                f"Error on parsing TOML config file. Please check the format. / TOML 形式の設定ファイルの読み込みに失敗しました。文法が正しいか確認してください。: {file}"
            )
            raise ValueError(f"{file}:dataset_config: {error}; correct the TOML syntax and file permissions") from error
    else:
        raise ValueError(f"not supported config file format / 対応していない設定ファイルの形式です: {file}")

    if not isinstance(config, dict):
        raise ValueError(f"{file}:root={config!r}: expected a mapping; provide general and datasets sections")

    deprecated_key_map = {
        "flux_kontext_no_resize_control": "no_resize_control",
        "qwen_image_edit_no_resize_control": "no_resize_control",
        "qwen_image_edit_control_resolution": "control_resolution",
    }

    def normalize_deprecated_keys(section: dict, section_name: str) -> None:
        for old_key, new_key in deprecated_key_map.items():
            if old_key not in section:
                continue
            try:
                Schema(ConfigSanitizer.IMAGE_DATASET_DISTINCT_SCHEMA[new_key])(section[old_key])
            except voluptuous.Invalid as error:
                raise ValueError(
                    f"{file}:{section_name}.{old_key}={section[old_key]!r}: {error.msg}; use a valid {new_key} value"
                ) from error
            if new_key in section:
                logger.warning(
                    f"Deprecated config key '{old_key}' is ignored because '{new_key}' is already set in {section_name}."
                )
            else:
                section[new_key] = section[old_key]
                logger.warning(f"Deprecated config key '{old_key}' found in {section_name}; use '{new_key}' instead.")
            del section[old_key]

    general_config = config.get("general")
    if isinstance(general_config, dict):
        normalize_deprecated_keys(general_config, "general")

    datasets_config = config.get("datasets", [])
    if isinstance(datasets_config, list):
        for idx, dataset_config in enumerate(datasets_config):
            if isinstance(dataset_config, dict):
                normalize_deprecated_keys(dataset_config, f"datasets[{idx}]")

    ConfigSanitizer().sanitize_user_config(config, source=str(file))
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
