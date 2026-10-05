"""Strict, source-linked Qwen validation caches, with no training-time encoding."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from PIL import Image
from safetensors import SafetensorError, safe_open
import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.bucket import BucketBatchManager, BucketSelector
from musubi_tuner.dataset.image_video_dataset import ImageDataset, ItemInfo
from musubi_tuner.dataset.media_utils import glob_images
from musubi_tuner.utils.model_utils import dtype_to_str


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()


def _preparation(dataset, size):
    bucket = BucketSelector(dataset.resolution, dataset.enable_bucket, dataset.bucket_no_upscale, dataset.architecture)
    return {
        "version": 1,
        "architecture": "qwen_image",
        "resize": "center_crop",
        "vae_encoding": "posterior_mode",
        "resolution": list(dataset.resolution),
        "enable_bucket": dataset.enable_bucket,
        "bucket_no_upscale": dataset.bucket_no_upscale,
        "original_size": list(size),
        "bucket_size": list(bucket.get_bucket_resolution(size)),
    }


def _provenance(image_sha256, caption, preparation):
    return {
        "qwen_source_sha256": image_sha256,
        "qwen_caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
        "qwen_preparation_sha256": _json_hash(preparation),
    }


@dataclass
class ValidationRecord:
    image_sha256: str
    source_path: str
    caption: str
    preparation: dict
    item: ItemInfo
    dataset: ImageDataset
    datasource_index: int
    latent_sha256: str = ""
    text_sha256: str = ""

    @property
    def provenance(self):
        return _provenance(self.image_sha256, self.caption, self.preparation)

    def identity(self):
        return {
            "image_sha256": self.image_sha256,
            "caption": self.caption,
            "preparation": self.preparation,
            "latent_sha256": self.latent_sha256,
            "text_sha256": self.text_sha256,
        }


def _source_records(datasets, source, validation=False):
    records = []
    cache_paths = set()
    for dataset_number, dataset in enumerate(datasets, 1):
        label = f"{source}: dataset {dataset_number}"
        if validation:
            for key in ("batch_size", "num_repeats"):
                if getattr(dataset, key) != 1:
                    raise ValueError(f"{label}: {key} must be 1 for validation; correct the dataset configuration")
            if dataset.image_directory:
                selected = {str(Path(path).resolve()) for path in dataset.datasource.image_paths}
                declared = {str(Path(path).resolve()) for path in glob_images(dataset.image_directory)}
                missing = declared - selected
                if missing:
                    raise ValueError(
                        f"{label}: missing caption for validation image {sorted(missing)[0]}; provide a matching caption file"
                    )
        if len(dataset.datasource) == 0:
            raise ValueError(f"{label}: empty effective image set; provide images with captions")
        for index in range(len(dataset.datasource)):
            image_path, caption = dataset.datasource.get_caption(index)
            with Image.open(image_path) as image:
                size = image.size
                image.verify()
            preparation = _preparation(dataset, size)
            item = ItemInfo(image_path, caption, size, tuple(preparation["bucket_size"]))
            item.latent_cache_path = dataset.get_latent_cache_path(item)
            item.text_encoder_output_cache_path = dataset.get_text_encoder_output_cache_path(item)
            # The text cache is stem-only even when original image dimensions differ.
            for path in (item.latent_cache_path, item.text_encoder_output_cache_path):
                resolved = str(Path(path).resolve())
                if resolved in cache_paths:
                    raise ValueError(
                        f"{label}: cache name collision at {path}; use separate cache directories or unique image stems"
                    )
                cache_paths.add(resolved)
            records.append(
                ValidationRecord(
                    file_sha256(image_path), str(Path(image_path).resolve()), caption, preparation, item, dataset, index
                )
            )
    return records


def _validate_cache(record, path, latent, require_provenance=True):
    label = f"validation cache {path} (source {record.source_path})"
    if not Path(path).is_file():
        raise ValueError(f"{label}: missing cache; run the corresponding Qwen cache command before training")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as opened:
            metadata = opened.metadata() or {}
            if metadata.get("architecture") != "qwen_image" or metadata.get("format_version") != "1.0.1":
                raise ValueError("unsupported architecture/format metadata")
            if require_provenance and any(metadata.get(key) != value for key, value in record.provenance.items()):
                raise ValueError("missing or stale source/caption/preparation provenance")
            keys = list(opened.keys())
            if len(keys) != 1:
                raise ValueError("expected exactly one original Qwen tensor")
            tensor = opened.get_tensor(keys[0])
            if not tensor.is_floating_point() or not torch.isfinite(tensor.float()).all().item():
                raise ValueError("tensor must contain finite floating point values")
            if latent:
                width, height = record.preparation["bucket_size"]
                expected_shape = (16, 1, height // 8, width // 8)
                if tuple(tensor.shape) != expected_shape:
                    raise ValueError(f"latent shape {tuple(tensor.shape)} does not match {expected_shape}")
                expected_key = f"latents_1x{height // 8}x{width // 8}_{dtype_to_str(tensor.dtype)}"
                if metadata.get("width") != str(record.item.original_size[0]) or metadata.get("height") != str(
                    record.item.original_size[1]
                ):
                    raise ValueError("source dimensions do not match cache metadata")
            else:
                if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] != 3584:
                    raise ValueError("text embedding must have shape [tokens>0,3584]")
                expected_key = f"varlen_vl_embed_{dtype_to_str(tensor.dtype)}"
                if metadata.get("caption1") != record.caption:
                    raise ValueError("cached caption differs from source caption")
            if keys[0] != expected_key:
                raise ValueError(f"tensor key {keys[0]!r} does not match {expected_key!r}")
    except (OSError, RuntimeError, ValueError, SafetensorError) as error:
        raise ValueError(f"{label}: invalid or stale cache: {error}; regenerate it with --validation before training") from error
    return file_sha256(path)


def prepare_cache_batch(batch, datasets, args):
    """Attach opt-in provenance to existing ItemInfo objects before either encoder."""
    if not (getattr(args, "experiment_mode", False) or getattr(args, "validation", False)):
        return
    for item in batch:
        dataset = datasets[item.dataset_index]
        with Image.open(item.item_key) as image:
            preparation = _preparation(dataset, image.size)
        item.cache_metadata = _provenance(file_sha256(item.item_key), item.caption, preparation)


def validate_cache_dataset(group, args):
    """Pre-model checks for new cache modes; --validation does not change path semantics."""
    if not (getattr(args, "experiment_mode", False) or getattr(args, "validation", False)):
        return
    records = _source_records(group.datasets, args.dataset_config, validation=getattr(args, "validation", False))
    if getattr(args, "validation", False) and getattr(args, "skip_existing", False):
        for record in records:
            for path, latent in ((record.item.latent_cache_path, True), (record.item.text_encoder_output_cache_path, False)):
                if Path(path).exists():
                    _validate_cache(record, path, latent)


def _build_records(args, train_group):
    val_args = argparse.Namespace(**vars(args))
    val_args.dataset_config = args.val_dataset_config
    declaration = config_utils.load_user_config(args.val_dataset_config)
    blueprint = config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(declaration, val_args, architecture="qi")
    # Bypass DatasetGroup's random seed: these datasets never shuffle or prepare train buckets.
    datasets = [ImageDataset(**asdict(item.params)) for item in blueprint.dataset_group.datasets]
    config_utils.validate_dataset_sources(argparse.Namespace(datasets=datasets), args.val_dataset_config)
    train_records = _source_records(train_group.datasets, args.dataset_config)
    records = _source_records(datasets, args.val_dataset_config, validation=True)
    train_hashes = {record.image_sha256 for record in train_records}
    val_hashes = {record.image_sha256 for record in records}
    overlap = train_hashes & val_hashes
    if overlap:
        raise ValueError(
            f"{args.val_dataset_config}: train/val image SHA-256 overlap {sorted(overlap)[0]}; provide independent val images"
        )
    if len(val_hashes) != len(records):
        raise ValueError(f"{args.val_dataset_config}: duplicate validation image contents; keep each image once")
    directories = [Path(dataset.cache_directory).resolve() for dataset in train_group.datasets + datasets]
    for index, directory in enumerate(directories):
        for other in directories[index + 1 :]:
            if directory == other or directory in other.parents or other in directory.parents:
                raise ValueError(
                    f"{args.val_dataset_config}: overlapping cache directories {directory} and {other}; use separate train/val caches"
                )
    expected_train = {str(Path(record.item.latent_cache_path).resolve()): record for record in train_records}
    effective_count = 0
    verified_train = set()
    for dataset in train_group.datasets:
        manager = dataset.batch_manager
        if manager is None:
            raise ValueError(f"{args.dataset_config}: training cache dataset is not prepared; build it before validation preflight")
        for bucket in manager.buckets.values():
            for item in bucket:
                key = str(Path(item.latent_cache_path).resolve())
                record = expected_train.get(key)
                if (
                    record is None
                    or Path(item.text_encoder_output_cache_path).resolve()
                    != Path(record.item.text_encoder_output_cache_path).resolve()
                ):
                    raise ValueError(
                        f"{args.dataset_config}: unexpected effective training cache {key}; remove stale caches or correct the source declaration"
                    )
                if key not in verified_train:
                    _validate_cache(record, item.latent_cache_path, True)
                    _validate_cache(record, item.text_encoder_output_cache_path, False)
                    verified_train.add(key)
                effective_count += 1
    if effective_count == 0:
        raise ValueError(f"{args.dataset_config}: empty effective training cache dataset; prepare training caches first")
    for record in records:
        record.latent_sha256 = _validate_cache(record, record.item.latent_cache_path, True)
        record.text_sha256 = _validate_cache(record, record.item.text_encoder_output_cache_path, False)
    return sorted(records, key=lambda record: record.image_sha256)


class ValidationInputs:
    def __init__(self, args, train_group, records):
        self._args = argparse.Namespace(**vars(args))
        self._train_group = train_group
        self.records = records
        self.manifest = {
            "schema": 1,
            "parameters": {
                key: getattr(args, key) for key in ("val_every_n_steps", "val_seed_noise", "val_level_noise_n", "val_seed_noise_n")
            },
            "images": [record.identity() for record in records],
        }
        self.fingerprint = _json_hash(self.manifest)

    def verify_unchanged(self):
        current = _build_records(self._args, self._train_group)
        if [record.identity() for record in current] != self.manifest["images"]:
            raise ValueError(
                f"{self._args.val_dataset_config}: validation inputs changed during this measurement series; restore the fixed inputs"
            )

    def load_batch(self, index):
        record = self.records[index]
        source, caption = record.dataset.datasource.get_caption(record.datasource_index)
        if file_sha256(source) != record.image_sha256 or caption != record.caption:
            raise ValueError(f"{source}: validation image/caption changed; restore the fixed inputs before continuing")
        for path, expected in (
            (record.item.latent_cache_path, record.latent_sha256),
            (record.item.text_encoder_output_cache_path, record.text_sha256),
        ):
            if not Path(path).is_file() or file_sha256(path) != expected:
                raise ValueError(f"{path}: validation cache changed or disappeared; restore the fixed cache before continuing")
        return BucketBatchManager({record.item.bucket_size: [record.item]}, batch_size=1)[0]


def prepare_validation_inputs(args, train_group):
    if not getattr(args, "val_dataset_config", None):
        return None
    return ValidationInputs(args, train_group, _build_records(args, train_group))
