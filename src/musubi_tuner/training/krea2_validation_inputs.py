"""Krea2-only, model-free inventory and frozen cache boundary."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import random

import numpy as np
from pathlib import Path

from PIL import Image
from safetensors import safe_open
import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.bucket import BucketSelector
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.dataset.media_utils import glob_images
from musubi_tuner.training.parser_common import validate_validation_options


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


@dataclass
class ValidationRecord:
    image_sha256: str
    image_path: str
    caption: str
    caption_sha256: str
    original_size: tuple
    bucket_size: tuple
    processing: dict
    latent_cache_path: str
    text_cache_path: str
    dataset_config: str

    def semantic(self):
        return {
            "image_sha256": self.image_sha256,
            "caption_sha256": self.caption_sha256,
            "original_size": list(self.original_size),
            "bucket_size": list(self.bucket_size),
            "processing": self.processing,
        }


def _inventory(config_path, args, validation=False):
    config = config_utils.load_user_config(config_path, resolve_paths=getattr(args, "experiment_mode", False))
    blueprint = config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(config, args)
    records, cache_dirs = [], []
    # Construct datasets without training preparation (which trusts cache filenames).
    from dataclasses import asdict
    from musubi_tuner.dataset.image_video_dataset import ImageDataset

    for entry in blueprint.dataset_group.datasets:
        params = entry.params
        if validation and (params.batch_size != 1 or params.num_repeats != 1):
            raise ValueError(f"{config_path}: validation batch_size and num_repeats must both be 1; correct the dataset config")
        if params.image_directory:
            for path in glob_images(params.image_directory):
                caption = Path(path).with_suffix(params.caption_extension)
                if not caption.is_file():
                    raise ValueError(f"{config_path}: {path}: missing caption {caption}; add the matching caption")
        dataset = ImageDataset(**asdict(params))
        cache_dirs.append(str(Path(dataset.cache_directory).resolve()))
        selector = BucketSelector(params.resolution, params.enable_bucket, params.bucket_no_upscale, params.architecture)
        processing = {
            "resolution": list(params.resolution),
            "enable_bucket": params.enable_bucket,
            "bucket_no_upscale": dataset.bucket_no_upscale,
            "resize": "lanczos-up-area-down-center-crop-v1",
            "color": "RGB-drop-alpha",
            "normalization": "qwen-image-vae-normalized-v1",
        }
        for index in range(len(dataset.datasource)):
            path, caption = dataset.datasource.get_caption(index)
            if not caption.strip():
                raise ValueError(f"{config_path}: {path}: empty caption; provide a nonempty caption")
            with Image.open(path) as image:
                size = image.size
                image.verify()
            bucket = selector.get_bucket_resolution(size)
            item = ItemInfo(path, caption, size, bucket)
            caption_path = Path(path).with_suffix(params.caption_extension) if params.image_directory else None
            # Hash original sidecar bytes as well as effective caption; whitespace changes are fixed-input drift.
            caption_hash = file_sha256(caption_path) if caption_path else _digest(caption)
            records.append(
                ValidationRecord(
                    file_sha256(path),
                    path,
                    caption,
                    caption_hash,
                    size,
                    bucket,
                    processing,
                    dataset.get_latent_cache_path(item),
                    dataset.get_text_encoder_output_cache_path(item),
                    str(config_path),
                )
            )
    if len(set(cache_dirs)) != len(cache_dirs):
        raise ValueError(f"{config_path}: cache_directory must be distinct for each dataset")
    if not records:
        raise ValueError(f"{config_path}: dataset is empty; provide captioned images")
    return records, cache_dirs


def _component_identities(args):
    result = {}
    for name in ("vae", "text_encoder", "dit"):
        value = getattr(args, name, None)
        if not value or not Path(value).is_file():
            raise ValueError(f"{name}: {value!r}: existing local model file required to verify cache/experiment identity")
        result[name] = file_sha256(value)
    tokenizer = getattr(args, "tokenizer_path", None)
    if not tokenizer:
        from huggingface_hub import snapshot_download

        try:
            tokenizer = snapshot_download("Qwen/Qwen3-VL-4B-Instruct", local_files_only=True)
        except Exception as error:
            raise ValueError(
                "tokenizer_path: official Qwen3-VL tokenizer is not cached; prepare it before training or set a local directory"
            ) from error
    directory = Path(tokenizer)
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix in (".json", ".txt", ".jinja") and not path.name.endswith("index.json")
    )
    if not any(path.name in ("tokenizer.json", "vocab.json") for path in files):
        raise ValueError(f"tokenizer_path {directory}: missing tokenizer assets; provide the complete official tokenizer")
    result["tokenizer"] = {path.name: file_sha256(path) for path in files}
    result["vae_representation"] = {
        "normalization": "qwen-image-vae-normalized-v1",
        "loader": file_sha256(Path(__file__).parents[1] / "qwen_image" / "qwen_image_utils.py"),
        "autoencoder": file_sha256(Path(__file__).parents[1] / "qwen_image" / "qwen_image_autoencoder_kl.py"),
    }
    # Pin the exact locally inventoried tokenizer for cache loading, never a moving Hub ref.
    args.tokenizer_path = str(directory.resolve())
    result["representation"] = {
        "layers": [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
        "max_length": 512,
        "hidden": 2560,
        "prompt": "krea2-qwen3-vl-v1",
        "dtype": "bfloat16",
        "encoder_implementation": file_sha256(Path(__file__).parents[1] / "krea2" / "krea2_encoder.py"),
    }
    return result


class ValidationInputs:
    def __init__(self, args, train_records, val_records, components):
        self.args = args
        self.train_records = train_records
        self.source_records = train_records + val_records
        self.records = _unique_val(val_records)
        self.components = components
        self.manifest = {}
        self.saved_manifest = None
        self.manifest_path = Path(args.output_dir or ".") / "val_manifest.json"
        if getattr(args, "resume", None):
            path = Path(args.resume) / "val_manifest.json"
            if not path.is_file():
                raise ValueError(f"{path}: resume manifest missing; use a checkpoint from this validation experiment")
            self.saved_manifest = json.loads(path.read_text(encoding="utf-8"))
        elif self.manifest_path.exists():
            self.saved_manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.fixed = self._fixed()
        if self.saved_manifest is not None and self.saved_manifest.get("fixed") != self.fixed:
            raise ValueError(
                f"{self.manifest_path}: fixed validation inputs/components/noise changed; restore original inputs or start a separate experiment"
            )
        missing = self.check_caches(allow_missing=getattr(args, "auto_cache", False) and not getattr(args, "resume", None))
        # Missing files must reach auto-cache before their bytes can be hashed.
        # Preparation still compares the complete result with the frozen manifest.
        if self.saved_manifest is not None and not missing:
            current = self._manifest()
            if current != self.saved_manifest:
                raise ValueError(f"{self.manifest_path}: frozen cache contents changed; restore original cache files")
            self.manifest = current

    def _fixed(self):
        return {
            "schema": 1,
            "records": [record.semantic() for record in self.records],
            "components": self.components,
            "noise": {key: getattr(self.args, key, None) for key in ("val_seed_noise", "val_level_noise_n", "val_seed_noise_n")},
            "generator": {"device": "cpu", "dtype": "float32"},
        }

    def provenance(self, record, kind):
        components = (
            {"vae": self.components["vae"]}
            if kind == "latent"
            else {key: self.components[key] for key in ("text_encoder", "tokenizer", "representation")}
        )
        semantic = record.semantic()
        if kind == "latent":
            semantic.pop("caption_sha256")
        return {"krea2_provenance": _digest({"source": semantic, "components": components, "kind": kind})}

    def check_caches(self, allow_missing=False):
        missing = []
        for record in self.source_records:
            for kind, path in (("latent", record.latent_cache_path), ("text", record.text_cache_path)):
                if not Path(path).is_file():
                    missing.append((record, kind))
                    if allow_missing:
                        continue
                    raise ValueError(
                        f"{record.dataset_config}: {path}: missing {kind} cache; enable auto_cache or explicitly prepare the cache"
                    )
                try:
                    with safe_open(path, framework="pt", device="cpu") as cache:
                        metadata = cache.metadata() or {}
                        if (
                            metadata.get("architecture") != "krea2"
                            or metadata.get("krea2_provenance") != self.provenance(record, kind)["krea2_provenance"]
                        ):
                            raise ValueError("architecture/source/component/processing provenance is incompatible or unknown")
                        keys = list(cache.keys())
                        prefix = "latents_" if kind == "latent" else "varlen_krea2_vl_embed_"
                        if len(keys) != 1 or not keys[0].startswith(prefix):
                            raise ValueError("unexpected tensor keys")
                        tensor = cache.get_slice(keys[0])
                        shape = tensor.get_shape()
                        expected = [16, 1, record.bucket_size[1] // 8, record.bucket_size[0] // 8]
                        if (kind == "latent" and shape != expected) or (
                            kind == "text" and (len(shape) != 3 or shape[0] < 1 or shape[1:] != [12, 2560])
                        ):
                            raise ValueError(f"incompatible {kind} shape {shape}")
                        dtype = tensor.get_dtype()
                        suffix = {"F32": "float32", "BF16": "bfloat16", "F16": "float16"}.get(dtype)
                        if suffix is None or not keys[0].endswith("_" + suffix):
                            raise ValueError("unsupported or mislabeled cache dtype")
                        expected_key = (
                            f"latents_1x{shape[2]}x{shape[3]}_{suffix}" if kind == "latent" else f"varlen_krea2_vl_embed_{suffix}"
                        )
                        if keys[0] != expected_key:
                            raise ValueError("tensor key does not match actual shape/dtype")
                        if metadata.get("format_version") != "1.0.1":
                            raise ValueError("unsupported cache format version")
                        if kind == "latent" and (
                            metadata.get("width") != str(record.original_size[0])
                            or metadata.get("height") != str(record.original_size[1])
                        ):
                            raise ValueError("original image dimensions mismatch")
                        if kind == "text" and metadata.get("caption1") != record.caption:
                            raise ValueError("caption mismatch")
                except Exception as error:
                    raise ValueError(
                        f"{record.dataset_config}: {path}: {error}; restore matching cache or explicitly recache in a separate directory"
                    ) from error
        return missing

    def _manifest(self):
        return {
            "fixed": self.fixed,
            "caches": [
                {
                    "image_sha256": record.image_sha256,
                    "latent": file_sha256(record.latent_cache_path),
                    "text": file_sha256(record.text_cache_path),
                }
                for record in self.records
            ],
        }

    def _validate_sources(self):
        train, _ = _inventory(self.args.dataset_config, self.args)
        if sorted((record.semantic() for record in train), key=lambda record: record["image_sha256"]) != sorted(
            (record.semantic() for record in self.train_records), key=lambda record: record["image_sha256"]
        ):
            raise ValueError("train dataset changed during cache preparation; restore inputs and retry")

    def validate_unchanged(self):
        records, _ = _inventory(self.args.val_dataset_config, self.args, validation=True) if self.records else ([], [])
        records = _unique_val(records)
        if [record.semantic() for record in records] != self.fixed["records"]:
            raise ValueError("val dataset: fixed images/captions/processing changed; restore the original validation inputs")
        self.check_caches()
        if self._manifest() != self.manifest:
            raise ValueError("val manifest: fixed cache bytes changed; restore original caches")

    @staticmethod
    def load_batch(record):
        from safetensors.torch import load_file

        latent = load_file(record.latent_cache_path)
        text = load_file(record.text_cache_path)
        return {"latents": next(iter(latent.values())).unsqueeze(0), "krea2_vl_embed": [next(iter(text.values()))]}


def _unique_val(records):
    unique = {}
    for record in sorted(records, key=lambda item: (item.image_sha256, item.image_path)):
        previous = unique.get(record.image_sha256)
        if previous and previous.semantic() != record.semantic():
            raise ValueError(
                f"{record.image_path}: duplicate image has incompatible caption/processing; remove the conflicting copy"
            )
        unique.setdefault(record.image_sha256, record)
    return list(unique.values())


def preflight_validation_inputs(args):
    validate_validation_options(args)
    if not (
        getattr(args, "val_dataset_config", None) or getattr(args, "experiment_mode", False) or getattr(args, "auto_cache", False)
    ):
        return None
    train, train_dirs = _inventory(args.dataset_config, args)
    val, val_dirs = (
        _inventory(args.val_dataset_config, args, validation=True) if getattr(args, "val_dataset_config", None) else ([], [])
    )
    if set(train_dirs) & set(val_dirs):
        raise ValueError("train/val cache_directory overlap; use separate cache/train and cache/val directories")
    overlap = {record.image_sha256 for record in train} & {record.image_sha256 for record in val}
    if overlap:
        raise ValueError(f"train/val image content overlap ({len(overlap)} SHA-256 IDs); provide separate datasets")
    expected_train = {str(Path(record.latent_cache_path).resolve()) for record in train}
    for directory in train_dirs:
        for path in Path(directory).glob("*_kr2.safetensors"):
            if str(path.resolve()) not in expected_train:
                raise ValueError(
                    f"{path}: stale train cache would add an uninventoried sample; use a separate clean cache directory"
                )
    paths = {}
    for record in train + val:
        for path in (record.latent_cache_path, record.text_cache_path):
            if path in paths and paths[path] != record.image_path:
                raise ValueError(
                    f"{path}: image stem cache collision between {paths[path]} and {record.image_path}; use separate cache directories"
                )
            paths[path] = record.image_path
    _unique_val(val)  # Reject conflicts before inspecting component files.
    return ValidationInputs(args, train, val, _component_identities(args))


def _prepare_validation_caches_main(args, inputs, accelerator):
    if inputs is None:
        return
    if accelerator.is_main_process:
        missing = inputs.check_caches(allow_missing=getattr(args, "auto_cache", False))
        from musubi_tuner import krea2_cache_latents, krea2_cache_text_encoder_outputs

        for config in sorted({record.dataset_config for record, _ in missing}):
            for kind, module in (("latent", krea2_cache_latents), ("text", krea2_cache_text_encoder_outputs)):
                if not any(record.dataset_config == config and missing_kind == kind for record, missing_kind in missing):
                    continue
                argv = ["--dataset_config", config, "--skip_existing", "--keep_cache", "--num_workers", "1", "--batch_size", "1"]
                argv += (
                    ["--vae", args.vae]
                    if kind == "latent"
                    else ["--text_encoder", args.text_encoder, "--tokenizer_path", args.tokenizer_path]
                )
                cache_args = module.setup_parser().parse_args(argv)
                cache_args.experiment_mode = getattr(args, "experiment_mode", False)
                lookup = {os.path.normcase(os.path.abspath(record.image_path)): record for record in inputs.source_records}
                cache_args.cache_provenance = lambda item, k=kind: inputs.provenance(
                    lookup[os.path.normcase(os.path.abspath(item.item_key))], k
                )
                module.main(cache_args)
        inputs._validate_sources()
        current, _ = _inventory(args.val_dataset_config, args, validation=True) if inputs.records else ([], [])
        if [record.semantic() for record in _unique_val(current)] != inputs.fixed["records"]:
            raise ValueError("val inputs changed during cache preparation; restore original validation inputs")
        inputs.check_caches()
        final = inputs._manifest()
        if inputs.saved_manifest is not None and final != inputs.saved_manifest:
            raise ValueError("val manifest changed during preparation; restore original inputs/caches")
        inputs.manifest = final
        inputs.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if inputs.saved_manifest is None or not inputs.manifest_path.exists():
            temporary = inputs.manifest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
            temporary.replace(inputs.manifest_path)


def prepare_validation_caches(args, inputs, accelerator):
    if inputs is None:
        return
    error = None
    if accelerator.is_main_process:
        try:
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            try:
                with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                    _prepare_validation_caches_main(args, inputs, accelerator)
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
        except Exception as exception:
            error = f"cache preparation failed: {exception}"
    status = [error]
    if getattr(accelerator, "num_processes", 1) > 1:
        from accelerate.utils import broadcast_object_list

        broadcast_object_list(status, from_process=0)
    if status[0] is not None:
        raise ValueError(status[0])
    accelerator.wait_for_everyone()
    inputs.check_caches()
    inputs.manifest = inputs._manifest()
    inputs.saved_manifest = json.loads(inputs.manifest_path.read_text(encoding="utf-8"))
    if inputs.manifest != inputs.saved_manifest:
        raise ValueError("val manifest differs between processes; use a shared consistent experiment directory")
