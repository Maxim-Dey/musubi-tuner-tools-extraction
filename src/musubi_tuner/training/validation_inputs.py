"""Strict Qwen-Image validation source and cache preparation.

This module prepares fixed records and deterministic noise checks. Validation
loss and resume-state persistence belong to later stages.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Iterator

from PIL import Image
from safetensors import safe_open
import torch
from torch.utils.data import DataLoader

from musubi_tuner.dataset.architectures import ARCHITECTURE_QWEN_IMAGE, ARCHITECTURE_QWEN_IMAGE_FULL
from musubi_tuner.dataset.bucket import BucketSelector
from musubi_tuner.dataset.config_utils import (
    BlueprintGenerator,
    ConfigSanitizer,
    generate_dataset_group_by_blueprint,
    load_user_config,
    validate_dataset_sources,
    validate_role_aware_sources,
)
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.dataset.media_utils import glob_images


ROLES = ("val_familiar", "val_unfamiliar")
_LATENT_KEY = re.compile(r"latents_(\d+)x(\d+)x(\d+)_(.+)")


@dataclass(frozen=True)
class ValidationItem:
    role: str
    image_id: str
    image_path: Path
    caption: str
    original_size: tuple[int, int]
    resolution: tuple[int, int]
    bucket_size: tuple[int, int]
    latent_cache_path: Path
    text_cache_path: Path
    caption_sha256: str
    latent_sha256: str
    text_sha256: str
    enable_bucket: bool
    bucket_no_upscale: bool
    declaration_path: Path
    declaration_sha256: str
    config_path: Path
    config_semantic_sha256: str
    source_directory: Path | None
    source_membership_sha256: str | None
    experiment_root: str | None = None


@dataclass(frozen=True)
class ValidationManifest:
    items_by_role: Mapping[str, tuple[ValidationItem, ...]]
    fingerprint: str

    @property
    def items(self) -> tuple[ValidationItem, ...]:
        return tuple(item for role in ROLES for item in self.items_by_role[role])


@dataclass(frozen=True)
class ValidationCheck:
    item: ValidationItem
    latent: torch.Tensor
    vl_embed: torch.Tensor | None
    i: int
    j: int
    t: float
    group: str
    seed: int
    epsilon: torch.Tensor
    noisy_latent: torch.Tensor
    timestep: float


def _require_cache(
    path: Path,
    kind: str,
    label: str,
    image_id: str,
    original_size: tuple[int, int],
    bucket_size: tuple[int, int],
    caption: str,
) -> torch.Tensor:
    if not path.is_file():
        raise ValueError(f"{label}: {kind} cache {path} is missing; rebuild this Qwen validation cache")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as cache:
            metadata = cache.metadata() or {}
            keys = list(cache.keys())
            if metadata.get("architecture") != ARCHITECTURE_QWEN_IMAGE_FULL:
                raise ValueError("architecture mismatch; rebuild the Qwen cache")
            if metadata.get("source_image_sha256") != image_id:
                raise ValueError("source_image_sha256 mismatch; rebuild the cache from this image")
            if kind == "latent":
                if metadata.get("width") != str(original_size[0]) or metadata.get("height") != str(original_size[1]):
                    raise ValueError("width/height metadata mismatch; rebuild the latent cache")
                matching = [key for key in keys if key.startswith("latents_")]
                if len(matching) != 1:
                    raise ValueError("expected exactly one Qwen latent tensor; rebuild the latent cache")
                key = matching[0]
                match = _LATENT_KEY.fullmatch(key)
                tensor = cache.get_tensor(key)
                if (
                    match is None
                    or tensor.ndim != 4
                    or tensor.shape[0] < 1
                    or tuple(tensor.shape[1:]) != tuple(map(int, match.groups()[:3]))
                    or tensor.shape[1] != 1
                    or tensor.shape[2] < 1
                    or tensor.shape[3] < 1
                ):
                    raise ValueError("invalid Qwen latent tensor key or [C,1,H,W] shape; rebuild the cache")
                expected_hw = (bucket_size[1] // 8, bucket_size[0] // 8)
                if tuple(tensor.shape[2:]) != expected_hw:
                    raise ValueError(f"latent shape does not match selected bucket {bucket_size}; rebuild the cache")
            else:
                if metadata.get("caption1") != caption:
                    raise ValueError("caption1 mismatch; rebuild the text cache from this caption")
                matching = [key for key in keys if key.startswith("varlen_vl_embed_")]
                if len(matching) != 1:
                    raise ValueError("expected exactly one Qwen text embedding; rebuild the text cache")
                tensor = cache.get_tensor(matching[0])
                if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] < 1:
                    raise ValueError("invalid Qwen text embedding [L,D] shape; rebuild the text cache")
            if not torch.isfinite(tensor.float()).all():
                raise ValueError("nonfinite tensor values; rebuild the cache")
    except ValueError as error:
        raise ValueError(f"{label}: {kind} cache {path}: {error}") from error
    except Exception as error:
        raise ValueError(f"{label}: {kind} cache {path} is invalid or corrupt safetensors; rebuild it: {error}") from error
    return tensor


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_record(item: ValidationItem) -> dict:
    return {
        "image_id": item.image_id,
        "caption_sha256": item.caption_sha256,
        "caption": item.caption,
        "original_size": list(item.original_size),
        "resolution": list(item.resolution),
        "enable_bucket": item.enable_bucket,
        "bucket_no_upscale": item.bucket_no_upscale,
        "bucket_size": list(item.bucket_size),
        "latent_sha256": item.latent_sha256,
        "text_sha256": item.text_sha256,
    }


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _effective_config_sha256(blueprint) -> str:
    entries = []
    for dataset in blueprint.dataset_group.datasets:
        params = asdict(dataset.params)
        params["cache_directory"] = params["cache_directory"] or params["image_directory"]
        for path_key in ("image_directory", "image_jsonl_file", "cache_directory"):
            if params[path_key] is not None:
                params[path_key] = os.path.normcase(os.path.realpath(params[path_key]))
        if not params["enable_bucket"]:
            params["bucket_no_upscale"] = False
        entries.append(params)
    entries.sort(key=lambda entry: (entry["role"], _canonical_json(entry)))
    return hashlib.sha256(_canonical_json(entries).encode("utf-8")).hexdigest()


def _directory_membership_sha256(directory: Path) -> str:
    names = [Path(path).name for path in glob_images(str(directory))]
    return hashlib.sha256(_canonical_json(names).encode("utf-8")).hexdigest()


def _fingerprint(items_by_role: Mapping[str, tuple[ValidationItem, ...]], protocol: dict) -> str:
    sets = {}
    for role in ROLES:
        records = [_identity_record(item) for item in items_by_role[role]]
        sets[role] = sorted(records, key=_canonical_json)
    identity = {"version": "qwen-image-validation-input-v1", "protocol": protocol, "sets": sets}
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def prepare_validation_inputs(args: argparse.Namespace, expected_fingerprint: str | None = None) -> ValidationManifest:
    """Validate both declared roles and every exact cache pair before model load."""
    config_path = getattr(args, "val_dataset_config", None)
    if not config_path:
        raise ValueError("val_dataset_config is absent; provide a validation dataset path")
    config_path = Path(config_path)
    experiment_root = getattr(args, "_experiment_root", None)
    user_config = load_user_config(config_path, experiment_root=experiment_root)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
        user_config,
        argparse.Namespace(dataset_config=str(config_path)),
        architecture=ARCHITECTURE_QWEN_IMAGE,
    )
    if any(dataset.params.role is None for dataset in blueprint.dataset_group.datasets):
        raise ValueError(
            f"{config_path}: every val_dataset_config dataset must declare role = 'val_familiar' or 'val_unfamiliar'"
        )
    config_semantic_sha256 = _effective_config_sha256(blueprint)
    group = generate_dataset_group_by_blueprint(blueprint.dataset_group, seed=0, experiment_root=experiment_root)
    validate_dataset_sources(group, str(config_path))
    validate_role_aware_sources(group, str(config_path))

    by_role: dict[str, tuple[ValidationItem, ...]] = {}
    for dataset in group.datasets:
        role = dataset.role
        selector = BucketSelector(dataset.resolution, dataset.enable_bucket, dataset.bucket_no_upscale, dataset.architecture)
        source_directory = Path(dataset.image_directory) if dataset.image_directory is not None else None
        source_membership_sha256 = _directory_membership_sha256(source_directory) if source_directory else None
        items: list[ValidationItem] = []
        for index in range(len(dataset.datasource)):
            label = f"{config_path}: {role} item {index + 1}"
            try:
                source, caption = dataset.datasource.get_caption(index)
                image_path = Path(source)
                image_id = _sha256_file(image_path)
                with Image.open(image_path) as image:
                    original_size = image.size
                    image.verify()
                bucket_size = selector.get_bucket_resolution(original_size)
            except (OSError, ValueError) as error:
                raise ValueError(f"{label}: invalid image or caption: {error}; correct the source") from error

            cache_item = ItemInfo(source, caption, original_size, bucket_size)
            latent_path = Path(dataset.get_latent_cache_path(cache_item))
            text_path = Path(dataset.get_text_encoder_output_cache_path(cache_item))
            _require_cache(latent_path, "latent", label, image_id, original_size, bucket_size, caption)
            _require_cache(text_path, "text", label, image_id, original_size, bucket_size, caption)
            if dataset.image_directory is not None:
                declaration_path = Path(os.path.splitext(source)[0] + dataset.caption_extension)
                caption_sha256 = _sha256_file(declaration_path)
            else:
                declaration_path = Path(dataset.image_jsonl_file)
                caption_sha256 = hashlib.sha256(caption.encode("utf-8")).hexdigest()
            items.append(
                ValidationItem(
                    role=role,
                    image_id=image_id,
                    image_path=image_path,
                    caption=caption,
                    original_size=original_size,
                    resolution=dataset.resolution,
                    bucket_size=bucket_size,
                    latent_cache_path=latent_path,
                    text_cache_path=text_path,
                    caption_sha256=caption_sha256,
                    latent_sha256=_sha256_file(latent_path),
                    text_sha256=_sha256_file(text_path),
                    enable_bucket=dataset.enable_bucket,
                    bucket_no_upscale=dataset.bucket_no_upscale,
                    declaration_path=declaration_path,
                    declaration_sha256=_sha256_file(declaration_path),
                    config_path=config_path,
                    config_semantic_sha256=config_semantic_sha256,
                    source_directory=source_directory,
                    source_membership_sha256=source_membership_sha256,
                    experiment_root=experiment_root,
                )
            )
        by_role[role] = tuple(items)
    frozen_sets = MappingProxyType(dict(by_role))
    protocol = {
        "val_every_n_steps": getattr(args, "val_every_n_steps", 200),
        "val_seed_noise": getattr(args, "val_seed_noise", 42),
        "val_level_noise_n": getattr(args, "val_level_noise_n", 10),
        "val_seed_noise_n": getattr(args, "val_seed_noise_n", 1),
    }
    fingerprint = _fingerprint(frozen_sets, protocol)
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError(f"{config_path}: validation input fingerprint changed; restore the frozen input or start a new experiment")
    return ValidationManifest(items_by_role=frozen_sets, fingerprint=fingerprint)


def stable_hash(val_seed_noise: int, image_sha256: str, i: int, j: int) -> int:
    """Derive the fixed 63-bit seed for one 1-based image/noise check."""
    if type(val_seed_noise) is not int or type(i) is not int or type(j) is not int or i < 1 or j < 1:
        raise ValueError("validation seed and 1-based i/j must be integers; correct the noise configuration")
    if re.fullmatch(r"[0-9a-f]{64}", image_sha256) is None:
        raise ValueError("image_sha256 must be a lowercase 64-character SHA-256 digest")
    message = f"qwen-image-val-noise-v1\n{val_seed_noise}\n{image_sha256}\n{i}\n{j}\n".encode("ascii")
    digest = hashlib.sha256(message).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def noise_levels(val_level_noise_n: int) -> tuple[float, ...]:
    """Return the fixed midpoint mixing levels, low half followed by high."""
    if type(val_level_noise_n) is not int or val_level_noise_n < 2 or val_level_noise_n % 2:
        raise ValueError("val_level_noise_n must be an even integer >= 2")
    return tuple(0.05 + (i - 0.5) * 0.90 / val_level_noise_n for i in range(1, val_level_noise_n + 1))


def _one_item(batch: list[ValidationItem]) -> ValidationItem:
    return batch[0]


def make_validation_loader(manifest: ValidationManifest) -> DataLoader:
    """Iterate one fixed item at a time without consuming global torch RNG."""
    role_order = {role: index for index, role in enumerate(ROLES)}
    items = sorted(manifest.items, key=lambda item: (role_order[item.role], _canonical_json(_identity_record(item))))
    generator = torch.Generator(device="cpu").manual_seed(0)
    return DataLoader(items, batch_size=1, shuffle=False, num_workers=0, collate_fn=_one_item, generator=generator)


def iter_noise_checks(
    item: ValidationItem,
    latent: torch.Tensor,
    *,
    val_seed_noise: int,
    val_level_noise_n: int,
    val_seed_noise_n: int,
    vl_embed: torch.Tensor | None = None,
) -> Iterator[ValidationCheck]:
    """Yield one noise tensor and mixed latent per check; never precompute them."""
    levels = noise_levels(val_level_noise_n)
    if type(val_seed_noise_n) is not int or val_seed_noise_n < 1:
        raise ValueError("val_seed_noise_n must be an integer >= 1")
    for i, t in enumerate(levels, 1):
        for j in range(1, val_seed_noise_n + 1):
            seed = stable_hash(val_seed_noise, item.image_id, i, j)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            epsilon = torch.randn(latent.shape, generator=generator, dtype=torch.float32, device="cpu")
            epsilon = epsilon.to(device=latent.device, dtype=latent.dtype)
            yield ValidationCheck(
                item=item,
                latent=latent,
                vl_embed=vl_embed,
                i=i,
                j=j,
                t=t,
                group="low" if t < 0.5 else "high",
                seed=seed,
                epsilon=epsilon,
                noisy_latent=(1 - t) * latent + t * epsilon,
                timestep=1000 * t,
            )


def read_validation_cache_pair(item: ValidationItem) -> tuple[torch.Tensor, torch.Tensor]:
    """Reject changes to the frozen item before reading one cache pair."""
    label = f"{item.role}: {item.image_path}"
    try:
        user_config = load_user_config(item.config_path, experiment_root=item.experiment_root)
        blueprint = BlueprintGenerator(ConfigSanitizer()).generate(
            user_config,
            argparse.Namespace(dataset_config=str(item.config_path)),
            architecture=ARCHITECTURE_QWEN_IMAGE,
        )
        current_config_sha256 = _effective_config_sha256(blueprint)
    except (OSError, ValueError) as error:
        raise ValueError(f"{label}: frozen validation config changed or is missing; restore it: {error}") from error
    if current_config_sha256 != item.config_semantic_sha256:
        raise ValueError(f"{label}: frozen validation config changed; restore the effective settings or start a new experiment")
    if item.source_directory is not None:
        if _directory_membership_sha256(item.source_directory) != item.source_membership_sha256:
            raise ValueError(f"{label}: validation set membership changed; restore the source directory")
    for source, expected, kind in (
        (item.declaration_path, item.declaration_sha256, "caption/JSONL"),
        (item.image_path, item.image_id, "source image"),
        (item.latent_cache_path, item.latent_sha256, "latent cache"),
        (item.text_cache_path, item.text_sha256, "text cache"),
    ):
        try:
            actual = _sha256_file(source)
        except OSError as error:
            raise ValueError(f"{label}: frozen {kind} {source} is missing; restore the validation input") from error
        if actual != expected:
            raise ValueError(f"{label}: frozen {kind} {source} changed; restore the validation input or start a new experiment")
    latent = _require_cache(
        item.latent_cache_path, "latent", label, item.image_id, item.original_size, item.bucket_size, item.caption
    )
    vl_embed = _require_cache(
        item.text_cache_path, "text", label, item.image_id, item.original_size, item.bucket_size, item.caption
    )
    return latent, vl_embed


def iter_validation_checks(
    manifest: ValidationManifest,
    *,
    val_seed_noise: int,
    val_level_noise_n: int,
    val_seed_noise_n: int,
) -> Iterator[ValidationCheck]:
    """Read one item at a time and yield its checks in deterministic order."""
    for item in make_validation_loader(manifest):
        latent, vl_embed = read_validation_cache_pair(item)
        yield from iter_noise_checks(
            item,
            latent,
            val_seed_noise=val_seed_noise,
            val_level_noise_n=val_level_noise_n,
            val_seed_noise_n=val_seed_noise_n,
            vl_embed=vl_embed,
        )
