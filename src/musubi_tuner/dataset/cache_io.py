from __future__ import annotations

import hashlib
import os
from typing import Optional, TYPE_CHECKING

import torch
from safetensors.torch import save_file

from musubi_tuner.dataset.architectures import ARCHITECTURE_QWEN_IMAGE_FULL
from musubi_tuner.utils import safetensors_utils
from musubi_tuner.utils.model_utils import dtype_to_str, remove_dtype_suffix

if TYPE_CHECKING:
    from musubi_tuner.dataset.image_video_dataset import ItemInfo

import logging

logger = logging.getLogger(__name__)


def source_image_sha256(image_path: str) -> str:
    digest = hashlib.sha256()
    with open(image_path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_latent_cache_qwen_image(
    item_info: ItemInfo, latent: torch.Tensor, *, source_image_sha256: Optional[str] = None
):
    """Original Qwen image cache: [C,1,H,W], unchanged serialized format."""
    if latent.ndim != 4 or latent.shape[1] != 1:
        raise ValueError(f"{item_info.item_key}: latent must have shape [C,1,H,W]; cache an original image")
    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    additional = {"source_image_sha256": source_image_sha256} if source_image_sha256 is not None else None
    save_latent_cache_common(item_info, sd, ARCHITECTURE_QWEN_IMAGE_FULL, additional_metadata=additional)


def _merge_cache_metadata(required: dict[str, str], additional: Optional[dict[str, str]]) -> dict[str, str]:
    metadata = dict(additional or {})
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()):
        raise ValueError("Safetensors metadata keys and values must be strings")
    metadata.update(required)
    return metadata


def save_latent_cache_common(
    item_info: ItemInfo,
    sd: dict[str, torch.Tensor],
    arch_fullname: str,
    additional_metadata: Optional[dict[str, str]] = None,
):
    metadata = _merge_cache_metadata(
        {
            "architecture": arch_fullname,
            "width": f"{item_info.original_size[0]}",
            "height": f"{item_info.original_size[1]}",
            "format_version": "1.0.1",
        },
        additional_metadata,
    )

    for key, value in sd.items():
        # NaN check and show warning, replace NaN with 0
        if torch.isnan(value).any():
            logger.warning(f"{key} tensor has NaN: {item_info.item_key}, replace NaN with 0")
            value[torch.isnan(value)] = 0

    latent_dir = os.path.dirname(item_info.latent_cache_path)
    os.makedirs(latent_dir, exist_ok=True)

    save_file(sd, item_info.latent_cache_path, metadata=metadata)


def save_text_encoder_output_cache_qwen_image(
    item_info: ItemInfo, embed: torch.Tensor, *, source_image_sha256: Optional[str] = None
):
    """Qwen-Image architecture."""
    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_vl_embed_{dtype_str}"] = embed.detach().cpu()

    additional = {"source_image_sha256": source_image_sha256} if source_image_sha256 is not None else None
    save_text_encoder_output_cache_common(
        item_info,
        sd,
        ARCHITECTURE_QWEN_IMAGE_FULL,
        merge_existing=source_image_sha256 is None,
        additional_metadata=additional,
    )


def save_text_encoder_output_cache_common(
    item_info: ItemInfo,
    sd: dict[str, torch.Tensor],
    arch_fullname: str,
    merge_existing: bool = True,
    additional_metadata: Optional[dict[str, str]] = None,
):
    # merge_existing keeps keys written by previous passes (e.g. HunyuanVideo caches LLM and CLIP separately).
    # Single-pass architectures that write their full key set at once should pass merge_existing=False so the
    # cache is overwritten fresh, dropping any stale keys (e.g. optionals/dtypes) left from an earlier run.
    for key, value in sd.items():
        # NaN check and show warning, replace NaN with 0
        if torch.isnan(value).any():
            logger.warning(f"{key} tensor has NaN: {item_info.item_key}, replace NaN with 0")
            value[torch.isnan(value)] = 0

    metadata = _merge_cache_metadata(
        {
            "architecture": arch_fullname,
            "caption1": item_info.caption,
            "format_version": "1.0.1",
        },
        additional_metadata,
    )
    if merge_existing and os.path.exists(item_info.text_encoder_output_cache_path):
        # load existing cache and update metadata
        new_key_bases = {remove_dtype_suffix(key) for key in sd}  # logical keys (dtype stripped) just written
        with safetensors_utils.MemoryEfficientSafeOpen(item_info.text_encoder_output_cache_path) as f:
            existing_metadata = f.metadata()
            for key in f.keys():
                # Skip any existing key superseded by a freshly written one. Comparing on the dtype-stripped base
                # (not the exact key) also drops a stale copy written in another precision, e.g. re-caching after
                # toggling fp8; otherwise both dtype variants would survive and collide under one key on load.
                if remove_dtype_suffix(key) in new_key_bases:
                    continue
                sd[key] = f.get_tensor(key)

        assert existing_metadata["architecture"] == metadata["architecture"], "architecture mismatch"
        if existing_metadata["caption1"] != metadata["caption1"]:
            logger.warning(f"caption mismatch: existing={existing_metadata['caption1']}, new={metadata['caption1']}, overwrite")
        # TODO verify format_version

        existing_metadata.pop("caption1", None)
        existing_metadata.pop("format_version", None)
        metadata.update(existing_metadata)  # copy existing metadata except caption and format_version
    else:
        text_encoder_output_dir = os.path.dirname(item_info.text_encoder_output_cache_path)
        os.makedirs(text_encoder_output_dir, exist_ok=True)

    safetensors_utils.mem_eff_save_file(sd, item_info.text_encoder_output_cache_path, metadata=metadata)
