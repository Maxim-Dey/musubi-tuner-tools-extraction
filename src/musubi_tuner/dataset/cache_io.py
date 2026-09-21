from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING
import torch
from safetensors.torch import save_file
from musubi_tuner.utils import safetensors_utils
from musubi_tuner.utils.model_utils import dtype_to_str, remove_dtype_suffix
import logging

if TYPE_CHECKING:
    from musubi_tuner.dataset.image_video_dataset import ItemInfo

logger = logging.getLogger(__name__)


def save_latent_cache_flux_2(
    item_info: ItemInfo, latent: torch.Tensor, control_latent: Optional[list[torch.Tensor]], arch_full: str
):
    """Flux 2 architecture"""
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"
    assert control_latent is None or all(cl.dim() == 3 for cl in control_latent), (
        "control_latent should be 3D tensor (channel, height, width) or None"
    )

    _, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    if control_latent is not None:
        for i, cl in enumerate(control_latent):
            _, H, W = cl.shape
            sd[f"latents_control_{i}_{H}x{W}_{dtype_str}"] = cl.detach().cpu().contiguous()

    save_latent_cache_common(item_info, sd, arch_full)


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
    if item_info.frame_count is not None:
        metadata["frame_count"] = f"{item_info.frame_count}"

    for key, value in sd.items():
        # NaN check and show warning, replace NaN with 0
        if torch.isnan(value).any():
            logger.warning(f"{key} tensor has NaN: {item_info.item_key}, replace NaN with 0")
            value[torch.isnan(value)] = 0

    latent_dir = os.path.dirname(item_info.latent_cache_path)
    os.makedirs(latent_dir, exist_ok=True)

    save_file(sd, item_info.latent_cache_path, metadata=metadata)


def save_text_encoder_output_cache_flux_2(item_info: ItemInfo, ctx_vec: torch.Tensor, arch_full: str):
    """Flux 2 architecture."""

    sd = {}
    dtype_str = dtype_to_str(ctx_vec.dtype)
    sd[f"ctx_vec_{dtype_str}"] = ctx_vec.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, arch_full)


def save_text_encoder_output_cache_common(
    item_info: ItemInfo,
    sd: dict[str, torch.Tensor],
    arch_fullname: str,
    merge_existing: bool = True,
    additional_metadata: Optional[dict[str, str]] = None,
):
    # Merge preserves other logical keys and replaces earlier dtype variants.
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
