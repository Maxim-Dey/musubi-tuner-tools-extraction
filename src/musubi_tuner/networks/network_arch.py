"""Architecture detection and configuration for network modules (LoHa, LoKr, etc.)."""

import logging

logger = logging.getLogger(__name__)


def detect_arch_config(unet):
    """Detect architecture from model structure.

    Returns: (target_replace_modules, default_exclude_patterns)
    """
    module_class_names = set()
    for module in unet.modules():
        module_class_names.add(type(module).__name__)

    if "QwenImageTransformerBlock" in module_class_names:
        from .lora_qwen_image import QWEN_IMAGE_TARGET_REPLACE_MODULES

        return QWEN_IMAGE_TARGET_REPLACE_MODULES, [r".*(_mod_).*"]

    raise ValueError(f"Cannot auto-detect architecture. Module classes found: {sorted(module_class_names)}")
