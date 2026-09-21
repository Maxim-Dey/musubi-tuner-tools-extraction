# LoRA module for FLUX.2

import ast
import math
import re
from typing import Dict, List, Optional
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora


FLUX_2_TARGET_REPLACE_MODULES = ["DoubleStreamBlock", "SingleStreamBlock"]


def validate_network_args(args):
    from musubi_tuner.training.parser_common import config_error, parse_nested_args

    if args.network_module not in ("networks.lora_flux_2", "musubi_tuner.networks.lora_flux_2"):
        raise config_error(
            args,
            "network_module",
            args.network_module,
            "only standard Dev LoRA is supported",
            "use networks.lora_flux_2 or musubi_tuner.networks.lora_flux_2",
        )
    values = parse_nested_args(args, "network_args", literal=False)
    allowed = {
        "conv_dim",
        "conv_alpha",
        "rank_dropout",
        "module_dropout",
        "verbose",
        "exclude_patterns",
        "include_patterns",
        "loraplus_lr_ratio",
    }
    for key, raw in values.items():
        try:
            if key not in allowed:
                raise ValueError("unknown or excluded adapter argument")
            value = ast.literal_eval(raw)
            if key.endswith("_patterns"):
                if not isinstance(value, list if key == "exclude_patterns" else (list, tuple)) or not all(
                    isinstance(pattern, str) for pattern in value
                ):
                    raise ValueError("expected a list of regular expressions")
                for pattern in value:
                    re.compile(pattern)
            elif key == "verbose":
                if type(value) is not bool:
                    raise ValueError("expected True or False")
            elif key == "conv_dim":
                if type(value) is not int or value <= 0:
                    raise ValueError("expected a positive integer")
            elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("expected a finite nonnegative number")
            elif key.endswith("_dropout") and (value > 1 or (key == "rank_dropout" and value == 1)):
                raise ValueError("dropout must be in [0, 1]; rank_dropout must be below 1")
        except (ValueError, SyntaxError, re.error) as error:
            raise config_error(
                args, "network_args", f"{key}={raw}", str(error), f"use a valid standard LoRA argument from {sorted(allowed)}"
            ) from error


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    # add default exclude patterns
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = [r".*(img_mod\.lin|txt_mod\.lin|modulation\.lin).*"]
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    # exclude if 'norm' in the name of the module
    exclude_patterns.append(r".*(norm).*")

    kwargs["exclude_patterns"] = exclude_patterns

    return lora.create_network(
        FLUX_2_TARGET_REPLACE_MODULES,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    return lora.create_network_from_weights(
        FLUX_2_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
