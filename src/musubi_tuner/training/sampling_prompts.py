"""Image prompt loading, sampling triggers and PNG saving used during training."""

import json
import math
import os
import re

import numpy as np
from PIL import Image
import torch
import torchvision
from einops import rearrange
import toml


PROMPT_FIELDS = {"prompt", "negative_prompt", "width", "height", "sample_steps", "cfg_scale", "seed", "enum"}


def _validate_prompt(prompt: dict, source: str) -> dict:
    if not isinstance(prompt, dict):
        raise ValueError(f"{source}: expected an image prompt object.")
    unknown = set(prompt) - PROMPT_FIELDS
    if unknown:
        raise ValueError(f"{source}: unsupported image prompt fields {sorted(unknown)}; remove them.")
    for key in ("prompt", "negative_prompt"):
        if key in prompt and not isinstance(prompt[key], str):
            raise ValueError(f"{source}: {key} must be a string.")
    for key in ("width", "height", "sample_steps", "seed", "enum"):
        if key in prompt and type(prompt[key]) is not int:
            raise ValueError(f"{source}: {key} must be an integer.")
    for key in ("width", "height"):
        if key in prompt and prompt[key] < 8:
            raise ValueError(f"{source}: {key} must be at least 8 before image-grid alignment.")
    if "sample_steps" in prompt and prompt["sample_steps"] < 1:
        raise ValueError(f"{source}: sample_steps must be positive.")
    if "seed" in prompt and not -(2**63) <= prompt["seed"] < 2**64:
        raise ValueError(f"{source}: seed must be in the PyTorch seed range [-2**63, 2**64-1].")
    if "cfg_scale" in prompt:
        value = prompt["cfg_scale"]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{source}: cfg_scale must be a finite number.")
    return prompt


def line_to_prompt_dict(line: str, source: str = "sample prompt") -> dict:
    prompt_args = line.split(" --")
    prompt = {"prompt": prompt_args[0]}
    names = {"w": "width", "h": "height", "d": "seed", "s": "sample_steps", "l": "cfg_scale", "n": "negative_prompt"}
    for option in prompt_args[1:]:
        match = re.fullmatch(r"([a-z]+)\s+(.+)", option, re.IGNORECASE)
        if match is None or match[1].lower() not in names:
            raise ValueError(f"{source}: unsupported or malformed option --{option}; use --w/--h/--d/--s/--l/--n.")
        name = names[match[1].lower()]
        raw = match[2]
        try:
            value = raw if name == "negative_prompt" else float(raw) if name == "cfg_scale" else int(raw)
        except ValueError as exc:
            raise ValueError(f"{source}: invalid value for --{match[1]}: {raw!r}.") from exc
        if name == "sample_steps":
            value = max(1, min(1000, value))
        prompt[name] = value
    return _validate_prompt(prompt, source)


def load_prompts(prompt_file: str) -> list[dict]:
    prompt_file = os.fspath(prompt_file)
    suffix = os.path.splitext(prompt_file)[1].lower()
    with open(prompt_file, "r", encoding="utf-8") as stream:
        if suffix == ".txt":
            prompts = [line.strip() for line in stream if line.strip() and not line.lstrip().startswith("#")]
        elif suffix == ".toml":
            data = toml.load(stream)
            if set(data) != {"prompt"} or not isinstance(data["prompt"], dict):
                raise ValueError(f"{prompt_file}: expected [prompt] and [[prompt.subset]] image prompts.")
            defaults = data["prompt"].copy()
            subsets = defaults.pop("subset", None)
            if not isinstance(subsets, list) or any(not isinstance(item, dict) for item in subsets):
                raise ValueError(f"{prompt_file}: prompt.subset must be a list of image prompt objects.")
            _validate_prompt(defaults, f"{prompt_file} [prompt]")
            prompts = [{**defaults, **subset} for subset in subsets]
        elif suffix == ".json":
            prompts = json.load(stream)
        else:
            raise ValueError(f"{prompt_file}: unsupported sample prompt format; use .txt, .toml or .json.")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"{prompt_file}: provide at least one image prompt.")
    result = []
    for index, prompt in enumerate(prompts):
        source = f"{prompt_file}: prompt {index + 1}"
        prompt = line_to_prompt_dict(prompt, source) if isinstance(prompt, str) else _validate_prompt(prompt, source).copy()
        prompt["enum"] = index
        result.append(prompt)
    return result


def should_sample_images(args, steps, epoch=None):
    if steps == 0:
        if not args.sample_at_first:
            return False
    else:
        should_sample_by_steps = args.sample_every_n_steps is not None and steps % args.sample_every_n_steps == 0
        should_sample_by_epochs = (
            args.sample_every_n_epochs is not None and epoch is not None and epoch % args.sample_every_n_epochs == 0
        )
        if not should_sample_by_steps and not should_sample_by_epochs:
            return False
    return True


def save_images_grid(
    videos: torch.Tensor, parent_dir: str, image_name: str, rescale: bool = False, n_rows: int = 1, create_subdir=True
) -> list[str]:
    """Preserve the training PNG grid format for (B,C,1,H,W) image tensors."""
    if videos.ndim != 5 or videos.shape[2] != 1:
        raise ValueError("image samples must have shape (B,C,1,H,W).")
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0, 1)
        outputs.append((x * 255).numpy().astype(np.uint8))
    output_dir = os.path.join(parent_dir, image_name) if create_subdir else parent_dir
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    for index, pixels in enumerate(outputs):
        image_path = os.path.join(output_dir, f"{image_name}_{index:03d}.png")
        image_paths.append(image_path)
        Image.fromarray(pixels).save(image_path)
    return image_paths
