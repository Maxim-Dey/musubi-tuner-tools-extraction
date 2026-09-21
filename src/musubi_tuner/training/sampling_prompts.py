"""Prompt loading and sampling trigger helpers used during training."""

import json
import math
import re
from pathlib import Path

import toml


def _validate_prompt(record, source, partial=False):
    if not isinstance(record, dict):
        raise ValueError(f"{source}:prompt={record!r}: expected a record; use a prompt string or image prompt mapping")
    supported = {
        "prompt",
        "width",
        "height",
        "frame_count",
        "seed",
        "sample_steps",
        "guidance_scale",
        "discrete_flow_shift",
        "control_image_path",
        "negative_prompt",
        "cfg_scale",
    }
    for key, value in record.items():
        if key not in supported:
            raise ValueError(f"{source}:{key}={value!r}: unknown or internal prompt field; use a supported image prompt field")
        if key in {"prompt", "negative_prompt"}:
            valid = isinstance(value, str) and (key != "prompt" or bool(value.strip()))
        elif key == "control_image_path":
            valid = (isinstance(value, str) and bool(value.strip())) or (
                isinstance(value, list) and bool(value) and all(isinstance(v, str) and bool(v.strip()) for v in value)
            )
        elif key in {"width", "height", "frame_count", "seed", "sample_steps"}:
            valid = type(value) is int and value >= (0 if key == "seed" else 1)
            if key in {"width", "height"}:
                valid = valid and value >= 8
            if key == "frame_count":
                valid = valid and value == 1
        else:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            if key == "discrete_flow_shift":
                valid = valid and value > 0
        if not valid:
            raise ValueError(
                f"{source}:{key}={value!r}: invalid image prompt value; use the documented field type/range (frame_count must be 1)"
            )
    if not partial and not record.get("prompt", "").strip():
        raise ValueError(f"{source}:prompt={record.get('prompt')!r}: empty prompt; provide nonempty text")
    return record


def line_to_prompt_dict(line: str, source="prompt") -> dict:
    prompt_args = line.split(" --")
    result = {"prompt": prompt_args[0]}
    options = {
        "w": "width",
        "h": "height",
        "f": "frame_count",
        "d": "seed",
        "s": "sample_steps",
        "g": "guidance_scale",
        "fs": "discrete_flow_shift",
        "l": "cfg_scale",
        "n": "negative_prompt",
        "ci": "control_image_path",
    }
    for raw in prompt_args[1:]:
        match = re.fullmatch(r"([a-z]+)\s+(.+)", raw.strip(), re.IGNORECASE)
        option = raw.split(maxsplit=1)[0].lower() if raw.strip() else "empty"
        if match is None or option not in options:
            raise ValueError(
                f"{source}:--{option}={raw!r}: unknown or malformed option; use an image prompt switch followed by its value"
            )
        value = match.group(2)
        key = options[option]
        try:
            if option in {"w", "h", "f", "d", "s"}:
                if not re.fullmatch(r"\d+", value):
                    raise ValueError("expected an integer token")
                value = int(value)
                if option == "s":
                    value = max(1, min(1000, value))
            elif option in {"g", "fs", "l"}:
                if not re.fullmatch(r"[\d.]+", value):
                    raise ValueError("expected a decimal token")
                value = float(value)
        except ValueError as error:
            raise ValueError(f"{source}:--{option}={value!r}: {error}; supply a complete numeric value") from error
        if option == "ci":
            result.setdefault(key, []).append(value.strip())
        else:
            result[key] = value
    return _validate_prompt(result, source)


def load_prompts(prompt_file: str) -> list[dict]:
    path = Path(prompt_file)
    try:
        with path.open(encoding="utf-8") as file:
            if path.suffix.lower() == ".txt":
                records = [
                    (line.strip(), f"{path}:line {index}")
                    for index, line in enumerate(file, 1)
                    if line.strip() and not line.lstrip().startswith("#")
                ]
            elif path.suffix.lower() == ".json":
                data = json.load(file)
                if not isinstance(data, list):
                    raise ValueError(f"{path}:prompt={data!r}: expected a JSON list; put prompt records in an array")
                records = [(record, f"{path}:prompt.{index}") for index, record in enumerate(data)]
            elif path.suffix.lower() == ".toml":
                data = toml.load(file)
                if set(data) != {"prompt"} or not isinstance(data["prompt"], dict):
                    raise ValueError(f"{path}:prompt={data!r}: expected a prompt section; use [prompt] with [[prompt.subset]]")
                base = data["prompt"].copy()
                subsets = base.pop("subset", None)
                _validate_prompt(base, f"{path}:prompt", partial=True)
                if not isinstance(subsets, list) or not subsets:
                    raise ValueError(f"{path}:prompt.subset={subsets!r}: expected a nonempty list; use [[prompt.subset]]")
                records = []
                for index, subset in enumerate(subsets):
                    source = f"{path}:prompt.subset.{index}"
                    _validate_prompt(subset, source, partial=True)
                    if base.keys() & subset.keys():
                        raise ValueError(f"{source}={subset!r}: duplicate base/subset fields; specify each field once")
                    records.append((dict(**base, **subset), source))
            else:
                raise ValueError(f"{path}:prompt_file: unsupported extension; use TXT, TOML or JSON")
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}:prompt_file: {error}; provide a readable, valid image prompt file") from error
    if not records:
        raise ValueError(f"{path}:prompt: no usable prompts; provide nonempty prompt text")
    prompts = []
    for index, (record, source) in enumerate(records):
        if isinstance(record, str):
            record = line_to_prompt_dict(record, source)
        else:
            record = _validate_prompt(record, source)
        if isinstance(record.get("control_image_path"), str):
            record = record | {"control_image_path": [record["control_image_path"]]}
        prompts.append(record | {"enum": index})
    return prompts


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
