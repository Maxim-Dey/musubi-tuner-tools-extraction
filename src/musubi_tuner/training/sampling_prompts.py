"""Original-image prompt readers and unchanged sampling triggers."""

import json
import math
import re
from pathlib import Path
import toml


PROMPT_TYPES = {
    "prompt": str,
    "negative_prompt": str,
    "width": int,
    "height": int,
    "seed": int,
    "sample_steps": int,
    "cfg_scale": float,
    "discrete_flow_shift": float,
}
TEXT_OPTIONS = {
    "w": "width",
    "h": "height",
    "d": "seed",
    "s": "sample_steps",
    "l": "cfg_scale",
    "fs": "discrete_flow_shift",
    "n": "negative_prompt",
}


def validate_prompt(prompt, source):
    if not isinstance(prompt, dict):
        raise ValueError(f"{source}: expected a prompt object or text line; correct the record")
    for key, value in prompt.items():
        if key not in PROMPT_TYPES:
            raise ValueError(f"{source}: unsupported prompt field {key}; remove it for original-image sampling")
        kind = PROMPT_TYPES[key]
        valid = type(value) is kind if kind is not float else type(value) in (int, float) and math.isfinite(value)
        if not valid:
            raise ValueError(f"{source}: {key} has invalid type/value; supply {kind.__name__}")
        minimum = 16 if key in ("width", "height") else 1 if key == "sample_steps" else 0
        if kind in (int, float) and (value < minimum or (key == "discrete_flow_shift" and value == 0)):
            raise ValueError(f"{source}: invalid {key}={value}; use a positive usable value (dimensions >=16)")
    return prompt


def line_to_prompt_dict(line: str, source="prompt") -> dict:
    parts = line.split(" --")
    result = {"prompt": parts[0]}
    for part in parts[1:]:
        match = re.fullmatch(r"([a-z]+)\s+(.+)", part, re.IGNORECASE)
        if not match or match[1].lower() not in TEXT_OPTIONS:
            raise ValueError(f"{source}: unsupported or malformed --{part}; use --w/--h/--d/--s/--l/--fs/--n")
        key = TEXT_OPTIONS[match[1].lower()]
        raw = match[2]
        try:
            if PROMPT_TYPES[key] is int and not re.fullmatch(r"[+-]?\d+", raw):
                raise ValueError("expected an integer")
            result[key] = PROMPT_TYPES[key](raw)
        except ValueError as error:
            raise ValueError(f"{source}: malformed {key}={raw!r}; supply a complete {PROMPT_TYPES[key].__name__} value") from error
    return validate_prompt(result, source)


def load_prompts(prompt_file: str) -> list[dict]:
    path = Path(prompt_file)
    try:
        if path.suffix == ".txt":
            records = [
                (line.strip(), f"{path}:{number}")
                for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
                if line.strip() and not line.lstrip().startswith("#")
            ]
        elif path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError("expected a list of prompts")
            records = [(record, f"{path}: record {i + 1}") for i, record in enumerate(data)]
        elif path.suffix == ".toml":
            data = toml.load(path)
            if set(data) != {"prompt"} or not isinstance(data["prompt"], dict):
                raise ValueError("expected [prompt] and [[prompt.subset]] records")
            common = data["prompt"].copy()
            subsets = common.pop("subset", None)
            validate_prompt(common, f"{path}: prompt defaults")
            if not isinstance(subsets, list) or any(not isinstance(item, dict) for item in subsets):
                raise ValueError("expected [[prompt.subset]] records")
            records = [({**common, **item}, f"{path}: subset {i + 1}") for i, item in enumerate(subsets)]
        else:
            raise ValueError("use a .txt, .toml or .json prompt file")
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"{path}: cannot read prompts: {error}; correct the file") from error
    if not records:
        raise ValueError(f"{path}: no prompts; provide at least one original-image prompt")
    prompts = []
    for record, source in records:
        prompt = line_to_prompt_dict(record, source) if isinstance(record, str) else validate_prompt(record, source)
        prompts.append({**prompt, "enum": len(prompts)})
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
