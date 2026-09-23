"""Explicit completed-update metadata for Qwen validation state saves."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile

import torch.distributed as dist


STATE_FILENAME = "val_loss_state.json"
STATE_VERSION = "qwen-image-val-loss-state-v1"
CONTROL_NAMES = ("val_every_n_steps", "val_seed_noise", "val_level_noise_n", "val_seed_noise_n")
_FINGERPRINT = re.compile(r"[0-9a-f]{64}")


def _validate_controls(values: dict) -> dict[str, int]:
    if any(type(value) is not int for value in values.values()):
        raise ValueError("validation controls must be exact integers; correct the effective training configuration")
    if values["val_every_n_steps"] < 1 or values["val_level_noise_n"] < 2 or values["val_level_noise_n"] % 2:
        raise ValueError("validation interval/level controls are invalid; correct the effective training configuration")
    if values["val_seed_noise_n"] < 1:
        raise ValueError("val_seed_noise_n must be >= 1; correct the effective training configuration")
    return values


def _controls(args) -> dict[str, int]:
    return _validate_controls({name: getattr(args, name) for name in CONTROL_NAMES})


def _validate_payload(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "version", "model_version", "absolute_completed_step", "validation_fingerprint", "controls"
    }:
        raise ValueError("invalid metadata fields")
    if value["version"] != STATE_VERSION:
        raise ValueError(f"unsupported version {value['version']!r}")
    if value["model_version"] != "original":
        raise ValueError("model_version must be 'original'")
    step = value["absolute_completed_step"]
    if type(step) is not int or step < 0:
        raise ValueError("absolute_completed_step must be a nonnegative exact integer")
    fingerprint = value["validation_fingerprint"]
    if not isinstance(fingerprint, str) or _FINGERPRINT.fullmatch(fingerprint) is None:
        raise ValueError("validation_fingerprint must be a lowercase 64-hex SHA-256 digest")
    controls = value["controls"]
    if not isinstance(controls, dict) or set(controls) != set(CONTROL_NAMES):
        raise ValueError("controls must contain exactly the four effective validation integers")
    _validate_controls(controls)
    return value


def build_validation_state(args, manifest, absolute_completed_step: int) -> dict:
    """Build the small sidecar payload from a completed update and frozen input."""
    payload = {
        "version": STATE_VERSION,
        "model_version": "original",
        "absolute_completed_step": absolute_completed_step,
        "validation_fingerprint": manifest.fingerprint,
        "controls": _controls(args),
    }
    return _validate_payload(payload)


def write_validation_state(state_dir: str | os.PathLike, payload: dict) -> Path:
    """Publish metadata only after all ranks complete their Accelerate state save."""
    _validate_payload(payload)
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    target = state_dir / STATE_FILENAME
    descriptor, temporary = tempfile.mkstemp(prefix=".val_loss_state-", suffix=".tmp", dir=state_dir)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def collective_error(accelerator, local_error: str | None) -> str | None:
    """Return the first rank-labelled error to every rank, if any."""
    if not dist.is_available() or not dist.is_initialized():
        if getattr(accelerator, "num_processes", 1) > 1:
            return "distributed process group is unavailable; restart the Accelerate run"
        return local_error
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_error)
    return next((f"rank {rank}: {error}" for rank, error in enumerate(gathered) if error), None)


def load_validation_resume_step(accelerator, input_dir: str | os.PathLike, args, manifest) -> int:
    """Require matching metadata from the exact directory passed to load_state."""
    path = Path(input_dir) / STATE_FILENAME
    step = None
    local_error = None
    try:
        if manifest is None:
            raise ValueError("validation inputs were not preflighted; correct val_dataset_config")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"missing or malformed {STATE_FILENAME}: {error}") from error
        _validate_payload(payload)
        if payload["validation_fingerprint"] != manifest.fingerprint:
            raise ValueError("validation fingerprint differs from the frozen Stage 1 input")
        if payload["controls"] != _controls(args):
            raise ValueError("validation controls differ from the saved effective protocol")
        step = payload["absolute_completed_step"]
    except (AttributeError, TypeError, ValueError) as error:
        local_error = str(error)

    error = collective_error(accelerator, local_error)
    if error is not None:
        raise ValueError(
            f"{path}: enabled validation resume rejected: {error}; restore the matching state and inputs or start a new experiment"
        )
    if dist.is_available() and dist.is_initialized():
        steps = [None] * dist.get_world_size()
        dist.all_gather_object(steps, step)
        if len(set(steps)) != 1:
            raise ValueError(f"{path}: ranks disagree on absolute_completed_step; restore a consistent state")
    return step
