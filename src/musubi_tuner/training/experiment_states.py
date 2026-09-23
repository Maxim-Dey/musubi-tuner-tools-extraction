"""Complete opt-in Qwen-Image training-state packages."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import struct
import tempfile

from safetensors import safe_open
from PIL import Image
import numpy as np
import torch
import torch.distributed as dist

from musubi_tuner.training.validation_state import build_validation_state, collective_error, write_validation_state


STATE_FILENAME = "experiment_state.json"
STATE_VERSION = "qwen-image-experiment-state-v1"
METRIC_NAMES = frozenset(
    {
        "train_eval_loss_mean", "train_eval_loss_low_noise", "train_eval_loss_high_noise",
        "val_loss_mean", "val_loss_low_noise", "val_loss_high_noise",
    }
)
SAVE_REASONS = frozenset({"periodic", "final", "epoch", "new_best", "sample"})


def _shared_error(accelerator, local_error: str | None, *, loading: bool = False) -> None:
    message = collective_error(accelerator, local_error)
    if message is not None:
        exception = ValueError if loading else RuntimeError
        raise exception(f"experiment state {'load' if loading else 'save'} rejected: {message}")


def _broadcast_main(accelerator, value):
    if dist.is_available() and dist.is_initialized():
        values = [value if accelerator.is_main_process else None]
        dist.broadcast_object_list(values, src=0)
        return values[0]
    if getattr(accelerator, "num_processes", 1) != 1:
        raise RuntimeError("distributed process group is unavailable")
    return value


def _output_root(args) -> Path:
    root = Path(args.experiment_dir).expanduser().resolve()
    output = root / "output"
    configured = Path(args.output_dir).expanduser().resolve()
    if os.path.normcase(os.path.abspath(output)) != os.path.normcase(os.path.abspath(configured)):
        raise ValueError("experiment output_dir must be <experiment_dir>/output")
    return output


def _output_name(args) -> str:
    name = getattr(args, "output_name", None)
    if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("experiment output_name must be a single nonempty path component")
    return name


def _expected_package(output: Path, name: str, step: int, destination: str) -> Path:
    if destination == "current":
        parent = output / "current_training_states"
    elif destination == "best":
        parent = output / "val_training_states" / "val-loss"
    else:
        raise ValueError("destination must be 'current' or 'best'")
    return parent / f"{name}-step-{step}"


def _metrics(value: dict | None) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != METRIC_NAMES:
        raise ValueError("metrics_at_step must contain exactly the six validation values")
    if any(isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) for number in value.values()):
        raise ValueError("metrics_at_step values must be finite numbers")
    return {name: float(value[name]) for name in sorted(METRIC_NAMES)}


def _reasons(value) -> list[str]:
    if isinstance(value, str):
        raise ValueError("save_reasons must be a sequence of reason names")
    try:
        reasons = list(value)
    except TypeError as error:
        raise ValueError("save_reasons must be a sequence of reason names") from error
    if not reasons or any(reason not in SAVE_REASONS for reason in reasons) or len(set(reasons)) != len(reasons):
        raise ValueError("save_reasons must be nonempty, known and unique")
    return reasons


def _payload(args, manifest, step: int, metrics, reasons, sample_count: int = 0) -> tuple[dict, dict]:
    if getattr(args, "save_precision", None) not in {None, "float", "fp32"}:
        raise ValueError("experiment state requires FP32 save_precision")
    if type(sample_count) is not int or sample_count < 0:
        raise ValueError("sample_count must be a nonnegative exact integer")
    stage2 = build_validation_state(args, manifest, step)
    experiment = {
        "version": STATE_VERSION,
        "output_name": _output_name(args),
        "model_version": "original",
        "absolute_completed_step": step,
        "validation_fingerprint": stage2["validation_fingerprint"],
        "controls": stage2["controls"],
        "save_reasons": _reasons(reasons),
        "metrics_at_step": _metrics(metrics),
        "sample_count": sample_count,
    }
    return stage2, experiment


def _write_json(path: Path, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"missing or malformed sidecar {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"malformed sidecar {path.name}: expected an object")
    return value


def _validate_rng_file(path: Path, accelerator) -> None:
    if not path.is_file():
        raise ValueError(f"missing rank RNG file {path.name}")
    try:
        states = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"corrupt rank RNG file {path.name}: {error}") from error
    required = {"step", "random_state", "numpy_random_seed", "torch_manual_seed"}
    if accelerator.device.type == "cuda":
        required.add("torch_cuda_manual_seed")
    if not isinstance(states, dict) or not required.issubset(states):
        raise ValueError(f"corrupt rank RNG file {path.name}: missing required RNG keys")
    if type(states["step"]) is not int or not isinstance(states["random_state"], tuple):
        raise ValueError(f"corrupt rank RNG file {path.name}: invalid RNG metadata")
    if not isinstance(states["numpy_random_seed"], tuple):
        raise ValueError(f"corrupt rank RNG file {path.name}: invalid NumPy state")
    tensor = states["torch_manual_seed"]
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8 or tensor.ndim != 1:
        raise ValueError(f"corrupt rank RNG file {path.name}: invalid torch state")
    try:
        random.Random(0).setstate(states["random_state"])
        np.random.RandomState(0).set_state(states["numpy_random_seed"])
        torch.Generator(device="cpu").set_state(tensor)
    except Exception as error:
        raise ValueError(f"corrupt rank RNG file {path.name}: unusable random state: {error}") from error
    if accelerator.device.type == "cuda":
        cuda_states = states["torch_cuda_manual_seed"]
        device_count = torch.cuda.device_count()
        if (
            device_count < 1
            or not isinstance(cuda_states, (list, tuple))
            or len(cuda_states) != device_count
        ):
            raise ValueError(f"corrupt rank RNG file {path.name}: invalid CUDA device state count")
        for cuda_state in cuda_states:
            if (
                not isinstance(cuda_state, torch.Tensor)
                or cuda_state.device.type != "cpu"
                or cuda_state.dtype != torch.uint8
                or cuda_state.ndim != 1
                or not cuda_state.is_contiguous()
                or cuda_state.numel() not in (8, 16)
            ):
                raise ValueError(f"corrupt rank RNG file {path.name}: invalid CUDA random state")
            if cuda_state.numel() == 16 and struct.unpack("=q", bytes(cuda_state[8:].tolist()))[0] % 4:
                raise ValueError(f"corrupt rank RNG file {path.name}: invalid CUDA random state offset")


def _validate_adapter(path: Path, accelerator, network) -> None:
    model_files = sorted(file.name for file in path.glob("*.safetensors"))
    if model_files != ["model.safetensors"] or list(path.glob("pytorch_model*.bin")) or list(path.glob("model*.bin")):
        raise ValueError("package must contain exactly one model.safetensors and no other model weights")
    expected = accelerator.unwrap_model(network).state_dict()
    try:
        with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as saved:
            names = set(saved.keys())
            if names != set(expected) or any(name.startswith("module.") for name in names):
                raise ValueError("model.safetensors has noncanonical adapter tensor names")
            if any(saved.get_slice(name).get_shape() != list(expected[name].shape) for name in names):
                raise ValueError("model.safetensors has incompatible adapter tensor shapes")
            if any(
                saved.get_slice(name).get_dtype() != ("F32" if expected[name].is_floating_point() else "I64")
                for name in names
            ):
                raise ValueError("model.safetensors must contain FP32 adapter weights and original integer metadata")
    except (OSError, RuntimeError) as error:
        raise ValueError(f"invalid model.safetensors: {error}") from error


def _validate_files(path: Path, accelerator, network) -> None:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"incomplete state package directory: {path}")
    for filename in ("optimizer.bin", "scheduler.bin", "val_loss_state.json", STATE_FILENAME):
        if not (path / filename).is_file():
            raise ValueError(f"incomplete state package: missing {filename}")
    _validate_adapter(path, accelerator, network)
    ranks = getattr(accelerator, "num_processes", 1)
    found = sorted(file.name for file in path.glob("random_states_*.pkl"))
    expected = sorted(f"random_states_{rank}.pkl" for rank in range(ranks))
    if found != expected:
        raise ValueError(f"incomplete state package: expected rank RNG files {expected}, found {found}")
    for filename in expected:
        _validate_rng_file(path / filename, accelerator)


def _verify_samples(path: Path, *, required: bool, expected_count: int | None = None) -> int:
    folder = path / "samples"
    if not folder.exists():
        if required:
            raise ValueError("incomplete sample set: samples directory is missing")
        return 0
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("incomplete sample set: samples path is not a directory")
    images = sorted(folder.glob("*.png"))
    if required and not images:
        raise ValueError("incomplete sample set: no PNG images were produced")
    if expected_count is not None and len(images) != expected_count:
        raise ValueError(f"incomplete sample set: expected {expected_count} PNG images, found {len(images)}")
    for image in images:
        if image.is_symlink() or not image.is_file() or image.stat().st_size == 0:
            raise ValueError(f"incomplete sample PNG: {image.name}")
        try:
            with Image.open(image) as loaded:
                if loaded.format != "PNG":
                    raise ValueError(f"sample image is not PNG: {image.name}")
                loaded.verify()
        except (OSError, SyntaxError) as error:
            raise ValueError(f"invalid sample PNG {image.name}: {error}") from error
    return len(images)


def _validate_sidecars(path: Path, args, manifest, step: int | None = None) -> tuple[int, int]:
    stage2 = _read_json(path / "val_loss_state.json")
    experiment = _read_json(path / STATE_FILENAME)
    if set(experiment) != {
        "version", "output_name", "model_version", "absolute_completed_step", "validation_fingerprint",
        "controls", "save_reasons", "metrics_at_step", "sample_count",
    } or experiment["version"] != STATE_VERSION:
        raise ValueError("invalid experiment sidecar metadata/version")
    loaded_step = experiment["absolute_completed_step"]
    if type(loaded_step) is not int or loaded_step < 0 or (step is not None and loaded_step != step):
        raise ValueError("experiment sidecar step mismatch")
    sample_count = experiment["sample_count"]
    expected_stage2, _ = _payload(
        args, manifest, loaded_step, experiment["metrics_at_step"], experiment["save_reasons"], sample_count
    )
    if stage2 != expected_stage2:
        raise ValueError("Stage 2 sidecar metadata differs from the effective validation protocol")
    if experiment["output_name"] != _output_name(args) or experiment["model_version"] != "original":
        raise ValueError("experiment sidecar output/model metadata mismatch")
    if experiment["validation_fingerprint"] != stage2["validation_fingerprint"] or experiment["controls"] != stage2["controls"]:
        raise ValueError("experiment and Stage 2 sidecar metadata mismatch")
    return loaded_step, sample_count


def _validate_package(path: Path, args, accelerator, network, manifest, *, published: bool) -> int:
    if published:
        output = _output_root(args)
        name = _output_name(args)
        match = re.fullmatch(re.escape(name) + r"-step-(0|[1-9][0-9]*)", path.name)
        if match is None:
            raise ValueError("state package name does not match <output_name>-step-<X>")
        step = int(match.group(1))
        expected_locations = {_expected_package(output, name, step, choice) for choice in ("current", "best")}
        if path.resolve() not in expected_locations or path.is_symlink():
            raise ValueError("state package is outside the selected current/best output trees")
        parent = path.parent
        while parent != output:
            if parent.is_symlink():
                raise ValueError("state package parent is a symlink")
            parent = parent.parent
    else:
        step = None
    _validate_files(path, accelerator, network)
    loaded_step, sample_count = _validate_sidecars(path, args, manifest, step)
    _verify_samples(path, required=sample_count > 0, expected_count=sample_count)
    return loaded_step


def is_new_best(metrics_at_step: dict | None, best_score: float | None) -> bool:
    """A complete finite unfamiliar-set mean wins only by strict improvement."""
    if metrics_at_step is None:
        return False
    metrics = _metrics(metrics_at_step)
    if best_score is not None:
        if isinstance(best_score, bool) or not isinstance(best_score, (int, float)) or not math.isfinite(best_score):
            raise ValueError("stored best validation score must be finite")
    return best_score is None or metrics["val_loss_mean"] < best_score


def read_best_package(args, accelerator, network, manifest) -> tuple[Path | None, float | None]:
    """Read the one owned, complete best package without changing training state."""
    output = _output_root(args)
    name = _output_name(args)
    parent = _expected_package(output, name, 0, "best").parent
    if not parent.exists():
        return None, None
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("best package directory is invalid or is a symlink")
    candidates = []
    for path in parent.iterdir():
        if re.fullmatch(re.escape(name) + r"-step-(0|[1-9][0-9]*)", path.name):
            _validate_package(path, args, accelerator, network, manifest, published=True)
            payload = _read_json(path / STATE_FILENAME)
            metrics = _metrics(payload["metrics_at_step"])
            if metrics is None or "new_best" not in payload["save_reasons"]:
                raise ValueError(f"best package metadata lacks a complete new-best validation event: {path}")
            candidates.append((path, metrics["val_loss_mean"]))
    if len(candidates) > 1:
        raise ValueError("more than one published best package exists")
    return candidates[0] if candidates else (None, None)


def decide_best_event(args, accelerator, network, manifest, metrics_at_step: dict | None) -> bool:
    """Main selects the best candidate; every rank receives its decision or error."""
    result = None
    if accelerator.is_main_process:
        try:
            _, prior_score = read_best_package(args, accelerator, network, manifest)
            result = {"decision": is_new_best(metrics_at_step, prior_score), "error": None}
        except (AttributeError, OSError, TypeError, ValueError) as error:
            result = {"decision": None, "error": f"best decision failed: {error}"}
    result = _broadcast_main(accelerator, result)
    if result["error"] is not None:
        raise ValueError(result["error"])
    return result["decision"]


def _retention_limit(args) -> int | None:
    limit = getattr(args, "save_last_n_steps", None)
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("save_last_n_steps must be a nonnegative integer")
    return limit


def _prune_owned_current(args, accelerator, network, manifest, absolute_step: int) -> None:
    limit = _retention_limit(args)
    if type(absolute_step) is not int or absolute_step < 0:
        raise ValueError("retention step must be a nonnegative exact integer")
    if limit is None:
        return
    output = _output_root(args)
    name = _output_name(args)
    parent = _expected_package(output, name, 0, "current").parent
    if not parent.exists():
        return
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("current package directory is invalid or is a symlink")
    for path in parent.iterdir():
        match = re.fullmatch(re.escape(name) + r"-step-(0|[1-9][0-9]*)", path.name)
        if match is None or path.is_symlink():
            continue
        step = int(match.group(1))
        if step >= absolute_step - limit:
            continue
        try:
            _validate_package(path, args, accelerator, network, manifest, published=True)
        except (OSError, TypeError, ValueError):
            # An incomplete or unrelated folder is not an owned retention target.
            continue
        expected = _expected_package(output, name, step, "current")
        if path.resolve() != expected or path.is_symlink():
            continue
        shutil.rmtree(path)


def prune_current_packages(args, accelerator, network, manifest, absolute_step: int) -> None:
    """Every rank enters; main removes only complete owned current packages."""
    local_error = None
    if accelerator.is_main_process:
        try:
            _prune_owned_current(args, accelerator, network, manifest, absolute_step)
        except (OSError, TypeError, ValueError) as error:
            local_error = f"retention failure: {error}"
    result = _broadcast_main(accelerator, local_error)
    if result is not None:
        raise RuntimeError(f"experiment state retention failed: {result}")


def merge_published_package_reasons(
    args, accelerator, network, manifest, absolute_step: int, additional_reasons,
) -> Path:
    """Add same-step reasons to one published package without another save or sample."""
    local_error = None
    package = None
    try:
        if isinstance(additional_reasons, str):
            raise ValueError("additional save reasons must be a sequence")
        additions = list(additional_reasons)
        if any(reason not in SAVE_REASONS for reason in additions):
            raise ValueError("additional save reasons contain an unknown value")
        output = _output_root(args)
        name = _output_name(args)
        current = _expected_package(output, name, absolute_step, "current")
        best = _expected_package(output, name, absolute_step, "best")
        locations = [path for path in (current, best) if path.exists() or path.is_symlink()]
        if len(locations) != 1:
            raise ValueError("expected exactly one published package at the completed step")
        package = locations[0]
        _validate_package(package, args, accelerator, network, manifest, published=True)
        if "new_best" in additions and package == current:
            raise ValueError("new_best requires promoting the current package to best")
    except (AttributeError, OSError, TypeError, ValueError) as error:
        local_error = str(error)
    _shared_error(accelerator, local_error)

    result = None
    if accelerator.is_main_process:
        try:
            original = _read_json(package / STATE_FILENAME)
            if "sample" in additions and original["sample_count"] == 0:
                raise ValueError("sample reason requires an existing complete PNG set")
            updated = dict(original)
            reasons = list(original["save_reasons"])
            for reason in additions:
                if reason not in reasons:
                    reasons.append(reason)
            updated["save_reasons"] = _reasons(reasons)
            if updated != original:
                _write_json(package / STATE_FILENAME, updated)
                try:
                    _validate_package(package, args, accelerator, network, manifest, published=True)
                except BaseException:
                    _write_json(package / STATE_FILENAME, original)
                    raise
            result = {"path": str(package), "error": None}
        except Exception as error:
            result = {"path": None, "error": f"published package reason merge failed: {error}"}
    result = _broadcast_main(accelerator, result)
    if result["error"] is not None:
        raise RuntimeError(result["error"])
    return Path(result["path"])


def promote_published_current_to_best(
    args, accelerator, network, manifest, absolute_step: int, metrics_at_step: dict,
) -> Path:
    """Promote an already saved resume-step package without saving its weights again."""
    local_error = None
    candidate = None
    try:
        metrics = _metrics(metrics_at_step)
        if metrics is None:
            raise ValueError("best promotion needs a complete validation event")
        output = _output_root(args)
        name = _output_name(args)
        current = _expected_package(output, name, absolute_step, "current")
        best = _expected_package(output, name, absolute_step, "best")
        if current.is_dir() or current.is_symlink():
            candidate = current
        elif best.is_dir() or best.is_symlink():
            candidate = best
        else:
            raise ValueError(f"no published current/best package exists at step {absolute_step}")
        _validate_package(candidate, args, accelerator, network, manifest, published=True)
    except (AttributeError, OSError, TypeError, ValueError) as error:
        local_error = str(error)
    _shared_error(accelerator, local_error)

    result = None
    if accelerator.is_main_process:
        try:
            previous, prior_score = read_best_package(args, accelerator, network, manifest)
            if not is_new_best(metrics, prior_score):
                raise ValueError("initial validation does not strictly improve the stored best")
            if candidate == current and (best.exists() or best.is_symlink()):
                raise ValueError(f"published best step collision: {best}")
            original = _read_json(candidate / STATE_FILENAME)
            updated = dict(original)
            updated["metrics_at_step"] = metrics
            reasons = list(original["save_reasons"])
            if "new_best" not in reasons:
                reasons.append("new_best")
            updated["save_reasons"] = _reasons(reasons)
            _write_json(candidate / STATE_FILENAME, updated)
            try:
                _validate_package(candidate, args, accelerator, network, manifest, published=True)
                if candidate == current:
                    old_current = None
                    if previous is not None:
                        old_step = _validate_package(previous, args, accelerator, network, manifest, published=True)
                        old_current = _expected_package(output, name, old_step, "current")
                        old_current.parent.mkdir(parents=True, exist_ok=True)
                        if old_current.exists() or old_current.is_symlink():
                            raise ValueError(f"previous best current-location collision: {old_current}")
                        previous.rename(old_current)
                    try:
                        best.parent.mkdir(parents=True, exist_ok=True)
                        candidate.rename(best)
                    except BaseException:
                        if previous is not None and old_current is not None and old_current.exists():
                            old_current.rename(previous)
                        raise
            except BaseException:
                if candidate.exists():
                    _write_json(candidate / STATE_FILENAME, original)
                raise
            _prune_owned_current(args, accelerator, network, manifest, absolute_step)
            result = {"path": str(best), "error": None}
        except Exception as error:
            result = {"path": None, "error": f"published best promotion failed: {error}"}
    result = _broadcast_main(accelerator, result)
    if result["error"] is not None:
        raise RuntimeError(result["error"])
    return Path(result["path"])


def _publish(staging: Path, args, accelerator, network, manifest, step: int, destination: str) -> Path:
    output = _output_root(args)
    name = _output_name(args)
    final = _expected_package(output, name, step, destination)
    final.parent.mkdir(parents=True, exist_ok=True)
    other = _expected_package(output, name, step, "best" if destination == "current" else "current")
    if final.exists() or final.is_symlink() or other.exists() or other.is_symlink():
        raise ValueError(f"published state already exists: {final}")
    if destination == "current":
        staging.rename(final)
        return final

    previous, old_score = read_best_package(args, accelerator, network, manifest)
    candidate_metrics = _read_json(staging / STATE_FILENAME)["metrics_at_step"]
    if not is_new_best(candidate_metrics, old_score):
        raise ValueError("candidate does not strictly improve the stored best validation score")
    old_current = None
    if previous is not None:
        old_step, _ = _validate_sidecars(previous, args, manifest)
        old_current = _expected_package(output, name, old_step, "current")
        old_current.parent.mkdir(parents=True, exist_ok=True)
        if old_current.exists():
            raise ValueError(f"cannot move previous best: current package already exists at {old_current}")
        previous.rename(old_current)
    try:
        staging.rename(final)
    except BaseException:
        if previous is not None and old_current is not None and old_current.exists():
            old_current.rename(previous)
        raise
    return final


def save_package(
    args, accelerator, network, manifest, absolute_step: int, metrics_at_step: dict | None, save_reasons,
    *, destination: str = "current", samples_enabled: bool = False, sample_callback=None,
) -> Path:
    """Save on every rank, verify after the rank saves, then publish on main."""
    local_error = None
    try:
        stage2, experiment = _payload(args, manifest, absolute_step, metrics_at_step, save_reasons)
        output = _output_root(args)
        final = _expected_package(output, _output_name(args), absolute_step, destination)
        other = _expected_package(output, _output_name(args), absolute_step, "best" if destination == "current" else "current")
        if type(samples_enabled) is not bool or (samples_enabled and not callable(sample_callback)):
            raise ValueError("enabled sampling requires a sample_callback")
        if not samples_enabled and sample_callback is not None:
            raise ValueError("sample_callback requires enabled sampling")
        if final.exists() or final.is_symlink() or other.exists() or other.is_symlink():
            raise ValueError(f"published state already exists: {final}")
    except (AttributeError, OSError, TypeError, ValueError) as error:
        local_error = str(error)
    _shared_error(accelerator, local_error)

    staging_name = None
    local_error = None
    if accelerator.is_main_process:
        try:
            output.mkdir(parents=True, exist_ok=True)
            staging_name = tempfile.mkdtemp(prefix=".experiment-stage-", dir=output)
            if samples_enabled:
                (Path(staging_name) / "samples").mkdir()
        except OSError as error:
            local_error = str(error)
    result = _broadcast_main(accelerator, {"path": staging_name, "error": local_error})
    if result["error"]:
        raise RuntimeError(f"experiment state staging failed: {result['error']}")
    staging = Path(result["path"])

    expected_count = None
    if samples_enabled:
        local_error = None
        try:
            expected_count = sample_callback(staging)
            if expected_count is not None and (type(expected_count) is not int or expected_count < 1):
                raise ValueError("sample_callback expected_count must be a positive exact integer")
        except Exception as error:
            local_error = f"sample generation failure: {error}"
        _shared_error(accelerator, local_error)

        local_error = None
        actual_count = None
        if accelerator.is_main_process:
            try:
                actual_count = _verify_samples(staging, required=True, expected_count=expected_count)
            except (OSError, TypeError, ValueError) as error:
                local_error = f"sample verification failure: {error}"
        result = _broadcast_main(
            accelerator, {"error": local_error, "sample_count": actual_count if accelerator.is_main_process else None}
        )
        if result["error"] is not None:
            raise RuntimeError(f"experiment state save rejected: {result['error']}")
        expected_count = result["sample_count"]
        experiment["sample_count"] = expected_count

    local_error = None
    try:
        accelerator.save_state(str(staging), safe_serialization=True)
    except Exception as error:
        local_error = f"state save failure: {error}"
    _shared_error(accelerator, local_error)

    local_error = None
    published = None
    if accelerator.is_main_process:
        try:
            write_validation_state(staging, stage2)
            _write_json(staging / STATE_FILENAME, experiment)
            _validate_package(staging, args, accelerator, network, manifest, published=False)
            if samples_enabled:
                _verify_samples(staging, required=True, expected_count=expected_count)
            published = _publish(staging, args, accelerator, network, manifest, absolute_step, destination)
            _prune_owned_current(args, accelerator, network, manifest, absolute_step)
        except Exception as error:
            local_error = f"package verification/publication failure: {error}"
    result = _broadcast_main(accelerator, {"path": str(published) if published else None, "error": local_error})
    if result["error"]:
        raise RuntimeError(f"experiment state save rejected: {result['error']}")
    return Path(result["path"])


def load_package(args, accelerator, network, manifest, package_dir: str | os.PathLike) -> int:
    """Reject incomplete or mismatched packages before any rank loads state."""
    package = Path(package_dir).expanduser()
    local_error = None
    step = None
    try:
        step = _validate_package(package, args, accelerator, network, manifest, published=True)
    except (AttributeError, OSError, TypeError, ValueError) as error:
        local_error = str(error)
    _shared_error(accelerator, local_error, loading=True)
    if dist.is_available() and dist.is_initialized():
        steps = [None] * dist.get_world_size()
        dist.all_gather_object(steps, step)
        if len(set(steps)) != 1:
            raise ValueError("ranks disagree on loaded absolute step")
    local_error = None
    try:
        accelerator.load_state(str(package))
    except Exception as error:
        local_error = f"accelerator.load_state failed: {error}"
    _shared_error(accelerator, local_error, loading=True)
    return step
