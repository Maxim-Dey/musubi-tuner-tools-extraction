"""Explicit optimizer steps and durable validation events for opt-in Qwen runs."""

import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

import torch


TRAINER_STATE_FILENAME = "trainer_state.json"
VALIDATION_TAGS = ("val_loss_mean", "val_loss_low_noise", "val_loss_high_noise")
_LEDGER_FILENAME = "validation_events.json"
CHECKPOINT_MANIFEST_FILENAME = "checkpoint_manifest.json"
CANONICAL_MODEL_FILENAME = "model.safetensors"


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: missing or invalid training metadata: {error}; resume from a complete opt-in state") from error


def _step(value):
    if type(value) is not int or value < 0:
        raise ValueError("optimizer_global_step must be a nonnegative integer; resume from a valid training state")
    return value


def _metrics(values):
    if set(values) != set(VALIDATION_TAGS):
        raise ValueError("validation metrics must contain exactly the three val_loss tags")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values.values()):
        raise ValueError("validation metrics must be finite numbers")
    return {tag: float(values[tag]) for tag in VALIDATION_TAGS}


def load_training_state(directory):
    """Read before loading models: neither filenames nor Accelerator.step supply the step."""
    path = Path(directory) / TRAINER_STATE_FILENAME
    state = _read_json(path)
    if not isinstance(state, dict) or type(state.get("schema")) is not int or state["schema"] != 1:
        raise ValueError(f"{path}: unsupported training metadata schema; use a complete compatible state")
    _step(state.get("optimizer_global_step"))
    for key in ("tensorboard_project_dir", "validation_fingerprint"):
        if state.get(key) is not None and not isinstance(state[key], str):
            raise ValueError(f"{path}: invalid {key}; use a complete compatible state")
    if not isinstance(state.get("tensorboard_run_name"), str) or not state["tensorboard_run_name"]:
        raise ValueError(f"{path}: missing TensorBoard run identity; use a complete compatible state")
    result = state.get("last_validation")
    if result is not None:
        if not isinstance(result, dict):
            raise ValueError(f"{path}: invalid last_validation metadata; use a complete compatible state")
        _step(result.get("step"))
        _metrics(result.get("metrics", {}))
        if not isinstance(result.get("adapter_identity"), str) or not isinstance(result.get("validation_fingerprint"), str):
            raise ValueError(f"{path}: invalid validation identity; use a complete compatible state")
    return state


def save_training_state(directory, state):
    """Store only durable fields; resolved locations and resume flags are process-local."""
    if state is None:
        return
    _step(state["optimizer_global_step"])
    _atomic_json(Path(directory) / TRAINER_STATE_FILENAME, {key: value for key, value in state.items() if not key.startswith("_")})


def _file_identity(path):
    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validation_computation_contract(args):
    """Capture fixed forward/loss inputs once, streaming model files without loading weights."""
    from musubi_tuner.utils.model_utils import str_to_dtype

    multipliers = getattr(args, "base_weights_multiplier", None) or []
    merged = [
        {"sha256": _file_identity(path), "multiplier": multipliers[index] if index < len(multipliers) else 1.0}
        for index, path in enumerate(getattr(args, "base_weights", None) or [])
    ]
    flags = (
        "fp8_base",
        "fp8_scaled",
        "sdpa",
        "flash_attn",
        "xformers",
        "split_attn",
        "cuda_allow_tf32",
        "cuda_cudnn_benchmark",
        "compile",
        "compile_backend",
        "compile_mode",
        "compile_dynamic",
        "compile_fullgraph",
        "dynamo_backend",
        "dynamo_mode",
        "dynamo_dynamic",
        "dynamo_fullgraph",
        "num_layers",
    )
    return {
        "version": 1,
        "weighting_scheme": getattr(args, "weighting_scheme", "none"),
        "dit_sha256": _file_identity(getattr(args, "dit", None)),
        "merged_base_weights": merged,
        "dit_dtype": str(str_to_dtype(getattr(args, "dit_dtype", None) or "bf16")),
        "mixed_precision": getattr(args, "mixed_precision", None) or os.environ.get("ACCELERATE_MIXED_PRECISION", "no"),
        "forward_options": {key: getattr(args, key, None) for key in flags},
    }


def prepare_training_state(args, validation_fingerprint=None):
    """Prepare mutable state before Accelerator/model construction; legacy remains untouched."""
    if not (getattr(args, "experiment_mode", False) or getattr(args, "val_dataset_config", None)):
        args._training_state = None
        return None
    root = Path(getattr(args, "_experiment_root", os.getcwd())).resolve()
    computation = _validation_computation_contract(args) if validation_fingerprint is not None else None
    resume = getattr(args, "resume", None)
    if resume:
        if getattr(args, "resume_from_huggingface", False):
            raise ValueError(
                "resume: opt-in training requires local trainer metadata before model loading; download the full state first"
            )
        if getattr(args, "experiment_mode", False):
            validate_experiment_checkpoint(resume)
        state = load_training_state(resume)
        prior_fingerprint = state.get("validation_fingerprint")
        if validation_fingerprint is not None and prior_fingerprint not in (None, validation_fingerprint):
            raise ValueError(
                "resume: validation fingerprint changed; restore the original inputs/caches/settings or start a new run"
            )
        if validation_fingerprint is not None:
            previous_computation = state.get("validation_computation_contract")
            if prior_fingerprint is not None and previous_computation != computation:
                raise ValueError(
                    "resume: validation computation contract changed (base weights, loss or precision/forward settings); "
                    "restore the original configuration or start a new run"
                )
            state["validation_fingerprint"] = validation_fingerprint
            state["validation_computation_contract"] = computation
    else:
        logging = getattr(args, "logging_dir", None)
        project_dir = None
        if logging is not None:
            prefix = getattr(args, "log_prefix", None) or ""
            project = Path(logging).resolve() / (prefix + datetime.now().strftime("%Y%m%d%H%M%S%f"))
            project_dir = str(project)
            if getattr(args, "experiment_mode", False) and project.is_relative_to(root):
                project_dir = project.relative_to(root).as_posix()
        state = {
            "schema": 1,
            "optimizer_global_step": 0,
            "validation_fingerprint": validation_fingerprint,
            "validation_computation_contract": computation,
            "last_validation": None,
            "tensorboard_project_dir": project_dir,
            "tensorboard_run_name": getattr(args, "log_tracker_name", None) or "network_train",
        }
    state["_experiment_root"] = str(root)
    state["_resumed"] = bool(resume)
    args._training_state = state
    return state


def resolved_project_dir(state):
    location = state["tensorboard_project_dir"]
    if location is None:
        return None
    path = Path(location)
    return str(path if path.is_absolute() else Path(state["_experiment_root"]) / path)


def _event_directory(state):
    project = resolved_project_dir(state)
    return None if project is None else Path(project) / state["tensorboard_run_name"]


def tracker_init(args, state, init_kwargs):
    """Reuse both directory components and preserve valid events at the resumed step."""
    name = getattr(args, "log_tracker_name", None) or "network_train"
    result = copy.deepcopy(init_kwargs)
    if state is not None:
        name = state["tensorboard_run_name"]
        if state.get("_resumed"):
            result.setdefault("tensorboard", {})["purge_step"] = state["optimizer_global_step"] + 1
            directory = _event_directory(state)
            if directory is not None and (directory / _LEDGER_FILENAME).exists():
                ledger = _read_ledger(state)
                ledger = {step: entry for step, entry in ledger.items() if int(step) <= state["optimizer_global_step"]}
                _atomic_json(directory / _LEDGER_FILENAME, ledger)
    return name, result


def adapter_identity(network):
    """Hash only adapter tensor contents, including names/shapes/dtypes, without RNG use."""
    digest = hashlib.sha256()
    for key, tensor in sorted(network.state_dict().items()):
        # Alpha is serialized as FP32 by the canonical adapter exporter even when
        # an integer constructor argument originally made this buffer int64.
        if key.endswith(".alpha"):
            tensor = tensor.float()
        descriptor = json.dumps([key, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode("utf-8")
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        contents = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def flush_trackers(accelerator):
    for tracker in accelerator.trackers:
        writer = getattr(tracker, "writer", None)
        if writer is not None:
            writer.flush()


def _read_ledger(state):
    directory = _event_directory(state)
    if directory is None or not (directory / _LEDGER_FILENAME).is_file():
        return {}
    ledger = _read_json(directory / _LEDGER_FILENAME)
    if not isinstance(ledger, dict) or any(not key.isdecimal() for key in ledger):
        raise ValueError(f"{directory}: invalid validation event identity ledger")
    return ledger


def _saved_result(state, step):
    result = state.get("last_validation")
    if result is not None and result["step"] == step:
        return result
    return _read_ledger(state).get(str(step))


def _same_identity(state, result, weight_identity):
    return (
        result is not None
        and result.get("adapter_identity") == weight_identity
        and result.get("validation_fingerprint") == state["validation_fingerprint"]
        and result.get("computation_identity") == _computation_identity(state)
    )


def _computation_identity(state):
    return hashlib.sha256(
        json.dumps(state.get("validation_computation_contract"), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _events_at(state, step):
    directory = _event_directory(state)
    if directory is None or not directory.is_dir() or not list(directory.glob("events.out.tfevents.*")):
        return {}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()
    available = set(events.Tags()["scalars"])
    return {
        tag: [event.value for event in events.Scalars(tag) if event.step == step] for tag in VALIDATION_TAGS if tag in available
    }


def _event_value(value):
    # SummaryWriter's scalar protobuf stores float32 even when the aggregate is float64.
    return struct.unpack("f", struct.pack("f", value))[0]


def saved_validation_metrics(state, step, weight_identity):
    """Recover measured values only when both loaded weights and fixed inputs match."""
    result = _saved_result(state, step)
    if not _same_identity(state, result, weight_identity):
        return None
    return _metrics(result["metrics"])


def validation_events_complete(state, step, weight_identity):
    result = _saved_result(state, step)
    if not _same_identity(state, result, weight_identity):
        return False
    actual = _events_at(state, step)
    complete = all(actual.get(tag) == [_event_value(result["metrics"][tag])] for tag in VALIDATION_TAGS)
    if complete:
        state["last_validation"] = result
    return complete


def pending_validation_events(state, step, metrics, weight_identity):
    """Repair a partially durable triplet without duplicating surviving scalar points."""
    metrics = _metrics(metrics)
    actual = {tag: values for tag, values in _events_at(state, step).items() if values}
    if actual:
        result = _saved_result(state, step)
        if not _same_identity(state, result, weight_identity):
            raise ValueError(
                f"validation step {step}: existing events have unknown/different weights or input identity; use a new run"
            )
        if any(actual[tag] != [_event_value(metrics[tag])] for tag in actual):
            raise ValueError(
                f"validation step {step}: conflicting or duplicate scalar metrics; restore the matching TensorBoard run"
            )
    return {tag: value for tag, value in metrics.items() if tag not in actual}


def record_validation(state, step, metrics, weight_identity):
    """Call after event logging and flush; metadata alone never proves event durability."""
    result = {
        "step": _step(step),
        "validation_fingerprint": state["validation_fingerprint"],
        "adapter_identity": weight_identity,
        "computation_identity": _computation_identity(state),
        "metrics": _metrics(metrics),
    }
    state["last_validation"] = result
    directory = _event_directory(state)
    if directory is not None:
        ledger = _read_ledger(state)
        ledger[str(step)] = result
        _atomic_json(directory / _LEDGER_FILENAME, ledger)


def checkpoint_directory(args, step):
    _step(step)
    name = args.output_name
    if not isinstance(name, str) or not name or any(character in name for character in "/\\:") or name in (".", ".."):
        raise ValueError("output_name must be a file name, without directory components")
    root = Path(args.output_dir).resolve()
    directory = root / f"{name}-step{step}"
    if directory.is_symlink() or directory.resolve().parent != root:
        raise ValueError(f"{directory}: unsafe checkpoint path outside the experiment output")
    return directory


def _owned_file(directory, name):
    directory = Path(directory).resolve()
    if not isinstance(name, str) or not name or name in (".", "..") or any(character in name for character in "/\\:"):
        raise ValueError(f"{directory}: unsafe manifest file name {name!r}")
    path = directory / name
    if path.is_symlink() or path.resolve().parent != directory:
        raise ValueError(f"{directory}: unsafe manifest file path {name!r}")
    return path


def _supported_state_file(name):
    return (
        name in (TRAINER_STATE_FILENAME, "scaler.pt")
        or re.fullmatch(
            r"(?:optimizer|scheduler|sampler)(?:_\d+)?\.bin|random_states_\d+\.pkl|dl_state_dict(?:_\d+)?\.bin|custom_checkpoint_\d+\.pkl",
            name,
        )
        is not None
    )


def _reject_unowned_tensor_files(directory, owned):
    for path in Path(directory).iterdir():
        if path.is_file() and path.suffix.lower() in (".safetensors", ".bin", ".pt", ".pth", ".pkl") and path.name not in owned:
            raise ValueError(f"{directory}: unexpected tensor artifact {path.name}; preserve it outside this checkpoint directory")


def _read_checkpoint_manifest(directory, verify=True):
    directory = Path(directory)
    path = directory / CHECKPOINT_MANIFEST_FILENAME
    if not path.is_file():
        raise ValueError(f"{directory}: no complete checkpoint manifest; samples-only or partial saves cannot resume")
    manifest = _read_json(path)
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema")) is not int
        or manifest["schema"] != 1
        or type(manifest.get("with_state")) is not bool
        or not isinstance(manifest.get("files"), dict)
    ):
        raise ValueError(f"{path}: invalid checkpoint manifest")
    _step(manifest.get("optimizer_global_step"))
    if type(manifest.get("process_count")) is not int or manifest["process_count"] < 1:
        raise ValueError(f"{path}: invalid checkpoint process count")
    files = manifest["files"]
    if CANONICAL_MODEL_FILENAME not in files:
        raise ValueError(f"{path}: canonical model.safetensors missing from checkpoint inventory")
    # Validate every ownership path before reading/deleting any of them.
    owned = {name: _owned_file(directory, name) for name in files}
    _reject_unowned_tensor_files(directory, owned)
    for name, item in owned.items():
        if name != CANONICAL_MODEL_FILENAME and not _supported_state_file(name):
            raise ValueError(f"{path}: unsupported manifest-owned file {name!r}")
        if verify:
            expected = files[name]
            if (
                not isinstance(expected, dict)
                or not item.is_file()
                or item.stat().st_size != expected.get("size")
                or _file_identity(item) != expected.get("sha256")
            ):
                raise ValueError(f"{directory}: missing/corrupt checkpoint file {name}; size or digest mismatch")
    if manifest["with_state"]:
        required = {TRAINER_STATE_FILENAME, "optimizer.bin", "scheduler.bin"}
        required.update(f"random_states_{rank}.pkl" for rank in range(manifest["process_count"]))
        if not required <= set(files):
            raise ValueError(f"{path}: incomplete resumable state inventory; missing {sorted(required - set(files))}")
    return manifest


def validate_experiment_checkpoint(directory, require_state=True):
    """Validate actual files before model setup; a weights/sample directory is not a state."""
    manifest = _read_checkpoint_manifest(directory)
    if require_state and not manifest["with_state"]:
        raise ValueError(f"{directory}: weights-only checkpoint has no resumable optimizer/scheduler/RNG state")
    if manifest["with_state"]:
        state = load_training_state(directory)
        if state["optimizer_global_step"] != manifest["optimizer_global_step"]:
            raise ValueError(f"{directory}: checkpoint and trainer optimizer_global_step disagree")
    return manifest


def _save_canonical_adapter(network, directory, metadata, dtype=torch.float32):
    if dtype == torch.float32 and any(
        parameter.dtype != torch.float32 for parameter in network.parameters() if parameter.requires_grad
    ):
        raise ValueError("experiment checkpoint requires FP32 trainable adapter parameters for lossless resume")
    network.save_weights(str(Path(directory) / CANONICAL_MODEL_FILENAME), dtype, dict(metadata))


def register_experiment_hooks(args, accelerator, network, state):
    """One adapter artifact replaces every Accelerate model-state serialization."""
    from safetensors.torch import load_file

    def save_hook(models, weights, output_dir):
        weights.clear()
        if accelerator.is_main_process:
            if not state.get("_checkpoint_reuse_model", False):
                _save_canonical_adapter(accelerator.unwrap_model(network), output_dir, state.get("_checkpoint_metadata", {}))
            save_training_state(output_dir, state)

    def load_hook(models, input_dir):
        manifest = validate_experiment_checkpoint(input_dir)
        if manifest["process_count"] != accelerator.num_processes:
            raise ValueError(f"{input_dir}: checkpoint rank count differs; resume with {manifest['process_count']} processes")
        unwrapped = accelerator.unwrap_model(network)
        tensors = load_file(str(Path(input_dir) / CANONICAL_MODEL_FILENAME))
        if any(tensor.dtype != torch.float32 for tensor in tensors.values()):
            raise ValueError(f"{input_dir}: canonical adapter tensors must all be FP32")
        unwrapped.load_state_dict(tensors, strict=True)
        # The retained LoRA/LoHa/LoKr modules cache this factor outside state_dict.
        for module in unwrapped.modules():
            if (
                isinstance(getattr(module, "alpha", None), torch.Tensor)
                and hasattr(module, "lora_dim")
                and hasattr(module, "scale")
            ):
                module.scale = module.alpha.item() / module.lora_dim
        if adapter_identity(unwrapped) != manifest["adapter_identity"]:
            raise ValueError(f"{input_dir}: restored adapter identity differs from saved weights")
        restored = load_training_state(input_dir)
        state.update(restored)
        models.clear()

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)


def _checkpoint_inventory(directory, with_state):
    names = [CANONICAL_MODEL_FILENAME]
    if with_state:
        names.extend(path.name for path in Path(directory).iterdir() if path.is_file() and _supported_state_file(path.name))
    return {
        name: {"size": _owned_file(directory, name).stat().st_size, "sha256": _file_identity(_owned_file(directory, name))}
        for name in sorted(names)
    }


def save_experiment_checkpoint(args, accelerator, network, state, metadata, with_state):
    """Every rank calls this; completion becomes durable only after all native state files."""
    from accelerate.utils import broadcast_object_list, gather_object
    from musubi_tuner.utils.train_utils import resolve_save_dtype

    step = _step(state["optimizer_global_step"])
    save_dtype = resolve_save_dtype(
        getattr(args, "save_precision", None), getattr(args, "full_fp16", False), getattr(args, "full_bf16", False)
    )
    if with_state and save_dtype != torch.float32:
        raise ValueError("save_precision: experiment state requires FP32; use save_precision=fp32")
    directory = checkpoint_directory(args, step)
    identity = adapter_identity(accelerator.unwrap_model(network))
    status = [None]
    if accelerator.is_main_process:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            marker = directory / CHECKPOINT_MANIFEST_FILENAME
            existing = validate_experiment_checkpoint(directory, require_state=False) if marker.exists() else None
            if existing is None:
                _reject_unowned_tensor_files(directory, {CANONICAL_MODEL_FILENAME})
            same = existing is not None and existing["adapter_identity"] == identity and existing["optimizer_global_step"] == step
            reuse_model = same and existing.get("save_dtype", "torch.float32") == str(save_dtype)
            done = same and (existing["with_state"] or (not with_state and reuse_model))
            if not done:
                if existing is not None:
                    # Retained state cannot accompany replacement weights from a different branch.
                    if not same:
                        for name in existing["files"]:
                            if name != CANONICAL_MODEL_FILENAME:
                                _owned_file(directory, name).unlink()
                    marker.unlink()
                elif any(
                    path.is_file() and (path.name == CANONICAL_MODEL_FILENAME or _supported_state_file(path.name))
                    for path in directory.iterdir()
                ):
                    raise ValueError(
                        f"{directory}: incomplete checkpoint files already exist; use a new output name or restore the save"
                    )
            status[0] = {"done": done, "reuse_model": reuse_model}
        except Exception as error:
            status[0] = {"error": str(error)}
    broadcast_object_list(status)
    if "error" in status[0]:
        raise ValueError(status[0]["error"])
    if status[0]["done"]:
        return directory
    state["_checkpoint_metadata"] = dict(metadata)
    state["_checkpoint_reuse_model"] = status[0]["reuse_model"]
    error = None
    try:
        flush_trackers(accelerator)
        if with_state:
            accelerator.save_state(str(directory))
        elif accelerator.is_main_process and not status[0]["reuse_model"]:
            _save_canonical_adapter(accelerator.unwrap_model(network), directory, metadata, save_dtype)
    except Exception as caught:
        error = str(caught)
    finally:
        state.pop("_checkpoint_metadata", None)
        state.pop("_checkpoint_reuse_model", None)
    errors = gather_object([error])
    if any(item is not None for item in errors):
        raise ValueError(f"{directory}: incomplete checkpoint save: {next(item for item in errors if item is not None)}")
    accelerator.wait_for_everyone()
    status = [None]
    if accelerator.is_main_process:
        try:
            inventory = _checkpoint_inventory(directory, with_state)
            manifest = {
                "schema": 1,
                "optimizer_global_step": step,
                "output_name": args.output_name,
                "with_state": bool(with_state),
                "process_count": accelerator.num_processes,
                "adapter_identity": identity,
                "save_dtype": str(save_dtype),
                "files": inventory,
            }
            if with_state:
                required = {"optimizer.bin", "scheduler.bin", TRAINER_STATE_FILENAME}
                required.update(f"random_states_{rank}.pkl" for rank in range(accelerator.num_processes))
                if not required <= set(inventory):
                    raise ValueError(f"incomplete state inventory: missing {sorted(required - set(inventory))}")
            _atomic_json(directory / CHECKPOINT_MANIFEST_FILENAME, manifest)
            prune_experiment_checkpoints(args, step)
            status[0] = {}
        except Exception as caught:
            status[0] = {"error": str(caught)}
    broadcast_object_list(status)
    if "error" in status[0]:
        raise ValueError(status[0]["error"])
    return directory


def prune_experiment_checkpoints(args, current_step):
    """Remove only exact-name, manifest-owned files inside the selected output root."""
    root = Path(args.output_dir).resolve()
    if not root.is_dir():
        return
    weights_window = getattr(args, "save_last_n_steps", None)
    state_window = getattr(args, "save_last_n_steps_state", None) or weights_window
    pattern = re.compile(re.escape(args.output_name) + r"-step(\d+)$")
    for directory in root.iterdir():
        match = pattern.fullmatch(directory.name)
        if not match or not directory.is_dir() or directory.is_symlink():
            continue
        if directory.resolve().parent != root:
            raise ValueError(f"{directory}: unsafe checkpoint path outside output root")
        marker = directory / CHECKPOINT_MANIFEST_FILENAME
        if not marker.is_file():
            continue
        manifest = _read_checkpoint_manifest(directory)
        step = int(match.group(1))
        if manifest.get("output_name") != args.output_name or manifest["optimizer_global_step"] != step:
            raise ValueError(f"{directory}: checkpoint ownership disagrees with output name/step")
        keep_state = manifest["with_state"] and (state_window is None or step >= current_step - state_window)
        keep_weights = keep_state or weights_window is None or step >= current_step - weights_window
        if keep_state:
            continue
        for name in manifest["files"]:
            if name != CANONICAL_MODEL_FILENAME or not keep_weights:
                _owned_file(directory, name).unlink()
        if keep_weights:
            manifest["with_state"] = False
            manifest["files"] = {CANONICAL_MODEL_FILENAME: manifest["files"][CANONICAL_MODEL_FILENAME]}
            _atomic_json(marker, manifest)
        else:
            marker.unlink()
            if not any(directory.iterdir()):
                directory.rmdir()
