"""Small read-only checks for short-case configs, events, and package files.

Usage: python inspect_evidence.py preflight <run>/experiments
       python inspect_evidence.py g04 --input <run>/evidence/G04/forwards.jsonl
       python inspect_evidence.py g07 --root <run> --out <json>
       python inspect_evidence.py g10 --root <run> --out <json>
Preflight checks TOML values only. T009 also requires actual CLI overrides,
source/environment/fixture/cache manifests, and model compatibility.
"""

import hashlib
import json
import math
import sys
import tomllib
from pathlib import Path

from safetensors.torch import load_file
import torch


TRAIN_BUDGETS = {
    "g01": 12,
    "g02": 10,
    "g05-on": 4,
    "g05-off": 4,
    "g06-current": 4,
    "g06-best": 4,
    "g09": 2,
}
VAL_INTERVALS = {"g01": 4, "g02": 4, "g05-on": 4, "g06-current": 4, "g06-best": 4, "g09": 1}


def flatten_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    flat = {}
    for section, value in data.items():
        if isinstance(value, dict):
            flat.update(value)  # same top-level section flattening as parser_common
        else:
            flat[section] = value
    return flat


def check(case: str, path: Path) -> dict:
    errors = []
    try:
        values = flatten_toml(path)
    except (OSError, ValueError) as exc:
        return {"case": case, "path": str(path), "errors": [f"cannot read TOML: {exc}"]}

    def require(key, expected):
        if values.get(key) != expected or key not in values:
            errors.append(f"{key}: expected explicit {expected!r}, got {values.get(key)!r}")

    budget = values.get("max_train_steps")
    if type(budget) is not int or not 0 <= budget <= 12 or budget == 1600:
        errors.append(f"max_train_steps: explicit integer 0..12 required, got {budget!r}")
    if case in TRAIN_BUDGETS:
        require("max_train_steps", TRAIN_BUDGETS[case])

    for key, expected in (
        ("model_version", "original"),
        ("network_dim", 16),
        ("network_alpha", 16),
        ("mixed_precision", "bf16"),
        ("optimizer_type", "adamw8bit"),
        ("learning_rate", 5e-5),
        ("lr_warmup_steps", 2),
        ("sdpa", True),
        ("gradient_checkpointing", True),
        ("fp8_base", False),
        ("fp8_scaled", False),
        ("fp8_vl", False),
        ("blocks_to_swap", 0),
    ):
        require(key, expected)
    if values.get("network_module") not in ("networks.lora_qwen_image", "musubi_tuner.networks.lora_qwen_image"):
        errors.append("network_module: original Qwen-Image LoRA module required")
    if values.get("save_precision") not in ("float", "fp32"):
        errors.append("save_precision: explicit FP32 required")
    for key in ("dit", "vae", "text_encoder"):
        value = values.get(key)
        if not isinstance(value, str) or not value or "REPLACE" in value:
            errors.append(f"{key}: real model path required")
            continue
        target = Path(value)
        if not target.is_absolute():
            target = path.parent / target
        if not target.is_file():
            errors.append(f"{key}: model file does not exist: {target}")
    if case in VAL_INTERVALS:
        require("val_every_n_steps", VAL_INTERVALS[case])
        require("val_seed_noise", 42)
        require("val_level_noise_n", 2 if case == "g09" else 4)
        require("val_seed_noise_n", 1 if case == "g09" else 2)
    if case == "g05-off":
        for key in ("experiment_dir", "val_dataset_config", "sample_prompts", "sample_every_n_steps"):
            if key in values:
                errors.append(f"{key}: remove from legacy disabled case")
        for key in ("dataset_config", "output_dir"):
            value = values.get(key)
            if not isinstance(value, str) or not Path(value).is_absolute():
                errors.append(f"{key}: explicit absolute legacy path required")
            elif key == "dataset_config" and not Path(value).is_file():
                errors.append(f"dataset_config: file does not exist: {value}")
    else:
        require("experiment_dir", ".")
        require("dataset_config", "train-dataset.toml")
        require("val_dataset_config", "val-dataset.toml")
        for name in ("train-dataset.toml", "val-dataset.toml"):
            dataset_path = path.parent / name
            try:
                with dataset_path.open("rb") as stream:
                    dataset = tomllib.load(stream)
                if dataset.get("general", {}).get("resolution") != [1024, 1024]:
                    errors.append(f"{name}: resolution must be [1024, 1024]")
                if dataset.get("general", {}).get("enable_bucket") is not True:
                    errors.append(f"{name}: enable_bucket must be true")
            except (OSError, ValueError) as exc:
                errors.append(f"{name}: cannot read dataset TOML: {exc}")
    return {"case": case, "path": str(path), "max_train_steps": budget, "errors": errors}


def preflight(cases_root: Path) -> int:
    names = (*TRAIN_BUDGETS, "g03-4x2", "g03-renamed", "g03-10x1", "g04", "g05-snapshot")
    results = [check(name, cases_root / name / "train.toml") for name in names]
    report = {
        "config_checks_pass": all(not item["errors"] for item in results),
        "declared_maximum_completed_updates": {"one_gpu": 38, "conditional_two_gpu": 40},
        "cases": results,
        "remaining_t009_gates": [
            "Compare exact launch commands and CLI overrides with these TOMLs",
            "Verify source/environment/model and fixture/cache manifests and hashes",
            "Verify memory and dependency imports; record conditional G09 GPU count",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["config_checks_pass"] else 1


TAGS = (
    "train_eval_loss_mean", "train_eval_loss_low_noise", "train_eval_loss_high_noise",
    "val_loss_mean", "val_loss_low_noise", "val_loss_high_noise",
)


CAPTURE_LIMIT = 512 * 1024 * 1024
RTOL, ATOL = 5e-3, 5e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _close(expected: float, actual: float, label: str) -> float:
    if not math.isfinite(expected) or not math.isfinite(actual):
        raise AssertionError(f"G04 {label} is nonfinite")
    difference = abs(expected - actual)
    if not math.isclose(expected, actual, rel_tol=RTOL, abs_tol=ATOL):
        raise AssertionError(f"G04 {label} differs beyond predeclared rtol={RTOL}, atol={ATOL}")
    return difference


def inspect_g04(input_file: Path) -> dict:
    input_file = input_file.resolve()
    base = input_file.parent
    observed = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in input_file.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != 40 or observed.get("capture_count") != 40:
        raise AssertionError("G04 requires exactly 40 actual forward captures")
    total_bytes = 0
    grouped: dict[tuple[str, int], list[dict]] = {}
    differences = []
    directed_count = 0
    for record in records:
        if record["role"] not in ("val_familiar", "val_unfamiliar"):
            raise AssertionError("G04 unknown validation role")
        i, j, t = record["i"], record["j"], record["t"]
        if i not in (1, 2, 3, 4) or j not in (1, 2):
            raise AssertionError("G04 unexpected check grid")
        midpoint = 0.05 + (i - 0.5) * 0.90 / 4
        if t != midpoint or record["timestep"] != 1000 * midpoint:
            raise AssertionError("G04 midpoint or timestep differs")
        capture = (base / record["capture_file"]).resolve()
        if not capture.is_relative_to(base / "captures") or capture.suffix != ".safetensors":
            raise AssertionError("G04 capture path leaves the fixed evidence folder")
        total_bytes += capture.stat().st_size
        if total_bytes > CAPTURE_LIMIT or _sha256(capture) != record["capture_sha256"]:
            raise AssertionError("G04 capture limit or SHA-256 mismatch")
        tensors = load_file(str(capture), device="cpu")
        if set(tensors) != {"latent", "epsilon", "noisy", "pred", "target"}:
            raise AssertionError("G04 capture does not contain exactly the five forward tensors")
        latent, epsilon, noisy, pred, target = (tensors[key] for key in ("latent", "epsilon", "noisy", "pred", "target"))
        for tensor, digest_key in (
            (latent, "latent_sha256"), (epsilon, "epsilon_sha256"), (noisy, "noisy_latent_sha256"),
            (pred, "prediction_sha256"), (target, "target_sha256"),
        ):
            if _tensor_hash(tensor) != record[digest_key]:
                raise AssertionError(f"G04 {digest_key} does not match captured tensor bytes")
        if str(epsilon.dtype) != record["epsilon_dtype"] or list(epsilon.shape[1:]) != record["epsilon_shape"]:
            raise AssertionError("G04 epsilon dtype/shape differs from the actual check")
        if not torch.equal(noisy, (1 - t) * latent + t * epsilon):
            raise AssertionError("G04 noisy latent does not equal independent flow mix")
        network_dtype = getattr(torch, record["network_dtype"].removeprefix("torch."))
        if not torch.equal(target, epsilon - latent.to(dtype=network_dtype)):
            raise AssertionError("G04 target does not equal epsilon minus latent")
        squared = (pred.to(network_dtype) - target).square()
        scheme = record["weighting_scheme"]
        sigma = torch.tensor(t, dtype=getattr(torch, record["dit_dtype"].removeprefix("torch.")))
        if scheme == "sigma_sqrt":
            squared *= (sigma ** -2.0).float()
        elif scheme == "cosmap":
            squared *= 2 / (math.pi * (1 - 2 * sigma + 2 * sigma**2))
        elif scheme != "none":
            raise AssertionError(f"G04 unsupported captured weighting scheme {scheme}")
        reference_loss = float(squared.mean().item())
        differences.append(_close(reference_loss, record["reported_loss"], "per-check weighted MSE"))
        if "directed_sigma_sqrt_reported" in record:
            directed_count += 1
            directed = float(
                ((pred.to(network_dtype) - target).square() * (sigma ** -2.0).float()).mean().item()
            )
            _close(directed, record["directed_sigma_sqrt_reported"], "directed exact-sigma loss")
        grouped.setdefault((record["role"], record["item_index"]), []).append(
            {"i": i, "j": j, "t": t, "loss": reference_loss, "image_id": record["image_id"]}
        )
    if directed_count != 1:
        raise AssertionError("G04 requires one directed sigma_sqrt check, without another forward")
    if sum(role == "val_familiar" for role, _ in grouped) != 2 or sum(role == "val_unfamiliar" for role, _ in grouped) != 3:
        raise AssertionError("G04 requires two familiar and three unfamiliar images")
    for checks in grouped.values():
        if len(checks) != 8 or {(x["i"], x["j"]) for x in checks} != {
            (i, j) for i in range(1, 5) for j in range(1, 3)
        } or len({x["image_id"] for x in checks}) != 1:
            raise AssertionError("G04 image has missing, duplicate or mismatched checks")
    reference = {}
    for role, prefix in (("val_familiar", "train_eval_loss"), ("val_unfamiliar", "val_loss")):
        images = [checks for (item_role, _), checks in grouped.items() if item_role == role]
        for suffix, condition in (
            ("mean", lambda t: True),
            ("low_noise", lambda t: t < 0.5),
            ("high_noise", lambda t: t >= 0.5),
        ):
            per_image = [
                math.fsum(x["loss"] for x in checks if condition(x["t"])) /
                sum(condition(x["t"]) for x in checks)
                for checks in images
            ]
            reference[prefix + "_" + suffix] = math.fsum(per_image) / len(per_image)
        _close(
            reference[prefix + "_mean"],
            (reference[prefix + "_low_noise"] + reference[prefix + "_high_noise"]) / 2,
            prefix + " half-mean",
        )
    metric_differences = {
        name: _close(reference[name], observed["observed_metrics"][name], name) for name in TAGS
    }
    result = {
        "status": "PASS",
        "forward_captures": 40,
        "capture_bytes": total_bytes,
        "capture_limit_bytes": CAPTURE_LIMIT,
        "reference_script_sha256": _sha256(Path(__file__).resolve()),
        "forward_file_sha256": _sha256(input_file),
        "reference_metrics": reference,
        "observed_metrics": observed["observed_metrics"],
        "metric_absolute_differences": metric_differences,
        "maximum_check_loss_absolute_difference": max(differences),
        "directed_sigma_sqrt_checks": directed_count,
        "tolerance": {"rtol": RTOL, "atol": ATOL},
        "completed_optimizer_updates": 0,
    }
    (base / "independent-reference.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def event_inventory(log_dir: Path) -> dict:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from tensorboard.util.tensor_util import make_ndarray
    except ImportError as exc:
        return {"files": [], "events": [], "errors": [f"TensorBoard unavailable: {exc}"]}
    files = sorted(log_dir.rglob("events.out.tfevents.*")) if log_dir.is_dir() else []
    events = []
    errors = []
    all_tags = set()
    for file in files:
        try:
            accumulator = EventAccumulator(str(file), size_guidance={"scalars": 0, "tensors": 0})
            accumulator.Reload()
            available = accumulator.Tags()
            all_tags.update(tag for group in available.values() if isinstance(group, list) for tag in group)
            for tag in TAGS:
                for item in accumulator.Scalars(tag) if tag in available.get("scalars", ()) else ():
                    events.append({"tag": tag, "step": item.step, "value": item.value, "file": str(file)})
                for item in accumulator.Tensors(tag) if tag in available.get("tensors", ()) else ():
                    value = make_ndarray(item.tensor_proto)
                    if value.size != 1:
                        errors.append(f"{file}: {tag} has non-scalar tensor value")
                    else:
                        events.append({"tag": tag, "step": item.step, "value": float(value.item()), "file": str(file)})
        except (OSError, ValueError) as exc:
            errors.append(f"{file}: {exc}")
    seen = set()
    duplicates = []
    for event in events:
        pair = (event["tag"], event["step"])
        if pair in seen:
            duplicates.append({"tag": pair[0], "step": pair[1]})
        seen.add(pair)
        if not math.isfinite(event["value"]):
            errors.append(f"nonfinite {event['tag']} at step {event['step']}")
    return {"files": [str(path) for path in files], "all_tags": sorted(all_tags), "events": events, "duplicate_tag_steps": duplicates, "errors": errors}


def package_inventory(output: Path) -> list[dict]:
    result = []
    for location, parent in (
        ("current", output / "current_training_states"),
        ("best", output / "val_training_states" / "val-loss"),
    ):
        if not parent.is_dir():
            continue
        for path in sorted(parent.iterdir()):
            if not path.is_dir() or "-step-" not in path.name:
                continue
            sidecars = {}
            errors = []
            for name in ("experiment_state.json", "val_loss_state.json"):
                try:
                    sidecars[name] = json.loads((path / name).read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    errors.append(f"{name}: {exc}")
            required = ("model.safetensors", "optimizer.bin", "scheduler.bin")
            for name in required:
                if not (path / name).is_file():
                    errors.append(f"missing {name}")
            images = sorted((path / "samples").glob("*.png"))
            metadata = sidecars.get("experiment_state.json", {})
            model_files = sorted(p.name for p in path.glob("*.safetensors"))
            if model_files != ["model.safetensors"]:
                errors.append(f"expected sole model.safetensors, found {model_files}")
            rank_rng_files = sorted(p.name for p in path.glob("random_states_*.pkl"))
            if not rank_rng_files:
                errors.append("missing rank RNG files")
            if metadata.get("sample_count") != len(images):
                errors.append("sample_count differs from PNG count")
            result.append({
                "location": location,
                "path": str(path),
                "sidecars": sidecars,
                "model_files": model_files,
                "rank_rng_files": rank_rng_files,
                "sample_pngs": [str(p) for p in images],
                "errors": errors,
            })
    return result


def inspect_g07(root: Path) -> dict:
    cases = {}
    for name in ("g01", "g02", "g06-current", "g06-best"):
        output = root / "experiments" / name / "output"
        cases[name] = {
            "events": event_inventory(output / "tensorboard"),
            "packages": package_inventory(output),
        }
    return {"cases": cases, "note": "Inventory only; compare attribution, retention and immutable G06 inputs with the protocol before assigning PASS."}


def inspect_g10(root: Path) -> dict:
    names = ("G01", "G02", "G03", "G04", "G05", "G06-current", "G06-best", "G07", "G08", "G09")
    cases = {}
    for name in names:
        folder = root / "evidence" / name
        files = {}
        for filename in ("command.txt", "pid.txt", "start.txt", "end.txt", "exit.txt"):
            path = folder / filename
            if path.is_file():
                files[filename] = path.read_text(encoding="utf-8").strip()
        cases[name] = {"folder": str(folder), "files": files}
    return {"cases": cases, "note": "Command/exit inventory only; measured PASS/FAIL/NOT RUN and numeric differences require manual G10 reconciliation."}


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "preflight":
        return preflight(Path(sys.argv[2]).resolve())
    if len(sys.argv) == 4 and sys.argv[1] == "g04" and sys.argv[2] == "--input":
        inspect_g04(Path(sys.argv[3]))
        return 0
    if len(sys.argv) == 6 and sys.argv[1] in ("g07", "g10") and sys.argv[2] == "--root" and sys.argv[4] == "--out":
        root = Path(sys.argv[3]).resolve()
        out = Path(sys.argv[5]).resolve()
        payload = inspect_g07(root) if sys.argv[1] == "g07" else inspect_g10(root)
        with out.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return 0
    print("usage: inspect_evidence.py preflight <experiments-directory> | g04 --input <forwards.jsonl> | {g07,g10} --root <run> --out <json>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
