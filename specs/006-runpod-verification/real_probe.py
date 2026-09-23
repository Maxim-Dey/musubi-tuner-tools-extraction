"""Bounded G03–G05 probes using the existing Qwen trainer."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import copy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from musubi_tuner.qwen_image import qwen_image_utils
from musubi_tuner.qwen_image_train_network import QwenImageNetworkTrainer, qwen_image_setup_parser
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.training import validation_inputs
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.trainer_base import NetworkTrainer


METRICS = (
    "train_eval_loss_mean",
    "train_eval_loss_low_noise",
    "train_eval_loss_high_noise",
    "val_loss_mean",
    "val_loss_low_noise",
    "val_loss_high_noise",
)
G04_CAPTURE_LIMIT = 512 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_hash(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _adapter_hash(network, accelerator) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(accelerator.unwrap_model(network).state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(bytes.fromhex(_tensor_hash(tensor)))
    return digest.hexdigest()


def _assert_selected_adapter_loaded(weights: Path, network, accelerator) -> None:
    """Reject a silent partial LoRA load before probing a selected checkpoint."""
    expected = load_file(str(weights), device="cpu")
    actual = accelerator.unwrap_model(network).state_dict()
    if expected.keys() != actual.keys():
        missing = sorted(expected.keys() - actual.keys())
        unexpected = sorted(actual.keys() - expected.keys())
        raise ValueError(f"selected LoRA keys differ: missing={missing}, unexpected={unexpected}")
    for name, selected in expected.items():
        loaded = actual[name].detach().cpu()
        if loaded.shape != selected.shape or loaded.dtype != selected.dtype or not torch.equal(loaded, selected):
            raise ValueError(f"selected LoRA tensor {name} was not loaded exactly")


def _read_effective_config(config: Path):
    parser = qwen_image_setup_parser(setup_parser_common())
    original_argv = sys.argv
    sys.argv = ["qwen_image_train_network.py", "--config_file", str(config)]
    try:
        return read_config_from_file(parser.parse_args(), parser)
    finally:
        sys.argv = original_argv


def _structured_hash(value) -> str:
    """Hash actual nested state values, including tensor bytes, without storing them."""
    if isinstance(value, torch.Tensor):
        serial = ["tensor", str(value.dtype), list(value.shape), _tensor_hash(value)]
    elif isinstance(value, np.ndarray):
        serial = ["ndarray", str(value.dtype), list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()]
    elif isinstance(value, dict):
        serial = ["dict", [[repr(key), _structured_hash(item)] for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))]]
    elif isinstance(value, (list, tuple)):
        serial = [type(value).__name__, [_structured_hash(item) for item in value]]
    else:
        serial = [type(value).__name__, repr(value)]
    return hashlib.sha256(json.dumps(serial, sort_keys=True).encode("utf-8")).hexdigest()


def _training_state(transformer, network, optimizer, scheduler, accelerator) -> dict:
    adapter = accelerator.unwrap_model(network)
    modes = {
        "transformer": [(name, module.training) for name, module in transformer.named_modules()],
        "network": [(name, module.training) for name, module in network.named_modules()],
    }
    gradients = [
        (name, None if parameter.grad is None else _tensor_hash(parameter.grad))
        for name, parameter in adapter.named_parameters()
    ]
    numpy_state = np.random.get_state()
    return {
        "modes": modes,
        "python_rng": _structured_hash(random.getstate()),
        "numpy_rng": _structured_hash(numpy_state),
        "torch_cpu_rng": _tensor_hash(torch.get_rng_state()),
        "torch_cuda_rng": [_tensor_hash(state) for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        "gradients": gradients,
        "adapter": _adapter_hash(network, accelerator),
        "optimizer": _structured_hash(optimizer.state_dict()),
        "scheduler": _structured_hash(scheduler.state_dict()),
    }


def _read_args(config: Path, weights: Path, *, levels: int, seeds: int):
    args = _read_effective_config(config)
    if args.resume or args.max_train_epochs is not None:
        raise ValueError("G03 is weights-only validation; remove resume and max_train_epochs")
    if not isinstance(args.max_train_steps, int) or not 1 <= args.max_train_steps <= 12:
        raise ValueError("G03 requires a short effective max_train_steps in 1..12; refuse inherited 1600")
    if (args.val_level_noise_n, args.val_seed_noise_n) != (levels, seeds):
        raise ValueError(f"G03 config requires val_level_noise_n={levels}, val_seed_noise_n={seeds}")
    if args.sample_prompts and (
        args.sample_at_first or args.sample_every_n_steps is not None or args.sample_every_n_epochs is not None
    ):
        raise ValueError("G03 must not sample; remove sample controls from its validation-only config")
    args.network_weights = str(weights)
    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"
    qwen_image_utils.resolve_model_version_args(args)
    return args


class _ValidationOnlyTrainer(QwenImageNetworkTrainer):
    def __init__(self, repeats: int):
        super().__init__()
        self.repeats = repeats
        self.records: list[dict] = []
        self.calls: list[dict] = []
        self._active_check_record: dict | None = None

    def _run_training_loop(self, args, accelerator, _session_id, _started_at, _dataset, _loader,
                           _epoch, transformer, network, _training_model, _optimizer,
                           _optimizer_name, _optimizer_args, _optimizer_train_fn, _optimizer_eval_fn,
                           _lr_scheduler, _lr_descriptions, _sample_resources, _sample_parameters,
                           dit_dtype, network_dtype):
        if self.validation_manifest is None:
            raise ValueError("G03 requires both frozen validation roles")
        _assert_selected_adapter_loaded(Path(args.network_weights), network, accelerator)
        before = _adapter_hash(network, accelerator)
        scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")
        original = validation_inputs.iter_noise_checks
        try:
            for call_index in range(self.repeats):
                checks = []
                item_indices = {}

                def observed_checks(*a, **kw):
                    for check in original(*a, **kw):
                        item_key = id(check.item)
                        if item_key not in item_indices:
                            item_indices[item_key] = len(item_indices)
                        record = {
                            "image_id": check.item.image_id,
                            "role": check.item.role,
                            "item_index": item_indices[item_key],
                            "i": check.i,
                            "j": check.j,
                            "t": check.t,
                            "timestep": check.timestep,
                            "seed": check.seed,
                            "epsilon_sha256": _tensor_hash(check.epsilon),
                            "epsilon_dtype": str(check.epsilon.dtype),
                            "epsilon_shape": list(check.epsilon.shape),
                        }
                        checks.append(record)
                        self._active_check_record = record
                        yield check

                validation_inputs.iter_noise_checks = observed_checks
                try:
                    metrics = self.evaluate_validation_event(
                        args, accelerator, transformer, network, scheduler, dit_dtype, network_dtype, 0
                    )
                finally:
                    validation_inputs.iter_noise_checks = original
                if set(metrics) != set(METRICS) or not all(math.isfinite(metrics[key]) for key in METRICS):
                    raise ValueError("G03 validation did not return six finite metrics")
                self.calls.append({
                    "call": call_index + 1,
                    "metrics": {key: float(metrics[key]) for key in METRICS},
                    "checks": checks,
                })
        finally:
            validation_inputs.iter_noise_checks = original
        after = _adapter_hash(network, accelerator)
        if before != after:
            raise RuntimeError("G03 validation changed adapter weights")
        self.records.append({
            "validation_fingerprint": self.validation_manifest.fingerprint,
            "adapter_before_sha256": before,
            "adapter_after_sha256": after,
            "calls": self.calls,
            "completed_optimizer_updates": 0,
        })


class _ReferenceTrainer(_ValidationOnlyTrainer):
    """Capture actual G04 forwards and independently reduce their outputs."""

    def __init__(self, capture_dir: Path):
        super().__init__(1)
        self._directed_done = False
        self.capture_dir = capture_dir
        self.capture_dir.mkdir(parents=True, exist_ok=False)
        self.capture_bytes = 0
        self.capture_count = 0

    def call_dit(self, args, accelerator, transformer, latents, batch, noise,
                 noisy_model_input, timesteps, network_dtype, **kwargs):
        output = super().call_dit(
            args, accelerator, transformer, latents, batch, noise,
            noisy_model_input, timesteps, network_dtype, **kwargs
        )
        record = self._active_check_record
        if record is None:
            raise RuntimeError("G04 forward occurred without a frozen validation check")
        t = record["t"]
        expected_mixed = (1 - t) * latents + t * noise
        expected_target = noise - latents.to(device=accelerator.device, dtype=network_dtype)
        mix_difference = float((noisy_model_input - expected_mixed).abs().max().item())
        target_difference = float((output.target - expected_target).abs().max().item())
        timestep_difference = float((timesteps - torch.tensor([1000 * t], device=timesteps.device)).abs().max().item())
        if mix_difference or target_difference or timestep_difference:
            raise AssertionError("G04 actual mixing, timestep or target differs from independent flow-matching formula")
        if self.capture_count >= 40:
            raise ValueError("G04 exceeded the predeclared 40-check capture limit")
        sources = {
            "latent": latents,
            "epsilon": noise,
            "noisy": noisy_model_input,
            "pred": output.pred,
            "target": output.target,
        }
        raw_bytes = sum(tensor.numel() * tensor.element_size() for tensor in sources.values())
        if self.capture_bytes + raw_bytes > G04_CAPTURE_LIMIT:
            raise ValueError("G04 captures would exceed the fixed 512 MiB limit")
        tensors = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in sources.items()}
        capture = self.capture_dir / f"check-{self.capture_count + 1:04d}.safetensors"
        save_file(tensors, str(capture))
        self.capture_bytes += capture.stat().st_size
        if self.capture_bytes > G04_CAPTURE_LIMIT:
            capture.unlink()
            raise ValueError("G04 captures exceeded the fixed 512 MiB limit")
        self.capture_count += 1
        squared = (output.pred.to(network_dtype) - expected_target).square()
        scheme = args.weighting_scheme
        if scheme in ("sigma_sqrt", "cosmap"):
            sigma = torch.tensor(t, device=squared.device, dtype=torch.bfloat16)
            if scheme == "sigma_sqrt":
                squared = squared * (sigma ** -2.0).float()
            else:
                squared = squared * (2 / (math.pi * (1 - 2 * sigma + 2 * sigma**2)))
        record.update({
            "noisy_latent_sha256": _tensor_hash(noisy_model_input),
            "latent_sha256": _tensor_hash(latents),
            "prediction_sha256": _tensor_hash(output.pred),
            "target_sha256": _tensor_hash(output.target),
            "weighting_scheme": args.weighting_scheme,
            "network_dtype": str(network_dtype),
            "dit_dtype": "torch.bfloat16",
            "capture_file": f"captures/{capture.name}",
            "capture_sha256": _sha256(capture),
            "mix_max_abs_difference": mix_difference,
            "timestep_max_abs_difference": timestep_difference,
            "target_max_abs_difference": target_difference,
            "reference_loss": float(squared.mean().item()),
        })
        if not self._directed_done:
            directed_args = copy(args)
            directed_args.weighting_scheme = "sigma_sqrt"
            scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")
            directed, _ = super().compute_loss(
                directed_args, output, timesteps, scheduler, torch.bfloat16,
                network_dtype, 0, exact_sigma=t
            )
            sigma = torch.tensor(t, device=squared.device, dtype=torch.bfloat16)
            independent = (
                (output.pred.to(network_dtype) - expected_target).square()
                * (sigma ** -2.0).float()
            ).mean()
            record["directed_sigma_sqrt_reported"] = float(directed.detach().item())
            record["directed_sigma_sqrt_reference"] = float(independent.item())
            self._directed_done = True
        return output

    def compute_loss(self, args, output, timesteps, noise_scheduler, dit_dtype,
                     network_dtype, global_step, *, exact_sigma=None):
        loss, metrics = super().compute_loss(
            args, output, timesteps, noise_scheduler, dit_dtype, network_dtype,
            global_step, exact_sigma=exact_sigma
        )
        if self._active_check_record is None:
            raise RuntimeError("G04 loss occurred without a frozen validation check")
        self._active_check_record["reported_loss"] = float(loss.detach().item())
        return loss, metrics


class _G05Trainer(QwenImageNetworkTrainer):
    """Observe the real loop through existing hooks; snapshot mode performs no update."""

    def __init__(self, mode: str, evidence: Path):
        super().__init__()
        self.mode = mode
        self.evidence = evidence
        self.batches: list[dict] = []
        self.updates: list[dict] = []
        self.evaluations: list[dict] = []
        self._pending: dict | None = None

    def _run_training_loop(self, *values):
        args, accelerator = values[:2]
        transformer, network, optimizer, scheduler = values[7], values[8], values[10], values[15]
        _assert_selected_adapter_loaded(Path(args.network_weights), network, accelerator)
        self._state_sources = (transformer, network, optimizer, scheduler, accelerator)
        self.initial_state = _training_state(*self._state_sources)
        if self.mode == "snapshot":
            if self.validation_manifest is None:
                raise ValueError("G05 snapshot requires prepared validation")
            noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")
            self.evaluate_validation_event(
                args, accelerator, accelerator.unwrap_model(transformer), accelerator.unwrap_model(network),
                noise_scheduler, values[19], values[20], 0
            )
        else:
            NetworkTrainer._run_training_loop(self, *values)
        self.final_state = _training_state(*self._state_sources)

    def evaluate_validation_event(self, *values):
        before = _training_state(*self._state_sources)
        metrics = None
        try:
            metrics = super().evaluate_validation_event(*values)
        finally:
            after = _training_state(*self._state_sources)
            self.evaluations.append({
                "absolute_step": values[-1],
                "before": before,
                "after": after,
                "metrics": metrics,
            })
            if before != after:
                raise AssertionError("G05 validation changed modes, RNG, gradients, adapter, optimizer or scheduler")
        return metrics

    def process_batch(self, args, accelerator, transformer, network, batch, latents,
                      noise, noise_scheduler, dit_dtype, network_dtype, sample_resources, global_step):
        self._pending = {
            "latent_sha256": _tensor_hash(latents),
            "text_sha256": _structured_hash(batch["vl_embed"]),
            "noise_sha256": _tensor_hash(noise),
            "global_step_before": global_step,
        }
        loss, metrics = super().process_batch(
            args, accelerator, transformer, network, batch, latents, noise,
            noise_scheduler, dit_dtype, network_dtype, sample_resources, global_step
        )
        self._pending["loss"] = float(loss.detach().item())
        self.batches.append(self._pending)
        self._pending = None
        return loss, metrics

    def get_noisy_model_input_and_timesteps(self, args, noise, latents, timesteps,
                                            noise_scheduler, device, dtype):
        noisy, selected = super().get_noisy_model_input_and_timesteps(
            args, noise, latents, timesteps, noise_scheduler, device, dtype
        )
        if self._pending is not None:
            self._pending["timesteps"] = [float(value) for value in selected.detach().cpu().flatten()]
            self._pending["noisy_sha256"] = _tensor_hash(noisy)
        return noisy, selected

    def on_post_optimizer_step(self, args, accelerator, network, transformer, sync_gradients, global_step):
        if not sync_gradients or accelerator.optimizer_step_was_skipped:
            return
        update = len(self.updates) + 1
        if update > 4:
            raise RuntimeError("G05 exceeded its four completed-update limit")
        snapshot = self.evidence / f"adapter-step-{update}.safetensors"
        adapter = accelerator.unwrap_model(network)
        save_file({name: value.detach().cpu().contiguous() for name, value in adapter.state_dict().items()}, str(snapshot))
        self.updates.append({
            "completed_update": update,
            "global_step_before": global_step,
            "adapter_sha256": _sha256(snapshot),
            "adapter_path": str(snapshot),
            "learning_rates": [float(value) for value in self._state_sources[3].get_last_lr()],
        })


def _reference_aggregates(checks: list[dict]) -> dict[str, float]:
    images: dict[tuple[str, int], list[dict]] = {}
    for record in checks:
        images.setdefault((record["role"], record["item_index"]), []).append(record)
        if not math.isclose(record["reference_loss"], record["reported_loss"], rel_tol=5e-3, abs_tol=5e-4):
            raise AssertionError("G04 independent per-check loss differs from evaluated loss")
        if "directed_sigma_sqrt_reported" in record and not math.isclose(
            record["directed_sigma_sqrt_reference"], record["directed_sigma_sqrt_reported"],
            rel_tol=5e-3, abs_tol=5e-4,
        ):
            raise AssertionError("G04 directed exact-sigma weighting differs")
    result = {}
    for role, prefix in (("val_familiar", "train_eval_loss"), ("val_unfamiliar", "val_loss")):
        members = [records for (item_role, _), records in images.items() if item_role == role]
        if not members:
            raise AssertionError(f"G04 missing role {role}")
        for suffix, group in (("mean", "all"), ("low_noise", "low"), ("high_noise", "high")):
            per_image = []
            for records in members:
                selected = [r["reference_loss"] for r in records if group == "all" or (r["t"] < 0.5) == (group == "low")]
                if not selected:
                    raise AssertionError(f"G04 missing {group} checks for an image")
                per_image.append(math.fsum(selected) / len(selected))
            result[f"{prefix}_{suffix}"] = math.fsum(per_image) / len(per_image)
    return result


def _worker(config: Path, weights: Path, output: Path, repeats: int, levels: int, seeds: int) -> None:
    args = _read_args(config, weights, levels=levels, seeds=seeds)
    trainer = _ValidationOnlyTrainer(repeats)
    trainer.train(args)
    if len(trainer.records) != 1 or len(trainer.calls) != repeats:
        raise RuntimeError("G03 setup did not reach the validation-only probe")
    payload = {
        "config": str(config.resolve()),
        "config_sha256": _sha256(config),
        "weights": str(weights.resolve()),
        "weights_sha256": _sha256(weights),
        "levels": levels,
        "seeds_per_level": seeds,
        "resume": None,
        **trainer.records[0],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _check_equal(first: dict, other: dict) -> dict:
    def keyed(call):
        return Counter(
            (check["image_id"], check["role"], check["i"], check["j"],
             check["t"], check["timestep"], check["seed"], check["epsilon_sha256"],
             check["epsilon_dtype"], tuple(check["epsilon_shape"]))
            for check in call["checks"]
        )

    left, right = keyed(first), keyed(other)
    if left != right:
        raise AssertionError("G03 fixed-check image-ID/i/j/level/seed/noise multiset differs")
    differences = {}
    for name in METRICS:
        a, b = first["metrics"][name], other["metrics"][name]
        differences[name] = abs(a - b)
        if not math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-4):
            raise AssertionError(f"G03 metric {name} differs beyond predeclared tolerance")
    return differences


def _run_worker(config: Path, weights: Path, output: Path, repeats: int, levels: int, seeds: int) -> dict:
    command = [
        sys.executable, str(Path(__file__).resolve()), "_g03_worker",
        "--config", str(config), "--weights", str(weights), "--out", str(output),
        "--repeat", str(repeats), "--levels", str(levels), "--seeds", str(seeds),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise RuntimeError(f"G03 worker failed with exit {result.returncode}: {command}")
    return json.loads(output.read_text(encoding="utf-8"))


def _selected_weights(checkpoint: Path) -> Path:
    selection = checkpoint.read_text(encoding="utf-8").strip()
    package = Path(selection).expanduser().resolve()
    weights = package / "model.safetensors"
    if not package.is_dir() or not weights.is_file():
        raise ValueError("selected checkpoint must name a package containing model.safetensors")
    return weights


def _g03(args) -> None:
    weights = _selected_weights(args.checkpoint)
    output = args.out or args.checkpoint.parent / "G03"
    output.mkdir(parents=True, exist_ok=True)
    first = _run_worker(args.config, weights, output / "same-process.json", 2, 4, 2)
    fresh = _run_worker(args.config, weights, output / "fresh-process.json", 1, 4, 2)
    renamed = _run_worker(args.renamed_config, weights, output / "renamed.json", 1, 4, 2)
    ten = _run_worker(args.ten_by_one_config, weights, output / "ten-by-one.json", 1, 10, 1)
    calls = [*first["calls"], *fresh["calls"], *renamed["calls"]]
    if len(calls) != 4 or len(ten["calls"]) != 1:
        raise AssertionError("G03 requires four 4x2 calls and one separate 10x1 call")
    if any(record["weights_sha256"] != first["weights_sha256"] for record in (fresh, renamed, ten)):
        raise AssertionError("G03 workers used different adapter weights")
    if any(record["adapter_before_sha256"] != first["adapter_before_sha256"] for record in (fresh, renamed, ten)):
        raise AssertionError("G03 workers loaded different adapter tensors")
    if fresh["validation_fingerprint"] != first["validation_fingerprint"]:
        raise AssertionError("G03 canonical config changed its validation fingerprint across processes")
    comparisons = [_check_equal(calls[0], call) for call in calls[1:]]
    if len(ten["calls"][0]["checks"]) != len(calls[0]["checks"]) * 10 // 8:
        raise AssertionError("G03 10x1 did not contain ten checks per image")
    summary = {
        "status": "PASS",
        "weights_sha256": first["weights_sha256"],
        "validation_fingerprints": {
            "canonical": first["validation_fingerprint"],
            "fresh_process": fresh["validation_fingerprint"],
            "renamed_copy": renamed["validation_fingerprint"],
            "ten_by_one": ten["validation_fingerprint"],
        },
        "four_by_two_calls": 4,
        "ten_by_one_calls": 1,
        "completed_optimizer_updates": 0,
        "metric_absolute_differences": comparisons,
        "tolerance": {"rtol": 1e-3, "atol": 1e-4},
        "worker_evidence": ["same-process.json", "fresh-process.json", "renamed.json", "ten-by-one.json"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _g04(args) -> None:
    weights = _selected_weights(args.checkpoint)
    config = _read_args(args.config, weights, levels=4, seeds=2)
    trainer = _ReferenceTrainer(args.out.parent / "captures")
    trainer.train(config)
    if len(trainer.records) != 1 or len(trainer.calls) != 1:
        raise RuntimeError("G04 did not complete exactly one real validation event")
    checks = trainer.calls[0]["checks"]
    if len(checks) != 40 or trainer.capture_count != 40:
        raise AssertionError("G04 requires exactly 40 captured checks from the frozen 2+3 validation images")
    reference = _reference_aggregates(checks)
    observed = trainer.calls[0]["metrics"]
    differences = {}
    for name in METRICS:
        differences[name] = abs(reference[name] - observed[name])
        if not math.isclose(reference[name], observed[name], rel_tol=5e-3, abs_tol=5e-4):
            raise AssertionError(f"G04 independent image-first aggregate differs for {name}")
    for prefix in ("train_eval_loss", "val_loss"):
        half_mean = (reference[prefix + "_low_noise"] + reference[prefix + "_high_noise"]) / 2
        if not math.isclose(half_mean, reference[prefix + "_mean"], rel_tol=5e-3, abs_tol=5e-4):
            raise AssertionError(f"G04 low/high half-mean relation differs for {prefix}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for record in checks:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    summary = {
        "status": "PASS",
        "config_sha256": _sha256(args.config),
        "weights_sha256": _sha256(weights),
        "validation_fingerprint": trainer.records[0]["validation_fingerprint"],
        "adapter_before_sha256": trainer.records[0]["adapter_before_sha256"],
        "adapter_after_sha256": trainer.records[0]["adapter_after_sha256"],
        "completed_optimizer_updates": 0,
        "validation_calls": 1,
        "reference_metrics": reference,
        "observed_metrics": observed,
        "absolute_differences": differences,
        "tolerance": {"rtol": 5e-3, "atol": 5e-4},
        "forwards_jsonl": str(args.out.resolve()),
        "capture_count": trainer.capture_count,
        "capture_bytes": trainer.capture_bytes,
        "capture_limit_bytes": G04_CAPTURE_LIMIT,
    }
    (args.out.parent / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _read_g05_args(config: Path, weights: Path, mode: str):
    args = _read_effective_config(config)
    expected_steps = 1 if mode == "snapshot" else 4
    if args.resume or args.max_train_epochs is not None or args.max_train_steps != expected_steps:
        raise ValueError(
            f"G05 {mode} requires weights-only start and max_train_steps={expected_steps}"
        )
    if args.sample_prompts or args.sample_at_first or args.sample_every_n_steps is not None or args.sample_every_n_epochs is not None:
        raise ValueError("G05 requires samples disabled in every branch")
    enabled = mode != "off"
    if enabled and (not args.experiment_dir or not args.val_dataset_config or args.val_every_n_steps != 4):
        raise ValueError("G05 enabled/snapshot branch requires experiment validation at steps 0 and 4")
    if not enabled and (args.experiment_dir or args.val_dataset_config):
        raise ValueError("G05 off branch must use legacy training without experiment_dir or val_dataset_config")
    if args.seed is None:
        raise ValueError("G05 requires one explicit shared training seed")
    args.network_weights = str(weights)
    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"
    qwen_image_utils.resolve_model_version_args(args)
    return args


def _g05_worker(args) -> None:
    config = _read_g05_args(args.config, args.weights, args.mode)
    args.out.mkdir(parents=True, exist_ok=False)
    trainer = _G05Trainer(args.mode, args.out)
    trainer.train(config)
    expected_updates = 0 if args.mode == "snapshot" else 4
    expected_events = [0] if args.mode == "snapshot" else ([0, 4] if args.mode == "on" else [])
    if len(trainer.updates) != expected_updates:
        raise AssertionError(f"G05 {args.mode} completed {len(trainer.updates)} rather than {expected_updates} updates")
    if [event["absolute_step"] for event in trainer.evaluations] != expected_events:
        raise AssertionError(f"G05 {args.mode} validation steps differ from {expected_events}")
    record = {
        "mode": args.mode,
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "weights_sha256": _sha256(args.weights),
        "initial_state": trainer.initial_state,
        "final_state": trainer.final_state,
        "batches": trainer.batches,
        "updates": trainer.updates,
        "evaluations": trainer.evaluations,
        "completed_optimizer_updates": len(trainer.updates),
    }
    (args.out / "trace.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _g05(args) -> None:
    selection = args.initial_adapter.read_text(encoding="utf-8").strip()
    weights = Path(selection).expanduser().resolve()
    if weights.is_dir():
        weights /= "model.safetensors"
    if not weights.is_file():
        raise ValueError("G05 selected adapter must be an existing model.safetensors file")
    output = args.out or args.initial_adapter.parent / "G05" / "probe"
    output.mkdir(parents=True, exist_ok=False)
    configs = {"snapshot": args.snapshot_config, "on": args.enabled, "off": args.disabled}
    preflight = {name: _read_g05_args(path, weights, name) for name, path in configs.items()}
    for key in ("seed", "gradient_accumulation_steps", "learning_rate", "optimizer_type", "lr_scheduler",
                "lr_warmup_steps", "dataset_config", "network_dim", "network_alpha"):
        values = [getattr(preflight[name], key) for name in ("snapshot", "on", "off")]
        if key != "dataset_config" and values[1] != values[2]:
            raise ValueError(f"G05 on/off effective {key} differs before training")
    records = {}
    for mode in ("snapshot", "on", "off"):
        folder = output / mode
        command = [
            sys.executable, str(Path(__file__).resolve()), "_g05_worker",
            "--mode", mode, "--config", str(configs[mode]), "--weights", str(weights), "--out", str(folder),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise RuntimeError(f"G05 {mode} worker failed with exit {completed.returncode}: {command}")
        records[mode] = json.loads((folder / "trace.json").read_text(encoding="utf-8"))
    snap, enabled, disabled = (records[name] for name in ("snapshot", "on", "off"))
    if snap["initial_state"] != snap["final_state"]:
        raise AssertionError("G05 snapshot changed training state around validation")
    if len(enabled["updates"]) != 4 or len(disabled["updates"]) != 4:
        raise AssertionError("G05 paired branches did not complete exactly four updates")
    starts = [records[name]["initial_state"]["adapter"] for name in ("snapshot", "on", "off")]
    if len(set(starts)) != 1:
        raise AssertionError("G05 branches did not load the same initial adapter")
    for key in ("optimizer", "scheduler", "python_rng", "numpy_rng", "torch_cpu_rng", "torch_cuda_rng"):
        if enabled["initial_state"][key] != disabled["initial_state"][key]:
            raise AssertionError(f"G05 on/off initial {key} differs before controlled updates")
    on_batches, off_batches = enabled["batches"], disabled["batches"]
    if len(on_batches) != len(off_batches) or not on_batches:
        raise AssertionError("G05 paired branches have different microbatch counts")
    loss_differences = []
    for on, off in zip(on_batches, off_batches):
        for key in ("latent_sha256", "text_sha256", "noise_sha256", "noisy_sha256", "timesteps", "global_step_before"):
            if on[key] != off[key]:
                raise AssertionError(f"G05 paired input/order/noise/timestep differs in {key}")
        loss_differences.append(abs(on["loss"] - off["loss"]))
        if not math.isclose(on["loss"], off["loss"], rel_tol=1e-3, abs_tol=1e-4):
            raise AssertionError("G05 paired loss differs beyond predeclared tolerance")
    adapter_differences = []
    changed_after_nonzero_lr = False
    previous = None
    for on, off in zip(enabled["updates"], disabled["updates"]):
        if on["learning_rates"] != off["learning_rates"]:
            raise AssertionError("G05 paired learning-rate schedules differ")
        on_state = load_file(on["adapter_path"], device="cpu")
        off_state = load_file(off["adapter_path"], device="cpu")
        if on_state.keys() != off_state.keys():
            raise AssertionError("G05 paired LoRA tensor names differ")
        differences = {}
        for name in on_state:
            left, right = on_state[name], off_state[name]
            if left.shape != right.shape or left.dtype != right.dtype:
                raise AssertionError(f"G05 paired LoRA tensor metadata differs for {name}")
            if left.is_floating_point():
                if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                    raise AssertionError(f"G05 nonfinite LoRA tensor {name}")
                if not torch.isfinite(torch.linalg.vector_norm(left.double())) or not torch.isfinite(torch.linalg.vector_norm(right.double())):
                    raise AssertionError(f"G05 nonfinite LoRA norm for {name}")
                differences[name] = float((left - right).abs().max().item())
                if not torch.allclose(left, right, rtol=5e-3, atol=1e-5):
                    raise AssertionError(f"G05 paired LoRA tensor {name} differs beyond predeclared tolerance")
            elif not torch.equal(left, right):
                raise AssertionError(f"G05 paired LoRA integer tensor differs for {name}")
        if previous is not None and any(rate > 0 for rate in on["learning_rates"]):
            changed_after_nonzero_lr |= any(
                not torch.equal(previous[name], on_state[name])
                for name in on_state if on_state[name].is_floating_point()
            )
        previous = on_state
        adapter_differences.append(differences)
    if not changed_after_nonzero_lr:
        raise AssertionError("G05 LoRA did not change after a nonzero learning-rate update")
    summary = {
        "status": "PASS",
        "weights_sha256": _sha256(weights),
        "snapshot_validation_calls": 1,
        "enabled_validation_steps": [event["absolute_step"] for event in enabled["evaluations"]],
        "disabled_validation_steps": [event["absolute_step"] for event in disabled["evaluations"]],
        "completed_optimizer_updates": {"snapshot": 0, "on": 4, "off": 4},
        "paired_loss_absolute_differences": loss_differences,
        "paired_adapter_max_abs_differences": adapter_differences,
        "loss_tolerance": {"rtol": 1e-3, "atol": 1e-4},
        "adapter_tolerance": {"rtol": 5e-3, "atol": 1e-5},
        "worker_evidence": ["snapshot/trace.json", "on/trace.json", "off/trace.json"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="case", required=True)
    g03 = subcommands.add_parser("g03", help="five real-model validation calls; zero optimizer updates")
    g03.add_argument("--checkpoint", type=Path, required=True, help="text file containing the selected package path")
    g03.add_argument("--config", type=Path, required=True)
    g03.add_argument("--renamed-config", type=Path, required=True)
    g03.add_argument("--ten-by-one-config", type=Path, required=True)
    g03.add_argument("--out", type=Path)
    g04 = subcommands.add_parser("g04", help="one real-model validation event with independent reduction")
    g04.add_argument("--checkpoint", type=Path, required=True)
    g04.add_argument("--config", type=Path, required=True)
    g04.add_argument("--out", type=Path, required=True)
    g05 = subcommands.add_parser("g05", help="one no-update state snapshot and two sequential four-update branches")
    g05.add_argument("--snapshot-config", type=Path, required=True)
    g05.add_argument("--enabled", type=Path, required=True)
    g05.add_argument("--disabled", type=Path, required=True)
    g05.add_argument("--initial-adapter", type=Path, required=True, help="text file containing adapter path")
    g05.add_argument("--out", type=Path)
    worker = subcommands.add_parser("_g03_worker", help=argparse.SUPPRESS)
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--weights", type=Path, required=True)
    worker.add_argument("--out", type=Path, required=True)
    worker.add_argument("--repeat", type=int, required=True)
    worker.add_argument("--levels", type=int, required=True)
    worker.add_argument("--seeds", type=int, required=True)
    g05_worker = subcommands.add_parser("_g05_worker", help=argparse.SUPPRESS)
    g05_worker.add_argument("--mode", choices=("snapshot", "on", "off"), required=True)
    g05_worker.add_argument("--config", type=Path, required=True)
    g05_worker.add_argument("--weights", type=Path, required=True)
    g05_worker.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.case == "g03":
        _g03(args)
    elif args.case == "g04":
        _g04(args)
    elif args.case == "g05":
        _g05(args)
    elif args.case == "_g05_worker":
        _g05_worker(args)
    else:
        if args.repeat not in (1, 2) or (args.levels, args.seeds) not in ((4, 2), (10, 1)):
            raise ValueError("G03 worker accepts only the predeclared 4x2/10x1 protocol")
        _worker(args.config, args.weights, args.out, args.repeat, args.levels, args.seeds)


if __name__ == "__main__":
    main()
