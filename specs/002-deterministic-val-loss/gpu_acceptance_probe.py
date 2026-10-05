"""Run after four real cache passes: normal Qwen setup, then bounded acceptance probes.

Example: python specs/002-deterministic-val-loss/gpu_acceptance_probe.py
    --config_file /workspace/acceptance/train.toml --probe_report output/probe.json

This is acceptance code. It reuses one loaded DiT and snapshots only the adapter,
optimizer, scheduler and RNG. Paired updates use identical real cached batches;
this does not claim exact restoration of the training loader's position.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

import accelerate
import numpy as np
import torch

from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner import qwen_image_train_network as qwen
from musubi_tuner.training import validation


LOSS_RTOL = 1e-4
LOSS_ATOL = 1e-5
# Declare before the real-model run: every paired update tensor must match exactly.
PAIRED_RTOL = 0.0
PAIRED_ATOL = 0.0


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    descriptor = json.dumps([str(value.dtype), list(value.shape)], separators=(",", ":")).encode("ascii")
    return hashlib.sha256(descriptor + value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def rng_state():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    )


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


def assert_exact(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        if left.dtype != right.dtype or not torch.equal(left, right):
            raise AssertionError(f"{path}: paired tensors differ")
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            raise AssertionError(f"{path}: paired NumPy values differ")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"{path}: paired state keys differ")
        for key in left:
            assert_exact(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"{path}: paired state lengths differ")
        for index, (a, b) in enumerate(zip(left, right)):
            assert_exact(a, b, f"{path}[{index}]")
    elif left != right:
        raise AssertionError(f"{path}: paired values differ: {left!r} != {right!r}")


def first_gradient_difference(names, left, right):
    """Describe the first exact mismatch without retaining tensors in the JSON report."""
    for name, a, b in zip(names, left, right):
        if a is None and b is None:
            continue
        if a is not None and b is not None and a.dtype == b.dtype and torch.equal(a, b):
            continue
        result = {"parameter": name}
        for label, tensor in (("control", a), ("replay", b)):
            result[label] = (
                None if tensor is None else {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": tensor_hash(tensor)}
            )
        if a is not None and b is not None and a.shape == b.shape:
            a, b = a.detach().double().cpu(), b.detach().double().cpu()
            difference = a - b
            result["unequal_elements"] = int(torch.count_nonzero(a != b))
            for label, value in (("max_abs", difference.abs().max().item()), ("rms", difference.square().mean().sqrt().item())):
                result[label] = value if math.isfinite(value) else "non-finite"
        return result
    return None


def run_probe(
    trainer, args, accelerator, transformer, network, optimizer, lr_scheduler, batches, dit_dtype, network_dtype, report=None
):
    if accelerator.num_processes != 1:
        raise ValueError("acceptance isolation probe uses one GPU/process; run the separate multi-process acceptance scenario")
    if len(batches) != args.gradient_accumulation_steps:
        raise ValueError("probe requires one real cached train batch for each accumulation microbatch")
    adapter = accelerator.unwrap_model(network)
    parameter_names = [name for name, _ in adapter.named_parameters()]
    model = accelerator.unwrap_model(transformer)
    adapter.train()
    inputs = trainer.validation_inputs
    noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")
    expected_count = len(inputs.records) * args.val_level_noise_n * args.val_seed_noise_n
    if report is None:
        report = {}
    report.update(
        {
            "status": "running",
            "torch": torch.__version__,
            "accelerate": accelerate.__version__,
            "device": str(accelerator.device),
            "device_name": torch.cuda.get_device_name(accelerator.device) if accelerator.device.type == "cuda" else "CPU fixture",
            "mixed_precision": accelerator.mixed_precision,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "network_dropout": args.network_dropout,
            "gradient_checkpointing": args.gradient_checkpointing,
            "validation_fingerprint": inputs.fingerprint,
            "image_count": len(inputs.records),
            "level_count": args.val_level_noise_n,
            "realization_count": args.val_seed_noise_n,
            "loss_tolerance": {"rtol": LOSS_RTOL, "atol": LOSS_ATOL},
            "paired_tolerance": {"rtol": PAIRED_RTOL, "atol": PAIRED_ATOL},
            "batch_latent_sha256": [tensor_hash(batch["latents"]) for batch in batches],
            "completed_updates": {},
            "microbatch_sync": {},
            "restored_state_exact": {},
            "determinism": {
                "requested": getattr(trainer, "probe_deterministic", False),
                "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
                "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "sdpa_enabled": {
                    "flash": torch.backends.cuda.flash_sdp_enabled(),
                    "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                    "math": torch.backends.cuda.math_sdp_enabled(),
                    "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
                },
            },
        }
    )

    def evaluate_with_noise_trace():
        original = validation.validation_noise_like
        traces = []

        def recorded(latents, seed):
            noise = original(latents, seed)
            traces.append({"seed": seed, "noise_sha256": tensor_hash(noise)})
            return noise

        validation.validation_noise_like = recorded
        try:
            result = validation.evaluate_validation(
                trainer, args, accelerator, transformer, network, inputs, noise_scheduler, dit_dtype, network_dtype
            )
        finally:
            validation.validation_noise_like = original
        if result.forward_count != expected_count or len(traces) != expected_count:
            raise AssertionError("validation forward/noise count differs from M*N1*N2")
        if not math.isclose(
            result.loss_mean, (result.loss_low_noise + result.loss_high_noise) / 2, rel_tol=LOSS_RTOL, abs_tol=LOSS_ATOL
        ):
            raise AssertionError("validation mean differs from (low+high)/2")
        return result, traces

    rng_before = copy.deepcopy(rng_state())
    first, first_noise = evaluate_with_noise_trace()
    second, second_noise = evaluate_with_noise_trace()
    report["repeatability"] = {
        "first": first.metrics,
        "second": second.metrics,
        "forward_count_each": first.forward_count,
        "noise": first_noise,
        "second_noise": second_noise,
    }
    assert_exact(first_noise, second_noise, "repeated_noise")
    assert_exact(rng_before, rng_state(), "repeated_evaluation_rng")
    for key in first.metrics:
        if not math.isclose(first.metrics[key], second.metrics[key], rel_tol=LOSS_RTOL, abs_tol=LOSS_ATOL):
            raise AssertionError(f"{key}: repeated loss exceeds predeclared tolerance")

    def snapshot():
        return copy.deepcopy(
            {
                "adapter": adapter.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "rng": rng_state(),
                "accelerator_step": accelerator.step,
                "scaler": accelerator.scaler.state_dict() if accelerator.scaler is not None else None,
                "gradients": [parameter.grad for parameter in adapter.parameters()],
                "modes": [module.training for root in (model, adapter) for module in root.modules()],
            }
        )

    def restore(saved):
        adapter.load_state_dict(saved["adapter"], strict=True)
        # Generic optimizers may retain tensor aliases to the passed dictionary.
        # Each branch must restore the same immutable initial snapshot.
        optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
        lr_scheduler.load_state_dict(copy.deepcopy(saved["scheduler"]))
        accelerator.step = saved["accelerator_step"]
        if accelerator.scaler is not None:
            accelerator.scaler.load_state_dict(saved["scaler"])
        for parameter, gradient in zip(adapter.parameters(), saved["gradients"]):
            parameter.grad = None if gradient is None else gradient.clone()
        for module, mode in zip((module for root in (model, adapter) for module in root.modules()), saved["modes"]):
            module.training = mode
        restore_rng(saved["rng"])

    def update(stage):
        trace, pre_clip_gradients, gradients, losses = [], [], [], []
        original_call = trainer.call_dit

        def observed(*positional, **keywords):
            trace.append(
                {"latent": tensor_hash(positional[3]), "noise": tensor_hash(positional[5]), "timesteps": tensor_hash(positional[7])}
            )
            return original_call(*positional, **keywords)

        trainer.call_dit = observed
        optimizer.zero_grad(set_to_none=True)
        completed = 0
        report["microbatch_sync"][stage] = []
        try:
            for batch in batches:
                with accelerator.accumulate(network):
                    report["microbatch_sync"][stage].append(accelerator.sync_gradients)
                    adapter.on_step_start()
                    latents = trainer.scale_shift_latents(batch["latents"])
                    noise = torch.randn_like(latents)
                    loss, _ = trainer.process_batch(
                        args,
                        accelerator,
                        transformer,
                        network,
                        batch,
                        latents,
                        noise,
                        noise_scheduler,
                        dit_dtype,
                        network_dtype,
                        None,
                        0,
                    )
                    losses.append(loss.detach().item())
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        pre_clip_gradients = [
                            parameter.grad.detach().clone() if parameter.grad is not None else None
                            for parameter in adapter.parameters()
                        ]
                        if args.max_grad_norm != 0.0:
                            accelerator.clip_grad_norm_(adapter.get_trainable_params(), args.max_grad_norm)
                        gradients = [
                            parameter.grad.detach().clone() if parameter.grad is not None else None
                            for parameter in adapter.parameters()
                        ]
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    completed += int(accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped)
        finally:
            trainer.call_dit = original_call
            report["completed_updates"][stage] = completed
        if completed != 1:
            raise AssertionError(f"probe expected one completed optimizer update, observed {completed}")
        return {"trace": trace, "pre_clip_gradients": pre_clip_gradients, "post_clip_gradients": gradients, "losses": losses}

    def compare_updates(control, replay, expected_state):
        differences = {
            stage: first_gradient_difference(parameter_names, control[stage], replay[stage])
            for stage in ("pre_clip_gradients", "post_clip_gradients")
        }
        errors = {}
        for name, a, b in (
            ("train_inputs_noise_timesteps", control["trace"], replay["trace"]),
            ("train_losses", control["losses"], replay["losses"]),
            ("next_update", expected_state, snapshot()),
        ):
            try:
                assert_exact(a, b, name)
            except AssertionError as error:
                errors[name] = str(error)
        return {
            "exact": not errors and all(value is None for value in differences.values()),
            "control_train_losses": control["losses"],
            "measured_train_losses": replay["losses"],
            "train_trace": control["trace"],
            "measured_train_trace": replay["trace"],
            "gradient_differences": differences,
            "errors": errors,
        }

    update("warmup")  # Initialize real optimizer state, including AdamW8bit allocations, before branching.
    initial = snapshot()
    control = update("control")
    expected = snapshot()
    restore(initial)
    assert_exact(initial, snapshot(), "restored_no_validation_state")
    report["restored_state_exact"]["control_replay"] = True
    control_replay = update("control_replay")
    report["no_validation_replay"] = compare_updates(control, control_replay, expected)
    del control_replay
    if not report["no_validation_replay"]["exact"]:
        raise AssertionError("same-state no-validation replay differs; validation isolation is not established")
    restore(initial)
    assert_exact(initial, snapshot(), "restored_validation_state")
    report["restored_state_exact"]["measured"] = True
    isolated, _ = evaluate_with_noise_trace()
    assert_exact(initial, snapshot(), "validation_unchanged_state")
    measured = update("measured")
    report["isolation"] = compare_updates(control, measured, expected)
    report["isolation"]["validation_metrics"] = isolated.metrics
    report["isolation"]["adapter_optimizer_scheduler_gradients_rng_equal"] = report["isolation"]["exact"]
    if not report["isolation"]["exact"]:
        raise AssertionError("validation next-update comparison differs; see recorded gradient/state diagnostics")
    report["status"] = "passed"
    return report


class AcceptanceTrainer(qwen.QwenImageNetworkTrainer):
    def __init__(self, report_path, deterministic=False):
        super().__init__()
        self.report_path = report_path
        self.probe_deterministic = deterministic

    def _run_training_loop(
        self,
        args,
        accelerator,
        session_id,
        training_started_at,
        train_dataset_group,
        train_dataloader,
        current_epoch,
        transformer,
        network,
        training_model,
        optimizer,
        optimizer_name,
        optimizer_args,
        optimizer_train_fn,
        optimizer_eval_fn,
        lr_scheduler,
        lr_descriptions,
        sample_resources,
        sample_parameters,
        dit_dtype,
        network_dtype,
    ):
        destination = Path(self.report_path)
        if not destination.is_absolute():
            destination = Path(getattr(args, "_experiment_root", Path(args._config_source).resolve().parent)) / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "failed",
            "loss_tolerance": {"rtol": LOSS_RTOL, "atol": LOSS_ATOL},
            "paired_tolerance": {"rtol": 0, "atol": 0},
        }
        try:
            if not args.val_dataset_config or args.val_level_noise_n != 10 or args.val_seed_noise_n != 2:
                raise ValueError("full acceptance probe requires validation with N1=10 and N2=2")
            if not args.gradient_checkpointing or not args.network_dropout or args.gradient_accumulation_steps <= 1:
                raise ValueError("acceptance probe requires checkpointing, dropout>0 and accumulation>1")
            batches = []
            iterator = iter(train_dataloader)
            try:
                for _ in range(args.gradient_accumulation_steps):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(train_dataloader)
                        batch = next(iterator)
                    batches.append(batch)
            finally:
                # Accelerate 1.6 calls end() after normal iteration, not in finally.
                # Closing at its last yield leaves a stale end-of-loader flush flag.
                try:
                    if hasattr(iterator, "close"):
                        iterator.close()
                finally:
                    train_dataloader.end()
            del iterator
            optimizer_train_fn()
            report = run_probe(
                self, args, accelerator, transformer, network, optimizer, lr_scheduler, batches, dit_dtype, network_dtype, report
            )
        except Exception as error:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--probe_report", required=True, help="Acceptance report path, relative to the experiment")
    parser.add_argument(
        "--probe_deterministic", action="store_true", help="Acceptance only: require deterministic PyTorch algorithms"
    )
    options, training_arguments = parser.parse_known_args()
    original_trainer, original_argv = qwen.QwenImageNetworkTrainer, sys.argv
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        if options.probe_deterministic:
            torch.use_deterministic_algorithms(True, warn_only=False)
        qwen.QwenImageNetworkTrainer = lambda: AcceptanceTrainer(options.probe_report, options.probe_deterministic)
        sys.argv = [original_argv[0], *training_arguments]
        qwen.main()
    finally:
        qwen.QwenImageNetworkTrainer, sys.argv = original_trainer, original_argv
        torch.use_deterministic_algorithms(original_deterministic, warn_only=original_warn_only)


if __name__ == "__main__":
    main()
