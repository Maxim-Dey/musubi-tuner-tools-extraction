"""Run normal Qwen training, proving restored state before its next update.

python specs/002-deterministic-val-loss/gpu_train_acceptance.py
    --resume_report output/resume-proof.json --config_file /experiment/train.toml
    --resume output/qwen-step3 --max_train_steps 5

Acceptance-only, for the observed single-GPU environment and Accelerate 1.6.
The report path is relative to the experiment/config directory. No base-model
snapshot is taken, and the verification restores the RNG state it observes.
"""

import argparse
import json
from pathlib import Path
import sys

import accelerate
from accelerate.utils import load
import torch

from gpu_acceptance_probe import assert_exact, rng_state
from musubi_tuner import qwen_image_train_network as qwen
from musubi_tuner.training.experiment_state import (
    adapter_identity,
    load_training_state,
    validate_experiment_checkpoint,
)
from musubi_tuner.training.validation import preserve_rng_state


def _cpu_state(value):
    """Compare optimizer tensors exactly across their saved/loaded devices."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_state(item) for item in value)
    return value


def verify_loaded_state(args, accelerator, network, directory):
    """Called immediately after native load_state, before trainer setup resumes."""
    with preserve_rng_state():
        observed_rng = rng_state()
        directory = Path(directory)
        manifest = validate_experiment_checkpoint(directory)
        saved_trainer = load_training_state(directory)
        step = saved_trainer["optimizer_global_step"]
        assert_exact(step, args._training_state["optimizer_global_step"], "optimizer_global_step")
        adapter = accelerator.unwrap_model(network)
        identity = adapter_identity(adapter)
        assert_exact(manifest["adapter_identity"], identity, "adapter_identity")

        checked = ["model.safetensors", "trainer_state.json"]
        # These are the actual prepared objects traversed by Accelerator 1.6 load_state.
        for kind, objects in (("optimizer", accelerator._optimizers), ("scheduler", accelerator._schedulers)):
            if not objects:
                raise AssertionError(f"resume proof requires a prepared {kind}")
            for index, item in enumerate(objects):
                filename = f"{kind}{'_' + str(index) if index else ''}.bin"
                expected = load(directory / filename, map_location="cpu")
                assert_exact(expected, _cpu_state(item.state_dict()), filename)
                checked.append(filename)
        if accelerator.scaler is not None:
            assert_exact(load(directory / "scaler.pt", map_location="cpu"), _cpu_state(accelerator.scaler.state_dict()), "scaler")
            checked.append("scaler.pt")

        filename = f"random_states_{accelerator.process_index}.pkl"
        saved_rng = load(directory / filename, map_location="cpu")
        checked.append(filename)
        expected_cuda = saved_rng.get("torch_cuda_manual_seed")
        if accelerator.device.type == "cuda" and expected_cuda is None:
            raise AssertionError("checkpoint has no CUDA RNG state")
        assert_exact(
            (saved_rng["random_state"], saved_rng["numpy_random_seed"], saved_rng["torch_manual_seed"], expected_cuda),
            observed_rng,
            "restored_rng",
        )
        assert_exact(saved_rng["step"], accelerator.step, "accelerator_step")
        return {
            "status": "passed",
            "verified_before_next_update": True,
            "comparison": "exact, with optimizer tensors compared on CPU",
            "torch": torch.__version__,
            "accelerate": accelerate.__version__,
            "checkpoint": str(directory.resolve()),
            "optimizer_global_step": step,
            "accelerator_step": accelerator.step,
            "adapter_identity": identity,
            "adapter_tensor_count": len(adapter.state_dict()),
            "optimizer_count": len(accelerator._optimizers),
            "scheduler_count": len(accelerator._schedulers),
            "rng": {"python": True, "numpy": True, "torch_cpu": True, "cuda_devices_checked": len(expected_cuda or [])},
            "validation_fingerprint": saved_trainer["validation_fingerprint"],
            "last_validation": saved_trainer["last_validation"],
            "checked_files": {name: manifest["files"][name]["sha256"] for name in checked},
        }


class ResumeProofTrainer(qwen.QwenImageNetworkTrainer):
    def __init__(self, report_path):
        super().__init__()
        self.report_path = report_path

    def _register_hooks_and_resume(self, args, accelerator, network):
        if not args.resume or not args.experiment_mode or accelerator.num_processes != 1:
            raise ValueError("resume acceptance requires a complete experiment checkpoint and exactly one process")
        destination = Path(self.report_path)
        if not destination.is_absolute():
            destination = Path(args._experiment_root) / destination
        native_load = accelerator.load_state

        def verified_load(directory, **kwargs):
            report = {"status": "failed", "checkpoint": str(directory)}
            try:
                result = native_load(directory, **kwargs)
                report = verify_loaded_state(args, accelerator, network, directory)
                return result
            except Exception as error:
                report["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                with preserve_rng_state():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

        accelerator.load_state = verified_load
        try:
            super()._register_hooks_and_resume(args, accelerator, network)
        finally:
            accelerator.load_state = native_load


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--resume_report", required=True)
    options, training_arguments = parser.parse_known_args()
    original_trainer, original_argv = qwen.QwenImageNetworkTrainer, sys.argv
    try:
        qwen.QwenImageNetworkTrainer = lambda: ResumeProofTrainer(options.resume_report)
        sys.argv = [original_argv[0], *training_arguments]
        qwen.main()
    finally:
        qwen.QwenImageNetworkTrainer, sys.argv = original_trainer, original_argv


if __name__ == "__main__":
    main()
