"""Acceptance wrapper observes actual native resume before the next CPU update."""

import importlib.util
import json
from pathlib import Path
import sys

from accelerate import Accelerator
from accelerate.utils import load
import pytest
import torch

from musubi_tuner.training.experiment_state import prepare_training_state, register_experiment_hooks, save_experiment_checkpoint
from test_experiment_state import arguments, tiny_adapter, update


def load_wrapper(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "specs/002-deterministic-val-loss"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("gpu_train_acceptance", directory / "gpu_train_acceptance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepared():
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    base, network = tiny_adapter()
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8)
    base, network, optimizer, scheduler = accelerator.prepare(base, network, optimizer, scheduler)
    return accelerator, network, optimizer, scheduler


@pytest.mark.parametrize("failure", [None, "optimizer", "rng"])
def test_actual_native_resume_proof_and_silent_load_failure_detection(tmp_path, monkeypatch, failure):
    wrapper = load_wrapper(monkeypatch)
    accelerator, network, optimizer, scheduler = prepared()
    args = arguments(tmp_path)
    state = prepare_training_state(args)
    register_experiment_hooks(args, accelerator, network, state)
    for _ in range(3):
        update(optimizer, scheduler, network)
    state["optimizer_global_step"] = 3
    accelerator.step = 6  # Deliberately distinct from completed optimizer updates.
    directory = save_experiment_checkpoint(args, accelerator, network, state, {}, with_state=True)
    saved_rng = load(directory / "random_states_0.pkl", map_location="cpu")
    accelerator.free_memory()

    accelerator, network, optimizer, scheduler = prepared()
    args = arguments(tmp_path, resume=str(directory))
    prepare_training_state(args)
    native_load = accelerator.load_state
    if failure is not None:
        def incomplete_load(*positional, **keywords):
            native_load(*positional, **keywords)
            if failure == "optimizer":
                optimizer.param_groups[0]["lr"] = 999.0
            else:
                torch.rand(1)

        monkeypatch.setattr(accelerator, "load_state", incomplete_load)
    original_load = accelerator.load_state
    trainer = wrapper.ResumeProofTrainer("output/resume-proof.json")
    try:
        if failure is None:
            trainer._register_hooks_and_resume(args, accelerator, network)
            wrapper.assert_exact(
                (saved_rng["random_state"], saved_rng["numpy_random_seed"], saved_rng["torch_manual_seed"], None),
                wrapper.rng_state(),
                "RNG after report writing",
            )
            report = json.loads((tmp_path / "output/resume-proof.json").read_text())
            assert report["status"] == "passed" and report["verified_before_next_update"]
            assert report["optimizer_global_step"] == 3 and report["accelerator_step"] == 6
            assert report["optimizer_count"] == report["scheduler_count"] == 1
            assert report["rng"] == {"python": True, "numpy": True, "torch_cpu": True, "cuda_devices_checked": 0}
            assert {"model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"} <= report["checked_files"].keys()
            update(optimizer, scheduler, network)
            assert all(item["step"].item() == 4 for item in optimizer.state.values())
        else:
            with pytest.raises(AssertionError, match="optimizer.bin|restored_rng"):
                trainer._register_hooks_and_resume(args, accelerator, network)
            report = json.loads((tmp_path / "output/resume-proof.json").read_text())
            assert report["status"] == "failed" and "AssertionError" in report["error"]
        assert accelerator.load_state == original_load
    finally:
        accelerator.free_memory()


def test_wrapper_delegates_normal_qwen_main_and_restores_arguments(monkeypatch):
    wrapper = load_wrapper(monkeypatch)
    original_trainer = wrapper.qwen.QwenImageNetworkTrainer
    arguments = ["gpu_train_acceptance.py", "--resume_report", "output/proof.json", "--config_file", "train.toml", "--resume", "state"]
    monkeypatch.setattr(sys, "argv", arguments)
    seen = {}

    def normal_main():
        seen["arguments"] = sys.argv[1:]
        seen["trainer"] = wrapper.qwen.QwenImageNetworkTrainer()

    monkeypatch.setattr(wrapper.qwen, "main", normal_main)
    wrapper.main()
    assert seen["arguments"] == ["--config_file", "train.toml", "--resume", "state"]
    assert isinstance(seen["trainer"], wrapper.ResumeProofTrainer)
    assert seen["trainer"].report_path == "output/proof.json"
    assert sys.argv is arguments and wrapper.qwen.QwenImageNetworkTrainer is original_trainer
