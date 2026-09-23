"""Reject unusable rank RNG payloads before Accelerate can ignore load errors."""

from __future__ import annotations

from types import SimpleNamespace
import random
import struct

from accelerate import Accelerator
import numpy as np
import pytest
import torch

from musubi_tuner.training.experiment_states import load_package, save_package


def _package_fixture(tmp_path):
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    args = SimpleNamespace(
        experiment_dir=str(tmp_path), output_dir=str(tmp_path / "output"), output_name="tiny",
        save_precision="fp32", save_last_n_steps=None,
        val_every_n_steps=2, val_seed_noise=42, val_level_noise_n=10, val_seed_noise_n=1,
    )
    manifest = SimpleNamespace(fingerprint="a" * 64)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    accelerator.register_save_state_pre_hook(
        lambda _models, weights, _directory: weights.__setitem__(slice(None), [accelerator.unwrap_model(model).state_dict()])
    )
    accelerator.register_load_state_pre_hook(
        lambda models, _directory: models.__setitem__(slice(None), [accelerator.unwrap_model(model)])
    )
    random.seed(103)
    np.random.seed(104)
    torch.manual_seed(105)
    package = save_package(args, accelerator, model, manifest, 3, None, ["final"])
    return accelerator, args, model, manifest, package


@pytest.mark.parametrize("damage", ["python", "numpy"])
def test_deserializable_unusable_rng_is_rejected_before_accelerate_load(tmp_path, monkeypatch, damage):
    accelerator, args, model, manifest, package = _package_fixture(tmp_path)
    try:
        rng_file = package / "random_states_0.pkl"
        payload = torch.load(rng_file, map_location="cpu", weights_only=False)
        payload["random_state" if damage == "python" else "numpy_random_seed"] = ()
        torch.save(payload, rng_file)
        monkeypatch.setattr(
            accelerator, "load_state", lambda *_args, **_kwargs: pytest.fail("accelerator.load_state preceded RNG preflight")
        )
        with pytest.raises((ValueError, RuntimeError), match="RNG|random|NumPy|invalid|corrupt"):
            load_package(args, accelerator, model, manifest, package)
    finally:
        accelerator.free_memory()


def test_valid_cpu_rng_payload_still_restores_all_three_streams(tmp_path):
    accelerator, args, model, manifest, package = _package_fixture(tmp_path)
    try:
        expected_python = random.getstate()
        expected_numpy = np.random.get_state()
        expected_torch = torch.get_rng_state().clone()
        random.random(), np.random.rand(), torch.rand(())
        assert load_package(args, accelerator, model, manifest, package) == 3
        assert random.getstate() == expected_python
        restored_numpy = np.random.get_state()
        assert restored_numpy[0] == expected_numpy[0]
        np.testing.assert_array_equal(restored_numpy[1], expected_numpy[1])
        assert restored_numpy[2:] == expected_numpy[2:]
        torch.testing.assert_close(torch.get_rng_state(), expected_torch, rtol=0, atol=0)
    finally:
        accelerator.free_memory()


def _cuda_state(*, offset=0):
    # PyTorch 2.7.1 CUDA Philox state: uint64 seed + int64 offset, both on CPU.
    return torch.tensor(list(struct.pack("=Qq", 7, offset)), dtype=torch.uint8)


@pytest.mark.parametrize(
    "damage",
    ["empty", "device-count", "dtype", "shape", "size", "offset"],
)
def test_simulated_cuda_rng_payload_fails_before_accelerate_load(tmp_path, monkeypatch, damage):
    accelerator, args, model, manifest, package = _package_fixture(tmp_path)
    try:
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda *_args: pytest.fail("CUDA execution during CPU preflight"))
        rng_file = package / "random_states_0.pkl"
        payload = torch.load(rng_file, map_location="cpu", weights_only=False)
        states = [_cuda_state(), _cuda_state()]
        if damage == "empty":
            states = []
        elif damage == "device-count":
            states.pop()
        elif damage == "dtype":
            states[0] = states[0].float()
        elif damage == "shape":
            states[0] = states[0].reshape(2, 8)
        elif damage == "size":
            states[0] = states[0][:3]
        elif damage == "offset":
            states[0] = _cuda_state(offset=1)
        payload["torch_cuda_manual_seed"] = states
        torch.save(payload, rng_file)
        simulated_cuda = SimpleNamespace(
            device=SimpleNamespace(type="cuda"), num_processes=1, is_main_process=True,
            unwrap_model=accelerator.unwrap_model,
            load_state=lambda *_args: pytest.fail("accelerator.load_state preceded CUDA RNG preflight"),
        )
        with pytest.raises((ValueError, RuntimeError), match="CUDA|cuda|RNG|random|state"):
            load_package(args, simulated_cuda, model, manifest, package)
    finally:
        accelerator.free_memory()


@pytest.mark.parametrize("legacy", [False, True])
def test_simulated_valid_cuda_rng_state_passes_preflight_without_cuda_execution(tmp_path, monkeypatch, legacy):
    accelerator, args, model, manifest, package = _package_fixture(tmp_path)
    try:
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda *_args: pytest.fail("CUDA execution during CPU preflight"))
        rng_file = package / "random_states_0.pkl"
        payload = torch.load(rng_file, map_location="cpu", weights_only=False)
        payload["torch_cuda_manual_seed"] = (
            [_cuda_state()[:8], _cuda_state()[:8]] if legacy else [_cuda_state(), _cuda_state(offset=4)]
        )
        torch.save(payload, rng_file)
        loaded = []
        simulated_cuda = SimpleNamespace(
            device=SimpleNamespace(type="cuda"), num_processes=1, is_main_process=True,
            unwrap_model=accelerator.unwrap_model, load_state=lambda path: loaded.append(path),
        )
        assert load_package(args, simulated_cuda, model, manifest, package) == 3
        assert loaded == [str(package)]
    finally:
        accelerator.free_memory()
