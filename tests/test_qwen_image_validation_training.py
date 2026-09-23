"""Controlled CPU contracts for Stage 2 Qwen-Image validation loss."""

import hashlib
import json
import math
import copy
from contextlib import contextmanager
from datetime import timedelta
import multiprocessing
import os
from pathlib import Path
import random
import socket
from types import SimpleNamespace

from accelerate import Accelerator
import numpy as np
from PIL import Image
import pytest
from safetensors.torch import save_file
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
import torch.distributed as dist

import musubi_tuner.training.trainer_base as trainer_base
from musubi_tuner.qwen_image_train_network import QwenImageNetworkTrainer
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer
from musubi_tuner.training.validation_inputs import iter_noise_checks, prepare_validation_inputs, read_validation_cache_pair


TAGS = {
    "train_eval_loss_mean",
    "train_eval_loss_low_noise",
    "train_eval_loss_high_noise",
    "val_loss_mean",
    "val_loss_low_noise",
    "val_loss_high_noise",
}


class TinyQwenBoundary(torch.nn.Module):
    """A small forward boundary for the existing Qwen call_dit path."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.timesteps = []
        self.inputs_require_grad = []

    def forward(self, *, hidden_states, timestep, **_kwargs):
        self.timesteps.append(timestep.detach().clone())
        self.inputs_require_grad.append(hidden_states.requires_grad)
        return hidden_states * self.scale


def _write_validation_item(root: Path, role: str, stem: str, size: tuple[int, int], marker: int) -> dict:
    image_dir = root / role / "images"
    cache_dir = root / role / "cache"
    image_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{stem}.png"
    caption = f"{role} {stem}"
    Image.new("RGB", size, (marker * 50, marker * 25, marker * 10)).save(image_path)
    (image_dir / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")
    image_id = hashlib.sha256(image_path.read_bytes()).hexdigest()
    metadata = {"architecture": "qwen_image", "format_version": "1.0.1", "source_image_sha256": image_id}
    latent_path = cache_dir / f"{stem}_{size[0]:04d}x{size[1]:04d}_qi.safetensors"
    text_path = cache_dir / f"{stem}_qi_te.safetensors"
    height, width = size[1] // 8, size[0] // 8
    save_file(
        {f"latents_1x{height}x{width}_float32": torch.full((2, 1, height, width), float(marker))},
        str(latent_path),
        metadata={**metadata, "width": str(size[0]), "height": str(size[1])},
    )
    save_file(
        {"varlen_vl_embed_float32": torch.ones((3, 4))},
        str(text_path),
        metadata={**metadata, "caption1": caption},
    )
    return {"image": image_path, "cache": cache_dir, "caption": caption, "latent": latent_path, "text": text_path}


@pytest.fixture
def validation_case(tmp_path):
    familiar = _write_validation_item(tmp_path, "val_familiar", "large", (64, 64), 1)
    small = _write_validation_item(tmp_path, "val_familiar", "small", (32, 64), 2)
    unfamiliar = _write_validation_item(tmp_path, "val_unfamiliar", "only", (64, 64), 3)
    config_path = tmp_path / "val-dataset.toml"
    config_path.write_text(
        "\n".join(
            [
                "[general]",
                "resolution = [64, 64]",
                "enable_bucket = true",
                "bucket_no_upscale = true",
                'caption_extension = ".txt"',
                "batch_size = 1",
                "num_repeats = 1",
                "",
                "[[datasets]]",
                'role = "val_familiar"',
                f'image_directory = {json.dumps(familiar["image"].parent.as_posix())}',
                f'cache_directory = {json.dumps(familiar["cache"].as_posix())}',
                "",
                "[[datasets]]",
                'role = "val_unfamiliar"',
                f'image_directory = {json.dumps(unfamiliar["image"].parent.as_posix())}',
                f'cache_directory = {json.dumps(unfamiliar["cache"].as_posix())}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        val_dataset_config=str(config_path),
        val_every_n_steps=2,
        val_seed_noise=42,
        val_level_noise_n=2,
        val_seed_noise_n=2,
        weighting_scheme="none",
        gradient_checkpointing=True,
        split_attn=True,
    )
    manifest = prepare_validation_inputs(args)
    return {"args": args, "manifest": manifest, "config": config_path, "familiar": familiar, "small": small, "unfamiliar": unfamiliar}


@pytest.fixture
def cpu_model_state():
    model = TinyQwenBoundary()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    return model, optimizer, scheduler


@pytest.fixture
def tracked_accelerator(tmp_path):
    accelerator = Accelerator(cpu=True, mixed_precision="no", log_with="tensorboard", project_dir=str(tmp_path / "logs"))
    accelerator.init_trackers("validation-test")
    yield accelerator
    accelerator.end_training()


def _event(trainer, case, accelerator, model, *, network=None, absolute_step=7):
    return trainer.evaluate_validation_event(
        case["args"],
        accelerator,
        model,
        torch.nn.Identity() if network is None else network,
        SimpleNamespace(sigmas=torch.tensor([0.2, 0.7]), timesteps=torch.tensor([200.0, 700.0])),
        torch.float32,
        torch.float32,
        absolute_step,
    )


def _controlled_losses(trainer, monkeypatch, *, overflow=False, nan_on_unfamiliar=False):
    """Replace expensive math only; event iteration and aggregation stay real."""
    seen = {}
    values = {1: ((1.0, 3.0), (5.0, 7.0)), 2: ((9.0, 11.0), (13.0, 15.0)), 3: ((21.0, 23.0), (25.0, 27.0))}

    def forward(args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs):
        marker = float(latents.flatten()[0])
        return DiTOutput(pred=torch.tensor([marker]), target=torch.zeros(1))

    def loss(args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step, *, exact_sigma=None):
        assert exact_sigma is not None
        t = float(timesteps.item()) / 1000
        level = 0 if math.isclose(t, 0.275, abs_tol=1e-6) else 1
        assert math.isclose(t, (0.275, 0.725)[level], abs_tol=1e-6)
        marker = int(output.pred.item())
        key = marker, level
        j = seen.get(key, 0) % 2
        seen[key] = seen.get(key, 0) + 1
        value = values[marker][level][j]
        if nan_on_unfamiliar and marker == 3:
            value = float("nan")
        if overflow:
            value = 1e308
        return torch.tensor(value, dtype=torch.float64), {}

    monkeypatch.setattr(trainer, "call_dit", forward)
    monkeypatch.setattr(trainer, "compute_loss", loss)
    return seen


def _event_scalars(root: Path) -> dict[str, list]:
    event_paths = list(root.rglob("events.out.tfevents*"))
    assert len(event_paths) == 1
    events = EventAccumulator(str(event_paths[0]))
    events.Reload()
    return {tag: events.Scalars(tag) for tag in events.Tags()["scalars"]}


def test_cpu_fixtures_have_real_two_role_cache_pairs_and_controlled_state(validation_case, cpu_model_state, tracked_accelerator):
    manifest = validation_case["manifest"]
    assert len(manifest.items_by_role["val_familiar"]) == 2
    assert len(manifest.items_by_role["val_unfamiliar"]) == 1
    assert {item.bucket_size for item in manifest.items} == {(64, 64), (32, 64)}
    assert {item.latent_cache_path.exists() and item.text_cache_path.exists() for item in manifest.items} == {True}
    model, optimizer, scheduler = cpu_model_state
    assert model.scale.device.type == "cpu"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert scheduler.last_epoch >= 0
    assert tracked_accelerator.device.type == "cpu"


def test_tensorboard_fixture_reads_existing_training_tag(tracked_accelerator, tmp_path):
    tracked_accelerator.log({"loss/current": 0.125}, step=7)
    tracked_accelerator.get_tracker("tensorboard").writer.flush()
    scalars = _event_scalars(tmp_path / "logs")
    assert set(scalars) == {"loss/current"}
    assert scalars["loss/current"][0].step == 7


def test_qwen_forward_uses_exact_t_and_flow_target_without_validation_input_grad(cpu_model_state):
    model, _, _ = cpu_model_state
    trainer = QwenImageNetworkTrainer()
    latents = torch.full((1, 2, 1, 8, 8), 0.25)
    epsilon = torch.full_like(latents, 0.75)
    t = 0.275
    noisy = (1 - t) * latents + t * epsilon
    args = SimpleNamespace(gradient_checkpointing=True, split_attn=True)
    with torch.no_grad():
        result = trainer.call_dit(
            args,
            Accelerator(cpu=True, mixed_precision="no"),
            model,
            latents,
            {"latents": latents, "vl_embed": [torch.ones((3, 4))]},
            epsilon,
            noisy,
            torch.tensor([1000 * t]),
            torch.float32,
        )
    torch.testing.assert_close(model.timesteps[-1], torch.tensor([t]))
    torch.testing.assert_close(result.target, epsilon - latents)
    torch.testing.assert_close(result.pred, noisy * model.scale)
    assert model.inputs_require_grad == [False]


@pytest.mark.parametrize("scheme", ["none", "sigma_sqrt", "cosmap"])
def test_shared_loss_default_keeps_scheduler_sigma(scheme):
    args = SimpleNamespace(weighting_scheme=scheme)
    output = DiTOutput(pred=torch.ones((1, 1, 1, 2, 2)), target=torch.zeros((1, 1, 1, 2, 2)))
    scheduler = SimpleNamespace(sigmas=torch.tensor([0.2, 0.7]), timesteps=torch.tensor([200.0, 700.0]))
    actual, metrics = NetworkTrainer().compute_loss(
        args, output, torch.tensor([275.0]), scheduler, torch.float32, torch.float32, 0
    )
    sigma = 0.2
    expected = {
        "none": 1.0,
        "sigma_sqrt": sigma**-2,
        "cosmap": 2 / (math.pi * (1 - 2 * sigma + 2 * sigma**2)),
    }[scheme]
    assert actual.item() == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert metrics == {}


@pytest.mark.parametrize("scheme", ["none", "sigma_sqrt", "cosmap"])
def test_shared_loss_uses_exact_off_schedule_sigma_and_preserves_default(scheme):
    trainer = NetworkTrainer()
    args = SimpleNamespace(weighting_scheme=scheme)
    pred = torch.tensor([[[[[1.0, 2.0], [3.0, 4.0]]]]])
    target = torch.tensor([[[[[0.5, 1.5], [1.0, 2.0]]]]])
    output = DiTOutput(pred=pred, target=target)
    scheduler = SimpleNamespace(sigmas=torch.tensor([0.2, 0.7]), timesteps=torch.tensor([200.0, 700.0]))
    timestep = torch.tensor([275.0])
    sigma = 0.275
    factor = {
        "none": 1.0,
        "sigma_sqrt": sigma**-2,
        "cosmap": 2 / (math.pi * (1 - 2 * sigma + 2 * sigma**2)),
    }[scheme]
    expected = ((pred - target).square() * factor).mean()
    actual, metrics = trainer.compute_loss(
        args, output, timestep, scheduler, torch.float32, torch.float32, 0, exact_sigma=torch.tensor([sigma])
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert metrics == {}

    # With no override, existing training still selects the schedule's nearest sigma.
    legacy, _ = trainer.compute_loss(args, output, timestep, scheduler, torch.float32, torch.float32, 0)
    old_sigma = 0.2
    old_factor = {
        "none": 1.0,
        "sigma_sqrt": old_sigma**-2,
        "cosmap": 2 / (math.pi * (1 - 2 * old_sigma + 2 * old_sigma**2)),
    }[scheme]
    torch.testing.assert_close(legacy, ((pred - target).square() * old_factor).mean(), rtol=1e-6, atol=1e-6)


def test_event_averages_checks_per_image_then_images_not_pixels(validation_case, cpu_model_state, tracked_accelerator, monkeypatch):
    trainer = QwenImageNetworkTrainer()
    trainer.validation_manifest = validation_case["manifest"]
    seen = _controlled_losses(trainer, monkeypatch)
    logged = []
    monkeypatch.setattr(tracked_accelerator, "log", lambda values, step: logged.append((dict(values), step)))
    _event(trainer, validation_case, tracked_accelerator, cpu_model_state[0])
    assert seen == {(marker, level): 2 for marker in (1, 2, 3) for level in (0, 1)}
    assert len(logged) == 1
    values, step = logged[0]
    assert step == 7 and set(values) == TAGS
    expected = {
        "train_eval_loss_mean": 8.0,
        "train_eval_loss_low_noise": 6.0,
        "train_eval_loss_high_noise": 10.0,
        "val_loss_mean": 24.0,
        "val_loss_low_noise": 22.0,
        "val_loss_high_noise": 26.0,
    }
    for tag, value in expected.items():
        assert values[tag] == pytest.approx(value, abs=1e-6)
    for prefix in ("train_eval_loss", "val_loss"):
        assert values[f"{prefix}_mean"] == pytest.approx(
            (values[f"{prefix}_low_noise"] + values[f"{prefix}_high_noise"]) / 2, abs=1e-6
        )


def test_event_uses_existing_qwen_forward_and_exact_weighted_loss(validation_case, cpu_model_state, tracked_accelerator):
    case = validation_case
    case["args"].weighting_scheme = "cosmap"
    trainer = QwenImageNetworkTrainer()
    trainer.validation_manifest = case["manifest"]
    model = cpu_model_state[0]
    actual = _event(trainer, case, tracked_accelerator, model)

    reference = {}
    for role, prefix in (("val_familiar", "train_eval_loss"), ("val_unfamiliar", "val_loss")):
        per_image = {"all": [], "low": [], "high": []}
        for item in case["manifest"].items_by_role[role]:
            latent, _ = read_validation_cache_pair(item)
            checks = iter_noise_checks(
                item,
                latent,
                val_seed_noise=case["args"].val_seed_noise,
                val_level_noise_n=case["args"].val_level_noise_n,
                val_seed_noise_n=case["args"].val_seed_noise_n,
            )
            values = {"all": [], "low": [], "high": []}
            for check in checks:
                pred = 2 * check.noisy_latent
                target = check.epsilon - check.latent
                weight = 2 / (math.pi * (1 - 2 * check.t + 2 * check.t**2))
                scalar = float(((pred - target).square() * weight).mean())
                values["all"].append(scalar)
                values["low" if check.t < 0.5 else "high"].append(scalar)
            for group in values:
                per_image[group].append(sum(values[group]) / len(values[group]))
        reference[prefix + "_mean"] = sum(per_image["all"]) / len(per_image["all"])
        reference[prefix + "_low_noise"] = sum(per_image["low"]) / len(per_image["low"])
        reference[prefix + "_high_noise"] = sum(per_image["high"]) / len(per_image["high"])

    assert set(actual) == TAGS
    for tag, value in reference.items():
        assert actual[tag] == pytest.approx(value, rel=1e-6, abs=1e-6)
    assert len(model.timesteps) == len(case["manifest"].items) * 4
    assert {round(float(t.item()), 3) for t in model.timesteps} == {0.275, 0.725}
    assert model.inputs_require_grad == [False] * len(model.timesteps)


def test_duplicate_declared_occurrence_keeps_equal_image_weight(validation_case, cpu_model_state, tracked_accelerator, monkeypatch, tmp_path):
    config_path = validation_case["config"]
    familiar = validation_case["familiar"]
    small = validation_case["small"]
    jsonl = tmp_path / "familiar.jsonl"
    entries = [
        {"image_path": str(source["image"]), "caption": source["caption"]}
        for source in (familiar, familiar, small)
    ]
    jsonl.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            f'image_directory = {json.dumps(familiar["image"].parent.as_posix())}',
            f'image_jsonl_file = {json.dumps(jsonl.as_posix())}',
        ),
        encoding="utf-8",
    )
    validation_case["manifest"] = prepare_validation_inputs(validation_case["args"])
    assert len(validation_case["manifest"].items_by_role["val_familiar"]) == 3
    trainer = QwenImageNetworkTrainer()
    trainer.validation_manifest = validation_case["manifest"]
    _controlled_losses(trainer, monkeypatch)
    logged = []
    monkeypatch.setattr(tracked_accelerator, "log", lambda values, step: logged.append((dict(values), step)))
    _event(trainer, validation_case, tracked_accelerator, cpu_model_state[0])
    values, _ = logged[0]
    assert values["train_eval_loss_mean"] == pytest.approx((4 + 4 + 12) / 3, abs=1e-6)
    assert values["train_eval_loss_low_noise"] == pytest.approx((2 + 2 + 10) / 3, abs=1e-6)
    assert values["train_eval_loss_high_noise"] == pytest.approx((6 + 6 + 14) / 3, abs=1e-6)


def test_complete_event_writes_exact_six_tags_to_real_tensorboard(validation_case, cpu_model_state, tracked_accelerator, monkeypatch, tmp_path):
    trainer = QwenImageNetworkTrainer()
    trainer.validation_manifest = validation_case["manifest"]
    _controlled_losses(trainer, monkeypatch)
    tracked_accelerator.log({"loss/current": 0.125}, step=7)
    _event(trainer, validation_case, tracked_accelerator, cpu_model_state[0])
    tracked_accelerator.get_tracker("tensorboard").writer.flush()
    scalars = _event_scalars(tmp_path / "logs")
    assert set(scalars) == TAGS | {"loss/current"}
    assert all(len(scalars[tag]) == 1 and scalars[tag][0].step == 7 for tag in TAGS)
    assert len(scalars["loss/current"]) == 1 and scalars["loss/current"][0].step == 7


@pytest.mark.parametrize("failure", ["late_read", "nonfinite_check", "nonfinite_aggregate"])
def test_failed_event_never_publishes_partial_tags(
    validation_case, cpu_model_state, tracked_accelerator, monkeypatch, failure
):
    trainer = QwenImageNetworkTrainer()
    trainer.validation_manifest = validation_case["manifest"]
    seen = _controlled_losses(
        trainer,
        monkeypatch,
        overflow=failure == "nonfinite_aggregate",
        nan_on_unfamiliar=failure == "nonfinite_check",
    )
    if failure == "late_read":
        validation_case["unfamiliar"]["text"].write_bytes(b"changed after preflight")
    logged = []
    monkeypatch.setattr(tracked_accelerator, "log", lambda values, step: logged.append((dict(values), step)))
    with pytest.raises(ValueError) as failure_info:
        _event(trainer, validation_case, tracked_accelerator, cpu_model_state[0])
    message = str(failure_info.value).lower()
    if failure == "late_read":
        assert "val_unfamiliar" in message and ("changed" in message or "restore" in message)
    elif failure == "nonfinite_check":
        assert "val_unfamiliar" in message and ("nonfinite" in message or "non-finite" in message)
    else:
        assert "val_familiar" in message and ("nonfinite" in message or "overflow" in message)
    assert any(marker in (1, 2) for marker, _level in seen)
    assert logged == []


class RandomTinyBoundary(TinyQwenBoundary):
    def __init__(self, network, fail_on_call=None):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
        self._network_getter = lambda: network
        self.fail_on_call = fail_on_call
        self.observed_modes = []
        self.observed_grad_modes = []

    def forward(self, *, hidden_states, timestep, **kwargs):
        network = self._network_getter()
        self.observed_modes.append(
            (self.training, self.dropout.training, network.training, network[0].training, network[1].training)
        )
        self.observed_grad_modes.append(torch.is_grad_enabled())
        random.random()
        np.random.random()
        torch.rand(())
        if len(self.observed_modes) == self.fail_on_call:
            raise RuntimeError("injected CPU forward failure")
        return super().forward(hidden_states=self.dropout(hidden_states), timestep=timestep, **kwargs)


def _rng_snapshot():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
        [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
    )


def _assert_rng_equal(actual, expected):
    assert actual[0] == expected[0]
    assert actual[1][0] == expected[1][0]
    np.testing.assert_array_equal(actual[1][1], expected[1][1])
    assert actual[1][2:] == expected[1][2:]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    if expected[3] is not None:
        assert len(actual[3]) == len(expected[3])
        for left, right in zip(actual[3], expected[3]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def _restore_rng(snapshot):
    random.setstate(snapshot[0])
    np.random.set_state(snapshot[1])
    torch.set_rng_state(snapshot[2])
    if snapshot[3] is not None:
        torch.cuda.set_rng_state_all(snapshot[3])


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


def _next_controlled_update(model, network, optimizer, scheduler):
    drive = random.random() + np.random.random() + float(torch.rand(()))
    loss = (model.scale + network[1].weight.sum()) * drive
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return drive, model.scale.detach().clone(), network[1].weight.detach().clone(), scheduler.get_last_lr()


@pytest.mark.parametrize("fail_on_call", [None, 2], ids=["success", "exception"])
def test_event_restores_mixed_modes_rng_and_training_state_before_next_update(
    validation_case, tracked_accelerator, fail_on_call
):
    network = torch.nn.Sequential(torch.nn.Dropout(0.25), torch.nn.Linear(1, 1))
    model = RandomTinyBoundary(network, fail_on_call=fail_on_call)
    model.train()
    model.dropout.eval()
    network.train()
    network[0].eval()
    parameters = list(model.parameters()) + list(network.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / 100)
    (model.scale.square() + network[1].weight.square().sum()).backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    model.scale.grad = torch.tensor(0.125)
    network[1].weight.grad = torch.full_like(network[1].weight, 0.25)
    modes_before = {id(module): module.training for root in (model, network) for module in root.modules()}
    weights_before = [parameter.detach().clone() for parameter in parameters]
    grads_before = [None if parameter.grad is None else parameter.grad.clone() for parameter in parameters]
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    original_rng = _rng_snapshot()
    try:
        random.seed(917)
        np.random.seed(917)
        torch.manual_seed(917)
        before = _rng_snapshot()
        reference = _next_controlled_update(model, network, optimizer, scheduler)
        with torch.no_grad():
            for parameter, expected_weight, expected_grad in zip(parameters, weights_before, grads_before):
                parameter.copy_(expected_weight)
                parameter.grad = None if expected_grad is None else expected_grad.clone()
        optimizer.load_state_dict(copy.deepcopy(optimizer_before))
        scheduler.load_state_dict(copy.deepcopy(scheduler_before))
        _restore_rng(before)

        trainer = QwenImageNetworkTrainer()
        trainer.validation_manifest = validation_case["manifest"]
        if fail_on_call is None:
            _event(trainer, validation_case, tracked_accelerator, model, network=network)
        else:
            with pytest.raises(RuntimeError, match="injected CPU forward failure"):
                _event(trainer, validation_case, tracked_accelerator, model, network=network)

        assert model.observed_modes
        assert all(flags == (False, False, False, False, False) for flags in model.observed_modes)
        assert not any(model.observed_grad_modes)
        assert not any(model.inputs_require_grad)
        assert {id(module): module.training for root in (model, network) for module in root.modules()} == modes_before
        _assert_rng_equal(_rng_snapshot(), before)
        for parameter, expected_weight, expected_grad in zip(parameters, weights_before, grads_before):
            torch.testing.assert_close(parameter, expected_weight, rtol=0, atol=0)
            if expected_grad is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(parameter.grad, expected_grad, rtol=0, atol=0)
        _assert_nested_equal(optimizer.state_dict(), optimizer_before)
        _assert_nested_equal(scheduler.state_dict(), scheduler_before)
        actual = _next_controlled_update(model, network, optimizer, scheduler)
        assert actual[0] == pytest.approx(reference[0], rel=0, abs=0)
        torch.testing.assert_close(actual[1], reference[1], rtol=0, atol=0)
        torch.testing.assert_close(actual[2], reference[2], rtol=0, atol=0)
        assert actual[3] == reference[3]
    finally:
        _restore_rng(original_rng)


@pytest.mark.parametrize(
    "budget,expected",
    [(5, [0, 2, 4, 5]), (4, [0, 2, 4])],
    ids=["final-outside-interval", "final-coincides-with-interval"],
)
def test_validation_boundary_runs_start_periodic_and_final_once(validation_case, cpu_model_state, monkeypatch, budget, expected):
    trainer = QwenImageNetworkTrainer()
    trainer.validation_manifest = validation_case["manifest"]
    observed = []
    monkeypatch.setattr(trainer, "evaluate_validation_event", lambda *args: observed.append(args[-1]))
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    scheduler = SimpleNamespace(sigmas=torch.tensor([0.2, 0.7]), timesteps=torch.tensor([200.0, 700.0]))
    for step in range(budget + 1):
        trainer._run_validation_boundary(
            validation_case["args"],
            accelerator,
            cpu_model_state[0],
            torch.nn.Identity(),
            scheduler,
            torch.float32,
            torch.float32,
            absolute_step=step,
            completed=step > 0,
            final=step == budget,
        )
    assert observed == expected


def test_example_validation_grid_through_1600_has_one_final_event(validation_case, cpu_model_state, monkeypatch):
    trainer = QwenImageNetworkTrainer()
    trainer.validation_manifest = validation_case["manifest"]
    validation_case["args"].val_every_n_steps = 200
    observed = []
    monkeypatch.setattr(trainer, "evaluate_validation_event", lambda *args: observed.append(args[-1]))
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    scheduler = SimpleNamespace(sigmas=torch.tensor([0.2, 0.7]), timesteps=torch.tensor([200.0, 700.0]))
    for step in range(1601):
        trainer._run_validation_boundary(
            validation_case["args"], accelerator, cpu_model_state[0], torch.nn.Identity(), scheduler,
            torch.float32, torch.float32, absolute_step=step, completed=step > 0, final=step == 1600,
        )
    assert observed == list(range(0, 1601, 200))


def test_enabled_loop_counts_only_completed_updates_and_logs_after_them(validation_case, tmp_path, monkeypatch):
    trace = []

    class LoopNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def on_epoch_start(self, _model):
            pass

        def on_step_start(self):
            pass

        def save_weights(self, path, _dtype, _metadata):
            Path(path).write_bytes(b"controlled CPU checkpoint")

    class LoopAccelerator:
        is_main_process = True
        is_local_main_process = True
        device = torch.device("cpu")

        def __init__(self):
            self.trackers = [object()]
            self.microbatches = 0
            self.sync_gradients = False
            self.optimizer_step_was_skipped = False
            self.logs = []

        @contextmanager
        def accumulate(self, _model):
            self.microbatches += 1
            self.sync_gradients = self.microbatches % 2 == 0
            self.optimizer_step_was_skipped = self.microbatches == 4
            yield

        def backward(self, loss):
            loss.backward()

        def unwrap_model(self, model):
            return model

        def init_trackers(self, *_args, **_kwargs):
            pass

        def print(self, *_args):
            pass

        def log(self, values, step):
            self.logs.append((step, dict(values)))

        def wait_for_everyone(self):
            pass

        def end_training(self):
            pass

    network = LoopNetwork()
    accelerator = LoopAccelerator()
    real_optimizer = torch.optim.SGD(network.parameters(), lr=0.1)

    class LoopOptimizer:
        def step(self):
            if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                real_optimizer.step()
                trace.append(("update", sum(kind == "update" for kind, *_ in trace) + 1))

        def zero_grad(self, **kwargs):
            if accelerator.sync_gradients:
                real_optimizer.zero_grad(**kwargs)

    class LoopScheduler:
        def step(self):
            pass

    class OptionalArgs(SimpleNamespace):
        def __getattr__(self, _name):
            return None

    args = OptionalArgs(
        **vars(validation_case["args"]),
        gradient_accumulation_steps=2,
        max_train_steps=3,
        output_name="controlled",
        output_dir=str(tmp_path),
        learning_rate=0.1,
        discrete_flow_shift=1.0,
        no_metadata=True,
        max_grad_norm=0.0,
        sample_at_first=False,
        save_state=False,
        save_state_on_train_end=False,
    )
    dataset = SimpleNamespace(batch_size=1, get_metadata=lambda: {})
    dataset_group = SimpleNamespace(datasets=[dataset], num_train_items=8)
    dataloader = [{"latents": torch.ones(1)} for _ in range(8)]
    trainer = QwenImageNetworkTrainer()
    trainer.validation_manifest = validation_case["manifest"]
    monkeypatch.setattr(trainer_base, "clean_memory_on_device", lambda _device: None)
    monkeypatch.setattr(trainer_base.sai_model_spec, "build_metadata", lambda *_a, **_kw: {})
    def process_batch(*_args, **_kwargs):
        trace.append(("batch", accelerator.microbatches))
        return network.weight.square(), {}

    monkeypatch.setattr(trainer, "process_batch", process_batch)
    monkeypatch.setattr(trainer, "generate_step_logs", lambda _a, loss, *_rest: {"loss/current": loss})
    monkeypatch.setattr(trainer, "evaluate_validation_event", lambda *_a: trace.append(("event", _a[-1])))
    trainer._run_training_loop(
        args,
        accelerator,
        "cpu-session",
        0.0,
        dataset_group,
        dataloader,
        SimpleNamespace(value=0),
        TinyQwenBoundary(),
        network,
        network,
        LoopOptimizer(),
        "SGD",
        [],
        lambda: None,
        lambda: None,
        LoopScheduler(),
        [],
        None,
        None,
        torch.float32,
        torch.float32,
    )
    assert [value for kind, value in trace if kind == "update"] == [1, 2, 3]
    assert [value for kind, value in trace if kind == "event"] == [0, 2, 3]
    assert trace.index(("event", 0)) < trace.index(("batch", 1))
    assert trace.index(("update", 2)) < trace.index(("event", 2)) < trace.index(("batch", 7))
    assert trace.index(("update", 3)) < trace.index(("event", 3))
    assert [(step, values["loss/current"]) for step, values in accelerator.logs if "loss/current" in values] == [
        (1, pytest.approx(1.0)),
        (2, pytest.approx(0.36)),
        (3, pytest.approx(0.1296)),
    ]


def _two_rank_boundary_worker(rank, port, config_path, report_dir, inject_failure, mismatch_no_due=False):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE="2",
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        ACCELERATE_USE_CPU="true",
    )
    result = {"rank": rank, "items": [], "logs": [], "step": 3 + rank if mismatch_no_due else 2, "error": None}
    try:
        dist.init_process_group("gloo", timeout=timedelta(seconds=15))
        accelerator = Accelerator(cpu=True, mixed_precision="no")
        args = SimpleNamespace(
            val_dataset_config=config_path,
            val_every_n_steps=2,
            val_seed_noise=42,
            val_level_noise_n=2,
            val_seed_noise_n=2,
            weighting_scheme="none",
            gradient_checkpointing=True,
            split_attn=True,
        )
        trainer = QwenImageNetworkTrainer()
        trainer.validation_manifest = prepare_validation_inputs(args)
        accelerator.log = lambda values, step: result["logs"].append((step, sorted(values)))

        def event(*event_args):
            for item in trainer.validation_manifest.items:
                result["items"].append((item.role, item.image_id))
                if inject_failure and len(result["items"]) == 2:
                    raise RuntimeError("injected main-rank input failure")
            accelerator.log({tag: 1.0 for tag in TAGS}, step=event_args[-1])

        trainer.evaluate_validation_event = event
        try:
            if mismatch_no_due:
                args.val_every_n_steps = 10
                trainer._assert_validation_update_agreement(
                    accelerator,
                    sync_gradients=False,
                    completed_update=False,
                    absolute_step=result["step"],
                )
            else:
                trainer._run_validation_boundary(
                    args,
                    accelerator,
                    TinyQwenBoundary(),
                    torch.nn.Identity(),
                    SimpleNamespace(sigmas=torch.tensor([0.2, 0.7]), timesteps=torch.tensor([200.0, 700.0])),
                    torch.float32,
                    torch.float32,
                    absolute_step=2,
                    completed=True,
                    final=False,
                )
        except Exception as exc:
            result["error"] = str(exc)
    except Exception as exc:
        result["setup_error"] = repr(exc)
    finally:
        Path(report_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("inject_failure", [False, True], ids=["success", "main-rank-error"])
def test_two_rank_cpu_validation_boundary_once_globally_and_releases_peers(validation_case, tmp_path, inject_failure):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_two_rank_boundary_worker,
            args=(rank, port, str(validation_case["config"]), str(tmp_path), inject_failure),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=25)
        assert all(not process.is_alive() for process in processes), "two-rank validation boundary deadlocked"
        assert all(process.exitcode == 0 for process in processes)
        results = [json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
        assert all("setup_error" not in result for result in results), results
        assert [result["step"] for result in results] == [2, 2]
        if inject_failure:
            assert all("injected main-rank input failure" in result["error"] for result in results)
            assert all(result["logs"] == [] for result in results)
            assert results[1]["items"] == []
        else:
            assert [tuple(item) for item in results[0]["items"]] == [
                (item.role, item.image_id) for item in validation_case["manifest"].items
            ]
            assert results[1]["items"] == []
            assert results[0]["logs"] == [[2, sorted(TAGS)]]
            assert results[1]["logs"] == []
            assert [result["error"] for result in results] == [None, None]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def test_two_rank_cpu_rejects_different_absolute_steps_without_due_event(validation_case, tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_two_rank_boundary_worker,
            args=(rank, port, str(validation_case["config"]), str(tmp_path), False, True),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=25)
        assert all(not process.is_alive() for process in processes), "two-rank optimizer boundary deadlocked"
        assert all(process.exitcode == 0 for process in processes)
        results = [json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
        assert all("setup_error" not in result for result in results), results
        assert [result["step"] for result in results] == [3, 4]
        assert all("boundary differs across ranks" in result["error"] for result in results)
        assert all(result["items"] == [] and result["logs"] == [] for result in results)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
