"""CPU state and resume checks for the Qwen validation timeline."""

import copy
import json
import os
import random
import socket
from types import SimpleNamespace

from accelerate import Accelerator
import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from musubi_tuner.qwen_image_train_network import QwenImageNetworkTrainer
from musubi_tuner.training.validation_state import build_validation_state, load_validation_resume_step
from musubi_tuner.utils import train_utils


FINGERPRINT = "a" * 64


def _args(path):
    return SimpleNamespace(
        output_dir=str(path),
        output_name="qwen",
        save_state_to_huggingface=False,
        save_last_n_steps_state=None,
        save_last_n_steps=None,
        save_every_n_steps=1,
        save_last_n_epochs_state=None,
        save_last_n_epochs=None,
        save_every_n_epochs=1,
        val_every_n_steps=2,
        val_seed_noise=42,
        val_level_noise_n=10,
        val_seed_noise_n=1,
        resume=None,
        resume_from_huggingface=False,
    )


def _rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _assert_rng_equal(expected):
    assert random.getstate() == expected[0]
    numpy_state = np.random.get_state()
    assert numpy_state[0] == expected[1][0]
    np.testing.assert_array_equal(numpy_state[1], expected[1][1])
    assert numpy_state[2:] == expected[1][2:]
    torch.testing.assert_close(torch.get_rng_state(), expected[2], rtol=0, atol=0)


def _tiny_state(accelerator):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    loss = model(torch.ones(1, 1)).square().sum()
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return model, optimizer, scheduler


def test_one_rank_accelerate_state_has_valid_sidecar_and_restores_training_state(tmp_path):
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    model, optimizer, scheduler = _tiny_state(accelerator)
    args = _args(tmp_path)
    manifest = SimpleNamespace(fingerprint=FINGERPRINT)
    random.seed(105)
    np.random.seed(106)
    torch.manual_seed(107)
    saved = (
        copy.deepcopy(accelerator.unwrap_model(model).state_dict()),
        copy.deepcopy(optimizer.state_dict()),
        copy.deepcopy(scheduler.state_dict()),
        _rng_snapshot(),
    )
    payload = build_validation_state(args, manifest, 5)
    train_utils.save_and_remove_state_stepwise(args, accelerator, 5, validation_state=payload)
    state_dir = tmp_path / "qwen-step00000005-state"
    assert json.loads((state_dir / "val_loss_state.json").read_text(encoding="utf-8")) == payload
    assert (state_dir / "random_states_0.pkl").is_file()
    assert load_validation_resume_step(accelerator, state_dir, args, manifest) == 5

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    optimizer.param_groups[0]["lr"] = 0.8
    scheduler.step()
    random.random(), np.random.rand(), torch.rand(1)
    trainer = QwenImageNetworkTrainer()
    trainer.validation_manifest = manifest
    args.val_dataset_config = "fixture.toml"
    args.resume = str(state_dir)
    trainer._register_hooks_and_resume(args, accelerator, model)
    assert trainer.validation_resume_step == 5
    for key, value in saved[0].items():
        torch.testing.assert_close(accelerator.unwrap_model(model).state_dict()[key], value, rtol=0, atol=0)
    assert optimizer.state_dict() == saved[1]
    assert scheduler.state_dict() == saved[2]
    _assert_rng_equal(saved[3])
    train_utils.save_and_remove_state_on_epoch_end(args, accelerator, 2, validation_state=payload)
    train_utils.save_state_on_train_end(args, accelerator, validation_state=payload)
    for state_name in ("qwen-000002-state", "qwen-state"):
        extra_state = tmp_path / state_name
        assert json.loads((extra_state / "val_loss_state.json").read_text(encoding="utf-8")) == payload
        assert (extra_state / "random_states_0.pkl").is_file()
    accelerator.free_memory()


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "malformed",
        "version",
        "model",
        "negative_step",
        "bool_step",
        "bad_fingerprint",
        "changed_fingerprint",
        "changed_interval",
        "bool_control",
    ],
)
def test_enabled_resume_rejects_missing_or_untrusted_metadata(tmp_path, change):
    args = _args(tmp_path)
    manifest = SimpleNamespace(fingerprint=FINGERPRINT)
    payload = build_validation_state(args, manifest, 5)
    path = tmp_path / "val_loss_state.json"
    if change == "malformed":
        path.write_text("{invalid", encoding="utf-8")
    elif change != "missing":
        if change == "version":
            payload["version"] = "unknown"
        elif change == "model":
            payload["model_version"] = "edit"
        elif change == "negative_step":
            payload["absolute_completed_step"] = -1
        elif change == "bool_step":
            payload["absolute_completed_step"] = True
        elif change == "bad_fingerprint":
            payload["validation_fingerprint"] = "bad"
        elif change == "changed_fingerprint":
            payload["validation_fingerprint"] = "b" * 64
        elif change == "changed_interval":
            payload["controls"]["val_every_n_steps"] = 3
        elif change == "bool_control":
            payload["controls"]["val_seed_noise"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="val_loss_state|validation|resume|step|fingerprint|controls"):
        load_validation_resume_step(Accelerator(cpu=True, mixed_precision="no"), tmp_path, args, manifest)


def test_legacy_resume_without_validation_sidecar_and_weights_only_new_timeline(tmp_path):
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    model, _, _ = _tiny_state(accelerator)
    legacy_dir = tmp_path / "legacy-state"
    accelerator.save_state(str(legacy_dir))
    assert not (legacy_dir / "val_loss_state.json").exists()

    args = _args(tmp_path)
    args.resume = str(legacy_dir)
    args.val_dataset_config = None
    trainer = QwenImageNetworkTrainer()
    trainer._register_hooks_and_resume(args, accelerator, model)
    assert trainer.validation_resume_step == 0

    weights_only = QwenImageNetworkTrainer()
    weights_only.validation_manifest = SimpleNamespace(fingerprint=FINGERPRINT)
    args.resume = None
    args.val_dataset_config = "fixture.toml"
    weights_only._register_hooks_and_resume(args, accelerator, model)
    assert weights_only.validation_resume_step == 0
    accelerator.free_memory()


def test_resumed_loop_uses_absolute_events_logs_and_state_names_with_new_run_budget(tmp_path, monkeypatch):
    from musubi_tuner.qwen_image_train_network import qwen_image_setup_parser
    from musubi_tuner.training.parser_common import setup_parser_common

    class TinyNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25))
            self.saved_steps = []

        def on_epoch_start(self, _transformer):
            self.train()

        def on_step_start(self):
            pass

        def get_trainable_params(self):
            return self.parameters()

        def save_weights(self, path, _dtype, metadata):
            self.saved_steps.append((os.path.basename(path), metadata["ss_steps"]))
            with open(path, "wb") as target:
                target.write(b"tiny adapter")

    accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=1)
    network = TinyNetwork()
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 4))
    network, optimizer, scheduler = accelerator.prepare(network, optimizer, scheduler)
    for _ in range(5):
        accelerator.backward(network.weight.square())
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    assert scheduler.scheduler.last_epoch == 5
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)

    args = qwen_image_setup_parser(setup_parser_common()).parse_args([])
    for key, value in vars(_args(tmp_path)).items():
        setattr(args, key, value)
    args.output_name = "timeline"
    args.max_train_steps = 3
    args.gradient_accumulation_steps = 1
    args.save_every_n_steps = 2
    args.sample_every_n_steps = 5
    args.save_state = True
    args.save_state_on_train_end = False
    args.save_every_n_epochs = None
    args.weighting_scheme = "none"
    args.network_args = None
    args.dit = None
    args.vae = None
    args.full_fp16 = False
    args.full_bf16 = False
    args.log_tracker_name = None
    args.log_tracker_config = None
    args.wandb_run_name = None
    manifest = SimpleNamespace(fingerprint=FINGERPRINT)
    state = build_validation_state(args, manifest, 5)
    train_utils.save_state_on_train_end(args, accelerator, validation_state=state)
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_scheduler = copy.deepcopy(scheduler.state_dict())
    with torch.no_grad():
        network.weight.add_(5)
    optimizer.param_groups[0]["lr"] = 0.5
    scheduler.step()

    trainer = QwenImageNetworkTrainer()
    trainer.validation_manifest = manifest
    args.val_dataset_config = "fixture.toml"
    args.resume = str(tmp_path / "timeline-state")
    trainer._register_hooks_and_resume(args, accelerator, network)
    assert trainer.validation_resume_step == 5
    assert optimizer.state_dict() == saved_optimizer
    assert scheduler.state_dict() == saved_scheduler
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)

    events = []
    samples = []
    training_hook_steps = []
    logs = []
    monkeypatch.setattr(
        trainer,
        "evaluate_validation_event",
        lambda *_args: events.append(_args[-1]),
    )

    def process_batch(_args, _accelerator, _transformer, current_network, _batch, latents, noise,
                      _noise_scheduler, _dit_dtype, _network_dtype, _sample_resources, global_step):
        training_hook_steps.append(global_step)
        return (current_network.weight * latents - noise).square().mean(), {}

    monkeypatch.setattr(trainer, "process_batch", process_batch)
    monkeypatch.setattr(
        trainer, "sample_images",
        lambda _accelerator, _args, epoch, step, *_rest: samples.append((epoch, step)),
    )
    monkeypatch.setattr(accelerator, "init_trackers", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(accelerator, "log", lambda values, step: logs.append((step, dict(values))))
    monkeypatch.setattr(accelerator, "end_training", lambda: None)
    accelerator.trackers = [object()]
    real_optimizer_step = optimizer.step
    attempts = 0

    def step_with_one_skip():
        nonlocal attempts
        attempts += 1
        optimizer._is_overflow = attempts == 1
        if attempts != 1:
            real_optimizer_step()

    monkeypatch.setattr(optimizer, "step", step_with_one_skip)
    latents = torch.ones(1, 1, 1, 1, 1)
    batches = [{"latents": latents}]
    group = SimpleNamespace(num_train_items=1, datasets=[SimpleNamespace(batch_size=1, get_metadata=lambda: {})])
    trainer._run_training_loop(
        args, accelerator, 1, 0.0, group, batches, SimpleNamespace(value=0), torch.nn.Identity(), network,
        network, optimizer, "AdamW", "", lambda: None, lambda: None, scheduler, None,
        None, None, torch.float32, torch.float32,
    )

    assert events == [5, 6, 8]
    assert samples == [(0, 5)]
    assert attempts == 4
    assert training_hook_steps == [0, 0, 1, 2]
    assert logs[0] == (5, {})
    assert [step for step, values in logs if "loss/current" in values] == [6, 7, 8]
    assert all("loss/current" not in values and "loss/epoch" not in values for step, values in logs if step == 5)
    assert scheduler.scheduler.last_epoch == 8
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert network.saved_steps == [
        ("timeline-step00000006.safetensors", "6"),
        ("timeline-step00000008.safetensors", "8"),
        ("timeline.safetensors", "8"),
    ]
    for step in (6, 8):
        saved_dir = tmp_path / f"timeline-step{step:08d}-state"
        assert json.loads((saved_dir / "val_loss_state.json").read_text(encoding="utf-8"))["absolute_completed_step"] == step
    assert json.loads((tmp_path / "timeline-state" / "val_loss_state.json").read_text(encoding="utf-8"))[
        "absolute_completed_step"
    ] == 8
    accelerator.free_memory()


def _two_rank_state_worker(rank, port, output_dir):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        ACCELERATE_USE_CPU="true",
    )
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    model, optimizer, scheduler = _tiny_state(accelerator)
    args = _args(output_dir)
    manifest = SimpleNamespace(fingerprint=FINGERPRINT)
    random.seed(201 + rank)
    np.random.seed(301 + rank)
    torch.manual_seed(401 + rank)
    rng = _rng_snapshot()
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    payload = build_validation_state(args, manifest, 5)
    train_utils.save_and_remove_state_stepwise(args, accelerator, 5, validation_state=payload)
    state_dir = os.path.join(output_dir, "qwen-step00000005-state")
    assert load_validation_resume_step(accelerator, state_dir, args, manifest) == 5
    random.random(), np.random.rand(), torch.rand(1)
    optimizer.param_groups[0]["lr"] = 0.9
    scheduler.step()
    trainer = QwenImageNetworkTrainer()
    trainer.validation_manifest = manifest
    args.val_dataset_config = "fixture.toml"
    args.resume = state_dir
    trainer._register_hooks_and_resume(args, accelerator, model)
    assert trainer.validation_resume_step == 5
    _assert_rng_equal(rng)
    assert optimizer.state_dict() == optimizer_before
    assert scheduler.state_dict() == scheduler_before
    local_manifest = manifest if rank == 0 else SimpleNamespace(fingerprint="b" * 64)
    with pytest.raises(ValueError, match="validation fingerprint"):
        load_validation_resume_step(accelerator, state_dir, args, local_manifest)
    from musubi_tuner.training import validation_state as state_module

    original_write = state_module.write_validation_state
    if rank == 0:
        def fail_main_write(*_args, **_kwargs):
            raise OSError("injected sidecar write failure")

        state_module.write_validation_state = fail_main_write
    try:
        with pytest.raises(RuntimeError, match="injected sidecar write failure"):
            train_utils.save_state_on_train_end(args, accelerator, validation_state=payload)
    finally:
        state_module.write_validation_state = original_write
    with open(os.path.join(output_dir, f"rank-{rank}.json"), "w", encoding="utf-8") as target:
        json.dump({"rank": rank, "step": 5}, target)
    accelerator.wait_for_everyone()
    accelerator.free_memory()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def test_two_rank_accelerate_state_restores_each_rank_rng_and_shared_sidecar(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_two_rank_state_worker, args=(port, str(tmp_path)), nprocs=2, join=True)
    state_dir = tmp_path / "qwen-step00000005-state"
    assert (state_dir / "val_loss_state.json").is_file()
    assert sorted(path.name for path in state_dir.glob("random_states_*.pkl")) == [
        "random_states_0.pkl",
        "random_states_1.pkl",
    ]
    assert [json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8")) for rank in range(2)] == [
        {"rank": 0, "step": 5},
        {"rank": 1, "step": 5},
    ]
