"""CPU acceptance tests for opt-in, complete Qwen-Image experiment states."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import socket
import time
import traceback
from types import SimpleNamespace

from accelerate import Accelerator
from PIL import Image
import numpy as np
import pytest
from safetensors.torch import load_file
import torch

from musubi_tuner.networks import lora_qwen_image
from musubi_tuner.qwen_image.qwen_image_model import QwenImageTransformerBlock
from musubi_tuner.qwen_image_train_network import QwenImageNetworkTrainer
from musubi_tuner.qwen_image_train_network import qwen_image_setup_parser
from musubi_tuner.training.parser_common import setup_parser_common
from musubi_tuner.training import trainer_base


FINGERPRINT = "a" * 64
METRICS = {
    "train_eval_loss_mean": 0.1,
    "train_eval_loss_low_noise": 0.11,
    "train_eval_loss_high_noise": 0.12,
    "val_loss_mean": 0.2,
    "val_loss_low_noise": 0.21,
    "val_loss_high_noise": 0.22,
}


def _experiment_states():
    # Import at use time: the fixture can run before the package implementation exists.
    return importlib.import_module("musubi_tuner.training.experiment_states")


def _args(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        experiment_dir=str(root),
        output_dir=str(root / "output"),
        output_name="tiny-qwen",
        model_version="original",
        save_precision="fp32",
        save_last_n_steps=None,
        val_dataset_config=str(root / "val-dataset.toml"),
        val_every_n_steps=2,
        val_seed_noise=42,
        val_level_noise_n=10,
        val_seed_noise_n=1,
        resume=None,
        resume_from_huggingface=False,
        sample_prompts=None,
        sample_every_n_steps=None,
        sample_every_n_epochs=None,
        sample_at_first=False,
    )


def _manifest():
    return SimpleNamespace(fingerprint=FINGERPRINT)


def _tiny_adapter():
    # The actual Qwen adapter factory and loader must accept the saved tensors.
    base = torch.nn.Sequential(QwenImageTransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4))
    base.requires_grad_(False)
    network = lora_qwen_image.create_arch_network(1, 2, 2, None, [], base)
    network.apply_to([], base, apply_text_encoder=False, apply_unet=True)
    assert list(network.parameters()) and all(p.dtype == torch.float32 for p in network.parameters())
    return base, network


def _prepared_state(accelerator: Accelerator):
    base, network = _tiny_adapter()
    # A registered frozen model exposes accidental full-model saves by the hook.
    frozen = torch.nn.Linear(3, 3)
    frozen.requires_grad_(False)
    # DDP cannot wrap an all-frozen module; register it for state saving directly.
    accelerator._models.append(frozen)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    network, optimizer, scheduler = accelerator.prepare(network, optimizer, scheduler)
    return base, frozen, network, optimizer, scheduler


def _update(accelerator, network, optimizer, scheduler, *, consume_rng=True):
    # A real optimizer update uses the rank-local RNG and populates AdamW state.
    optimizer.zero_grad()
    random_factor = 0.2 + random.random() + float(np.random.rand()) + float(torch.rand(())) if consume_rng else 1.0
    loss = sum(p.float().square().sum() for p in accelerator.unwrap_model(network).parameters()) * random_factor
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def _rng_snapshot():
    return (copy.deepcopy(random.getstate()), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone())


def _assert_rng_equal(expected):
    assert random.getstate() == expected[0]
    current_numpy = np.random.get_state()
    assert current_numpy[0] == expected[1][0]
    np.testing.assert_array_equal(current_numpy[1], expected[1][1])
    assert current_numpy[2:] == expected[1][2:]
    torch.testing.assert_close(torch.get_rng_state(), expected[2], rtol=0, atol=0)


def _state_snapshot(accelerator, network, optimizer, scheduler):
    return (
        {key: value.detach().cpu().clone() for key, value in accelerator.unwrap_model(network).state_dict().items()},
        copy.deepcopy(optimizer.state_dict()),
        copy.deepcopy(scheduler.state_dict()),
        _rng_snapshot(),
    )


def _assert_same_structure(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same_structure(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_same_structure(left, right)
    else:
        assert actual == expected


def _assert_state_equal(accelerator, network, optimizer, scheduler, expected):
    actual = _state_snapshot(accelerator, network, optimizer, scheduler)
    for left, right in zip(actual[:3], expected[:3]):
        _assert_same_structure(left, right)
    _assert_rng_equal(expected[3])


def _package(root: Path, destination: str, step: int) -> Path:
    branch = "current_training_states" if destination == "current" else "val_training_states/val-loss"
    return root / "output" / branch / f"tiny-qwen-step-{step}"


def _assert_package_files(package: Path, step: int, *, ranks: int, metrics):
    assert package.name == f"tiny-qwen-step-{step}"
    assert package.is_dir()
    assert sorted(path.name for path in package.glob("*.safetensors")) == ["model.safetensors"]
    assert (package / "optimizer.bin").is_file()
    assert (package / "scheduler.bin").is_file()
    assert sorted(path.name for path in package.glob("random_states_*.pkl")) == [
        f"random_states_{rank}.pkl" for rank in range(ranks)
    ]
    assert not list(package.rglob("*.png"))
    stage2 = json.loads((package / "val_loss_state.json").read_text(encoding="utf-8"))
    experiment = json.loads((package / "experiment_state.json").read_text(encoding="utf-8"))
    assert stage2["absolute_completed_step"] == experiment["absolute_completed_step"] == step
    assert stage2["validation_fingerprint"] == experiment["validation_fingerprint"] == FINGERPRINT
    assert stage2["model_version"] == experiment["model_version"] == "original"
    assert stage2["controls"] == experiment["controls"]
    assert experiment["output_name"] == "tiny-qwen"
    assert experiment["metrics_at_step"] == metrics
    tensors = load_file(str(package / "model.safetensors"))
    assert tensors and all(
        tensor.dtype == (torch.float32 if tensor.is_floating_point() else torch.int64)
        for tensor in tensors.values()
    )
    assert all(not key.startswith("module.") for key in tensors)
    assert any("lora_down" in key for key in tensors)
    return tensors


def _assert_qwen_loader_accepts(package: Path, tensors: dict):
    clone_base = torch.nn.Sequential(QwenImageTransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4))
    clone_base.requires_grad_(False)
    cloned = lora_qwen_image.create_arch_network_from_weights(1, tensors, unet=clone_base)
    cloned.apply_to(None, clone_base, apply_text_encoder=False, apply_unet=True)
    result = cloned.load_weights(str(package / "model.safetensors"))
    assert not result.missing_keys and not result.unexpected_keys
    for name, tensor in cloned.state_dict().items():
        torch.testing.assert_close(tensor, tensors[name].to(tensor.dtype), rtol=0, atol=0)


def _register(accelerator, network, args, manifest):
    trainer = QwenImageNetworkTrainer()
    trainer.validation_manifest = manifest
    trainer._register_hooks_and_resume(args, accelerator, network)
    return trainer


def test_tiny_qwen_adapter_fixture_has_trainable_fp32_weights_and_frozen_base():
    base, network = _tiny_adapter()
    assert any(p.requires_grad for p in network.parameters())
    network_parameter_ids = {id(parameter) for parameter in network.parameters()}
    assert all(not p.requires_grad for p in base.parameters() if id(p) not in network_parameter_ids)


def test_one_rank_package_roundtrip_restores_exact_next_update_without_sampling(tmp_path):
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    args, manifest = _args(tmp_path), _manifest()
    _, frozen, network, optimizer, scheduler = _prepared_state(accelerator)
    _register(accelerator, network, args, manifest)
    random.seed(11), np.random.seed(12), torch.manual_seed(13)
    _update(accelerator, network, optimizer, scheduler)
    before = _state_snapshot(accelerator, network, optimizer, scheduler)
    states = _experiment_states()
    states.save_package(args, accelerator, network, manifest, 5, None, ["final"], destination="current", samples_enabled=False)
    package = _package(tmp_path, "current", 5)
    tensors = _assert_package_files(package, 5, ranks=1, metrics=None)
    assert tensors.keys() == before[0].keys()
    _assert_same_structure(tensors, before[0])
    assert not any("weight" in path.name and path.name != "model.safetensors" for path in package.iterdir())
    assert not (tmp_path / "output" / "sample").exists()
    assert len(accelerator._models) >= 2 and frozen is not None

    _update(accelerator, network, optimizer, scheduler)
    expected_next = _state_snapshot(accelerator, network, optimizer, scheduler)
    _assert_qwen_loader_accepts(package, tensors)
    for parameter in accelerator.unwrap_model(network).parameters():
        parameter.data.add_(0.75)
    optimizer.param_groups[0]["lr"] = 0.91
    scheduler.step()
    random.random(), np.random.rand(), torch.rand(())
    assert states.load_package(args, accelerator, network, manifest, package) == 5
    _assert_state_equal(accelerator, network, optimizer, scheduler, before)
    _update(accelerator, network, optimizer, scheduler)
    _assert_state_equal(accelerator, network, optimizer, scheduler, expected_next)
    accelerator.free_memory()


@pytest.mark.parametrize("destination", ["current", "best"])
def test_complete_package_is_loadable_from_current_or_best_with_no_samples(tmp_path, destination):
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    args, manifest = _args(tmp_path), _manifest()
    _, _, network, optimizer, scheduler = _prepared_state(accelerator)
    _register(accelerator, network, args, manifest)
    _update(accelerator, network, optimizer, scheduler)
    expected = _state_snapshot(accelerator, network, optimizer, scheduler)
    states = _experiment_states()
    metrics = METRICS if destination == "best" else None
    states.save_package(
        args, accelerator, network, manifest, 8, metrics, ["new_best"] if metrics else ["periodic"],
        destination=destination, samples_enabled=False,
    )
    package = _package(tmp_path, destination, 8)
    _assert_package_files(package, 8, ranks=1, metrics=metrics)
    assert not list((tmp_path / "output").rglob("*.png"))
    for parameter in accelerator.unwrap_model(network).parameters():
        parameter.data.zero_()
    assert states.load_package(args, accelerator, network, manifest, package) == 8
    _assert_state_equal(accelerator, network, optimizer, scheduler, expected)
    accelerator.free_memory()


def test_resumed_initial_validation_promotes_published_current_without_resaving(tmp_path, monkeypatch, request):
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    request.addfinalizer(accelerator.free_memory)
    args, manifest = _args(tmp_path), _manifest()
    _, _, network, _, _ = _prepared_state(accelerator)
    _register(accelerator, network, args, manifest)
    states = _experiment_states()
    previous_metrics = {**METRICS, "val_loss_mean": 1.0, "val_loss_low_noise": 1.0, "val_loss_high_noise": 1.0}
    states.save_package(
        args, accelerator, network, manifest, 2, previous_metrics, ["new_best"],
        destination="best", samples_enabled=False,
    )
    states.save_package(
        args, accelerator, network, manifest, 5, None, ["periodic"],
        destination="current", samples_enabled=False,
    )
    args.resume = str(_package(tmp_path, "current", 5))
    monkeypatch.setattr(accelerator, "save_state", lambda *_args, **_kwargs: pytest.fail("resume promotion resaved state"))

    promoted = states.promote_published_current_to_best(args, accelerator, network, manifest, 5, METRICS)

    assert promoted == _package(tmp_path, "best", 5)
    assert not _package(tmp_path, "current", 5).exists()
    assert _package(tmp_path, "current", 2).is_dir()
    assert not _package(tmp_path, "best", 2).exists()
    assert len(list((tmp_path / "output").rglob("tiny-qwen-step-5"))) == 1
    updated = json.loads((promoted / "experiment_state.json").read_text(encoding="utf-8"))
    assert updated["metrics_at_step"] == METRICS
    assert set(updated["save_reasons"]) == {"periodic", "new_best"}
    assert states.read_best_package(args, accelerator, network, manifest) == (promoted, METRICS["val_loss_mean"])


def test_resumed_initial_validation_uses_promotion_without_new_save(tmp_path, monkeypatch):
    class OptionalArgs(SimpleNamespace):
        def __getattr__(self, _name):
            return None

    class StopAtFirstBatch(Exception):
        pass

    class LoopNetwork(torch.nn.Linear):
        def on_epoch_start(self, _transformer):
            pass

        def on_step_start(self):
            pass

    @contextmanager
    def accumulate(_model):
        yield

    accelerator = SimpleNamespace(
        is_main_process=True, is_local_main_process=True, num_processes=1, device=torch.device("cpu"),
        trackers=[], print=lambda *_args: None, init_trackers=lambda *_args, **_kwargs: None,
        unwrap_model=lambda model: model, accumulate=accumulate,
    )
    args = OptionalArgs(**vars(qwen_image_setup_parser(setup_parser_common()).parse_args([])))
    args.experiment_dir = str(tmp_path)
    args.output_dir = str(tmp_path / "output")
    args.output_name = "tiny-qwen"
    args.resume = str(_package(tmp_path, "current", 5))
    args.val_dataset_config = str(tmp_path / "val-dataset.toml")
    args.val_every_n_steps = 10
    args.gradient_accumulation_steps = 1
    args.max_train_steps = 1
    args.save_every_n_steps = None
    args.save_every_n_epochs = None
    args.sample_prompts = None
    args.save_precision = "fp32"
    args.no_metadata = True
    args.max_grad_norm = 0.0
    trainer = QwenImageNetworkTrainer()
    trainer.validation_manifest = _manifest()
    trainer.validation_resume_step = 5
    monkeypatch.setattr(trainer, "evaluate_validation_event", lambda *_args: METRICS)
    monkeypatch.setattr(trainer, "process_batch", lambda *_args: (_ for _ in ()).throw(StopAtFirstBatch()))
    monkeypatch.setattr(trainer, "_run_experiment_package_event", lambda *_args, **_kwargs: pytest.fail("resaved step 5"))
    monkeypatch.setattr(trainer_base, "clean_memory_on_device", lambda _device: None)
    monkeypatch.setattr(trainer_base.sai_model_spec, "build_metadata", lambda *_args, **_kwargs: {})
    states = _experiment_states()
    monkeypatch.setattr(states, "decide_best_event", lambda *_args: True)
    promoted_steps = []
    monkeypatch.setattr(
        states, "promote_published_current_to_best",
        lambda _args, _accelerator, _network, _manifest, step, metrics: promoted_steps.append((step, metrics)),
        raising=False,
    )
    network = LoopNetwork(1, 1)
    dataset = SimpleNamespace(batch_size=1, get_metadata=lambda: {})
    with pytest.raises(StopAtFirstBatch):
        trainer._run_training_loop(
            args, accelerator, "cpu-session", 0.0,
            SimpleNamespace(datasets=[dataset], num_train_items=1), [{"latents": torch.ones(1)}],
            SimpleNamespace(value=0), _SampleTransformer(), network, network, None, "SGD", [],
            lambda: None, lambda: None, None, [], None, None, torch.float32, torch.float32,
        )
    assert promoted_steps == [(5, METRICS)]


@pytest.mark.parametrize("damage", ["missing-sidecar", "mismatched-sidecar", "save-failure"])
def test_incomplete_package_never_publishes_or_loads(tmp_path, monkeypatch, damage):
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    args, manifest = _args(tmp_path), _manifest()
    _, _, network, _, _ = _prepared_state(accelerator)
    _register(accelerator, network, args, manifest)
    states = _experiment_states()
    package = _package(tmp_path, "current", 3)
    if damage == "save-failure":
        def fail_save(*_args, **_kwargs):
            raise OSError("injected state save failure")

        monkeypatch.setattr(accelerator, "save_state", fail_save)
        with pytest.raises((OSError, RuntimeError), match="injected state save failure"):
            states.save_package(args, accelerator, network, manifest, 3, None, ["final"], samples_enabled=False)
        assert not package.exists()
        assert not any(path.name == package.name for path in (tmp_path / "output").rglob("*"))
    else:
        states.save_package(args, accelerator, network, manifest, 3, None, ["final"], samples_enabled=False)
        metadata = package / "experiment_state.json"
        if damage == "missing-sidecar":
            metadata.unlink()
        else:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            payload["absolute_completed_step"] = 4
            metadata.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(accelerator, "load_state", lambda *_a, **_kw: pytest.fail("load_state preceded preflight"))
        with pytest.raises((ValueError, RuntimeError), match="sidecar|metadata|incomplete|mismatch|step"):
            states.load_package(args, accelerator, network, manifest, package)
    accelerator.free_memory()


def _set_rank_environment(rank: int, port: int):
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank),
        WORLD_SIZE="2", LOCAL_WORLD_SIZE="2", ACCELERATE_USE_CPU="true", OMP_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="-1",
    )


def _distributed_worker(rank: int, port: int, root_name: str, damage: str | None):
    root = Path(root_name)
    try:
        _set_rank_environment(rank, port)
        accelerator = Accelerator(cpu=True, mixed_precision="no")
        args, manifest = _args(root), _manifest()
        _, _, network, optimizer, scheduler = _prepared_state(accelerator)
        assert accelerator.num_processes == 2
        assert isinstance(network, torch.nn.parallel.DistributedDataParallel)
        _register(accelerator, network, args, manifest)
        _update(accelerator, network, optimizer, scheduler, consume_rng=False)
        random.seed(21 + rank), np.random.seed(31 + rank), torch.manual_seed(41 + rank)
        expected = _state_snapshot(accelerator, network, optimizer, scheduler)
        states = _experiment_states()
        states.save_package(args, accelerator, network, manifest, 6, None, ["periodic"], samples_enabled=False)
        package = _package(root, "current", 6)
        accelerator.wait_for_everyone()
        if damage and rank == 0:
            rng_file = package / "random_states_1.pkl"
            if damage == "missing":
                rng_file.unlink()
            else:
                rng_file.write_bytes(b"corrupt rank RNG state")
        accelerator.wait_for_everyone()
        if damage:
            accelerator.load_state = lambda *_a, **_kw: pytest.fail("load_state preceded rank-file preflight")
            with pytest.raises((ValueError, RuntimeError), match="random_states_1|rank|RNG|corrupt|missing"):
                states.load_package(args, accelerator, network, manifest, package)
        else:
            for parameter in accelerator.unwrap_model(network).parameters():
                parameter.data.add_(0.75)
            optimizer.param_groups[0]["lr"] = 0.91
            scheduler.step()
            random.random(), np.random.rand(), torch.rand(())
            assert states.load_package(args, accelerator, network, manifest, package) == 6
            _assert_state_equal(accelerator, network, optimizer, scheduler, expected)
        (root / f"rank-{rank}-{damage or 'roundtrip'}.json").write_text(
            json.dumps({"rank": rank, "step": 6, "passed": True}), encoding="utf-8"
        )
        accelerator.wait_for_everyone()
        accelerator.free_memory()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except BaseException:
        (root / f"rank-{rank}-{damage or 'roundtrip'}-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def _run_two_rank(root: Path, damage: str | None = None):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_distributed_worker, args=(rank, port, str(root), damage)) for rank in range(2)]
    deadline = time.monotonic() + 70
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert all(not process.is_alive() for process in processes), "two-rank CPU state test exceeded 70 seconds"
        errors = [path.read_text(encoding="utf-8") for path in root.glob("rank-*-error.txt")]
        assert [process.exitcode for process in processes] == [0, 0], "\n".join(errors)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)


@pytest.mark.parametrize("damage", [None, "missing", "corrupt"])
def test_two_rank_ddp_package_has_canonical_adapter_and_rank_local_rng(tmp_path, damage):
    _run_two_rank(tmp_path, damage)
    package = _package(tmp_path, "current", 6)
    if damage is None:
        _assert_package_files(package, 6, ranks=2, metrics=None)
    for rank in range(2):
        result = json.loads((tmp_path / f"rank-{rank}-{damage or 'roundtrip'}.json").read_text(encoding="utf-8"))
        assert result == {"rank": rank, "step": 6, "passed": True}


@pytest.fixture
def package_context(tmp_path):
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    args, manifest = _args(tmp_path), _manifest()
    _, _, network, optimizer, scheduler = _prepared_state(accelerator)
    _register(accelerator, network, args, manifest)
    context = SimpleNamespace(
        root=tmp_path, args=args, manifest=manifest, accelerator=accelerator,
        network=network, optimizer=optimizer, scheduler=scheduler, states=_experiment_states(),
    )
    yield context
    accelerator.free_memory()


def _save_at(context, step: int, *, score: float | None = None, destination: str = "current") -> Path:
    metrics = None if score is None else {**METRICS, "val_loss_mean": score}
    reasons = ["new_best"] if destination == "best" else ["periodic"]
    return context.states.save_package(
        context.args, context.accelerator, context.network, context.manifest,
        step, metrics, reasons, destination=destination, samples_enabled=False,
    )


def test_best_decision_uses_only_strict_finite_unfamiliar_mean():
    states = _experiment_states()
    assert states.is_new_best(METRICS, None)  # the first result may occur at step 0
    assert states.is_new_best({**METRICS, "val_loss_mean": 0.19, "train_eval_loss_mean": 99}, 0.2)
    assert not states.is_new_best({**METRICS, "val_loss_mean": 0.2, "train_eval_loss_mean": 0}, 0.2)
    assert not states.is_new_best({**METRICS, "val_loss_mean": 0.3, "train_eval_loss_mean": 0}, 0.2)
    assert not states.is_new_best(None, 0.2)
    with pytest.raises(ValueError, match="metrics|validation|finite"):
        states.is_new_best({"val_loss_mean": 0.1}, 0.2)
    with pytest.raises(ValueError, match="metrics|validation|finite"):
        states.is_new_best({**METRICS, "val_loss_mean": float("nan")}, 0.2)


def test_step_zero_best_and_strict_replacement_move_whole_package(package_context):
    context = package_context
    assert context.states.read_best_package(context.args, context.accelerator, context.network, context.manifest) == (None, None)
    first = _save_at(context, 0, score=0.2, destination="best")
    assert first == _package(context.root, "best", 0)
    assert context.states.read_best_package(context.args, context.accelerator, context.network, context.manifest) == (first, 0.2)
    original_model_file = first / "model.safetensors"
    original_file_id = original_model_file.stat().st_ino
    assert not context.states.is_new_best({**METRICS, "val_loss_mean": 0.2}, 0.2)
    assert sorted(path.name for path in first.parent.iterdir()) == [first.name]

    assert context.states.is_new_best({**METRICS, "val_loss_mean": 0.1}, 0.2)
    second = _save_at(context, 4, score=0.1, destination="best")
    former = _package(context.root, "current", 0)
    assert second == _package(context.root, "best", 4)
    assert former.is_dir() and not first.exists()
    if original_file_id:
        assert (former / "model.safetensors").stat().st_ino == original_file_id
    assert context.states.read_best_package(context.args, context.accelerator, context.network, context.manifest) == (second, 0.1)
    assert sorted(path.name for path in second.parent.iterdir()) == [second.name]


@pytest.mark.parametrize("failure", ["save", "publish"])
def test_failed_new_best_keeps_previous_complete_best(package_context, monkeypatch, failure):
    context = package_context
    first = _save_at(context, 2, score=0.2, destination="best")
    before = (first / "model.safetensors").read_bytes()
    if failure == "save":
        def fail_save(*_args, **_kwargs):
            raise RuntimeError("injected candidate save failure")
        monkeypatch.setattr(context.accelerator, "save_state", fail_save)
    else:
        original_rename = Path.rename

        def fail_candidate_publish(path, target):
            if path.name.startswith(".experiment-stage-") and Path(target) == _package(context.root, "best", 4):
                raise OSError("injected candidate publish failure")
            return original_rename(path, target)

        monkeypatch.setattr(Path, "rename", fail_candidate_publish)
    with pytest.raises(RuntimeError, match="candidate|publish|save"):
        _save_at(context, 4, score=0.1, destination="best")
    assert first.is_dir() and (first / "model.safetensors").read_bytes() == before
    assert not _package(context.root, "best", 4).exists()
    assert not _package(context.root, "current", 2).exists()


def test_whole_package_retention_has_inclusive_step_boundary_and_protects_best(package_context):
    context = package_context
    best = _save_at(context, 0, score=0.1, destination="best")
    for step in (3, 4, 12):
        _save_at(context, step)
    context.states.prune_current_packages(
        context.args, context.accelerator, context.network, context.manifest, 12
    )
    assert all(_package(context.root, "current", step).is_dir() for step in (3, 4, 12))
    context.args.save_last_n_steps = 8
    context.states.prune_current_packages(
        context.args, context.accelerator, context.network, context.manifest, 12
    )
    assert not _package(context.root, "current", 3).exists()
    assert all(_package(context.root, "current", step).is_dir() for step in (4, 12))
    assert best.is_dir()  # best step 0 lies outside the current window


@pytest.mark.parametrize("former_step,kept", [(4, True), (3, False)])
def test_replacing_best_retains_or_removes_former_best_immediately(package_context, former_step, kept):
    context = package_context
    context.args.save_last_n_steps = 8
    _save_at(context, former_step, score=0.2, destination="best")
    _save_at(context, 12, score=0.1, destination="best")
    assert _package(context.root, "best", 12).is_dir()
    assert _package(context.root, "current", former_step).is_dir() is kept
    assert not _package(context.root, "best", former_step).exists()


def test_retention_ignores_foreign_and_incomplete_folders_and_input_data(package_context):
    context = package_context
    owned = _save_at(context, 1)
    current = owned.parent
    foreign = current / "another-run-step-0"
    incomplete = current / "tiny-qwen-step-0"
    for directory in (foreign, incomplete):
        directory.mkdir()
        (directory / "user-note.txt").write_text("keep", encoding="utf-8")
    dataset = context.root / "dataset" / "source.png"
    cache = context.root / "cache" / "source.safetensors"
    dataset.parent.mkdir()
    cache.parent.mkdir()
    dataset.write_bytes(b"image input")
    cache.write_bytes(b"cache input")
    context.args.save_last_n_steps = 0
    context.states.prune_current_packages(context.args, context.accelerator, context.network, context.manifest, 12)
    assert not owned.exists()
    assert (foreign / "user-note.txt").read_text(encoding="utf-8") == "keep"
    assert (incomplete / "user-note.txt").read_text(encoding="utf-8") == "keep"
    assert dataset.read_bytes() == b"image input" and cache.read_bytes() == b"cache input"


def test_retention_does_not_follow_symlink_to_external_directory(package_context):
    context = package_context
    target = context.root / "outside-output"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("external", encoding="utf-8")
    link = _package(context.root, "current", 1)
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")
    context.args.save_last_n_steps = 0
    context.states.prune_current_packages(context.args, context.accelerator, context.network, context.manifest, 12)
    assert link.is_symlink() and marker.read_text(encoding="utf-8") == "external"


def test_older_current_resume_preserves_its_own_step_and_later_best_score(package_context):
    context = package_context
    _save_at(context, 2, score=0.2, destination="best")
    older = _save_at(context, 4, score=0.4)
    later = _save_at(context, 12, score=0.1, destination="best")
    assert context.states.load_package(context.args, context.accelerator, context.network, context.manifest, older) == 4
    older_metadata = json.loads((older / "experiment_state.json").read_text(encoding="utf-8"))
    assert older_metadata["absolute_completed_step"] == 4
    assert older_metadata["metrics_at_step"]["val_loss_mean"] == 0.4
    assert context.states.read_best_package(context.args, context.accelerator, context.network, context.manifest) == (later, 0.1)
    assert not context.states.is_new_best({**METRICS, "val_loss_mean": 0.15}, 0.1)
    assert context.states.is_new_best({**METRICS, "val_loss_mean": 0.05}, 0.1)

    before = (later / "model.safetensors").read_bytes()
    with pytest.raises((ValueError, RuntimeError), match="published|exists|collision"):
        _save_at(context, 12, destination="current")
    assert (later / "model.safetensors").read_bytes() == before
    changed_manifest = SimpleNamespace(fingerprint="b" * 64)
    context.args.val_seed_noise = 43
    with pytest.raises((ValueError, RuntimeError), match="fingerprint|protocol|sidecar|metadata|controls"):
        context.states.load_package(context.args, context.accelerator, context.network, changed_manifest, older)


def _best_decision_worker(rank: int, port: int, root_name: str, damaged_best: bool):
    root = Path(root_name)
    try:
        _set_rank_environment(rank, port)
        accelerator = Accelerator(cpu=True, mixed_precision="no")
        args, manifest = _args(root), _manifest()
        _, _, network, _, _ = _prepared_state(accelerator)
        assert accelerator.num_processes == 2
        assert isinstance(network, torch.nn.parallel.DistributedDataParallel)
        accelerator.save_state = lambda *_a, **_kw: pytest.fail("best decision must precede state save")
        states = _experiment_states()
        try:
            decision = states.decide_best_event(
                args, accelerator, network, manifest, METRICS if rank == 0 else None
            )
        except (ValueError, RuntimeError) as error:
            assert damaged_best
            assert any(word in str(error).lower() for word in ("best", "sidecar", "incomplete", "metadata"))
            decision = "error"
        assert decision == ("error" if damaged_best else True)
        (root / f"decision-rank-{rank}.json").write_text(
            json.dumps({"rank": rank, "decision": decision}), encoding="utf-8"
        )
        accelerator.free_memory()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except BaseException:
        (root / f"decision-rank-{rank}-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


@pytest.mark.parametrize("damaged_best", [False, True])
def test_two_rank_best_decision_or_error_agrees_before_save_without_deadlock(tmp_path, damaged_best):
    if damaged_best:
        malformed_best = _package(tmp_path, "best", 2)
        malformed_best.mkdir(parents=True)
        (malformed_best / "experiment_state.json").write_text("{incomplete", encoding="utf-8")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_best_decision_worker, args=(rank, port, str(tmp_path), damaged_best))
        for rank in range(2)
    ]
    deadline = time.monotonic() + 70
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert all(not process.is_alive() for process in processes), "two-rank best decision exceeded 70 seconds"
        errors = [path.read_text(encoding="utf-8") for path in tmp_path.glob("decision-rank-*-error.txt")]
        assert [process.exitcode for process in processes] == [0, 0], "\n".join(errors)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
    outcomes = [json.loads((tmp_path / f"decision-rank-{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    assert [outcome["decision"] for outcome in outcomes] == ["error" if damaged_best else True] * 2


class _SampleTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)])
        self.layers[1].eval()

    def switch_block_swap_for_inference(self):
        pass

    def switch_block_swap_for_training(self):
        pass


class _SampleVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.placement = "cpu"

    def to(self, device, *args, **kwargs):
        self.placement = str(device)
        return self


def _run_sample_event(context, monkeypatch, *, step, reasons, metrics, enabled=True, trainer=None, transformer=None, vae=None):
    trainer = trainer or QwenImageNetworkTrainer()
    transformer = transformer or _SampleTransformer()
    vae = vae or _SampleVAE()
    context.args.sample_prompts = "fixture-prompts.txt" if enabled else None
    context.args.sample_every_n_steps = (1 if reasons == ["sample"] else 4) if enabled else None
    context.args.sample_at_first = False
    calls = {"save": 0, "generation": 0}
    original_save = context.accelerator.save_state

    def save_once(*args, **kwargs):
        calls["save"] += 1
        return original_save(*args, **kwargs)

    def create_png(_accelerator, _args, _transformer, _dtype, _vae, save_dir, *_rest):
        calls["generation"] += 1
        Image.new("RGB", (8, 8), (12, 34, 56)).save(Path(save_dir) / "sample.png")

    monkeypatch.setattr(context.accelerator, "save_state", save_once)
    monkeypatch.setattr(trainer, "sample_image_inference", create_png)
    trainer._run_experiment_package_event(
        context.args, context.accelerator, transformer, context.network, context.manifest,
        absolute_step=step, reasons=reasons, metrics_at_step=metrics,
        sample_resources=vae if enabled else None,
        sample_parameters=[{"prompt": "fixture", "enum": 0}] if enabled else None,
        dit_dtype=torch.float32, epoch=None,
    )
    return calls


@pytest.mark.parametrize(
    "step,reasons,score,enabled,destination",
    [
        (1, ["sample"], None, True, "current"),
        (0, ["new_best"], 0.2, True, "best"),
        (4, ["periodic", "epoch", "final", "new_best", "sample"], 0.1, True, "best"),
        (1, ["final"], None, False, "current"),
    ],
)
def test_sample_reasons_publish_one_complete_package_and_one_png(
    package_context, monkeypatch, step, reasons, score, enabled, destination,
):
    context = package_context
    metrics = None if score is None else {**METRICS, "val_loss_mean": score}
    calls = _run_sample_event(context, monkeypatch, step=step, reasons=reasons, metrics=metrics, enabled=enabled)
    package = _package(context.root, destination, step)
    assert package.is_dir()
    assert calls == {"save": 1, "generation": int(enabled)}
    assert len(list((package / "samples").glob("*.png"))) == int(enabled)
    assert json.loads((package / "experiment_state.json").read_text(encoding="utf-8"))["save_reasons"] == reasons
    assert not (context.root / "output" / "sample").exists()
    assert sorted(path.name for path in (context.root / "output").rglob("*.safetensors")) == ["model.safetensors"]
    assert not list((context.root / "output").rglob("*-state"))


@pytest.mark.parametrize("failure", ["none-image", "missing-png"])
def test_sample_failure_never_publishes_partial_or_replaces_best(package_context, monkeypatch, failure):
    context = package_context
    previous = _save_at(context, 2, score=0.2, destination="best")
    before = (previous / "model.safetensors").read_bytes()
    trainer = QwenImageNetworkTrainer()
    transformer, vae = _SampleTransformer(), _SampleVAE()
    context.args.sample_prompts = "fixture-prompts.txt"
    context.args.sample_every_n_steps = 4
    if failure == "none-image":
        monkeypatch.setattr(trainer, "do_inference", lambda *_a, **_kw: None)
    else:
        monkeypatch.setattr(trainer, "sample_image_inference", lambda *_a, **_kw: None)
    with pytest.raises((RuntimeError, ValueError), match="sample|image|PNG|incomplete"):
        trainer._run_experiment_package_event(
            context.args, context.accelerator, transformer, context.network, context.manifest,
            absolute_step=4, reasons=["new_best"], metrics_at_step={**METRICS, "val_loss_mean": 0.1},
            sample_resources=vae, sample_parameters=[{"prompt": "fixture", "enum": 0}],
            dit_dtype=torch.float32, epoch=None,
        )
    assert previous.is_dir() and (previous / "model.safetensors").read_bytes() == before
    assert not _package(context.root, "best", 4).exists()
    assert not _package(context.root, "current", 2).exists()
    assert not (context.root / "output" / "sample").exists()


def _all_rng_snapshot():
    return _rng_snapshot(), [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None


def _assert_all_rng_equal(expected):
    _assert_rng_equal(expected[0])
    current_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    assert (current_cuda is None) == (expected[1] is None)
    if current_cuda is not None:
        assert len(current_cuda) == len(expected[1])
        for actual, previous in zip(current_cuda, expected[1]):
            torch.testing.assert_close(actual, previous, rtol=0, atol=0)


@pytest.mark.parametrize("failure", [False, True])
def test_opt_in_sample_preparation_restores_all_rng_even_on_error(tmp_path, monkeypatch, failure):
    trainer = QwenImageNetworkTrainer()
    args = _args(tmp_path)
    args.sample_prompts = "fixture-prompts.txt"
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    prepared = ([{"prompt": "fixture"}], _SampleVAE())

    def consume_rng(*_args):
        random.random(), np.random.rand(), torch.rand(())
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(19)
        if failure:
            raise RuntimeError("injected preparation failure")
        return prepared

    monkeypatch.setattr(trainer, "prepare_sampling", consume_rng)
    before = _all_rng_snapshot()
    if failure:
        with pytest.raises(RuntimeError, match="preparation failure"):
            trainer._prepare_experiment_sampling(args, accelerator, torch.float32)
    else:
        assert trainer._prepare_experiment_sampling(args, accelerator, torch.float32) is prepared
    _assert_all_rng_equal(before)
    accelerator.free_memory()


@pytest.mark.parametrize("failure", [None, "before", "inference", "after"])
def test_sample_event_restores_modes_rng_gradients_and_next_update(package_context, monkeypatch, failure):
    context = package_context
    trainer = QwenImageNetworkTrainer()
    transformer, vae = _SampleTransformer(), _SampleVAE()
    modules = [*transformer.modules(), *context.accelerator.unwrap_model(context.network).modules(), *vae.modules()]
    modes_before = [module.training for module in modules]
    params = list(context.accelerator.unwrap_model(context.network).parameters())
    for parameter in params:
        parameter.grad = torch.ones_like(parameter)
    gradients_before = [parameter.grad.clone() for parameter in params]
    before = _state_snapshot(context.accelerator, context.network, context.optimizer, context.scheduler)
    all_rng_before = _all_rng_snapshot()
    _update(context.accelerator, context.network, context.optimizer, context.scheduler)
    expected_next = _state_snapshot(context.accelerator, context.network, context.optimizer, context.scheduler)
    context.accelerator.unwrap_model(context.network).load_state_dict(before[0])
    context.optimizer.load_state_dict(copy.deepcopy(before[1]))
    context.scheduler.load_state_dict(copy.deepcopy(before[2]))
    random.setstate(before[3][0]), np.random.set_state(before[3][1]), torch.set_rng_state(before[3][2])
    if all_rng_before[1] is not None:
        torch.cuda.set_rng_state_all(all_rng_before[1])
    for parameter, gradient in zip(params, gradients_before):
        parameter.grad = gradient.clone()

    def disturb(stage):
        random.random(), np.random.rand(), torch.rand(())
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(23)
        for module in modules:
            module.training = not module.training
        vae.to("sample-device")
        if failure == stage:
            raise RuntimeError(f"injected {stage} failure")

    def before_hook(*_args):
        disturb("before")

    def infer(_accelerator, _args, _transformer, _dtype, _vae, save_dir, *_rest):
        disturb("inference")
        Image.new("RGB", (8, 8), (2, 4, 6)).save(Path(save_dir) / "sample.png")

    def after_hook(*_args):
        disturb("after")

    monkeypatch.setattr(trainer, "on_before_sample_images", before_hook)
    monkeypatch.setattr(trainer, "sample_image_inference", infer)
    monkeypatch.setattr(trainer, "on_after_sample_images", after_hook)
    context.args.sample_prompts = "fixture-prompts.txt"
    context.args.sample_every_n_steps = 4
    if failure:
        with pytest.raises((RuntimeError, ValueError), match="injected|sample"):
            trainer._run_experiment_package_event(
                context.args, context.accelerator, transformer, context.network, context.manifest,
                absolute_step=4, reasons=["sample"], metrics_at_step=None,
                sample_resources=vae, sample_parameters=[{"prompt": "fixture", "enum": 0}],
                dit_dtype=torch.float32, epoch=None,
            )
        assert not _package(context.root, "current", 4).exists()
    else:
        trainer._run_experiment_package_event(
            context.args, context.accelerator, transformer, context.network, context.manifest,
            absolute_step=4, reasons=["sample"], metrics_at_step=None,
            sample_resources=vae, sample_parameters=[{"prompt": "fixture", "enum": 0}],
            dit_dtype=torch.float32, epoch=None,
        )
        assert len(list((_package(context.root, "current", 4) / "samples").glob("*.png"))) == 1
    assert [module.training for module in modules] == modes_before
    assert vae.placement == "cpu"
    _assert_state_equal(context.accelerator, context.network, context.optimizer, context.scheduler, before)
    _assert_all_rng_equal(all_rng_before)
    for parameter, gradient in zip(params, gradients_before):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    _update(context.accelerator, context.network, context.optimizer, context.scheduler)
    _assert_state_equal(context.accelerator, context.network, context.optimizer, context.scheduler, expected_next)


def test_experiment_loop_samples_and_saves_validation_weights_with_mode_switching_optimizer(package_context, monkeypatch):
    context = package_context
    class OptionalArgs(SimpleNamespace):
        def __getattr__(self, _name):
            return None

    defaults = vars(qwen_image_setup_parser(setup_parser_common()).parse_args([]))
    defaults.update(vars(context.args))
    args = OptionalArgs(**defaults)
    args.max_train_steps = 2
    args.gradient_accumulation_steps = 1
    args.val_every_n_steps = 1
    args.save_every_n_steps = 1
    args.save_every_n_epochs = None
    args.sample_prompts = "fixture-prompts.txt"
    args.sample_every_n_steps = 1
    args.sample_at_first = False
    args.no_metadata = True
    args.max_grad_norm = 0.0
    trainer = QwenImageNetworkTrainer()
    trainer.validation_manifest = context.manifest
    network = context.accelerator.unwrap_model(context.network)
    parameter_name, parameter = next(iter(network.named_parameters()))
    mode = {"eval": False}
    validation_weights = {}
    sampled_weights = {}
    after_first = {}

    def optimizer_eval():
        if not mode["eval"]:
            with torch.no_grad():
                parameter.view(-1)[0].neg_()
            mode["eval"] = True

    def optimizer_train():
        if mode["eval"]:
            with torch.no_grad():
                parameter.view(-1)[0].neg_()
            mode["eval"] = False

    def evaluate(*event_args):
        step = event_args[-1]
        if step == 0:
            return None
        validation_weights[step] = parameter.view(-1)[0].detach().clone()
        if step == 1:
            after_first["network"] = {key: value.detach().clone() for key, value in network.state_dict().items()}
            after_first["optimizer"] = copy.deepcopy(context.optimizer.state_dict())
        return {**METRICS, "val_loss_mean": float(abs(validation_weights[step]))}

    def create_png(_accelerator, _args, _transformer, _dtype, _vae, save_dir, _prompt, _epoch, step):
        sampled_weights[step] = parameter.view(-1)[0].detach().clone()
        Image.new("RGB", (8, 8), (12, 34, 56)).save(Path(save_dir) / "sample.png")

    monkeypatch.setattr(trainer, "evaluate_validation_event", evaluate)
    monkeypatch.setattr(trainer, "process_batch", lambda *_args: (parameter.view(-1)[0].square(), {}))
    monkeypatch.setattr(trainer, "sample_image_inference", create_png)
    monkeypatch.setattr(trainer_base, "clean_memory_on_device", lambda _device: None)
    monkeypatch.setattr(trainer_base.sai_model_spec, "build_metadata", lambda *_args, **_kwargs: {})
    dataset = SimpleNamespace(batch_size=1, get_metadata=lambda: {})
    trainer._run_training_loop(
        args, context.accelerator, "cpu-session", 0.0,
        SimpleNamespace(datasets=[dataset], num_train_items=1), [{"latents": torch.ones(1)}],
        SimpleNamespace(value=0), _SampleTransformer(), context.network, context.network,
        context.optimizer, "AdamW", [], optimizer_train, optimizer_eval, context.scheduler, [],
        _SampleVAE(), [{"prompt": "fixture", "enum": 0}], torch.float32, torch.float32,
    )
    assert set(validation_weights) == set(sampled_weights) == {1, 2}
    network.load_state_dict(after_first["network"])
    context.optimizer.load_state_dict(after_first["optimizer"])
    mode["eval"] = False
    context.optimizer.zero_grad()
    context.accelerator.backward(parameter.view(-1)[0].square())
    context.optimizer.step()
    torch.testing.assert_close(parameter.view(-1)[0].detach(), validation_weights[2], rtol=0, atol=0)
    for step in (1, 2):
        package = next(path for location in ("current", "best") if (path := _package(context.root, location, step)).exists())
        saved = load_file(str(package / "model.safetensors"))[parameter_name].view(-1)[0]
        torch.testing.assert_close(sampled_weights[step], validation_weights[step], rtol=0, atol=0)
        torch.testing.assert_close(saved, validation_weights[step], rtol=0, atol=0)


def test_controlled_loop_coalesces_all_step_one_reasons_into_one_event(tmp_path, monkeypatch):
    class OptionalArgs(SimpleNamespace):
        def __getattr__(self, _name):
            return None

    class LoopNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def on_epoch_start(self, _transformer):
            pass

        def on_step_start(self):
            pass

        def save_weights(self, path, *_args):
            Path(path).write_bytes(b"legacy save should not run in experiment mode")

    class LoopAccelerator:
        is_main_process = True
        is_local_main_process = True
        num_processes = 1
        device = torch.device("cpu")
        sync_gradients = True
        optimizer_step_was_skipped = False
        trackers = []

        @contextmanager
        def accumulate(self, _model):
            yield

        def backward(self, loss):
            loss.backward()

        def unwrap_model(self, model):
            return model

        def init_trackers(self, *_args, **_kwargs):
            pass

        def print(self, *_args):
            pass

        def log(self, *_args, **_kwargs):
            pass

        def wait_for_everyone(self):
            pass

        def end_training(self):
            pass

    args = OptionalArgs(**vars(qwen_image_setup_parser(setup_parser_common()).parse_args([])))
    args.experiment_dir = str(tmp_path)
    args.output_dir = str(tmp_path / "output")
    args.output_name = "tiny-qwen"
    args.val_dataset_config = str(tmp_path / "val-dataset.toml")
    args.val_every_n_steps = 1
    args.gradient_accumulation_steps = 1
    args.max_train_steps = 1
    args.max_train_epochs = None
    args.save_every_n_steps = 1
    args.save_every_n_epochs = 1
    args.sample_prompts = "fixture-prompts.txt"
    args.sample_every_n_steps = 1
    args.sample_at_first = False
    args.save_state = False
    args.save_state_on_train_end = False
    args.save_precision = "fp32"
    args.no_metadata = True
    args.log_with = None
    args.max_grad_norm = 0.0
    network, transformer, accelerator = LoopNetwork(), _SampleTransformer(), LoopAccelerator()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1)
    trainer = QwenImageNetworkTrainer()
    trainer.validation_manifest = _manifest()
    calls = []

    def event(*_args, **kwargs):
        calls.append((kwargs["absolute_step"], list(kwargs["reasons"]), kwargs["metrics_at_step"]))

    monkeypatch.setattr(trainer, "_run_experiment_package_event", event, raising=False)
    monkeypatch.setattr(trainer, "evaluate_validation_event", lambda *_args: METRICS if _args[-1] == 1 else None)
    monkeypatch.setattr(trainer, "process_batch", lambda *_args: (network.weight.square(), {}))
    monkeypatch.setattr(trainer, "generate_step_logs", lambda *_args: {})
    monkeypatch.setattr(trainer, "sample_images", lambda *_args, **_kwargs: pytest.fail("legacy sampling ran"))
    monkeypatch.setattr(trainer_base, "clean_memory_on_device", lambda _device: None)
    monkeypatch.setattr(trainer_base.sai_model_spec, "build_metadata", lambda *_args, **_kwargs: {})
    dataset = SimpleNamespace(batch_size=1, get_metadata=lambda: {})
    trainer._run_training_loop(
        args, accelerator, "cpu-session", 0.0,
        SimpleNamespace(datasets=[dataset], num_train_items=1), [{"latents": torch.ones(1)}],
        SimpleNamespace(value=0), transformer, network, network, optimizer, "SGD", [],
        lambda: None, lambda: None, scheduler, [], _SampleVAE(), [{"prompt": "fixture", "enum": 0}],
        torch.float32, torch.float32,
    )
    assert len(calls) == 1
    step, reasons, metrics = calls[0]
    assert step == 1 and set(reasons) == {"periodic", "epoch", "final", "new_best", "sample"}
    assert len(reasons) == len(set(reasons)) and metrics == METRICS
    assert not list((tmp_path / "output").rglob("*.safetensors"))


def test_epoch_package_uses_last_completed_step_after_skipped_final_attempt(tmp_path, monkeypatch):
    class OptionalArgs(SimpleNamespace):
        def __getattr__(self, _name):
            return None

    class LoopNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def on_epoch_start(self, _transformer):
            pass

        def on_step_start(self):
            pass

    class LoopAccelerator:
        is_main_process = True
        is_local_main_process = True
        num_processes = 1
        device = torch.device("cpu")
        sync_gradients = True
        optimizer_step_was_skipped = False
        trackers = []
        attempts = 0

        @contextmanager
        def accumulate(self, _model):
            self.attempts += 1
            self.optimizer_step_was_skipped = self.attempts == 2
            yield

        def backward(self, loss):
            loss.backward()

        def unwrap_model(self, model):
            return model

        def init_trackers(self, *_args, **_kwargs):
            pass

        def print(self, *_args):
            pass

        def log(self, *_args, **_kwargs):
            pass

        def wait_for_everyone(self):
            pass

        def end_training(self):
            pass

    args = OptionalArgs(**vars(qwen_image_setup_parser(setup_parser_common()).parse_args([])))
    args.experiment_dir = str(tmp_path)
    args.output_dir = str(tmp_path / "output")
    args.output_name = "tiny-qwen"
    args.val_dataset_config = str(tmp_path / "val-dataset.toml")
    args.val_every_n_steps = 10
    args.gradient_accumulation_steps = 1
    args.max_train_steps = 2
    args.save_every_n_steps = None
    args.save_every_n_epochs = 1
    args.sample_prompts = "fixture-prompts.txt"
    args.sample_every_n_steps = None
    args.sample_every_n_epochs = 1
    args.sample_at_first = False
    args.save_precision = "fp32"
    args.no_metadata = True
    args.max_grad_norm = 0.0
    network, transformer, accelerator = LoopNetwork(), _SampleTransformer(), LoopAccelerator()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1)
    actual_optimizer_step = optimizer.step
    optimizer_steps = []

    def step_unless_skipped(*step_args, **step_kwargs):
        if not accelerator.optimizer_step_was_skipped:
            optimizer_steps.append(accelerator.attempts)
            return actual_optimizer_step(*step_args, **step_kwargs)

    monkeypatch.setattr(optimizer, "step", step_unless_skipped)
    trainer = QwenImageNetworkTrainer()
    trainer.validation_manifest = _manifest()
    calls = []
    monkeypatch.setattr(
        trainer, "_run_experiment_package_event",
        lambda *_args, **kwargs: calls.append((kwargs["absolute_step"], list(kwargs["reasons"]))),
    )
    monkeypatch.setattr(trainer, "evaluate_validation_event", lambda *_args: None)
    monkeypatch.setattr(trainer, "process_batch", lambda *_args: (network.weight.square(), {}))
    monkeypatch.setattr(trainer, "generate_step_logs", lambda *_args: {})
    monkeypatch.setattr(trainer_base, "clean_memory_on_device", lambda _device: None)
    monkeypatch.setattr(trainer_base.sai_model_spec, "build_metadata", lambda *_args, **_kwargs: {})
    dataset = SimpleNamespace(batch_size=1, get_metadata=lambda: {})
    trainer._run_training_loop(
        args, accelerator, "cpu-session", 0.0,
        SimpleNamespace(datasets=[dataset], num_train_items=2),
        [{"latents": torch.ones(1)}, {"latents": torch.ones(1)}],
        SimpleNamespace(value=0), transformer, network, network, optimizer, "SGD", [],
        lambda: None, lambda: None, scheduler, [], _SampleVAE(), [{"prompt": "fixture", "enum": 0}],
        torch.float32, torch.float32,
    )
    assert optimizer_steps == [1, 3]
    assert [reasons for step, reasons in calls if step == 1] == [["epoch", "sample"]]
    assert len(calls) == 2 and calls[1][0] == 2


def test_skipped_epoch_end_adds_reasons_to_existing_periodic_package(tmp_path, monkeypatch, request):
    class OptionalArgs(SimpleNamespace):
        def __getattr__(self, _name):
            return None

    class StopAfterFirstEpoch(Exception):
        pass

    class LoopNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def on_epoch_start(self, _transformer):
            pass

        def on_step_start(self):
            pass

    class LoopAccelerator:
        is_main_process = True
        is_local_main_process = True
        num_processes = 1
        device = torch.device("cpu")
        sync_gradients = True
        optimizer_step_was_skipped = False
        trackers = []
        attempts = 0

        @contextmanager
        def accumulate(self, _model):
            self.attempts += 1
            self.optimizer_step_was_skipped = self.attempts == 2
            yield

        def backward(self, loss):
            loss.backward()

        def unwrap_model(self, model):
            return model

        def init_trackers(self, *_args, **_kwargs):
            pass

        def print(self, *_args):
            pass

        def wait_for_everyone(self):
            pass

    defaults = vars(qwen_image_setup_parser(setup_parser_common()).parse_args([]))
    defaults.update(vars(_args(tmp_path)))
    args = OptionalArgs(**defaults)
    args.gradient_accumulation_steps = 1
    args.max_train_steps = 2
    args.save_every_n_steps = 1
    args.save_every_n_epochs = 1
    args.sample_prompts = "fixture-prompts.txt"
    args.sample_every_n_epochs = 1
    args.save_precision = "fp32"
    args.no_metadata = True
    args.max_grad_norm = 0.0
    real_accelerator = Accelerator(cpu=True, mixed_precision="no")
    request.addfinalizer(real_accelerator.free_memory)
    network = LoopNetwork()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1)
    network, optimizer, scheduler = real_accelerator.prepare(network, optimizer, scheduler)
    trainer = _register(real_accelerator, network, args, _manifest())
    sample_generations = []

    def create_sample(staging):
        sample_generations.append(1)
        Image.new("RGB", (8, 8), (12, 34, 56)).save(staging / "samples" / "sample.png")
        return 1

    states = _experiment_states()
    states.save_package(
        args, real_accelerator, network, trainer.validation_manifest, 1, None, ["periodic"],
        samples_enabled=True, sample_callback=create_sample,
    )
    package = _package(tmp_path, "current", 1)
    sample_bytes = (package / "samples" / "sample.png").read_bytes()
    monkeypatch.setattr(real_accelerator, "save_state", lambda *_args, **_kwargs: pytest.fail("saved state twice"))
    accelerator = LoopAccelerator()
    actual_optimizer_step = optimizer.step
    completed_updates = []

    def step_unless_skipped(*step_args, **step_kwargs):
        if not accelerator.optimizer_step_was_skipped:
            completed_updates.append(accelerator.attempts)
            return actual_optimizer_step(*step_args, **step_kwargs)

    monkeypatch.setattr(optimizer, "step", step_unless_skipped)
    package_calls = []
    monkeypatch.setattr(
        trainer, "_run_experiment_package_event",
        lambda *_args, **kwargs: package_calls.append((kwargs["absolute_step"], list(kwargs["reasons"]))),
    )
    monkeypatch.setattr(trainer, "evaluate_validation_event", lambda *_args: None)
    monkeypatch.setattr(trainer, "sample_images", lambda *_args, **_kwargs: pytest.fail("generated samples twice"))

    def process_batch(*_args):
        if accelerator.attempts == 3:
            raise StopAfterFirstEpoch()
        return network.weight.square(), {}

    monkeypatch.setattr(trainer, "process_batch", process_batch)
    monkeypatch.setattr(trainer_base, "clean_memory_on_device", lambda _device: None)
    monkeypatch.setattr(trainer_base.sai_model_spec, "build_metadata", lambda *_args, **_kwargs: {})
    dataset = SimpleNamespace(batch_size=1, get_metadata=lambda: {})
    with pytest.raises(StopAfterFirstEpoch):
        trainer._run_training_loop(
            args, accelerator, "cpu-session", 0.0,
            SimpleNamespace(datasets=[dataset], num_train_items=2),
            [{"latents": torch.ones(1)}, {"latents": torch.ones(1)}],
            SimpleNamespace(value=0), _SampleTransformer(), network, network, optimizer, "SGD", [],
            lambda: None, lambda: None, scheduler, [], _SampleVAE(), [{"prompt": "fixture", "enum": 0}],
            torch.float32, torch.float32,
        )
    assert completed_updates == [1]
    assert package_calls == [(1, ["periodic"])]
    assert sample_generations == [1]
    assert (package / "samples" / "sample.png").read_bytes() == sample_bytes
    assert len(list((tmp_path / "output").rglob("tiny-qwen-step-1"))) == 1
    metadata = json.loads((package / "experiment_state.json").read_text(encoding="utf-8"))
    assert set(metadata["save_reasons"]) == {"periodic", "epoch", "sample"}


@pytest.mark.parametrize("damage", ["one-png", "all-png", "samples-directory"])
def test_published_sample_package_rejects_removed_pngs_before_accelerator_load(package_context, monkeypatch, damage):
    context = package_context

    def create_two_samples(staging):
        for index in range(2):
            Image.new("RGB", (8, 8), (index * 60, 20, 30)).save(staging / "samples" / f"sample-{index}.png")
        return 2

    package = context.states.save_package(
        context.args, context.accelerator, context.network, context.manifest,
        5, None, ["sample"], samples_enabled=True, sample_callback=create_two_samples,
    )
    samples = package / "samples"
    assert len(list(samples.glob("*.png"))) == 2
    if damage == "samples-directory":
        shutil.rmtree(samples)
    else:
        images = sorted(samples.glob("*.png"))
        for image in images[:1] if damage == "one-png" else images:
            image.unlink()
    monkeypatch.setattr(context.accelerator, "load_state", lambda *_a, **_kw: pytest.fail("load_state preceded sample integrity preflight"))
    with pytest.raises((ValueError, RuntimeError), match="sample|PNG|incomplete"):
        context.states.load_package(context.args, context.accelerator, context.network, context.manifest, package)
