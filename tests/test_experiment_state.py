"""Actual CPU Accelerate canonical adapter and resumable state round trips."""

from argparse import Namespace
import copy
import json
import random

from accelerate import Accelerator
import numpy as np
import pytest
from safetensors import safe_open
from safetensors.torch import load_file
import torch

from musubi_tuner.networks import lora_qwen_image
from musubi_tuner.qwen_image.qwen_image_model import QwenImageTransformerBlock
from musubi_tuner.training.experiment_state import (
    adapter_identity,
    checkpoint_directory,
    prepare_training_state,
    record_validation,
    register_experiment_hooks,
    save_experiment_checkpoint,
    validate_experiment_checkpoint,
)


def arguments(root, **overrides):
    values = dict(
        experiment_mode=True,
        val_dataset_config=None,
        resume=None,
        resume_from_huggingface=False,
        _experiment_root=str(root),
        output_dir=str(root / "output"),
        output_name="qwen",
        logging_dir=None,
        log_prefix=None,
        log_tracker_name=None,
        save_last_n_steps=None,
        save_last_n_steps_state=None,
    )
    values.update(overrides)
    return Namespace(**values)


def tiny_adapter(alpha=2):
    base = torch.nn.Sequential(QwenImageTransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4))
    base.requires_grad_(False)
    network = lora_qwen_image.create_arch_network(1, 2, alpha, None, [], base)
    network.apply_to([], base, apply_text_encoder=False, apply_unet=True)
    return base, network


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_equal(left, right)
    else:
        assert a == b


def update(optimizer, scheduler, network):
    for parameter in network.parameters():
        parameter.grad = torch.ones_like(parameter) * 0.25
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def test_actual_accelerate_one_fp32_adapter_roundtrip_rng_metadata_and_next_update(tmp_path):
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    base, network = tiny_adapter()
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8)
    base, network, optimizer, scheduler = accelerator.prepare(base, network, optimizer, scheduler)
    args = arguments(tmp_path)
    state = prepare_training_state(args, "fixture-validation")
    state["optimizer_global_step"] = 3
    register_experiment_hooks(args, accelerator, network, state)
    update(optimizer, scheduler, network)
    record_validation(
        state, 3, {"val_loss_mean": 2.0, "val_loss_low_noise": 1.0, "val_loss_high_noise": 3.0}, adapter_identity(network)
    )
    expected_validation = copy.deepcopy(state["last_validation"])
    random.seed(73), np.random.seed(73), torch.manual_seed(73)
    before = copy.deepcopy(
        (
            network.state_dict(),
            optimizer.state_dict(),
            scheduler.state_dict(),
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
    )
    projection = network.unet_loras[0].org_forward.__self__
    sample = torch.ones(2, projection.in_features)
    forward_before = projection(sample).detach().clone()
    directory = save_experiment_checkpoint(args, accelerator, network, state, {"ss_steps": "3"}, with_state=True)
    manifest = validate_experiment_checkpoint(directory)
    assert directory == checkpoint_directory(args, 3)
    assert manifest["with_state"] and manifest["process_count"] == 1
    assert {path.name for path in directory.glob("*.safetensors")} == {"model.safetensors"}
    tensors = load_file(str(directory / "model.safetensors"))
    assert tensors.keys() == network.state_dict().keys()
    assert all(tensor.dtype == torch.float32 for tensor in tensors.values())
    assert {"optimizer.bin", "scheduler.bin", "random_states_0.pkl", "trainer_state.json"} <= set(manifest["files"])
    assert not list(directory.glob("pytorch_model*"))
    with safe_open(str(directory / "model.safetensors"), framework="pt") as opened:
        assert opened.metadata()["ss_steps"] == "3"
    update(optimizer, scheduler, network)
    expected_next = copy.deepcopy((network.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
    for module in network.unet_loras:
        module.alpha.fill_(1)
        module.scale = 0.5
    state["optimizer_global_step"] = 999
    random.random(), np.random.random(), torch.rand(2)
    accelerator.load_state(str(directory))
    after = (
        network.state_dict(),
        optimizer.state_dict(),
        scheduler.state_dict(),
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    assert_equal(before, after)
    assert state["optimizer_global_step"] == 3
    assert state["last_validation"] == expected_validation
    torch.testing.assert_close(projection(sample), forward_before, rtol=0, atol=0)
    update(optimizer, scheduler, network)
    assert_equal(expected_next, (network.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
    accelerator.free_memory()


def test_weights_only_upgrade_and_final_dedup_write_adapter_once(tmp_path, monkeypatch):
    accelerator = Accelerator(cpu=True)
    base, network = tiny_adapter()
    optimizer = torch.optim.AdamW(network.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    base, network, optimizer, scheduler = accelerator.prepare(base, network, optimizer, scheduler)
    args = arguments(tmp_path)
    state = prepare_training_state(args)
    state["optimizer_global_step"] = 2
    register_experiment_hooks(args, accelerator, network, state)
    saved = []
    original = network.save_weights

    def counted(*args, **kwargs):
        saved.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(network, "save_weights", counted)
    directory = save_experiment_checkpoint(args, accelerator, network, state, {}, with_state=False)
    assert not (directory / "trainer_state.json").exists()
    with pytest.raises(ValueError, match="state|resum"):
        validate_experiment_checkpoint(directory)
    save_experiment_checkpoint(args, accelerator, network, state, {}, with_state=True)
    assert validate_experiment_checkpoint(directory)["with_state"]
    save_experiment_checkpoint(args, accelerator, network, state, {}, with_state=True)
    save_experiment_checkpoint(args, accelerator, network, state, {}, with_state=False)
    assert len(saved) == 1
    accelerator.free_memory()


def test_partial_and_samples_only_are_not_resume_states(tmp_path):
    directory = tmp_path / "output/qwen-step3"
    (directory / "samples").mkdir(parents=True)
    (directory / "samples/image.png").write_bytes(b"sample")
    args = arguments(tmp_path, resume=str(directory))
    with pytest.raises(ValueError, match="complete|manifest"):
        prepare_training_state(args)


@pytest.mark.parametrize("name", ["model_1.safetensors", "pytorch_model.bin"])
def test_unowned_tensor_artifact_rejected_without_deleting_it(tmp_path, name):
    accelerator = Accelerator(cpu=True)
    base, network = tiny_adapter()
    args = arguments(tmp_path)
    state = prepare_training_state(args)
    directory = checkpoint_directory(args, 0)
    (directory / "samples").mkdir(parents=True)
    foreign = directory / name
    foreign.write_bytes(b"user tensor artifact")
    with pytest.raises(ValueError, match="unexpected tensor artifact"):
        save_experiment_checkpoint(args, accelerator, network, state, {}, with_state=False)
    assert foreign.read_bytes() == b"user tensor artifact"
    foreign.unlink()
    save_experiment_checkpoint(args, accelerator, network, state, {}, with_state=False)
    foreign.write_bytes(b"user tensor artifact")
    with pytest.raises(ValueError, match="unexpected tensor artifact"):
        validate_experiment_checkpoint(directory, require_state=False)
    assert foreign.read_bytes() == b"user tensor artifact"
    accelerator.free_memory()


def test_weights_only_precision_honored_and_lossless_state_requires_fp32(tmp_path):
    accelerator = Accelerator(cpu=True)
    base, network = tiny_adapter()
    optimizer = torch.optim.AdamW(network.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    base, network, optimizer, scheduler = accelerator.prepare(base, network, optimizer, scheduler)
    args = arguments(tmp_path, save_precision="bf16")
    state = prepare_training_state(args)
    register_experiment_hooks(args, accelerator, network, state)
    directory = save_experiment_checkpoint(args, accelerator, network, state, {}, with_state=False)
    assert all(tensor.dtype == torch.bfloat16 for tensor in load_file(str(directory / "model.safetensors")).values())
    with pytest.raises(ValueError, match="FP32"):
        save_experiment_checkpoint(args, accelerator, network, state, {}, with_state=True)
    args.save_precision = "fp32"
    save_experiment_checkpoint(args, accelerator, network, state, {}, with_state=True)
    assert validate_experiment_checkpoint(directory)["with_state"]
    assert all(tensor.dtype == torch.float32 for tensor in load_file(str(directory / "model.safetensors")).values())
    accelerator.free_memory()


def test_missing_or_modified_inventory_file_rejected_before_models(tmp_path):
    accelerator = Accelerator(cpu=True)
    base, network = tiny_adapter()
    optimizer = torch.optim.AdamW(network.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    base, network, optimizer, scheduler = accelerator.prepare(base, network, optimizer, scheduler)
    args = arguments(tmp_path)
    state = prepare_training_state(args)
    register_experiment_hooks(args, accelerator, network, state)
    directory = save_experiment_checkpoint(args, accelerator, network, state, {}, with_state=True)
    (directory / "random_states_0.pkl").write_bytes(b"incomplete")
    args.resume = str(directory)
    with pytest.raises(ValueError, match="random_states_0.pkl|digest"):
        prepare_training_state(args)
    manifest = json.loads((directory / "checkpoint_manifest.json").read_text())
    manifest["files"]["../outside"] = {"sha256": "0" * 64, "size": 0}
    (directory / "checkpoint_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe|outside|file name"):
        validate_experiment_checkpoint(directory)
    accelerator.free_memory()
