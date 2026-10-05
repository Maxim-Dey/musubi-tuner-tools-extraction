from contextlib import nullcontext
import copy
from datetime import timedelta
import json
from pathlib import Path
import random
from types import SimpleNamespace

from accelerate import Accelerator
import numpy as np
import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

from musubi_tuner.networks import lora_qwen_image
from musubi_tuner.qwen_image.qwen_image_model import QwenImageTransformer2DModel
from musubi_tuner.qwen_image_train_network import QwenImageNetworkTrainer
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer
from musubi_tuner.training.validation import evaluate_validation


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def rng_state():
    return copy.deepcopy((random.getstate(), np.random.get_state(), torch.get_rng_state()))


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


class Inputs:
    def __init__(self, fail=False):
        self.records = [SimpleNamespace(image_sha256="f" * 64), SimpleNamespace(image_sha256="0" * 64)]
        self.loads = []
        self.verified = 0
        self.fail = fail

    def verify_unchanged(self):
        self.verified += 1
        # Deliberate setup randomness must not reach training.
        random.random(), np.random.rand(), torch.rand(3)

    def load_batch(self, index):
        self.loads.append(index)
        random.random(), np.random.rand(), torch.rand(3)
        if self.fail:
            raise ValueError("fixture cache read failure")
        width = 4 if index else 6
        return {"latents": torch.ones(1, 2, 1, 4, width) * 0.2, "vl_embed": [torch.ones(3, 8)], "timesteps": None}


def arguments(n1=2, n2=2):
    return SimpleNamespace(
        split_attn=False,
        gradient_checkpointing=True,
        weighting_scheme="none",
        val_seed_noise=42,
        val_level_noise_n=n1,
        val_seed_noise_n=n2,
        timestep_sampling="uniform",
        min_timestep=None,
        max_timestep=None,
        preserve_distribution_shape=False,
    )


def tiny_qwen():
    model = QwenImageTransformer2DModel(
        in_channels=8,
        out_channels=2,
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=8,
        joint_attention_dim=8,
        axes_dims_rope=(2, 2, 4),
    )
    model.requires_grad_(False)
    model.enable_gradient_checkpointing()
    network = lora_qwen_image.create_arch_network(
        1, 2, 2, None, [], model, neuron_dropout=0.4, rank_dropout="0.2", module_dropout="0.1"
    )
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    for adapter in network.unet_loras:
        torch.nn.init.constant_(adapter.lora_up.weight, 0.03)
    model.train()
    network.train()
    return model, network


@pytest.mark.parametrize("n1,n2", [(2, 1), (10, 2)])
def test_actual_qwen_lora_stream_count_repeatability_and_modes(n1, n2):
    model, network = tiny_qwen()
    # Preserve mixed nested modes, not only the two roots.
    model.txt_norm.eval()
    network.unet_loras[0].eval()
    modes = [module.training for root in (model, network) for module in root.modules()]
    parameters = list(network.parameters())
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 0.125)
    before = copy.deepcopy((model.state_dict(), network.state_dict(), [p.grad for p in parameters], rng_state()))
    inputs = Inputs()
    observed = []

    def inspect_forward(module, args, kwargs):
        assert not torch.is_grad_enabled()
        assert not any(child.training for root in (model, network) for child in root.modules())
        observed.append(kwargs["timestep"].item())
        random.random(), np.random.rand(), torch.rand(1)

    handle = model.register_forward_pre_hook(inspect_forward, with_kwargs=True)
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    trainer = QwenImageNetworkTrainer()
    result = evaluate_validation(
        trainer, arguments(n1, n2), accelerator, model, network, inputs, None, torch.float32, torch.float32
    )
    assert inputs.loads == [1, 0] and inputs.verified == 1
    assert result.image_count == 2 and result.forward_count == len(observed) == 2 * n1 * n2
    assert result.loss_mean == pytest.approx((result.loss_low_noise + result.loss_high_noise) / 2, abs=1e-7)
    repeated = evaluate_validation(
        trainer, arguments(n1, n2), accelerator, model, network, inputs, None, torch.float32, torch.float32
    )
    assert result == repeated
    assert modes == [module.training for root in (model, network) for module in root.modules()]
    assert_equal(before, (model.state_dict(), network.state_dict(), [p.grad for p in parameters], rng_state()))
    handle.remove()


@pytest.mark.parametrize("failure", ["read", "forward"])
def test_error_restores_rng_nested_modes_and_offloader(failure):
    model, network = tiny_qwen()
    model.txt_norm.eval()
    modes = [module.training for root in (model, network) for module in root.modules()]
    transitions = []

    class Offloader:
        forward_only = False

        def set_forward_only(self, value):
            self.forward_only = value
            transitions.append(value)

    model.blocks_to_swap = 1
    model.offloader = Offloader()
    model.prepare_block_swap_before_forward = lambda: None

    def fail_forward(*args, **kwargs):
        assert model.offloader.forward_only
        random.random(), np.random.rand(), torch.rand(3)
        raise ValueError("fixture model forward failure")

    model.forward = fail_forward
    before = rng_state()
    with pytest.raises((ValueError, RuntimeError), match="fixture"):
        evaluate_validation(
            QwenImageNetworkTrainer(),
            arguments(),
            Accelerator(cpu=True),
            model,
            network,
            Inputs(fail=failure == "read"),
            None,
            torch.float32,
            torch.float32,
        )
    assert_equal(before, rng_state())
    assert modes == [module.training for root in (model, network) for module in root.modules()]
    assert model.offloader.forward_only is False and transitions == [True, False]
    assert model.gradient_checkpointing


def test_validation_preserves_real_next_train_update_optimizer_scheduler_and_gradients():
    model, network = tiny_qwen()
    trainer, args, accelerator = QwenImageNetworkTrainer(), arguments(), Accelerator(cpu=True, mixed_precision="no")
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    batch = Inputs().load_batch(0)

    def update():
        optimizer.zero_grad(set_to_none=True)
        noise = torch.randn_like(batch["latents"])
        loss, _ = trainer.process_batch(
            args,
            accelerator,
            model,
            network,
            batch,
            trainer.scale_shift_latents(batch["latents"]),
            noise,
            None,
            torch.float32,
            torch.float32,
            None,
            1,
        )
        loss.backward()
        gradients = [parameter.grad.clone() if parameter.grad is not None else None for parameter in network.parameters()]
        optimizer.step()
        scheduler.step()
        return copy.deepcopy(
            (noise, loss.detach(), gradients, network.state_dict(), optimizer.state_dict(), scheduler.state_dict(), rng_state())
        )

    update()  # Initialize actual AdamW state before the branching comparison.
    initial = copy.deepcopy((network.state_dict(), optimizer.state_dict(), scheduler.state_dict(), rng_state()))
    control = update()
    network.load_state_dict(initial[0])
    optimizer.load_state_dict(initial[1])
    scheduler.load_state_dict(initial[2])
    restore_rng(initial[3])
    optimizer.zero_grad(set_to_none=True)
    evaluate_validation(trainer, args, accelerator, model, network, Inputs(), None, torch.float32, torch.float32, global_step=1)
    with_validation = update()
    assert_equal(control, with_validation)


class DistributedAccelerator:
    device = torch.device("cpu")
    num_processes = 2

    def __init__(self, rank):
        self.is_main_process = rank == 0

    def wait_for_everyone(self):
        distributed.barrier()

    def unwrap_model(self, model):
        return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    def autocast(self):
        return nullcontext()


class DistributedTrainer(NetworkTrainer):
    calls = 0

    def scale_shift_latents(self, latents):
        return latents

    def call_dit(self, args, accelerator, transformer, latents, batch, noise, noisy, timesteps, network_dtype):
        self.calls += 1
        return DiTOutput(pred=transformer(noisy.unsqueeze(-1)).squeeze(-1), target=noise - latents)


def distributed_worker(rank, rendezvous, output):
    distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=40))
    try:
        torch.manual_seed(90)
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1, 1))
        network = torch.nn.Identity()
        trainer, accelerator = DistributedTrainer(), DistributedAccelerator(rank)
        before = rng_state()
        result = evaluate_validation(
            trainer, arguments(), accelerator, model, network, Inputs(), None, torch.float32, torch.float32
        )
        assert_equal(before, rng_state())
        error = ""
        try:
            evaluate_validation(
                trainer, arguments(), accelerator, model, network, Inputs(fail=True), None, torch.float32, torch.float32
            )
        except RuntimeError as caught:
            error = str(caught)
        assert "fixture cache read failure" in error
        assert_equal(before, rng_state())
        Path(output, f"rank{rank}.json").write_text(
            json.dumps({"calls": trainer.calls, "metrics": result.metrics, "count": result.forward_count, "error": error}),
            encoding="utf-8",
        )
    finally:
        distributed.destroy_process_group()


def test_actual_two_process_gloo_exact_count_results_rng_and_errors(tmp_path):
    rendezvous = (tmp_path / "rendezvous").resolve().as_uri()
    multiprocessing.spawn(distributed_worker, args=(rendezvous, str(tmp_path)), nprocs=2, join=True)
    results = [json.loads((tmp_path / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    assert results[0]["calls"] == results[0]["count"] == 8
    assert results[1]["calls"] == 0 and results[1]["count"] == 8
    assert results[0]["metrics"] == results[1]["metrics"]
    assert results[0]["error"] == results[1]["error"]
