from contextlib import nullcontext
from types import SimpleNamespace
import math
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from musubi_tuner.training.krea2_validation import (
    fixed_noise, fixed_weight, noise_levels, noise_seed, run_validation, validation_context,
)


def test_levels_and_cross_process_seed_vector():
    assert noise_levels(2) == pytest.approx([0.275, 0.725], abs=1e-12)
    assert noise_levels(10) == pytest.approx([.095, .185, .275, .365, .455, .545, .635, .725, .815, .905], abs=1e-12)
    assert noise_seed(42, "0" * 64, 1, 1) == 2528747074945454131
    script = "from musubi_tuner.training.krea2_validation import noise_seed; print(noise_seed(42,'0'*64,1,1))"
    assert subprocess.check_output([sys.executable, "-c", script], text=True).strip() == "2528747074945454131"
    first = fixed_noise((2, 3), noise_seed(42, "a", 1, 1))
    torch.randn(100)
    assert torch.equal(first, fixed_noise((2, 3), noise_seed(42, "a", 1, 1)))
    assert not torch.equal(first, fixed_noise((2, 3), noise_seed(42, "a", 1, 2)))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.dropout = torch.nn.Dropout(.9)
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = True
        self.offloader = SimpleNamespace(forward_only=False)
        self.offloader.set_forward_only = lambda flag: setattr(self.offloader, "forward_only", flag)
        self.prepared = 0

    def prepare_block_swap_before_forward(self):
        self.prepared += 1


class Inputs:
    def __init__(self, records):
        self.records = records

    def validate_unchanged(self):
        pass

    def load_batch(self, record):
        return {"latents": torch.full(record.shape, record.value), "krea2_vl_embed": [torch.zeros(1, 1, 1)]}


def accelerator(rank=0, world=1, reduce=None):
    return SimpleNamespace(device=torch.device("cpu"), process_index=rank, num_processes=world,
                           unwrap_model=lambda x: x, reduce=reduce or (lambda value, reduction: value), autocast=nullcontext)


@pytest.mark.parametrize("scheme", ["none", "sigma_sqrt", "cosmap", "logit_normal", "mode"])
def test_direct_math_equal_image_means_and_reorder(scheme):
    records = [SimpleNamespace(image_sha256="a", shape=(1, 1, 1, 1, 1), value=1.),
               SimpleNamespace(image_sha256="b", shape=(1, 1, 1, 3, 4), value=3.)]
    model, network = Model(), Model()
    args = SimpleNamespace(val_level_noise_n=2, val_seed_noise_n=2, val_seed_noise=42, weighting_scheme=scheme)

    def forward(args, acc, model, latents, batch, noise, noisy, timesteps, dtype):
        t = timesteps.item() / 1000
        assert torch.allclose(noisy, (1 - t) * latents + t * noise, rtol=1e-5, atol=1e-6)
        assert not torch.is_grad_enabled() and not model.training and not network.training
        return SimpleNamespace(pred=noisy * model.weight, target=noise - latents)

    trainer = SimpleNamespace(call_dit=forward)
    actual = run_validation(trainer, args, accelerator(), model, network, Inputs(records), torch.float32)
    reference = []
    for record in records:
        values = []
        for i, t in enumerate((.275, .725), 1):
            for j in (1, 2):
                # Independent arithmetic in float64; noise itself stays exact FP32.
                noise = fixed_noise(record.shape, noise_seed(42, record.image_sha256, i, j)).double()
                x = (1 - t) * record.value + t * noise
                mse = ((.5 * x - (noise - record.value)) ** 2).mean().item()
                weight = 1 if scheme not in ("sigma_sqrt", "cosmap") else 1/t**2 if scheme == "sigma_sqrt" else 2/(math.pi*((1-t)**2+t**2))
                values.append(mse * weight)
        reference.append([sum(values)/4, sum(values[:2])/2, sum(values[2:])/2])
    expected = np.mean(reference, axis=0)
    np.testing.assert_allclose(list(actual.values()), expected, rtol=1e-5, atol=1e-6)
    assert actual["val_loss_mean"] == pytest.approx((actual["val_loss_low_noise"]+actual["val_loss_high_noise"])/2, abs=1e-12)
    reordered = run_validation(trainer, args, accelerator(), model, network, Inputs(records[::-1]), torch.float32)
    assert actual == reordered


@pytest.mark.parametrize("raises", [False, True])
def test_context_restores_rng_modes_existing_gradients_and_optimizer(raises):
    model, network = Model(), Model()
    model.dropout.eval()
    network.eval()
    model.weight.grad = torch.tensor(7.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    optimizer_before, scheduler_before = optimizer.state_dict(), scheduler.state_dict()
    modes = [m.training for root in (model, network) for m in root.modules()]
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    try:
        with validation_context(model, network):
            assert not model.gradient_checkpointing and model.offloader.forward_only
            random.random(), np.random.rand(), torch.rand(10)
            if raises:
                raise RuntimeError("injected")
    except RuntimeError:
        assert raises
    assert modes == [m.training for root in (model, network) for m in root.modules()]
    assert model.gradient_checkpointing and model.activation_cpu_offloading and not model.offloader.forward_only
    assert model.weight.item() == .5 and model.weight.grad.item() == 7
    assert optimizer.state_dict() == optimizer_before and scheduler.state_dict() == scheduler_before
    assert python == random.getstate()
    assert np.array_equal(numpy[1], np.random.get_state()[1])
    assert torch.equal(cpu, torch.get_rng_state())


@pytest.mark.parametrize("world", [2,5])
def test_empty_and_unequal_ranks_reduce_sums_not_rank_means(world):
    records = [SimpleNamespace(image_sha256=str(i), shape=(1, 1, 1, 1, 1), value=float(i)) for i in range(3)]
    args = SimpleNamespace(val_level_noise_n=2, val_seed_noise_n=1, val_seed_noise=42, weighting_scheme="none")
    trainer = SimpleNamespace(call_dit=lambda a, ac, m, lat, b, n, x, t, dt: SimpleNamespace(pred=lat, target=torch.zeros_like(lat)))
    partials = []
    for rank in range(world):
        def capture(value, reduction):
            partials.append(value.clone())
            return value.new_tensor([5., 5., 5., 3.])
        result = run_validation(trainer, args, accelerator(rank, world, capture), Model(), Model(), Inputs(records), torch.float32)
        assert result["val_loss_mean"] == pytest.approx(5/3)
    assert torch.equal(sum(partials), torch.tensor([5., 5., 5., 3.], dtype=torch.float64))
    if world == 5:
        assert partials[-1].tolist() == [0., 0., 0., 0.]
    else:
        assert [p[-1].item() for p in partials] == [2,1]


def test_actual_krea2_forward_does_not_enable_gradients_during_validation():
    from musubi_tuner.krea2_train_network import Krea2NetworkTrainer

    class DiT(torch.nn.Module):
        config = SimpleNamespace(patch=2)
        def forward(self, **inputs):
            assert not inputs["img"].requires_grad and not inputs["context"].requires_grad
            assert inputs["t"].item() == pytest.approx(.275, abs=1e-7)
            assert inputs["mask"].tolist() == [[True, True, True]]
            return torch.zeros_like(inputs["img"])

    latents = torch.ones(1,16,1,2,2)
    batch = {"latents":latents,"krea2_vl_embed":[torch.ones(2,2,4)]}
    with torch.no_grad():
        result = Krea2NetworkTrainer().call_dit(SimpleNamespace(gradient_checkpointing=True), accelerator(), DiT(),
            latents,batch,torch.zeros_like(latents),latents*.725,torch.tensor([275.]),torch.float32)
    assert torch.equal(result.target,-latents) and not result.pred.requires_grad
