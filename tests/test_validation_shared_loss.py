"""Validation crosses the same original-Qwen forward and weighted loss boundaries."""

from types import SimpleNamespace

from accelerate import Accelerator
import pytest
import torch

from musubi_tuner.qwen_image_train_network import QwenImageNetworkTrainer


@pytest.mark.parametrize("weighting", ["none", "sigma_sqrt", "cosmap"])
def test_fixed_sigma_shared_forward_target_loss_without_autograd(weighting):
    trainer = QwenImageNetworkTrainer()
    args = SimpleNamespace(gradient_checkpointing=True, split_attn=False, weighting_scheme=weighting)
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    latents = torch.arange(32, dtype=torch.float32).reshape(1, 2, 1, 4, 4) / 100
    noise = torch.ones_like(latents)
    sigma = torch.tensor([0.275], dtype=torch.float32)
    mixed = (1 - sigma) * latents + sigma * noise
    captured = {}

    def forward(**kwargs):
        captured.update(kwargs)
        assert not torch.is_grad_enabled()
        assert not kwargs["hidden_states"].requires_grad
        assert not kwargs["encoder_hidden_states"].requires_grad
        return kwargs["hidden_states"] * 2

    with torch.no_grad():
        output = trainer.call_dit(
            args,
            accelerator,
            forward,
            latents,
            {"latents": latents, "vl_embed": [torch.ones(2, 8)]},
            noise,
            mixed,
            sigma * 1000,
            torch.float32,
        )
        # A scheduler with deliberately incompatible shifted values must be ignored.
        scheduler = SimpleNamespace(sigmas=torch.tensor([0.99]), timesteps=torch.tensor([999.0]))
        loss, metrics = trainer.compute_loss(
            args, output, sigma * 1000, scheduler, torch.bfloat16, torch.float32, 0, fixed_sigmas=sigma
        )
    torch.testing.assert_close(captured["timestep"], sigma, rtol=0, atol=0)
    torch.testing.assert_close(output.target, noise - latents, rtol=0, atol=0)
    expected = (2 * mixed - (noise - latents)).square()
    if weighting == "sigma_sqrt":
        expected *= sigma**-2
    elif weighting == "cosmap":
        expected *= 2 / (torch.pi * (1 - 2 * sigma + 2 * sigma**2))
    torch.testing.assert_close(loss, expected.mean(), rtol=1e-6, atol=1e-7)
    assert metrics == {}
