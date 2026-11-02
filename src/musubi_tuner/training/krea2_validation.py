"""Fixed Krea2 validation inputs, noise and equal-image loss reduction."""

from contextlib import contextmanager
import hashlib
import json
import math
import random

import numpy as np
import torch


def noise_levels(count):
    return tuple(0.05 + (i - 0.5) * 0.90 / count for i in range(1, count + 1))


def noise_seed(seed, image_sha256, level, realization):
    payload = json.dumps([seed, image_sha256, level, realization], separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def fixed_noise(shape, seed, *, dtype=torch.float32):
    # CPU/FP32 generation is independent of CUDA topology and autocast. Transfer
    # only the current realization; the manifest freezes this representation.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype, device="cpu")


def fixed_weight(scheme, t):
    if scheme == "sigma_sqrt":
        return t ** -2
    if scheme == "cosmap":
        return 2 / (math.pi * (1 - 2 * t + 2 * t * t))
    return 1.0


@contextmanager
def preserve_rng():
    python = random.getstate()
    numpy = np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


@contextmanager
def validation_context(transformer, network):
    # LoRA modules are not necessarily registered under the DiT, and mixed
    # train/eval submodule modes must be restored individually.
    modules = {id(m): m for root in (transformer, network) for m in root.modules()}
    modes = [(m, m.training) for m in modules.values()]
    checkpointing = getattr(transformer, "gradient_checkpointing", None)
    offloading = getattr(transformer, "activation_cpu_offloading", None)
    offloader = getattr(transformer, "offloader", None)
    forward_only = getattr(offloader, "forward_only", None)
    with preserve_rng():
        try:
            transformer.eval()
            network.eval()
            if checkpointing is not None:
                transformer.gradient_checkpointing = False
            if offloading is not None:
                transformer.activation_cpu_offloading = False
            if offloader is not None:
                offloader.set_forward_only(True)
                transformer.prepare_block_swap_before_forward()
            with torch.no_grad():
                yield
        finally:
            for module, training in modes:
                module.training = training
            if checkpointing is not None:
                transformer.gradient_checkpointing = checkpointing
            if offloading is not None:
                transformer.activation_cpu_offloading = offloading
            if offloader is not None:
                offloader.set_forward_only(forward_only)
                transformer.prepare_block_swap_before_forward()


def run_validation(trainer, args, accelerator, transformer, network, inputs, network_dtype):
    inputs.validate_unchanged()
    # Do not use DDP forward: different ranks can have different numbers of
    # images, including zero. Only the final sum/count is collective.
    transformer = accelerator.unwrap_model(transformer)
    network = accelerator.unwrap_model(network)
    totals = torch.zeros(4, dtype=torch.float64, device=accelerator.device)
    levels = noise_levels(args.val_level_noise_n)
    records = inputs.records[accelerator.process_index :: accelerator.num_processes]
    with validation_context(transformer, network):
        for record in records:
            batch = inputs.load_batch(record)
            latents = batch["latents"].to(device=accelerator.device, dtype=network_dtype)
            batch["latents"] = latents
            image_sum = low_sum = high_sum = 0.0
            for i, t in enumerate(levels, 1):
                for j in range(1, args.val_seed_noise_n + 1):
                    noise = fixed_noise(latents.shape, noise_seed(args.val_seed_noise, record.image_sha256, i, j))
                    noise = noise.to(device=latents.device, dtype=latents.dtype)
                    noisy = (1 - t) * latents + t * noise
                    timesteps = torch.full((latents.shape[0],), 1000 * t, device=latents.device, dtype=torch.float32)
                    output = trainer.call_dit(
                        args, accelerator, transformer, latents, batch, noise, noisy, timesteps, network_dtype
                    )
                    loss = (output.pred.float() - output.target.float()).square().mean().item() * fixed_weight(args.weighting_scheme, t)
                    image_sum += loss
                    if t < 0.5:
                        low_sum += loss
                    else:
                        high_sum += loss
            combinations = len(levels) * args.val_seed_noise_n
            totals += totals.new_tensor([image_sum / combinations, low_sum * 2 / combinations, high_sum * 2 / combinations, 1])
        totals = accelerator.reduce(totals, reduction="sum")
    if totals[3].item() == 0:
        raise ValueError("Validation contains no unique images")
    values = (totals[:3] / totals[3]).tolist()
    return dict(zip(("val_loss_mean", "val_loss_low_noise", "val_loss_high_noise"), values))
