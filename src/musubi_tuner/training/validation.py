"""Deterministic Qwen validation primitives; training uses its existing loss hook."""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import random

import numpy as np
import torch


def _integer(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int) or (minimum is not None and value < minimum):
        suffix = "" if minimum is None else f" >= {minimum}"
        raise ValueError(f"{name}: expected an integer{suffix} (not bool)")
    return value


def validation_noise_levels(count):
    """Final mixing coefficients, without the training shift or scheduler grid."""
    _integer(count, "val_level_noise_n", 2)
    if count % 2:
        raise ValueError("val_level_noise_n: expected an even integer >= 2")
    return tuple(0.05 + (i - 0.5) * 0.90 / count for i in range(1, count + 1))


def stable_noise_seed(seed, image_sha256, i, j):
    _integer(seed, "val_seed_noise")
    _integer(i, "noise level index", 1)
    _integer(j, "noise realization index", 1)
    image_sha256 = image_sha256.lower()
    if len(image_sha256) != 64 or any(character not in "0123456789abcdef" for character in image_sha256):
        raise ValueError("image_sha256: expected a SHA-256 hexadecimal digest")
    serialized = json.dumps([seed, image_sha256, i, j], ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return int.from_bytes(hashlib.sha256(serialized).digest()[:8], "big") % (2**63)


def validation_noise_like(latents, seed):
    """Generate one CPU FP32 realization, then cast to the cached latent dtype/device."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(latents.shape, generator=generator, device="cpu", dtype=torch.float32)
    return noise.to(device=latents.device, dtype=latents.dtype)


def fixed_validation_inputs(latents, noise, t):
    """Return mixed input, trainer-scale timestep and the exact fixed FP32 sigma."""
    sigmas = torch.full((latents.shape[0],), t, device=latents.device, dtype=torch.float32)
    coefficient = sigmas.reshape((-1,) + (1,) * (latents.ndim - 1))
    noisy = (1 - coefficient) * latents + coefficient * noise
    return noisy, sigmas * 1000, sigmas


@contextmanager
def preserve_rng_state():
    """Cover setup/cache reads as well as evaluation, restoring state even after errors."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@dataclass(frozen=True)
class ValidationResult:
    loss_mean: float
    loss_low_noise: float
    loss_high_noise: float
    image_count: int
    forward_count: int

    @property
    def metrics(self):
        return {
            "val_loss_mean": self.loss_mean,
            "val_loss_low_noise": self.loss_low_noise,
            "val_loss_high_noise": self.loss_high_noise,
        }


def reduce_validation_losses(image_losses, level_count, realization_count):
    """Consume scalar image losses in SHA/i/j order, with equal weight per image.

    Each image iterable yields (one-based level index, scalar training loss).
    Only detached Python scalars are retained; forwards/noise are streamed by the caller.
    """
    validation_noise_levels(level_count)
    _integer(realization_count, "val_seed_noise_n", 1)
    image_means = []
    low_means = []
    high_means = []
    for losses in image_losses:
        low, high = [], []
        counts = [0] * level_count
        for level, value in losses:
            _integer(level, "noise level index", 1)
            if level > level_count:
                raise ValueError("validation loss: level index exceeds val_level_noise_n")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("validation loss: non-finite scalar loss")
            counts[level - 1] += 1
            (low if level <= level_count // 2 else high).append(value)
        if counts != [realization_count] * level_count:
            raise ValueError("validation loss: every image must have exactly N1*N2 measurements, N2 per level")
        low_means.append(math.fsum(low) / (level_count // 2 * realization_count))
        high_means.append(math.fsum(high) / (level_count // 2 * realization_count))
        image_means.append(math.fsum(low + high) / (level_count * realization_count))
    count = len(image_means)
    if not count:
        raise ValueError("validation loss: the effective validation dataset is empty")
    return ValidationResult(
        math.fsum(image_means) / count,
        math.fsum(low_means) / count,
        math.fsum(high_means) / count,
        count,
        count * level_count * realization_count,
    )


@contextmanager
def validation_model_state(transformer, network):
    """Temporarily disable all dropout and backward-only block transfers."""
    modules = {id(module): module for root in (transformer, network) for module in root.modules()}
    modes = [(module, module.training) for module in modules.values()]
    offloader = getattr(transformer, "offloader", None) if getattr(transformer, "blocks_to_swap", None) else None
    forward_only = offloader.forward_only if offloader is not None else None
    with preserve_rng_state():
        try:
            transformer.eval()
            network.eval()
            if offloader is not None:
                offloader.set_forward_only(True)
                transformer.prepare_block_swap_before_forward()
            with torch.no_grad():
                yield
        finally:
            # Direct assignment preserves mixed nested modes; calling train() would recurse.
            for module, training in modes:
                module.training = training
            if offloader is not None:
                offloader.set_forward_only(forward_only)
                transformer.prepare_block_swap_before_forward()


def evaluate_validation(
    trainer,
    args,
    accelerator,
    transformer,
    network,
    inputs,
    noise_scheduler,
    dit_dtype,
    network_dtype,
    global_step=0,
):
    """Evaluate each fixed pair once on rank zero and return identical results on all ranks.

    The unwrapped transformer avoids DDP forward collectives while the other ranks wait.
    A structured error is broadcast on ordinary evaluation failures, so waiting peers also exit.
    """
    with preserve_rng_state():
        accelerator.wait_for_everyone()
        payload = [None]
        if accelerator.is_main_process:
            try:
                model = accelerator.unwrap_model(transformer)
                adapter = accelerator.unwrap_model(network)
                with validation_model_state(model, adapter):
                    inputs.verify_unchanged()
                    levels = validation_noise_levels(args.val_level_noise_n)
                    _integer(args.val_seed_noise_n, "val_seed_noise_n", 1)

                    def image_losses(index, record):
                        batch = inputs.load_batch(index)
                        batch["latents"] = batch["latents"].to(accelerator.device)
                        latents = trainer.scale_shift_latents(batch["latents"])
                        for i, t in enumerate(levels, 1):
                            for j in range(1, args.val_seed_noise_n + 1):
                                seed = stable_noise_seed(args.val_seed_noise, record.image_sha256, i, j)
                                noise = validation_noise_like(latents, seed)
                                noisy, timesteps, sigmas = fixed_validation_inputs(latents, noise, t)
                                output = trainer.call_dit(
                                    args, accelerator, model, latents, batch, noise, noisy, timesteps, network_dtype
                                )
                                loss, _ = trainer.compute_loss(
                                    args,
                                    output,
                                    timesteps,
                                    noise_scheduler,
                                    dit_dtype,
                                    network_dtype,
                                    global_step,
                                    fixed_sigmas=sigmas,
                                )
                                yield i, loss.detach().item()

                    ordered = sorted(enumerate(inputs.records), key=lambda item: item[1].image_sha256)
                    result = reduce_validation_losses(
                        (image_losses(index, record) for index, record in ordered),
                        args.val_level_noise_n,
                        args.val_seed_noise_n,
                    )
                payload[0] = {"result": result, "error": None}
            except Exception as error:
                if accelerator.num_processes == 1:
                    raise
                payload[0] = {"result": None, "error": f"{type(error).__name__}: {error}"}
        if accelerator.num_processes > 1:
            torch.distributed.broadcast_object_list(payload, src=0, device=accelerator.device)
        if payload[0]["error"] is not None:
            raise RuntimeError(f"Validation failed on main process: {payload[0]['error']}")
        return payload[0]["result"]
