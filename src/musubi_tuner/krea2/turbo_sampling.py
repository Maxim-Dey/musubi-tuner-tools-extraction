"""Sample-only RAW/Turbo weight swap, adapted from kohya-ss/musubi-tuner.

Upstream f8a1b03794a49239a3539015075f5123d6c07d66 keeps the live
Parameter objects so the current trainee LoRA remains attached. A snapshot of
the actual frozen RAW base also preserves merged base adapters on rollback.
"""
from contextlib import contextmanager
import itertools

import torch

from musubi_tuner.krea2.krea2_utils import KREA2_FP8_OPTIMIZATION_TARGET_KEYS, KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS
from musubi_tuner.training.krea2_validation import validation_context
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.utils.safetensors_utils import load_safetensors


def load_sampling_state(path, *, fp8_scaled, calc_device, result_device):
    """Load weights only: no second DiT, no copied trainee adapter."""
    if fp8_scaled:
        return load_safetensors_with_lora_and_fp8(
            model_files=path, lora_weights_list=None, lora_multipliers=None,
            fp8_optimization=True, calc_device=torch.device(calc_device),
            move_to_device=torch.device(calc_device) == torch.device(result_device),
            dit_weight_dtype=None, target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS,
            exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS,
        )
    return load_safetensors(path, device=result_device, disable_mmap=True, dtype=torch.bfloat16)


def live_tensors(model):
    # LoRA lives in the separate network; these are just frozen base tensors.
    return {key.replace("._orig_mod.", "."): value
            for key, value in itertools.chain(model.named_parameters(), model.named_buffers())}


def check_state(state, shapes):
    missing = shapes.keys() - state.keys()
    unexpected = state.keys() - shapes.keys()
    if missing or unexpected:
        raise ValueError(f"Turbo DiT keys differ from RAW: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for key, shape in shapes.items():
        if tuple(state[key].shape) != shape:
            raise ValueError(f"Turbo DiT tensor {key}: expected shape {shape}, got {tuple(state[key].shape)}")


def overwrite(tensors, state):
    for key, tensor in tensors.items():
        tensor.data.copy_(state[key])


def release(tensors):
    for tensor in tensors.values():
        tensor.data = tensor.data.new_empty((0,))


@contextmanager
def sample_on_turbo(trainer, args, accelerator, transformer, network):
    model = accelerator.unwrap_model(transformer)
    adapter = accelerator.unwrap_model(network)
    tensors = live_tensors(model)
    shapes = {key: tuple(value.shape) for key, value in tensors.items()}
    devices = {key: value.device for key, value in tensors.items()}
    dtypes = {key: value.dtype for key, value in tensors.items()}
    cached = args.turbo_dit_cache
    # The guard covers loader/quantizer RNG too, not just the denoising loop.
    with validation_context(model, adapter):
        if cached:
            if trainer._turbo_raw_state is None:
                trainer._turbo_raw_state = {key: value.detach().to("cpu", copy=True) for key, value in tensors.items()}
            raw = trainer._turbo_raw_state
        else:
            # Transient CPU RAW copy is necessary for exact exception rollback,
            # including an already merged base LoRA; no disk re-read can replace it.
            raw = {key: value.detach().to("cpu", copy=True) for key, value in tensors.items()}
        state = None
        changed = False
        try:
            if cached:
                if trainer._turbo_cached_state is None:
                    trainer._turbo_cached_state = load_sampling_state(
                        args.turbo_dit, fp8_scaled=args.fp8_scaled,
                        calc_device=accelerator.device, result_device="cpu",
                    )
                state = trainer._turbo_cached_state
                check_state(state, shapes)
                changed = True  # also roll back a failure midway through copy_
                overwrite(tensors, state)
            else:
                # Validate the complete candidate before touching live weights.
                # The CPU payload is temporary; GPU placement happens tensor by
                # tensor after freeing RAW, avoiding two GPU DiTs at once.
                state = load_sampling_state(args.turbo_dit, fp8_scaled=args.fp8_scaled,
                                            calc_device=accelerator.device, result_device="cpu")
                check_state(state, shapes)
                changed = True
                release(tensors)
                for key, tensor in tensors.items():
                    tensor.data = state[key].to(device=devices[key], dtype=dtypes[key])
                state = None
            trainer._turbo_sampling_active = True
            yield
        finally:
            trainer._turbo_sampling_active = False
            state = None
            if changed:
                if cached:
                    overwrite(tensors, raw)
                else:
                    release(tensors)
                    for key, tensor in tensors.items():
                        tensor.data = raw[key].to(device=devices[key], dtype=dtypes[key])
