import hashlib
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from musubi_tuner.training.timesteps import compute_loss_weighting_for_sd3
from musubi_tuner.training.validation import (
    fixed_validation_inputs,
    preserve_rng_state,
    reduce_validation_losses,
    stable_noise_seed,
    validation_noise_levels,
    validation_noise_like,
)


@pytest.mark.parametrize(
    "count,expected",
    [(2, (0.275, 0.725)), (10, (0.095, 0.185, 0.275, 0.365, 0.455, 0.545, 0.635, 0.725, 0.815, 0.905))],
)
def test_fixed_levels_and_equal_groups(count, expected):
    levels = validation_noise_levels(count)
    assert levels == pytest.approx(expected, abs=1e-15)
    assert sum(t < 0.5 for t in levels) == count // 2
    assert sum(t >= 0.5 for t in levels) == count // 2


@pytest.mark.parametrize("count", [True, False, 0, 1, 3, 2.0, "2", None])
def test_invalid_level_counts(count):
    with pytest.raises(ValueError, match="val_level_noise_n"):
        validation_noise_levels(count)


@pytest.mark.parametrize(
    "seed,image_sha,i,j,expected",
    [
        (42, "0" * 64, 1, 1, 2528747074945454131),
        (42, "0" * 64, 1, 2, 3513524646004844088),
        (42, "f" * 64, 10, 2, 8961402154011714235),
        (-17, "0123456789abcdef" * 4, 1, 1, 491017634604846590),
    ],
)
def test_seed_byte_contract(seed, image_sha, i, j, expected):
    assert stable_noise_seed(seed, image_sha, i, j) == expected
    assert stable_noise_seed(seed, image_sha.upper(), i, j) == expected


def test_noise_new_process_reorder_and_move(tmp_path):
    source = tmp_path / "original.img"
    source.write_bytes(b"source image bytes")
    image_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    latents = torch.zeros(1, 2, 1, 4, 6)
    expected = {(i, j): validation_noise_like(latents, stable_noise_seed(42, image_sha, i, j)) for i in (1, 2) for j in (1, 2)}
    moved = tmp_path / "renamed.img"
    source.rename(moved)
    moved_sha = hashlib.sha256(moved.read_bytes()).hexdigest()
    for i, j in reversed(list(expected)):
        actual = validation_noise_like(latents, stable_noise_seed(42, moved_sha, i, j))
        torch.testing.assert_close(actual, expected[i, j], rtol=0, atol=0)
    assert len({stable_noise_seed(42, image_sha, i, j) for i, j in expected}) == 4
    assert all(not torch.equal(expected[1, 1], expected[pair]) for pair in expected if pair != (1, 1))
    program = """
import json, torch
from musubi_tuner.training.validation import stable_noise_seed, validation_noise_like
seed = stable_noise_seed(42, __import__('sys').argv[1], 1, 1)
print(json.dumps(validation_noise_like(torch.zeros(1,2,1,4,6), seed).flatten().tolist()))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    actual = subprocess.check_output([sys.executable, "-c", program, image_sha], env=environment, text=True)
    torch.testing.assert_close(torch.tensor(json.loads(actual)).reshape(latents.shape), expected[1, 1], rtol=0, atol=0)


def test_fixed_mixing_timestep_and_sigma_do_not_use_scheduler():
    latents = torch.full((1, 2, 1, 4, 6), 2.0)
    noise = torch.full_like(latents, 6.0)
    noisy, timesteps, sigmas = fixed_validation_inputs(latents, noise, 0.275)
    torch.testing.assert_close(noisy, torch.full_like(latents, 3.1))
    torch.testing.assert_close(timesteps, torch.tensor([275.0]), rtol=0, atol=0)
    torch.testing.assert_close(sigmas, torch.tensor([0.275]), rtol=0, atol=0)
    wrong_scheduler = SimpleNamespace(sigmas=torch.tensor([0.9]), timesteps=torch.tensor([276.0]))
    for scheme in ("none", "sigma_sqrt", "cosmap"):
        actual = compute_loss_weighting_for_sd3(scheme, wrong_scheduler, timesteps, "cpu", torch.bfloat16, fixed_sigmas=sigmas)
        if scheme == "none":
            assert actual is None
        else:
            expected = sigmas**-2 if scheme == "sigma_sqrt" else 2 / (math.pi * (1 - 2 * sigmas + 2 * sigmas**2))
            torch.testing.assert_close(actual.flatten(), expected, rtol=0, atol=0)
            assert actual.shape == (1, 1, 1, 1, 1)
            assert actual.dtype == torch.float32


def test_legacy_weighting_keeps_scheduler_sigma_and_dtype():
    scheduler = SimpleNamespace(sigmas=torch.tensor([0.25, 0.5]), timesteps=torch.tensor([250.0, 500.0]))
    actual = compute_loss_weighting_for_sd3("sigma_sqrt", scheduler, scheduler.timesteps, "cpu", torch.bfloat16)
    torch.testing.assert_close(actual.flatten(), torch.tensor([16.0, 4.0]), rtol=0, atol=0)


@pytest.mark.parametrize("n1,n2", [(2, 1), (2, 2), (10, 1), (10, 2)])
def test_equal_image_averages_counts_and_order(n1, n2):
    # Scalar image loss is already the training reduction; pixel count must not weight the image again.
    image_errors = [torch.full((1, 1, 1, 2, 2), 2.0), torch.full((1, 1, 1, 8, 8), 10.0)]
    calls = []

    def losses(image_index):
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                calls.append((image_index, i, j))
                yield i, image_errors[image_index].mean().item() + (0 if i <= n1 // 2 else 4)

    result = reduce_validation_losses((losses(k) for k in range(2)), n1, n2)
    assert result.metrics == {"val_loss_mean": 8.0, "val_loss_low_noise": 6.0, "val_loss_high_noise": 10.0}
    assert result.image_count == 2 and result.forward_count == 2 * n1 * n2
    assert len(calls) == result.forward_count
    assert result.loss_mean == (result.loss_low_noise + result.loss_high_noise) / 2


@pytest.mark.parametrize("losses", [[], [[(1, 2.0)]], [[(1, 2.0), (1, 3.0)]], [[(1, float("nan")), (2, 2.0)]]])
def test_reduction_rejects_empty_incomplete_duplicate_or_nonfinite_measurements(losses):
    with pytest.raises(ValueError):
        reduce_validation_losses(losses, 2, 1)


@pytest.mark.parametrize("raises", [False, True])
def test_guard_and_noise_preserve_training_rng(raises):
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    with pytest.raises(RuntimeError) if raises else nullcontext():
        with preserve_rng_state():
            random.random(), np.random.rand(), torch.rand(3)
            validation_noise_like(torch.zeros(2, 3), 812)
            if raises:
                raise RuntimeError("fixture evaluation failure")
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    assert before[0] == after[0]
    np.testing.assert_equal(before[1], after[1])
    torch.testing.assert_close(before[2], after[2], rtol=0, atol=0)
    validation_noise_like(torch.zeros(2, 3), 812)
    torch.testing.assert_close(before[2], torch.get_rng_state(), rtol=0, atol=0)
