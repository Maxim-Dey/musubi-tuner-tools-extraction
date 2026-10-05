"""Automatic preparation uses native cache commands before training can start."""

import copy
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import toml
import torch

from musubi_tuner.qwen_image import qwen_image_utils
from musubi_tuner.training import auto_cache


def inventory(root, role="train", *, latent=True, text=True):
    return SimpleNamespace(
        dataset_config=str(root / f"{role}.toml"),
        validation=role == "val",
        missing_latents=(str(root / role / "one_qi.safetensors"),) if latent else (),
        missing_text=(str(root / role / "one_qi_te.safetensors"),) if text else (),
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    args = SimpleNamespace(
        auto_cache=True,
        experiment_mode=True,
        _config_source=str(tmp_path / "train config.toml"),
        dataset_config=str(tmp_path / "train.toml"),
        val_dataset_config=str(tmp_path / "val.toml"),
        model_version="original",
        vae=str(tmp_path / "vae weights.safetensors"),
        text_encoder=str(tmp_path / "text weights.safetensors"),
        fp8_vl=False,
    )
    Path(args.vae).write_bytes(b"fixture encoder boundary")
    Path(args.text_encoder).write_bytes(b"fixture encoder boundary")
    for key in (*auto_cache._SIZE_VARIABLES, *auto_cache._RANK_VARIABLES, "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(key, raising=False)
    launches, tokenizer_calls = [], []
    monkeypatch.setattr(auto_cache.subprocess, "run", lambda command, **kwargs: launches.append((command, kwargs)))
    monkeypatch.setattr(
        qwen_image_utils.Qwen2Tokenizer,
        "from_pretrained",
        lambda *positional, **keywords: tokenizer_calls.append((positional, keywords)) or object(),
    )
    return args, launches, tokenizer_calls


def set_inventory(monkeypatch, values):
    remaining = iter(values)
    monkeypatch.setattr(auto_cache.validation_inputs, "inspect_cache_inputs", lambda args: next(remaining), raising=False)


def test_disabled_returns_before_inventory_scope_or_resources(setup, monkeypatch):
    args, launches, tokenizers = setup
    args.auto_cache, args.experiment_mode = False, False
    monkeypatch.setenv("WORLD_SIZE", "8")

    def unexpected(args):
        pytest.fail("disabled automatic preparation inspected sources")

    monkeypatch.setattr(auto_cache.validation_inputs, "inspect_cache_inputs", unexpected, raising=False)
    auto_cache.prepare_missing_caches(args)
    assert not launches and not tokenizers


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true"])
def test_non_boolean_rejected_before_inventory(setup, monkeypatch, value):
    args, launches, tokenizers = setup
    args.auto_cache = value
    set_inventory(monkeypatch, [])
    with pytest.raises(ValueError, match="auto_cache.*bool"):
        auto_cache.prepare_missing_caches(args)
    assert not launches and not tokenizers


@pytest.mark.parametrize(
    "key,value",
    [
        ("WORLD_SIZE", "2"),
        ("RANK", "1"),
        ("LOCAL_RANK", "1"),
        ("PMI_SIZE", "2"),
        ("OMPI_COMM_WORLD_RANK", "1"),
        ("MV2_COMM_WORLD_LOCAL_SIZE", "2"),
    ],
)
def test_distributed_rejected_before_inventory(setup, monkeypatch, key, value):
    args, launches, tokenizers = setup
    monkeypatch.setenv(key, value)
    set_inventory(monkeypatch, [])
    with pytest.raises(ValueError, match="single|one process"):
        auto_cache.prepare_missing_caches(args)
    assert not launches and not tokenizers


def test_experiment_required_before_inventory(setup, monkeypatch):
    args, launches, _ = setup
    args.experiment_mode = False
    set_inventory(monkeypatch, [])
    with pytest.raises(ValueError, match="experiment_mode"):
        auto_cache.prepare_missing_caches(args)
    assert not launches


def test_all_sources_checked_before_resources_and_children(setup, monkeypatch):
    args, launches, tokenizers = setup

    def invalid(args):
        raise ValueError("val.toml: source overlap; provide independent validation images")

    monkeypatch.setattr(auto_cache.validation_inputs, "inspect_cache_inputs", invalid, raising=False)
    with pytest.raises(ValueError, match="source overlap"):
        auto_cache.prepare_missing_caches(args)
    assert not launches and not tokenizers


def test_all_model_resources_checked_before_first_child(setup, monkeypatch, tmp_path):
    args, launches, tokenizers = setup
    Path(args.text_encoder).unlink()
    set_inventory(monkeypatch, [(inventory(tmp_path, text=False), inventory(tmp_path, "val", latent=False))])
    with pytest.raises(ValueError, match="val.*text.*text_encoder"):
        auto_cache.prepare_missing_caches(args)
    assert not launches and not tokenizers


def test_tokenizer_failure_precedes_even_latent_child(setup, monkeypatch, tmp_path):
    args, launches, _ = setup
    set_inventory(monkeypatch, [(inventory(tmp_path),)])

    def missing(*positional, **keywords):
        raise OSError("fixture tokenizer unavailable offline")

    monkeypatch.setattr(qwen_image_utils.Qwen2Tokenizer, "from_pretrained", missing)
    with pytest.raises(ValueError, match="train.*text.*tokenizer.*unavailable"):
        auto_cache.prepare_missing_caches(args)
    assert not launches


def test_complete_inventory_requires_no_unused_models_or_tokenizer(setup, monkeypatch, tmp_path):
    args, launches, tokenizers = setup
    args.vae = args.text_encoder = None
    set_inventory(monkeypatch, [(inventory(tmp_path, latent=False, text=False),)])
    auto_cache.prepare_missing_caches(args)
    assert not launches and not tokenizers


def test_four_stages_native_argv_and_single_rank_child_environment(setup, monkeypatch, tmp_path):
    args, launches, tokenizers = setup
    args.fp8_vl = True
    parent_argv, parent_cwd, before_args = list(sys.argv), Path.cwd(), copy.deepcopy(vars(args))
    for key, value in {
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29511",
        "PMI_SIZE": "1",
        "PMI_RANK": "0",
        "CUDA_VISIBLE_DEVICES": "3",
        "PYTHONPATH": "fixture path",
        "ACCELERATE_MIXED_PRECISION": "bf16",
    }.items():
        monkeypatch.setenv(key, value)
    parent_environment = dict(os.environ)
    set_inventory(
        monkeypatch,
        [
            (inventory(tmp_path), inventory(tmp_path, "val")),
            (inventory(tmp_path, latent=False, text=False), inventory(tmp_path, "val", latent=False, text=False)),
        ],
    )
    auto_cache.prepare_missing_caches(args)
    assert len(tokenizers) == 1
    assert tokenizers[0] == ((qwen_image_utils.QWEN_IMAGE_ID,), {"subfolder": "tokenizer"})
    assert [command[2].rsplit("_cache_", 1)[1] for command, _ in launches] == ["latents", "text_encoder_outputs"] * 2
    for index, (command, keywords) in enumerate(launches):
        assert command[:2] == [sys.executable, "-m"]
        assert command[command.index("--dataset_config") + 1] == str(tmp_path / ("train.toml" if index < 2 else "val.toml"))
        assert "--validation" in command if index >= 2 else "--validation" not in command
        assert "--fp8_vl" in command if index % 2 else "--fp8_vl" not in command
        assert all(flag in command for flag in ("--experiment_mode", "--skip_existing", "--keep_cache"))
        assert command[command.index("--batch_size") + 1] == command[command.index("--num_workers") + 1] == "1"
        model_key = "--text_encoder" if index % 2 else "--vae"
        assert command[command.index(model_key) + 1] == getattr(args, model_key[2:])
        assert keywords["check"] is True and not keywords.get("shell", False)
        assert all(
            key not in keywords["env"]
            for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "PMI_SIZE", "PMI_RANK")
        )
        assert keywords["env"]["CUDA_VISIBLE_DEVICES"] == "3"
        assert keywords["env"]["PYTHONPATH"] == "fixture path"
        assert keywords["env"]["ACCELERATE_MIXED_PRECISION"] == "bf16"
    assert dict(os.environ) == parent_environment and list(sys.argv) == parent_argv and Path.cwd() == parent_cwd
    assert vars(args) == before_args


@pytest.mark.parametrize("failure", ["child", "incomplete", "invalid"])
def test_failed_or_incomplete_preparation_stops_with_context(setup, monkeypatch, tmp_path, failure):
    args, launches, _ = setup
    set_inventory(monkeypatch, [(inventory(tmp_path, text=False),), (inventory(tmp_path, text=False),)])
    if failure == "child":

        def failed(command, **kwargs):
            raise subprocess.CalledProcessError(7, command)

        monkeypatch.setattr(auto_cache.subprocess, "run", failed)
    elif failure == "invalid":
        calls = []

        def invalid_after(args):
            calls.append(True)
            if len(calls) == 1:
                return (inventory(tmp_path, text=False),)
            raise ValueError("train.toml: corrupt latent cache; recreate it")

        monkeypatch.setattr(auto_cache.validation_inputs, "inspect_cache_inputs", invalid_after)
    with pytest.raises(ValueError, match="train.*latent|corrupt latent"):
        auto_cache.prepare_missing_caches(args)
    assert len(launches) <= 1


@pytest.mark.parametrize("fail", [False, True])
def test_parent_rng_restored_during_inspection_tokenizer_and_failure(setup, monkeypatch, tmp_path, fail):
    args, launches, _ = setup
    before = copy.deepcopy((random.getstate(), np.random.get_state(), torch.get_rng_state()))
    inspected = []

    def consume():
        random.random()
        np.random.rand()
        torch.rand(1)

    def inspect(args):
        consume()
        inspected.append(True)
        return (inventory(tmp_path, latent=False, text=len(inspected) == 1),)

    def tokenizer(*positional, **keywords):
        consume()
        if fail:
            raise OSError("fixture tokenizer failure")
        return object()

    monkeypatch.setattr(auto_cache.validation_inputs, "inspect_cache_inputs", inspect, raising=False)
    monkeypatch.setattr(qwen_image_utils.Qwen2Tokenizer, "from_pretrained", tokenizer)
    if fail:
        with pytest.raises(ValueError, match="tokenizer failure"):
            auto_cache.prepare_missing_caches(args)
    else:
        auto_cache.prepare_missing_caches(args)
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert np.random.get_state()[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])
    assert len(launches) == int(not fail)


def source_configuration(root, role, colors):
    images = root / "images" / role
    images.mkdir(parents=True)
    for index, color in enumerate(colors):
        Image.new("RGB", (32, 32), color).save(images / f"image {index}.png")
        (images / f"image {index}.txt").write_text(f"caption {role} {index}", encoding="utf-8")
    config = root / f"{role}.toml"
    config.write_text(
        toml.dumps(
            {
                "general": {
                    "resolution": [32, 32],
                    "caption_extension": ".txt",
                    "batch_size": 1 if role == "val" else 2,
                    "num_repeats": 1 if role == "val" else 2,
                    "enable_bucket": True,
                    "bucket_no_upscale": True,
                },
                "datasets": [{"image_directory": f"images/{role}", "cache_directory": f"cache/{role}"}],
            }
        ),
        encoding="utf-8",
    )
    return str(config)


def native_cache_children(monkeypatch):
    """Exercise real parsers/readers/cache writers; substitute only process and encoders."""
    from musubi_tuner import qwen_image_cache_latents, qwen_image_cache_text_encoder_outputs

    encoded, commands = [], []

    class Vae:
        device, dtype = torch.device("cpu"), torch.bfloat16

        def to(self, device):
            assert device.type == "cpu"
            return self

        def eval(self):
            return self

        def encode_pixels_to_latents(self, pixels):
            encoded.append(("latent", len(pixels)))
            return torch.ones(len(pixels), 16, 1, pixels.shape[-2] // 8, pixels.shape[-1] // 8, dtype=self.dtype)

    class TextEncoder:
        def eval(self):
            return self

    def text_embeddings(tokenizer, text_encoder, prompts):
        encoded.append(("text", len(prompts)))
        return torch.ones(len(prompts), 2, 3584, dtype=torch.bfloat16), torch.ones(len(prompts), 2, dtype=torch.int64)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(qwen_image_utils, "load_vae", lambda *positional, **keywords: Vae())
    monkeypatch.setattr(qwen_image_utils, "load_qwen2_5_vl", lambda **keywords: (object(), TextEncoder()))
    monkeypatch.setattr(qwen_image_utils, "get_qwen_prompt_embeds", text_embeddings)

    def run(command, *, check, env):
        assert check is True
        commands.append(command)
        module = qwen_image_cache_latents if command[2].endswith("_latents") else qwen_image_cache_text_encoder_outputs
        with monkeypatch.context() as child:
            child.setattr(sys, "argv", [command[2], *command[3:]])
            for key in set(os.environ) - env.keys():
                child.delenv(key)
            for key, value in env.items():
                child.setenv(key, value)
            module.main()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(auto_cache.subprocess, "run", run)
    return commands, encoded, run


@pytest.mark.parametrize("with_validation", [False, True])
def test_native_cache_mains_missing_only_reuse_and_other_cwd(setup, monkeypatch, tmp_path, with_validation):
    args, _, tokenizer_calls = setup
    root = tmp_path / "experiment with spaces"
    root.mkdir()
    args.dataset_config = source_configuration(root, "train", ["red", "green"])
    args.val_dataset_config = source_configuration(root, "val", ["blue"]) if with_validation else None
    extra = root / "cache/train/unrelated_qi_te.safetensors"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b"unrelated file; automatic preparation must retain it")
    commands, encoded, _ = native_cache_children(monkeypatch)
    other_cwd = tmp_path / "another working directory"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    auto_cache.prepare_missing_caches(args)
    assert len(commands) == (4 if with_validation else 2)
    assert sum(count for kind, count in encoded if kind == "latent") == (3 if with_validation else 2)
    assert sum(count for kind, count in encoded if kind == "text") == (3 if with_validation else 2)
    before = {path: path.read_bytes() for path in (root / "cache").rglob("*") if path.is_file()}
    commands.clear()
    encoded.clear()
    tokenizer_calls.clear()
    auto_cache.prepare_missing_caches(args)
    assert not commands and not encoded and not tokenizer_calls
    assert all(path.read_bytes() == contents for path, contents in before.items())

    missing = next(
        path for path in before if path.name != extra.name and path.parent.name == "train" and path.name.endswith("_te.safetensors")
    )
    missing.unlink()
    preserved = {path: contents for path, contents in before.items() if path != missing}
    auto_cache.prepare_missing_caches(args)
    assert len(commands) == 1 and commands[0][2].endswith("qwen_image_cache_text_encoder_outputs")
    assert encoded == [("text", 1)] and missing.is_file()
    assert all(path.read_bytes() == contents for path, contents in preserved.items())
    assert all(
        not item.missing_latents and not item.missing_text for item in auto_cache.validation_inputs.inspect_cache_inputs(args)
    )


def test_native_retry_keeps_successful_stage_files(setup, monkeypatch, tmp_path):
    args, _, _ = setup
    args.dataset_config = source_configuration(tmp_path, "train", ["red", "green"])
    args.val_dataset_config = source_configuration(tmp_path, "val", ["blue"])
    commands, encoded, native_run = native_cache_children(monkeypatch)

    def fail_text(command, **keywords):
        if command[2].endswith("_text_encoder_outputs"):
            raise subprocess.CalledProcessError(9, command)
        return native_run(command, **keywords)

    monkeypatch.setattr(auto_cache.subprocess, "run", fail_text)
    with pytest.raises(ValueError, match="train.*text.*preparation failed"):
        auto_cache.prepare_missing_caches(args)
    retained = {path: path.read_bytes() for path in (tmp_path / "cache/train").glob("*.safetensors")}
    assert len(retained) == 2 and all(not path.name.endswith("_te.safetensors") for path in retained)
    monkeypatch.setattr(auto_cache.subprocess, "run", native_run)
    commands.clear()
    encoded.clear()
    auto_cache.prepare_missing_caches(args)
    assert len(commands) == 3 and commands[0][2].endswith("_text_encoder_outputs")
    assert encoded == [("text", 1), ("text", 1), ("latent", 1), ("text", 1)]
    assert all(path.read_bytes() == contents for path, contents in retained.items())
