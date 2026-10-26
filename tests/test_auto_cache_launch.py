"""Native Qwen launch boundaries for opt-in cache preparation, without model loads."""

import copy
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import toml
import torch
from transformers import Qwen2Tokenizer

from musubi_tuner import qwen_image_cache_latents as latent_cache
from musubi_tuner import qwen_image_cache_text_encoder_outputs as text_cache
from musubi_tuner import qwen_image_train_network as training
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.validation import preserve_rng_state
from test_validation_events import save_complete_state
from test_validation_inputs import make_data


class ReachedModels(Exception):
    """Stop after actual dataset, validation and resume checks, before Accelerator."""


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_equal(left, right):
    assert left[0] == right[0]
    assert left[1][0] == right[1][0]
    assert np.array_equal(left[1][1], right[1][1])
    assert left[1][2:] == right[1][2:]
    assert torch.equal(left[2], right[2])


@pytest.fixture
def launch(tmp_path, monkeypatch):
    root = tmp_path / "experiment with spaces"
    make_data(root, "train", "red")
    make_data(root, "val", "blue")
    weights = root / "model files"
    weights.mkdir()
    for name in ("dit", "vae", "text encoder"):
        (weights / f"{name}.safetensors").write_bytes(b"fixture checkpoint; never load weights")
    config = dict(
        experiment_mode=True,
        auto_cache=True,
        dataset_config="train.toml",
        val_dataset_config="val.toml",
        dit="model files/dit.safetensors",
        vae="model files/vae.safetensors",
        text_encoder="model files/text encoder.safetensors",
        network_module="networks.lora_qwen_image",
        optimizer_type="AdamW",
        sdpa=True,
        output_dir="output",
        output_name="fixture",
        max_data_loader_n_workers=0,
        min_timestep=20,
        max_timestep=980,
        preserve_distribution_shape=True,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        network_dropout=0.05,
        mixed_precision="bf16",
        seed=47,
    )
    events, observed = [], {}
    native_init = training.QwenImageNetworkTrainer._init_session

    def init_session(trainer, args):
        events.append("session")
        observed["pre_session_rng"] = rng_state()
        return native_init(trainer, args)

    def stop_before_models(trainer, args):
        events.append("models")
        observed["args"] = args
        observed["inputs"] = getattr(trainer, "validation_inputs", None)
        raise ReachedModels

    monkeypatch.setattr(training.QwenImageNetworkTrainer, "_init_session", init_session)
    monkeypatch.setattr(training.QwenImageNetworkTrainer, "_prepare_accelerator_and_dtypes", stop_before_models)
    monkeypatch.chdir(tmp_path)
    with preserve_rng_state():
        yield SimpleNamespace(root=root, config=config, events=events, observed=observed)


def invoke(launch, monkeypatch, cli=()):
    path = launch.root / "training settings.toml"
    path.write_text(toml.dumps(launch.config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["qwen_image_train_network.py", "--config_file", str(path), *cli])
    training.main()


def erase_fixture_caches(launch, role=None, kind=None):
    suffix = {None: "*.safetensors", "latent": "*_qi.safetensors", "text": "*_te.safetensors"}[kind]
    roots = [launch.root / "cache" / role] if role else [launch.root / "cache/train", launch.root / "cache/val"]
    for root in roots:
        for path in root.glob(suffix):
            path.unlink()


def native_children(monkeypatch, launch, failure=None):
    """Dispatch native cache CLIs; only process isolation and expensive encoders are substituted."""
    calls = []

    class Vae:
        device = torch.device("cpu")
        dtype = torch.float32

        def to(self, device):
            return self

        def eval(self):
            return self

        def encode_pixels_to_latents(self, pixels):
            return torch.ones(pixels.shape[0], 16, 1, pixels.shape[-2] // 8, pixels.shape[-1] // 8)

    def tokenizer(*args, **kwargs):
        launch.events.append("tokenizer")
        random.random()
        np.random.random()
        torch.rand(1)
        return object()

    monkeypatch.setattr(Qwen2Tokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(latent_cache.qwen_image_utils, "load_vae", lambda *a, **kw: Vae())
    monkeypatch.setattr(text_cache.qwen_image_utils, "load_qwen2_5_vl", lambda **kw: (None, SimpleNamespace(eval=lambda: None)))
    monkeypatch.setattr(
        text_cache.qwen_image_utils,
        "get_qwen_prompt_embeds",
        lambda tokenizer, model, prompts: (torch.ones(len(prompts), 3, 3584), torch.ones(len(prompts), 3)),
    )

    def run(argv, *, check, env):
        assert check is True and argv[:2] == [sys.executable, "-m"]
        module = {
            "musubi_tuner.qwen_image_cache_latents": latent_cache,
            "musubi_tuner.qwen_image_cache_text_encoder_outputs": text_cache,
        }[argv[2]]
        role = "val" if "--validation" in argv else "train"
        kind = "latent" if module is latent_cache else "text"
        stage = f"{role}:{kind}"
        launch.events.append(stage)
        calls.append(list(argv))
        assert Path(argv[argv.index("--dataset_config") + 1]) == launch.root / f"{role}.toml"
        model_flag = "--vae" if kind == "latent" else "--text_encoder"
        assert Path(argv[argv.index(model_flag) + 1]).is_absolute()
        assert "--skip_existing" in argv and "--keep_cache" in argv
        assert env.get("CUDA_VISIBLE_DEVICES") == os.environ.get("CUDA_VISIBLE_DEVICES")
        if failure and stage == "train:text":
            if failure == "child":
                raise subprocess.CalledProcessError(7, argv)
            return subprocess.CompletedProcess(argv, 0)
        with monkeypatch.context() as child:
            child.setattr(sys, "argv", [argv[2], *argv[3:], "--device", "cpu"])
            module.main()
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


@pytest.mark.parametrize(
    "configured,cli,expected",
    [(None, (), False), (True, (), True), (False, ("--auto_cache",), True), (True, ("--no_auto_cache",), False)],
)
def test_effective_auto_cache_boolean_and_cli_priority(tmp_path, monkeypatch, configured, cli, expected):
    path = tmp_path / "settings.toml"
    path.write_text(toml.dumps({} if configured is None else {"auto_cache": configured}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["fixture", "--config_file", str(path), *cli])
    parser = training.qwen_image_setup_parser(setup_parser_common())
    args = read_config_from_file(parser.parse_args(), parser)
    assert args.auto_cache is expected


@pytest.mark.parametrize("value", ["false", 0, 1])
def test_non_boolean_auto_cache_rejected(tmp_path, monkeypatch, value):
    path = tmp_path / "settings.toml"
    path.write_text(toml.dumps({"auto_cache": value}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["fixture", "--config_file", str(path)])
    parser = training.qwen_image_setup_parser(setup_parser_common())
    with pytest.raises(ValueError, match="auto_cache.*boolean"):
        read_config_from_file(parser.parse_args(), parser)


def test_enable_and_disable_cli_are_mutually_exclusive(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fixture", "--auto_cache", "--no_auto_cache"])
    with pytest.raises(SystemExit, match="2"):
        training.main()
    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize("validation", [False, True])
def test_native_launch_prepares_before_session_datasets_and_models(launch, monkeypatch, validation):
    if not validation:
        launch.config.pop("val_dataset_config")
    erase_fixture_caches(launch)
    calls = native_children(monkeypatch, launch)
    before_rng, before_env, before_cwd = rng_state(), dict(os.environ), Path.cwd()
    with pytest.raises(ReachedModels):
        invoke(launch, monkeypatch)
    expected = ["train:latent", "train:text"] + (["val:latent", "val:text"] if validation else [])
    assert launch.events == ["tokenizer", *expected, "session", "models"]
    assert len(calls) == len(expected)
    assert_rng_equal(before_rng, launch.observed["pre_session_rng"])
    assert dict(os.environ) == before_env and Path.cwd() == before_cwd
    args = launch.observed["args"]
    for key in (
        "min_timestep",
        "max_timestep",
        "preserve_distribution_shape",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "network_dropout",
        "mixed_precision",
        "seed",
    ):
        assert getattr(args, key) == launch.config[key]
    assert args.dit_dtype == args.vae_dtype == "bfloat16"
    assert args._training_state["validation_fingerprint"] == (launch.observed["inputs"].fingerprint if validation else None)
    if validation:
        launch.observed["inputs"].verify_unchanged()


def test_native_invalid_setting_stops_before_cache_work(launch, monkeypatch):
    erase_fixture_caches(launch)
    calls = native_children(monkeypatch, launch)
    launch.config["output_name"] = "../invalid"
    with pytest.raises(ValueError, match="output_name"):
        invoke(launch, monkeypatch)
    assert calls == [] and launch.events == []


@pytest.mark.parametrize("failure", ["child", "incomplete"])
def test_failed_preparation_preserves_completed_files_and_stops_training(launch, monkeypatch, failure):
    erase_fixture_caches(launch)
    native_children(monkeypatch, launch, failure)
    before_rng = rng_state()
    with pytest.raises((ValueError, RuntimeError)) as error:
        invoke(launch, monkeypatch)
    message = str(error.value)
    assert "train" in message and "text" in message and str(launch.root / "train.toml") in message
    assert "session" not in launch.events and "models" not in launch.events
    assert len(list((launch.root / "cache/train").glob("*_qi.safetensors"))) == 1
    assert list((launch.root / "cache/train").glob("*_te.safetensors")) == []
    assert_rng_equal(before_rng, rng_state())


def test_cli_disable_keeps_manual_cached_launch_without_prep_resources(launch, monkeypatch):
    calls = native_children(monkeypatch, launch)
    launch.config.pop("vae")
    launch.config.pop("text_encoder")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ReachedModels):
        invoke(launch, monkeypatch, ["--no_auto_cache"])
    assert calls == [] and launch.events == ["session", "models"]
    assert launch.observed["args"].auto_cache is False


@pytest.mark.parametrize("unsupported", ["experiment", "multiple_processes"])
def test_unsupported_auto_mode_stops_before_encoding(launch, monkeypatch, unsupported):
    calls = native_children(monkeypatch, launch)
    if unsupported == "experiment":
        launch.config["experiment_mode"] = False
        for key in ("dataset_config", "val_dataset_config", "dit", "vae", "text_encoder", "output_dir"):
            launch.config[key] = str(launch.root / launch.config[key])
    else:
        monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError) as error:
        invoke(launch, monkeypatch)
    cause = "requires experiment_mode" if unsupported == "experiment" else "single training process"
    assert "auto_cache" in str(error.value) and cause in str(error.value)
    assert calls == [] and launch.events == []


@pytest.mark.parametrize("regenerate", [False, True])
def test_native_resume_fingerprint_stays_authoritative_after_preparation(launch, monkeypatch, regenerate):
    calls = native_children(monkeypatch, launch)
    with pytest.raises(ReachedModels):
        invoke(launch, monkeypatch)
    state = copy.deepcopy(launch.observed["args"]._training_state)
    state["optimizer_global_step"] = 3
    original_fingerprint = state["validation_fingerprint"]
    save_complete_state(launch.root / "checkpoint", state)
    launch.config["resume"] = "checkpoint"
    launch.events.clear()
    if regenerate:
        erase_fixture_caches(launch, "val", "text")
        with pytest.raises(ValueError, match="resume: validation fingerprint changed"):
            invoke(launch, monkeypatch)
        assert "models" not in launch.events
        assert len(calls) == 1 and "--validation" in calls[0]
        assert len(list((launch.root / "cache/val").glob("*_te.safetensors"))) == 1
    else:
        with pytest.raises(ReachedModels):
            invoke(launch, monkeypatch)
        assert calls == [] and launch.events == ["session", "models"]
        resumed = launch.observed["args"]._training_state
        assert resumed["optimizer_global_step"] == 3
        assert resumed["validation_fingerprint"] == original_fingerprint
