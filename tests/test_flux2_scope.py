"""Production input boundaries, with sentinels at the real weight loaders."""

import functools
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import toml
import torch
from PIL import Image

from musubi_tuner import flux_2_cache_latents, flux_2_cache_text_encoder_outputs, flux_2_train_network
from musubi_tuner.dataset.image_video_dataset import BaseDataset, ItemInfo
from musubi_tuner.dataset.cache_io import save_latent_cache_flux_2, save_text_encoder_output_cache_flux_2
from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.sampling_prompts import load_prompts


class LoaderReached(RuntimeError):
    pass


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    image = images / "cup.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image)
    image.with_suffix(".txt").write_text("cup", encoding="utf-8")
    cache = tmp_path / "cache"
    dataset = BaseDataset(cache_directory=str(cache), architecture="f2d")
    item = ItemInfo(str(image), "cup", (16, 16))
    item.latent_cache_path = dataset.get_latent_cache_path(item)
    item.text_encoder_output_cache_path = dataset.get_text_encoder_output_cache_path(item)
    save_latent_cache_flux_2(item, torch.zeros(1, 1, 1), None, "flux_2_dev")
    save_text_encoder_output_cache_flux_2(item, torch.zeros(2, 3), "flux_2_dev")
    dataset_path = tmp_path / "dataset.toml"
    dataset_path.write_text(
        toml.dumps(
            {
                "general": {"resolution": 16, "caption_extension": ".txt"},
                "datasets": [{"image_directory": str(images), "cache_directory": str(cache)}],
            }
        ),
        encoding="utf-8",
    )
    weights = tmp_path / "weights.safetensors"
    weights.touch()  # Existence only; no test opens this as a checkpoint.
    config = {
        "model_version": "dev",
        "network_module": "networks.lora_flux_2",
        "dataset_config": str(dataset_path),
        "dit": str(weights),
        "vae": str(weights),
        "text_encoder": str(weights),
        "output_dir": str(tmp_path / "output"),
        "output_name": "cup",
        "sdpa": True,
        "optimizer_type": "AdamW",
        "mixed_precision": "no",
        "max_data_loader_n_workers": 0,
    }
    calls = []

    def loader(*args, **kwargs):
        calls.append(True)
        raise LoaderReached("weight loader sentinel")

    for name in ["load_flow_model", "load_ae", "load_text_embedder"]:
        monkeypatch.setattr(flux2_utils, name, loader)

    # Isolate cache availability from the boundary matrix. No import/model is stubbed;
    # the actual cached processor/tokenization probe is a separate environment check.
    def cached_processor(name, **kwargs):
        assert name == flux2_utils.M3_TOKENIZER_ID
        assert kwargs == {"use_fast": False, "local_files_only": True}
        return object()

    monkeypatch.setattr(flux2_utils.AutoProcessor, "from_pretrained", cached_processor)
    return tmp_path, config, calls


def run_entry(inputs, monkeypatch, command="train", changes=None, extra=()):
    root, config, _ = inputs
    config = config | (changes or {})
    if command == "train":
        path = root / "train.toml"
        path.write_text(toml.dumps(config), encoding="utf-8")
        argv = ["--config_file", str(path), *extra]
        module = flux_2_train_network
    else:
        module = flux_2_cache_latents if command == "latent" else flux_2_cache_text_encoder_outputs
        weight_key = "vae" if command == "latent" else "text_encoder"
        argv = [
            "--dataset_config",
            config["dataset_config"],
            f"--{weight_key}",
            config[weight_key],
            "--model_version",
            config["model_version"],
            "--device",
            "cpu",
            *extra,
        ]
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


@pytest.mark.parametrize("command", ["train", "latent", "text"])
@pytest.mark.parametrize("model", ["klein-4b", "klein-base-4b", "klein-9b", "klein-base-9b", "wan"])
def test_reject_model_before_weights(inputs, monkeypatch, capsys, command, model):
    with pytest.raises((ValueError, SystemExit)) as error:
        run_entry(inputs, monkeypatch, command, {"model_version": model})
    message = str(error.value) + capsys.readouterr().err
    assert "model_version" in message and model in message
    assert inputs[2] == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("network_module", "networks.lora"),
        ("network_module", "networks.lora_wan"),
        ("network_module", "networks.loha"),
        ("network_module", "networks.lokr"),
        ("network_module", "lycoris.kohya"),
        ("network_args", ["algo=loha"]),
        ("network_args", ["module_class=other.Factory"]),
        ("network_args", ["rank_dropout=0.1", "rank_dropout=0.2"]),
        ("network_args", ["exclude_patterns=['[']"]),
        ("network_args", ["conv_dim=bad"]),
        ("save_merged_model", True),
        ("self_flow", True),
        ("video_length", 16),
        ("learnng_rate", 1e-4),
        ("learning_rate", "invalid"),
        ("network_dim", True),
        ("mixed_precision", "invalid"),
        ("fp8_text_encoder", True),
        ("optimizer_args", ["betas=(0.9, 0.999)", "made_up=1"]),
        ("lr_scheduler_args", ["made_up=1"]),
    ],
)
def test_training_rejects_sources_before_weights(inputs, monkeypatch, key, value):
    with pytest.raises(ValueError) as error:
        run_entry(inputs, monkeypatch, changes={key: value})
    assert key in str(error.value)
    assert "train.toml" in str(error.value)
    assert inputs[2] == []


@pytest.mark.parametrize(
    "changes,key,value",
    [
        ({"lr_scheduler": "constant", "lr_warmup_steps": 100}, "lr_warmup_steps", "100"),
        ({"optimizer_type": "SGD", "optimizer_args": ["nesterov=True"]}, "optimizer_args", "True"),
        (
            {"optimizer_type": "torch.optim.SGD", "optimizer_args": ["nesterov=True", "momentum=0.9", "dampening=0.1"]},
            "optimizer_args",
            "0.1",
        ),
        ({"blocks_to_swap": 100}, "blocks_to_swap", "100"),
        ({"blocks_to_swap": 30}, "blocks_to_swap", "30"),
        ({"min_timestep": 1001, "max_timestep": 1002}, "min_timestep", "1001"),
        ({"max_timestep": 1001}, "max_timestep", "1001"),
        ({"min_timestep": -1}, "min_timestep", "-1"),
    ],
)
def test_round1_invalid_effective_values_before_weights(inputs, monkeypatch, changes, key, value):
    with pytest.raises(ValueError) as error:
        run_entry(inputs, monkeypatch, changes=changes)
    message = str(error.value)
    assert "train.toml" in message and key in message and value in message and ";" in message
    assert inputs[2] == []


@pytest.mark.parametrize(
    "changes",
    [
        {"lr_scheduler": "constant", "lr_warmup_steps": 0},
        {"lr_scheduler": "constant", "lr_warmup_steps": 0.1, "max_train_steps": 1},
        {"optimizer_type": "SGD", "optimizer_args": ["nesterov=True", "momentum=0.9"]},
        {"blocks_to_swap": 29},
        {"min_timestep": 0, "max_timestep": 1000},
        {"min_timestep": 999, "max_timestep": 1000},
        {"min_timestep": 10, "max_timestep": 10},
    ],
)
def test_round1_valid_boundaries(inputs, monkeypatch, changes):
    with pytest.raises(LoaderReached):
        run_entry(inputs, monkeypatch, changes=changes)
    assert inputs[2] == [True]


@pytest.mark.parametrize("command", ["train", "latent", "text"])
@pytest.mark.parametrize("extra", [["--unknown_option", "1"], ["--save_merged_model"], ["--video_length", "2"]])
def test_unknown_cli(inputs, monkeypatch, capsys, command, extra):
    with pytest.raises(SystemExit):
        run_entry(inputs, monkeypatch, command, extra=extra)
    assert extra[0] in capsys.readouterr().err
    assert inputs[2] == []


@pytest.mark.parametrize("section", ["model", "lora", "arbitrary_user_group"])
def test_grouped_training_precedence(tmp_path, monkeypatch, section):
    path = tmp_path / "train.toml"
    path.write_text(
        toml.dumps({section: {"model_version": "klein-4b", "network_module": "networks.lora_flux_2", "learning_rate": 0.0001}}),
        encoding="utf-8",
    )
    parser = setup_parser_common()
    flux_2_train_network.flux2_setup_parser(parser)
    monkeypatch.setattr(sys, "argv", ["train", "--config_file", str(path), "--model_version", "dev", "--learning_rate", "0.0002"])
    args = read_config_from_file(parser.parse_args(), parser)
    assert args.model_version == "dev"
    assert args.learning_rate == 0.0002
    assert args.network_module == "networks.lora_flux_2"


@pytest.mark.parametrize(
    "payload,key",
    [({"model": {"learnng_rate": 1e-4}}, "model.learnng_rate"), ({"model": {"nested": {"learning_rate": 1e-4}}}, "model.nested")],
)
def test_group_raw_keys_cannot_disappear(tmp_path, monkeypatch, payload, key):
    path = tmp_path / "train.toml"
    path.write_text(toml.dumps(payload), encoding="utf-8")
    parser = setup_parser_common()
    flux_2_train_network.flux2_setup_parser(parser)
    monkeypatch.setattr(sys, "argv", ["train", "--config_file", str(path), "--learning_rate", "0.0002"])
    with pytest.raises(ValueError) as error:
        read_config_from_file(parser.parse_args(), parser)
    assert key in str(error.value) and str(path) in str(error.value)


@pytest.mark.parametrize("command", ["train", "latent", "text"])
@pytest.mark.parametrize("extension", ["toml", "json"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("video_directory", "videos"),
        ("audio_sample_rate", 24000),
        ("fp_latent_window_size", 9),
        ("unknown", 1),
        ("batch_size", True),
    ],
)
def test_invalid_dataset_before_weights(inputs, monkeypatch, command, extension, field, value):
    path = Path(inputs[1]["dataset_config"])
    data = toml.loads(path.read_text(encoding="utf-8"))
    data["datasets"][0][field] = value
    path = path.with_suffix(f".{extension}")
    path.write_text(toml.dumps(data) if extension == "toml" else json.dumps(data), encoding="utf-8")
    inputs[1]["dataset_config"] = str(path)
    with pytest.raises(ValueError) as error:
        run_entry(inputs, monkeypatch, command)
    assert field in str(error.value) and str(path) in str(error.value)
    assert inputs[2] == []


@pytest.mark.parametrize(
    "extension,text,key",
    [
        ("txt", "cup --unknown 1", "unknown"),
        ("txt", "cup --w 64junk", "w"),
        ("txt", "cup --f 2", "f"),
        ("txt", "cup --w", "w"),
        ("txt", "# empty", "prompt"),
        ("json", '[{"prompt":"cup","enum":1}]', "enum"),
        ("json", '[{"prompt":"cup","width":true}]', "width"),
        ("json", '{"prompt":"cup"}', "prompt"),
        ("toml", '[prompt]\nprompt="cup"\nunknown=1\n[[prompt.subset]]\nseed=1', "unknown"),
    ],
)
def test_prompt_source_validation(tmp_path, extension, text, key):
    path = tmp_path / f"prompts.{extension}"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_prompts(str(path))
    assert key in str(error.value) and str(path) in str(error.value)


@pytest.mark.parametrize("extension", ["txt", "json", "toml"])
def test_valid_prompt_formats(tmp_path, extension):
    path = tmp_path / f"prompts.{extension}"
    record = {
        "prompt": "d 12 monkeys",
        "width": 64,
        "negative_prompt": "blur",
        "cfg_scale": 2.0,
        "control_image_path": ["one.png", "two.png"],
    }
    contents = {
        "txt": "d 12 monkeys --w 64 --n blur --l 2 --ci one.png --ci two.png",
        "json": json.dumps([record]),
        "toml": toml.dumps({"prompt": {"subset": [record]}}),
    }
    path.write_text(contents[extension], encoding="utf-8")
    assert load_prompts(str(path)) == [record | {"enum": 0}]


@pytest.mark.parametrize(
    "text,key",
    [
        (None, "log_tracker_config"),
        ("[broken", "log_tracker_config"),
        ('[unknown_tracker]\nname="cup"', "unknown_tracker"),
        ("[tensorboard]\nmade_up=1", "tensorboard.made_up"),
        ('tensorboard="invalid"', "tensorboard"),
        ('[tensorboard]\nflush_secs="invalid"', "tensorboard.flush_secs"),
    ],
)
def test_tracker_file_errors_before_weights(inputs, monkeypatch, text, key):
    path = inputs[0] / "tracker.toml"
    if text is not None:
        path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        run_entry(
            inputs,
            monkeypatch,
            changes={"log_tracker_config": str(path), "log_with": "tensorboard", "logging_dir": str(inputs[0] / "logs")},
        )
    assert str(path) in str(error.value) and key in str(error.value)
    assert inputs[2] == []


@pytest.mark.parametrize("command", ["train", "latent", "text"])
def test_valid_entry_reaches_only_guarded_loader(inputs, monkeypatch, command):
    with pytest.raises(LoaderReached):
        run_entry(inputs, monkeypatch, command)
    assert inputs[2] == [True]


@pytest.mark.parametrize(
    "package,changes",
    [
        ("bitsandbytes", {"optimizer_type": "AdamW8bit"}),
        ("tensorboard", {"log_with": "tensorboard"}),
        ("xformers", {"sdpa": False, "xformers": True}),
    ],
)
def test_selected_dependency_is_required_before_weights(inputs, monkeypatch, package, changes):
    original_import, original_spec = importlib.import_module, importlib.util.find_spec

    def denied_import(name, *args, **kwargs):
        if name == package or name.startswith(package + "."):
            raise ImportError(f"fixture: {package} unavailable")
        return original_import(name, *args, **kwargs)

    def denied_spec(name, *args, **kwargs):
        if name == package or name.startswith(package + "."):
            return None
        return original_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", denied_import)
    monkeypatch.setattr(importlib.util, "find_spec", denied_spec)
    changes = changes | {"logging_dir": str(inputs[0] / "logs")}
    with pytest.raises(ValueError) as error:
        run_entry(inputs, monkeypatch, changes=changes)
    assert package in str(error.value)
    assert inputs[2] == []


def test_invalid_prompt_entry_stops_before_mistral(inputs, monkeypatch):
    path = inputs[0] / "prompts.txt"
    path.write_text("cup --unknown 1", encoding="utf-8")
    with pytest.raises(ValueError) as error:
        run_entry(inputs, monkeypatch, changes={"sample_prompts": str(path)})
    assert str(path) in str(error.value) and "unknown" in str(error.value)
    assert inputs[2] == []


def test_valid_tracker_configuration_does_not_start_tracker(inputs, monkeypatch):
    path = inputs[0] / "tracker.toml"
    path.write_text('[tensorboard]\nflush_secs=30\nmax_queue=10\nfilename_suffix=".local"', encoding="utf-8")
    from torch.utils.tensorboard import SummaryWriter

    @functools.wraps(SummaryWriter.__init__)
    def forbidden(*args, **kwargs):
        pytest.fail("tracker initialized before weights")

    monkeypatch.setattr(SummaryWriter, "__init__", forbidden)
    with pytest.raises(LoaderReached):
        run_entry(
            inputs,
            monkeypatch,
            changes={"log_tracker_config": str(path), "log_with": "tensorboard", "logging_dir": str(inputs[0] / "logs")},
        )
    assert inputs[2] == [True]


@pytest.mark.parametrize(
    "changes",
    [
        {
            "network_module": "musubi_tuner.networks.lora_flux_2",
            "network_args": [
                "conv_dim=4",
                "conv_alpha=4",
                "rank_dropout=0.1",
                "module_dropout=0.1",
                "verbose=True",
                "exclude_patterns=['.*norm.*']",
                "include_patterns=['.*linear.*']",
                "loraplus_lr_ratio=2",
            ],
        },
        {
            "optimizer_type": "SGD",
            "optimizer_args": ["momentum=0.9"],
            "lr_scheduler_type": "StepLR",
            "lr_scheduler_args": ["step_size=10", "gamma=0.5"],
        },
        {"optimizer_type": "torch.optim.AdamW", "optimizer_args": ["betas=(0.9, 0.99)"]},
        {
            "optimizer_type": "Adafactor",
            "optimizer_args": ["relative_step=False", "scale_parameter=False"],
            "lr_scheduler": "constant_with_warmup",
        },
        {"optimizer_type": "Adafactor"},
        {"num_timestep_buckets": 0, "timestep_sampling": "qwen_shift", "fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 2},
    ],
)
def test_valid_nested_dev_options(inputs, monkeypatch, changes):
    with pytest.raises(LoaderReached):
        run_entry(inputs, monkeypatch, changes=changes)
    assert inputs[2] == [True]


@pytest.mark.parametrize("command,key", [("train", "dit"), ("latent", "vae"), ("text", "text_encoder")])
def test_missing_checkpoint_stops_before_weights(inputs, monkeypatch, command, key):
    inputs[1][key] = str(inputs[0] / "missing.safetensors")
    with pytest.raises(ValueError, match=key):
        run_entry(inputs, monkeypatch, command)
    assert not inputs[2]


def test_missing_companion_shard_before_weights(inputs, monkeypatch):
    shard = inputs[0] / "model-00001-of-00002.safetensors"
    shard.touch()
    with pytest.raises(ValueError, match="companion shard"):
        run_entry(inputs, monkeypatch, "text", changes={"text_encoder": str(shard)})
    assert not inputs[2]


def test_missing_processor_before_weights(inputs, monkeypatch):
    def missing(*args, **kwargs):
        assert kwargs["local_files_only"] is True
        raise OSError("fixture: processor cache missing")

    monkeypatch.setattr(flux2_utils.AutoProcessor, "from_pretrained", missing)
    with pytest.raises(ValueError, match="cached processor"):
        run_entry(inputs, monkeypatch, "text")
    assert not inputs[2]


def test_preflight_does_not_consume_rng(inputs, monkeypatch):
    import random

    path = inputs[0] / "train.toml"
    path.write_text(toml.dumps(inputs[1]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train", "--config_file", str(path)])
    parser = flux_2_train_network.flux2_setup_parser(setup_parser_common())
    args = read_config_from_file(parser.parse_args(), parser)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    flux_2_train_network.validate_training_inputs(args)
    assert random.getstate() == before[0]
    np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
    assert np.random.get_state()[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])


@pytest.mark.parametrize(
    "changes,key",
    [
        ({"sdpa": False}, "sdpa"),
        ({"blocks_to_swap": 2, "block_swap_h2d_only": True, "gradient_checkpointing": False}, "block_swap_h2d_only"),
        ({"compile": True, "compile_backend": "nonexistent_backend"}, "compile_backend"),
        ({"network_args": ["rank_dropout=1"]}, "network_args"),
    ],
)
def test_selected_backend_coupling_before_weights(inputs, monkeypatch, changes, key):
    with pytest.raises(ValueError, match=key):
        run_entry(inputs, monkeypatch, changes=changes)
    assert not inputs[2]
