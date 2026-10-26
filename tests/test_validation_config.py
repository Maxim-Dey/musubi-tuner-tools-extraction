import argparse
import json
from pathlib import Path
import sys

import pytest
import toml

from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.experiment_config import add_cache_arguments, configure_cache_args
from musubi_tuner.dataset import config_utils


def resolve(monkeypatch, config, cli=()):
    parser = setup_parser_common()
    monkeypatch.setattr(sys, "argv", ["fixture", "--config_file", str(config), *cli])
    return read_config_from_file(parser.parse_args(), parser)


@pytest.mark.parametrize(
    "key,value",
    [
        ("val_seed_noise", True),
        ("val_level_noise_n", 3),
        ("val_level_noise_n", 0),
        ("val_seed_noise_n", 0),
        ("val_every_n_steps", 0),
        ("val_seed_noise_n", 1.5),
    ],
)
def test_invalid_effective_validation_settings(tmp_path, monkeypatch, key, value):
    path = tmp_path / "train.toml"
    path.write_text(toml.dumps({"val_dataset_config": "val.toml", key: value}), encoding="utf-8")
    with pytest.raises(ValueError, match=key):
        resolve(monkeypatch, path)


def test_validation_defaults_explicit_without_dataset_and_unknown(tmp_path, monkeypatch):
    path = tmp_path / "train.toml"
    path.write_text('val_dataset_config="val.toml"', encoding="utf-8")
    args = resolve(monkeypatch, path)
    assert (args.val_every_n_steps, args.val_seed_noise, args.val_level_noise_n, args.val_seed_noise_n) == (50, 42, 10, 2)
    path.write_text("val_every_n_steps=50", encoding="utf-8")
    with pytest.raises(ValueError, match="val_dataset_config"):
        resolve(monkeypatch, path)
    path.write_text("val_misspelling=50", encoding="utf-8")
    with pytest.raises(ValueError, match="val_misspelling"):
        resolve(monkeypatch, path)


def test_experiment_paths_cli_priority_and_legacy(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    root.mkdir()
    path = root / "train.toml"
    values = dict(
        experiment_mode=True,
        dataset_config="train-dataset.toml",
        val_dataset_config="val.toml",
        output_dir="output",
        logging_dir="output/tensorboard",
        sample_prompts="prompts.txt",
        resume="output/state",
        val_seed_noise=42,
    )
    path.write_text(toml.dumps(values), encoding="utf-8")
    expected = None
    for cwd in (tmp_path, root):
        monkeypatch.chdir(cwd)
        args = resolve(monkeypatch, path, ["--val_seed_noise", "19"])
        assert args.val_seed_noise == 19
        current = tuple(str(getattr(args, key)) for key in ("dataset_config", "output_dir", "resume", "sample_prompts"))
        assert all(Path(value).is_absolute() for value in current)
        if expected is not None:
            assert current == expected
        expected = current
    values["experiment_mode"] = False
    path.write_text(toml.dumps(values), encoding="utf-8")
    args = resolve(monkeypatch, path)
    assert str(args.dataset_config) == "train-dataset.toml" and args.resume == "output/state"


def test_server_absolute_paths_preserved_on_local_platform(tmp_path, monkeypatch):
    path = tmp_path / "train.toml"
    path.write_text('experiment_mode=true\ndit="/workspace/models/dit.safetensors"', encoding="utf-8")
    assert resolve(monkeypatch, path).dit == "/workspace/models/dit.safetensors"


def test_cache_flags_are_independent_and_resolve_dataset_paths(tmp_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config")
    parser.add_argument("--vae")
    add_cache_arguments(parser)
    dataset = tmp_path / "dataset.toml"
    dataset.write_text(
        '[general]\ncaption_extension=".txt"\n[[datasets]]\nimage_directory="dataset/val"\ncache_directory="cache/val"',
        encoding="utf-8",
    )
    args = configure_cache_args(parser.parse_args(["--dataset_config", str(dataset), "--vae", "vae.safetensors", "--validation"]))
    assert args.vae == "vae.safetensors" and not args.experiment_mode
    args = configure_cache_args(
        parser.parse_args(["--dataset_config", str(dataset), "--vae", "vae.safetensors", "--experiment_mode"])
    )
    assert args.vae == str(tmp_path / "vae.safetensors")
    blueprint = config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
        config_utils.load_user_config(dataset), args, architecture="qi"
    )
    assert blueprint.dataset_group.datasets[0].params.image_directory == str(tmp_path / "dataset/val")


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
@pytest.mark.parametrize("state_flag", ["save_state", "save_state_on_train_end"])
def test_state_precision_rejected_early(tmp_path, monkeypatch, precision, state_flag):
    path = tmp_path / "train.toml"
    path.write_text(toml.dumps(dict(experiment_mode=True, save_precision=precision, **{state_flag: True})), encoding="utf-8")
    with pytest.raises(ValueError, match="save_precision.*FP32"):
        resolve(monkeypatch, path)


def test_direct_full_precision_state_rejected(tmp_path):
    from musubi_tuner.training.experiment_config import configure_training_args

    args = argparse.Namespace(
        experiment_mode=True, _config_source=str(tmp_path / "train.toml"), save_state_on_train_end=True, full_bf16=True
    )
    with pytest.raises(ValueError, match="full_fp16/full_bf16"):
        configure_training_args(args)


def test_jsonl_paths_follow_config_from_other_cwd_only_in_experiment(tmp_path, monkeypatch):
    from PIL import Image

    root = tmp_path / "experiment"
    records = root / "records"
    records.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(records / "image.png")
    jsonl = records / "images.jsonl"
    original = json.dumps({"image_path": "image.png", "caption": "caption"}) + "\n"
    jsonl.write_text(original, encoding="utf-8")
    path = root / "dataset.toml"
    path.write_text('[[datasets]]\nimage_jsonl_file="records/images.jsonl"\ncache_directory="cache"', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(dataset_config=str(path), experiment_mode=True)
    blueprint = config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
        config_utils.load_user_config(path), args, architecture="qi"
    )
    group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    assert group.datasets[0].datasource.get_caption(0)[0] == str(records / "image.png")
    assert jsonl.read_text(encoding="utf-8") == original
    from musubi_tuner.dataset.datasources import ImageJsonlDatasource

    with pytest.raises(ValueError, match="working directory"):
        ImageJsonlDatasource(str(jsonl))
