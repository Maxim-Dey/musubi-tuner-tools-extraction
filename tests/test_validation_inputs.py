import argparse
from pathlib import Path
import random
import shutil
import sys
from types import SimpleNamespace

from PIL import Image
import pytest
from safetensors.torch import load_file, save_file
import toml
import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.cache_io import save_latent_cache_qwen_image, save_text_encoder_output_cache_qwen_image
from musubi_tuner.training.validation_inputs import (
    prepare_validation_inputs,
    prepare_cache_batch,
    validate_cache_dataset,
)


def make_data(root, name, color, size=(32, 32)):
    directory = root / "dataset" / name
    directory.mkdir(parents=True)
    Image.new("RGB", size, color).save(directory / "one.png")
    (directory / "one.txt").write_text("one caption", encoding="utf-8")
    config = root / f"{name}.toml"
    config.write_text(
        toml.dumps(
            dict(
                general=dict(
                    resolution=[32, 32],
                    caption_extension=".txt",
                    batch_size=1,
                    num_repeats=1,
                    enable_bucket=True,
                    bucket_no_upscale=True,
                ),
                datasets=[dict(image_directory=f"dataset/{name}", cache_directory=f"cache/{name}")],
            )
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        dataset_config=str(config), experiment_mode=True, validation=name == "val", batch_size=None, skip_existing=False
    )
    blueprint = config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
        config_utils.load_user_config(config), args, architecture="qi"
    )
    group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    for dataset in group.datasets:
        for _, batch in dataset.retrieve_latent_cache_batches(1):
            prepare_cache_batch(batch, group.datasets, args)
            for item in batch:
                save_latent_cache_qwen_image(item, torch.ones(16, 1, item.bucket_size[1] // 8, item.bucket_size[0] // 8))
                save_text_encoder_output_cache_qwen_image(item, torch.ones(2, 3584))
        dataset.prepare_for_training()
    group.num_train_items = sum(dataset.num_train_items for dataset in group.datasets)
    return config, group


@pytest.fixture
def fixture(tmp_path):
    train, group = make_data(tmp_path, "train", "red")
    val, _ = make_data(tmp_path, "val", "blue")
    args = argparse.Namespace(
        dataset_config=str(train),
        val_dataset_config=str(val),
        experiment_mode=True,
        val_seed_noise=42,
        val_every_n_steps=50,
        val_level_noise_n=10,
        val_seed_noise_n=2,
    )
    return args, group


def test_strict_inputs_batch_manifest_rng_and_moving(tmp_path, fixture):
    args, group = fixture
    before = random.getstate()
    inputs = prepare_validation_inputs(args, group)
    assert random.getstate() == before
    assert len(inputs.records) == 1
    batch = inputs.load_batch(0)
    assert batch["latents"].shape == (1, 16, 1, 4, 4)
    assert batch["vl_embed"][0].shape == (2, 3584)
    assert batch["timesteps"] is None
    inputs.verify_unchanged()
    moved = tmp_path / "moved"
    shutil.copytree(tmp_path / "dataset", moved / "dataset")
    shutil.copytree(tmp_path / "cache", moved / "cache")
    shutil.copy(args.dataset_config, moved / "train.toml")
    shutil.copy(args.val_dataset_config, moved / "val.toml")
    moved_args = argparse.Namespace(**vars(args))
    moved_args.dataset_config, moved_args.val_dataset_config = str(moved / "train.toml"), str(moved / "val.toml")
    blueprint = config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
        config_utils.load_user_config(moved_args.dataset_config), moved_args, architecture="qi"
    )
    moved_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group, training=True)
    moved_inputs = prepare_validation_inputs(moved_args, moved_group)
    assert moved_inputs.fingerprint == inputs.fingerprint


@pytest.mark.parametrize(
    "kind",
    ["latent_missing", "text_missing", "corrupt", "nonfinite", "stale_caption", "stale_image", "bad_shape", "extra_train_cache"],
)
def test_bad_caches_fail_before_model(tmp_path, fixture, kind):
    args, group = fixture
    latent = next((tmp_path / "cache/val").glob("*_qi.safetensors"))
    text = next((tmp_path / "cache/val").glob("*_te.safetensors"))
    if kind == "latent_missing":
        latent.unlink()
    elif kind == "text_missing":
        text.unlink()
    elif kind == "corrupt":
        latent.write_bytes(b"broken")
    elif kind in ("nonfinite", "bad_shape"):
        from safetensors import safe_open

        with safe_open(str(latent), framework="pt") as opened:
            metadata = opened.metadata()
        tensors = load_file(str(latent))
        key = next(iter(tensors))
        tensors[key] = tensors[key][:2].clone() if kind == "bad_shape" else torch.full_like(tensors[key], float("nan"))
        save_file(tensors, str(latent), metadata=metadata)
    elif kind == "stale_caption":
        (tmp_path / "dataset/val/one.txt").write_text("changed", encoding="utf-8")
    elif kind == "stale_image":
        Image.new("RGB", (32, 32), "green").save(tmp_path / "dataset/val/one.png")
    else:
        shutil.copy(
            next((tmp_path / "cache/train").glob("*_qi.safetensors")), tmp_path / "cache/train/foreign_0032x0032_qi.safetensors"
        )
        shutil.copy(next((tmp_path / "cache/train").glob("*_te.safetensors")), tmp_path / "cache/train/foreign_qi_te.safetensors")
        group.datasets[0].prepare_for_training()
    with pytest.raises(ValueError):
        prepare_validation_inputs(args, group)


def test_overlap_and_runtime_mutation(tmp_path, fixture):
    args, group = fixture
    inputs = prepare_validation_inputs(args, group)
    (tmp_path / "dataset/val/one.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed|stale|provenance"):
        inputs.verify_unchanged()
    shutil.copy(tmp_path / "dataset/train/one.png", tmp_path / "dataset/val/one.png")
    with pytest.raises(ValueError, match="overlap"):
        prepare_validation_inputs(args, group)


@pytest.mark.parametrize(
    "key,value", [("batch_size", 2), ("num_repeats", 2), ("caption_dropout_rate", 0.1), ("shuffle_caption", True)]
)
def test_invalid_val_declaration(tmp_path, fixture, key, value):
    args, group = fixture
    config = toml.load(args.val_dataset_config)
    config["general"][key] = value
    Path(args.val_dataset_config).write_text(toml.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=key):
        prepare_validation_inputs(args, group)


def test_collision_and_empty(tmp_path, fixture):
    args, group = fixture
    Image.new("RGB", (32, 32), "green").save(tmp_path / "dataset/val/one.jpg")
    with pytest.raises(ValueError, match="collision"):
        prepare_validation_inputs(args, group)
    (tmp_path / "dataset/val/one.jpg").unlink()
    (tmp_path / "dataset/val/one.png").unlink()
    with pytest.raises(ValueError, match="no images|empty"):
        prepare_validation_inputs(args, group)


def test_validation_image_without_caption_cannot_be_filtered_out(tmp_path, fixture):
    args, group = fixture
    Image.new("RGB", (32, 32), "green").save(tmp_path / "dataset/val/missing.png")
    with pytest.raises(ValueError, match="missing caption.*missing.png"):
        prepare_validation_inputs(args, group)


def test_cache_validation_rejects_bad_batch_and_stale_skip(tmp_path, fixture):
    args, group = fixture
    cache_args = argparse.Namespace(
        dataset_config=args.val_dataset_config, validation=True, experiment_mode=True, skip_existing=True
    )
    declaration = config_utils.load_user_config(args.val_dataset_config)
    blueprint = config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(declaration, cache_args, architecture="qi")
    val_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    validate_cache_dataset(val_group, cache_args)
    (tmp_path / "dataset/val/one.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance|stale"):
        validate_cache_dataset(val_group, cache_args)
    val_group.datasets[0].batch_size = 2
    with pytest.raises(ValueError, match="batch_size"):
        validate_cache_dataset(val_group, cache_args)


def test_multiple_val_datasets_order_and_renamed_files(tmp_path, fixture):
    args, group = fixture
    extra_config, _ = make_data(tmp_path, "other", "green", (16, 32))
    config = toml.load(args.val_dataset_config)
    config["datasets"].extend(toml.load(extra_config)["datasets"])
    Path(args.val_dataset_config).write_text(toml.dumps(config), encoding="utf-8")
    inputs = prepare_validation_inputs(args, group)
    assert len(inputs.records) == 2
    assert [record.image_sha256 for record in inputs.records] == sorted(record.image_sha256 for record in inputs.records)
    assert {tuple(inputs.load_batch(index)["latents"].shape[-2:]) for index in range(2)} == {(4, 4), (4, 2)}
    config["datasets"].reverse()
    Path(args.val_dataset_config).write_text(toml.dumps(config), encoding="utf-8")
    for directory in (tmp_path / "dataset/val", tmp_path / "cache/val"):
        for path in list(directory.glob("one*")):
            path.rename(path.with_name(path.name.replace("one", "renamed", 1)))
    assert prepare_validation_inputs(args, group).fingerprint == inputs.fingerprint


@pytest.mark.parametrize("which", ["latent", "text"])
def test_real_cache_entrypoints_write_validation_provenance(tmp_path, monkeypatch, which):
    from musubi_tuner import qwen_image_cache_latents as latent_cache
    from musubi_tuner import qwen_image_cache_text_encoder_outputs as text_cache
    from safetensors import safe_open

    config, _ = make_data(tmp_path, "val", "blue")
    weight = tmp_path / "weights.safetensors"
    weight.touch()
    calls = []

    class Vae:
        device = torch.device("cpu")
        dtype = torch.float32

        def to(self, device):
            return self

        def eval(self):
            calls.append("eval")

        def encode_pixels_to_latents(self, pixels):
            return torch.ones(pixels.shape[0], 16, 1, pixels.shape[-2] // 8, pixels.shape[-1] // 8)

    monkeypatch.setattr(latent_cache.qwen_image_utils, "load_vae", lambda *a, **kw: Vae())
    monkeypatch.setattr(
        text_cache.qwen_image_utils, "load_qwen2_5_vl", lambda **kw: (None, SimpleNamespace(eval=lambda: calls.append("eval")))
    )
    monkeypatch.setattr(
        text_cache.qwen_image_utils,
        "get_qwen_prompt_embeds",
        lambda tokenizer, model, prompts: (torch.ones(len(prompts), 3, 3584), torch.ones(len(prompts), 3)),
    )
    # Exercise each actual CLI after deleting only the corresponding fixture cache.
    suffix = "*_qi.safetensors" if which == "latent" else "*_te.safetensors"
    for path in (tmp_path / "cache/val").glob(suffix):
        path.unlink()
    monkeypatch.chdir(tmp_path.parent)
    flag = "--vae" if which == "latent" else "--text_encoder"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fixture",
            "--dataset_config",
            str(config),
            flag,
            "weights.safetensors",
            "--experiment_mode",
            "--validation",
            "--num_workers",
            "1",
            "--device",
            "cpu",
        ],
    )
    (latent_cache if which == "latent" else text_cache).main()
    assert calls == ["eval"]
    with safe_open(str(next((tmp_path / "cache/val").glob(suffix))), framework="pt") as opened:
        assert {"qwen_source_sha256", "qwen_caption_sha256", "qwen_preparation_sha256"} <= opened.metadata().keys()
