"""Source-complete cache discovery, without encoders or training buckets."""

import argparse
from dataclasses import FrozenInstanceError
from pathlib import Path
import random
import shutil

from PIL import Image
import pytest
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import toml
import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.cache_io import save_latent_cache_qwen_image, save_text_encoder_output_cache_qwen_image
from musubi_tuner.training import validation_inputs as inputs


def declaration(root, name, colors, **general_overrides):
    directory = root / "dataset" / name
    directory.mkdir(parents=True)
    for index, color in enumerate(colors):
        Image.new("RGB", (32, 32), color).save(directory / f"image{index}.png")
        (directory / f"image{index}.txt").write_text(f"caption {index}", encoding="utf-8")
    path = root / f"{name}.toml"
    path.write_text(
        toml.dumps(
            {
                "general": {
                    "resolution": [32, 32],
                    "caption_extension": ".txt",
                    "batch_size": 1,
                    "num_repeats": 1,
                    "enable_bucket": True,
                    "bucket_no_upscale": True,
                    **general_overrides,
                },
                "datasets": [{"image_directory": f"dataset/{name}", "cache_directory": f"cache/{name}"}],
            }
        ),
        encoding="utf-8",
    )
    return path


def arguments(root, validation=True):
    train = declaration(root, "train", ["red", "green"], batch_size=2, num_repeats=3)
    val = declaration(root, "val", ["blue"]) if validation else None
    return argparse.Namespace(
        dataset_config=str(train),
        val_dataset_config=str(val) if val else None,
        experiment_mode=True,
        val_seed_noise=42,
        val_every_n_steps=2,
        val_level_noise_n=2,
        val_seed_noise_n=1,
    )


def source_group(args, validation=False, training=False):
    local = argparse.Namespace(**vars(args))
    local.dataset_config = args.val_dataset_config if validation else args.dataset_config
    blueprint = config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
        config_utils.load_user_config(local.dataset_config), local, architecture="qi"
    )
    return config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group, training=training)


def write_caches(args, validation=False):
    group = source_group(args, validation)
    cache_args = argparse.Namespace(experiment_mode=True, validation=validation)
    paths = []
    for dataset in group.datasets:
        for _, batch in dataset.retrieve_latent_cache_batches(1):
            inputs.prepare_cache_batch(batch, group.datasets, cache_args)
            for item in batch:
                width, height = item.bucket_size
                save_latent_cache_qwen_image(item, torch.ones(16, 1, height // 8, width // 8))
                save_text_encoder_output_cache_qwen_image(item, torch.ones(2, 3584))
                paths.append((Path(item.latent_cache_path), Path(item.text_encoder_output_cache_path)))
    return sorted(paths)


def file_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("validation", [False, True])
def test_absent_inventory_is_source_complete_absolute_and_read_only(tmp_path, monkeypatch, validation):
    root = tmp_path / "experiment with spaces"
    args = arguments(root, validation)
    monkeypatch.chdir(tmp_path)
    before = file_bytes(root)
    rng = random.getstate()
    result = inputs.inspect_cache_inputs(args)
    assert isinstance(result, tuple) and len(result) == (2 if validation else 1)
    for inventory, count, role in zip(result, [2, 1], [False, True]):
        assert isinstance(inventory, inputs.CacheInventory)
        assert inventory.validation is role
        assert Path(inventory.dataset_config).is_absolute()
        assert len(inventory.missing_latents) == len(inventory.missing_text) == count
        assert isinstance(inventory.missing_latents, tuple) and isinstance(inventory.missing_text, tuple)
        assert all(Path(path).is_absolute() for path in inventory.missing_latents + inventory.missing_text)
        with pytest.raises(FrozenInstanceError):
            inventory.validation = not role
    assert random.getstate() == rng
    assert file_bytes(root) == before
    assert not (root / "cache").exists()


@pytest.mark.parametrize("split", ["train", "val"])
def test_missing_caption_is_not_silently_excluded_in_auto_mode(tmp_path, split):
    args = arguments(tmp_path)
    (tmp_path / "dataset" / split / "image0.txt").unlink()
    before = file_bytes(tmp_path)
    with pytest.raises(ValueError, match="caption") as error:
        inputs.inspect_cache_inputs(args)
    assert str(tmp_path / f"{split}.toml") in str(error.value)
    assert file_bytes(tmp_path) == before
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("kind", ["overlap", "collision", "shared_cache", "nested_cache", "val_batch", "val_repeats"])
def test_source_errors_precede_any_cache_creation(tmp_path, kind):
    args = arguments(tmp_path)
    val = toml.load(args.val_dataset_config)
    if kind == "overlap":
        shutil.copyfile(tmp_path / "dataset/train/image0.png", tmp_path / "dataset/val/image0.png")
    elif kind == "collision":
        Image.new("RGB", (64, 32), "yellow").save(tmp_path / "dataset/train/image0.jpg")
    elif kind in ("shared_cache", "nested_cache"):
        val["datasets"][0]["cache_directory"] = "cache/train" + ("/val" if kind == "nested_cache" else "")
    else:
        val["general"]["batch_size" if kind == "val_batch" else "num_repeats"] = 2
    Path(args.val_dataset_config).write_text(toml.dumps(val), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap|collision|must be 1"):
        inputs.inspect_cache_inputs(args)
    assert not (tmp_path / "cache").exists()


def test_complete_and_partial_inventory_preserve_existing_bytes(tmp_path):
    args = arguments(tmp_path)
    train = write_caches(args)
    val = write_caches(args, validation=True)
    unrelated = tmp_path / "cache/train/notes.txt"
    unrelated.write_text("preserved", encoding="utf-8")
    before = file_bytes(tmp_path)
    complete = inputs.inspect_cache_inputs(args)
    assert all(not item.missing_latents and not item.missing_text for item in complete)
    assert file_bytes(tmp_path) == before

    train[0][1].unlink()
    val[0][0].unlink()
    before = file_bytes(tmp_path)
    partial = inputs.inspect_cache_inputs(args)
    assert partial[0].missing_latents == ()
    assert partial[0].missing_text == (str(train[0][1]),)
    assert partial[1].missing_latents == (str(val[0][0]),)
    assert partial[1].missing_text == ()
    assert file_bytes(tmp_path) == before


@pytest.mark.parametrize("validation", [False, True])
@pytest.mark.parametrize("kind", ["corrupt_latent", "corrupt_text", "provenance", "caption", "image", "shape", "nonfinite"])
def test_present_invalid_cache_is_an_error_not_missing(tmp_path, validation, kind):
    args = arguments(tmp_path)
    paths = write_caches(args, validation)
    latent, text = paths[0]
    split = "val" if validation else "train"
    if kind.startswith("corrupt_"):
        (text if kind == "corrupt_text" else latent).write_bytes(b"not safetensors")
    elif kind == "caption":
        (tmp_path / "dataset" / split / "image0.txt").write_text("changed", encoding="utf-8")
    elif kind == "image":
        Image.new("RGB", (32, 32), "yellow").save(tmp_path / "dataset" / split / "image0.png")
    else:
        with safe_open(str(latent), framework="pt") as opened:
            metadata = opened.metadata()
        tensors = {key: value.clone() for key, value in load_file(str(latent)).items()}
        key = next(iter(tensors))
        if kind == "provenance":
            metadata.pop("qwen_source_sha256")
        elif kind == "shape":
            tensors[key] = tensors[key][:2].clone()
        else:
            tensors[key] = torch.full_like(tensors[key], float("nan"))
        save_file(tensors, str(latent), metadata=metadata)
    before = file_bytes(tmp_path)
    with pytest.raises(ValueError, match="invalid or stale cache") as error:
        inputs.inspect_cache_inputs(args)
    assert str(tmp_path / f"{split}.toml") in str(error.value)
    assert file_bytes(tmp_path) == before


@pytest.mark.parametrize("kind", ["directory", "broken_link"])
@pytest.mark.parametrize("stage", ["latent", "text"])
def test_present_non_file_path_is_not_a_creation_target(tmp_path, monkeypatch, kind, stage):
    args = arguments(tmp_path, validation=False)
    inventory = inputs.inspect_cache_inputs(args)[0]
    target = Path((inventory.missing_latents if stage == "latent" else inventory.missing_text)[0])
    target.parent.mkdir(parents=True)
    if kind == "directory":
        target.mkdir()
    else:
        try:
            target.symlink_to(tmp_path / "absent-link-target")
        except OSError:
            # Windows can deny symlink creation. Exercise its observed metadata boundary
            # (exists=False, is_symlink=True) without requiring elevated filesystem access.
            original = Path.is_symlink
            monkeypatch.setattr(Path, "is_symlink", lambda path: path == target or original(path))
    with pytest.raises(ValueError, match="regular file") as error:
        inputs.inspect_cache_inputs(args)
    assert str(target) in str(error.value)
    assert target.is_dir() if kind == "directory" else target.is_symlink()


def test_unexpected_training_latent_is_rejected_even_without_text_pair(tmp_path):
    args = arguments(tmp_path, validation=False)
    paths = write_caches(args)
    unexpected = paths[0][0].parent / "foreign_0032x0032_qi.safetensors"
    shutil.copyfile(paths[0][0], unexpected)
    before = file_bytes(tmp_path)
    with pytest.raises(ValueError, match="unexpected training latent cache"):
        inputs.inspect_cache_inputs(args)
    assert file_bytes(tmp_path) == before


def test_unrelated_text_and_val_cache_files_are_preserved(tmp_path):
    args = arguments(tmp_path)
    write_caches(args)
    write_caches(args, validation=True)
    for name in ["cache/train/foreign_qi_te.safetensors", "cache/val/foreign_0032x0032_qi.safetensors"]:
        (tmp_path / name).write_bytes(b"not consumed by training or validation")
    before = file_bytes(tmp_path)
    assert all(not row.missing_latents and not row.missing_text for row in inputs.inspect_cache_inputs(args))
    assert file_bytes(tmp_path) == before


def test_multiple_source_datasets_aggregate_under_one_inventory(tmp_path):
    args = arguments(tmp_path, validation=False)
    extra = declaration(tmp_path, "extra", ["yellow"])
    config = toml.load(args.dataset_config)
    config["datasets"].extend(toml.load(extra)["datasets"])
    Path(args.dataset_config).write_text(toml.dumps(config), encoding="utf-8")
    rows = inputs.inspect_cache_inputs(args)
    assert len(rows) == 1
    assert len(rows[0].missing_latents) == len(rows[0].missing_text) == 3
    assert {Path(path).parent.name for path in rows[0].missing_latents} == {"train", "extra"}


def test_jsonl_inventory_uses_jsonl_root_from_another_cwd(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    nested = tmp_path / "declarations"
    nested.mkdir()
    records = nested / "images.jsonl"
    records.write_text('{"image_path":"../dataset/val/image0.png","caption":"JSONL caption"}\n', encoding="utf-8")
    cfg = toml.load(args.val_dataset_config)
    cfg["datasets"] = [{"image_jsonl_file": "declarations/images.jsonl", "cache_directory": "cache/jsonl"}]
    Path(args.val_dataset_config).write_text(toml.dumps(cfg), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    val = inputs.inspect_cache_inputs(args)[1]
    assert val.missing_latents == (str(tmp_path / "cache/jsonl/image0_0032x0032_qi.safetensors"),)
    assert val.missing_text == (str(tmp_path / "cache/jsonl/image0_qi_te.safetensors"),)


def test_inventory_does_not_replace_fixed_validation_fingerprint_guard(tmp_path):
    args = arguments(tmp_path)
    write_caches(args)
    val = write_caches(args, validation=True)
    group = source_group(args, training=True)
    fixed = inputs.prepare_validation_inputs(args, group)
    original = val[0][1].read_bytes()
    val[0][1].unlink()
    assert inputs.inspect_cache_inputs(args)[1].missing_text == (str(val[0][1]),)
    val[0][1].write_bytes(original)
    assert inputs.prepare_validation_inputs(args, group).fingerprint == fixed.fingerprint
    with safe_open(str(val[0][1]), framework="pt") as opened:
        metadata = opened.metadata()
    tensors = {key: value.clone() for key, value in load_file(str(val[0][1])).items()}
    save_file({key: value + 1 for key, value in tensors.items()}, str(val[0][1]), metadata=metadata)
    assert not inputs.inspect_cache_inputs(args)[1].missing_text
    with pytest.raises(ValueError, match="validation inputs changed"):
        fixed.verify_unchanged()


def test_blank_caption_remains_valid_and_manual_train_filtering_unchanged(tmp_path):
    args = arguments(tmp_path)
    (tmp_path / "dataset/train/image0.txt").write_text("", encoding="utf-8")
    assert len(inputs.inspect_cache_inputs(args)[0].missing_latents) == 2
    write_caches(args)
    write_caches(args, validation=True)
    (tmp_path / "dataset/train/image1.txt").unlink()
    # Existing manual loading ignores the image without a caption and its absent cache.
    for path in (tmp_path / "cache/train").glob("image1_*.safetensors"):
        path.unlink()
    group = source_group(args, training=True)
    assert inputs.prepare_validation_inputs(args, group) is not None
    with pytest.raises(ValueError, match="missing caption"):
        inputs.inspect_cache_inputs(args)
