"""Image declarations, actual cache IO and batching; no model weights required."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
from safetensors import safe_open
from safetensors.torch import load_file
import toml
import torch

from musubi_tuner import cache_latents, cache_text_encoder_outputs
from musubi_tuner import qwen_image_cache_latents, qwen_image_cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.cache_io import save_latent_cache_qwen_image, save_text_encoder_output_cache_qwen_image
from musubi_tuner.dataset.datasources import ImageJsonlDatasource
from musubi_tuner.dataset.image_video_dataset import ImageDataset


def image_source(path):
    path.mkdir(parents=True)
    for index, name in enumerate(("a", "b")):
        Image.new("RGB", (64, 64), (index * 255, 127, 0)).save(path / f"{name}.png")
        (path / f"{name}.txt").write_text(f" caption {name} \n", encoding="utf-8")
    return path


def dataset(path, **overrides):
    return ImageDataset(
        resolution=(64, 64),
        caption_extension=".txt",
        batch_size=2,
        num_repeats=2,
        enable_bucket=True,
        bucket_no_upscale=True,
        image_directory=str(path),
        cache_directory=str(path / "cache"),
        architecture="qi",
        **overrides,
    )


def role_declaration(tmp_path):
    familiar = image_source(tmp_path / "familiar")
    unfamiliar = image_source(tmp_path / "unfamiliar")
    declaration = {
        "general": {"resolution": 64, "caption_extension": ".txt", "batch_size": 1, "num_repeats": 1},
        "datasets": [
            {"role": "val_familiar", "image_directory": str(familiar), "cache_directory": str(tmp_path / "familiar_cache")},
            {"role": "val_unfamiliar", "image_directory": str(unfamiliar), "cache_directory": str(tmp_path / "unfamiliar_cache")},
        ],
    }
    return declaration, familiar, unfamiliar


def run_cache_preflight(tmp_path, monkeypatch, declaration, module, extra_args=()):
    config = tmp_path / "val-dataset.toml"
    config.write_text(toml.dumps(declaration), encoding="utf-8")
    weight = tmp_path / "unused.safetensors"
    weight.touch()
    model_loader = "load_vae" if module is qwen_image_cache_latents else "load_qwen2_5_vl"
    weight_option = "--vae" if module is qwen_image_cache_latents else "--text_encoder"
    reached = []

    def stop_at_model(*args, **kwargs):
        reached.append(True)
        raise RuntimeError("model boundary")

    monkeypatch.setattr(module.qwen_image_utils, model_loader, stop_at_model)
    monkeypatch.setattr(sys, "argv", ["fixture", "--dataset_config", str(config), weight_option, str(weight), *extra_args])
    return config, reached


@pytest.mark.parametrize("module", [qwen_image_cache_latents, qwen_image_cache_text_encoder_outputs])
def test_role_dataset_reaches_cache_model_only_after_preflight(tmp_path, monkeypatch, module):
    declaration, _, _ = role_declaration(tmp_path)
    _, reached = run_cache_preflight(tmp_path, monkeypatch, declaration, module)
    with pytest.raises(RuntimeError, match="model boundary"):
        module.main()
    assert reached == [True]


@pytest.mark.parametrize("module", [qwen_image_cache_latents, qwen_image_cache_text_encoder_outputs])
def test_role_cache_skip_existing_rejected_before_model(tmp_path, monkeypatch, module):
    declaration, _, _ = role_declaration(tmp_path)
    config, reached = run_cache_preflight(tmp_path, monkeypatch, declaration, module, ["--skip_existing"])
    with pytest.raises(ValueError, match="skip_existing") as error:
        module.main()
    assert str(config) in str(error.value)
    assert reached == []


@pytest.mark.parametrize("module", [qwen_image_cache_latents, qwen_image_cache_text_encoder_outputs])
@pytest.mark.parametrize("failure", ["missing_caption", "empty_role", "cache_collision"])
def test_role_source_errors_precede_cache_models(tmp_path, monkeypatch, module, failure):
    declaration, familiar, unfamiliar = role_declaration(tmp_path)
    if failure == "missing_caption":
        Image.new("RGB", (64, 64)).save(familiar / "orphan.png")
    elif failure == "empty_role":
        for path in unfamiliar.glob("*.png"):
            path.unlink()
    elif failure == "cache_collision":
        Image.new("RGB", (64, 64), (1, 2, 3)).save(familiar / "a.jpg")
    config, reached = run_cache_preflight(tmp_path, monkeypatch, declaration, module)
    with pytest.raises(ValueError) as error:
        module.main()
    assert str(config) in str(error.value)
    assert reached == []


@pytest.mark.parametrize("module", [qwen_image_cache_latents, qwen_image_cache_text_encoder_outputs])
@pytest.mark.parametrize("difference", ["none", "bucket", "preprocessing"])
def test_shared_source_latent_cache_requires_matching_preprocessing(tmp_path, monkeypatch, module, difference):
    declaration, familiar, _ = role_declaration(tmp_path)
    shared_cache = tmp_path / "shared_cache"
    declaration["datasets"][0]["cache_directory"] = str(shared_cache)
    declaration["datasets"][1]["image_directory"] = str(familiar)
    # Distinct spellings reach the same files and pass the existing string-only cache-dir check.
    declaration["datasets"][1]["cache_directory"] = str(shared_cache) + "/."
    if difference == "bucket":
        declaration["datasets"][1]["resolution"] = 128
    elif difference == "preprocessing":
        declaration["datasets"][1]["enable_bucket"] = True
    config, reached = run_cache_preflight(tmp_path, monkeypatch, declaration, module)
    if difference == "none":
        with pytest.raises(RuntimeError, match="model boundary"):
            module.main()
        assert reached == [True]
    else:
        with pytest.raises(ValueError, match="latent cache.*preprocessing") as error:
            module.main()
        assert str(config) in str(error.value)
        assert reached == []


@pytest.mark.parametrize("roles", [["val_familiar"], ["val_familiar", "val_familiar"], ["val_familiar", "wrong"]])
def test_role_declaration_requires_both_known_roles(tmp_path, roles):
    declaration, _, _ = role_declaration(tmp_path)
    declaration["datasets"] = declaration["datasets"][: len(roles)]
    for entry, role in zip(declaration["datasets"], roles):
        entry["role"] = role
    with pytest.raises(ValueError, match="role"):
        config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
            declaration, argparse.Namespace(dataset_config="val-dataset.toml"), architecture="qi"
        )


@pytest.mark.parametrize("key,value", [("batch_size", 2), ("num_repeats", 2)])
def test_role_declaration_rejects_effective_training_batch_or_repeats(tmp_path, key, value):
    declaration, _, _ = role_declaration(tmp_path)
    declaration["general"][key] = value
    with pytest.raises(ValueError, match=key):
        config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
            declaration, argparse.Namespace(dataset_config="val-dataset.toml"), architecture="qi"
        )


def test_role_jsonl_keeps_exact_duplicates_but_rejects_conflicting_caption(tmp_path):
    declaration, familiar, _ = role_declaration(tmp_path)
    image_path = familiar / "a.png"
    jsonl = tmp_path / "familiar.jsonl"
    records = [{"image_path": str(image_path), "caption": "caption a"}] * 2
    jsonl.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    declaration["datasets"][0].pop("image_directory")
    declaration["datasets"][0]["image_jsonl_file"] = str(jsonl)

    def preflight():
        blueprint = config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
            declaration, argparse.Namespace(dataset_config=str(tmp_path / "val-dataset.toml")), architecture="qi"
        )
        group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group, seed=0)
        config_utils.validate_role_aware_sources(group, str(tmp_path / "val-dataset.toml"))
        return group

    group = preflight()
    assert len(group.datasets[0].datasource) == 2
    records[1] = {**records[1], "caption": "another caption"}
    jsonl.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    with pytest.raises(ValueError, match="cache"):
        preflight()


def test_role_cache_writers_bind_source_and_replace_text_metadata(tmp_path, monkeypatch):
    source = image_source(tmp_path / "images")
    ds = dataset(source, role="val_familiar")
    ds.dataset_index = 0
    image_item = next(ds.retrieve_latent_cache_batches(1))[1][0]
    expected_sha = hashlib.sha256(Path(image_item.item_key).read_bytes()).hexdigest()

    class EncoderBoundary:
        device = torch.device("cpu")
        dtype = torch.float32

        def encode_pixels_to_latents(self, pixels):
            return pixels[:, :2, :, ::8, ::8]

    qwen_image_cache_latents.encode_and_save_batch(EncoderBoundary(), [image_item], ["val_familiar"])
    with safe_open(image_item.latent_cache_path, framework="pt") as saved:
        assert saved.metadata()["source_image_sha256"] == expected_sha
        assert set(saved.keys()) == {"latents_1x8x8_float32"}

    text_item = next(ds.retrieve_text_encoder_output_cache_batches(1))[0]
    save_text_encoder_output_cache_qwen_image(text_item, torch.ones(2, 4), source_image_sha256="0" * 64)

    def encoder_boundary(tokenizer, encoder, prompts):
        return torch.ones(1, 3, 4), torch.tensor([[1, 1, 0]])

    monkeypatch.setattr(qwen_image_cache_text_encoder_outputs.qwen_image_utils, "get_qwen_prompt_embeds", encoder_boundary)
    qwen_image_cache_text_encoder_outputs.encode_and_save_batch(
        None, None, [text_item], torch.device("cpu"), None, ["val_familiar"]
    )
    with safe_open(text_item.text_encoder_output_cache_path, framework="pt") as saved:
        assert saved.metadata()["source_image_sha256"] == expected_sha
        assert saved.metadata()["caption1"] == text_item.caption
        assert set(saved.keys()) == {"varlen_vl_embed_float32"}
        assert saved.get_tensor("varlen_vl_embed_float32").shape == (2, 4)


@pytest.mark.parametrize("extension", ["toml", "json"])
@pytest.mark.parametrize("source_kind", ["directory", "jsonl"])
def test_real_declarations_fallback_and_association(tmp_path, extension, source_kind):
    first = image_source(tmp_path / "first")
    second = image_source(tmp_path / "second")
    declaration = {
        "general": {
            "resolution": [64, 64],
            "caption_extension": ".txt",
            "enable_bucket": True,
            "bucket_no_upscale": True,
            "batch_size": 2,
            "num_repeats": 3,
        },
        "datasets": [
            {"image_directory": str(first), "cache_directory": str(first / "cache")},
            {"image_directory": str(second), "cache_directory": str(second / "cache"), "batch_size": 1, "num_repeats": 4},
        ],
    }
    if source_kind == "jsonl":
        records = [{"image_path": str(first / f"{name}.png"), "caption": f"caption {name}"} for name in ("a", "b")]
        jsonl = tmp_path / "images.jsonl"
        jsonl.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        declaration["datasets"][0].pop("image_directory")
        declaration["datasets"][0]["image_jsonl_file"] = str(jsonl)
    config = tmp_path / f"dataset.{extension}"
    config.write_text(toml.dumps(declaration) if extension == "toml" else json.dumps(declaration), encoding="utf-8")
    parsed = config_utils.load_user_config(config)
    blueprint = config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
        parsed, argparse.Namespace(batch_size=9, num_repeats=9), architecture="qi"
    )
    group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    assert [(d.batch_size, d.num_repeats) for d in group.datasets] == [(2, 3), (1, 4)]
    for index, ds in enumerate(group.datasets):
        items = [item for _, batch in ds.retrieve_latent_cache_batches(1) for item in batch]
        assert sorted((item.caption, item.datasource_index) for item in items) == [("caption a", 0), ("caption b", 1)]
        assert {item.dataset_index for item in items} == {index}
        assert all(item.content.shape == (64, 64, 3) for item in items)
        assert all(item.bucket_size == (64, 64) for item in items)


def test_no_upscale_and_cache_round_trip(tmp_path):
    path = image_source(tmp_path / "images")
    ds = dataset(path)
    ds.resolution = (128, 128)
    items = [item for _, batch in ds.retrieve_latent_cache_batches(1) for item in batch]
    for index, item in enumerate(items):
        assert item.content.shape == (64, 64, 3)
        assert Path(item.latent_cache_path).name == f"{Path(item.item_key).stem}_0064x0064_qi.safetensors"
        save_latent_cache_qwen_image(item, torch.full((2, 1, 8, 8), float(index)))
        save_text_encoder_output_cache_qwen_image(item, torch.full((index + 2, 4), float(index)))
        with safe_open(item.latent_cache_path, framework="pt") as saved:
            assert set(saved.keys()) == {"latents_1x8x8_float32"}
            assert saved.metadata() == {"architecture": "qwen_image", "format_version": "1.0.1", "width": "64", "height": "64"}
        with safe_open(item.text_encoder_output_cache_path, framework="pt") as saved:
            assert saved.metadata()["caption1"] == item.caption
    ds.prepare_for_training()
    ds.set_seed(7, SimpleNamespace(value=0))
    assert ds.num_train_items == 4 and len(ds) == 2
    cached_items = next(iter(ds.batch_manager.buckets.values()))
    assert cached_items[0] is cached_items[1] and cached_items[2] is cached_items[3]
    batches = [ds[i] for i in range(len(ds))]
    assert all(batch["latents"].shape == (2, 2, 1, 8, 8) for batch in batches)
    assert sorted(x.shape[0] for batch in batches for x in batch["vl_embed"]) == [2, 2, 3, 3]
    assert all(batch["timesteps"] is None for batch in batches)


def test_dtype_replacement_preserves_unrelated_keys(tmp_path):
    ds = dataset(image_source(tmp_path / "images"))
    item = next(ds.retrieve_latent_cache_batches(1))[1][0]
    save_text_encoder_output_cache_qwen_image(item, torch.ones(2, 4, dtype=torch.float16))
    from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_common

    save_text_encoder_output_cache_common(item, {"extra_float32": torch.tensor([3.0])}, "qwen_image")
    save_text_encoder_output_cache_qwen_image(item, torch.full((3, 4), 2.0, dtype=torch.bfloat16))
    tensors = load_file(item.text_encoder_output_cache_path)
    assert set(tensors) == {"varlen_vl_embed_bfloat16", "extra_float32"}
    torch.testing.assert_close(tensors["extra_float32"], torch.tensor([3.0]))


def test_missing_text_cache_warns_and_skips(tmp_path, caplog):
    ds = dataset(image_source(tmp_path / "images"))
    item = next(ds.retrieve_latent_cache_batches(1))[1][0]
    save_latent_cache_qwen_image(item, torch.ones(2, 1, 8, 8))
    ds.prepare_for_training()
    assert len(ds) == 0 and ds.num_train_items == 0
    assert "Text encoder output cache file not found" in caplog.text


@pytest.mark.parametrize("keep", [True, False])
def test_latent_orchestration_skip_and_cleanup(tmp_path, keep):
    ds = dataset(image_source(tmp_path / "images"))
    cache = Path(ds.cache_directory)
    cache.mkdir()
    stale = cache / "stale_0064x0064_qi.safetensors"
    stale.write_bytes(b"stale")
    calls = []

    class EncoderBoundary:
        device = torch.device("cpu")
        dtype = torch.float32

        def encode_pixels_to_latents(self, pixels):
            calls.append(pixels.clone())
            assert pixels.shape == (1, 3, 1, 64, 64)
            assert pixels.min() >= -1 and pixels.max() <= 1
            return pixels[:, :2, :, ::8, ::8]

    def encode(items):
        qwen_image_cache_latents.encode_and_save_batch(EncoderBoundary(), items)

    args = argparse.Namespace(num_workers=1, batch_size=1, skip_existing=False, keep_cache=keep)
    cache_latents.encode_datasets([ds], encode, args)
    assert len(calls) == 2 and stale.exists() == keep
    args.skip_existing = True
    (Path(ds.image_directory) / "a.txt").write_text("changed caption", encoding="utf-8")
    cache_latents.encode_datasets([ds], encode, args)
    assert len(calls) == 2  # existence-only reuse, independent of caption freshness
    for tensor in calls:
        torch.testing.assert_close(tensor[0, 1], torch.full((1, 64, 64), 127 / 127.5 - 1))


@pytest.mark.parametrize("keep", [True, False])
def test_text_orchestration_mask_trim_skip_and_cleanup(tmp_path, monkeypatch, keep):
    ds = dataset(image_source(tmp_path / "images"))
    cache = Path(ds.cache_directory)
    cache.mkdir()
    stale = cache / "stale_qi_te.safetensors"
    stale.write_bytes(b"stale")
    calls = []

    def encoder_boundary(tokenizer, encoder, prompts):
        calls.extend(prompts)
        embedding = torch.arange(len(prompts) * 4 * 3).reshape(len(prompts), 4, 3).float()
        mask = torch.tensor([[1, 1, 0, 0] if p.endswith("a") else [1, 1, 1, 0] for p in prompts])
        return embedding, mask

    monkeypatch.setattr(qwen_image_cache_text_encoder_outputs.qwen_image_utils, "get_qwen_prompt_embeds", encoder_boundary)

    def encode(items):
        qwen_image_cache_text_encoder_outputs.encode_and_save_batch(None, None, items, torch.device("cpu"), None)

    for skip in (False, True):
        existing, current = cache_text_encoder_outputs.prepare_cache_files_and_paths([ds])
        cache_text_encoder_outputs.process_text_encoder_batches(1, skip, 2, [ds], existing, current, encode)
        cache_text_encoder_outputs.post_process_cache_files([ds], existing, current, keep)
    assert sorted(calls) == ["caption a", "caption b"]
    assert stale.exists() == keep
    assert load_file(str(cache / "a_qi_te.safetensors"))["varlen_vl_embed_float32"].shape == (2, 3)
    assert load_file(str(cache / "b_qi_te.safetensors"))["varlen_vl_embed_float32"].shape == (3, 3)


@pytest.mark.parametrize(
    "field,value",
    [
        ("control_directory", "x"),
        ("multiple_target", True),
        ("fp_1f_target_index", 1),
        ("no_resize_control", True),
        ("unknown", 1),
        ("batch_size", True),
    ],
)
def test_reject_excluded_or_invalid_declaration(tmp_path, field, value):
    declaration = {"datasets": [{"image_directory": str(tmp_path), field: value}]}
    with pytest.raises(ValueError, match=field):
        config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
            declaration, argparse.Namespace(), architecture="qi"
        )


def test_reject_video_declaration(tmp_path):
    with pytest.raises(ValueError, match="video_directory"):
        config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
            {"datasets": [{"video_directory": str(tmp_path)}]}, argparse.Namespace(), architecture="qi"
        )


@pytest.mark.parametrize("field", ["image_path_1", "control_path", "video_path", "audio_path", "references", "unknown"])
def test_reject_excluded_jsonl_content(tmp_path, field):
    source = image_source(tmp_path / "images")
    path = tmp_path / "items.jsonl"
    path.write_text(
        json.dumps({"image_path": str(source / "a.png"), "caption": "caption a", field: "bad"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=field):
        ImageJsonlDatasource(str(path))


def test_reject_jsonl_without_cache_location(tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="cache_directory"):
        config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
            {"datasets": [{"image_jsonl_file": str(path)}]}, argparse.Namespace(), architecture="qi"
        )
