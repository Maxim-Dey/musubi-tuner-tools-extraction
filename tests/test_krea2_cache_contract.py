"""CPU regressions for the real image, cache and training-dataset chain."""

import argparse
from multiprocessing import Value
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open

from musubi_tuner import krea2_cache_latents, krea2_cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_KREA2
from musubi_tuner.dataset.image_video_dataset import save_text_encoder_output_cache_krea2
from musubi_tuner import cache_latents, cache_text_encoder_outputs


def _group(tmp_path, *, jsonl=False, training=False, repeats=2):
    image_dir = tmp_path / "images"
    image_dir.mkdir(exist_ok=True)
    for index, caption in enumerate(["one", "two"]):
        Image.fromarray(np.full((32, 48, 3), index * 40 + 100, dtype=np.uint8)).save(image_dir / f"image_{index}.png")
        (image_dir / f"image_{index}.txt").write_text(caption, encoding="utf-8")
    dataset = {"image_directory": str(image_dir)}
    if jsonl:
        import json

        records = [
            {"image_path": str(image_dir / f"image_{i}.png"), "caption": caption} for i, caption in enumerate(["one", "two"])
        ]
        jsonl_file = tmp_path / "images.jsonl"
        jsonl_file.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
        dataset = {"image_jsonl_file": str(jsonl_file)}
    config = {
        "general": {"resolution": [32, 32], "caption_extension": ".txt", "batch_size": 2, "num_repeats": repeats},
        "datasets": [{**dataset, "cache_directory": str(tmp_path / "cache")}],
    }
    blueprint = config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
        config, argparse.Namespace(), architecture=ARCHITECTURE_KREA2
    )
    return config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group, training=training, shared_epoch=Value("i", 0))


@pytest.mark.parametrize("jsonl", [False, True])
def test_real_image_cache_roundtrip(tmp_path, monkeypatch, jsonl):
    group = _group(tmp_path, jsonl=jsonl)
    dataset = group.datasets[0]
    _, items = next(dataset.retrieve_latent_cache_batches(1))
    assert {item.caption for item in items} == {"one", "two"}
    assert all(item.content.shape == (32, 32, 3) for item in items)

    class VaeBoundary:
        device = torch.device("cpu")
        dtype = torch.float32

        def encode_pixels_to_latents(self, pixels):
            assert pixels.shape == (2, 3, 1, 32, 32)
            assert pixels.min() >= -1 and pixels.max() <= 1
            return torch.arange(2 * 16 * 4 * 4, dtype=torch.float32).reshape(2, 16, 1, 4, 4)

    krea2_cache_latents.encode_and_save_batch(VaeBoundary(), items)
    captions = [item.caption for item in items]

    def encoder_boundary(encoder, prompts):
        assert prompts == captions
        hidden = torch.arange(2 * 3 * 12 * 4, dtype=torch.float32).reshape(2, 3, 12, 4)
        return hidden, torch.tensor([[True, False, True], [True, True, True]])

    monkeypatch.setattr(krea2_cache_text_encoder_outputs.krea2_utils, "get_krea2_prompt_embeds", encoder_boundary)
    krea2_cache_text_encoder_outputs.encode_and_save_batch(object(), items)
    with safe_open(items[0].latent_cache_path, framework="pt") as cache:
        assert cache.keys() == ["latents_1x4x4_float32"]
        assert cache.metadata() == {"architecture": "krea2", "width": "48", "height": "32", "format_version": "1.0.1"}
    with safe_open(items[0].text_encoder_output_cache_path, framework="pt") as cache:
        assert cache.keys() == ["varlen_krea2_vl_embed_float32"]
        assert cache.metadata()["caption1"] == items[0].caption
        assert cache.get_tensor(cache.keys()[0]).shape == (2, 12, 4)
    loaded_group = _group(tmp_path, jsonl=jsonl, training=True, repeats=1)
    loaded_dataset = loaded_group.datasets[0]
    assert loaded_dataset.num_train_items == 2
    batch = loaded_dataset.batch_manager[0]
    assert batch["latents"].shape == (2, 16, 1, 4, 4)
    assert sorted(entry.shape[0] for entry in batch["krea2_vl_embed"]) == [2, 3]
    assert sorted(float(latent.flatten()[0]) for latent in batch["latents"]) == [0.0, 256.0]
    assert _group(tmp_path, jsonl=jsonl, training=True).num_train_items == 4


def test_bucket_resolution_retains_unaligned_area_setting():
    from musubi_tuner.dataset.bucket import BucketSelector

    config = {
        "general": {"caption_extension": ".txt", "resolution": 1000, "enable_bucket": True},
        "datasets": [{"image_directory": "images"}],
    }
    blueprint = config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(config, argparse.Namespace())
    params = blueprint.dataset_group.datasets[0].params
    resolution = BucketSelector(params.resolution, params.enable_bucket, architecture=params.architecture).get_bucket_resolution(
        (1000, 1000)
    )
    assert resolution == (992, 1008)
    config["general"]["enable_bucket"] = False
    with pytest.raises(ValueError, match="resolution"):
        config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(config, argparse.Namespace())


def test_text_cache_replaces_old_dtype(tmp_path):
    dataset = _group(tmp_path).datasets[0]
    item = next(dataset.retrieve_text_encoder_output_cache_batches(1))[0]
    save_text_encoder_output_cache_krea2(item, torch.ones(2, 12, 4))
    save_text_encoder_output_cache_krea2(item, torch.ones(3, 12, 4, dtype=torch.bfloat16))
    with safe_open(item.text_encoder_output_cache_path, framework="pt") as cache:
        assert cache.keys() == ["varlen_krea2_vl_embed_bfloat16"]


@pytest.mark.parametrize(
    "field,value",
    [("video_directory", "v"), ("control_directory", "c"), ("multiple_target", True), ("fp_latent_window_size", 9), ("unknown", 1)],
)
def test_dataset_rejects_excluded_or_unknown_fields(field, value):
    with pytest.raises(ValueError, match=field):
        config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
            {"datasets": [{"image_directory": "images", field: value}]}, argparse.Namespace(), architecture=ARCHITECTURE_KREA2
        )


@pytest.mark.parametrize(
    "module,model_flag", [(krea2_cache_latents, "--vae"), (krea2_cache_text_encoder_outputs, "--text_encoder")]
)
def test_cache_parsers_reject_unknown_and_invalid_options(module, model_flag):
    parser = module.setup_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset_config", "data.toml", model_flag, "model", "--turbo_dit", "turbo"])
    args = parser.parse_args(["--dataset_config", "data.toml", model_flag, "model", "--batch_size", "0"])
    with pytest.raises(ValueError, match="batch_size"):
        module.validate_cache_args(args)


def test_starter_dataset_uses_supported_schema():
    root = Path(__file__).resolve().parents[1]
    config = config_utils.load_user_config(str(root / "config_for_krea2_raw_lora/dataset.toml"))
    blueprint = config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
        config, argparse.Namespace(), architecture=ARCHITECTURE_KREA2
    )
    assert blueprint.dataset_group.datasets[0].params.resolution == (1024, 1024)


@pytest.mark.parametrize("keep_cache", [False, True])
def test_cache_lifecycle_preserves_skip_batching_and_pruning(tmp_path, keep_cache):
    dataset = _group(tmp_path).datasets[0]
    stale_latent = Path(dataset.cache_directory) / "stale_0048x0032_kr2.safetensors"
    stale_text = Path(dataset.cache_directory) / "stale_kr2_te.safetensors"
    stale_latent.parent.mkdir()
    stale_latent.touch()
    stale_text.touch()
    encoded = []

    def save_latents(batch):
        encoded.append(len(batch))
        for item in batch:
            krea2_cache_latents.save_latent_cache_krea2(item, torch.zeros(16, 1, 4, 4))

    args = argparse.Namespace(num_workers=1, batch_size=1, skip_existing=True, keep_cache=keep_cache)
    cache_latents.encode_datasets([dataset], save_latents, args)
    assert encoded == [1, 1]
    cache_latents.encode_datasets([dataset], save_latents, args)
    assert encoded == [1, 1]
    assert stale_latent.exists() is keep_cache

    def save_text(batch):
        encoded.append(len(batch))
        for item in batch:
            save_text_encoder_output_cache_krea2(item, torch.ones(2, 12, 4))

    def text_pass():
        files, paths = cache_text_encoder_outputs.prepare_cache_files_and_paths([dataset])
        cache_text_encoder_outputs.process_text_encoder_batches(1, True, 1, [dataset], files, paths, save_text)
        cache_text_encoder_outputs.post_process_cache_files([dataset], files, paths, keep_cache)

    text_pass()
    assert encoded == [1, 1, 1, 1]
    text_pass()
    assert encoded == [1, 1, 1, 1]
    assert stale_text.exists() is keep_cache


@pytest.mark.parametrize(
    "module,model_flag,loader_owner,loader_name",
    [
        (krea2_cache_latents, "--vae", krea2_cache_latents.qwen_image_utils, "load_vae"),
        (
            krea2_cache_text_encoder_outputs,
            "--text_encoder",
            krea2_cache_text_encoder_outputs.krea2_utils,
            "load_krea2_text_encoder",
        ),
    ],
)
@pytest.mark.parametrize("bad_dataset", ['video_directory = "video"', "unknown_field = true"])
def test_invalid_dataset_fails_before_model_loading(
    tmp_path, monkeypatch, module, model_flag, loader_owner, loader_name, bad_dataset
):
    config = tmp_path / "bad.toml"
    config.write_text(f'[[datasets]]\nimage_directory = "images"\n{bad_dataset}\n', encoding="utf-8")
    weights = tmp_path / "weights.safetensors"
    weights.touch()
    monkeypatch.setattr("sys.argv", ["cache", "--device", "cpu", "--dataset_config", str(config), model_flag, str(weights)])

    def unexpected_load(*args, **kwargs):
        pytest.fail("Model loader must not run for an invalid dataset")

    monkeypatch.setattr(loader_owner, loader_name, unexpected_load)
    with pytest.raises(ValueError, match=str(config.name)):
        module.main()


def test_jsonl_rejects_control_and_missing_caption_before_model(tmp_path):
    from musubi_tuner.dataset.datasources import ImageJsonlDatasource
    import json

    jsonl_file = tmp_path / "invalid.jsonl"
    for record, message in [
        ({"image_path": "image.png", "caption": "c", "control_path": "control.png"}, "control_path"),
        ({"image_path": "image.png"}, "caption"),
    ]:
        jsonl_file.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            ImageJsonlDatasource(str(jsonl_file))


@pytest.mark.parametrize(
    "module,model_flag,loader_owner,loader_name",
    [
        (krea2_cache_latents, "--vae", krea2_cache_latents.qwen_image_utils, "load_vae"),
        (
            krea2_cache_text_encoder_outputs,
            "--text_encoder",
            krea2_cache_text_encoder_outputs.krea2_utils,
            "load_krea2_text_encoder",
        ),
    ],
)
def test_missing_model_file_fails_before_loader(tmp_path, monkeypatch, module, model_flag, loader_owner, loader_name):
    import toml

    dataset = _group(tmp_path).datasets[0]
    config = tmp_path / "valid.toml"
    config.write_text(
        toml.dumps(
            {
                "general": {"caption_extension": ".txt", "resolution": 32},
                "datasets": [{"image_directory": dataset.image_directory, "cache_directory": dataset.cache_directory}],
            }
        ),
        encoding="utf-8",
    )
    missing = tmp_path / "missing.safetensors"
    monkeypatch.setattr("sys.argv", ["cache", "--device", "cpu", "--dataset_config", str(config), model_flag, str(missing)])

    def unexpected_load(*args, **kwargs):
        pytest.fail("Missing checkpoint must be rejected before the model loader")

    monkeypatch.setattr(loader_owner, loader_name, unexpected_load)
    with pytest.raises(ValueError, match=model_flag) as error:
        module.main()
    assert str(missing) in str(error.value)
    assert "existing local" in str(error.value)

    missing.touch()
    args = module.setup_parser().parse_args(["--dataset_config", str(config), model_flag, str(missing)])
    module.validate_cache_args(args)


def test_latent_debug_reads_images_without_vae(tmp_path, monkeypatch):
    import toml

    dataset = _group(tmp_path).datasets[0]
    config = tmp_path / "debug.toml"
    config.write_text(
        toml.dumps(
            {
                "general": {"caption_extension": ".txt", "resolution": 32},
                "datasets": [{"image_directory": dataset.image_directory, "cache_directory": dataset.cache_directory}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["cache", "--device", "cpu", "--dataset_config", str(config), "--debug_mode", "console", "--console_num_images", "1"],
    )
    seen = []

    def preview(image, *args):
        seen.append(image.shape)
        return ord(" ")

    def unexpected_load(*args, **kwargs):
        pytest.fail("Dataset debug must not load a VAE")

    monkeypatch.setattr(cache_latents, "show_console", preview)
    monkeypatch.setattr(krea2_cache_latents.qwen_image_utils, "load_vae", unexpected_load)
    krea2_cache_latents.main()
    assert seen == [(32, 32, 3)]
