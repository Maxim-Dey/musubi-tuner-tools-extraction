"""Model-free Krea2 validation input/cache regressions."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import torch
from safetensors.torch import save_file
import toml

from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common, validate_validation_options
from musubi_tuner.training.krea2_validation_inputs import preflight_validation_inputs, prepare_validation_caches


def _experiment(tmp_path):
    for split, value in (("train", 90), ("val", 150)):
        directory = tmp_path / split
        directory.mkdir()
        Image.fromarray(np.full((32, 48, 3), value, dtype=np.uint8)).save(directory / "image.png")
        (directory / "image.txt").write_text(split, encoding="utf-8")
        (tmp_path / (split + ".toml")).write_text(
            toml.dumps(
                {
                    "general": {"resolution": 32, "caption_extension": ".txt"},
                    "datasets": [{"image_directory": split, "cache_directory": "cache/" + split}],
                }
            ),
            encoding="utf-8",
        )
    for name in ("vae", "text_encoder", "dit"):
        (tmp_path / (name + ".safetensors")).write_bytes(name.encode())
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
    return argparse.Namespace(
        dataset_config=str(tmp_path / "train.toml"),
        val_dataset_config=str(tmp_path / "val.toml"),
        experiment_mode=True,
        auto_cache=True,
        val_every_n_steps=2,
        val_seed_noise=42,
        val_level_noise_n=2,
        val_seed_noise_n=1,
        vae=str(tmp_path / "vae.safetensors"),
        text_encoder=str(tmp_path / "text_encoder.safetensors"),
        dit=str(tmp_path / "dit.safetensors"),
        tokenizer_path=str(tokenizer),
        output_dir=str(tmp_path / "output"),
        resume=None,
    )


def _write_cache(inputs, record, kind, *, metadata=None, shape=None):
    path = Path(record.latent_cache_path if kind == "latent" else record.text_cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "architecture": "krea2",
        "format_version": "1.0.1",
        "width": str(record.original_size[0]),
        "height": str(record.original_size[1]),
        **inputs.provenance(record, kind),
    }
    if kind == "text":
        info["caption1"] = record.caption
    info.update(metadata or {})
    tensor = torch.ones(
        shape or ([16, 1, record.bucket_size[1] // 8, record.bucket_size[0] // 8] if kind == "latent" else [2, 12, 2560])
    )
    key = "latents_1x4x4_float32" if kind == "latent" else "varlen_krea2_vl_embed_float32"
    save_file({key: tensor}, str(path), metadata=info)


def _populate(inputs):
    for record in inputs.train_records + inputs.records:
        for kind in ("latent", "text"):
            _write_cache(inputs, record, kind)


class AcceleratorBoundary:
    is_main_process = True
    num_processes = 1
    barriers = 0

    def wait_for_everyone(self):
        self.barriers += 1


def test_legacy_opt_out():
    assert preflight_validation_inputs(argparse.Namespace()) is None


@pytest.mark.parametrize(
    "key,value", [("val_seed_noise", True), ("val_every_n_steps", 0), ("val_level_noise_n", 3), ("val_seed_noise_n", 0)]
)
def test_explicit_options_rejected(tmp_path, key, value):
    args = _experiment(tmp_path)
    setattr(args, key, value)
    with pytest.raises(ValueError, match=key):
        validate_validation_options(args)


def test_partial_group_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        validate_validation_options(argparse.Namespace(val_every_n_steps=5))


def test_real_parser_merge_paths_other_cwd(tmp_path, monkeypatch):
    config_dir = tmp_path / "arbitrary-name"
    config_dir.mkdir()
    config = config_dir / "train.toml"
    config.write_text(
        'experiment_mode=true\nauto_cache=true\ndataset_config="train-data.toml"\noutput_dir="output"\nnetwork_module="networks.lora_krea2"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    parser = setup_parser_common()
    argv = ["--config_file", str(config), "--output_dir", "cli-output"]
    args = read_config_from_file(parser.parse_args(argv), parser, argv)
    assert args.dataset_config == str(config_dir / "train-data.toml")
    assert args.output_dir == str(config_dir / "cli-output")
    assert args.network_module == "networks.lora_krea2"
    assert args._config_sources["output_dir"] == "CLI --output_dir"
    config.write_text('output_dir="relative"\n', encoding="utf-8")
    args = read_config_from_file(parser.parse_args(argv[:2]), parser, argv[:2])
    assert args.output_dir == "relative"
    config.write_text("val_every_n_steps=true\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        read_config_from_file(parser.parse_args(argv[:2]), parser, argv[:2])
    config.write_text("unknown_new_option=true\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        read_config_from_file(parser.parse_args(argv[:2]), parser, argv[:2])


@pytest.mark.parametrize("problem", ["missing", "empty", "empty_dataset", "overlap", "batch", "repeat", "collision", "conflict"])
def test_preflight_rejects_inputs_before_components(tmp_path, problem, monkeypatch):
    args = _experiment(tmp_path)
    caption = tmp_path / "val/image.txt"
    if problem == "missing":
        caption.unlink()
    elif problem == "empty":
        caption.write_text("  ")
    elif problem == "empty_dataset":
        (tmp_path / "val/image.png").unlink()
    elif problem == "overlap":
        shutil.copyfile(tmp_path / "train/image.png", tmp_path / "val/image.png")
    elif problem in ("batch", "repeat"):
        path = tmp_path / "val.toml"
        content = toml.loads(path.read_text())
        content["general"]["batch_size" if problem == "batch" else "num_repeats"] = 2
        path.write_text(toml.dumps(content))
    elif problem == "collision":
        Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8)).save(tmp_path / "val/image.jpg")
    else:
        shutil.copyfile(tmp_path / "val/image.png", tmp_path / "val/duplicate.png")
        (tmp_path / "val/duplicate.txt").write_text("different caption")
    monkeypatch.setattr(
        "musubi_tuner.training.krea2_validation_inputs._component_identities",
        lambda args: pytest.fail("models must not be inventoried for invalid data"),
    )
    with pytest.raises(ValueError):
        preflight_validation_inputs(args)


def test_compatible_copies_deduplicate_content(tmp_path):
    args = _experiment(tmp_path)
    shutil.copyfile(tmp_path / "val/image.png", tmp_path / "val/copy.png")
    shutil.copyfile(tmp_path / "val/image.txt", tmp_path / "val/copy.txt")
    inputs = preflight_validation_inputs(args)
    assert len(inputs.records) == 1


def test_cache_prepare_missing_only_and_reuse(tmp_path, monkeypatch):
    args = _experiment(tmp_path)
    inputs = preflight_validation_inputs(args)
    calls = []
    from musubi_tuner import krea2_cache_latents, krea2_cache_text_encoder_outputs

    for module, kind in ((krea2_cache_latents, "latent"), (krea2_cache_text_encoder_outputs, "text")):

        def generate(cache_args, kind=kind):
            calls.append((cache_args.dataset_config, kind))
            assert cache_args.skip_existing and cache_args.keep_cache
            for record in inputs.train_records + inputs.records:
                path = record.latent_cache_path if kind == "latent" else record.text_cache_path
                if record.dataset_config == cache_args.dataset_config and not Path(path).exists():
                    _write_cache(inputs, record, kind)

        monkeypatch.setattr(module, "main", generate)
    _write_cache(inputs, inputs.train_records[0], "latent")
    accelerator = AcceleratorBoundary()
    prepare_validation_caches(args, inputs, accelerator)
    assert len(calls) == 3
    assert accelerator.barriers == 1
    original = inputs.manifest
    inputs.validate_unchanged()
    assert inputs.load_batch(inputs.records[0])["latents"].shape == (1, 16, 1, 4, 4)
    again = preflight_validation_inputs(args)
    prepare_validation_caches(args, again, accelerator)
    assert len(calls) == 3
    assert again.manifest == original


@pytest.mark.parametrize("change", ["caption", "image", "cache", "bucket", "model", "missing_manifest", "missing_cache"])
def test_frozen_manifest_resume_and_active_drift(tmp_path, change):
    args = _experiment(tmp_path)
    inputs = preflight_validation_inputs(args)
    _populate(inputs)
    prepare_validation_caches(args, inputs, AcceleratorBoundary())
    state = tmp_path / "checkpoint"
    state.mkdir()
    (state / "val_manifest.json").write_text(json.dumps(inputs.manifest))
    args.resume = str(state)
    if change == "caption":
        (tmp_path / "val/image.txt").write_text("changed")
    elif change == "image":
        Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8)).save(tmp_path / "val/image.png")
    elif change == "cache":
        _write_cache(inputs, inputs.records[0], "text", shape=[3, 12, 2560])
    elif change == "bucket":
        path = tmp_path / "val.toml"
        path.write_text(path.read_text().replace("resolution = 32", "resolution = 64"))
    elif change == "model":
        Path(args.vae).write_bytes(b"new-vae")
    elif change == "missing_manifest":
        (state / "val_manifest.json").unlink()
    else:
        Path(inputs.records[0].latent_cache_path).unlink()
    with pytest.raises(ValueError):
        preflight_validation_inputs(args)
    if change in ("caption", "image", "cache", "bucket", "missing_cache"):
        with pytest.raises(ValueError):
            inputs.validate_unchanged()


@pytest.mark.parametrize(
    "kind,metadata,shape",
    [("text", {"architecture": "qwen_image"}, None), ("latent", {"krea2_provenance": ""}, None), ("latent", {}, [4, 1, 4, 4])],
)
def test_unknown_and_incompatible_cache_rejected(tmp_path, kind, metadata, shape):
    args = _experiment(tmp_path)
    inputs = preflight_validation_inputs(args)
    _write_cache(inputs, inputs.records[0], kind, metadata=metadata, shape=shape)
    with pytest.raises(ValueError, match="recache"):
        preflight_validation_inputs(args)


def test_missing_without_auto_cache_and_prepare_exception(tmp_path, monkeypatch):
    args = _experiment(tmp_path)
    inputs = preflight_validation_inputs(args)
    args.auto_cache = False
    with pytest.raises(ValueError, match="missing"):
        preflight_validation_inputs(args)
    args.auto_cache = True
    from musubi_tuner import krea2_cache_latents

    def fail(args):
        raise RuntimeError("injected encoder failure")

    monkeypatch.setattr(krea2_cache_latents, "main", fail)
    with pytest.raises(ValueError, match="injected encoder failure"):
        prepare_validation_caches(args, inputs, AcceleratorBoundary())
    assert not inputs.manifest_path.exists()


def test_relocation_and_rename_preserve_semantics(tmp_path):
    root = tmp_path / "first"
    root.mkdir()
    args = _experiment(root)
    inputs = preflight_validation_inputs(args)
    _populate(inputs)
    prepare_validation_caches(args, inputs, AcceleratorBoundary())
    relocated = tmp_path / "moved"
    shutil.copytree(root, relocated)
    for key in ("dataset_config", "val_dataset_config", "output_dir", "vae", "text_encoder", "dit", "tokenizer_path"):
        setattr(args, key, str(getattr(args, key)).replace(str(root), str(relocated)))
    copied = preflight_validation_inputs(args)
    assert copied.manifest == inputs.manifest
    # Rename source + matching cache files without changing their contents.
    for path in (relocated / "val").iterdir():
        path.rename(path.with_name(path.name.replace("image", "renamed")))
    for path in (relocated / "cache/val").iterdir():
        path.rename(path.with_name(path.name.replace("image", "renamed")))
    assert preflight_validation_inputs(args).manifest == inputs.manifest


def test_no_val_train_cache_mode(tmp_path):
    args = _experiment(tmp_path)
    for key in ("val_dataset_config", "val_every_n_steps", "val_seed_noise", "val_level_noise_n", "val_seed_noise_n"):
        setattr(args, key, None)
    inputs = preflight_validation_inputs(args)
    assert inputs is not None and inputs.records == []


def test_stale_train_cache_rejected(tmp_path):
    args = _experiment(tmp_path)
    directory = tmp_path / "cache/train"
    directory.mkdir(parents=True)
    (directory / "stale_0048x0032_kr2.safetensors").touch()
    with pytest.raises(ValueError, match="stale train cache"):
        preflight_validation_inputs(args)


def test_cache_preparation_preserves_rng_even_on_failure(tmp_path, monkeypatch):
    import random

    args = _experiment(tmp_path)
    inputs = preflight_validation_inputs(args)
    from musubi_tuner import krea2_cache_latents

    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()

    def fail(cache_args):
        random.random()
        np.random.random()
        torch.rand(4)
        raise RuntimeError("producer failed")

    monkeypatch.setattr(krea2_cache_latents, "main", fail)
    with pytest.raises(ValueError, match="producer failed"):
        prepare_validation_caches(args, inputs, AcceleratorBoundary())
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_rank_serialization_and_error_broadcast(tmp_path, monkeypatch):
    args = _experiment(tmp_path)
    inputs = preflight_validation_inputs(args)
    _populate(inputs)
    status = []

    def broadcast(objects, from_process):
        assert from_process == 0
        if not status:
            status.extend(objects)
        else:
            objects[:] = status

    monkeypatch.setattr("accelerate.utils.broadcast_object_list", broadcast)
    primary = AcceleratorBoundary()
    primary.num_processes = 2
    prepare_validation_caches(args, inputs, primary)
    follower = AcceleratorBoundary()
    follower.num_processes = 2
    follower.is_main_process = False
    # Writer is never invoked on a secondary rank.
    monkeypatch.setattr(
        "musubi_tuner.training.krea2_validation_inputs._prepare_validation_caches_main",
        lambda *args: pytest.fail("secondary writer"),
    )
    prepare_validation_caches(args, inputs, follower)
    assert follower.barriers == 1
    status[:] = ["primary rank cache error"]
    with pytest.raises(ValueError, match="primary rank cache error"):
        prepare_validation_caches(args, inputs, follower)


@pytest.mark.parametrize("module_name,kind", [("krea2_cache_latents", "latent"), ("krea2_cache_text_encoder_outputs", "text")])
@pytest.mark.parametrize("failure", [False, True])
def test_existing_cache_entrypoints_release_models(tmp_path, monkeypatch, module_name, kind, failure):
    import weakref
    import gc
    import importlib

    args = _experiment(tmp_path)
    module = importlib.import_module("musubi_tuner." + module_name)
    references = []

    class Model:
        def to(self, device):
            return self

    def load(*args, **kwargs):
        model = Model()
        references.append(weakref.ref(model))
        return model

    if kind == "latent":
        monkeypatch.setattr(module.qwen_image_utils, "load_vae", load)
        producer = module.cache_latents

        def process(*args):
            if failure:
                raise RuntimeError("injected")

        monkeypatch.setattr(producer, "encode_datasets", process)
    else:
        monkeypatch.setattr(module.krea2_utils, "load_krea2_text_encoder", load)
        producer = module.cache_text_encoder_outputs

        def process(*args):
            if failure:
                raise RuntimeError("injected")

        monkeypatch.setattr(producer, "process_text_encoder_batches", process)
        monkeypatch.setattr(producer, "post_process_cache_files", lambda *args: None)
    argv = ["--dataset_config", args.dataset_config, "--device", "cpu"]
    argv += ["--vae", args.vae] if kind == "latent" else ["--text_encoder", args.text_encoder]
    cache_args = module.setup_parser().parse_args(argv)
    cache_args.experiment_mode = True
    if failure:
        with pytest.raises(RuntimeError, match="injected"):
            module.main(cache_args)
    else:
        module.main(cache_args)
    gc.collect()
    assert references and references[0]() is None


@pytest.mark.parametrize("incompatible_kind", ["latent", "text"])
@pytest.mark.parametrize("missing_kinds", [("latent",), ("text",), ("latent", "text")])
def test_duplicate_auto_cache_uses_original_producer_inventory(tmp_path, monkeypatch, incompatible_kind, missing_kinds):
    """Actual cache entrypoints walk both copies, while metric records stay unique."""
    args = _experiment(tmp_path)
    shutil.copyfile(tmp_path / "val/image.png", tmp_path / "val/copy.png")
    shutil.copyfile(tmp_path / "val/image.txt", tmp_path / "val/copy.txt")
    inputs = preflight_validation_inputs(args)
    assert len(inputs.records) == 1
    assert len(inputs.source_records) == 3  # One train image and both validation paths.
    from musubi_tuner import krea2_cache_latents, krea2_cache_text_encoder_outputs

    loads = []
    encoded_captions = []

    class VaeBoundary:
        device = torch.device("cpu")
        dtype = torch.float32

        def to(self, device):
            return self

        def encode_pixels_to_latents(self, pixels):
            return torch.zeros(pixels.shape[0], 16, 1, pixels.shape[-2] // 8, pixels.shape[-1] // 8)

    def load_vae(*args, **kwargs):
        loads.append("vae")
        return VaeBoundary()

    def load_encoder(*args, **kwargs):
        loads.append("encoder")
        return object()

    def embed(encoder, captions):
        encoded_captions.extend(captions)
        return torch.ones(len(captions), 2, 12, 2560), torch.ones(len(captions), 2, dtype=torch.bool)

    monkeypatch.setattr(krea2_cache_latents.qwen_image_utils, "load_vae", load_vae)
    monkeypatch.setattr(krea2_cache_text_encoder_outputs.krea2_utils, "load_krea2_text_encoder", load_encoder)
    monkeypatch.setattr(krea2_cache_text_encoder_outputs.krea2_utils, "get_krea2_prompt_embeds", embed)
    prepare_validation_caches(args, inputs, AcceleratorBoundary())
    assert len(inputs.records) == 1
    assert encoded_captions.count("val") == 2
    assert len(loads) == 4  # Existing producer entrypoint per split and component.
    cache_paths = [Path(path) for record in inputs.source_records for path in (record.latent_cache_path, record.text_cache_path)]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in cache_paths}
    again = preflight_validation_inputs(args)
    prepare_validation_caches(args, again, AcceleratorBoundary())
    assert len(again.records) == 1
    assert len(loads) == 4
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in cache_paths} == before

    # A copied experiment may retain its frozen manifest but omit working caches.
    # Exercise actual producer entrypoints, not a replacement for cache preparation.
    relocated = tmp_path / "relocated"
    shutil.copytree(tmp_path, relocated)
    for key in ("dataset_config", "val_dataset_config", "output_dir", "vae", "text_encoder", "dit", "tokenizer_path"):
        setattr(args, key, str(relocated / Path(getattr(args, key)).relative_to(tmp_path)))
    copied = preflight_validation_inputs(args)
    copied_before = {
        Path(path): (Path(path).read_bytes(), Path(path).stat().st_mtime_ns)
        for record in copied.source_records
        for path in (record.latent_cache_path, record.text_cache_path)
    }
    manifest_before = (copied.manifest_path.read_bytes(), copied.manifest_path.stat().st_mtime_ns)
    missing = {
        Path(copied.records[0].latent_cache_path if kind == "latent" else copied.records[0].text_cache_path)
        for kind in missing_kinds
    }
    for path in missing:
        path.unlink()
    args.auto_cache = False
    with pytest.raises(ValueError, match="missing .* cache; enable auto_cache"):
        preflight_validation_inputs(args)
    args.auto_cache = True
    recovered = preflight_validation_inputs(args)  # Must not hash missing files here.
    assert recovered.saved_manifest == inputs.manifest
    error = None
    try:
        prepare_validation_caches(args, recovered, AcceleratorBoundary())
    except ValueError as exception:
        error = exception
    assert loads[4:] == ["vae" if kind == "latent" else "encoder" for kind in missing_kinds]
    assert all(path.is_file() for path in missing)
    # Safetensors serialization may change bytes even for identical model outputs.
    # Regeneration never authorizes changing the frozen byte identity.
    if recovered._manifest() != recovered.saved_manifest:
        assert error is not None and "val manifest changed during preparation" in str(error)
    else:
        assert error is None
    assert (copied.manifest_path.read_bytes(), copied.manifest_path.stat().st_mtime_ns) == manifest_before
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in copied_before if path not in missing
    } == {path: value for path, value in copied_before.items() if path not in missing}
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in cache_paths} == before

    # Continue the original duplicate-provenance check in its original directory.
    for key in ("dataset_config", "val_dataset_config", "output_dir", "vae", "text_encoder", "dit", "tokenizer_path"):
        setattr(args, key, str(tmp_path / Path(getattr(args, key)).relative_to(relocated)))

    selected = inputs.records[0].image_path
    nonselected = next(
        record
        for record in inputs.source_records
        if record.dataset_config == args.val_dataset_config and record.image_path != selected
    )
    _write_cache(inputs, nonselected, incompatible_kind, metadata={"architecture": "qwen_image"})
    with pytest.raises(ValueError, match="incompatible"):
        preflight_validation_inputs(args)
