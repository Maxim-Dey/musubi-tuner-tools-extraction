"""CPU path and preflight contracts for the opted-in Qwen experiment folder."""

import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from PIL import Image
import pytest
from safetensors.torch import save_file
import toml
import torch

from musubi_tuner import qwen_image_cache_latents, qwen_image_cache_text_encoder_outputs, qwen_image_train_network
from musubi_tuner import cache_latents, cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.validation_inputs import prepare_validation_inputs, read_validation_cache_pair


class ReachedModelBoundary(Exception):
    """Stop a successful preflight before a real model is loaded."""


def _write_image_and_cache(root: Path, subset: str, stem: str, caption: str) -> dict:
    images = root / "dataset" / subset
    cache = root / "cache" / subset
    images.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    image = images / f"{stem}.png"
    Image.new("RGB", (64, 64), (63, 91, len(subset) * 9)).save(image)
    (images / f"{stem}.txt").write_text(caption + "\n", encoding="utf-8")
    image_id = hashlib.sha256(image.read_bytes()).hexdigest()
    metadata = {"architecture": "qwen_image", "format_version": "1.0.1", "source_image_sha256": image_id}
    latent = cache / f"{stem}_0064x0064_qi.safetensors"
    text = cache / f"{stem}_qi_te.safetensors"
    save_file(
        {"latents_1x8x8_float32": torch.full((2, 1, 8, 8), 0.25)},
        str(latent),
        metadata={**metadata, "width": "64", "height": "64"},
    )
    save_file(
        {"varlen_vl_embed_float32": torch.ones((3, 4))},
        str(text),
        metadata={**metadata, "caption1": caption},
    )
    return {"image": image, "caption": images / f"{stem}.txt", "latent": latent, "text": text}


def make_experiment(root: Path) -> dict:
    """Create real train/val sources and caches with no absolute path in the TOMLs."""
    root.mkdir(parents=True)
    train = _write_image_and_cache(root, "train", "train", "training caption")
    familiar = _write_image_and_cache(root, "val_familiar", "familiar", "familiar caption")
    unfamiliar = _write_image_and_cache(root, "val_unfamiliar", "unfamiliar", "unfamiliar caption")
    jsonl = root / "dataset" / "val_unfamiliar" / "items.jsonl"
    jsonl.write_text(
        json.dumps({"image_path": "dataset/val_unfamiliar/unfamiliar.png", "caption": "unfamiliar caption"}) + "\n",
        encoding="utf-8",
    )
    general = {
        "resolution": [64, 64],
        "enable_bucket": True,
        "bucket_no_upscale": True,
        "caption_extension": ".txt",
        "batch_size": 1,
        "num_repeats": 1,
    }
    train_config = root / "train-dataset.toml"
    train_config.write_text(
        toml.dumps(
            {"general": general, "datasets": [{"image_directory": "dataset/train", "cache_directory": "cache/train"}]}
        ),
        encoding="utf-8",
    )
    val_config = root / "val-dataset.toml"
    val_config.write_text(
        toml.dumps(
            {
                "general": general,
                "datasets": [
                    {
                        "role": "val_familiar",
                        "image_directory": "dataset/val_familiar",
                        "cache_directory": "cache/val_familiar",
                    },
                    {
                        "role": "val_unfamiliar",
                        "image_jsonl_file": "dataset/val_unfamiliar/items.jsonl",
                        "cache_directory": "cache/val_unfamiliar",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    models = root / "models"
    models.mkdir()
    for name in ("dit.safetensors", "vae.safetensors", "text.safetensors", "adapter.safetensors", "base.safetensors"):
        (models / name).touch()  # Path preflight only; no model loader reads these files.
    (root / "sample_prompts.txt").write_text("fixture portrait --s 2\n", encoding="utf-8")
    (root / "tracker.toml").write_text("[run]\nname = 'fixture'\n", encoding="utf-8")
    (root / "resume-state").mkdir()
    settings = {
        "experiment_dir": ".",
        "model_version": "original",
        "dataset_config": "train-dataset.toml",
        "val_dataset_config": "val-dataset.toml",
        "dit": "models/dit.safetensors",
        "vae": "models/vae.safetensors",
        "text_encoder": "models/text.safetensors",
        "network_module": "networks.lora_qwen_image",
        "sdpa": True,
        "optimizer_type": "AdamW",
        "output_name": "portrait",
        "max_train_steps": 2,
        "max_data_loader_n_workers": 0,
        "val_every_n_steps": 11,
        "log_with": "tensorboard",
    }
    train_toml = root / "train.toml"
    train_toml.write_text(toml.dumps(settings), encoding="utf-8")
    return {
        "root": root,
        "train_toml": train_toml,
        "train_dataset": train_config,
        "val_dataset": val_config,
        "jsonl": jsonl,
        "items": {"train": train, "val_familiar": familiar, "val_unfamiliar": unfamiliar},
    }


@pytest.fixture
def experiment(tmp_path):
    return make_experiment(tmp_path / "portrait")


def _parse_trainer(monkeypatch, train_toml: Path | None, *cli: str):
    parser = qwen_image_train_network.qwen_image_setup_parser(setup_parser_common())
    argv = ["qwen_image_train_network"]
    if train_toml is not None:
        argv += ["--config_file", str(train_toml)]
    monkeypatch.setattr(sys, "argv", argv + list(cli))
    args = read_config_from_file(parser.parse_args(), parser)
    qwen_image_train_network.QwenImageNetworkTrainer().validate_training_inputs(args)
    return args


def _cache_main(monkeypatch, module, argv: list[str]):
    # These parser/path checks deliberately reach the model boundary even when fixtures have complete caches.
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv, *([] if "--rebuild" in argv else ["--rebuild"])])

    def stop_at_model(*_args, **_kwargs):
        raise ReachedModelBoundary

    monkeypatch.setattr(module.qwen_image_utils, "load_vae", stop_at_model)
    monkeypatch.setattr(module.qwen_image_utils, "load_qwen2_5_vl", stop_at_model)
    with pytest.raises(ReachedModelBoundary):
        module.main()


def test_fixture_contains_real_train_two_role_jsonl_and_cache_pairs(experiment):
    assert experiment["train_toml"].is_file()
    assert experiment["train_dataset"].is_file() and experiment["val_dataset"].is_file()
    assert "dataset/val_unfamiliar/unfamiliar.png" in experiment["jsonl"].read_text(encoding="utf-8")
    for item in experiment["items"].values():
        assert item["image"].is_file() and item["caption"].is_file()
        assert item["latent"].is_file() and item["text"].is_file()


def test_fixture_cache_pairs_are_valid_for_existing_stage1_reader(experiment):
    """An absolute-path copy isolates fixture validity from the new path resolver."""
    root = experiment["root"]
    jsonl = root / "dataset" / "val_unfamiliar" / "absolute-items.jsonl"
    jsonl.write_text(
        json.dumps({"image_path": str(experiment["items"]["val_unfamiliar"]["image"]), "caption": "unfamiliar caption"}) + "\n",
        encoding="utf-8",
    )
    declaration = toml.load(experiment["val_dataset"])
    for dataset in declaration["datasets"]:
        dataset["cache_directory"] = str(root / dataset["cache_directory"])
        if "image_directory" in dataset:
            dataset["image_directory"] = str(root / dataset["image_directory"])
        else:
            dataset["image_jsonl_file"] = str(jsonl)
    absolute_config = root / "val-absolute.toml"
    absolute_config.write_text(toml.dumps(declaration), encoding="utf-8")
    manifest = prepare_validation_inputs(SimpleNamespace(val_dataset_config=str(absolute_config)))
    assert {role: len(items) for role, items in manifest.items_by_role.items()} == {
        "val_familiar": 1,
        "val_unfamiliar": 1,
    }
    for item in manifest.items:
        latent, text = read_validation_cache_pair(item)
        assert latent.numel() > 0 and text.numel() > 0


def test_trainer_portable_root_precedence_and_effective_paths(experiment, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    before_cwd = Path.cwd()
    args = _parse_trainer(
        monkeypatch,
        experiment["train_toml"],
        "--val_every_n_steps", "7",
        "--dit", str(experiment["root"] / "models" / "dit.safetensors"),
    )
    root = experiment["root"].resolve()
    assert Path.cwd() == before_cwd
    assert Path(args.experiment_dir) == root
    assert args.val_seed_noise == 42  # parser default
    assert args.val_every_n_steps == 7  # CLI overrides train.toml's 11
    for name, expected in (
        ("dataset_config", "train-dataset.toml"),
        ("val_dataset_config", "val-dataset.toml"),
        ("dit", "models/dit.safetensors"),
        ("vae", "models/vae.safetensors"),
        ("text_encoder", "models/text.safetensors"),
    ):
        assert Path(getattr(args, name)) == root / expected
    assert Path(args.output_dir) == root / "output"
    assert Path(args.logging_dir) == root / "output" / "tensorboard"
    absolute_override = _parse_trainer(monkeypatch, experiment["train_toml"], "--experiment_dir", str(root))
    assert Path(absolute_override.experiment_dir) == root


def test_trainer_rebases_optional_paths(experiment, tmp_path, monkeypatch):
    settings = toml.load(experiment["train_toml"])
    settings.update(
        sample_prompts="sample_prompts.txt",
        log_tracker_config="tracker.toml",
        network_weights="models/adapter.safetensors",
        base_weights=["models/base.safetensors"],
        resume="resume-state",
    )
    experiment["train_toml"].write_text(toml.dumps(settings), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    args = _parse_trainer(monkeypatch, experiment["train_toml"])
    root = experiment["root"].resolve()
    assert Path(args.sample_prompts) == root / "sample_prompts.txt"
    assert Path(args.log_tracker_config) == root / "tracker.toml"
    assert Path(args.network_weights) == root / "models" / "adapter.safetensors"
    assert [Path(path) for path in args.base_weights] == [root / "models" / "base.safetensors"]
    assert Path(args.resume) == root / "resume-state"
    assert Path(args.dit) == root / "models" / "dit.safetensors"


@pytest.mark.parametrize("adapter", ["networks.loha", "networks.lokr"])
def test_experiment_mode_requires_qwen_lora_adapter(experiment, monkeypatch, adapter):
    with pytest.raises(ValueError, match="network_module.*lora_qwen_image"):
        _parse_trainer(monkeypatch, experiment["train_toml"], "--network_module", adapter)


def test_selected_train_toml_must_match_experiment_root(experiment, tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    settings = toml.load(experiment["train_toml"])
    settings["experiment_dir"] = str(experiment["root"])
    selected = outside / "train.toml"
    selected.write_text(toml.dumps(settings), encoding="utf-8")
    monkeypatch.chdir(outside)
    with pytest.raises(ValueError) as error:
        _parse_trainer(monkeypatch, selected)
    message = str(error.value)
    assert "unsupported parameter" not in message
    assert str(experiment["root"]) in message
    assert "train.toml" in message and ("selected" in message or "config_file" in message)


@pytest.mark.parametrize("root_arg", ["relative", "absolute"])
def test_trainer_requires_selected_root_train_toml_and_validation(experiment, tmp_path, monkeypatch, root_arg):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    value = "." if root_arg == "relative" else str(experiment["root"])
    with pytest.raises(ValueError, match=r"(?i)(config_file|train\.toml)"):
        _parse_trainer(monkeypatch, None, "--experiment_dir", value)
    settings = toml.load(experiment["train_toml"])
    settings.pop("val_dataset_config")
    experiment["train_toml"].write_text(toml.dumps(settings), encoding="utf-8")
    with pytest.raises(ValueError, match=r"val_dataset_config"):
        _parse_trainer(monkeypatch, experiment["train_toml"])


@pytest.mark.parametrize("missing", ["train-dataset.toml", "val-dataset.toml"])
def test_missing_dataset_toml_fails_before_model(experiment, tmp_path, monkeypatch, missing):
    (experiment["root"] / missing).unlink()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with pytest.raises(ValueError, match=missing):
        _parse_trainer(monkeypatch, experiment["train_toml"])


@pytest.mark.parametrize("setting,canonical", [("dataset_config", "train-dataset.toml"), ("val_dataset_config", "val-dataset.toml")])
@pytest.mark.parametrize("source", ["toml_relative", "cli_absolute"])
def test_trainer_requires_canonical_dataset_tomls(experiment, tmp_path, monkeypatch, setting, canonical, source):
    alternative = (experiment["root"] if source == "toml_relative" else tmp_path / "outside") / f"other-{canonical}"
    alternative.parent.mkdir(exist_ok=True)
    shutil.copyfile(experiment["root"] / canonical, alternative)
    cli = ()
    if source == "toml_relative":
        settings = toml.load(experiment["train_toml"])
        settings[setting] = alternative.name
        experiment["train_toml"].write_text(toml.dumps(settings), encoding="utf-8")
    else:
        cli = (f"--{setting}", str(alternative))
    with pytest.raises(ValueError) as error:
        _parse_trainer(monkeypatch, experiment["train_toml"], *cli)
    message = str(error.value)
    assert setting in message and canonical in message
    assert (f"CLI --{setting}" if cli else "train.toml") in message


def test_trainer_accepts_absolute_canonical_dataset_tomls(experiment, monkeypatch):
    args = _parse_trainer(
        monkeypatch, experiment["train_toml"],
        "--dataset_config", str(experiment["train_dataset"]),
        "--val_dataset_config", str(experiment["val_dataset"]),
    )
    assert Path(args.dataset_config) == experiment["train_dataset"]
    assert Path(args.val_dataset_config) == experiment["val_dataset"]


@pytest.mark.parametrize(
    "setting,value,correction",
    [
        ("save_precision", "bf16", "fp32"),
        ("save_precision", "fp16", "fp32"),
        ("save_last_n_steps", -1, "save_last_n_steps"),
        ("save_last_n_steps_state", 4, "save_last_n_steps"),
        ("save_last_n_epochs", 4, "save_last_n_epochs"),
        ("save_last_n_epochs_state", 4, "save_last_n_epochs_state"),
        ("save_state_to_huggingface", True, "save_state_to_huggingface"),
        ("output_dir", "other-output", "output"),
        ("logging_dir", "other-logs", "tensorboard"),
        ("output_name", "../escape", "output_name"),
    ],
)
def test_conflicting_effective_settings_fail_in_preflight(experiment, tmp_path, monkeypatch, setting, value, correction):
    settings = toml.load(experiment["train_toml"])
    settings[setting] = value
    experiment["train_toml"].write_text(toml.dumps(settings), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with pytest.raises(ValueError) as error:
        _parse_trainer(monkeypatch, experiment["train_toml"])
    message = str(error.value)
    assert experiment["train_toml"].name in message
    assert setting in message and correction in message


@pytest.mark.parametrize("setting,value", [("output_dir", "other-output"), ("dit", "missing.safetensors")])
def test_cli_path_errors_identify_cli_and_experiment_root(experiment, monkeypatch, setting, value):
    with pytest.raises(ValueError) as error:
        _parse_trainer(monkeypatch, experiment["train_toml"], f"--{setting}", value)
    message = str(error.value)
    assert f"CLI --{setting}" in message
    assert "experiment_dir" in message
    assert "process CWD" not in message


def test_legacy_parser_keeps_cwd_relative_paths_without_experiment_root(tmp_path, monkeypatch):
    config = tmp_path / "legacy.toml"
    config.write_text(toml.dumps({"dataset_config": "legacy-dataset.toml", "dit": "legacy-dit.safetensors"}), encoding="utf-8")
    parser = qwen_image_train_network.qwen_image_setup_parser(setup_parser_common())
    monkeypatch.setattr(sys, "argv", ["qwen_image_train_network", "--config_file", str(config)])
    args = read_config_from_file(parser.parse_args(), parser)
    assert args.dataset_config == "legacy-dataset.toml"
    assert args.dit == "legacy-dit.safetensors"


def test_moving_experiment_keeps_validation_identity_and_later_read(experiment, tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    original_args = _parse_trainer(monkeypatch, experiment["train_toml"])
    original = prepare_validation_inputs(original_args)
    moved_root = tmp_path / "renamed-experiment"
    shutil.copytree(experiment["root"], moved_root)
    moved_args = _parse_trainer(monkeypatch, moved_root / "train.toml")
    moved = prepare_validation_inputs(moved_args, expected_fingerprint=original.fingerprint)
    assert moved.fingerprint == original.fingerprint
    assert {item.role for item in moved.items} == {"val_familiar", "val_unfamiliar"}
    assert all(str(item.image_path).startswith(str(moved_root)) for item in moved.items)
    for item in moved.items:
        latent, text = read_validation_cache_pair(item)
        assert latent.numel() > 0 and text.numel() > 0
    assert not (experiment["root"] / "output").exists()


@pytest.mark.parametrize("module,checkpoint_key", [(qwen_image_cache_latents, "vae"), (qwen_image_cache_text_encoder_outputs, "text_encoder")])
def test_cache_commands_preflight_both_val_roles_from_other_cwd(experiment, tmp_path, monkeypatch, module, checkpoint_key):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    seen_roles = []
    original_validator = module.config_utils.validate_role_aware_sources

    def capture_roles(group, source):
        seen_roles.extend(dataset.role for dataset in group.datasets)
        return original_validator(group, source)

    monkeypatch.setattr(module.config_utils, "validate_role_aware_sources", capture_roles)
    _cache_main(
        monkeypatch,
        module,
        [
            "--train_config", str(experiment["train_toml"]),
            "--dataset_config", "val-dataset.toml",
            f"--{checkpoint_key}", f"models/{'vae' if checkpoint_key == 'vae' else 'text'}.safetensors",
        ],
    )
    assert seen_roles == ["val_familiar", "val_unfamiliar"]


@pytest.mark.parametrize("module", [qwen_image_cache_latents, qwen_image_cache_text_encoder_outputs])
def test_cache_commands_read_nested_toml_experiment_root(experiment, tmp_path, monkeypatch, module):
    settings = toml.load(experiment["train_toml"])
    settings["training"] = {"experiment_dir": settings.pop("experiment_dir")}
    experiment["train_toml"].write_text(toml.dumps(settings), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    checkpoint = "--vae" if module is qwen_image_cache_latents else "--text_encoder"
    filename = "vae.safetensors" if module is qwen_image_cache_latents else "text.safetensors"
    _cache_main(
        monkeypatch,
        module,
        ["--train_config", str(experiment["train_toml"]), "--dataset_config", "val-dataset.toml", checkpoint, f"models/{filename}"],
    )


@pytest.mark.parametrize("failure", [None, "stale_source", "missing_text"])
def test_train_cache_pairs_match_declared_sources(experiment, failure):
    if failure == "stale_source":
        experiment["items"]["train"]["image"].unlink()
        experiment["items"]["train"]["caption"].unlink()
        replacement = experiment["root"] / "dataset" / "train" / "replacement.png"
        Image.new("RGB", (64, 64), (1, 2, 3)).save(replacement)
        replacement.with_suffix(".txt").write_text("replacement caption", encoding="utf-8")
    elif failure == "missing_text":
        experiment["items"]["train"]["text"].unlink()
    user_config = config_utils.load_user_config(experiment["train_dataset"], experiment_root=str(experiment["root"]))
    blueprint = config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
        user_config, SimpleNamespace(dataset_config=str(experiment["train_dataset"])), architecture="qi"
    )
    group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group, experiment_root=str(experiment["root"]))
    if failure is None:
        config_utils.validate_training_cache_bindings(group, str(experiment["train_dataset"]))
    else:
        with pytest.raises(ValueError, match=r"cache.*(missing|stale|extra|source)|source.*cache"):
            config_utils.validate_training_cache_bindings(group, str(experiment["train_dataset"]))


@pytest.mark.parametrize("module", [qwen_image_cache_latents, qwen_image_cache_text_encoder_outputs])
def test_absolute_cache_root_needs_no_train_config(experiment, tmp_path, monkeypatch, module):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    checkpoint = "--vae" if module is qwen_image_cache_latents else "--text_encoder"
    filename = "vae.safetensors" if module is qwen_image_cache_latents else "text.safetensors"
    _cache_main(
        monkeypatch,
        module,
        ["--experiment_dir", str(experiment["root"]), "--dataset_config", "val-dataset.toml", checkpoint, f"models/{filename}"],
    )


@pytest.mark.parametrize("module", [qwen_image_cache_latents, qwen_image_cache_text_encoder_outputs])
@pytest.mark.parametrize("selector", ["train_config", "dataset_relative", "dataset_absolute"])
def test_cache_requires_canonical_experiment_files(experiment, tmp_path, monkeypatch, module, selector):
    root = experiment["root"]
    checkpoint = "--vae" if module is qwen_image_cache_latents else "--text_encoder"
    filename = "vae.safetensors" if module is qwen_image_cache_latents else "text.safetensors"
    train_config = experiment["train_toml"]
    dataset_config = "val-dataset.toml"
    if selector == "train_config":
        train_config = root / "other-train.toml"
        shutil.copyfile(experiment["train_toml"], train_config)
    else:
        alternative = (root if selector == "dataset_relative" else tmp_path / "outside") / "other-val-dataset.toml"
        alternative.parent.mkdir(exist_ok=True)
        shutil.copyfile(experiment["val_dataset"], alternative)
        dataset_config = alternative.name if selector == "dataset_relative" else str(alternative)
    monkeypatch.setattr(sys, "argv", [
        module.__name__, "--train_config", str(train_config),
        "--dataset_config", dataset_config, checkpoint, f"models/{filename}",
    ])
    monkeypatch.setattr(module.qwen_image_utils, "load_vae", lambda *_a, **_k: pytest.fail("model reached before path preflight"))
    monkeypatch.setattr(module.qwen_image_utils, "load_qwen2_5_vl", lambda *_a, **_k: pytest.fail("model reached before path preflight"))
    with pytest.raises(ValueError) as error:
        module.main()
    message = str(error.value)
    assert ("--train_config" if selector == "train_config" else "--dataset_config") in message
    assert ("train.toml" if selector == "train_config" else "val-dataset.toml") in message


@pytest.mark.parametrize("module", [qwen_image_cache_latents, qwen_image_cache_text_encoder_outputs])
def test_cache_accepts_absolute_canonical_dataset_toml(experiment, monkeypatch, module):
    checkpoint = "--vae" if module is qwen_image_cache_latents else "--text_encoder"
    filename = "vae.safetensors" if module is qwen_image_cache_latents else "text.safetensors"
    _cache_main(monkeypatch, module, [
        "--train_config", str(experiment["train_toml"]),
        "--dataset_config", str(experiment["val_dataset"]),
        checkpoint, f"models/{filename}",
    ])


@pytest.mark.parametrize("module", [qwen_image_cache_latents, qwen_image_cache_text_encoder_outputs])
def test_cache_accepts_canonical_train_dataset_toml(experiment, monkeypatch, module):
    checkpoint = "--vae" if module is qwen_image_cache_latents else "--text_encoder"
    filename = "vae.safetensors" if module is qwen_image_cache_latents else "text.safetensors"
    _cache_main(monkeypatch, module, [
        "--train_config", str(experiment["train_toml"]),
        "--dataset_config", "train-dataset.toml",
        checkpoint, f"models/{filename}",
    ])


@pytest.mark.parametrize("module", [qwen_image_cache_latents, qwen_image_cache_text_encoder_outputs])
def test_cache_legacy_allows_alternative_dataset_toml(experiment, monkeypatch, module):
    alternative = experiment["root"] / "other-train-dataset.toml"
    shutil.copyfile(experiment["train_dataset"], alternative)
    monkeypatch.chdir(experiment["root"])
    checkpoint = "--vae" if module is qwen_image_cache_latents else "--text_encoder"
    filename = "vae.safetensors" if module is qwen_image_cache_latents else "text.safetensors"
    _cache_main(monkeypatch, module, [
        "--dataset_config", alternative.name,
        checkpoint, f"models/{filename}",
    ])


@pytest.mark.parametrize("module", [qwen_image_cache_latents, qwen_image_cache_text_encoder_outputs])
def test_cache_missing_caption_fails_before_model(experiment, tmp_path, monkeypatch, module):
    experiment["items"]["val_familiar"]["caption"].unlink()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    checkpoint = "--vae" if module is qwen_image_cache_latents else "--text_encoder"
    filename = "vae.safetensors" if module is qwen_image_cache_latents else "text.safetensors"
    monkeypatch.setattr(sys, "argv", [module.__name__, "--train_config", str(experiment["train_toml"]), "--dataset_config", "val-dataset.toml", checkpoint, f"models/{filename}"])
    monkeypatch.setattr(module.qwen_image_utils, "load_vae", lambda *_a, **_k: pytest.fail("model loaded before caption preflight"))
    monkeypatch.setattr(module.qwen_image_utils, "load_qwen2_5_vl", lambda *_a, **_k: pytest.fail("model loaded before caption preflight"))
    with pytest.raises(ValueError, match=r"val_familiar.*caption"):
        module.main()


@pytest.mark.parametrize("module", [qwen_image_cache_latents, qwen_image_cache_text_encoder_outputs])
def test_cache_collision_fails_before_model(experiment, tmp_path, monkeypatch, module):
    declaration = toml.load(experiment["val_dataset"])
    declaration["datasets"][1]["cache_directory"] = "cache/val_familiar"
    experiment["val_dataset"].write_text(toml.dumps(declaration), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    checkpoint = "--vae" if module is qwen_image_cache_latents else "--text_encoder"
    filename = "vae.safetensors" if module is qwen_image_cache_latents else "text.safetensors"
    monkeypatch.setattr(sys, "argv", [module.__name__, "--train_config", str(experiment["train_toml"]), "--dataset_config", "val-dataset.toml", checkpoint, f"models/{filename}"])
    monkeypatch.setattr(module.qwen_image_utils, "load_vae", lambda *_a, **_k: pytest.fail("model loaded before cache collision preflight"))
    monkeypatch.setattr(module.qwen_image_utils, "load_qwen2_5_vl", lambda *_a, **_k: pytest.fail("model loaded before cache collision preflight"))
    with pytest.raises(ValueError, match=r"cache"):
        module.main()


def test_cache_legacy_parser_does_not_infer_experiment_root(tmp_path, monkeypatch):
    for module, setup in (
        (qwen_image_cache_latents, cache_latents.setup_parser_common),
        (qwen_image_cache_text_encoder_outputs, cache_text_encoder_outputs.setup_parser_common),
    ):
        parser = module.qwen_image_setup_parser(setup())
        args = parser.parse_args(["--dataset_config", "legacy-dataset.toml", "--vae" if module is qwen_image_cache_latents else "--text_encoder", "legacy-model.safetensors"])
        assert args.dataset_config == "legacy-dataset.toml"
        assert getattr(args, "experiment_dir", None) is None
