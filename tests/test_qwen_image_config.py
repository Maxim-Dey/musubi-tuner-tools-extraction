"""Real readers and early command-boundary validation, with no model execution."""

from pathlib import Path
import hashlib
import json
import sys

import pytest
import toml
import torch
from PIL import Image
from safetensors.torch import save_file

from musubi_tuner import qwen_image_train_network as training
from musubi_tuner.training.parser_common import setup_parser_common, read_config_from_file
from musubi_tuner.training.sampling_prompts import load_prompts

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "config_for_qwen_image_lora"


def resolve(monkeypatch, tmp_path, config=None, cli=()):
    parser = training.qwen_image_setup_parser(setup_parser_common())
    argv = ["fixture"]
    if config is not None:
        path = tmp_path / "settings.toml"
        path.write_text(toml.dumps(config), encoding="utf-8")
        argv += ["--config_file", str(path)]
    monkeypatch.setattr(sys, "argv", argv + list(cli))
    return read_config_from_file(parser.parse_args(), parser)


def test_all_training_template_values_and_cli_precedence(tmp_path, monkeypatch):
    config = toml.load(TEMPLATES / "train.toml")
    assert len(config) == 40
    args = resolve(monkeypatch, tmp_path, config)
    for key, value in config.items():
        assert getattr(args, key) == value
    args = resolve(monkeypatch, tmp_path, config, ["--network_dim", "8", "--learning_rate", "0.002", "--fp8_base"])
    assert (args.network_dim, args.learning_rate, args.fp8_base) == (8, 0.002, True)
    assert args.max_train_steps == 1600
    assert type(args.lr_warmup_steps) is int and args.lr_warmup_steps == 200


def test_validation_defaults_toml_and_cli_precedence(tmp_path, monkeypatch):
    defaults = resolve(monkeypatch, tmp_path)
    assert (
        defaults.val_dataset_config,
        defaults.val_every_n_steps,
        defaults.val_seed_noise,
        defaults.val_level_noise_n,
        defaults.val_seed_noise_n,
    ) == (None, 200, 42, 10, 1)
    training.validate_validation_args(defaults)

    config = {
        "val_dataset_config": "data/val-dataset.toml",
        "val_every_n_steps": 17,
        "val_seed_noise": -9,
        "val_level_noise_n": 4,
        "val_seed_noise_n": 3,
    }
    from_toml = resolve(monkeypatch, tmp_path, config)
    assert all(getattr(from_toml, key) == value for key, value in config.items())
    training.validate_validation_args(from_toml)
    overridden = resolve(
        monkeypatch,
        tmp_path,
        config,
        ["--val_every_n_steps", "5", "--val_seed_noise", "0", "--val_level_noise_n", "2", "--val_seed_noise_n", "1"],
    )
    assert (overridden.val_every_n_steps, overridden.val_seed_noise, overridden.val_level_noise_n, overridden.val_seed_noise_n) == (
        5, 0, 2, 1
    )
    assert overridden.val_dataset_config == config["val_dataset_config"]
    training.validate_validation_args(overridden)


@pytest.mark.parametrize(
    "key,value",
    [
        ("val_every_n_steps", 0),
        ("val_every_n_steps", True),
        ("val_every_n_steps", 1.5),
        ("val_seed_noise", False),
        ("val_seed_noise", "42"),
        ("val_level_noise_n", 0),
        ("val_level_noise_n", 1),
        ("val_level_noise_n", 3),
        ("val_level_noise_n", True),
        ("val_seed_noise_n", 0),
        ("val_seed_noise_n", False),
        ("val_dataset_config", ""),
        ("val_unknown", 2),
    ],
)
def test_validation_effective_errors_precede_model(tmp_path, monkeypatch, invocation, key, value):
    effects = forbid_effects(monkeypatch)
    config = {**invocation, key: value}
    path = tmp_path / "validation-settings.toml"
    path.write_text(toml.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["fixture", "--config_file", str(path)])
    with pytest.raises(ValueError) as error:
        training.main()
    assert key in str(error.value) and str(path) in str(error.value)
    assert effects == []


def test_sections_suffix_and_supported_override(tmp_path, monkeypatch):
    args = resolve(
        monkeypatch, tmp_path, {"max_train_steps": 100, "later": {"max_train_steps": "bad"}}, ["--max_train_steps", "27"]
    )
    assert args.max_train_steps == 27
    path = tmp_path / "settings"
    parser = training.qwen_image_setup_parser(setup_parser_common())
    monkeypatch.setattr(sys, "argv", ["fixture", "--config_file", str(path), "--max_train_steps", "28"])
    assert read_config_from_file(parser.parse_args(), parser).max_train_steps == 28


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_train_step", 4),
        ("mixed_precision", "bad"),
        ("max_train_steps", "1600"),
        ("max_train_steps", True),
        ("fp8_base", 1),
        ("network_alpha", "16"),
        ("network_args", "rank_dropout=0.1"),
        ("optimizer_args", [1]),
        ("weighting_scheme", "bad"),
        ("timestep_sampling", "bad"),
        ("learning_rate", float("nan")),
        ("model_version", "edit"),
        ("edit", True),
        ("network_module", "networks.lora_wan"),
    ],
)
def test_invalid_training_toml_has_source(tmp_path, monkeypatch, key, value):
    with pytest.raises((ValueError, SystemExit)) as error:
        resolve(monkeypatch, tmp_path, {"settings": {key: value}})
    message = str(error.value)
    assert "settings.toml" in message and key in message


@pytest.mark.parametrize(
    "raw,cli",
    [
        ({"model_version": "edit"}, ["--model_version", "original"]),
        ({"network_module": "networks.lora_wan"}, ["--network_module", "networks.lora_qwen_image"]),
        ({"edit": True}, ["--model_version", "original"]),
        ({"max_train_step": 4}, ["--max_train_steps", "4"]),
    ],
)
def test_raw_excluded_fields_cannot_be_masked(tmp_path, monkeypatch, raw, cli):
    with pytest.raises(ValueError):
        resolve(monkeypatch, tmp_path, raw, cli)


def test_prompt_templates_and_all_readers(tmp_path):
    supplied = load_prompts(str(TEMPLATES / "sample_prompts.txt"))
    assert len(supplied) == 2
    assert [p["enum"] for p in supplied] == [0, 1]
    assert all(p["sample_steps"] == 30 and p["cfg_scale"] == 4 for p in supplied)
    path = tmp_path / "samples.toml"
    path.write_text(
        '[prompt]\nwidth=512\nheight=768\n[[prompt.subset]]\nprompt="one"\n[[prompt.subset]]\nprompt="two"\nwidth=640\n',
        encoding="utf-8",
    )
    assert [p["width"] for p in load_prompts(str(path))] == [512, 640]
    path = tmp_path / "samples.json"
    path.write_text(
        json.dumps([{"prompt": "one", "sample_steps": 3}, "two --w 528 --h 536 --d 9 --s 5 --l 4.5 --fs 2.2 --n bad"]),
        encoding="utf-8",
    )
    assert load_prompts(str(path))[1] == dict(
        prompt="two",
        width=528,
        height=536,
        seed=9,
        sample_steps=5,
        cfg_scale=4.5,
        discrete_flow_shift=2.2,
        negative_prompt="bad",
        enum=1,
    )


@pytest.mark.parametrize(
    "line",
    [
        "scene --w 512junk",
        "scene --s 0",
        "scene --g 2",
        "scene --f 1",
        "scene --ci x.png",
        "scene --unknown 2",
        "scene --l nan",
        "scene --fs inf",
        "scene --w 7",
        "scene --h -1",
        "scene --s 2 extra",
        "scene --o result",
    ],
)
def test_invalid_prompt_text_has_line_context(tmp_path, line):
    path = tmp_path / "bad.txt"
    path.write_text("# comment\n" + line, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_prompts(str(path))
    assert str(path) in str(error.value) and "2" in str(error.value)


@pytest.mark.parametrize(
    "record",
    [
        {"prompt": "scene", "control_image_path": []},
        {"width": True},
        {"sample_steps": 0},
        {"seed": -1},
        {"cfg_scale": float("inf")},
        {"prompt": []},
        {"frame_count": 1},
        {"enum": 9},
    ],
)
def test_invalid_prompt_json_records(tmp_path, record):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([record]), encoding="utf-8")
    with pytest.raises(ValueError, match="bad.json"):
        load_prompts(str(path))


@pytest.fixture
def invocation(tmp_path):
    from PIL import Image

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (32, 32)).save(images / "one.png")
    (images / "one.txt").write_text("scene", encoding="utf-8")
    dataset = tmp_path / "dataset.toml"
    dataset.write_text(
        toml.dumps(
            {
                "general": {"resolution": 32, "caption_extension": ".txt"},
                "datasets": [{"image_directory": str(images), "cache_directory": str(tmp_path / "cache")}],
            }
        ),
        encoding="utf-8",
    )
    weights = tmp_path / "unused.safetensors"
    weights.touch()  # existence only; no loader may read this file
    return dict(
        dataset_config=str(dataset),
        dit=str(weights),
        vae=str(weights),
        text_encoder=str(weights),
        network_module="networks.lora_qwen_image",
        sdpa=True,
        optimizer_type="AdamW",
        output_dir=str(tmp_path / "output"),
        output_name="adapter",
        max_data_loader_n_workers=0,
    )


def forbid_effects(monkeypatch):
    effects = []

    def forbidden(*args, **kwargs):
        effects.append("called")
        pytest.fail("invalid input reached a model/tokenizer/tracker/cache/training boundary")

    from musubi_tuner.training import trainer_base
    from musubi_tuner import cache_latents, cache_text_encoder_outputs

    for name in ("load_vae", "load_qwen2_5_vl"):
        monkeypatch.setattr(training.qwen_image_utils, name, forbidden)
    monkeypatch.setattr(training.qwen_image_model, "load_qwen_image_model", forbidden)
    monkeypatch.setattr(trainer_base, "prepare_accelerator", forbidden)
    monkeypatch.setattr(trainer_base.NetworkTrainer, "_run_training_loop", forbidden)
    monkeypatch.setattr(cache_latents, "encode_datasets", forbidden)
    monkeypatch.setattr(cache_text_encoder_outputs, "prepare_cache_files_and_paths", forbidden)
    monkeypatch.setattr(cache_text_encoder_outputs, "post_process_cache_files", forbidden)
    return effects


def test_training_dataset_rejects_validation_roles_before_model(tmp_path, monkeypatch, invocation):
    declaration = toml.load(invocation["dataset_config"])
    first = declaration["datasets"][0]
    first["role"] = "val_familiar"
    declaration["datasets"].append(
        {**first, "role": "val_unfamiliar", "cache_directory": str(tmp_path / "other-cache")}
    )
    Path(invocation["dataset_config"]).write_text(toml.dumps(declaration), encoding="utf-8")
    settings = tmp_path / "training-with-validation-as-data.toml"
    settings.write_text(toml.dumps(invocation), encoding="utf-8")
    effects = forbid_effects(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["fixture", "--config_file", str(settings)])

    with pytest.raises(ValueError, match="role"):
        training.main()
    assert effects == []


def test_validation_input_preflight_runs_before_accelerator(tmp_path, monkeypatch, invocation):
    # The original training reader only checks cache paths at this boundary.
    train_cache = tmp_path / "cache"
    train_cache.mkdir()
    (train_cache / "one_0032x0032_qi.safetensors").touch()
    (train_cache / "one_qi_te.safetensors").touch()

    datasets = []
    for role, color in (("val_familiar", (255, 0, 0)), ("val_unfamiliar", (0, 0, 255))):
        image_dir = tmp_path / role
        image_dir.mkdir()
        image_path = image_dir / "one.png"
        Image.new("RGB", (32, 32), color).save(image_path)
        (image_dir / "one.txt").write_text(role, encoding="utf-8")
        image_id = hashlib.sha256(image_path.read_bytes()).hexdigest()
        cache_dir = tmp_path / f"{role}-cache"
        cache_dir.mkdir()
        metadata = {"architecture": "qwen_image", "format_version": "1.0.1", "source_image_sha256": image_id}
        save_file(
            {"latents_1x4x4_float32": torch.ones((2, 1, 4, 4))},
            str(cache_dir / "one_0032x0032_qi.safetensors"),
            metadata={**metadata, "width": "32", "height": "32"},
        )
        save_file(
            {"varlen_vl_embed_float32": torch.ones((2, 4))},
            str(cache_dir / "one_qi_te.safetensors"),
            metadata={**metadata, "caption1": role},
        )
        datasets.append({"role": role, "image_directory": str(image_dir), "cache_directory": str(cache_dir)})
    val_config = tmp_path / "val-dataset.toml"
    val_config.write_text(
        toml.dumps({"general": {"resolution": 32, "caption_extension": ".txt"}, "datasets": datasets}), encoding="utf-8"
    )
    settings = tmp_path / "training-with-validation.toml"
    settings.write_text(toml.dumps({**invocation, "val_dataset_config": str(val_config)}), encoding="utf-8")

    from musubi_tuner.training import trainer_base

    observed = []

    def stop_after_preflight(trainer, args):
        observed.append(trainer.validation_manifest)
        raise RuntimeError("accelerator boundary")

    monkeypatch.setattr(trainer_base.NetworkTrainer, "_prepare_accelerator_and_dtypes", stop_after_preflight)
    monkeypatch.setattr(sys, "argv", ["fixture", "--config_file", str(settings)])
    with pytest.raises(RuntimeError, match="accelerator boundary"):
        training.main()
    assert len(observed) == 1
    assert len(observed[0].items_by_role["val_familiar"]) == 1
    assert len(observed[0].items_by_role["val_unfamiliar"]) == 1


@pytest.mark.parametrize(
    "change,key",
    [
        ({"fp8_scaled": True}, "fp8_scaled"),
        ({"network_args": ["rank_dropuot=0.1"]}, "rank_dropuot"),
        ({"persistent_data_loader_workers": True}, "persistent_data_loader_workers"),
        ({"network_dim": -1}, "network_dim"),
        ({"network_dropout": 1.1}, "network_dropout"),
        ({"blocks_to_swap": 60}, "blocks_to_swap"),
        ({"max_train_steps": 0}, "max_train_steps"),
        ({"sample_every_n_steps": 0}, "sample_every_n_steps"),
        ({"save_every_n_steps": 0}, "save_every_n_steps"),
        ({"optimizer_args": ["bad"]}, "optimizer_args"),
        ({"lr_scheduler": "constant", "lr_warmup_steps": 200}, "lr_warmup_steps"),
        ({"sage_attn": True}, "sage_attn"),
        ({"sdpa": False}, "attention"),
        ({"sdpa": False, "flash3": True}, "flash3"),
        ({"lr_scheduler_args": ["misspelled=2"]}, "lr_scheduler_args"),
        ({"optimizer_args": ["misspelled=2"]}, "optimizer_args"),
        ({"optimizer_args": ["betas=(1.5,0.999)"]}, "betas"),
        ({"lr_scheduler_type": "StepLR"}, "step_size"),
        ({"lr_scheduler": "adafactor:0.001"}, "Adafactor"),
        ({"lr_scheduler": "piecewise_constant", "lr_scheduler_args": ["step_rules='bad'"]}, "step_rules"),
        ({"optimizer_type": "SGD", "optimizer_args": ["momentum=-1"]}, "momentum"),
        ({"lr_scheduler": "polynomial", "learning_rate": 1e-8}, "lr_end"),
        ({"lr_scheduler": "cosine_with_min_lr"}, "lr_scheduler_min_lr_ratio"),
        ({"lr_scheduler": "inverse_sqrt", "lr_scheduler_timescale": 0}, "lr_scheduler_timescale"),
        (
            {"lr_scheduler": "warmup_stable_decay", "lr_decay_steps": 2, "lr_scheduler_args": ["decay_type='bad'"]},
            "decay_type",
        ),
        ({"compile": True, "compile_backend": "not_a_backend"}, "compile_backend"),
        ({"timestep_sampling": "logsnr", "logit_std": -1}, "logit_std"),
        ({"timestep_sampling": "qinglong_flux", "logit_std": -1}, "logit_std"),
        ({"timestep_sampling": "sigma", "weighting_scheme": "logit_normal", "logit_std": -1}, "logit_std"),
    ],
)
def test_training_early_errors_precede_every_effect(tmp_path, monkeypatch, invocation, change, key):
    effects = forbid_effects(monkeypatch)
    config = {**invocation, **change}
    path = tmp_path / "training.toml"
    path.write_text(toml.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["fixture", "--config_file", str(path)])
    with pytest.raises(ValueError) as error:
        training.main()
    assert key in str(error.value)
    assert str(path) in str(error.value)
    assert effects == []


@pytest.mark.parametrize("module_name", ["qwen_image_cache_latents", "qwen_image_cache_text_encoder_outputs"])
@pytest.mark.parametrize("flag,value", [("--batch_size", "0"), ("--num_workers", "-1")])
def test_cache_cli_errors_precede_effects(tmp_path, monkeypatch, invocation, module_name, flag, value):
    import importlib

    module = importlib.import_module("musubi_tuner." + module_name)
    effects = forbid_effects(monkeypatch)
    weight_flag = "--vae" if module_name.endswith("latents") else "--text_encoder"
    monkeypatch.setattr(
        sys, "argv", ["fixture", "--dataset_config", invocation["dataset_config"], weight_flag, invocation["dit"], flag, value]
    )
    with pytest.raises(ValueError, match=flag[2:]):
        module.main()
    assert effects == []


@pytest.mark.parametrize(
    "module_name",
    [
        "networks.lora_qwen_image",
        "networks.loha",
        "networks.lokr",
        "musubi_tuner.networks.lora_qwen_image",
        "musubi_tuner.networks.loha",
        "musubi_tuner.networks.lokr",
    ],
)
def test_non_template_options_remain_valid(tmp_path, monkeypatch, invocation, module_name):
    extra = dict(
        network_module=module_name,
        network_dim=8,
        network_alpha=4,
        network_dropout=0.1,
        network_args=["rank_dropout=0.1", "module_dropout=0.05", "include_patterns=['.*attn.*']", "loraplus_lr_ratio=8"],
        optimizer_type="torch.optim.AdamW",
        optimizer_args=["betas=(0.8, 0.95)", "weight_decay=0.01"],
        max_train_epochs=3,
        gradient_accumulation_steps=2,
        max_data_loader_n_workers=1,
        persistent_data_loader_workers=True,
        save_every_n_epochs=2,
        save_state_on_train_end=True,
        save_last_n_epochs=0,
        save_last_n_epochs_state=0,
        save_precision="fp16",
        mixed_precision="no",
        fp8_base=True,
        fp8_scaled=True,
        blocks_to_swap=3,
        num_layers=6,
        block_swap_h2d_only=True,
        block_swap_ring_size=1,
        gradient_checkpointing=True,
        gradient_checkpointing_cpu_offload=True,
        compile=True,
        compile_backend="eager",
        compile_mode="default",
        split_attn=True,
        timestep_sampling="krea2_shift",
        weighting_scheme="sigma_sqrt",
        min_timestep=50,
        max_timestep=950,
        lr_scheduler="cosine",
        lr_warmup_steps=0.1,
        log_grad_metrics=True,
        metadata_title="fixture",
        huggingface_repo_id="fixture/repo",
        huggingface_repo_type="model",
        async_upload=True,
    )
    if module_name.endswith("lokr"):
        extra["network_args"].append("factor=2")
    args = resolve(monkeypatch, tmp_path, {**invocation, **extra})
    training.validate_training_args(args)
    for key, value in extra.items():
        assert getattr(args, key) == value


def test_unused_attention_flags_have_no_prerequisites(tmp_path, monkeypatch, invocation):
    args = resolve(monkeypatch, tmp_path, {**invocation, "xformers": True, "flash_attn": True, "flash3": True})
    original = training.importlib.util.find_spec

    def check(name, *a, **kw):
        assert name not in ("flash_attn", "xformers")
        return original(name, *a, **kw)

    monkeypatch.setattr(training.importlib.util, "find_spec", check)
    training.validate_training_args(args)


@pytest.mark.parametrize("source_kind", ["prompt", "dataset", "jsonl", "missing_caption"])
def test_training_source_errors_precede_loaders(tmp_path, monkeypatch, invocation, source_kind):
    effects = forbid_effects(monkeypatch)
    if source_kind == "prompt":
        prompt = tmp_path / "bad.txt"
        prompt.write_text("scene --f 3", encoding="utf-8")
        invocation.update(sample_prompts=str(prompt), sample_at_first=True)
    elif source_kind == "dataset":
        Path(invocation["dataset_config"]).write_text('[[datasets]]\nvideo_directory="excluded"', encoding="utf-8")
    elif source_kind == "jsonl":
        source = tmp_path / "bad.jsonl"
        source.write_text(
            json.dumps({"image_path": str(tmp_path / "images/one.png"), "caption": "scene", "image_path_1": "excluded"}),
            encoding="utf-8",
        )
        Path(invocation["dataset_config"]).write_text(
            toml.dumps({"datasets": [{"image_jsonl_file": str(source), "cache_directory": str(tmp_path / "cache")}]}),
            encoding="utf-8",
        )
    elif source_kind == "missing_caption":
        (tmp_path / "images/one.txt").unlink()
    path = tmp_path / "training.toml"
    path.write_text(toml.dumps(invocation), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["fixture", "--config_file", str(path)])
    with pytest.raises(ValueError) as error:
        training.main()
    assert str(tmp_path) in str(error.value)
    assert effects == []


@pytest.mark.parametrize("module_name", ["qwen_image_cache_latents", "qwen_image_cache_text_encoder_outputs"])
def test_cache_source_content_errors_precede_effects(tmp_path, monkeypatch, invocation, module_name):
    import importlib

    effects = forbid_effects(monkeypatch)
    module = importlib.import_module("musubi_tuner." + module_name)
    (tmp_path / "images/one.png").write_text("not an image", encoding="utf-8")
    weight_flag = "--vae" if module_name.endswith("latents") else "--text_encoder"
    monkeypatch.setattr(sys, "argv", ["fixture", "--dataset_config", invocation["dataset_config"], weight_flag, invocation["dit"]])
    with pytest.raises(ValueError, match="source"):
        module.main()
    assert effects == []


@pytest.mark.parametrize("warmup", [2, 0.2])
def test_actual_scheduler_integer_and_ratio_progression(tmp_path, monkeypatch, invocation, warmup):
    import torch

    args = resolve(
        monkeypatch,
        tmp_path,
        {
            **invocation,
            "lr_scheduler": "constant_with_warmup",
            "lr_warmup_steps": warmup,
            "max_train_steps": 10,
            "learning_rate": 0.01,
        },
    )
    training.validate_training_args(args)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=args.learning_rate)
    scheduler = training.NetworkTrainer().get_lr_scheduler(args, optimizer, 1)
    observed = [scheduler.get_last_lr()[0]]
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        observed.append(scheduler.get_last_lr()[0])
    assert observed == pytest.approx([0, 0.005, 0.01, 0.01, 0.01])


def test_scheduler_float_ratio_percent_and_unused_fields(tmp_path, monkeypatch, invocation):
    args = resolve(monkeypatch, tmp_path, {**invocation, "lr_scheduler": "constant_with_warmup", "lr_warmup_steps": 200.0})
    assert type(args.lr_warmup_steps) is float
    with pytest.raises(ValueError, match="ratio"):
        training.validate_scheduler_args(args)
    args = resolve(monkeypatch, tmp_path, {**invocation, "lr_scheduler": "constant_with_warmup"}, ["--lr_warmup_steps", "20%"])
    assert args.lr_warmup_steps == 0.2
    args.lr_scheduler_type = "StepLR"
    args.lr_scheduler_args = ["step_size=2"]
    args.lr_warmup_steps = 200.0
    training.validate_training_args(args)
    args.lr_scheduler_type = ""
    args.optimizer_type = "FixtureScheduleFree"
    training.validate_scheduler_args(args)  # existing early return does not consume warmup


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"optimizer_args": ["betas=(0.8,0.95)"]}, [0.01] * 4),
        ({"lr_scheduler_type": "StepLR", "lr_scheduler_args": ["step_size=2", "gamma=0.5"]}, [0.01, 0.01, 0.005, 0.005]),
        ({"lr_scheduler": "piecewise_constant", "lr_scheduler_args": ["step_rules='1:2,0.5'"]}, [0.01, 0.01, 0.005, 0.005]),
        ({"optimizer_type": "Adafactor"}, [0.01] * 4),
        ({"optimizer_type": "Adafactor", "optimizer_args": ["relative_step=False", "warmup_init=True"]}, [0.01] * 4),
        ({"optimizer_type": "SGD", "optimizer_args": ["momentum=0"]}, [0.01] * 4),
        (
            {"lr_scheduler": "polynomial", "learning_rate": 1e-6, "max_train_steps": 4},
            [1e-6, 7.75e-7, 5.5e-7, 3.25e-7],
        ),
        (
            {"lr_scheduler": "polynomial", "learning_rate": 1e-8, "max_train_steps": 4, "lr_scheduler_args": ["lr_end=1e-10"]},
            [1e-8, 7.525e-9, 5.05e-9, 2.575e-9],
        ),
        (
            {"lr_scheduler": "inverse_sqrt", "lr_scheduler_timescale": 4},
            [0.01, 0.01 * (4 / 5) ** 0.5, 0.01 * (4 / 6) ** 0.5, 0.01 * (4 / 7) ** 0.5],
        ),
        (
            {"lr_scheduler": "cosine_with_min_lr", "lr_scheduler_min_lr_ratio": 0.1, "max_train_steps": 3},
            [0.01, 0.00775, 0.00325, 0.001],
        ),
        (
            {"lr_scheduler": "cosine_with_min_lr", "lr_scheduler_args": ["min_lr=0.001"], "max_train_steps": 3},
            [0.01, 0.00775, 0.00325, 0.001],
        ),
        (
            {
                "lr_scheduler": "warmup_stable_decay",
                "lr_decay_steps": 2,
                "max_train_steps": 3,
                "lr_scheduler_args": ["decay_type='linear'"],
            },
            [0.01, 0.01, 0.005, 0],
        ),
        ({"lr_scheduler": "constant", "learning_rate": 1e-8, "lr_scheduler_timescale": 0}, [1e-8] * 4),
        (
            {
                "lr_scheduler_type": "StepLR",
                "lr_scheduler": "polynomial",
                "learning_rate": 1e-8,
                "lr_scheduler_args": ["step_size=2", "gamma=0.5"],
            },
            [1e-8, 1e-8, 5e-9, 5e-9],
        ),
        ({"optimizer_type": "Adafactor", "lr_scheduler": "cosine_with_min_lr"}, [0.01] * 4),
    ],
)
def test_optimizer_scheduler_preflight_matches_consumers(tmp_path, monkeypatch, invocation, config, expected):
    import torch

    args = resolve(monkeypatch, tmp_path, {**invocation, "learning_rate": 0.01, **config})
    before = vars(args).copy()
    rng = torch.get_rng_state().clone()
    training.validate_training_args(args)
    assert vars(args) == before
    assert torch.equal(torch.get_rng_state(), rng)
    trainer = training.NetworkTrainer()
    optimizer = trainer.get_optimizer(args, [torch.nn.Parameter(torch.ones(1))])[2]
    scheduler = trainer.get_lr_scheduler(args, optimizer, 1)
    observed = [scheduler.get_last_lr()[0]]
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        observed.append(scheduler.get_last_lr()[0])
    assert observed == pytest.approx(expected)
    if config.get("optimizer_type") == "Adafactor":
        assert args.learning_rate is None and args.lr_scheduler == "adafactor:0.01"


@pytest.mark.parametrize("kind", ["missing_tracker", "malformed_tracker", "output_file", "output_parent_file"])
def test_execution_paths_fail_before_loaders(tmp_path, monkeypatch, invocation, kind):
    effects = forbid_effects(monkeypatch)
    if kind.endswith("tracker"):
        tracker = tmp_path / "tracker.toml"
        if kind == "malformed_tracker":
            tracker.write_text("[broken", encoding="utf-8")
        invocation["log_tracker_config"] = str(tracker)
        key = "log_tracker_config"
    else:
        output = tmp_path / "output-file"
        output.touch()
        invocation["output_dir"] = str(output if kind == "output_file" else output / "new")
        key = "output_dir"
    resolve(monkeypatch, tmp_path, invocation)
    with pytest.raises(ValueError) as error:
        training.main()
    assert key in str(error.value) and "settings.toml" in str(error.value)
    assert "correct" in str(error.value)
    assert effects == []


@pytest.mark.parametrize(
    "config",
    [
        {"compile": True, "compile_backend": "eager", "timestep_sampling": "logsnr", "logit_std": 0},
        {"compile": False, "compile_backend": "not_a_backend", "timestep_sampling": "uniform", "logit_std": -1},
        {"timestep_sampling": "qinglong_qwen", "logit_std": -1},
        {"timestep_sampling": "sigma", "weighting_scheme": "none", "logit_std": -1},
    ],
)
def test_valid_execution_inputs_and_unused_options(tmp_path, monkeypatch, invocation, config):
    tracker = tmp_path / "tracker.toml"
    tracker.write_text('[wandb]\nname="fixture"\n', encoding="utf-8")
    output = tmp_path / "new" / "output"
    args = resolve(
        monkeypatch,
        tmp_path,
        {**invocation, **config, "log_tracker_config": str(tracker), "output_dir": str(output)},
    )
    training.validate_training_args(args)
    assert not output.exists()


@pytest.mark.parametrize("equals", [False, True])
def test_raw_abbreviated_network_cannot_be_masked(tmp_path, monkeypatch, invocation, equals):
    effects = forbid_effects(monkeypatch)
    cli = ["--network_m=networks.lora_wan"] if equals else ["--network_m", "networks.lora_wan"]
    cli += ["--network_module", "networks.lora_qwen_image"]
    path = tmp_path / "training.toml"
    path.write_text(toml.dumps(invocation), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["fixture", "--config_file", str(path), *cli])
    with pytest.raises(ValueError, match="CLI.*network_module.*networks.lora_wan"):
        training.main()
    assert effects == []


@pytest.mark.parametrize("equals", [False, True])
def test_valid_abbreviated_network_override(tmp_path, monkeypatch, invocation, equals):
    cli = ["--network_m=networks.lora_qwen_image"] if equals else ["--network_m", "networks.lora_qwen_image"]
    args = resolve(monkeypatch, tmp_path, {**invocation, "network_module": "networks.loha"}, cli)
    training.validate_training_args(args)
    assert args.network_module == "networks.lora_qwen_image"


@pytest.mark.parametrize("size,no_upscale,valid", [(8, True, False), (8, False, True), (32, True, True)])
def test_latent_bucket_dimensions_before_vae(tmp_path, monkeypatch, invocation, size, no_upscale, valid):
    from PIL import Image
    from musubi_tuner import qwen_image_cache_latents as cache

    Image.new("RGB", (size, size)).save(tmp_path / "images/one.png")
    path = Path(invocation["dataset_config"])
    config = toml.load(path)
    config["general"].update(enable_bucket=True, bucket_no_upscale=no_upscale)
    path.write_text(toml.dumps(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["fixture", "--dataset_config", str(path), "--vae", invocation["vae"]])
    reached = []

    class AtLoader(Exception):
        pass

    def stop_at_loader(*args, **kwargs):
        reached.append(True)
        raise AtLoader

    monkeypatch.setattr(cache.qwen_image_utils, "load_vae", stop_at_loader)
    if valid:
        with pytest.raises(AtLoader):
            cache.main()
        assert reached == [True]
    else:
        with pytest.raises(ValueError) as error:
            cache.main()
        message = str(error.value)
        assert all(part in message for part in (str(path), "dataset 1", "item 1", "one.png", "bucket", "0"))
        assert reached == []
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("selection", [None, "SGD"])
def test_parser_optimizer_default_and_explicit_selection(tmp_path, monkeypatch, invocation, selection):
    import torch

    del invocation["optimizer_type"]
    if selection is not None:
        invocation["optimizer_type"] = selection
    args = resolve(monkeypatch, tmp_path, {**invocation, "learning_rate": 0.002})
    assert args.optimizer_type == (selection or "AdamW")
    training.validate_training_args(args)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = training.NetworkTrainer().get_optimizer(args, [parameter])[2]
    assert type(optimizer) is (torch.optim.SGD if selection == "SGD" else torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.002
    assert optimizer.param_groups[0]["params"][0] is parameter


def test_h2d_block_swap_requires_checkpointing_before_effects(tmp_path, monkeypatch, invocation):
    effects = forbid_effects(monkeypatch)
    resolve(monkeypatch, tmp_path, {**invocation, "blocks_to_swap": 1, "block_swap_h2d_only": True})
    with pytest.raises(ValueError) as error:
        training.main()
    message = str(error.value)
    assert all(part in message for part in ("settings.toml", "blocks_to_swap", "block_swap_h2d_only", "gradient_checkpointing"))
    assert "requires" in message and "disable" in message
    assert effects == []


@pytest.mark.parametrize("blocks,h2d,checkpointing", [(1, True, True), (0, True, False), (1, False, False)])
def test_valid_h2d_block_swap_controls(tmp_path, monkeypatch, invocation, blocks, h2d, checkpointing):
    import torch
    from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig

    args = resolve(
        monkeypatch,
        tmp_path,
        {**invocation, "blocks_to_swap": blocks, "block_swap_h2d_only": h2d, "gradient_checkpointing": checkpointing},
    )
    before = vars(args).copy()
    training.validate_training_args(args)
    assert vars(args) == before
    if blocks:
        policy = BlockSwapConfig.from_args(args, torch.device("cpu"), supports_backward=True)
        assert policy.h2d_only == h2d


def test_rex_scheduler_spelling_equivalence(tmp_path, monkeypatch, invocation):
    import torch

    observed = []
    for spelling in ("rex", "REX"):
        args = resolve(monkeypatch, tmp_path, {**invocation, "lr_scheduler": spelling, "max_train_steps": 3, "learning_rate": 0.01})
        before = vars(args).copy()
        training.validate_training_args(args)
        assert vars(args) == before
        trainer = training.NetworkTrainer()
        optimizer = trainer.get_optimizer(args, [torch.nn.Parameter(torch.ones(1))])[2]
        scheduler = trainer.get_lr_scheduler(args, optimizer, 1)
        rates = [scheduler.get_last_lr()[0]]
        for _ in range(3):
            optimizer.step()
            scheduler.step()
            rates.append(scheduler.get_last_lr()[0])
        observed.append(rates)
    assert observed[0] == observed[1]
    assert observed[0][0] == pytest.approx(0.01)
    assert observed[0][-1] == pytest.approx(0.0001)


@pytest.mark.parametrize("module_name", ["qwen_image_cache_latents", "qwen_image_cache_text_encoder_outputs"])
@pytest.mark.parametrize("kind", ["file", "parent_file", "ancestor_file", "new"])
def test_cache_directory_ancestors_before_loaders(tmp_path, monkeypatch, invocation, module_name, kind):
    import importlib

    module = importlib.import_module("musubi_tuner." + module_name)
    parent = tmp_path / "cache-parent"
    if kind != "new":
        parent.write_text("preserve this file", encoding="utf-8")
    cache = parent if kind == "file" else parent / "cache" if kind == "parent_file" else parent / "nested" / "cache"
    path = Path(invocation["dataset_config"])
    config = toml.load(path)
    config["datasets"][0]["cache_directory"] = str(cache)
    path.write_text(toml.dumps(config), encoding="utf-8")
    weight_flag = "--vae" if module_name.endswith("latents") else "--text_encoder"
    monkeypatch.setattr(sys, "argv", ["fixture", "--device", "cpu", "--dataset_config", str(path), weight_flag, invocation["dit"]])
    reached = []

    class AtLoader(Exception):
        pass

    def stop_at_loader(*args, **kwargs):
        reached.append(True)
        raise AtLoader

    if kind == "new":
        loader = "load_vae" if module_name.endswith("latents") else "load_qwen2_5_vl"
        monkeypatch.setattr(module.qwen_image_utils, loader, stop_at_loader)
        with pytest.raises(AtLoader):
            module.main()
        assert reached == [True]
        assert not parent.exists()
    else:
        with pytest.raises(ValueError) as error:
            module.main()
        message = str(error.value)
        assert all(
            part in message for part in (str(path), "dataset 1", "cache_directory", str(cache), str(parent), "file", "directory")
        )
        assert "supply" in message
        assert reached == []
        assert parent.read_text(encoding="utf-8") == "preserve this file"
