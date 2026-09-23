"""Physical Stage 4 example contract and small reusable CPU sources."""

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
from types import SimpleNamespace

from accelerate import Accelerator
import pytest
from safetensors.torch import load_file
import toml
import torch

from musubi_tuner import qwen_image_cache_latents, qwen_image_cache_text_encoder_outputs
from musubi_tuner.qwen_image_train_network import qwen_image_setup_parser
from musubi_tuner.qwen_image_train_network import QwenImageNetworkTrainer
from musubi_tuner.dataset import config_utils
import musubi_tuner.training.trainer_base as trainer_base
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.training.experiment_states import (
    decide_best_event, load_package, promote_published_current_to_best,
    prune_current_packages, read_best_package, save_package,
)
from musubi_tuner.training.validation_inputs import (
    iter_noise_checks, prepare_validation_inputs, read_validation_cache_pair, stable_hash,
)

from test_qwen_image_experiment_paths import _cache_main, make_experiment
from test_qwen_image_experiment_states import _assert_qwen_loader_accepts, _prepared_state, _tiny_adapter, _update
from test_qwen_image_validation_training import TAGS, TinyQwenBoundary, _event_scalars, _write_validation_item


EXAMPLE = Path(__file__).resolve().parents[1] / "qwen_image_lora_val_example"
EXAMPLE_FILES = ("train.toml", "train-dataset.toml", "val-dataset.toml", "sample_prompts.txt")
GENERAL = {
    "resolution": [1024, 1024],
    "enable_bucket": True,
    "bucket_no_upscale": True,
    "caption_extension": ".txt",
    "batch_size": 1,
    "num_repeats": 1,
}
TRAIN_VALUES = {
    "model_version": "original",
    "network_module": "networks.lora_qwen_image",
    "network_dim": 16,
    "network_alpha": 16,
    "experiment_dir": ".",
    "dataset_config": "train-dataset.toml",
    "val_dataset_config": "val-dataset.toml",
    "output_dir": "output",
    "output_name": "qwen_image_lora",
    "mixed_precision": "bf16",
    "fp8_base": False,
    "fp8_scaled": False,
    "fp8_vl": False,
    "blocks_to_swap": 0,
    "sdpa": True,
    "gradient_checkpointing": True,
    "max_train_steps": 1600,
    "gradient_accumulation_steps": 1,
    "seed": 42,
    "optimizer_type": "adamw8bit",
    "learning_rate": 5e-5,
    "lr_scheduler": "constant_with_warmup",
    "lr_warmup_steps": 200,
    "max_grad_norm": 1.0,
    "timestep_sampling": "shift",
    "discrete_flow_shift": 2.2,
    "weighting_scheme": "none",
    "max_data_loader_n_workers": 0,
    "persistent_data_loader_workers": False,
    "val_every_n_steps": 200,
    "val_seed_noise": 42,
    "val_level_noise_n": 10,
    "val_seed_noise_n": 1,
    "save_precision": "fp32",
    "save_every_n_steps": 200,
    "save_last_n_steps": 1000,
    "save_state": True,
    "log_with": "tensorboard",
    "logging_dir": "output/tensorboard",
    "sample_prompts": "sample_prompts.txt",
    "sample_every_n_steps": 200,
    "sample_at_first": False,
}
PROMPTS = [
    "TOK, a gold coin on a plain background. --w 1024 --h 1024 --d 42 --s 30 --l 4.0 --fs 2.2",
    "TOK, a copper teapot on a table. --w 1024 --h 1024 --d 42 --s 30 --l 4.0 --fs 2.2",
]


@pytest.fixture
def small_experiment(tmp_path):
    """Reuse Stage 1/2/3 helpers; production example files are never modified."""
    case = make_experiment(tmp_path / "small-experiment")
    root = case["root"]
    small = _write_validation_item(tmp_path / "source", "val_familiar", "small", (32, 64), 2)
    for source, target in (
        (small["image"], root / "dataset/val_familiar/small.png"),
        (small["image"].with_suffix(".txt"), root / "dataset/val_familiar/small.txt"),
        (small["latent"], root / "cache/val_familiar/small_0032x0064_qi.safetensors"),
        (small["text"], root / "cache/val_familiar/small_qi_te.safetensors"),
    ):
        source.replace(target)
    case["tiny_adapter_factory"] = _tiny_adapter
    return case


def test_small_experiment_has_captioned_train_and_two_sizes_with_real_caches(small_experiment):
    case = small_experiment
    manifest = prepare_validation_inputs(
        SimpleNamespace(val_dataset_config=str(case["val_dataset"]), _experiment_root=str(case["root"]))
    )
    assert {role: len(items) for role, items in manifest.items_by_role.items()} == {
        "val_familiar": 2,
        "val_unfamiliar": 1,
    }
    assert {item.bucket_size for item in manifest.items} == {(64, 64), (32, 64)}
    assert case["items"]["train"]["image"].is_file()
    assert case["items"]["train"]["caption"].is_file()
    assert case["items"]["train"]["latent"].is_file()
    assert case["items"]["train"]["text"].is_file()
    assert callable(case["tiny_adapter_factory"])
    for item in manifest.items:
        latent, text = read_validation_cache_pair(item)
        assert latent.numel() > 0 and text.numel() > 0


def test_example_files_and_values(monkeypatch):
    assert {name for name in EXAMPLE_FILES if (EXAMPLE / name).is_file()} == set(EXAMPLE_FILES)

    settings = toml.load(EXAMPLE / "train.toml")
    assert set(settings) == set(TRAIN_VALUES) | {"dit", "vae", "text_encoder"}
    for key, expected in TRAIN_VALUES.items():
        assert settings[key] == expected, key
        assert type(settings[key]) is type(expected), key
    for key in ("dit", "vae", "text_encoder"):
        assert isinstance(settings[key], str) and PurePosixPath(settings[key]).is_absolute(), key
    assert "save_last_n_steps_state" not in settings

    parser = qwen_image_setup_parser(setup_parser_common())
    assert not set(settings) - {action.dest for action in parser._actions}
    monkeypatch.setattr("sys.argv", ["qwen_image_train_network", "--config_file", str(EXAMPLE / "train.toml")])
    args = read_config_from_file(parser.parse_args(), parser)
    for key, expected in TRAIN_VALUES.items():
        assert getattr(args, key) == expected, key

    train_dataset = toml.load(EXAMPLE / "train-dataset.toml")
    val_dataset = toml.load(EXAMPLE / "val-dataset.toml")
    assert train_dataset == {
        "general": GENERAL,
        "datasets": [{"image_directory": "dataset/train", "cache_directory": "cache/train"}],
    }
    assert val_dataset == {
        "general": GENERAL,
        "datasets": [
            {"role": "val_familiar", "image_directory": "dataset/val_familiar", "cache_directory": "cache/val_familiar"},
            {"role": "val_unfamiliar", "image_directory": "dataset/val_unfamiliar", "cache_directory": "cache/val_unfamiliar"},
        ],
    }

    assert (EXAMPLE / "sample_prompts.txt").read_text(encoding="utf-8").splitlines() == PROMPTS
    assert len(load_prompts(str(EXAMPLE / "sample_prompts.txt"))) == 2
    for subtree in ("dataset", "cache"):
        assert not [path for path in (EXAMPLE / subtree).rglob("*") if path.is_file() and path.name != ".gitkeep"]
    for path in (
        "dataset/train", "dataset/val_familiar", "dataset/val_unfamiliar",
        "cache/train", "cache/val_familiar", "cache/val_unfamiliar", "output",
    ):
        assert (EXAMPLE / path).is_dir()


@pytest.fixture
def copied_example(small_experiment):
    """Copy the shipped files over small real sources, replacing only CPU test inputs."""
    case = small_experiment
    root = case["root"]
    for name in EXAMPLE_FILES:
        shutil.copyfile(EXAMPLE / name, root / name)
    settings = toml.load(root / "train.toml")
    settings.update(
        dit=str(root / "models/dit.safetensors"),
        vae=str(root / "models/vae.safetensors"),
        text_encoder=str(root / "models/text.safetensors"),
        optimizer_type="AdamW",  # Avoid the server's bitsandbytes dependency in CPU preflight.
    )
    (root / "train.toml").write_text(toml.dumps(settings), encoding="utf-8")
    for name in ("train-dataset.toml", "val-dataset.toml"):
        config = toml.load(root / name)
        config["general"]["resolution"] = [64, 64]
        (root / name).write_text(toml.dumps(config), encoding="utf-8")
    prompts = (root / "sample_prompts.txt").read_text(encoding="utf-8")
    (root / "sample_prompts.txt").write_text(prompts.replace("TOK", "CPU_TRIGGER"), encoding="utf-8")
    return case


def _preflight_copied_example(monkeypatch, case, *cli):
    parser = qwen_image_setup_parser(setup_parser_common())
    monkeypatch.setattr("sys.argv", ["qwen_image_train_network", "--config_file", str(case["train_toml"]), *cli])
    args = read_config_from_file(parser.parse_args(), parser)
    trainer = QwenImageNetworkTrainer()
    trainer.validate_training_inputs(args)
    return args, trainer


def test_copied_example_real_readers_and_distinct_sources_from_other_cwd(copied_example, tmp_path, monkeypatch):
    case = copied_example
    source = Path(__file__).resolve().parents[1] / "config_for_qwen_image_lora"
    user_files = {name: (source / name).read_bytes() for name in ("train.toml", "sample_prompts.txt")}
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    args, trainer = _preflight_copied_example(monkeypatch, case)
    root = case["root"].resolve()
    assert Path(args.experiment_dir) == root
    for key, expected in (
        ("dataset_config", "train-dataset.toml"),
        ("val_dataset_config", "val-dataset.toml"),
        ("sample_prompts", "sample_prompts.txt"),
        ("output_dir", "output"),
        ("logging_dir", "output/tensorboard"),
    ):
        assert Path(getattr(args, key)) == root / expected
    assert {Path(args.dit), Path(args.vae), Path(args.text_encoder)} == {
        root / "models/dit.safetensors", root / "models/vae.safetensors", root / "models/text.safetensors",
    }

    user_config = config_utils.load_user_config(args.dataset_config, experiment_root=args._experiment_root)
    blueprint = config_utils.BlueprintGenerator(config_utils.ConfigSanitizer()).generate(
        user_config, SimpleNamespace(dataset_config=args.dataset_config), architecture="qi"
    )
    group = config_utils.generate_dataset_group_by_blueprint(
        blueprint.dataset_group, experiment_root=args._experiment_root
    )
    config_utils.validate_training_cache_bindings(group, args.dataset_config)
    assert len(group.datasets) == 1 and group.datasets[0].role is None
    assert len(group.datasets[0].datasource) == 1
    train_image, _ = group.datasets[0].datasource.get_caption(0)
    assert Path(train_image).resolve() == case["items"]["train"]["image"].resolve()
    assert Path(group.datasets[0].cache_directory) == root / "cache/train"

    manifest = trainer.validation_manifest
    assert {role: len(items) for role, items in manifest.items_by_role.items()} == {
        "val_familiar": 2, "val_unfamiliar": 1,
    }
    assert {item.bucket_size for item in manifest.items} == {(64, 64), (32, 64)}
    assert all(item.image_path != Path(train_image) for item in manifest.items)
    assert {item.latent_cache_path.parent for item in manifest.items} == {
        root / "cache/val_familiar", root / "cache/val_unfamiliar",
    }
    for item in manifest.items:
        latent, text = read_validation_cache_pair(item)
        assert latent.numel() > 0 and text.numel() > 0
    prompts = load_prompts(args.sample_prompts)
    assert len(prompts) == 2 and all("CPU_TRIGGER" in prompt["prompt"] for prompt in prompts)
    assert {name: (source / name).read_bytes() for name in user_files} == user_files


@pytest.mark.parametrize(
    "damage,pattern",
    [
        ("placeholder", "dit.*missing input file"),
        ("caption", "caption|val_familiar"),
        ("cache", "cache|missing"),
        ("conflict", "output_dir.*experiment_dir"),
        ("unknown", "unsupported parameter"),
    ],
)
def test_copied_example_rejects_invalid_input_before_model(copied_example, monkeypatch, damage, pattern):
    case = copied_example
    train_toml = case["train_toml"]
    settings = toml.load(train_toml)
    if damage == "placeholder":
        settings["dit"] = toml.load(EXAMPLE / "train.toml")["dit"]
    elif damage == "caption":
        case["items"]["val_familiar"]["caption"].unlink()
    elif damage == "cache":
        case["items"]["val_unfamiliar"]["text"].unlink()
    elif damage == "conflict":
        settings["output_dir"] = "other-output"
    else:
        settings["unknown_training_setting"] = True
    train_toml.write_text(toml.dumps(settings), encoding="utf-8")
    with pytest.raises(ValueError, match=pattern):
        _preflight_copied_example(monkeypatch, case)


@pytest.mark.parametrize(
    "module,checkpoint_flag,checkpoint_name",
    [
        (qwen_image_cache_latents, "--vae", "vae.safetensors"),
        (qwen_image_cache_text_encoder_outputs, "--text_encoder", "text.safetensors"),
    ],
    ids=["latent", "text"],
)
@pytest.mark.parametrize("dataset_name", ["train-dataset.toml", "val-dataset.toml"], ids=["train", "validation"])
def test_four_copied_example_cache_cli_selections_from_other_cwd(
    copied_example, tmp_path, monkeypatch, module, checkpoint_flag, checkpoint_name, dataset_name
):
    root = copied_example["root"].resolve()
    elsewhere = tmp_path / "cache-command-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    observed = []
    original_validator = module.config_utils.validate_role_aware_sources

    def capture_sources(group, source):
        observed.append((Path(source).resolve(), [(dataset.role, len(dataset.datasource), Path(dataset.cache_directory))
                                                  for dataset in group.datasets]))
        return original_validator(group, source)

    monkeypatch.setattr(module.config_utils, "validate_role_aware_sources", capture_sources)
    _cache_main(
        monkeypatch,
        module,
        [
            "--train_config", str(root / "train.toml"),
            "--dataset_config", dataset_name,
            checkpoint_flag, str(root / "models" / checkpoint_name),
            "--model_version", "original",
        ],
    )
    assert len(observed) == 1
    assert observed[0][0] == root / dataset_name
    expected = (
        [(None, 1, root / "cache/train")]
        if dataset_name == "train-dataset.toml"
        else [("val_familiar", 2, root / "cache/val_familiar"),
              ("val_unfamiliar", 1, root / "cache/val_unfamiliar")]
    )
    assert observed[0][1] == expected


def _build_saved_example(copied_example, monkeypatch):
    case = copied_example
    args, trainer = _preflight_copied_example(monkeypatch, case)
    manifest = trainer.validation_manifest
    assert args.val_level_noise_n == 10 and args.val_seed_noise_n == 1
    assert args.timestep_sampling == "shift" and args.discrete_flow_shift == 2.2
    assert len(manifest.items_by_role["val_familiar"]) == 2
    assert len(manifest.items_by_role["val_unfamiliar"]) == 1
    assert {item.bucket_size for item in manifest.items} == {(64, 64), (32, 64)}
    expected_levels = [0.05 + (i - 0.5) * 0.90 / 10 for i in range(1, 11)]
    for item in manifest.items:
        assert item.image_id == hashlib.sha256(item.image_path.read_bytes()).hexdigest()
        latent, text = read_validation_cache_pair(item)
        assert latent.numel() > 0 and text.numel() > 0
        checks = list(iter_noise_checks(
            item, latent, val_seed_noise=args.val_seed_noise,
            val_level_noise_n=args.val_level_noise_n, val_seed_noise_n=args.val_seed_noise_n,
        ))
        assert len(checks) == 10
        assert [check.t for check in checks] == pytest.approx(expected_levels)
        assert [check.i for check in checks] == list(range(1, 11))
        assert all(check.j == 1 and check.seed == stable_hash(42, item.image_id, check.i, 1) for check in checks)
        assert [check.timestep for check in checks] == pytest.approx([1000 * t for t in expected_levels])
        repeated = iter_noise_checks(
            item, latent, val_seed_noise=args.val_seed_noise,
            val_level_noise_n=args.val_level_noise_n, val_seed_noise_n=args.val_seed_noise_n,
        )
        for first, second in zip(checks, repeated):
            torch.testing.assert_close(first.epsilon, second.epsilon, rtol=0, atol=0)

    accelerator = Accelerator(cpu=True, mixed_precision="no", log_with="tensorboard", project_dir=args.logging_dir)
    accelerator.init_trackers("stage4-example")
    try:
        _, frozen, network, optimizer, scheduler = _prepared_state(accelerator)
        trainer._register_hooks_and_resume(args, accelerator, network)
        _update(accelerator, network, optimizer, scheduler)
        transformer = TinyQwenBoundary()
        noise_scheduler = SimpleNamespace(sigmas=torch.tensor([0.2, 0.7]), timesteps=torch.tensor([200.0, 700.0]))
        metrics = trainer.evaluate_validation_event(
            args, accelerator, transformer, network, noise_scheduler, torch.float32, torch.float32, 200,
        )
        assert set(metrics) == TAGS and all(math.isfinite(value) for value in metrics.values())
        assert len(transformer.timesteps) == 30
        assert sorted({round(float(t.item()), 6) for t in transformer.timesteps}) == pytest.approx(expected_levels)
        assert transformer.inputs_require_grad == [False] * 30
        accelerator.get_tracker("tensorboard").writer.flush()
        series = _event_scalars(Path(args.logging_dir))
        assert set(series) == TAGS
        for tag, value in metrics.items():
            assert len(series[tag]) == 1 and series[tag][0].step == 200
            assert series[tag][0].value == pytest.approx(value, rel=1e-6, abs=1e-6)

        package = save_package(args, accelerator, network, manifest, 200, metrics, ["periodic"])
        assert package == Path(args.output_dir) / "current_training_states/qwen_image_lora-step-200"
        assert sorted(path.name for path in package.glob("*.safetensors")) == ["model.safetensors"]
        assert {"optimizer.bin", "scheduler.bin", "random_states_0.pkl", "val_loss_state.json", "experiment_state.json"} <= {
            path.name for path in package.iterdir()
        }
        assert not list(package.rglob("*.png"))
        assert not list(Path(args.output_dir).rglob("pytorch_model*.bin"))
        assert frozen is not None and len(accelerator._models) >= 2
        tensors = load_file(str(package / "model.safetensors"))
        assert tensors and all(
            tensor.dtype == (torch.float32 if tensor.is_floating_point() else torch.int64)
            for tensor in tensors.values()
        )
        assert all(not name.startswith("module.") for name in tensors)
        assert any("lora_down" in name for name in tensors)
        _assert_qwen_loader_accepts(package, tensors)
        stage2 = json.loads((package / "val_loss_state.json").read_text(encoding="utf-8"))
        state = json.loads((package / "experiment_state.json").read_text(encoding="utf-8"))
        assert stage2["absolute_completed_step"] == state["absolute_completed_step"] == 200
        assert stage2["validation_fingerprint"] == state["validation_fingerprint"] == manifest.fingerprint
        assert stage2["controls"] == state["controls"]
        assert state["metrics_at_step"] == metrics
        assert state["save_reasons"] == ["periodic"] and state["sample_count"] == 0
        assert len(list(Path(args.output_dir).glob("current_training_states/*"))) == 1
        return case, args, manifest, metrics, package
    finally:
        accelerator.end_training()
        accelerator.free_memory()


def test_copied_config_fixed_validation_to_one_complete_cpu_package(copied_example, monkeypatch):
    _build_saved_example(copied_example, monkeypatch)


def test_copied_config_metrics_select_one_best_and_retain_inclusive_current_boundary(copied_example, monkeypatch):
    _, args, manifest, metrics, current_200 = _build_saved_example(copied_example, monkeypatch)
    assert args.save_last_n_steps == 1000
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    try:
        _, _, network, _, _ = _prepared_state(accelerator)
        trainer = QwenImageNetworkTrainer()
        trainer.validation_manifest = manifest
        trainer._register_hooks_and_resume(args, accelerator, network)

        assert decide_best_event(args, accelerator, network, manifest, metrics)
        best_200 = promote_published_current_to_best(args, accelerator, network, manifest, 200, metrics)
        assert not current_200.exists()
        assert read_best_package(args, accelerator, network, manifest) == (best_200, metrics["val_loss_mean"])
        assert sorted(path.name for path in best_200.parent.iterdir()) == [best_200.name]
        assert load_package(args, accelerator, network, manifest, best_200) == 200

        old = save_package(args, accelerator, network, manifest, 199, None, ["epoch"])
        boundary = save_package(args, accelerator, network, manifest, 201, None, ["epoch"])
        assert old.is_dir() and boundary.is_dir()
        prune_current_packages(args, accelerator, network, manifest, 1201)
        assert not old.exists()
        assert boundary.is_dir() and load_package(args, accelerator, network, manifest, boundary) == 201
        assert {
            "experiment_state.json", "model.safetensors", "optimizer.bin", "random_states_0.pkl",
            "scheduler.bin", "val_loss_state.json",
        } <= {path.name for path in boundary.iterdir() if path.is_file()}
        assert read_best_package(args, accelerator, network, manifest) == (best_200, metrics["val_loss_mean"])
    finally:
        accelerator.end_training()
        accelerator.free_memory()


def test_copied_example_root_move_keeps_paths_and_frozen_validation_identity(copied_example, tmp_path, monkeypatch):
    case = copied_example
    root = case["root"]
    server_models = tmp_path / "server-models"
    server_models.mkdir()
    settings = toml.load(root / "train.toml")
    for key in ("dit", "vae", "text_encoder"):
        model = server_models / f"{key}.safetensors"
        model.touch()
        settings[key] = str(model)
    (root / "train.toml").write_text(toml.dumps(settings), encoding="utf-8")
    resume_relative = Path("output/current_training_states/qwen_image_lora-step-200")
    (root / resume_relative).mkdir(parents=True)

    before_cwd = tmp_path / "before-cwd"
    before_cwd.mkdir()
    monkeypatch.chdir(before_cwd)
    before_args, before_trainer = _preflight_copied_example(monkeypatch, case, "--resume", str(resume_relative))
    before_manifest = before_trainer.validation_manifest
    assert Path(before_args.resume) == root / resume_relative

    moved = tmp_path / "renamed-experiment"
    shutil.move(str(root), str(moved))
    relocated_case = {**case, "root": moved, "train_toml": moved / "train.toml"}
    after_cwd = tmp_path / "after-cwd"
    after_cwd.mkdir()
    monkeypatch.chdir(after_cwd)
    after_args, after_trainer = _preflight_copied_example(monkeypatch, relocated_case, "--resume", str(resume_relative))
    assert after_trainer.validation_manifest.fingerprint == before_manifest.fingerprint
    for key, relative in (
        ("dataset_config", "train-dataset.toml"),
        ("val_dataset_config", "val-dataset.toml"),
        ("sample_prompts", "sample_prompts.txt"),
        ("output_dir", "output"),
        ("logging_dir", "output/tensorboard"),
    ):
        assert Path(getattr(after_args, key)) == moved / relative
    assert Path(after_args.resume) == moved / resume_relative
    for key in ("dit", "vae", "text_encoder"):
        assert Path(getattr(after_args, key)) == server_models / f"{key}.safetensors"
    before_identity = sorted(
        (item.role, item.image_id, item.latent_sha256, item.text_sha256, item.bucket_size)
        for item in before_manifest.items
    )
    after_identity = sorted(
        (item.role, item.image_id, item.latent_sha256, item.text_sha256, item.bucket_size)
        for item in after_trainer.validation_manifest.items
    )
    assert after_identity == before_identity
    for item in after_trainer.validation_manifest.items:
        latent, text = read_validation_cache_pair(item)
        assert latent.numel() > 0 and text.numel() > 0


def test_copied_example_complete_package_resumes_on_absolute_timeline(copied_example, monkeypatch):
    case, args, manifest, saved_metrics, package = _build_saved_example(copied_example, monkeypatch)
    class OptionalArgs(SimpleNamespace):
        def __getattr__(self, _name):
            return None

    args = OptionalArgs(**vars(args))
    args.resume = str(package)
    args.max_train_steps = 2
    args.max_grad_norm = 0.0
    accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=1,
                              log_with="tensorboard", project_dir=args.logging_dir)
    try:
        _, _, network, optimizer, scheduler = _prepared_state(accelerator)
        trainer = QwenImageNetworkTrainer()
        trainer.validation_manifest = prepare_validation_inputs(args, expected_fingerprint=manifest.fingerprint)
        trainer._register_hooks_and_resume(args, accelerator, network)
        assert trainer.validation_resume_step == 200
        logs = []
        original_log = accelerator.log

        def capture_log(values, *, step):
            logs.append((step, dict(values)))
            return original_log(values, step=step)

        def controlled_batch(_args, _accelerator, _transformer, current_network, *_rest):
            return sum(parameter.float().square().sum() for parameter in current_network.parameters()), {}

        monkeypatch.setattr(accelerator, "log", capture_log)
        monkeypatch.setattr(trainer, "process_batch", controlled_batch)
        monkeypatch.setattr(trainer, "generate_step_logs", lambda _args, loss, *_rest: {"loss/current": loss})
        monkeypatch.setattr(trainer_base, "clean_memory_on_device", lambda _device: None)
        monkeypatch.setattr(trainer_base.sai_model_spec, "build_metadata", lambda *_args, **_kwargs: {})
        dataset = SimpleNamespace(batch_size=1, get_metadata=lambda: {})
        trainer._run_training_loop(
            args, accelerator, "stage4-cpu-session", 0.0,
            SimpleNamespace(datasets=[dataset], num_train_items=1),
            [{"latents": torch.ones((1, 2, 1, 8, 8))}], SimpleNamespace(value=0),
            TinyQwenBoundary(), network, network, optimizer, "AdamW", [],
            lambda: None, lambda: None, scheduler, [], None, None,
            torch.float32, torch.float32,
        )
        metric_logs = [(step, values) for step, values in logs if set(values) == TAGS]
        assert [step for step, _ in metric_logs] == [200, 202]
        assert metric_logs[0][1] == pytest.approx(saved_metrics)
        assert [step for step, values in logs if "loss/current" in values] == [201, 202]
        assert all("loss/current" not in values for step, values in logs if step == 200)
        final = Path(args.output_dir) / "current_training_states/qwen_image_lora-step-202"
        assert final.is_dir()
        assert json.loads((final / "experiment_state.json").read_text(encoding="utf-8"))["absolute_completed_step"] == 202
    finally:
        accelerator.end_training()
        accelerator.free_memory()
