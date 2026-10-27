"""CPU regressions for the real Raw parser, retained LoRA setup and sampling/state contracts."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from accelerate import Accelerator
import pytest
from PIL import Image
from safetensors.torch import load_file, save_file
import torch
from torch import nn
import toml

from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser
from musubi_tuner.networks import lora_krea2
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.utils import sai_model_spec


def _parser():
    return krea2_setup_parser(setup_parser_common())


def _args(tmp_path, *extra):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[general]\n", encoding="utf-8")
    dit = tmp_path / "raw.safetensors"
    dit.touch()
    return _parser().parse_args(
        [
            "--dataset_config",
            str(dataset),
            "--dit",
            str(dit),
            "--output_dir",
            str(tmp_path / "out"),
            "--output_name",
            "adapter",
            "--network_module",
            "networks.lora_krea2",
            "--sdpa",
            *extra,
        ]
    )


def _from_toml(tmp_path, contents, *cli):
    config = tmp_path / "train.toml"
    config.write_text(contents, encoding="utf-8")
    parser = _parser()
    argv = ["--config_file", str(config), *cli]
    return read_config_from_file(parser.parse_args(argv), parser, argv)


def test_effective_config_cli_overrides_toml_even_when_equal_to_defaults(tmp_path):
    args = _from_toml(
        tmp_path,
        "learning_rate=0.005\nmax_train_steps=55\nnetwork_dim=32\n",
        "--learning_rate",
        "0.000002",
        "--max_train_steps",
        "1600",
    )
    assert args.learning_rate == 2e-6
    assert args.max_train_steps == 1600
    assert args.network_dim == 32
    assert args.network_alpha == 1  # A value absent from TOML and CLI keeps its existing default.
    assert args._config_sources["learning_rate"] == "CLI --learning_rate"


@pytest.mark.parametrize("value, expected_type, expected", [("1.0", float, 1.0), ("1", int, 1), ('"100%"', float, 1.0)])
def test_toml_warmup_preserves_step_count_vs_ratio_semantics(tmp_path, value, expected_type, expected):
    args = _from_toml(tmp_path, f'lr_warmup_steps={value}\nlr_scheduler="linear"\nmax_train_steps=100')
    assert type(args.lr_warmup_steps) is expected_type
    assert args.lr_warmup_steps == expected
    trainer = Krea2NetworkTrainer()
    optimizer = trainer.get_optimizer(args, [nn.Parameter(torch.zeros(1))])[2]
    scheduler = trainer.get_lr_scheduler(args, optimizer, 1)
    assert scheduler.lr_lambdas[0](1) == pytest.approx(0.01 if expected_type is float else 1.0)


@pytest.mark.parametrize("buckets", [0, 1])
def test_disabled_timestep_buckets_full_warmup_and_zero_retention_remain_valid(tmp_path, buckets):
    args = _args(
        tmp_path,
        "--num_timestep_buckets",
        str(buckets),
        "--lr_scheduler",
        "constant_with_warmup",
        "--lr_warmup_steps",
        "100%",
        "--save_last_n_steps",
        "0",
        "--save_last_n_epochs",
        "0",
    )
    trainer = Krea2NetworkTrainer()
    assert args.lr_warmup_steps == 1.0 and isinstance(args.lr_warmup_steps, float)
    assert trainer._validate_args_and_init(args)
    trainer.num_timestep_buckets = buckets
    assert 0 <= trainer.get_bucketed_timestep() <= 1
    assert trainer.timestep_range_pool == []


@pytest.mark.parametrize(
    "contents, key",
    [
        ("unknown_option=1", "unknown_option"),
        ('turbo_dit="turbo.safetensors"', "turbo_dit"),
        ('vae_dtype="bf16"', "vae_dtype"),
        ('mixed_precision="invalid"', "mixed_precision"),
        ('max_train_steps="100"', "max_train_steps"),
        ('gradient_checkpointing="false"', "gradient_checkpointing"),
        ('optimizer_args="weight_decay=0.01"', "optimizer_args"),
        ("network_dim=true", "network_dim"),
        ("learning_rate=nan", "learning_rate"),
        ("[one]\nnetwork_dim=4\n[two]\nnetwork_dim=8", "duplicate parameter"),
    ],
)
def test_toml_unknown_types_choices_and_collisions_fail_with_source(tmp_path, contents, key, capsys):
    with pytest.raises(SystemExit):
        _from_toml(tmp_path, contents)
    error = capsys.readouterr().err
    assert key in error and "train.toml" in error


@pytest.mark.parametrize(
    "extra, key",
    [
        (("--network_module", "networks.lora"), "network_module"),
        (("--network_module", "networks.lora_wan"), "network_module"),
        (("--network_args", "unknown=1"), "network_args"),
        (("--network_args", "exclude_patterns=['[']"), "network_args"),
        (("--network_args", "rank_dropout=1"), "network_args"),
        (("--optimizer_args", "not_a_parameter=1"), "optimizer_type"),
        (("--optimizer_args", "betas=(0.9,2)"), "optimizer_type"),
        (("--optimizer_args", "weight_decay=invalid"), "optimizer_args"),
        (("--lr_scheduler_args", "not_a_parameter=1"), "lr_scheduler_args"),
        (("--lr_scheduler", "adafactor"), "lr_scheduler"),
        (("--lr_warmup_steps", "10"), "lr_warmup_steps"),
        (("--lr_scheduler_type", "torch.optim.lr_scheduler.StepLR"), "lr_scheduler_type"),
        (("--flash_attn",), "sdpa"),
        (("--blocks_to_swap", "27"), "blocks_to_swap"),
        (("--block_swap_h2d_only", "--blocks_to_swap", "4"), "block_swap_h2d_only"),
        (("--gradient_checkpointing_cpu_offload",), "gradient_checkpointing_cpu_offload"),
        (("--dim_from_weights",), "dim_from_weights"),
        (("--sample_every_n_steps", "2"), "sample_prompts"),
        (("--save_every_n_steps", "0"), "save_every_n_steps"),
        (("--min_timestep", "500", "--max_timestep", "400"), "min_timestep"),
    ],
)
def test_invalid_effective_settings_fail_before_any_loader(tmp_path, extra, key, monkeypatch):
    trainer = Krea2NetworkTrainer()
    monkeypatch.setattr(trainer, "load_transformer", lambda *a, **k: pytest.fail("model loader was called"))
    monkeypatch.setattr(trainer, "process_sample_prompts", lambda *a, **k: pytest.fail("text encoder loader was called"))
    with pytest.raises(ValueError, match=key):
        trainer._validate_args_and_init(_args(tmp_path, *extra))


def test_supported_dynamic_optimizer_scheduler_and_network_options_remain(tmp_path):
    args = _args(
        tmp_path,
        "--optimizer_type",
        "torch.optim.SGD",
        "--optimizer_args",
        "momentum=0.8",
        "--lr_scheduler_type",
        "torch.optim.lr_scheduler.StepLR",
        "--lr_scheduler_args",
        "step_size=4",
        "--network_args",
        "loraplus_lr_ratio=2",
        "exclude_patterns=['first']",
        "include_patterns=['linear']",
    )
    trainer = Krea2NetworkTrainer()
    assert trainer._validate_args_and_init(args)
    optimizer = trainer.get_optimizer(args, [nn.Parameter(torch.zeros(1))])[2]
    scheduler = trainer.get_lr_scheduler(args, optimizer, 1)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["momentum"] == 0.8
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert scheduler.step_size == 4


def test_documented_default_and_empty_optimizer_alias_are_adamw(tmp_path):
    trainer = Krea2NetworkTrainer()
    args = _args(tmp_path)
    for alias in (args.optimizer_type, ""):
        args.optimizer_type = alias
        assert isinstance(trainer.get_optimizer(args, [nn.Parameter(torch.zeros(1))])[2], torch.optim.AdamW)


@pytest.mark.parametrize("initial_scheduler", ["constant", "cosine_with_min_lr"])
def test_adafactor_relative_and_explicit_scheduler_keep_existing_pairing(tmp_path, initial_scheduler):
    args = _args(tmp_path, "--optimizer_type", "Adafactor", "--lr_scheduler", initial_scheduler)
    trainer = Krea2NetworkTrainer()
    assert trainer._validate_args_and_init(args)
    optimizer = trainer.get_optimizer(args, [nn.Parameter(torch.zeros(1))])[2]
    assert args.learning_rate is None and args.lr_scheduler.startswith("adafactor:")
    assert trainer.get_lr_scheduler(args, optimizer, 1).__class__.__name__ == "AdafactorSchedule"


@pytest.mark.parametrize("from_toml", [False, True])
def test_tracker_toml_is_parsed_before_model_load_with_source(tmp_path, from_toml, monkeypatch):
    tracker = tmp_path / "tracker.toml"
    tracker.write_text('[wandb\nproject = "training"', encoding="utf-8")
    args = _args(tmp_path, "--log_tracker_config", str(tracker))
    if from_toml:
        settings = {
            key: str(getattr(args, key)) for key in ("dataset_config", "dit", "output_dir", "output_name", "network_module")
        }
        settings.update(sdpa=True, log_tracker_config=str(tracker))
        args = _from_toml(tmp_path, toml.dumps(settings))
    trainer = Krea2NetworkTrainer()
    monkeypatch.setattr(trainer, "load_transformer", lambda *a, **k: pytest.fail("model loader called"))
    monkeypatch.setattr(trainer, "process_sample_prompts", lambda *a, **k: pytest.fail("encoder loader called"))
    with pytest.raises(ValueError, match="log_tracker_config") as error:
        trainer._validate_args_and_init(args)
    assert "tracker.toml" in str(error.value)
    if from_toml:
        assert "train.toml" in str(error.value)


def test_valid_tracker_toml_retains_backend_init_kwargs(tmp_path):
    tracker = tmp_path / "tracker.toml"
    tracker.write_text('[wandb]\nproject="training"\nname="raw-run"', encoding="utf-8")
    trainer = Krea2NetworkTrainer()
    assert trainer._validate_args_and_init(_args(tmp_path, "--log_tracker_config", str(tracker)))
    assert trainer._tracker_init_kwargs == {"wandb": {"project": "training", "name": "raw-run"}}


def test_bad_metadata_resolution_fails_before_model_loading(tmp_path, monkeypatch):
    trainer = Krea2NetworkTrainer()
    monkeypatch.setattr(trainer, "load_transformer", lambda *a, **k: pytest.fail("model loader called"))
    with pytest.raises(ValueError, match="metadata_reso"):
        trainer._validate_args_and_init(_args(tmp_path, "--metadata_reso", "garbage"))


@pytest.mark.parametrize("resolution, expected", [("1024", "1024x1024"), ("1024,768", "1024x768")])
def test_valid_metadata_resolution_uses_actual_save_consumer(tmp_path, resolution, expected):
    trainer = Krea2NetworkTrainer()
    args = _args(tmp_path, "--metadata_reso", resolution)
    assert trainer._validate_args_and_init(args)
    metadata = sai_model_spec.build_metadata(None, trainer.architecture, 0, reso=args.metadata_reso)
    assert metadata["modelspec.resolution"] == expected


def test_cosine_with_min_lr_requires_a_minimum_before_models(tmp_path):
    with pytest.raises(ValueError, match="lr_scheduler_min_lr_ratio"):
        Krea2NetworkTrainer()._validate_args_and_init(_args(tmp_path, "--lr_scheduler", "cosine_with_min_lr"))


@pytest.mark.parametrize(
    "extra, expected",
    [
        (("--lr_scheduler_min_lr_ratio", "0.1"), 0.1),
        (("--lr_scheduler_args", "min_lr=0.0000002"), 0.1),
        (("--lr_scheduler_min_lr_ratio", "0"), 0),
    ],
)
def test_cosine_with_min_lr_constructs_and_reaches_configured_minimum(tmp_path, extra, expected):
    trainer = Krea2NetworkTrainer()
    args = _args(tmp_path, "--lr_scheduler", "cosine_with_min_lr", *extra)
    assert trainer._validate_args_and_init(args)
    optimizer = trainer.get_optimizer(args, [nn.Parameter(torch.zeros(1))])[2]
    scheduler = trainer.get_lr_scheduler(args, optimizer, 1)
    assert scheduler.lr_lambdas[0](args.max_train_steps) == pytest.approx(expected)


@pytest.mark.parametrize("step_size", ["'bad'", "0"])
def test_dynamic_torch_step_lr_invalid_arithmetic_is_rejected_early(tmp_path, step_size):
    args = _args(
        tmp_path, "--lr_scheduler_type", "torch.optim.lr_scheduler.StepLR", "--lr_scheduler_args", f"step_size={step_size}"
    )
    with pytest.raises(ValueError, match="lr_scheduler_type"):
        Krea2NetworkTrainer()._validate_args_and_init(args)


def test_valid_dynamic_torch_step_lr_updates_tiny_cpu_optimizer(tmp_path):
    args = _args(
        tmp_path,
        "--optimizer_type",
        "torch.optim.SGD",
        "--learning_rate",
        "0.1",
        "--lr_scheduler_type",
        "torch.optim.lr_scheduler.StepLR",
        "--lr_scheduler_args",
        "step_size=2",
        "gamma=0.5",
    )
    trainer = Krea2NetworkTrainer()
    assert trainer._validate_args_and_init(args)
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = trainer.get_optimizer(args, [parameter])[2]
    scheduler = trainer.get_lr_scheduler(args, optimizer, 1)
    for _ in range(2):
        optimizer.step()  # No gradient/model: exercise the actual scheduler's step semantics.
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_dim_from_weights_reads_adapter_file_and_restores_rank(tmp_path):
    weights = tmp_path / "adapter.safetensors"
    save_file(
        {
            "lora_unet_linear.lora_down.weight": torch.ones(3, 4),
            "lora_unet_linear.lora_up.weight": torch.ones(2, 3),
            "lora_unet_linear.alpha": torch.tensor(3.0),
        },
        str(weights),
    )
    args = _args(tmp_path, "--dim_from_weights", "--network_weights", str(weights))
    model = nn.ModuleDict({"linear": nn.Linear(4, 2)})
    accelerator = SimpleNamespace(print=lambda *a: None)
    network = Krea2NetworkTrainer()._build_network(args, accelerator, model, None, torch.float32)
    assert network.unet_loras[0].lora_down.weight.shape == (3, 4)
    assert torch.equal(network.unet_loras[0].lora_up.weight, torch.ones(2, 3))


def test_base_lora_is_passed_to_loader_before_quantization_and_not_merged_twice(tmp_path, monkeypatch):
    path = tmp_path / "base.safetensors"
    weights = {"lora_unet_linear.lora_down.weight": torch.ones(2, 4)}
    save_file(weights, str(path))
    args = _args(tmp_path, "--base_weights", str(path), "--base_weights_multiplier", "0.5", "--fp8_base", "--fp8_scaled")
    recorded = {}
    sentinel = object()

    def load(path, **kwargs):
        recorded.update(kwargs)
        return sentinel

    monkeypatch.setattr("musubi_tuner.krea2.krea2_utils.load_krea2_dit", load)
    accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer = Krea2NetworkTrainer()
    assert trainer.load_transformer(accelerator, args, args.dit, "torch", False, "cpu", None) is sentinel
    assert recorded["fp8_scaled"] is True and recorded["lora_multipliers"] == [0.5]
    assert torch.equal(
        recorded["lora_weights"][0]["lora_unet_linear.lora_down.weight"], weights["lora_unet_linear.lora_down.weight"]
    )
    # A second merge would dereference this non-model sentinel and fail.
    trainer.merge_base_weights(args, accelerator, sentinel, None, torch.float32)


def test_real_batch_hook_preserves_flow_target_text_padding_and_mse(tmp_path):
    class Prediction(nn.Module):
        config = SimpleNamespace(patch=2)

        def forward(self, **inputs):
            self.inputs = inputs
            return torch.zeros_like(inputs["img"])

    trainer = Krea2NetworkTrainer()
    args = _args(tmp_path, "--gradient_checkpointing")
    model = Prediction()
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
    latents = torch.zeros(2, 16, 1, 4, 4)
    noise = torch.ones_like(latents)
    batch = {"latents": latents, "krea2_vl_embed": [torch.ones(2, 2, 4), torch.ones(1, 2, 4)]}
    timesteps = torch.tensor([100.0, 500.0])
    output = trainer.call_dit(args, accelerator, model, latents, batch, noise, noise * 0.5, timesteps, torch.float32)
    assert output.pred.shape == latents.shape and torch.equal(output.target, noise - latents)
    assert model.inputs["context"].shape == (2, 2, 2, 4)
    assert torch.count_nonzero(model.inputs["context"][1, 1]) == 0
    assert model.inputs["mask"].tolist() == [[True] * 6, [True] * 5 + [False]]
    assert torch.equal(model.inputs["t"], timesteps / 1000)
    loss, metrics = trainer.compute_loss(args, output, timesteps, None, torch.bfloat16, torch.float32, 0)
    assert loss.item() == 1.0 and metrics == {}


def test_starter_train_and_sample_files_use_the_actual_parsers():
    root = Path(__file__).resolve().parents[1]
    config = root / "config_for_krea2_raw_lora" / "train.toml"
    parser = _parser()
    argv = ["--config_file", str(config)]
    args = read_config_from_file(parser.parse_args(argv), parser, argv)
    assert args.network_module == "networks.lora_krea2" and args.optimizer_type == "AdamW"
    assert args.lr_scheduler == "constant" and args.lr_warmup_steps == 0
    assert args.mixed_precision == "bf16" and args.sdpa
    assert args.save_state and args.save_state_on_train_end
    assert (root / args.dataset_config).is_file()
    prompts = load_prompts(root / args.sample_prompts)
    assert prompts[0]["negative_prompt"] and prompts[0]["cfg_scale"] > 1


@pytest.mark.parametrize(
    "format, content",
    [
        ("txt", "subject --f 4"),
        ("json", '[{"prompt":"subject","image_path":"edit.png"}]'),
        ("toml", '[prompt]\n[[prompt.subset]]\nprompt="subject"\nreference_jsonl="refs.jsonl"'),
    ],
)
def test_removed_sample_options_rejected_in_each_format_before_models(tmp_path, format, content):
    path = tmp_path / f"prompts.{format}"
    path.write_text(content, encoding="utf-8")
    vae = tmp_path / "vae.safetensors"
    text = tmp_path / "text.safetensors"
    vae.touch()
    text.touch()
    args = _args(tmp_path, "--sample_prompts", str(path), "--vae", str(vae), "--text_encoder", str(text))
    with pytest.raises(ValueError, match="sample_prompts"):
        Krea2NetworkTrainer()._validate_args_and_init(args)


@pytest.mark.parametrize(
    "format, content",
    [
        ("txt", "subject --w 64 --h 80 --s 8 --l 5.5 --n blurred --d 42"),
        (
            "json",
            '[{"prompt":"subject","width":64,"height":80,"sample_steps":8,"cfg_scale":5.5,"negative_prompt":"blurred","seed":42}]',
        ),
        (
            "toml",
            '[prompt]\nwidth=64\nheight=80\nsample_steps=8\ncfg_scale=5.5\nnegative_prompt="blurred"\nseed=42\n[[prompt.subset]]\nprompt="subject"',
        ),
    ],
)
def test_image_prompt_formats_preserve_sample_values(tmp_path, format, content):
    path = tmp_path / f"prompts.{format}"
    path.write_text(content, encoding="utf-8")
    prompt = load_prompts(path)[0]
    assert prompt == {
        "prompt": "subject",
        "width": 64,
        "height": 80,
        "sample_steps": 8,
        "cfg_scale": 5.5,
        "negative_prompt": "blurred",
        "seed": 42,
        "enum": 0,
    }


class _SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.swap_mode = "training"

    def switch_block_swap_for_inference(self):
        self.swap_mode = "inference"

    def switch_block_swap_for_training(self):
        self.swap_mode = "training"


@pytest.mark.parametrize("fails", [False, True])
def test_sampling_preserves_rng_model_mode_and_swap_and_saves_same_png(tmp_path, fails):
    class Sampler(Krea2NetworkTrainer):
        def do_inference(self, *args, **kwargs):
            torch.rand(7)
            if fails:
                raise RuntimeError("sample failure")
            return torch.full((1, 3, 1, 8, 8), 0.5)

    args = _args(tmp_path)
    args.sample_every_n_steps = 1
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    model, vae = _SampleModel().train(), nn.Identity()
    state = torch.get_rng_state().clone()
    call = lambda: Sampler().sample_images(
        accelerator, args, None, 1, vae, model, [{"prompt": "subject", "seed": 42, "enum": 0}], torch.bfloat16
    )
    if fails:
        with pytest.raises(RuntimeError, match="sample failure"):
            call()
    else:
        call()
        paths = list((Path(args.output_dir) / "sample").glob("adapter_000001_00_*_42_000.png"))
        assert len(paths) == 1
        image = Image.open(paths[0])
        assert image.size == (8, 8) and image.getpixel((0, 0)) == (127, 127, 127)
    assert torch.equal(torch.get_rng_state(), state)
    assert model.training and model.swap_mode == "training"


def test_accelerate_state_saves_only_lora_and_restores_optimizer_scheduler_rng(tmp_path):
    model = nn.ModuleDict({"linear": nn.Linear(4, 2)})
    network = lora_krea2.create_arch_network(1, 2, 2, None, None, model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    model, network, optimizer, scheduler = accelerator.prepare(model, network, optimizer, scheduler)
    args = SimpleNamespace(resume=None, resume_from_huggingface=False)
    trainer = Krea2NetworkTrainer()
    trainer._register_hooks_and_resume(args, accelerator, network)
    checkpoint = tmp_path / "state"
    saved_weights = {key: value.clone() for key, value in network.state_dict().items()}
    saved_rng = torch.get_rng_state().clone()
    accelerator.save_state(str(checkpoint))
    files = list(checkpoint.glob("model*.safetensors"))
    assert len(files) == 1
    assert set(load_file(str(files[0]))) == set(saved_weights)
    assert all(key.startswith("lora_unet_") for key in saved_weights)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.fill_(7)
    optimizer.param_groups[0]["lr"] = 0.5
    scheduler.scheduler.last_epoch = 9
    torch.manual_seed(777)
    args.resume = str(checkpoint)
    assert trainer.resume_from_local_or_hf_if_specified(accelerator, args)
    assert optimizer.param_groups[0]["lr"] == 0.001
    assert scheduler.scheduler.last_epoch == 0
    assert torch.equal(torch.get_rng_state(), saved_rng)
    assert all(torch.equal(value, saved_weights[key]) for key, value in network.state_dict().items())
