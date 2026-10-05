"""Exercise the acceptance probe through its real tiny Qwen/Accelerate loop seam on CPU."""

from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from accelerate import Accelerator
import pytest
import torch

from musubi_tuner.networks import lora_qwen_image
from musubi_tuner.qwen_image.qwen_image_model import QwenImageTransformer2DModel
from musubi_tuner.qwen_image_train_network import qwen_image_setup_parser
from musubi_tuner.training.parser_common import setup_parser_common


def load_probe():
    source = Path(__file__).resolve().parents[1] / "specs/002-deterministic-val-loss/gpu_acceptance_probe.py"
    spec = importlib.util.spec_from_file_location("gpu_acceptance_probe", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixedValidation:
    records = [SimpleNamespace(image_sha256="7" * 64)]
    fingerprint = "CPU tiny fixture; real GPU inputs require cache provenance"

    def __init__(self):
        self.evaluations = 0

    def verify_unchanged(self):
        self.evaluations += 1
        torch.rand(1)

    def load_batch(self, index):
        assert index == 0
        torch.rand(1)
        return batch()


def batch():
    return {"latents": torch.full((1, 2, 1, 4, 4), 0.2), "vl_embed": [torch.ones(2, 8)], "timesteps": None}


@pytest.mark.parametrize(
    "configured_vae,cli_vae,expected_vae",
    [(None, None, "bfloat16"), ("float32", None, "float32"), ("float16", "float32", "float32")],
)
def test_probe_cli_uses_native_qwen_setup(tmp_path, monkeypatch, configured_vae, cli_vae, expected_vae):
    probe = load_probe()
    config = tmp_path / "train.toml"
    config.write_text(
        'model_version = "original"\nseed = 17\nmax_train_steps = 9\n'
        + (f'vae_dtype = "{configured_vae}"\n' if configured_vae else ""),
        encoding="utf-8",
    )
    original_trainer = probe.qwen.QwenImageNetworkTrainer
    native_resolve = probe.qwen.qwen_image_utils.resolve_model_version_args
    resolved, trained = [], []

    def resolve(args):
        result = native_resolve(args)
        resolved.append(args)
        return result

    def train(self, args):
        trained.append((self, args))

    monkeypatch.setattr(probe.qwen.qwen_image_utils, "resolve_model_version_args", resolve)
    monkeypatch.setattr(probe.AcceptanceTrainer, "train", train)
    command = ["probe", "--probe_report", "output/probe.json", "--config_file", str(config), "--max_train_steps", "3"]
    if cli_vae:
        command += ["--vae_dtype", cli_vae]
    monkeypatch.setattr(sys, "argv", command)
    probe.main()
    assert len(trained) == 1 and resolved == [trained[0][1]]
    trainer, args = trained[0]
    assert trainer.report_path == "output/probe.json" and not hasattr(args, "probe_report")
    assert args.dit_dtype == "bfloat16" and args.vae_dtype == expected_vae
    assert args.model_version == "original" and args.seed == 17 and args.max_train_steps == 3
    assert Path(args._config_source) == config
    assert probe.qwen.QwenImageNetworkTrainer is original_trainer and sys.argv is command


@pytest.mark.parametrize(
    "requested,prior_enabled,prior_warn_only", [(False, False, False), (True, False, False), (True, True, True)]
)
@pytest.mark.parametrize("fail", [False, True])
def test_probe_cli_restores_native_factory_arguments_and_determinism(monkeypatch, requested, prior_enabled, prior_warn_only, fail):
    probe = load_probe()
    original_trainer = probe.qwen.QwenImageNetworkTrainer
    command = ["probe", "--probe_report", "output/probe.json"]
    if requested:
        command.append("--probe_deterministic")

    def train(self, args):
        assert args.dit_dtype == args.vae_dtype == "bfloat16" and args.model_version == "original"
        assert self.probe_deterministic is requested
        assert torch.are_deterministic_algorithms_enabled() is (True if requested else prior_enabled)
        assert torch.is_deterministic_algorithms_warn_only_enabled() is (False if requested else prior_warn_only)
        assert not hasattr(args, "probe_deterministic")
        if fail:
            raise RuntimeError("fixture train setup failure")

    monkeypatch.setattr(probe.AcceptanceTrainer, "train", train)
    monkeypatch.setattr(sys, "argv", command)
    original_enabled = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(prior_enabled, warn_only=prior_warn_only)
        with pytest.raises(RuntimeError, match="fixture train setup failure") if fail else nullcontext():
            probe.main()
        assert probe.qwen.QwenImageNetworkTrainer is original_trainer and sys.argv is command
        assert torch.are_deterministic_algorithms_enabled() is prior_enabled
        assert torch.is_deterministic_algorithms_warn_only_enabled() is prior_warn_only
    finally:
        torch.use_deterministic_algorithms(original_enabled, warn_only=original_warn_only)


@pytest.mark.parametrize("loader_batches,failure", [(1, None), (2, None), (3, None), (2, "update"), (2, "baseline_gradient")])
@pytest.mark.parametrize("deterministic", [False, True])
def test_probe_full_cpu_loop_seam_actual_qwen_lora(tmp_path, monkeypatch, loader_batches, failure, deterministic):
    probe = load_probe()
    accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=2)
    args = qwen_image_setup_parser(setup_parser_common()).parse_args([])
    vars(args).update(
        _config_source=str(tmp_path / "train.toml"),
        _experiment_root=str(tmp_path),
        val_dataset_config="CPU memory cache fixture",
        val_level_noise_n=10,
        val_seed_noise_n=2,
        val_seed_noise=42,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        network_dropout=0.2,
        max_grad_norm=1.0,
        timestep_sampling="shift",
        discrete_flow_shift=2.2,
    )
    trainer = probe.AcceptanceTrainer("output/probe.json", deterministic)
    trainer.validation_inputs = FixedValidation()
    model = QwenImageTransformer2DModel(
        in_channels=8,
        out_channels=2,
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=8,
        joint_attention_dim=8,
        axes_dims_rope=(2, 2, 4),
    )
    model.requires_grad_(False)
    model.enable_gradient_checkpointing()
    network = lora_qwen_image.create_arch_network(1, 2, 2, None, [], model, neuron_dropout=0.2)
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    loader = torch.utils.data.DataLoader([batch() for _ in range(loader_batches)], batch_size=1, collate_fn=lambda items: items[0])
    model, network, optimizer, loader, scheduler, training_model, dtype = trainer._prepare_with_accelerator(
        args, accelerator, model, network, optimizer, loader, scheduler, torch.float32, torch.float32, torch.float32
    )
    loader_references = list(accelerator.gradient_state.dataloader_references)
    active_loader = accelerator.gradient_state.active_dataloader
    native_probe = probe.run_probe
    probe_entries = []

    def checked_probe(*positional, **keywords):
        assert accelerator.gradient_state.dataloader_references == loader_references
        assert accelerator.gradient_state.active_dataloader is active_loader
        assert not accelerator.gradient_state.end_of_dataloader
        probe_entries.append(True)
        return native_probe(*positional, **keywords)

    monkeypatch.setattr(probe, "run_probe", checked_probe)
    if failure == "update":

        def fail_update(*positional, **keywords):
            raise RuntimeError("fixture update failure after repeatability")

        monkeypatch.setattr(trainer, "process_batch", fail_update)
    elif failure == "baseline_gradient":
        native_backward = accelerator.backward
        backward_calls = []

        def changed_gradient(*positional, **keywords):
            native_backward(*positional, **keywords)
            backward_calls.append(True)
            if len(backward_calls) == 6:  # Second microbatch of the no-validation replay.
                next(network.parameters()).grad.view(-1)[0].add_(0.25)

        monkeypatch.setattr(accelerator, "backward", changed_gradient)
    before = torch.get_num_threads()
    original_enabled = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    try:
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=False)
        expected_error = (
            pytest.raises(RuntimeError, match="fixture update failure")
            if failure == "update"
            else pytest.raises(AssertionError, match="no-validation replay")
            if failure == "baseline_gradient"
            else nullcontext()
        )
        with expected_error:
            trainer._run_training_loop(
                args,
                accelerator,
                1,
                0.0,
                None,
                loader,
                SimpleNamespace(value=0),
                model,
                network,
                training_model,
                optimizer,
                "AdamW",
                "",
                lambda: None,
                lambda: None,
                scheduler,
                None,
                None,
                None,
                torch.float32,
                dtype,
            )
        assert probe_entries == [True]
        assert accelerator.gradient_state.dataloader_references == loader_references
        assert accelerator.gradient_state.active_dataloader is active_loader
    finally:
        accelerator.free_memory()
        torch.set_num_threads(before)
        torch.use_deterministic_algorithms(original_enabled, warn_only=original_warn_only)
    report = json.loads((tmp_path / "output/probe.json").read_text(encoding="utf-8"))
    assert report["repeatability"]["forward_count_each"] == 20
    assert len(report["repeatability"]["noise"]) == 20
    assert len({pair["noise_sha256"] for pair in report["repeatability"]["noise"]}) == 20
    assert report["repeatability"]["noise"] == report["repeatability"]["second_noise"]
    assert report["paired_tolerance"] == {"rtol": 0, "atol": 0}
    assert report["loss_tolerance"] == {"rtol": 1e-4, "atol": 1e-5}
    assert report["determinism"]["requested"] is deterministic
    assert report["determinism"]["algorithms_enabled"] is (True if deterministic else original_enabled)
    assert report["determinism"]["warn_only"] is (False if deterministic else original_warn_only)
    if failure == "update":
        assert report["status"] == "failed" and "fixture update failure" in report["error"]
        assert report["completed_updates"] == {"warmup": 0}
        assert "isolation" not in report
        return
    baseline = report["no_validation_replay"]
    assert baseline["train_trace"] == baseline["measured_train_trace"]
    assert baseline["control_train_losses"] == baseline["measured_train_losses"]
    if failure == "baseline_gradient":
        assert report["status"] == "failed" and not baseline["exact"] and "isolation" not in report
        assert trainer.validation_inputs.evaluations == 2
        assert report["completed_updates"] == {"warmup": 1, "control": 1, "control_replay": 1}
        before_clip = baseline["gradient_differences"]["pre_clip_gradients"]
        after_clip = baseline["gradient_differences"]["post_clip_gradients"]
        assert before_clip["parameter"] == after_clip["parameter"] == next(iter(network.named_parameters()))[0]
        assert before_clip["unequal_elements"] == 1 and before_clip["max_abs"] > 0
        for difference in (before_clip, after_clip):
            assert difference["control"]["shape"] == list(next(network.parameters()).shape)
            assert difference["control"]["dtype"] == "torch.float32"
            assert difference["control"]["sha256"] != difference["replay"]["sha256"]
        return
    assert report["status"] == "passed"
    assert baseline["exact"] and trainer.validation_inputs.evaluations == 3
    assert baseline["gradient_differences"] == {"pre_clip_gradients": None, "post_clip_gradients": None}
    assert report["restored_state_exact"] == {"control_replay": True, "measured": True}
    assert report["isolation"]["exact"]
    assert len(report["isolation"]["train_trace"]) == 2
    assert report["isolation"]["train_trace"] == report["isolation"]["measured_train_trace"]
    assert report["completed_updates"] == {"warmup": 1, "control": 1, "control_replay": 1, "measured": 1}
    assert report["microbatch_sync"] == {stage: [False, True] for stage in ("warmup", "control", "control_replay", "measured")}
