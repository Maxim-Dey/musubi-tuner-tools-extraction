"""Actual CPU trainer-loop acceptance with tiny Qwen/LoRA and real event/state files."""

import copy
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from musubi_tuner.networks import lora_qwen_image
from musubi_tuner.qwen_image.qwen_image_model import QwenImageTransformer2DModel
from musubi_tuner.qwen_image_train_network import QwenImageNetworkTrainer, qwen_image_setup_parser
from musubi_tuner.training.accelerator_setup import prepare_accelerator
from musubi_tuner.training.parser_common import setup_parser_common


VAL_TAGS = ("val_loss_mean", "val_loss_low_noise", "val_loss_high_noise")


class SmallDataset(torch.utils.data.Dataset):
    batch_size = 1
    num_train_items = 24

    @property
    def datasets(self):
        return [self]

    def get_metadata(self):
        return {"batch_size": 1, "resolution": [32, 32], "num_repeats": 1}

    def __len__(self):
        return self.num_train_items

    def __getitem__(self, index):
        # Include all host RNGs in batch production, making next-batch isolation observable.
        jitter = random.random() / 100 + float(np.random.random()) / 100 + torch.rand(()).item() / 100
        return {
            "latents": torch.full((1, 2, 1, 4, 4), index / 100 + jitter),
            "vl_embed": [torch.arange(16, dtype=torch.float32).reshape(2, 8) / 100],
            "timesteps": None,
        }


class MemoryValidation:
    """Only the external cache boundary is replaced; evaluator and model stay real."""

    fingerprint = "a" * 64
    records = [SimpleNamespace(image_sha256="1" * 64)]
    manifest = {"fixture": "one independent validation image"}

    def __init__(self, failure=False):
        self.checks = 0
        self.loads = 0
        self.failure = failure

    def verify_unchanged(self):
        self.checks += 1
        # Setup randomness must be isolated too, not only generated Gaussian noise.
        random.random(), np.random.random(), torch.rand(())
        if self.failure is True or self.failure == "verify":
            raise ValueError("fixture validation failure")

    def load_batch(self, index):
        assert index == 0
        self.loads += 1
        random.random(), np.random.random(), torch.rand(())
        if self.failure == "load":
            raise ValueError("fixture validation load failure")
        return {"latents": torch.full((1, 2, 1, 4, 4), 0.2), "vl_embed": [torch.ones(2, 8)], "timesteps": None}


class ObservedTrainer(QwenImageNetworkTrainer):
    def __init__(self):
        super().__init__()
        self.train_inputs = []
        self.train_times = []
        self.val_forwards = 0
        self.updates = []

    def process_batch(
        self,
        args,
        accelerator,
        transformer,
        network,
        batch,
        latents,
        noise,
        noise_scheduler,
        dit_dtype,
        network_dtype,
        sample_resources,
        global_step,
    ):
        self.train_inputs.append((global_step, latents.detach().clone(), noise.detach().clone()))
        return super().process_batch(
            args,
            accelerator,
            transformer,
            network,
            batch,
            latents,
            noise,
            noise_scheduler,
            dit_dtype,
            network_dtype,
            sample_resources,
            global_step,
        )

    def call_dit(
        self, args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
    ):
        if torch.is_grad_enabled():
            self.train_times.append(timesteps.detach().clone())
        else:
            self.val_forwards += 1
        return super().call_dit(
            args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
        )

    def on_post_optimizer_step(self, args, accelerator, network, transformer, sync_gradients, global_step):
        if sync_gradients and not accelerator.optimizer_step_was_skipped:
            self.updates.append(global_step)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    count = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(count)


def scalar_events(root):
    events = {}
    run_dirs = {file.parent for file in Path(root).rglob("events.out.tfevents.*")}
    for directory in run_dirs:
        reader = EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()
        for tag in reader.Tags()["scalars"]:
            events.setdefault(tag, []).extend(reader.Scalars(tag))
    return {tag: sorted(values, key=lambda event: (event.step, event.wall_time)) for tag, values in events.items()}, run_dirs


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def run_loop(
    root,
    *,
    target=5,
    accumulation=1,
    validation=True,
    resume=None,
    save=False,
    skip_update=None,
    failure=False,
    experiment=True,
    save_interval=None,
    periodic_state=False,
    sample_interval=None,
    weights_window=None,
    state_window=None,
):
    from musubi_tuner.training.experiment_state import prepare_training_state

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    random.seed(71)
    np.random.seed(71)
    torch.manual_seed(71)
    args = qwen_image_setup_parser(setup_parser_common()).parse_args([])
    vars(args).update(
        experiment_mode=experiment,
        _experiment_root=str(root),
        _config_source=str(root / "train.toml"),
        output_dir=str(root / "output"),
        output_name="tiny",
        logging_dir=str(root / "output/tensorboard"),
        log_with="tensorboard",
        log_prefix="tiny",
        val_dataset_config="memory" if validation else None,
        val_every_n_steps=2 if validation else None,
        val_seed_noise=42 if validation else None,
        val_level_noise_n=2 if validation else None,
        val_seed_noise_n=1 if validation else None,
        network_module="networks.lora_qwen_image",
        network_dim=2,
        network_alpha=2,
        network_dropout=0.15,
        max_train_steps=target,
        gradient_accumulation_steps=accumulation,
        gradient_checkpointing=True,
        mixed_precision="no",
        learning_rate=0.003,
        max_grad_norm=1.0,
        timestep_sampling="shift",
        discrete_flow_shift=2.2,
        save_precision="fp32",
        save_state=periodic_state,
        save_state_on_train_end=save,
        save_every_n_steps=save_interval,
        sample_every_n_steps=sample_interval,
        save_last_n_steps=weights_window,
        save_last_n_steps_state=state_window,
        seed=71,
        resume=str(resume) if resume else None,
    )
    trainer = ObservedTrainer()
    trainer.validation_inputs = MemoryValidation(failure=failure) if validation else None
    state = prepare_training_state(args, trainer.validation_inputs.fingerprint if validation else None)
    accelerator = prepare_accelerator(args)
    transformer = QwenImageTransformer2DModel(
        in_channels=8,
        out_channels=2,
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=8,
        joint_attention_dim=8,
        axes_dims_rope=(2, 2, 4),
    )
    transformer.requires_grad_(False)
    network = lora_qwen_image.create_arch_network(1, 2, 2, None, [], transformer, neuron_dropout=0.15)
    network.apply_to([], transformer, apply_text_encoder=False, apply_unet=True)
    saves = []
    save_weights = network.save_weights

    def observed_save(path, dtype, metadata):
        saves.append(Path(path))
        return save_weights(path, dtype, metadata)

    network.save_weights = observed_save
    transformer.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    dataset = SmallDataset()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: batch[0], generator=torch.Generator().manual_seed(29)
    )
    transformer, network, optimizer, loader, scheduler, training_model, dtype = trainer._prepare_with_accelerator(
        args, accelerator, transformer, network, optimizer, loader, scheduler, torch.float32, torch.float32, torch.float32
    )
    trainer._register_hooks_and_resume(args, accelerator, network)
    step_impl = optimizer.step
    attempts = [0]
    gradients = []

    def step(*step_args, **step_kwargs):
        if accelerator.sync_gradients:
            attempts[0] += 1
            gradients.append(
                [parameter.grad.detach().clone() if parameter.grad is not None else None for parameter in network.parameters()]
            )
            # Real CPU Accelerate optimizer/scheduler, replacing only unavailable CUDA overflow.
            optimizer._is_overflow = attempts[0] == skip_update
            if optimizer._is_overflow:
                return
        return step_impl(*step_args, **step_kwargs)

    optimizer.step = step

    def snapshot():
        return copy.deepcopy(
            (
                network.state_dict(),
                optimizer.state_dict(),
                scheduler.state_dict(),
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state(),
                [module.training for module in transformer.modules()] + [module.training for module in network.modules()],
                [parameter.grad for parameter in network.parameters()],
            )
        )

    before_loop = snapshot()
    sample_parameters = None
    sample_resources = None
    if sample_interval:
        sample_parameters = [dict(prompt="fixture", width=32, height=32, seed=42, sample_steps=1, enum=0)]
        sample_resources = torch.nn.Identity()

        def sample_encoder_boundary(
            accelerator, args, prompt, vae, dtype, transformer, shift, steps, width, height, generator, do_cfg, cfg
        ):
            assert not torch.is_grad_enabled() and not transformer.training
            return torch.full((1, 3, 1, height, width), 0.5)

        trainer.do_inference = sample_encoder_boundary
    try:
        trainer._run_training_loop(
            args,
            accelerator,
            123,
            0.0,
            dataset,
            loader,
            SimpleNamespace(value=0),
            transformer,
            network,
            training_model,
            optimizer,
            "torch.optim.AdamW",
            "",
            lambda: None,
            lambda: None,
            scheduler,
            None,
            sample_resources,
            sample_parameters,
            torch.float32,
            dtype,
        )
        result = SimpleNamespace(
            args=args,
            state=copy.deepcopy(state),
            trainer=trainer,
            attempts=attempts[0],
            network=copy.deepcopy(network.state_dict()),
            optimizer=copy.deepcopy(optimizer.state_dict()),
            scheduler=copy.deepcopy(scheduler.state_dict()),
            gradients=copy.deepcopy(gradients),
            rng=(random.getstate(), np.random.get_state(), torch.get_rng_state()),
            modes=[module.training for module in transformer.modules()] + [module.training for module in network.modules()],
            saves=saves,
        )
    except Exception as error:
        error.loop_states = (before_loop, snapshot())
        raise
    finally:
        accelerator.end_training()
        accelerator.free_memory()
    return result


@pytest.mark.parametrize("target,accumulation,expected", [(5, 1, [0, 2, 4, 5]), (4, 2, [0, 2, 4])])
def test_real_loop_steps_and_exact_event_tags(tmp_path, target, accumulation, expected):
    result = run_loop(tmp_path, target=target, accumulation=accumulation)
    events, runs = scalar_events(tmp_path)
    assert len(runs) == 1
    assert {tag for tag in events if tag.startswith("val_")} == set(VAL_TAGS)
    for tag in VAL_TAGS:
        assert [event.step for event in events[tag]] == expected
    assert [event.step for event in events["loss/current"]] == list(range(1, target + 1))
    assert result.state["optimizer_global_step"] == target
    assert len(result.trainer.train_inputs) == target * accumulation
    assert result.trainer.val_forwards == len(expected) * 2
    assert len(result.trainer.updates) == target
    for mean, low, high in zip(*(events[tag] for tag in VAL_TAGS)):
        assert mean.value == pytest.approx((low.value + high.value) / 2, rel=1e-6, abs=1e-7)


def test_skipped_optimizer_update_does_not_advance_loop_step(tmp_path):
    result = run_loop(tmp_path, target=3, accumulation=2, skip_update=1)
    events, _ = scalar_events(tmp_path)
    assert [event.step for event in events["val_loss_mean"]] == [0, 2, 3]
    assert [event.step for event in events["loss/current"]] == [1, 2, 3]
    assert result.attempts == 4
    assert len(result.trainer.updates) == 3
    assert result.state["optimizer_global_step"] == 3


def test_real_loop_validation_preserves_training_sequence_and_update(tmp_path):
    control = run_loop(tmp_path / "control", target=2, accumulation=2, validation=False)
    measured = run_loop(tmp_path / "measured", target=2, accumulation=2, validation=True)
    for key in ("network", "optimizer", "scheduler", "gradients", "rng", "modes"):
        assert_nested_equal(getattr(control, key), getattr(measured, key))
    assert_nested_equal(control.trainer.train_inputs, measured.trainer.train_inputs)
    assert_nested_equal(control.trainer.train_times, measured.trainer.train_times)
    assert control.trainer.val_forwards == 0 and measured.trainer.val_forwards == 4


def checkpoint_at(root, step):
    from musubi_tuner.training.experiment_state import TRAINER_STATE_FILENAME, load_training_state

    found = [
        path.parent
        for path in Path(root).rglob(TRAINER_STATE_FILENAME)
        if load_training_state(path.parent)["optimizer_global_step"] == step
    ]
    assert len(found) == 1
    return found[0]


def test_real_saved_resume_preserves_absolute_step_and_deduplicates(tmp_path):
    from musubi_tuner.training.experiment_state import load_training_state

    first = run_loop(tmp_path, target=3, save=True)
    checkpoint = checkpoint_at(tmp_path, 3)
    saved = load_training_state(checkpoint)
    assert saved["optimizer_global_step"] == 3
    assert saved["validation_fingerprint"] == MemoryValidation.fingerprint
    assert saved["last_validation"]["step"] == 3
    resumed = run_loop(tmp_path, target=5, resume=checkpoint)
    events, runs = scalar_events(tmp_path)
    assert len(runs) == 1
    assert [event.step for event in events["val_loss_mean"]] == [0, 2, 3, 4, 5]
    assert [event.step for event in events["loss/current"]] == [1, 2, 3, 4, 5]
    assert resumed.state["optimizer_global_step"] == 5
    assert [entry[0] for entry in resumed.trainer.train_inputs] == [3, 4]
    assert resumed.trainer.val_forwards == 4
    assert first.trainer.val_forwards == 6
    assert any(not torch.equal(first.network[key], resumed.network[key]) for key in first.network)


def test_resume_without_durable_events_measures_loaded_step(tmp_path):
    run_loop(tmp_path, target=3, save=True)
    checkpoint = checkpoint_at(tmp_path, 3)
    for path in tmp_path.rglob("events.out.tfevents.*"):
        path.unlink()
    resumed = run_loop(tmp_path, target=5, resume=checkpoint)
    events, _ = scalar_events(tmp_path)
    assert [event.step for event in events["val_loss_mean"]] == [3, 4, 5]
    assert [event.step for event in events["loss/current"]] == [4, 5]
    # Durable metadata may re-emit verified metrics; it must not invent a training point.
    assert resumed.trainer.val_forwards in (4, 6)


@pytest.mark.parametrize("target", [2, 3])
def test_resume_at_target_has_no_extra_update(tmp_path, target):
    first = run_loop(tmp_path, target=3, save=True)
    checkpoint = checkpoint_at(tmp_path, 3)
    resumed = run_loop(tmp_path, target=target, resume=checkpoint)
    assert resumed.trainer.train_inputs == []
    assert resumed.trainer.val_forwards == 0
    assert_nested_equal(first.network, resumed.network)
    assert_nested_equal(first.optimizer, resumed.optimizer)
    assert_nested_equal(first.scheduler, resumed.scheduler)
    assert_nested_equal(first.rng, resumed.rng)
    events, _ = scalar_events(tmp_path)
    assert [event.step for event in events["val_loss_mean"]] == [0, 2, 3]
    assert [event.step for event in events["loss/current"]] == [1, 2, 3]


def test_resume_rollback_purges_abandoned_future_events(tmp_path):
    run_loop(tmp_path, target=3, save=True)
    checkpoint = checkpoint_at(tmp_path, 3)
    run_loop(tmp_path, target=5, resume=checkpoint)
    rolled_back = run_loop(tmp_path, target=4, resume=checkpoint)
    events, runs = scalar_events(tmp_path)
    assert len(runs) == 1
    for tag in VAL_TAGS:
        assert [event.step for event in events[tag]] == [0, 2, 3, 4]
    assert [event.step for event in events["loss/current"]] == [1, 2, 3, 4]
    assert rolled_back.trainer.val_forwards == 2


@pytest.mark.parametrize("failure", ["verify", "load"])
def test_real_loop_validation_error_restores_modes_rng_and_optimizer(tmp_path, failure):
    with pytest.raises((ValueError, RuntimeError), match="fixture validation") as caught:
        run_loop(tmp_path, failure=failure)
    before, after = caught.value.loop_states
    assert_nested_equal(before, after)


def test_val_only_legacy_layout_saves_explicit_step_and_resumes(tmp_path):
    run_loop(tmp_path, target=3, save=True, experiment=False)
    checkpoint = checkpoint_at(tmp_path, 3)
    assert checkpoint.name == "tiny-state"
    resumed = run_loop(tmp_path, target=5, resume=checkpoint, experiment=False)
    events, runs = scalar_events(tmp_path)
    assert len(runs) == 1
    assert [event.step for event in events["val_loss_mean"]] == [0, 2, 3, 4, 5]
    assert [event.step for event in events["loss/current"]] == [1, 2, 3, 4, 5]
    assert resumed.state["optimizer_global_step"] == 5


def test_canonical_checkpoint_contains_one_fp32_adapter_and_no_base(tmp_path):
    from safetensors.torch import load_file

    result = run_loop(tmp_path, target=2, save=True)
    checkpoint = checkpoint_at(tmp_path, 2)
    assert checkpoint.name == "tiny-step2"
    tensors = list(checkpoint.glob("*.safetensors"))
    assert [path.name for path in tensors] == ["model.safetensors"]
    saved = load_file(str(tensors[0]))
    assert saved.keys() == result.network.keys()
    assert all(key.startswith("lora_unet_") for key in saved)
    assert all(tensor.dtype == torch.float32 for tensor in saved.values())
    for key, tensor in saved.items():
        torch.testing.assert_close(tensor, result.network[key].float(), rtol=0, atol=0)
    assert not list(checkpoint.glob("pytorch_model*"))
    assert (checkpoint / "optimizer.bin").is_file()
    assert (checkpoint / "scheduler.bin").is_file()
    assert (checkpoint / "random_states_0.pkl").is_file()
    assert len(result.saves) == 1


def test_periodic_final_checkpoint_is_written_once(tmp_path):
    result = run_loop(tmp_path, target=4, save=True, save_interval=2, periodic_state=True)
    assert [path.parent.name for path in result.saves] == ["tiny-step2", "tiny-step4"]
    assert all(path.name == "model.safetensors" for path in result.saves)
    assert checkpoint_at(tmp_path, 4).name == "tiny-step4"


def test_weights_only_and_samples_only_are_not_resumable_states(tmp_path):
    from PIL import Image

    result = run_loop(tmp_path, target=5, save=False, save_interval=3, sample_interval=2)
    output = tmp_path / "output"
    assert [path.parent.name for path in result.saves] == ["tiny-step3", "tiny-step5"]
    for step in (2, 4):
        samples = list((output / f"tiny-step{step}" / "samples").glob("*.png"))
        assert len(samples) == 1
        assert f"_{step:06d}_" in samples[0].name
        assert np.asarray(Image.open(samples[0])).max() == 127
        assert not (samples[0].parent.parent / "model.safetensors").exists()
    for step in (2, 3, 5):
        with pytest.raises(ValueError, match="state|checkpoint|metadata"):
            run_loop(tmp_path, target=6, resume=output / f"tiny-step{step}")
    assert not (output / "sample").exists()


def test_real_loop_retains_weights_beyond_shorter_state_window(tmp_path):
    run_loop(tmp_path, target=7, save=True, save_interval=2, periodic_state=True, weights_window=3, state_window=1)
    output = tmp_path / "output"
    weight_steps = sorted(int(path.parent.name.removeprefix("tiny-step")) for path in output.glob("tiny-step*/model.safetensors"))
    state_steps = sorted(int(path.parent.name.removeprefix("tiny-step")) for path in output.glob("tiny-step*/optimizer.bin"))
    assert weight_steps == [4, 6, 7]
    assert state_steps == [6, 7]
