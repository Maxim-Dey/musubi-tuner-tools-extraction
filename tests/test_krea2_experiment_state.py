from pathlib import Path
from types import SimpleNamespace
from multiprocessing import Value
import random

import numpy as np
import pytest
from accelerate import Accelerator
from safetensors.torch import load_file, save_file
import torch

from musubi_tuner.training.krea2_experiment_state import (
    ExperimentState, checkpoint_name, retention_boundary, remove_expired_checkpoints, save_checkpoint, choose_tensorboard_directory,
)
from musubi_tuner.training.parser_common import setup_parser_common
from musubi_tuner.training.trainer_base import NetworkTrainer
import musubi_tuner.training.trainer_base as trainer_module


@pytest.mark.parametrize("step,window,interval,expected", [(1000,1000,200,None),(1200,1000,200,0),
    (1400,1000,200,200),(1250,1000,200,200),(5,None,2,None),(5,0,2,4)])
def test_retention_boundary(step, window, interval, expected):
    assert retention_boundary(step, window, interval) == expected


def test_single_retention_window_removes_whole_bundles_and_preserves_unrelated_files(tmp_path):
    for name in ("run-step000002", "run-step000004", "other-step000002", "run-step000003.incomplete"):
        path = tmp_path / name
        (path / "samples").mkdir(parents=True)
        (path / "samples" / "x.png").write_bytes(b"x")
        for filename in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
            (path / filename).write_bytes(b"state")
        (path / "checkpoint_complete").touch()
    args = SimpleNamespace(output_dir=str(tmp_path), output_name="run", save_last_n_steps=2, save_every_n_steps=2)
    remove_expired_checkpoints(args, 6)
    assert not (tmp_path / "run-step000002").exists()
    assert (tmp_path / "run-step000004").exists()
    assert (tmp_path / "other-step000002").exists()
    assert (tmp_path / "run-step000003.incomplete").exists()


def test_retention_main_window_and_nonperiodic_final_step(tmp_path):
    args = SimpleNamespace(output_dir=str(tmp_path), output_name="run", save_last_n_steps=1000, save_every_n_steps=200)
    for step in (200, 400, 600, 800, 1000):
        folder = tmp_path / checkpoint_name("run", step)
        folder.mkdir()
        (folder / "checkpoint_complete").touch()
    remove_expired_checkpoints(args, 1000)
    assert len(list(tmp_path.iterdir())) == 5
    final = tmp_path / checkpoint_name("run", 1250)
    final.mkdir()
    (final / "checkpoint_complete").touch()
    remove_expired_checkpoints(args, 1250)
    assert {path.name for path in tmp_path.iterdir()} == {
        checkpoint_name("run", step) for step in (400, 600, 800, 1000, 1250)
    }


def test_absent_retention_window_keeps_all_bundles(tmp_path):
    folder = tmp_path / checkpoint_name("run", 2)
    folder.mkdir()
    (folder / "checkpoint_complete").touch()
    args = SimpleNamespace(output_dir=str(tmp_path), output_name="run", save_last_n_steps=None, save_every_n_steps=2)
    remove_expired_checkpoints(args, 10000)
    assert folder.is_dir()


def test_legacy_step_state_uses_same_retention_key_without_removed_attribute(tmp_path):
    from musubi_tuner.utils import train_utils

    args = SimpleNamespace(output_dir=str(tmp_path), output_name="run", save_last_n_steps=2,
                           save_every_n_steps=2, save_state_to_huggingface=False)
    for step in (2, 4):
        folder = tmp_path / train_utils.STEP_STATE_NAME.format("run", step)
        folder.mkdir()
        (folder / "optimizer.bin").write_bytes(b"optimizer")
    def save_state(directory):
        folder = Path(directory)
        folder.mkdir()
        (folder / "optimizer.bin").write_bytes(b"optimizer")
    train_utils.save_and_remove_state_stepwise(args, SimpleNamespace(save_state=save_state), 6)
    assert not (tmp_path / train_utils.STEP_STATE_NAME.format("run", 2)).exists()
    for step in (4, 6):
        assert (tmp_path / train_utils.STEP_STATE_NAME.format("run", step) / "optimizer.bin").is_file()


def test_schedule_restore_skips_completed_validation_and_incomplete_state(tmp_path):
    state = ExperimentState()
    observed = []
    for step in range(6):
        state.completed_optimizer_steps = step
        if state.validation_due(2, initial=step==0, final=step==5):
            observed.append(step)
            state.last_validation_step = step
    assert observed == [0, 2, 4, 5]
    state.save(tmp_path)
    with pytest.raises(ValueError, match="Incomplete"):
        ExperimentState.load(tmp_path)
    (tmp_path / "checkpoint_complete").touch()
    restored = ExperimentState.load(tmp_path)
    assert not restored.validation_due(2, initial=True, final=True)


class TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.5))

    def save_weights(self, path, dtype, metadata):
        save_file({k: v.detach().cpu().to(dtype) for k, v in self.state_dict().items()}, str(path))

    def load_weights(self, path):
        self.load_state_dict(load_file(str(path)))

    def on_epoch_start(self, transformer):
        self.train()

    def on_step_start(self):
        pass

    def get_trainable_params(self):
        return self.parameters()


class ToyDataset(torch.utils.data.Dataset):
    batch_size = num_repeats = 1
    num_train_items = 7

    def __init__(self, epoch, size=7):
        self.shared_epoch = epoch
        self.current_epoch = 0
        self.order = list(range(1, size + 1))
        self.num_train_items = size

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        while self.current_epoch < self.shared_epoch.value:
            self.current_epoch += 1
            random.seed(42 + self.current_epoch)
            random.shuffle(self.order)
        return {"latents": torch.tensor([[[[float(self.order[index])]]]])}

    def get_metadata(self):
        return {}


class ToyTrainer(NetworkTrainer):
    architecture = "krea2"
    architecture_full_name = "krea2"

    def __init__(self, network):
        super().__init__()
        self.network = network
        self.seen = []

    def process_batch(self, args, accelerator, transformer, network, batch, latents, noise, *unused):
        self.seen.append(latents.item())
        return ((network.weight * latents - noise) ** 2).mean(), {}

    def scale_shift_latents(self, latents):
        return latents

    def call_dit(self, args, accelerator, transformer, latents, batch, noise, noisy, timesteps, dtype):
        return SimpleNamespace(pred=self.network.weight * noisy, target=noise-latents)

    def sample_images(self, accelerator, args, epoch, steps, *unused):
        path = Path(self._sample_save_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "sample.txt").write_text(str(steps))


class ToyInputs:
    manifest = {"test": "fixed"}
    records = [SimpleNamespace(image_sha256="a")]

    def validate_unchanged(self):
        pass

    def load_batch(self, record):
        return {"latents": torch.ones(1,1,1,1,1), "krea2_vl_embed": []}


def run_loop(root, *, resume=None, validation=True, stop_after=None, monkeypatch=None, workers=0, sample_interval=None,
             skip_updates=0, dataset_size=7, tracking=False, event_sink=None, epoch_hook_sink=None, sample_sink=None):
    args = setup_parser_common().parse_args(["--dataset_config", "unused", "--dit", "unused"])
    args.output_dir = str(root)
    args.output_name = "run"
    args.max_train_steps = 5
    args.gradient_accumulation_steps = 2
    args.save_every_n_steps = 2
    args.save_state = True
    assert args.save_state_on_train_end is False  # Unified final step saves state without this legacy flag.
    args.save_last_n_steps = None
    args.seed = 42
    args.full_fp16 = args.full_bf16 = False
    args.fp8_scaled = False
    args.resume = str(resume) if resume else None
    args.val_dataset_config = "unused" if validation else None
    args.val_every_n_steps = 2
    args.val_level_noise_n = 2
    args.val_seed_noise_n = 1
    args.val_seed_noise = 42
    args.weighting_scheme = "none"
    args.sample_every_n_steps = sample_interval
    args.sample_at_first = sample_interval is not None
    args.network_module = "networks.lora_krea2"
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    acc = Accelerator(cpu=True, gradient_accumulation_steps=2, log_with="tensorboard" if tracking else None,
                      project_dir=str(root/"tensorboard") if tracking else None)
    network, transformer = TinyNetwork(), TinyNetwork()
    trainer = ToyTrainer(network)
    if epoch_hook_sink is not None:
        trainer.on_epoch_end = lambda args, accelerator, network, transformer, epoch: epoch_hook_sink.append(epoch)
    if sample_sink is not None:
        original_sample = trainer.sample_images
        def record_sample(*values):
            sample_sink.append(values[3])
            return original_sample(*values)
        trainer.sample_images = record_sample
    trainer._validation_inputs = ToyInputs()
    trainer._unified_checkpoint = True
    trainer._experiment_state = ExperimentState.load(resume) if resume else ExperimentState()
    trainer._data_generator = torch.Generator().manual_seed(42)
    epoch = Value("i", 0)
    dataset = ToyDataset(epoch, dataset_size)
    loader = torch.utils.data.DataLoader(dataset, shuffle=True, generator=trainer._data_generator, num_workers=workers)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.7)
    network, optimizer, loader, scheduler = acc.prepare(network, optimizer, loader, scheduler)
    if skip_updates:
        original_step = optimizer.step
        skipped = 0
        def maybe_skip(*values, **kw):
            nonlocal skipped
            if acc.sync_gradients and skipped < skip_updates:
                skipped += 1
                optimizer._is_overflow = True
                return
            optimizer._is_overflow = False
            return original_step(*values, **kw)
        optimizer.step = maybe_skip
    trainer._register_hooks_and_resume(args, acc, network)
    class Group:
        num_train_items = len(dataset)
        datasets = [dataset]
        def __len__(self):
            return len(dataset)
    events = []
    native_log = acc.log
    def record_log(values, step=None):
        events.append((step, values.copy()))
        native_log(values, step=step)
    acc.log = record_log
    if stop_after is not None:
        original_save = trainer_module.save_checkpoint
        def stop(*values, **kw):
            result = original_save(*values, **kw)
            if values[2].completed_optimizer_steps == stop_after:
                raise InterruptedError("controlled stop")
            return result
        monkeypatch.setattr(trainer_module, "save_checkpoint", stop)
    try:
        trainer._run_training_loop(
            args, acc, 1, 0, Group(), loader, epoch, transformer, network, network,
            optimizer, "AdamW", [], lambda: None, lambda: None, scheduler, ["unet"], None, None,
            torch.float32, torch.float32,
        )
    except InterruptedError:
        assert stop_after is not None
        acc.end_training()
    finally:
        if stop_after is not None:
            monkeypatch.setattr(trainer_module, "save_checkpoint", original_save)
    result = (network.weight.detach().clone(), trainer.seen, [(s,v) for s,v in events if "val_loss_mean" in v],
              optimizer.state_dict(), scheduler.state_dict(), torch.get_rng_state().clone())
    if event_sink is not None:
        event_sink.extend(events)
    acc.free_memory()
    return result


@pytest.mark.parametrize("workers", [0, 2])
def test_actual_loop_accumulation_pause_resume_and_validation_no_delta(tmp_path, monkeypatch, workers):
    full = run_loop(tmp_path/"full", workers=workers)
    without_val = run_loop(tmp_path/"no-val", validation=False, workers=workers)
    assert torch.equal(full[0], without_val[0])
    assert torch.equal(full[-1], without_val[-1])
    assert full[1] == without_val[1]
    stopped = run_loop(tmp_path/"split", stop_after=2, monkeypatch=monkeypatch, workers=workers)
    checkpoint = tmp_path/"split"/checkpoint_name("run", 2)
    resumed = run_loop(tmp_path/"split", resume=checkpoint, workers=workers)
    assert stopped[1] + resumed[1] == full[1]
    assert torch.equal(resumed[0], full[0])
    assert torch.equal(resumed[-1], full[-1])
    assert resumed[3]["param_groups"] == full[3]["param_groups"]
    for key, values in full[3]["state"].items():
        for name, value in values.items():
            assert torch.equal(resumed[3]["state"][key][name], value)
    assert resumed[4] == full[4]
    assert [s for s,v in full[2]] == [0,2,4,5]
    assert [s for s,v in stopped[2]+resumed[2]] == [0,2,4,5]
    for (_, original), (_, restored) in zip(full[2], stopped[2]+resumed[2]):
        assert original == restored
    for step in (2,4,5):
        path = tmp_path/"split"/checkpoint_name("run", step)
        assert (path/"checkpoint_complete").is_file()
        tensors = load_file(str(path/"model.safetensors"))
        assert all(value.dtype == torch.float32 for value in tensors.values())
        assert list(path.glob("*.safetensors")) == [path/"model.safetensors"]
        assert not list(path.glob("pytorch_model*"))
        assert (path/"optimizer.bin").is_file() and (path/"scheduler.bin").is_file()
        reloaded = TinyNetwork()
        reloaded.load_weights(path/"model.safetensors")
    assert not list((tmp_path/"split").glob("*.incomplete"))
    assert {path.name for path in (tmp_path/"split").glob("run-step*")} == {
        checkpoint_name("run", step) for step in (2, 4, 5)
    }
    assert not list((tmp_path/"split").glob("*.safetensors"))


@pytest.mark.parametrize("interval", [2,3])
def test_samples_belong_to_exact_weight_step(tmp_path, interval):
    run_loop(tmp_path, sample_interval=interval)
    assert (tmp_path/"samples"/"step000000"/"sample.txt").read_text() == "0"
    if interval == 2:
        for step in (2,4):
            assert (tmp_path/checkpoint_name("run",step)/"samples"/"sample.txt").read_text() == str(step)
        assert not (tmp_path/"samples"/"step000004").exists()
    else:
        assert (tmp_path/"samples"/"step000003"/"sample.txt").read_text() == "3"
        assert not (tmp_path/checkpoint_name("run",4)/"samples").exists()


def test_skipped_updates_do_not_advance_validation_or_stop_early(tmp_path):
    result = run_loop(tmp_path, skip_updates=4)
    assert [s for s,v in result[2]] == [0,2,4,5]
    assert len(result[1]) > 14  # four failed updates require a third epoch
    assert ExperimentState.load(tmp_path/checkpoint_name("run",5)).completed_optimizer_steps == 5


def test_interrupted_write_never_publishes_ready_checkpoint(tmp_path):
    args = SimpleNamespace(output_dir=str(tmp_path), output_name="run")
    def fail(directory):
        Path(directory,"optimizer.bin").write_bytes(b"partial")
        raise OSError("injected write failure")
    acc = SimpleNamespace(is_main_process=True, process_index=0, wait_for_everyone=lambda:None, save_state=fail)
    with pytest.raises(OSError, match="injected"):
        save_checkpoint(args,acc,ExperimentState(completed_optimizer_steps=2),{},None)
    assert not (tmp_path/checkpoint_name("run",2)).exists()
    incomplete = tmp_path/(checkpoint_name("run",2)+".incomplete")
    assert (incomplete/"optimizer.bin").exists() and not (incomplete/"checkpoint_complete").exists()


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_new_checkpoint_rejects_explicit_lower_save_precision_before_models(monkeypatch, precision):
    import musubi_tuner.training.krea2_validation_inputs as inputs_module
    trainer = NetworkTrainer()
    args = setup_parser_common().parse_args(["--dataset_config","unused","--dit","unused"])
    args.experiment_mode = args.save_state = True
    args.save_precision = precision
    trainer.validate_training_args = lambda args: None
    monkeypatch.setattr(inputs_module,"preflight_validation_inputs",lambda args:ToyInputs())
    with pytest.raises(ValueError,match="save_precision"):
        trainer._validate_args_and_init(args)


def test_older_checkpoint_continuation_uses_separate_tensorboard_directory(tmp_path):
    old_event = tmp_path/"events.old"
    old_event.write_bytes(b"later incompatible events")
    args = SimpleNamespace(logging_dir=str(tmp_path),resume="step2")
    choose_tensorboard_directory(args, ExperimentState(completed_optimizer_steps=2))
    assert Path(args.logging_dir) == tmp_path/"continuation-step000002"
    assert old_event.read_bytes() == b"later incompatible events"


def tensorboard_scalars(root, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    values = []
    for path in Path(root).rglob("events.out.tfevents.*"):
        events = EventAccumulator(str(path), size_guidance={"scalars": 0})
        events.Reload()
        if tag in events.Tags().get("scalars", []):
            values.extend(events.Scalars(tag))
    return sorted(values, key=lambda point: point.step)


@pytest.mark.parametrize("dataset_size", [2, 7])
def test_tensorboard_resume_boundary_and_partial_epoch_preserve_state_and_saved_samples(tmp_path, monkeypatch, dataset_size):
    import hashlib

    full = run_loop(tmp_path/"full", dataset_size=dataset_size, tracking=True, sample_interval=2)
    split = tmp_path/"split"
    stopped = run_loop(split, dataset_size=dataset_size, tracking=True, sample_interval=2,
                       stop_after=2, monkeypatch=monkeypatch)
    checkpoint = split/checkpoint_name("run", 2)
    state = ExperimentState.load(checkpoint)
    assert state.next_microbatch == (2 if dataset_size == 2 else 4)
    sample = checkpoint/"samples"/"sample.txt"
    before = (hashlib.sha256(sample.read_bytes()).hexdigest(), sample.stat().st_mtime_ns)
    resumed_events, resumed_hooks, resumed_samples = [], [], []
    resumed = run_loop(split, resume=checkpoint, dataset_size=dataset_size, tracking=True, sample_interval=2,
                       event_sink=resumed_events, epoch_hook_sink=resumed_hooks, sample_sink=resumed_samples)
    assert torch.equal(resumed[0], full[0]) and torch.equal(resumed[-1], full[-1])
    assert stopped[1] + resumed[1] == full[1]
    assert resumed[3]["param_groups"] == full[3]["param_groups"]
    for key, values in full[3]["state"].items():
        for name, value in values.items():
            assert torch.equal(resumed[3]["state"][key][name], value)
    assert resumed[4] == full[4]
    assert resumed_hooks[0] == state.epoch + 1  # consumed epoch hook still replays
    assert not any(step == 2 and "loss/epoch" in values for step, values in resumed_events)
    assert any("loss/epoch" in values for step, values in resumed_events)
    assert 2 not in resumed_samples
    assert before == (hashlib.sha256(sample.read_bytes()).hexdigest(), sample.stat().st_mtime_ns)
    assert not (split/"samples"/"step000002").exists()
    for tag in ("val_loss_mean", "val_loss_low_noise", "val_loss_high_noise"):
        points = tensorboard_scalars(split/"tensorboard", tag)
        assert [point.step for point in points] == [0, 2, 4, 5]
        assert all(np.isfinite(point.value) for point in points)
    epoch_points = tensorboard_scalars(split/"tensorboard", "loss/epoch")
    assert epoch_points and all(np.isfinite(point.value) for point in epoch_points)


def test_tensorboard_epoch_with_all_skipped_updates_still_logs_real_losses(tmp_path):
    events = []
    run_loop(tmp_path, dataset_size=2, tracking=True, skip_updates=2, event_sink=events)
    initial_epoch_losses = [values["loss/epoch"] for step, values in events if step == 0 and "loss/epoch" in values]
    assert len(initial_epoch_losses) == 2 and all(np.isfinite(value) for value in initial_epoch_losses)
    assert [point.step for point in tensorboard_scalars(tmp_path/"tensorboard", "val_loss_mean")] == [0, 2, 4, 5]


def test_resume_without_saved_sample_directory_does_not_mark_sample_complete(tmp_path, monkeypatch):
    import shutil

    run_loop(tmp_path, dataset_size=2, tracking=True, sample_interval=2, stop_after=2, monkeypatch=monkeypatch)
    checkpoint = tmp_path/checkpoint_name("run", 2)
    shutil.rmtree(checkpoint/"samples")  # isolated fixture: sample was never present in the restored state
    resumed_samples = []
    run_loop(tmp_path, resume=checkpoint, dataset_size=2, tracking=True, sample_interval=2, sample_sink=resumed_samples)
    assert resumed_samples.count(2) == 1
    assert (tmp_path/"samples"/"step000002"/"sample.txt").read_text() == "2"
