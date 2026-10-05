"""Durable TensorBoard/checkpoint identity, using actual event writers/readers."""

from argparse import Namespace
import copy
import hashlib
import json
from pathlib import Path
import shutil

from accelerate import Accelerator
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
from torch.utils.tensorboard import SummaryWriter

from musubi_tuner.training.experiment_state import (
    adapter_identity,
    flush_trackers,
    load_training_state,
    pending_validation_events,
    prepare_training_state,
    record_validation,
    resolved_project_dir,
    saved_validation_metrics,
    save_training_state,
    tracker_init,
    validation_events_complete,
)


METRICS = {"val_loss_mean": 0.3, "val_loss_low_noise": 0.2, "val_loss_high_noise": 0.4}


def arguments(tmp_path, **kwargs):
    values = dict(
        experiment_mode=True,
        val_dataset_config="val.toml",
        resume=None,
        resume_from_huggingface=False,
        logging_dir=str(tmp_path / "output/tensorboard"),
        log_prefix="fixture",
        log_tracker_name=None,
        log_with="tensorboard",
        _experiment_root=str(tmp_path),
    )
    values.update(kwargs)
    return Namespace(**values)


def writer_for(state, **kwargs):
    return SummaryWriter(str(Path(resolved_project_dir(state)) / state["tensorboard_run_name"]), **kwargs)


def write_metrics(writer, values, step):
    for tag, value in values.items():
        writer.add_scalar(tag, value, step)
    writer.flush()


def events(state):
    return EventAccumulator(
        str(Path(resolved_project_dir(state)) / state["tensorboard_run_name"]), size_guidance={"scalars": 0}
    ).Reload()


def save_complete_state(directory, state):
    """Real native CPU state lets metadata tests keep the production completeness gate."""
    accelerator = Accelerator(cpu=True)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    accelerator.save_state(str(directory))
    save_training_state(directory, state)
    files = {
        path.name: {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in Path(directory).iterdir()
        if path.is_file()
    }
    manifest = dict(
        schema=1,
        optimizer_global_step=state["optimizer_global_step"],
        output_name="fixture",
        with_state=True,
        process_count=1,
        adapter_identity=adapter_identity(model),
        files=files,
    )
    (Path(directory) / "checkpoint_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    accelerator.free_memory()


def test_legacy_and_experiment_without_validation(tmp_path):
    args = arguments(tmp_path, experiment_mode=False, val_dataset_config=None)
    assert prepare_training_state(args) is None
    args.experiment_mode = True
    state = prepare_training_state(args)
    assert state["optimizer_global_step"] == 0 and state["validation_fingerprint"] is None
    state["optimizer_global_step"] = 7
    save_complete_state(tmp_path / "state", state)
    args.resume = str(tmp_path / "state")
    assert prepare_training_state(args)["optimizer_global_step"] == 7


def test_resume_metadata_required_and_fingerprint_checked(tmp_path):
    args = arguments(tmp_path, resume=str(tmp_path / "legacy"))
    with pytest.raises(ValueError, match="optimizer_global_step|metadata|manifest"):
        prepare_training_state(args, "inputs")
    args.resume = None
    state = prepare_training_state(args, "inputs")
    save_complete_state(tmp_path / "state", state)
    args.resume = str(tmp_path / "state")
    with pytest.raises(ValueError, match="validation.*changed|fingerprint"):
        prepare_training_state(args, "changed")
    path = tmp_path / "state/trainer_state.json"
    path.write_text('{"schema":1,"optimizer_global_step":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="optimizer_global_step"):
        load_training_state(tmp_path / "state")


def test_moved_experiment_keeps_relative_run_and_original_name(tmp_path):
    original = tmp_path / "original"
    state = prepare_training_state(arguments(original, log_tracker_name="saved-run"), "inputs")
    assert not Path(state["tensorboard_project_dir"]).is_absolute()
    state["optimizer_global_step"] = 3
    save_complete_state(original / "output/state", state)
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    args = arguments(moved, resume=str(moved / "output/state"), log_tracker_name="changed-run")
    resumed = prepare_training_state(args, "inputs")
    assert Path(resolved_project_dir(resumed)).is_relative_to(moved)
    name, kwargs = tracker_init(args, resumed, {"tensorboard": {"flush_secs": 2}, "wandb": {"name": "other"}})
    assert name == "saved-run"
    assert kwargs == {"tensorboard": {"flush_secs": 2, "purge_step": 4}, "wandb": {"name": "other"}}


def test_val_only_legacy_layout_uses_absolute_saved_run(tmp_path):
    state = prepare_training_state(arguments(tmp_path, experiment_mode=False), "inputs")
    assert Path(state["tensorboard_project_dir"]).is_absolute()
    save_training_state(tmp_path / "old-layout-state", state)
    assert (tmp_path / "old-layout-state/trainer_state.json").is_file()


@pytest.mark.parametrize("change", ["weighting", "dit", "merged", "multiplier", "precision", "attention"])
def test_changed_validation_computation_rejects_resume(tmp_path, change):
    dit = tmp_path / "dit.safetensors"
    merged = tmp_path / "base.safetensors"
    dit.write_bytes(b"base model bytes")
    merged.write_bytes(b"merged adapter bytes")
    args = arguments(
        tmp_path,
        dit=str(dit),
        base_weights=[str(merged)],
        base_weights_multiplier=[0.75],
        weighting_scheme="none",
        mixed_precision="bf16",
        sdpa=True,
    )
    state = prepare_training_state(args, "inputs")
    save_complete_state(tmp_path / "state", state)
    args.resume = str(tmp_path / "state")
    if change == "weighting":
        args.weighting_scheme = "sigma_sqrt"
    elif change == "dit":
        dit.write_bytes(b"different base model bytes")
    elif change == "merged":
        merged.write_bytes(b"different merged adapter bytes")
    elif change == "multiplier":
        args.base_weights_multiplier = [0.5]
    elif change == "precision":
        args.mixed_precision = "fp16"
    else:
        args.sdpa = False
    with pytest.raises(ValueError, match="computation contract changed"):
        prepare_training_state(args, "inputs")


def test_computation_identity_ignores_model_file_locations(tmp_path):
    dit = tmp_path / "dit.safetensors"
    merged = tmp_path / "base.safetensors"
    dit.write_bytes(b"base model bytes")
    merged.write_bytes(b"merged adapter bytes")
    args = arguments(tmp_path, dit=str(dit), base_weights=[str(merged)], weighting_scheme="none")
    state = prepare_training_state(args, "inputs")
    save_complete_state(tmp_path / "state", state)
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy(dit, moved / "renamed-dit.safetensors")
    shutil.copy(merged, moved / "renamed-base.safetensors")
    args.dit, args.base_weights = str(moved / "renamed-dit.safetensors"), [str(moved / "renamed-base.safetensors")]
    args.resume = str(tmp_path / "state")
    resumed = prepare_training_state(args, "inputs")
    assert resumed["validation_computation_contract"] == state["validation_computation_contract"]


def test_adapter_identity_is_order_independent_and_detects_weights_without_rng_changes():
    model = torch.nn.Linear(2, 2)
    before = torch.get_rng_state().clone()
    first = adapter_identity(model)
    clone = copy.deepcopy(model)
    assert adapter_identity(clone) == first
    with torch.no_grad():
        clone.weight[0, 0] += 1
    assert adapter_identity(clone) != first
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


def test_actual_accelerate_tracker_complete_missing_and_partial_tags(tmp_path):
    args = arguments(tmp_path)
    state = prepare_training_state(args, "inputs")
    accelerator = Accelerator(cpu=True, log_with="tensorboard", project_dir=resolved_project_dir(state))
    name, kwargs = tracker_init(args, state, {})
    accelerator.init_trackers(name, init_kwargs=kwargs)
    accelerator.log(METRICS, step=3)
    flush_trackers(accelerator)
    record_validation(state, 3, METRICS, "weights")
    assert validation_events_complete(state, 3, "weights")
    assert not validation_events_complete(state, 3, "changed")
    assert pending_validation_events(state, 3, METRICS, "weights") == {}
    assert set(events(state).Tags()["scalars"]) == set(METRICS)
    accelerator.end_training()
    accelerator.free_memory()


def test_metadata_alone_is_not_a_durable_measurement(tmp_path):
    state = prepare_training_state(arguments(tmp_path), "inputs")
    record_validation(state, 3, METRICS, "weights")
    assert not validation_events_complete(state, 3, "weights")
    assert pending_validation_events(state, 3, METRICS, "weights") == METRICS


def test_partial_tags_reemit_only_missing_and_ledger_recovers_stale_checkpoint_metadata(tmp_path):
    state = prepare_training_state(arguments(tmp_path), "inputs")
    with writer_for(state) as writer:
        write_metrics(writer, {"val_loss_mean": METRICS["val_loss_mean"]}, 3)
    record_validation(state, 3, METRICS, "weights")
    state["last_validation"] = None  # checkpoint saved before durable evaluation metadata
    assert not validation_events_complete(state, 3, "weights")
    saved = saved_validation_metrics(state, 3, "weights")
    assert saved == METRICS
    assert saved_validation_metrics(state, 3, "other-weights") is None
    assert saved_validation_metrics(state, 2, "weights") is None
    # No second floating-point forward is needed: durable values repair the partial triplet.
    missing = pending_validation_events(state, 3, saved, "weights")
    assert set(missing) == {"val_loss_low_noise", "val_loss_high_noise"}
    with writer_for(state) as writer:
        write_metrics(writer, missing, 3)
    assert validation_events_complete(state, 3, "weights")
    assert all(len(events(state).Scalars(tag)) == 1 for tag in METRICS)


def test_unidentified_or_conflicting_existing_events_are_never_silently_reused(tmp_path):
    state = prepare_training_state(arguments(tmp_path), "inputs")
    with writer_for(state) as writer:
        write_metrics(writer, METRICS, 3)
    with pytest.raises(ValueError, match="identity"):
        pending_validation_events(state, 3, METRICS, "weights")
    record_validation(state, 3, METRICS, "weights")
    with pytest.raises(ValueError, match="identity"):
        pending_validation_events(state, 3, METRICS, "different-weights")
    with pytest.raises(ValueError, match="metric|conflict"):
        pending_validation_events(state, 3, {**METRICS, "val_loss_mean": 9.0}, "weights")


def test_rollback_purges_later_trajectory_and_preserves_checkpoint_point(tmp_path):
    args = arguments(tmp_path)
    state = prepare_training_state(args, "inputs")
    state["optimizer_global_step"] = 3
    with writer_for(state) as writer:
        for step in (0, 2, 3):
            write_metrics(writer, METRICS, step)
            record_validation(state, step, METRICS, "weights")
    save_complete_state(tmp_path / "state", state)
    with writer_for(state) as writer:
        write_metrics(writer, {**METRICS, "val_loss_mean": 9.0}, 4)
    record_validation(state, 4, {**METRICS, "val_loss_mean": 9.0}, "discarded")
    resumed = prepare_training_state(arguments(tmp_path, resume=str(tmp_path / "state")), "inputs")
    _, kwargs = tracker_init(args, resumed, {})
    with writer_for(resumed, **kwargs["tensorboard"]) as writer:
        write_metrics(writer, METRICS, 4)
        record_validation(resumed, 4, METRICS, "new-weights")
    accumulator = events(resumed)
    assert [event.step for event in accumulator.Scalars("val_loss_mean")] == [0, 2, 3, 4]
    assert accumulator.Scalars("val_loss_mean")[-1].value == pytest.approx(0.3)
    assert validation_events_complete(resumed, 3, "weights")
