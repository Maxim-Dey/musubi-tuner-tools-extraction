"""Inclusive optimizer-step retention windows, scoped to manifest-owned files."""

from argparse import Namespace
import hashlib
import json

import pytest

from musubi_tuner.training.experiment_state import prune_experiment_checkpoints


def checkpoint(root, name, step, state=True):
    directory = root / f"{name}-step{step}"
    (directory / "samples").mkdir(parents=True)
    (directory / "samples/image.png").write_bytes(b"sample")
    names = ["model.safetensors"] + (
        ["optimizer.bin", "scheduler.bin", "random_states_0.pkl", "trainer_state.json"] if state else []
    )
    inventory = {}
    for filename in names:
        contents = filename.encode()
        (directory / filename).write_bytes(contents)
        inventory[filename] = {"sha256": hashlib.sha256(contents).hexdigest(), "size": len(contents)}
    manifest = dict(
        schema=1,
        optimizer_global_step=step,
        output_name=name,
        with_state=state,
        process_count=1,
        adapter_identity="fixture",
        files=inventory,
    )
    (directory / "checkpoint_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


@pytest.mark.parametrize(
    "weight_window,state_window,weights,states",
    [
        (40, 20, list(range(60, 101, 10)), list(range(80, 101, 10))),
        (40, 60, list(range(40, 101, 10)), list(range(40, 101, 10))),
        (40, None, list(range(60, 101, 10)), list(range(60, 101, 10))),
        (40, 0, list(range(60, 101, 10)), list(range(60, 101, 10))),
        (None, None, list(range(10, 101, 10)), list(range(10, 101, 10))),
        (None, 20, list(range(10, 101, 10)), list(range(80, 101, 10))),
    ],
)
def test_retention_windows_and_output_name_isolation(tmp_path, weight_window, state_window, weights, states):
    for step in range(10, 101, 10):
        checkpoint(tmp_path, "qwen", step)
    other = checkpoint(tmp_path, "qwen-other", 10)
    notes = tmp_path / "qwen-step10/notes.txt"
    notes.write_text("user notes")
    args = Namespace(
        output_dir=str(tmp_path), output_name="qwen", save_last_n_steps=weight_window, save_last_n_steps_state=state_window
    )
    prune_experiment_checkpoints(args, 100)
    present_weights = [step for step in range(10, 101, 10) if (tmp_path / f"qwen-step{step}/model.safetensors").exists()]
    present_states = [step for step in range(10, 101, 10) if (tmp_path / f"qwen-step{step}/optimizer.bin").exists()]
    assert present_weights == weights and present_states == states
    assert (other / "optimizer.bin").exists() and notes.read_text() == "user notes"
    assert all((tmp_path / f"qwen-step{step}/samples/image.png").exists() for step in range(10, 101, 10))


def test_missing_steps_and_final_between_periods_are_not_invented(tmp_path):
    for step in (10, 20, 35):
        checkpoint(tmp_path, "a.b", step)
    untouched = checkpoint(tmp_path, "axb", 1)
    args = Namespace(output_dir=str(tmp_path), output_name="a.b", save_last_n_steps=15, save_last_n_steps_state=5)
    prune_experiment_checkpoints(args, 35)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.b-step10", "a.b-step20", "a.b-step35", "axb-step1"]
    assert not (tmp_path / "a.b-step10/model.safetensors").exists()
    assert (tmp_path / "a.b-step20/model.safetensors").exists()
    assert not (tmp_path / "a.b-step20/optimizer.bin").exists()
    assert (tmp_path / "a.b-step35/optimizer.bin").exists() and (untouched / "optimizer.bin").exists()


def test_manifest_escape_cannot_delete_user_file(tmp_path):
    directory = checkpoint(tmp_path, "qwen", 1)
    protected = tmp_path / "protected.bin"
    protected.write_bytes(b"user data")
    manifest_path = directory / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["../protected.bin"] = {"size": 9, "sha256": hashlib.sha256(b"user data").hexdigest()}
    manifest_path.write_text(json.dumps(manifest))
    args = Namespace(output_dir=str(tmp_path), output_name="qwen", save_last_n_steps=1, save_last_n_steps_state=1)
    with pytest.raises(ValueError, match="unsafe|file name"):
        prune_experiment_checkpoints(args, 100)
    assert protected.read_bytes() == b"user data"
