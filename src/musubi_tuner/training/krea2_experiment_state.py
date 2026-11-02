"""Small filesystem/state helpers for the opt-in Krea2 experiment format."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import shutil

import torch


@dataclass
class ExperimentState:
    completed_optimizer_steps: int = 0
    epoch: int = 0
    next_microbatch: int = 0
    last_validation_step: int | None = None
    validation_metrics: dict = field(default_factory=dict)
    topology: dict = field(default_factory=dict)
    tensorboard_dir: str | None = None

    def validation_due(self, interval, *, initial=False, final=False):
        step = self.completed_optimizer_steps
        return self.last_validation_step != step and (initial or final or step % interval == 0)

    def save(self, directory):
        Path(directory, "experiment_state.json").write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        root = Path(directory)
        if not (root / "checkpoint_complete").is_file():
            raise ValueError(f"Incomplete experiment checkpoint: {root}; resume from a completed step directory")
        return cls(**json.loads((root / "experiment_state.json").read_text(encoding="utf-8")))


def checkpoint_name(output_name, step):
    return f"{output_name}-step{step:06d}"


def retention_boundary(step, window, interval):
    if window is None or interval is None:
        return None
    boundary = (step - window - 1) // interval * interval
    return boundary if boundary >= 0 else None


def remove_expired_checkpoints(args, step):
    window = args.save_last_n_steps
    boundary = retention_boundary(step, window, args.save_every_n_steps)
    if boundary is None:
        return
    pattern = re.compile(re.escape(args.output_name) + r"-step(\d+)$")
    for path in Path(args.output_dir).iterdir():
        match = pattern.fullmatch(path.name)
        if path.is_dir() and match and int(match[1]) <= boundary and (path / "checkpoint_complete").is_file():
            shutil.rmtree(path)


def save_checkpoint(args, accelerator, state, manifest, epoch_generator_state):
    """All ranks save native state; only complete directories become resumable."""
    root = Path(args.output_dir)
    final = root / checkpoint_name(args.output_name, state.completed_optimizer_steps)
    stage = root / (final.name + ".incomplete")
    if final.exists():
        raise ValueError(f"Checkpoint already exists: {final}; resume that experiment or choose a new output directory")
    if accelerator.is_main_process:
        root.mkdir(parents=True, exist_ok=True)
        if stage.exists():
            raise ValueError(f"Previous interrupted checkpoint exists: {stage}; preserve it and choose a clean output")
        stage.mkdir()
    accelerator.wait_for_everyone()
    accelerator.save_state(str(stage))
    torch.save(epoch_generator_state, stage / f"data_position_{accelerator.process_index}.pt")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        state.save(stage)
        (stage / "val_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        samples = root / "samples" / f"step{state.completed_optimizer_steps:06d}"
        if samples.exists():
            shutil.move(str(samples), str(stage / "samples"))
        (stage / "checkpoint_complete").write_text("complete\n", encoding="utf-8")
        stage.rename(final)
        remove_expired_checkpoints(args, state.completed_optimizer_steps)
    accelerator.wait_for_everyone()
    return final


def training_topology(args, accelerator, dataset):
    return {
        "world_size": accelerator.num_processes,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "dataset_batches": len(dataset),
        "batch_sizes": [d.batch_size for d in dataset.datasets],
        "num_repeats": [d.num_repeats for d in dataset.datasets],
        "seed": args.seed,
        "max_train_steps": args.max_train_steps,
    }


def choose_tensorboard_directory(args, state):
    """Each continuation is its own trajectory, even from an older state."""
    if not args.logging_dir:
        return
    root = Path(args.logging_dir)
    if args.resume:
        # A separate run also avoids mixing already written later events with
        # an older branch. The resumed validation point is not logged again.
        root = root / f"continuation-step{state.completed_optimizer_steps:06d}"
        root.mkdir(parents=True, exist_ok=True)
    args.logging_dir = str(root)


def restore_data_iterator(loader, cursor, generator, generator_state):
    from musubi_tuner.training.krea2_validation import preserve_rng

    with preserve_rng():
        if generator_state is not None:
            generator.set_state(generator_state)
        iterator = iter(loader)
        for _ in range(cursor):
            try:
                next(iterator)
            except StopIteration as exc:
                raise ValueError("Saved train cursor exceeds current dataloader length") from exc
    return iterator
