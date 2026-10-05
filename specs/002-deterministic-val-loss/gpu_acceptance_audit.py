"""Read actual GPU run artifacts; no training or model loading."""

import argparse
import json
import math
from pathlib import Path

from safetensors import safe_open
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from musubi_tuner.training.experiment_state import validate_experiment_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--weights", type=int, nargs="+", required=True)
    parser.add_argument("--states", type=int, nargs="+", required=True)
    parser.add_argument("--samples", type=int, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    runs = sorted({path.parent for path in (args.output / "tensorboard").rglob("events.out.tfevents.*")})
    assert len(runs) == 1, f"Expected one continued event run, found {runs}"
    reader = EventAccumulator(str(runs[0]), size_guidance={"scalars": 0}).Reload()
    tags = ("val_loss_mean", "val_loss_low_noise", "val_loss_high_noise")
    assert sorted(tag for tag in reader.Tags()["scalars"] if tag.startswith("val_")) == sorted(tags)
    values = {tag: [{"step": event.step, "value": event.value} for event in reader.Scalars(tag)] for tag in tags}
    for tag in tags:
        assert [event["step"] for event in values[tag]] == args.steps, (tag, values[tag])
        assert all(math.isfinite(event["value"]) for event in values[tag])
    for mean, low, high in zip(*(values[tag] for tag in tags)):
        assert math.isclose(mean["value"], (low["value"] + high["value"]) / 2, rel_tol=1e-4, abs_tol=1e-5)
    train_steps = [event.step for event in reader.Scalars("loss/current")]
    assert train_steps == list(range(1, max(args.steps) + 1)), train_steps
    inventory, weights, states, samples = {}, [], [], []
    for directory in sorted(args.output.glob(args.name + "-step*")):
        step = int(directory.name.removeprefix(args.name + "-step"))
        entry = {"files": sorted(path.name for path in directory.iterdir() if path.is_file())}
        tensors = list(directory.glob("*.safetensors"))
        if tensors:
            assert [path.name for path in tensors] == ["model.safetensors"]
            weights.append(step)
            with safe_open(str(tensors[0]), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                assert keys and all(key.startswith("lora_") for key in keys), keys[:5]
                assert all(handle.get_slice(key).get_dtype() == "F32" for key in keys)
                entry["adapter_tensor_count"] = len(keys)
            manifest = validate_experiment_checkpoint(directory, require_state=False)
            entry["manifest"] = manifest
            if manifest["with_state"]:
                validate_experiment_checkpoint(directory)
                states.append(step)
        pngs = sorted((directory / "samples").glob("*.png"))
        if pngs:
            samples.append(step)
            entry["samples"] = [str(path.relative_to(args.output)) for path in pngs]
        inventory[str(step)] = entry
    assert sorted(weights) == sorted(args.weights), weights
    assert sorted(states) == sorted(args.states), states
    assert sorted(samples) == sorted(args.samples), samples
    report = {"status": "passed", "run": str(runs[0]), "validation": values, "train_steps": train_steps, "inventory": inventory}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "passed", "val_steps": args.steps, "weights": weights, "states": states, "samples": samples}))


if __name__ == "__main__":
    main()
