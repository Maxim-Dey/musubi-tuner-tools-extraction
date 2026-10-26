"""Two real Accelerate CPU ranks save one adapter and restore independent RNG states."""

from argparse import Namespace
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import random

from accelerate import Accelerator
import numpy as np
from safetensors.torch import load_file
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

from musubi_tuner.networks import lora_qwen_image
from musubi_tuner.qwen_image.qwen_image_model import QwenImageTransformerBlock
from musubi_tuner.training.experiment_state import (
    prepare_training_state,
    register_experiment_hooks,
    save_experiment_checkpoint,
    validate_experiment_checkpoint,
)


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def checkpoint_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29511",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        accelerator = Accelerator(cpu=True, mixed_precision="no")
        assert accelerator.num_processes == 2 and accelerator.process_index == rank
        torch.manual_seed(318)
        base = torch.nn.Sequential(QwenImageTransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4))
        base.requires_grad_(False)
        network = lora_qwen_image.create_arch_network(1, 2, 2, None, [], base)
        network.apply_to([], base, apply_text_encoder=False, apply_unet=True)
        optimizer = torch.optim.AdamW(network.parameters(), lr=0.003)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        # Accelerate 1.6 MULTI_CPU wraps even frozen models in DDP (unlike MULTI_GPU).
        # Register the frozen base natively without DDP; its weights still reach the save hooks.
        base = accelerator.prepare_model(base, evaluation_mode=True)
        network, optimizer, scheduler = accelerator.prepare(network, optimizer, scheduler)
        for parameter in network.parameters():
            parameter.grad = torch.ones_like(parameter) * 0.125
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        root = Path(output)
        args = Namespace(
            experiment_mode=True,
            val_dataset_config=None,
            resume=None,
            resume_from_huggingface=False,
            _experiment_root=str(root),
            output_dir=str(root / "output"),
            output_name="distributed",
            logging_dir=None,
            log_prefix=None,
            log_tracker_name=None,
            save_last_n_steps=None,
            save_last_n_steps_state=None,
        )
        state = prepare_training_state(args)
        state["optimizer_global_step"] = 3
        register_experiment_hooks(args, accelerator, network, state)
        adapter = accelerator.unwrap_model(network)
        random.seed(907 + rank)
        np.random.seed(807 + rank)
        torch.manual_seed(707 + rank)
        before = copy.deepcopy(
            (
                adapter.state_dict(),
                optimizer.state_dict(),
                scheduler.state_dict(),
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state(),
            )
        )
        directory = save_experiment_checkpoint(args, accelerator, network, state, {"ss_steps": "3"}, with_state=True)
        manifest = validate_experiment_checkpoint(directory)
        assert manifest["process_count"] == 2
        assert {"random_states_0.pkl", "random_states_1.pkl"} <= manifest["files"].keys()
        assert {path.name for path in directory.glob("*.safetensors")} == {"model.safetensors"}
        tensors = load_file(str(directory / "model.safetensors"))
        assert tensors.keys() == adapter.state_dict().keys()
        assert all(tensor.dtype == torch.float32 for tensor in tensors.values())
        assert not list(directory.glob("pytorch_model*"))

        with torch.no_grad():
            for parameter in network.parameters():
                parameter.add_(50 + rank)
        optimizer.param_groups[0]["lr"] = 0.9
        scheduler.step()
        state["optimizer_global_step"] = 91
        random.random(), np.random.rand(), torch.rand(5)
        accelerator.load_state(str(directory))
        after = (
            adapter.state_dict(),
            optimizer.state_dict(),
            scheduler.state_dict(),
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
        assert_equal(before, after)
        assert state["optimizer_global_step"] == 3
        # Different next values prove each process loaded its own saved RNG, not rank zero's copy.
        result = {"rank": rank, "python": random.random(), "numpy": float(np.random.rand()), "torch": torch.rand(1).item()}
        (root / f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
        accelerator.wait_for_everyone()
        accelerator.free_memory()
    finally:
        distributed.destroy_process_group()


def test_actual_two_rank_accelerate_canonical_adapter_and_rng_roundtrip(tmp_path):
    rendezvous = (tmp_path / "rendezvous").resolve().as_uri()
    multiprocessing.spawn(checkpoint_worker, args=(rendezvous, str(tmp_path)), nprocs=2, join=True)
    results = [json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    for key in ("python", "numpy", "torch"):
        assert results[0][key] != results[1][key]
