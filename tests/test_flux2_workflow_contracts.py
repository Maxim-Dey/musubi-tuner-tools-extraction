"""CPU contracts captured before narrowing bf478828's Dev workflow.

Only tiny tensors, callbacks and temporary artifacts; no real-model training.
"""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import re
import sys
import toml
import json
import copy
import random
from multiprocessing import Value

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from musubi_tuner import cache_latents, cache_text_encoder_outputs
from musubi_tuner.dataset.image_video_dataset import BaseDataset, BucketBatchManager, ItemInfo
from musubi_tuner.dataset.cache_io import save_latent_cache_flux_2, save_text_encoder_output_cache_flux_2
from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.flux_2_cache_latents import preprocess_contents_flux_2
from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer
from musubi_tuner.utils.image_utils import save_images_grid
from musubi_tuner.training.sampling_prompts import should_sample_images
from musubi_tuner.training.trainer_base import NetworkTrainer
from musubi_tuner.utils.train_utils import get_remove_step_no


def test_image_control_cache_round_trip(tmp_path):
    dataset = BaseDataset(cache_directory=str(tmp_path), architecture="f2d")
    item = ItemInfo("nested/cup.png", "ceramic cup", (32, 16), content=np.array([[[0, 127, 255]]], dtype=np.uint8))
    item.control_content = [np.array([[[255, 0, 127, 1]]], dtype=np.uint8)]
    item.latent_cache_path = dataset.get_latent_cache_path(item)
    item.text_encoder_output_cache_path = dataset.get_text_encoder_output_cache_path(item)
    assert Path(item.latent_cache_path).name == "cup_0032x0016_f2d.safetensors"
    assert Path(item.text_encoder_output_cache_path).name == "cup_f2d_te.safetensors"
    pixels, controls = preprocess_contents_flux_2([item])
    torch.testing.assert_close(pixels.flatten(), torch.tensor([-1, -0.003921568393707275, 1]), rtol=0, atol=0)
    torch.testing.assert_close(controls[0][0].flatten(), torch.tensor([1, -1, -0.003921568393707275]), rtol=0, atol=0)
    latent = torch.tensor([[[float("nan"), 2.0]]])
    save_latent_cache_flux_2(item, latent, [torch.ones(1, 1, 2)], "flux_2_dev")
    save_text_encoder_output_cache_flux_2(item, torch.ones(2, 3, dtype=torch.bfloat16), "flux_2_dev")
    # Replacing a logical text key changes dtype without retaining its old variant.
    save_text_encoder_output_cache_flux_2(item, torch.full((2, 3), 2.0), "flux_2_dev")
    assert set(load_file(item.latent_cache_path)) == {"latents_1x2_float32", "latents_control_0_1x2_float32"}
    assert set(load_file(item.text_encoder_output_cache_path)) == {"ctx_vec_float32"}
    with safe_open(item.latent_cache_path, framework="pt") as stored:
        assert stored.metadata() == {"architecture": "flux_2_dev", "width": "32", "height": "16", "format_version": "1.0.1"}
    with safe_open(item.text_encoder_output_cache_path, framework="pt") as stored:
        assert stored.metadata() == {"architecture": "flux_2_dev", "caption1": "ceramic cup", "format_version": "1.0.1"}
    batch = BucketBatchManager({(32, 16): [item, item]}, batch_size=2)[0]
    assert batch["timesteps"] is None
    torch.testing.assert_close(batch["latents"], torch.tensor([[[[0.0, 2.0]]]]).repeat(2, 1, 1, 1), rtol=0, atol=0)
    torch.testing.assert_close(batch["ctx_vec"], torch.full((2, 2, 3), 2.0), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["latent", "text"])
@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("custom_current", [False, True])
def test_cache_callbacks_skip_batch_and_cleanup(tmp_path, kind, keep, custom_current):
    items = [ItemInfo(str(i), "cup", (1, 1), content=np.zeros((1, 1, 4), dtype=np.uint8)) for i in range(4)]
    for i, item in enumerate(items):
        item.latent_cache_path = item.text_encoder_output_cache_path = str(tmp_path / f"{i}.safetensors")
    Path(items[0].latent_cache_path).touch()
    stale = tmp_path / "stale.safetensors"
    stale.touch()
    dataset = SimpleNamespace(
        retrieve_latent_cache_batches=lambda workers: iter([((1, 1), items)]),
        retrieve_text_encoder_output_cache_batches=lambda workers: iter([items]),
        get_all_latent_cache_files=lambda: [str(p) for p in tmp_path.glob("*.safetensors")],
        get_all_text_encoder_output_cache_files=lambda: [str(p) for p in tmp_path.glob("*.safetensors")],
    )
    batches = []
    current = (lambda item: item is items[1]) if custom_current else None
    if kind == "latent":
        args = SimpleNamespace(num_workers=1, skip_existing=True, batch_size=2, keep_cache=keep)
        cache_latents.encode_datasets(
            [dataset], lambda batch: batches.append([x.item_key for x in batch]), args, cache_is_current=current
        )
        assert all(item.content.shape[-1] == 3 for item in items)
    else:
        existing, expected = cache_text_encoder_outputs.prepare_cache_files_and_paths([dataset])
        cache_text_encoder_outputs.process_text_encoder_batches(
            1,
            True,
            2,
            [dataset],
            existing,
            expected,
            lambda batch: batches.append([x.item_key for x in batch]),
            cache_is_current=current,
        )
        cache_text_encoder_outputs.post_process_cache_files([dataset], existing, expected, keep)
    assert batches == ([["0", "2"], ["3"]] if custom_current else [["1", "2"], ["3"]])
    assert stale.exists() is keep
    assert Path(items[0].latent_cache_path).exists()


def test_packing_control_coordinates_and_flow_target():
    latents = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    packed, ids = flux2_utils.prc_img(latents)
    assert ids.tolist() == [[[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 0]]]
    torch.testing.assert_close(flux2_utils.scatter_ids(packed, ids)[0].squeeze(2), latents, rtol=0, atol=0)
    refs, ref_ids = flux2_utils.pack_control_latent([latents, latents + 1])
    assert ref_ids[0, :, 0].tolist() == [10] * 4 + [20] * 4
    torch.testing.assert_close(refs[:, :4], packed, rtol=0, atol=0)
    calls = []

    def transformer(**kwargs):
        calls.append(kwargs)
        return kwargs["x"] * 2

    trainer = Flux2NetworkTrainer()
    args = SimpleNamespace(gradient_checkpointing=False, weighting_scheme="none")
    rng = torch.get_rng_state().clone()
    output = trainer.call_dit(
        args,
        SimpleNamespace(device="cpu"),
        transformer,
        latents,
        {"ctx_vec": torch.zeros(1, 3, 4), "latents_control_0": latents},
        torch.ones_like(latents),
        latents + 0.5,
        torch.tensor([501.0]),
        torch.float32,
    )
    torch.testing.assert_close(output.pred, latents * 2 + 1, rtol=0, atol=0)
    torch.testing.assert_close(output.target, 1 - latents, rtol=0, atol=0)
    torch.testing.assert_close(calls[0]["timesteps"], torch.tensor([0.501]), rtol=0, atol=0)
    assert calls[0]["guidance"].tolist() == [1.0]
    assert calls[0]["x"].shape == (1, 8, 2)
    loss, metrics = trainer.compute_loss(args, output, torch.tensor([501.0]), None, torch.float32, torch.float32, 0)
    assert loss.item() == 157.5
    assert metrics == {}
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("subdir,rescale", [(False, False), (True, True)])
def test_png_pixels_paths_and_return_value(tmp_path, subdir, rescale):
    pixels = torch.tensor([[[[[-1.0, 0, 1, 2]]]]]).repeat(1, 3, 1, 1, 1)
    paths = save_images_grid(pixels, str(tmp_path), "sample", rescale=rescale, create_subdir=subdir)
    expected = tmp_path / "sample" / "sample_000.png" if subdir else tmp_path / "sample_000.png"
    assert paths == [str(expected)]
    image = np.asarray(Image.open(expected))
    assert image.shape == (1, 4, 3)
    assert image[0, :, 0].tolist() == ([0, 127, 255, 255] if rescale else [0, 0, 255, 255])


@pytest.mark.parametrize(
    "step,sample,remove",
    [
        (0, True, None),
        (249, False, None),
        (250, True, None),
        (1000, True, None),
        (1250, True, 0),
        (1500, True, 250),
        (2000, True, 750),
    ],
)
def test_sample_and_retention_boundaries(step, sample, remove):
    args = SimpleNamespace(
        sample_at_first=True, sample_every_n_steps=250, sample_every_n_epochs=None, save_every_n_steps=250, save_last_n_steps=1000
    )
    assert should_sample_images(args, step) is sample
    assert get_remove_step_no(args, step) == remove


def test_sample_callback_restores_rng_and_swap(tmp_path):
    transitions = []
    transformer = SimpleNamespace(
        switch_block_swap_for_inference=lambda: transitions.append("inference"),
        switch_block_swap_for_training=lambda: transitions.append("training"),
    )
    accelerator = SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda model: model, autocast=nullcontext)
    args = SimpleNamespace(sample_at_first=True, sample_every_n_steps=250, sample_every_n_epochs=None, output_dir=str(tmp_path))
    trainer = NetworkTrainer()
    trainer.sample_image_inference = lambda *args: torch.rand(3)
    torch.manual_seed(123)
    rng = torch.get_rng_state().clone()
    trainer.sample_images(accelerator, args, None, 0, None, transformer, [{"prompt": "cup"}], torch.float32)
    assert torch.equal(rng, torch.get_rng_state())
    assert transitions == ["inference", "training"]


def test_sample_mode_defaults_filename_and_tracker(tmp_path, monkeypatch):
    from musubi_tuner.training import trainer_base

    trainer = Flux2NetworkTrainer()
    trainer.handle_model_specific_args(SimpleNamespace(model_version="dev", mixed_precision="bf16"))
    model = torch.nn.Linear(1, 1)
    calls, logged = [], []

    def infer(*args, **kwargs):
        assert not model.training
        calls.append(args)
        return torch.full((1, 3, 1, 1, 1), 0.5)

    trainer.do_inference = infer
    monkeypatch.setattr(trainer_base.time, "strftime", lambda *args: "20000102030405")
    monkeypatch.setattr(
        trainer_base,
        "wandb_tracker_and_module",
        lambda accelerator: (
            SimpleNamespace(log=lambda data, step: logged.append((data, step))),
            SimpleNamespace(Image=lambda path: path),
        ),
    )
    trainer.sample_image_inference(
        SimpleNamespace(device=torch.device("cpu")),
        SimpleNamespace(output_name="cup"),
        model,
        torch.bfloat16,
        torch.nn.Identity(),
        str(tmp_path),
        {"prompt": "cup", "enum": 2, "seed": 3},
        None,
        250,
    )
    assert model.training
    assert calls[0][6:11] == (None, 20, 256, 256, 1)
    assert calls[0][13] == 4.0
    path = str(tmp_path / "cup_000250_02_20000102030405_3_000.png")
    assert logged == [({"sample_2": path}, 250)]
    assert np.asarray(Image.open(path)).tolist() == [[[127, 127, 127]]]


def test_tensorboard_step_and_scalar_round_trip(tmp_path):
    logs = NetworkTrainer().generate_step_logs(
        SimpleNamespace(optimizer_type="AdamW"), 0.25, 0.5, SimpleNamespace(get_last_lr=lambda: [0.0001]), ["unet"]
    )
    with SummaryWriter(str(tmp_path)) as writer:
        for key, value in logs.items():
            writer.add_scalar(key, value, 250)
    events = EventAccumulator(str(tmp_path)).Reload()
    assert set(events.Tags()["scalars"]) == {"loss/current", "loss/average", "lr/unet"}
    event = events.Scalars("loss/current")[0]
    assert (event.step, event.value) == (250, 0.25)
    from musubi_tuner.utils.train_utils import get_sanitized_config_or_none

    assert get_sanitized_config_or_none(
        SimpleNamespace(log_config=True, learning_rate=1e-4, _config_sources={"learning_rate": "file"}, _tracker_init_kwargs={})
    ) == {"log_config": True, "learning_rate": 1e-4}


def test_cpu_state_hooks_and_resume_without_training_loop(tmp_path):
    accelerator = Accelerator(cpu=True)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    trainer = NetworkTrainer()
    trainer._register_hooks_and_resume(SimpleNamespace(resume=None), accelerator, model)
    # One toy update creates optimizer state. This never calls a trainer loop.
    accelerator.backward(model(torch.ones(1, 2)).square().mean())
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    saved_lr = optimizer.param_groups[0]["lr"]
    state = tmp_path / "state"
    accelerator.save_state(str(state))
    expected_rng = torch.get_rng_state().clone()
    assert {"model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"} <= {p.name for p in state.iterdir()}
    with torch.no_grad():
        for param in model.parameters():
            param.add_(10)
    assert trainer.resume_from_local_or_hf_if_specified(
        accelerator, SimpleNamespace(resume=str(state), resume_from_huggingface=False)
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    assert optimizer.param_groups[0]["lr"] == saved_lr == 0.005
    assert torch.equal(torch.get_rng_state(), expected_rng)
    accelerator.free_memory()


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("workers,persistent", [(0, False), (1, False), (2, True)])
def test_cpu_resume_matches_continuous_updates(tmp_path, monkeypatch, workers, persistent):
    from accelerate.utils import set_seed
    from safetensors.torch import save_file
    from musubi_tuner.dataset.image_video_dataset import DatasetGroup, ImageDataset
    from musubi_tuner.training.accelerator_setup import collator_class
    from musubi_tuner.training.parser_common import setup_parser_common
    from musubi_tuner.utils import train_utils

    # Five cached toy items in two datasets exercise lazy per-dataset shuffling,
    # an accumulation remainder, epoch transitions and deterministic cache skips.
    def make_dataset(epoch):
        datasets = []
        for group_index, count in enumerate((3, 2)):
            dataset = ImageDataset(
                (16, 16),
                ".txt",
                1,
                1,
                False,
                False,
                image_directory=str(tmp_path),
                cache_directory=str(tmp_path),
                architecture="f2d",
            )
            items = []
            for index in range(count):
                key = group_index * 3 + index
                latent = tmp_path / f"latent{key}.safetensors"
                context = tmp_path / f"context{key}.safetensors"
                save_file({"latents_1x1_float32": torch.tensor([key + 1.0, key + 2.0]).reshape(2, 1, 1)}, latent)
                save_file({"ctx_vec_float32": torch.zeros(1, 2)}, context)
                item = ItemInfo(str(key), "toy", (16, 16))
                item.latent_cache_path, item.text_encoder_output_cache_path = str(latent), str(context)
                items.append(item)
            dataset.batch_manager = BucketBatchManager({(16, 16): items}, 1, num_timestep_buckets=3)
            dataset.num_train_items = count
            dataset.set_seed(71, epoch)
            datasets.append(dataset)
        return DatasetGroup(datasets)

    class ToyTrainer(Flux2NetworkTrainer):
        def scale_shift_latents(self, latents):
            return latents

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
            x = latents.flatten(1)
            self.seen.append((global_step, x.tolist(), batch["timesteps"]))
            target = noise.flatten(1) + random.random() + float(np.random.random()) + self.get_bucketed_timestep()
            return (transformer.double_blocks[0].proj(x) - target).square().mean(), {}

        def sample_images(self, accelerator, args, epoch, steps, *rest):
            self.samples.append((epoch, steps))

    def run(output, resume=None):
        set_seed(123)
        args = setup_parser_common().parse_args([])
        args.network_module = "networks.lora_flux_2"
        args.optimizer_type, args.learning_rate = "AdamW", 0.001
        args.lr_scheduler, args.lr_warmup_steps = "linear", 0
        args.max_train_steps, args.gradient_accumulation_steps = 6, 2
        args.num_timestep_buckets = 3
        args.output_dir, args.output_name = str(output), "toy"
        args.save_every_n_steps, args.save_every_n_epochs, args.save_state = 1, 1, True
        args.sample_at_first, args.sample_every_n_steps = True, 2
        args.resume, args.full_fp16, args.full_bf16 = str(resume) if resume else None, False, False
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
        epoch = Value("i", 0)
        dataset = make_dataset(epoch)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=workers,
            persistent_workers=persistent,
            collate_fn=collator_class(epoch, dataset if workers == 0 else None),
        )
        model, network = _tiny_dev_lora()
        trainer = ToyTrainer()
        trainer.model_version_info = flux2_utils.FLUX2_MODEL_INFO["dev"]
        trainer.seen, trainer.samples = [], []
        trainer.num_timestep_buckets = 3
        name, optimizer_args, optimizer, train_fn, eval_fn = trainer.get_optimizer(args, list(network.parameters()))
        scheduler = trainer.get_lr_scheduler(args, optimizer, 1)
        network, optimizer, loader, scheduler = accelerator.prepare(network, optimizer, loader, scheduler)
        trainer._register_hooks_and_resume(args, accelerator, network)
        logs, saves, save_marks = [], [], {}
        # Backend startup may consume RNG; reinitializing it on resume must not
        # alter the saved training RNG. No tracker/service is started here.
        monkeypatch.setattr(accelerator, "init_trackers", lambda *a, **kw: (random.random(), np.random.random(), torch.rand(1)))
        accelerator.trackers = [object()]
        monkeypatch.setattr(accelerator, "log", lambda values, step: logs.append((step, copy.deepcopy(values))))
        monkeypatch.setattr(accelerator, "end_training", lambda: None)
        original_save = accelerator.save_state

        def save_state(path, **kwargs):
            original_save(path, **kwargs)
            saves.append(Path(path).name)
            save_marks[Path(path).name] = (len(trainer.samples), len(logs), len(saves))

        monkeypatch.setattr(accelerator, "save_state", save_state)
        trainer._run_training_loop(
            args,
            accelerator,
            1,
            0,
            dataset,
            loader,
            epoch,
            model,
            network,
            network,
            optimizer,
            name,
            optimizer_args,
            train_fn,
            eval_fn,
            scheduler,
            None,
            None,
            None,
            torch.float32,
            torch.float32,
        )
        result = copy.deepcopy(
            (
                network.state_dict(),
                optimizer.state_dict(),
                scheduler.state_dict(),
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state(),
                trainer._training_progress,
                accelerator.step,
                trainer.seen,
                trainer.samples,
                logs,
                saves,
                save_marks,
            )
        )
        accelerator.trackers = []
        accelerator.free_memory()
        return result

    continuous = run(tmp_path / "continuous")
    # Step 1: mid-epoch; step 3: final partial accumulation; step 4: second
    # epoch; epoch state: after epoch events; final state: no further updates.
    for state_name, completed_batches in [
        ("toy-step00000001-state", 2),
        ("toy-step00000003-state", 5),
        ("toy-step00000004-state", 7),
        ("toy-000001-state", 5),
        ("toy-step00000006-state", 10),
        ("toy-state", 10),
    ]:
        resumed = run(tmp_path / f"resumed-{state_name}", tmp_path / "continuous" / state_name)
        _assert_nested_equal(resumed[:6], continuous[:6])
        for key in ("epoch", "next_batch", "global_step", "loss_list", "loss_total", "timestep_range_pool"):
            _assert_nested_equal(resumed[6][key], continuous[6][key])
        assert resumed[7] == continuous[7]
        assert resumed[8] == continuous[8][completed_batches:]
        sample_index, log_index, save_index = continuous[12][state_name]
        assert resumed[9] == continuous[9][sample_index:]
        _assert_nested_equal([row for row in resumed[10] if row[1]], [row for row in continuous[10][log_index:] if row[1]])
        assert resumed[11] == (continuous[11][save_index:] if state_name != "toy-state" else ["toy-state"])
        assert (0, 0) not in resumed[9]
        assert all(
            step >= train_utils.load_training_progress(tmp_path / "continuous" / state_name)["global_step"]
            for step, values in resumed[10]
            if "loss/current" in values
        )
        assert resumed[6]["global_step"] == 6


def test_legacy_resume_warns_and_corrupt_progress_is_rejected(tmp_path, caplog):
    from musubi_tuner.utils import train_utils

    accelerator = Accelerator(cpu=True)
    model = accelerator.prepare(torch.nn.Linear(2, 1))
    state = tmp_path / "legacy"
    accelerator.save_state(state)
    expected = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        model.weight.add_(10)
    trainer = NetworkTrainer()
    trainer._register_hooks_and_resume(SimpleNamespace(resume=str(state), resume_from_huggingface=False), accelerator, model)
    _assert_nested_equal(model.state_dict(), expected)
    assert trainer._training_progress is None
    assert "Legacy state" in caplog.text and "restarting epoch/global_step" in caplog.text
    torch.save({"version": 1, "epoch": 0}, state / train_utils.TRAINING_PROGRESS_NAME)
    with pytest.raises(ValueError, match="resume progress.*global_step|resume progress.*next_batch"):
        train_utils.load_training_progress(state)
    accelerator.free_memory()


@pytest.mark.parametrize("network_module", ["networks.lora_flux_2", "musubi_tuner.networks.lora_flux_2"])
def test_complete_supplied_template_contract(tmp_path, monkeypatch, network_module):
    from musubi_tuner.flux_2_train_network import flux2_setup_parser, validate_training_inputs
    from musubi_tuner.training.parser_common import setup_parser_common, read_config_from_file, validate_effective_args
    from musubi_tuner.training.accelerator_setup import validate_tracker_config, validate_attention_dependencies
    from musubi_tuner.networks.lora_flux_2 import validate_network_args
    from musubi_tuner.dataset.config_utils import load_user_config, BlueprintGenerator, ConfigSanitizer
    from musubi_tuner.training.sampling_prompts import load_prompts

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    baseline = (root / "specs/001-scope-flux2-dev-lora/baseline.md").read_text(encoding="utf-8")
    snapshot = baseline.split("### Literal original templates", 1)[1]
    train_before, dataset_before = re.findall(r"```toml\n(.*?)\n```", snapshot, re.S)[:2]
    expected = toml.loads(train_before)
    expected.update(dataset_config="./flux2dev_lora/dataset.toml", sample_prompts="./flux2dev_lora/sample_prompts.txt")
    train_file = root / "flux2dev_lora/train.toml"
    actual_text = train_file.read_text(encoding="utf-8")
    actual = toml.loads(actual_text)
    assert actual == expected  # every original active value, not a subset whitelist
    assert (
        actual_text.strip()
        == train_before.replace("./configs/flux2_dev_style/dataset.toml", expected["dataset_config"])
        .replace("./configs/flux2_dev_style/sample_prompts.txt", expected["sample_prompts"])
        .strip()
    )
    assert "# blocks_to_swap = 20" in actual_text
    assert Path(actual["dataset_config"]).is_file()
    assert load_prompts(actual["sample_prompts"]) == [{"prompt": "A ceramic cup on a wooden table.", "enum": 0}]
    dataset = load_user_config(actual["dataset_config"])
    assert dataset == toml.loads(dataset_before)
    blueprint = BlueprintGenerator(ConfigSanitizer()).generate(dataset, SimpleNamespace(), architecture="f2d")
    params = blueprint.dataset_group.datasets[0].params
    assert (params.resolution, params.batch_size, params.num_repeats, params.enable_bucket, params.bucket_no_upscale) == (
        (1024, 1024),
        1,
        1,
        True,
        True,
    )

    # Substitute only external prerequisites in temporary copies. No user resource is read.
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (16, 16)).save(images / "cup.png")
    (images / "cup.txt").write_text("cup", encoding="utf-8")
    dataset["datasets"][0].update(image_directory=str(images), cache_directory=str(tmp_path / "cache"))
    dataset_path = tmp_path / "dataset.toml"
    dataset_path.write_text(toml.dumps(dataset), encoding="utf-8")
    weights = tmp_path / "external.safetensors"
    weights.touch()
    external = {
        "dataset_config": str(dataset_path),
        "dit": str(weights),
        "vae": str(weights),
        "text_encoder": str(weights),
        "output_dir": str(tmp_path / "output"),
        "logging_dir": str(tmp_path / "logs"),
    }
    path = tmp_path / "train.toml"
    path.write_text(toml.dumps(actual | external | {"network_module": network_module}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train", "--config_file", str(path)])
    parser = flux2_setup_parser(setup_parser_common())
    args = read_config_from_file(parser.parse_args(), parser)
    validate_effective_args(args, parser)
    for key, value in (actual | external | {"network_module": network_module}).items():
        assert getattr(args, key) == value
    validate_network_args(args)
    trainer = Flux2NetworkTrainer()
    trainer.validate_optimizer_and_scheduler(args)  # actual bitsandbytes selection, no optimizer construction
    validate_attention_dependencies(args)
    validate_tracker_config(args)

    def forbidden(*args, **kwargs):
        pytest.fail("template parsing attempted a weight load")

    for name in ("load_flow_model", "load_ae", "load_text_embedder"):
        monkeypatch.setattr(flux2_utils, name, forbidden)

    def processor_boundary(name, **kwargs):
        assert name == flux2_utils.M3_TOKENIZER_ID
        assert kwargs == {"local_files_only": True, "use_fast": False}
        return object()  # real processor availability is a separately recorded environment gate

    monkeypatch.setattr(flux2_utils.AutoProcessor, "from_pretrained", processor_boundary)
    validate_training_inputs(args)


def test_image_order_dataset_indices_repeats_and_control_buckets(tmp_path):
    from musubi_tuner.dataset.image_video_dataset import ImageDataset, DatasetGroup

    for name in ("b", "a"):
        Image.new("RGB", (32, 32), (10, 20, 30)).save(tmp_path / f"{name}.png")
        (tmp_path / f"{name}.txt").write_text(f"caption {name}", encoding="utf-8")
    datasets = [
        ImageDataset(
            resolution=(32, 32),
            caption_extension=".txt",
            batch_size=2,
            num_repeats=2,
            enable_bucket=True,
            bucket_no_upscale=True,
            image_directory=str(tmp_path),
            cache_directory=str(tmp_path / f"cache{i}"),
            architecture="f2d",
        )
        for i in range(2)
    ]
    assert [d.dataset_index for d in datasets] == [None, None]
    DatasetGroup(datasets)
    assert [d.dataset_index for d in datasets] == [0, 1]
    dataset = datasets[1]
    assert [Path(p).name for p in dataset.datasource.image_paths] == ["a.png", "b.png"]
    assert [dataset.datasource.get_caption(i)[1] for i in range(2)] == ["caption a", "caption b"]
    items = [item for _, batch in dataset.retrieve_latent_cache_batches(1) for item in batch]
    assert sorted((item.dataset_index, item.datasource_index) for item in items) == [(1, 0), (1, 1)]
    for i, item in enumerate(items):
        save_latent_cache_flux_2(item, torch.full((2, 1, 1), float(i)), [torch.zeros(2, 1, 1) for _ in range(i + 1)], "flux_2_dev")
        item.text_encoder_output_cache_path = dataset.get_text_encoder_output_cache_path(item)
        save_text_encoder_output_cache_flux_2(item, torch.full((2, 3), float(i)), "flux_2_dev")
    dataset.prepare_for_training()
    assert dataset.num_train_items == 4 and len(dataset) == 2
    batches = [dataset.batch_manager[i] for i in range(2)]
    assert sorted(len([k for k in batch if k.startswith("latents_control_")]) for batch in batches) == [1, 2]
    assert all(batch["latents"].shape == (2, 2, 1, 1) for batch in batches)
    assert all(torch.equal(batch["latents"][0], batch["latents"][1]) for batch in batches)


def test_image_jsonl_paths_cwd_first_then_jsonl_directory(tmp_path, monkeypatch):
    from musubi_tuner.dataset.datasources import ImageJsonlDatasource

    directory, cwd = tmp_path / "dataset", tmp_path / "cwd"
    directory.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    for base, name, color in [(cwd, "shadow.png", 10), (directory, "shadow.png", 20), (directory, "nearby.png", 30)]:
        Image.new("RGB", (16, 16), (color, 0, 0)).save(base / name)
    path = directory / "images.jsonl"
    records = [
        {"image_path": name, "caption": "cup", "control_path": "nearby.png"} for name in ("shadow.png", "nearby.png", "missing.png")
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    datasource = ImageJsonlDatasource(str(path))
    assert [item["image_path"] for item in datasource.data] == [
        str(cwd / "shadow.png"),
        str(directory / "nearby.png"),
        "missing.png",
    ]
    assert all(item["control_path_0"] == str(directory / "nearby.png") for item in datasource.data)
    assert [datasource.get_image_data(i)[1][0].getpixel((0, 0))[0] for i in (0, 1)] == [10, 30]
    with pytest.raises(FileNotFoundError):
        datasource.get_image_data(2)


def test_mistral_layer_padding_feature_and_cache_contract(tmp_path):
    embedder = flux2_utils.Mistral3Embedder.__new__(flux2_utils.Mistral3Embedder)
    torch.nn.Module.__init__(embedder)

    def tokenize(messages, **kwargs):
        assert kwargs["max_length"] == 512 and kwargs["padding"] == "max_length"
        assert kwargs["truncation"] and kwargs["tokenize"] and not kwargs["add_generation_prompt"]
        return {"input_ids": torch.zeros(1, 512, dtype=torch.long), "attention_mask": torch.ones(1, 512, dtype=torch.long)}

    class HiddenStates(torch.nn.Module):
        device = torch.device("cpu")

        def forward(self, input_ids, attention_mask, output_hidden_states, use_cache):
            assert input_ids.shape == attention_mask.shape == (1, 512)
            assert output_hidden_states and not use_cache
            return SimpleNamespace(hidden_states=[torch.tensor(float(i)).expand(1, 512, 5120) for i in range(31)])

    embedder.tokenizer = SimpleNamespace(apply_chat_template=tokenize)
    embedder.mistral3 = HiddenStates()
    output = embedder(["cup"])
    assert output.shape == (1, 512, 15360)
    assert output[0, 0, [0, 5120, 10240]].tolist() == [10, 20, 30]
    item = ItemInfo("cup.png", "cup", (16, 16))
    item.text_encoder_output_cache_path = str(tmp_path / "cup_f2d_te.safetensors")
    save_text_encoder_output_cache_flux_2(item, output[0], "flux_2_dev")
    assert load_file(item.text_encoder_output_cache_path)["ctx_vec_float32"].shape == (512, 15360)


def _tiny_dev_lora(dropout=0.05):
    from musubi_tuner.networks import lora_flux_2

    class DoubleStreamBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2, bias=False)

    model = torch.nn.Module()
    model.double_blocks = torch.nn.ModuleList([DoubleStreamBlock()])
    model.requires_grad_(False)
    network = lora_flux_2.create_arch_network(1.0, 32, 32, None, None, model, neuron_dropout=dropout, loraplus_lr_ratio="2")
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    with torch.no_grad():
        model.double_blocks[0].proj.weight.fill_(0.25)
        network.unet_loras[0].lora_down.weight.fill_(0.1)
        network.unet_loras[0].lora_up.weight.fill_(0.2)
    return model, network


@pytest.mark.parametrize("spaced", [False, True])
def test_nested_names_reach_optimizer_scheduler_and_network(spaced):
    from musubi_tuner.training.parser_common import setup_parser_common

    separator = " =" if spaced else "="
    args = setup_parser_common().parse_args([])
    args.optimizer_type = "AdamW"
    args.optimizer_args = [f"weight_decay{separator}0.1", f"betas{separator}(0.8, 0.95)"]
    args.lr_scheduler_type = "StepLR"
    args.lr_scheduler_args = [f"step_size{separator}2", f"gamma{separator}0.25"]
    args.network_module = "networks.lora_flux_2"
    args.network_args = [f"rank_dropout{separator}0.5"]
    trainer = NetworkTrainer()
    trainer.validate_optimizer_and_scheduler(args)
    parameter = torch.nn.Parameter(torch.ones(1))
    _, _, optimizer, _, _ = trainer.get_optimizer(args, [parameter])
    scheduler = trainer.get_lr_scheduler(args, optimizer, 1)
    assert optimizer.defaults["weight_decay"] == 0.1 and optimizer.defaults["betas"] == (0.8, 0.95)
    assert scheduler.step_size == 2 and scheduler.gamma == 0.25
    model, _ = _tiny_dev_lora()
    network = trainer._build_network(args, SimpleNamespace(print=lambda *a: None), model, None, torch.float32)
    assert network.unet_loras[0].rank_dropout == 0.5


@pytest.mark.parametrize("from_weights", [False, True])
def test_build_network_initial_weights_restores_rank_alpha_and_values(tmp_path, from_weights):
    from musubi_tuner.training.parser_common import setup_parser_common
    from safetensors.torch import save_file

    model, original = _tiny_dev_lora()
    state = original.state_dict()
    prefix = "lora_unet_double_blocks_0_proj."
    state[prefix + "alpha"] = torch.tensor(6.0)
    state[prefix + "lora_down.weight"] = torch.full((3, 2), 0.3)
    state[prefix + "lora_up.weight"] = torch.full((2, 3), 0.4)
    path = tmp_path / "initial.safetensors"
    save_file(state, path)
    args = setup_parser_common().parse_args([])
    args.network_module = "networks.lora_flux_2"
    args.network_weights = str(path)
    args.dim_from_weights = from_weights
    args.network_dim = 17 if from_weights else 3
    args.network_alpha = 17 if from_weights else 6
    network = NetworkTrainer()._build_network(args, SimpleNamespace(print=lambda *a: None), model, None, torch.float32)
    adapter = network.unet_loras[0]
    assert adapter.lora_dim == 3 and adapter.alpha.item() == 6 and adapter.scale == 2
    for key, value in network.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0, check_dtype=False)


def test_generic_lora_excludes_conv3d_and_preserves_conv2d():
    from musubi_tuner.networks import lora

    model = torch.nn.ModuleDict({"image": torch.nn.Conv2d(2, 2, 1), "volume": torch.nn.Conv3d(2, 2, 1)})
    network = lora.create_network(None, "lora_unet", 1.0, 2, 2, None, None, model)
    assert [adapter.lora_name for adapter in network.unet_loras] == ["lora_unet_image"]
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    model["image"](torch.ones(1, 2, 4, 4)).sum().backward()
    assert network.unet_loras[0].lora_up.weight.grad is not None


@pytest.mark.parametrize("extension", ["json", "toml"])
@pytest.mark.parametrize("multiple", [False, True])
def test_structured_control_paths_reach_sampling_consumer(tmp_path, monkeypatch, extension, multiple):
    from musubi_tuner.training.sampling_prompts import load_prompts

    paths = [str(tmp_path / "control one.png"), str(tmp_path / "control two.png")]
    for path in paths:
        Image.new("RGB", (16, 16)).save(path)
    control = paths if multiple else paths[0]
    record = {"prompt": "cup", "control_image_path": control}
    path = tmp_path / f"sample.{extension}"
    path.write_text(json.dumps([record]) if extension == "json" else toml.dumps({"prompt": {"subset": [record]}}), encoding="utf-8")
    parameter = load_prompts(str(path))[0] | {"ctx_vec": torch.zeros(1, 2, 3), "negative_ctx_vec": torch.zeros(1, 2, 3)}
    seen = []

    class StopSampling(Exception):
        pass

    def preprocess(image_path, limit_size):
        seen.append(image_path)
        assert Path(image_path).is_file()
        return torch.zeros(1, 3, 16, 16), None, None

    def stop(*a, **kw):
        raise StopSampling

    def no_weights(*a, **kw):
        pytest.fail("weight loader called")

    monkeypatch.setattr(flux2_utils, "preprocess_control_image", preprocess)
    monkeypatch.setattr(flux2_utils, "pack_control_latent", stop)
    for name in ("load_flow_model", "load_ae", "load_text_embedder"):
        monkeypatch.setattr(flux2_utils, name, no_weights)
    vae = SimpleNamespace(to=lambda *a: None, eval=lambda: None, encode=lambda x: x, dtype=torch.float32)
    generator = torch.Generator().manual_seed(7)
    before = torch.get_rng_state().clone()
    with pytest.raises(StopSampling):
        Flux2NetworkTrainer().do_inference(
            SimpleNamespace(device=torch.device("cpu")),
            None,
            parameter,
            vae,
            torch.float32,
            None,
            1,
            1,
            16,
            16,
            1,
            generator,
            False,
            4,
            1,
        )
    assert seen == (paths if multiple else paths[:1])
    assert torch.equal(before, torch.get_rng_state())


def test_dev_lora_dropout_gradients_groups_and_enable_restore():
    model, network = _tiny_dev_lora()
    projection = model.double_blocks[0].proj
    adapter = network.unet_loras[0]
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    base = torch.nn.functional.linear(x, projection.weight)
    torch.manual_seed(17)
    adapted = projection(x)
    after = torch.get_rng_state()
    torch.manual_seed(17)
    delta = torch.nn.functional.linear(
        torch.nn.functional.dropout(torch.nn.functional.linear(x, adapter.lora_down.weight), p=0.05), adapter.lora_up.weight
    )
    torch.testing.assert_close(adapted, base + delta, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), after)
    adapted.square().mean().backward()
    assert projection.weight.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in network.parameters())
    groups, names = network.prepare_optimizer_params(1e-4)
    assert [g["lr"] for g in groups] == [1e-4, 2e-4]
    assert names == ["unet", "unet plus"]
    network.eval()
    adapted = projection(x)
    network.set_enabled(False)
    torch.testing.assert_close(projection(x), base)
    network.set_enabled(True)
    torch.testing.assert_close(projection(x), adapted)
    assert not torch.allclose(adapted, base)


def test_cpu_accumulation_optimizer_then_scheduler():
    from musubi_tuner.training.parser_common import setup_parser_common

    args = setup_parser_common().parse_args(
        [
            "--optimizer_type",
            "SGD",
            "--learning_rate",
            "0.1",
            "--lr_scheduler_type",
            "StepLR",
            "--lr_scheduler_args",
            "step_size=1",
            "gamma=0.5",
        ]
    )
    trainer = NetworkTrainer()
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    _, _, optimizer, _, _ = trainer.get_optimizer(args, [parameter])
    scheduler = trainer.get_lr_scheduler(args, optimizer, 1)
    # One scalar toy update with four accumulated microbatches; never a trainer loop.
    for value in (1.0, 2.0, 3.0, 4.0):
        ((parameter * value).square() / 4).backward()
    assert parameter.grad.item() == 15.0
    optimizer.step()
    assert parameter.item() == pytest.approx(-0.5, abs=1e-7)
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    assert scheduler.get_last_lr() == [0.05] and parameter.grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_lora_safetensors_keys_metadata_and_reload(tmp_path, dtype):
    from musubi_tuner.utils.train_utils import get_step_ckpt_name

    model, network = _tiny_dev_lora()
    path = tmp_path / get_step_ckpt_name("cup", 250)
    assert path.name == "cup-step00000250.safetensors"
    metadata = {"ss_network_module": "networks.lora_flux_2", "ss_network_dim": "32", "ss_network_alpha": "32", "ss_steps": "250"}
    network.save_weights(str(path), dtype, metadata)
    saved = load_file(path)
    assert set(saved) == {f"lora_unet_double_blocks_0_proj.{key}" for key in ("alpha", "lora_down.weight", "lora_up.weight")}
    assert {v.dtype for v in saved.values()} == {dtype}
    with safe_open(path, framework="pt") as stored:
        assert all(stored.metadata()[k] == v for k, v in metadata.items())
    _, restored = _tiny_dev_lora()
    restored.load_weights(str(path))
    for key, value in restored.state_dict().items():
        if key.endswith(".alpha"):
            assert value.dtype == torch.int64 and value.item() == saved[key].item() == 32
        else:
            torch.testing.assert_close(value, saved[key].float(), rtol=0, atol=0)


def test_state_retention_names_and_train_end(tmp_path):
    from musubi_tuner.utils.train_utils import save_and_remove_state_stepwise, save_state_on_train_end

    args = SimpleNamespace(
        output_dir=str(tmp_path),
        output_name="cup",
        save_state_to_huggingface=False,
        save_last_n_steps_state=1000,
        save_last_n_steps=1000,
        save_every_n_steps=250,
    )
    old, retained = tmp_path / "cup-step00000250-state", tmp_path / "cup-step00000500-state"
    old.mkdir()
    retained.mkdir()
    saved = []
    accelerator = SimpleNamespace(save_state=lambda path: saved.append(Path(path).name))
    save_and_remove_state_stepwise(args, accelerator, 1500)
    assert saved == ["cup-step00001500-state"]
    assert not old.exists() and retained.is_dir()
    save_state_on_train_end(args, accelerator)
    assert saved[-1] == "cup-state"


def test_common_fp8_patch_matches_dequantized_reference():
    from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8, bias=False)).to(torch.bfloat16)
    state = model.state_dict()
    for name, shape in (("0", (8, 1)), ("1", (1,))):
        state[f"{name}.weight"] = state[f"{name}.weight"].to(torch.float8_e4m3fn)
        state[f"{name}.scale_weight"] = torch.full(shape, 0.5, dtype=torch.bfloat16)
    apply_fp8_monkey_patch(model, state, use_scaled_mm=False)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    assert not missing and not unexpected
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    reference = torch.nn.functional.linear(x, state["0.weight"].to(torch.bfloat16) * state["0.scale_weight"], state["0.bias"])
    reference = torch.nn.functional.linear(reference, state["1.weight"].to(torch.bfloat16) * state["1.scale_weight"])
    actual = model(x)
    assert torch.equal(actual, reference)  # original TE shared arithmetic required exact equality
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)  # original DiT tolerance retained
