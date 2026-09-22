"""CPU characterizations of the retained numerical and artifact boundaries."""

from types import SimpleNamespace
import copy
import random

from accelerate import Accelerator
import numpy as np
from PIL import Image
import pytest
import torch

from musubi_tuner.qwen_image.qwen_image_modules import attention
from musubi_tuner.utils.model_utils import is_fp8
from musubi_tuner.utils.image_utils import save_images_grid
from musubi_tuner.qwen_image import qwen_image_utils
from musubi_tuner.qwen_image_train_network import QwenImageNetworkTrainer
from musubi_tuner.training.trainer_base import NetworkTrainer
from musubi_tuner.training.sampling_prompts import should_sample_images
from musubi_tuner.qwen_image.qwen_image_model import QwenImageTransformer2DModel, QwenImageTransformerBlock
from musubi_tuner.networks import lora_qwen_image
from musubi_tuner.utils import train_utils


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("qkv_list", [False, True])
def test_attention_outputs_and_gradients(split, qkv_list):
    generator = torch.Generator().manual_seed(71)
    q, k, v = [torch.randn(2, 4, 2, 3, generator=generator, dtype=torch.float64).requires_grad_() for _ in range(3)]
    reference_inputs = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    lengths = torch.tensor([4, 2])
    mask = torch.arange(4)[None, :] < lengths[:, None]
    expected = []
    for i, size in enumerate(lengths.tolist()):
        rq, rk, rv = [x[i, : size if split else 4].transpose(0, 1) for x in reference_inputs]
        scores = rq @ rk.transpose(-1, -2) / (3**0.5)
        if not split:
            scores = scores.masked_fill(~mask[i][None, None, :], -torch.inf)
        result = (scores.softmax(-1) @ rv).transpose(0, 1).reshape(-1, 6)
        if split:
            result = torch.nn.functional.pad(result, (0, 0, 0, 4 - size))
        expected.append(result)
    expected = torch.stack(expected)
    values = [q, k, v]
    kwargs = {"mode": "torch", "total_len": lengths if split else None, "attn_mask": None if split else mask[:, None, None, :]}
    actual = attention(values, **kwargs) if qkv_list else attention(q, k, v, **kwargs)
    if qkv_list:
        assert values == []  # releases temporary QKV references
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for original, reference in zip((q, k, v), reference_inputs):
        torch.testing.assert_close(original.grad, reference.grad)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz])
def test_fp8_classification(dtype):
    assert is_fp8(dtype)
    assert not is_fp8(torch.float16)
    assert not is_fp8(torch.bfloat16)
    assert not is_fp8(torch.float32)
    assert not is_fp8(torch.int8)


def test_png_grid_pixels_and_layout(tmp_path):
    pixels = torch.zeros(2, 3, 1, 2, 2)
    pixels[0, 0] = 1  # red
    pixels[1, 1] = 0.5  # half green -> uint8 127
    paths = save_images_grid(pixels, str(tmp_path), "sample", n_rows=2, create_subdir=False)
    assert len(paths) == 1
    actual = np.asarray(Image.open(paths[0]))
    expected = np.zeros((6, 10, 3), dtype=np.uint8)
    expected[2:4, 2:4, 0] = 255
    expected[2:4, 6:8, 1] = 127
    np.testing.assert_array_equal(actual, expected)


def test_original_packing_order_and_round_trip():
    latents = torch.arange(2 * 3 * 4 * 6).reshape(2, 3, 1, 4, 6).float()
    packed = qwen_image_utils.pack_latents(latents)
    expected = torch.stack([latents[:, :, 0, y : y + 2, x : x + 2].reshape(2, -1) for y in (0, 2) for x in (0, 2, 4)], dim=1)
    torch.testing.assert_close(packed, expected)
    restored = qwen_image_utils.unpack_latents(packed, 32, 48)
    torch.testing.assert_close(restored, latents)


@pytest.mark.parametrize("split", [False, True])
def test_original_forward_target_and_weighted_loss(split):
    trainer = QwenImageNetworkTrainer()
    trainer.is_edit = False
    trainer.is_layered = False
    args = SimpleNamespace(is_layered=False, split_attn=split, gradient_checkpointing=True, weighting_scheme="sigma_sqrt")
    latents = torch.arange(64).reshape(2, 2, 1, 4, 4).float() / 100
    noise = torch.ones_like(latents)
    noisy = latents * 0.75 + noise * 0.25
    timesteps = torch.tensor([250.0, 500.0])
    embeds = [torch.ones(2, 8), torch.ones(3, 8) * 2]
    captured = {}

    def model_boundary(**kwargs):
        captured.update(kwargs)
        return kwargs["hidden_states"] * 2

    result = trainer.call_dit(
        args,
        Accelerator(cpu=True, mixed_precision="no"),
        model_boundary,
        latents,
        {"latents": latents, "vl_embed": embeds},
        noise,
        noisy,
        timesteps,
        torch.float32,
    )
    torch.testing.assert_close(result.pred, noisy * 2)
    torch.testing.assert_close(result.target, noise - latents)
    torch.testing.assert_close(captured["timestep"], torch.tensor([0.25, 0.5]))
    assert captured["img_shapes"] == [[(1, 2, 2)]]
    assert captured["txt_seq_lens"] == [2, 3]
    torch.testing.assert_close(captured["encoder_hidden_states"][0, 2], torch.zeros(8))
    if split:
        assert captured["encoder_hidden_states_mask"] is None
    else:
        assert captured["encoder_hidden_states_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert captured["hidden_states"].requires_grad
    scheduler = SimpleNamespace(sigmas=torch.tensor([0.25, 0.5]), timesteps=timesteps)
    loss, metrics = NetworkTrainer().compute_loss(args, result, timesteps, scheduler, torch.float32, torch.float32, 0)
    expected = ((noisy * 2 - (noise - latents)).square() * torch.tensor([16.0, 4.0]).reshape(2, 1, 1, 1, 1)).mean()
    torch.testing.assert_close(loss, expected)
    assert metrics == {}
    loss.backward()
    assert captured["hidden_states"].grad is not None


@pytest.mark.parametrize(
    "initial,steps,epoch,expected",
    [(False, 0, 2, False), (True, 0, None, True), (False, 4, 1, True), (False, 3, 2, True), (False, 3, 1, False)],
)
def test_sample_triggers_use_initial_then_step_or_epoch(initial, steps, epoch, expected):
    args = SimpleNamespace(sample_at_first=initial, sample_every_n_steps=4, sample_every_n_epochs=2)
    assert should_sample_images(args, steps, epoch) is expected


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def test_sampling_and_logs_preserve_observed_training_state(tmp_path, monkeypatch):
    model = QwenImageTransformer2DModel(
        in_channels=8,
        out_channels=2,
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=8,
        joint_attention_dim=8,
        axes_dims_rope=(2, 2, 4),
    )
    model.train()
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    optimizer = torch.optim.AdamW([parameter], lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
    before = copy.deepcopy(
        (
            model.state_dict(),
            parameter.grad,
            optimizer.state_dict(),
            scheduler.state_dict(),
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
    )
    trainer = QwenImageNetworkTrainer()
    trainer.is_edit = trainer.is_layered = False
    trainer._i2v_training = trainer._control_training = False
    trainer.default_guidance_scale = 1
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        output_name="adapter",
        sample_at_first=True,
        sample_every_n_steps=4,
        sample_every_n_epochs=2,
        sample_prompts="fixture",
        optimizer_type="AdamW",
    )
    transitions = []
    for name in ("switch_block_swap_for_inference", "switch_block_swap_for_training"):
        original = getattr(model, name)

        def record(original=original, name=name):
            transitions.append(name)
            original()

        monkeypatch.setattr(model, name, record)
    captured = {}

    def generation_boundary(
        accelerator, args, prompt, vae, dtype, transformer, shift, steps, width, height, generator, do_cfg, cfg, **kwargs
    ):
        assert not transformer.training and not torch.is_grad_enabled()
        captured.update(
            shift=shift, steps=steps, width=width, height=height, cfg=cfg, seed=generator.initial_seed(), prompt=prompt["prompt"]
        )
        torch.rand(3)  # observed boundary must restore this RNG consumption
        return torch.ones(1, 3, 1, height, width) * 0.5

    monkeypatch.setattr(trainer, "do_inference", generation_boundary)
    prompt = dict(
        prompt="scene",
        width=35,
        height=39,
        seed=42,
        sample_steps=3,
        cfg_scale=4.5,
        discrete_flow_shift=2.2,
        negative_prompt=" ",
        enum=0,
    )
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    trainer.sample_images(accelerator, args, None, 0, torch.nn.Identity(), model, [prompt], torch.float32)
    logs = trainer.generate_step_logs(args, 0.2, 0.3, scheduler, None)
    assert logs == {"loss/current": 0.2, "loss/average": 0.3, "lr/unet": 0.001}
    assert captured == dict(shift=2.2, steps=3, width=32, height=32, cfg=4.5, seed=42, prompt="scene")
    assert transitions == ["switch_block_swap_for_inference", "switch_block_swap_for_training"]
    assert model.training
    files = list((tmp_path / "sample").glob("*.png"))
    assert len(files) == 1
    np.testing.assert_array_equal(np.asarray(Image.open(files[0])), np.full((32, 32, 3), 127, dtype=np.uint8))
    after = (
        model.state_dict(),
        parameter.grad,
        optimizer.state_dict(),
        scheduler.state_dict(),
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    assert_state_equal(before, after)


def tiny_adapter():
    base = torch.nn.Sequential(QwenImageTransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4))
    base.requires_grad_(False)
    network = lora_qwen_image.create_arch_network(1, 2, 2, None, [], base)
    network.apply_to([], base, apply_text_encoder=False, apply_unet=True)
    return base, network


@pytest.mark.parametrize("module_name", ["lora_qwen_image", "loha", "lokr"])
def test_qwen_adapter_factories_have_gradients_and_frozen_base(module_name):
    import importlib
    import re

    module = importlib.import_module("musubi_tuner.networks." + module_name)
    base = torch.nn.Sequential(QwenImageTransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4))
    base[0].add_module("fixture_mod_projection", torch.nn.Linear(8, 8))
    base.requires_grad_(False)
    before = copy.deepcopy(base.state_dict())
    network = module.create_arch_network(1, 2, 2, None, [], base, rank_dropout="0.1", module_dropout="0.0")
    network.apply_to([], base, apply_text_encoder=False, apply_unet=True)
    assert network.unet_loras
    expected = {
        "lora_unet_" + name.replace(".", "_")
        for name, layer in base.named_modules()
        if isinstance(layer, torch.nn.Linear) and not re.fullmatch(r".*(_mod_).*", name)
    }
    assert {item.lora_name for item in network.unet_loras} == expected
    base[0].attn.to_q(torch.ones(1, 2, 8)).sum().backward()
    assert any(parameter.grad is not None for parameter in network.parameters())
    assert all(parameter.grad is None for parameter in base.parameters())
    assert_state_equal(before, base.state_dict())


@pytest.mark.parametrize("module_name", ["lora_qwen_image", "loha", "lokr"])
def test_build_network_infers_rank_and_loads_serialized_weights(tmp_path, module_name):
    import importlib
    from safetensors.torch import load_file
    from musubi_tuner.training.parser_common import setup_parser_common

    module = importlib.import_module("musubi_tuner.networks." + module_name)
    base = torch.nn.Sequential(QwenImageTransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4))
    base.requires_grad_(False)
    original = module.create_arch_network(1, 1, 3, None, [], base)
    original.apply_to([], base, apply_text_encoder=False, apply_unet=True)
    with torch.no_grad():
        for index, parameter in enumerate(original.parameters()):
            parameter.fill_((index + 1) / 100)
    path = tmp_path / (module_name + ".safetensors")
    original.save_weights(str(path), torch.float32, {})
    saved = load_file(str(path))
    args = setup_parser_common().parse_args(
        [
            "--network_module",
            module.__name__,
            "--network_weights",
            str(path),
            "--dim_from_weights",
            "--network_dim",
            "7",
            "--network_alpha",
            "9",
        ]
    )
    target = torch.nn.Sequential(QwenImageTransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4))
    target.requires_grad_(False)
    before = copy.deepcopy(target.state_dict())
    restored = NetworkTrainer()._build_network(args, Accelerator(cpu=True), target, None, torch.float32)
    assert restored.unet_loras
    assert all(adapter.lora_dim == 1 and adapter.alpha.item() == 3 for adapter in restored.unet_loras)
    assert_state_equal(saved, restored.state_dict())
    assert_state_equal(before, target.state_dict())
    assert args.resume is None


def test_sample_text_embedding_cache_and_negative_default(tmp_path, monkeypatch):
    trainer = QwenImageNetworkTrainer()
    path = tmp_path / "prompts.txt"
    path.write_text("scene\nscene --n negative\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(qwen_image_utils, "load_qwen2_5_vl", lambda *a, **kw: (object(), object()))

    def encoder_boundary(tokenizer, encoder, prompt):
        calls.append(prompt)
        return torch.ones(1, 4, 8), torch.tensor([[1, 1, 0, 0]])

    monkeypatch.setattr(qwen_image_utils, "get_qwen_prompt_embeds", encoder_boundary)
    result = trainer.process_sample_prompts(SimpleNamespace(fp8_vl=False, text_encoder="fixture"), Accelerator(cpu=True), str(path))
    assert calls == ["scene", " ", "negative"]
    assert result[0]["negative_prompt"] == " "
    assert result[0]["vl_embed"] is result[1]["vl_embed"]
    assert result[0]["vl_embed"].shape == (1, 2, 8)


def test_adapter_safetensors_precision_metadata_and_frozen_base(tmp_path):
    from safetensors import safe_open

    base, network = tiny_adapter()
    before = copy.deepcopy(base.state_dict())
    for adapter in network.unet_loras:
        torch.nn.init.constant_(adapter.lora_up.weight, 0.1)
    # Exercise actual attached adapter projection and gradients without a training loop.
    base[0].attn.to_q(torch.ones(1, 2, 8)).sum().backward()
    assert all(parameter.grad is None for parameter in base.parameters())
    assert any(parameter.grad is not None for parameter in network.parameters())
    path = tmp_path / "adapter.safetensors"
    network.save_weights(str(path), torch.bfloat16, {"ss_output_name": "fixture"})
    with safe_open(path, framework="pt") as saved:
        assert saved.metadata()["ss_output_name"] == "fixture"
        assert all(saved.get_tensor(key).dtype == torch.bfloat16 for key in saved.keys())
        tensors = {key: saved.get_tensor(key).float() for key in saved.keys()}
    for parameter in network.parameters():
        parameter.data.zero_()
    network.load_weights(str(path))
    for key, tensor in network.state_dict().items():
        expected_dtype = torch.int64 if key.endswith(".alpha") else torch.float32
        assert tensor.dtype == expected_dtype
        torch.testing.assert_close(tensor.float(), tensors[key])
    assert_state_equal(before, base.state_dict())


def test_qwen_inference_uses_original_scheduler_cfg_and_decode(monkeypatch):
    trainer = QwenImageNetworkTrainer()
    trainer.is_edit = trainer.is_layered = False
    calls = []

    class ModelBoundary:
        in_channels = 8

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return torch.ones_like(kwargs["hidden_states"]) * kwargs["encoder_hidden_states"].mean()

    class VaeBoundary(torch.nn.Module):
        def decode_to_pixels(self, latents):
            self.latents = latents.clone()
            return torch.ones(1, 3, 32, 48) * 0.5

    vae = VaeBoundary()
    prompt = {"vl_embed": torch.ones(1, 3, 8) * 2, "negative_vl_embed": torch.ones(1, 2, 8)}
    generator = torch.Generator().manual_seed(17)
    original = qwen_image_utils.prepare_latents(1, 2, 32, 48, torch.bfloat16, torch.device("cpu"), generator)
    scheduler = qwen_image_utils.get_scheduler(2.2)
    times, _ = qwen_image_utils.retrieve_timesteps(
        scheduler, 3, "cpu", sigmas=np.linspace(1, 1 / 3, 3), mu=qwen_image_utils.calculate_shift_qwen_image(original.shape[1])
    )
    scheduler.set_begin_index(0)
    expected = original
    for time in times:
        # CFG: 1 + 4.5 * (2 - 1), then original conditional-norm rescaling => 2.
        expected = scheduler.step(torch.ones_like(expected) * 2, time, expected, return_dict=False)[0]
    pixels = trainer.do_inference(
        Accelerator(cpu=True),
        SimpleNamespace(is_layered=False),
        prompt,
        vae,
        torch.bfloat16,
        ModelBoundary(),
        2.2,
        3,
        48,
        32,
        torch.Generator().manual_seed(17),
        True,
        4.5,
    )
    assert len(calls) == 6
    for index, call in enumerate(calls):
        assert call["txt_seq_lens"] == ([3] if index % 2 == 0 else [2])
        assert call["img_shapes"] == [(1, 2, 3)]
        assert call["guidance"] is None and call["encoder_hidden_states_mask"] is None
        torch.testing.assert_close(call["timestep"], times[index // 2].expand(1).to(torch.bfloat16) / 1000)
    torch.testing.assert_close(vae.latents, qwen_image_utils.unpack_latents(expected, 32, 48))
    torch.testing.assert_close(pixels, torch.full((1, 3, 1, 32, 48), 0.5))


def test_actual_accelerator_hooks_restore_adapter_optimizer_scheduler_rng(tmp_path):
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    base, network = tiny_adapter()
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    base, network, optimizer, scheduler = accelerator.prepare(base, network, optimizer, scheduler)
    # Initialize optimizer state with a tiny explicit unit update, never trainer.train/_run_training_loop.
    for parameter in network.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    args = SimpleNamespace(resume=None, resume_from_huggingface=False)
    trainer = NetworkTrainer()
    trainer._register_hooks_and_resume(args, accelerator, network)
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    before = copy.deepcopy(
        (
            network.state_dict(),
            optimizer.state_dict(),
            scheduler.state_dict(),
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
    )
    state = tmp_path / "state"
    accelerator.save_state(str(state))
    from safetensors.torch import load_file

    serialized = list(state.glob("*.safetensors"))
    assert len(serialized) == 1
    assert set(load_file(str(serialized[0]))) == set(network.state_dict())
    for parameter in network.parameters():
        parameter.data.add_(5)
    optimizer.param_groups[0]["lr"] = 0.99
    scheduler.step()
    random.random(), np.random.rand(), torch.rand(1)
    args.resume = str(state)
    assert trainer.resume_from_local_or_hf_if_specified(accelerator, args)
    after = (
        network.state_dict(),
        optimizer.state_dict(),
        scheduler.state_dict(),
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    assert_state_equal(before, after)
    accelerator.free_memory()


def test_names_and_independent_retention_windows(tmp_path):
    args = SimpleNamespace(
        output_dir=str(tmp_path),
        output_name="qwen",
        save_every_n_steps=200,
        save_last_n_steps=1000,
        save_last_n_steps_state=400,
        save_state_to_huggingface=False,
        save_every_n_epochs=2,
        save_last_n_epochs=3,
        save_last_n_epochs_state=1,
    )
    assert train_utils.get_step_ckpt_name("qwen", 200) == "qwen-step00000200.safetensors"
    assert train_utils.get_epoch_ckpt_name("qwen", 2) == "qwen-000002.safetensors"
    assert train_utils.get_last_ckpt_name("qwen") == "qwen.safetensors"
    assert [train_utils.get_remove_step_no(args, n) for n in (1000, 1200, 1400, 1600)] == [None, 0, 200, 400]
    assert train_utils.get_remove_epoch_no(args, 8) == 2
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    for step in (200, 400, 600, 800):
        train_utils.save_and_remove_state_stepwise(args, accelerator, step)
    assert not (tmp_path / "qwen-step00000200-state").exists()
    assert (tmp_path / "qwen-step00000400-state").exists()
    args.save_last_n_steps_state = 0  # existing truthy fallback to checkpoint window
    train_utils.save_and_remove_state_stepwise(args, accelerator, 1000)
    assert (tmp_path / "qwen-step00000400-state").exists()
    train_utils.save_and_remove_state_on_epoch_end(args, accelerator, 2)
    train_utils.save_and_remove_state_on_epoch_end(args, accelerator, 4)
    assert not (tmp_path / "qwen-000002-state").exists()
    assert (tmp_path / "qwen-000004-state").exists()
    train_utils.save_state_on_train_end(args, accelerator)
    assert (tmp_path / "qwen-state").is_dir()
