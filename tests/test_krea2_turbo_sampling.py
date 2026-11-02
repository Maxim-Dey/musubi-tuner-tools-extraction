from contextlib import nullcontext
from pathlib import Path
import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser
from musubi_tuner.krea2 import turbo_sampling
from musubi_tuner.networks import lora_krea2
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common


class TinyDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(2, 2, bias=False)
        self.first.register_buffer("scale_weight", torch.tensor(1.25))
        self.unused_dropout = torch.nn.Dropout(.5)
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = True
        self.offloader = None
        with torch.no_grad():
            self.first.weight.fill_(7.)  # actual RAW includes an already merged base adapter

    def forward(self, x):
        return self.first(x)


def fixture_models(monkeypatch):
    model = TinyDiT().requires_grad_(False)
    model.unused_dropout.eval()
    network = lora_krea2.create_arch_network(1., 1, 1, None, [], model, neuron_dropout=0.)
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    with torch.no_grad():
        for module in network.unet_loras:
            module.lora_down.weight.fill_(.5)
            module.lora_up.weight.fill_(.25)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.01)
    loss = model(torch.tensor([[1., 2.]])).square().mean()
    loss.backward()
    optimizer.step()  # populated optimizer state and existing gradients must survive
    accelerator = SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda value: value,
                                  process_index=0, num_processes=1, reduce=lambda value, reduction: value)
    trainer = Krea2NetworkTrainer()
    loads = []
    def load(path, **kwargs):
        loads.append((path, kwargs))
        random.random(), np.random.rand(), torch.rand(1)  # loader RNG belongs to the sample transaction
        return {"first.weight": torch.full((2, 2), 3., dtype=torch.bfloat16), "first.scale_weight": torch.tensor(9.)}
    monkeypatch.setattr(turbo_sampling, "load_sampling_state", load)
    return trainer, model, network, optimizer, accelerator, loads


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng(before):
    assert before[0] == random.getstate()
    now = np.random.get_state()
    assert before[1][0] == now[0] and np.array_equal(before[1][1], now[1]) and before[1][2:] == now[2:]
    assert torch.equal(before[2], torch.get_rng_state())


def assert_optimizer(left, right):
    assert left["param_groups"] == right["param_groups"]
    for key, values in left["state"].items():
        for name, value in values.items():
            assert torch.equal(value, right["state"][key][name])


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("failure", [None, "load", "keys", "shape", "partial_swap", "inference"])
def test_current_lora_and_exact_raw_rollback(monkeypatch, cached, failure):
    trainer, model, network, optimizer, accelerator, loads = fixture_models(monkeypatch)
    args = SimpleNamespace(turbo_dit="turbo.safetensors", turbo_dit_cache=cached, fp8_scaled=False)
    raw = {key: value.detach().clone() for key, value in turbo_sampling.live_tensors(model).items()}
    adapter = copy.deepcopy(network.state_dict())
    gradients = [p.grad.clone() for p in network.parameters()]
    identities = [id(p) for p in list(model.parameters()) + list(network.parameters())]
    requires_grad = [p.requires_grad for p in list(model.parameters()) + list(network.parameters())]
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    modes = [m.training for root in (model, network) for m in root.modules()]
    before_rng = rng_state()
    loader = turbo_sampling.load_sampling_state
    if failure in ("load", "keys", "shape"):
        def faulty_load(*values, **kwargs):
            state = loader(*values, **kwargs)
            if failure == "load":
                raise OSError("injected loader failure")
            if failure == "keys":
                state.pop("first.weight")
            else:
                state["first.weight"] = torch.zeros(3, 2)
            # Full key/shape prevalidation must leave actual RAW untouched.
            assert all(torch.equal(value, raw[key]) for key, value in turbo_sampling.live_tensors(model).items())
            return state
        monkeypatch.setattr(turbo_sampling, "load_sampling_state", faulty_load)
    if failure == "partial_swap":
        if cached:
            original = turbo_sampling.overwrite
            calls = 0
            def partial(tensors, state):
                nonlocal calls
                calls += 1
                if calls == 1:
                    tensors["first.weight"].data.copy_(state["first.weight"])
                    raise RuntimeError("injected midway copy failure")
                return original(tensors, state)
            monkeypatch.setattr(turbo_sampling, "overwrite", partial)
        else:
            original = turbo_sampling.release
            calls = 0
            def partial(tensors):
                nonlocal calls
                calls += 1
                original(tensors)
                if calls == 1:
                    raise RuntimeError("injected release failure")
            monkeypatch.setattr(turbo_sampling, "release", partial)
    caught = None
    try:
        with turbo_sampling.sample_on_turbo(trainer, args, accelerator, model, network):
            assert not torch.is_grad_enabled() and not model.training and not network.training
            assert not model.gradient_checkpointing and not model.activation_cpu_offloading
            assert torch.equal(model.first.weight, torch.full((2, 2), 3.))
            x = torch.tensor([[1., 2.]])
            module = network.unet_loras[0]
            expected = torch.nn.functional.linear(x, torch.full((2, 2), 3.)) + module.lora_up(module.lora_down(x))
            assert torch.equal(model(x), expected)  # same current trained adapter, no snapshot/frozen trainee
            if failure == "inference":
                raise RuntimeError("injected denoising failure")
    except (RuntimeError, ValueError, OSError) as error:
        caught = error
    assert (caught is not None) == (failure is not None)
    assert all(torch.equal(value, raw[key]) for key, value in turbo_sampling.live_tensors(model).items())
    assert [id(p) for p in list(model.parameters()) + list(network.parameters())] == identities
    assert [p.requires_grad for p in list(model.parameters()) + list(network.parameters())] == requires_grad
    assert modes == [m.training for root in (model, network) for m in root.modules()]
    assert model.gradient_checkpointing and model.activation_cpu_offloading and not trainer._turbo_sampling_active
    for key, value in network.state_dict().items():
        assert torch.equal(value, adapter[key])
    assert all(torch.equal(p.grad, gradient) for p, gradient in zip(network.parameters(), gradients))
    assert_optimizer(optimizer_before, optimizer.state_dict())
    assert_rng(before_rng)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("fail_sample", [False, True])
def test_second_sample_uses_updated_adapter_and_next_training_step_matches(monkeypatch, cached, fail_sample):
    from musubi_tuner.training.krea2_validation import run_validation

    _, baseline, baseline_network, baseline_optimizer, _, _ = fixture_models(monkeypatch)
    trainer, model, network, optimizer, accelerator, loads = fixture_models(monkeypatch)
    args = SimpleNamespace(turbo_dit="turbo", turbo_dit_cache=cached, fp8_scaled=False,
                           val_level_noise_n=2, val_seed_noise_n=1, val_seed_noise=42, weighting_scheme="none")
    x = torch.tensor([[1., 2.]])
    outputs = []
    inputs = SimpleNamespace(records=[SimpleNamespace(image_sha256="a")], validate_unchanged=lambda: None,
                             load_batch=lambda record: {"latents":torch.ones(1,2,1,1,1)})
    def val_forward(current_model):
        def call(cfg, acc, transformer, latents, batch, noise, noisy, times, dtype):
            assert torch.equal(current_model.first.weight, torch.full((2,2),7.))
            return SimpleNamespace(pred=current_model(x).reshape(latents.shape), target=noise-latents)
        return SimpleNamespace(call_dit=call)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="sample exception") if fail_sample else nullcontext():
            with turbo_sampling.sample_on_turbo(trainer, args, accelerator, model, network):
                outputs.append(model(x).clone())
                if fail_sample:
                    raise RuntimeError("sample exception")
        measured = run_validation(val_forward(model), args, accelerator, model, network, inputs, torch.float32)
        reference = run_validation(val_forward(baseline), args, accelerator, baseline, baseline_network, inputs, torch.float32)
        assert measured == reference
        optimizer.zero_grad()
        model(x).square().mean().backward()
        optimizer.step()
        baseline_optimizer.zero_grad()
        baseline(x).square().mean().backward()
        baseline_optimizer.step()
        assert torch.equal(model(x), baseline(x))  # immediate RAW validation/inference state
        for key, value in network.state_dict().items():
            assert torch.equal(value, baseline_network.state_dict()[key])
        assert_optimizer(optimizer.state_dict(), baseline_optimizer.state_dict())
    assert not torch.equal(outputs[0], outputs[1])
    assert len(loads) == (1 if cached else 2)
    assert trainer._turbo_raw_state is not None if cached else trainer._turbo_raw_state is None


def test_compiled_name_normalization_and_fp8_scale_buffers():
    model = torch.nn.Module()
    model.block = torch.nn.Module()
    model.block._orig_mod = torch.nn.Linear(2, 2, bias=False)
    model.block._orig_mod.register_buffer("scale_weight", torch.tensor(.125))
    tensors = turbo_sampling.live_tensors(model)
    assert set(tensors) == {"block.weight", "block.scale_weight"}


def parser():
    return krea2_setup_parser(setup_parser_common())


@pytest.mark.parametrize("extra, message", [(["--turbo_dit_cache"], "requires turbo_dit"),
    (["--turbo_dit", "missing.safetensors"], "does not exist"),
    (["--blocks_to_swap", "1"], "blocks_to_swap"), (["--convrot_int8"], "convrot_int8")])
def test_turbo_options_fail_before_model_loading(tmp_path, extra, message):
    for name in ("dataset.toml", "raw.safetensors", "turbo.safetensors"):
        (tmp_path/name).touch()
    argv = ["--dataset_config", str(tmp_path/"dataset.toml"), "--dit", str(tmp_path/"raw.safetensors"),
            "--network_module", "networks.lora_krea2", "--output_dir", str(tmp_path), "--output_name", "run", "--sdpa"]
    if message in ("blocks_to_swap", "convrot_int8"):
        argv += ["--turbo_dit", str(tmp_path/"turbo.safetensors")]
    args = parser().parse_args(argv + extra)
    with pytest.raises(ValueError, match=message):
        Krea2NetworkTrainer().validate_training_args(args)


def test_toml_turbo_path_resolution_and_defaults(tmp_path):
    file = tmp_path/"train.toml"
    file.write_text('experiment_mode=true\ndataset_config="dataset.toml"\ndit="raw.safetensors"\nturbo_dit="turbo.safetensors"\nturbo_dit_cache=true\n')
    p = parser()
    argv = ["--config_file", str(file)]
    args = read_config_from_file(p.parse_args(argv), p, argv)
    assert args.turbo_dit == str(tmp_path/"turbo.safetensors") and args.turbo_dit_cache is True
    assert p.parse_args(["--dataset_config", "a", "--dit", "b"]).turbo_dit is None


def test_sampling_loader_reuses_existing_bf16_and_fp8_paths(monkeypatch):
    seen = []
    monkeypatch.setattr(turbo_sampling, "load_safetensors", lambda *a, **kw: seen.append((a, kw)) or {})
    monkeypatch.setattr(turbo_sampling, "load_safetensors_with_lora_and_fp8", lambda **kw: seen.append(((), kw)) or {})
    turbo_sampling.load_sampling_state("turbo", fp8_scaled=False, calc_device="cpu", result_device="cpu")
    assert seen[-1][1]["dtype"] == torch.bfloat16 and seen[-1][1]["disable_mmap"]
    turbo_sampling.load_sampling_state("turbo", fp8_scaled=True, calc_device="cpu", result_device="cpu")
    assert seen[-1][1]["fp8_optimization"] and seen[-1][1]["lora_weights_list"] is None


@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
def test_before_hook_failure_still_runs_cleanup_in_actual_loop(tmp_path, monkeypatch, error):
    from test_krea2_experiment_state import ToyTrainer, run_loop
    calls = []
    def before(self, *args):
        calls.append("before")
        raise error("injected entry failure")
    def after(self, *args):
        calls.append("after")
    monkeypatch.setattr(ToyTrainer, "on_before_sample_images", before)
    monkeypatch.setattr(ToyTrainer, "on_after_sample_images", after)
    with pytest.raises(error, match="entry failure"):
        run_loop(tmp_path, sample_interval=2)
    assert calls == ["before", "after"]


@pytest.mark.parametrize("explicit", [False, True])
def test_turbo_prompt_defaults_honor_explicit_fields(tmp_path, monkeypatch, explicit):
    from musubi_tuner import krea2_train_network as module
    prompt = tmp_path/"prompts.txt"
    prompt.write_text("subject" + (" --s 17 --l 4.5" if explicit else "") + "\n")
    monkeypatch.setattr(module.krea2_utils, "load_krea2_text_encoder", lambda *a, **kw: object())
    monkeypatch.setattr(module.krea2_utils, "get_krea2_prompt_embeds",
                        lambda *a: (torch.ones(1, 2, 2, 4), torch.ones(1, 2, dtype=torch.bool)))
    args = SimpleNamespace(text_encoder="encoder", turbo_dit="turbo", tokenizer_path=None)
    accelerator = SimpleNamespace(device=torch.device("cpu"))
    parameters = Krea2NetworkTrainer().process_sample_prompts(args, accelerator, str(prompt))
    assert parameters[0]["sample_steps"] == (17 if explicit else 8)
    assert parameters[0]["cfg_scale"] == (4.5 if explicit else 1.)


@pytest.mark.parametrize("active", [False, True])
def test_turbo_schedule_is_fixed_only_while_swapped(tmp_path, monkeypatch, active):
    from musubi_tuner import krea2_train_network as module
    seen = []
    original = module.krea2_sampling.timesteps
    def schedule(*values, **kw):
        seen.append(kw)
        return original(*values, **kw)
    monkeypatch.setattr(module.krea2_sampling, "timesteps", schedule)
    trainer = Krea2NetworkTrainer()
    trainer._turbo_sampling_active = active
    class Model(torch.nn.Module):
        config = SimpleNamespace(patch=2, channels=1)
        def forward(self, img, **kw):
            return torch.zeros_like(img)
    class Vae(torch.nn.Module):
        dtype = torch.bfloat16
        def decode_to_pixels(self, value):
            return torch.zeros(1, 3, 16, 16)
    trainer.do_inference(SimpleNamespace(device=torch.device("cpu")), SimpleNamespace(turbo_dit="configured"),
                         {"krea2_vl_embed":torch.ones(2,2,4)}, Vae(), torch.bfloat16, Model(),
                         1, 16, 16, torch.Generator().manual_seed(42), False, 1.)
    assert seen[0].get("mu") == (1.15 if active else None)


@pytest.mark.parametrize("cached", [False, True])
def test_fp8_parameter_and_scale_buffer_rollback(monkeypatch, cached):
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("This Torch build has no float8 dtype")
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(2, 2).to(torch.float8_e4m3fn), requires_grad=False)
    model.register_buffer("scale_weight", torch.tensor(.125))
    network = torch.nn.Linear(2, 2)
    identity = id(model.weight)
    raw_bytes = model.weight.view(torch.uint8).clone()
    trainer = Krea2NetworkTrainer()
    args = SimpleNamespace(turbo_dit="turbo", turbo_dit_cache=cached, fp8_scaled=True)
    monkeypatch.setattr(turbo_sampling, "load_sampling_state", lambda *a, **kw: {
        "weight":torch.full((2, 2), 3.).to(torch.float8_e4m3fn), "scale_weight":torch.tensor(.25)})
    with pytest.raises(RuntimeError, match="after fp8 swap"):
        with turbo_sampling.sample_on_turbo(trainer, args, SimpleNamespace(device="cpu", unwrap_model=lambda x:x), model, network):
            assert model.weight.dtype == torch.float8_e4m3fn and model.scale_weight.item() == .25
            raise RuntimeError("after fp8 swap")
    assert id(model.weight) == identity and model.weight.dtype == torch.float8_e4m3fn
    assert torch.equal(model.weight.view(torch.uint8), raw_bytes) and model.scale_weight.item() == .125


def test_turbo_options_preserve_existing_frozen_manifest_and_cache_bytes(tmp_path):
    from test_krea2_validation_inputs import _experiment, _populate, AcceleratorBoundary
    from musubi_tuner.training.krea2_validation_inputs import preflight_validation_inputs, prepare_validation_caches

    args = _experiment(tmp_path)
    inputs = preflight_validation_inputs(args)
    _populate(inputs)
    prepare_validation_caches(args, inputs, AcceleratorBoundary())
    paths = [inputs.manifest_path] + [path for record in inputs.source_records
                                     for path in (Path(record.latent_cache_path), Path(record.text_cache_path))]
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    args.turbo_dit = str(tmp_path/"turbo.safetensors")
    Path(args.turbo_dit).write_bytes(b"sample-only Turbo checkpoint")
    args.turbo_dit_cache = True
    new = preflight_validation_inputs(args)
    prepare_validation_caches(args, new, AcceleratorBoundary())
    assert new.manifest == inputs.manifest
    assert before == [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]


def test_hooks_restore_vae_modes_even_when_sampling_fails(monkeypatch):
    trainer, model, network, _, accelerator, _ = fixture_models(monkeypatch)
    args = SimpleNamespace(turbo_dit="turbo", turbo_dit_cache=False, fp8_scaled=False)
    vae = torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Linear(2, 2))
    vae[0].eval()
    modes = [m.training for m in vae.modules()]
    hook_args = (accelerator, args, None, 2, vae, model, network, [], torch.float32)
    trainer.on_before_sample_images(*hook_args)
    try:
        vae.eval()
        raise RuntimeError("decode failed")
    except RuntimeError:
        trainer.on_after_sample_images(*hook_args)
    assert [m.training for m in vae.modules()] == modes
    assert torch.equal(model.first.weight, torch.full((2, 2), 7.))
