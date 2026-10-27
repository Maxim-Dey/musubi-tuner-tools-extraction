"""Real CPU contracts for helpers retained from removed model pipelines."""

import copy

import torch
from safetensors.torch import load_file, save_file

from musubi_tuner.networks import lora_krea2
from musubi_tuner.qwen_image.qwen_image_utils import convert_comfyui_state_dict
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8


def test_vae_key_conversion_preserves_tensor_identity():
    tensors = {
        "conv1.bias": torch.arange(16.0),
        "encoder.downsamples.5.time_conv.weight": torch.ones(2, 2, 1, 1, 1),
        "decoder.head.2.weight": torch.ones(3, 2, 1, 3, 3),
    }
    converted = convert_comfyui_state_dict(tensors)
    assert converted["quant_conv.bias"] is tensors["conv1.bias"]
    assert converted["encoder.down_blocks.5.time_conv.weight"] is tensors["encoder.downsamples.5.time_conv.weight"]
    assert converted["decoder.conv_out.weight"] is tensors["decoder.head.2.weight"]
    assert convert_comfyui_state_dict(converted) is converted


def test_krea2_adapter_save_load_and_cpu_step(tmp_path):
    torch.manual_seed(17)
    base = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 4))
    original = copy.deepcopy(base)
    original_state = copy.deepcopy(base.state_dict())
    base.requires_grad_(False)
    network = lora_krea2.create_arch_network(1.0, 2, 2, None, None, base)
    network.apply_to(None, base, apply_text_encoder=False, apply_unet=True)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
    inputs = torch.randn(3, 4)
    loss = base(inputs).square().mean()
    loss.backward()
    optimizer.step()
    for name, value in original_state.items():
        torch.testing.assert_close(base.state_dict()[name], value, rtol=0, atol=0)
    assert any(param.grad is not None for param in network.parameters())

    path = str(tmp_path / "adapter.safetensors")
    network.save_weights(path, torch.float32, {"modelspec.architecture": "Krea-2/lora"})
    state = load_file(path)
    assert state and all(name.startswith("lora_unet_") for name in state)
    restored = lora_krea2.create_arch_network_from_weights(1.0, state, unet=original)
    restored.apply_to(None, original, apply_text_encoder=False, apply_unet=True)
    restored.load_state_dict(state, strict=True)
    network.eval()
    restored.eval()
    torch.testing.assert_close(base(inputs), original(inputs), rtol=0, atol=0)


def test_base_adapter_streaming_merge_preserves_scale(tmp_path):
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    path = str(tmp_path / "base.safetensors")
    save_file({"layer.weight": weight, "layer.bias": torch.zeros(3)}, path)
    down = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    up = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    adapter = {
        "lora_unet_layer.lora_down.weight": down,
        "lora_unet_layer.lora_up.weight": up,
        "lora_unet_layer.alpha": torch.tensor(4.0),
    }
    merged = load_safetensors_with_lora_and_fp8(path, [adapter], [0.5], False, torch.device("cpu"), dit_weight_dtype=torch.float32)
    torch.testing.assert_close(merged["layer.weight"], weight + up @ down, rtol=0, atol=0)
    torch.testing.assert_close(merged["layer.bias"], torch.zeros(3), rtol=0, atol=0)
