"""Actual offline command imports/help and adapter factories under an exclusion guard."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ("qwen_image_train_network", "qwen_image_cache_latents", "qwen_image_cache_text_encoder_outputs")
# Exact T004 removal inventory, condensed only at wholly excluded package boundaries.
EXCLUDED = (
    "musubi_tuner.caption_images_by_qwen_vl",
    "musubi_tuner.convert_lora",
    "musubi_tuner.dataset.audio_utils",
    "musubi_tuner.flux",
    "musubi_tuner.flux_2",
    "musubi_tuner.flux_2_cache_latents",
    "musubi_tuner.flux_2_cache_text_encoder_outputs",
    "musubi_tuner.flux_2_generate_image",
    "musubi_tuner.flux_2_train_network",
    "musubi_tuner.flux_2_train_network_self_flow",
    "musubi_tuner.flux_kontext_cache_latents",
    "musubi_tuner.flux_kontext_cache_text_encoder_outputs",
    "musubi_tuner.flux_kontext_generate_image",
    "musubi_tuner.flux_kontext_train_network",
    "musubi_tuner.fpack_cache_latents",
    "musubi_tuner.fpack_cache_text_encoder_outputs",
    "musubi_tuner.fpack_generate_video",
    "musubi_tuner.fpack_train_network",
    "musubi_tuner.frame_pack",
    "musubi_tuner.gui",
    "musubi_tuner.hidream_o1",
    "musubi_tuner.hidream_o1_cache_pixel",
    "musubi_tuner.hidream_o1_cache_text_encoder_outputs",
    "musubi_tuner.hidream_o1_generate_image",
    "musubi_tuner.hidream_o1_train",
    "musubi_tuner.hidream_o1_train_network",
    "musubi_tuner.hunyuan_model",
    "musubi_tuner.hunyuan_video_1_5",
    "musubi_tuner.hv_1_5_cache_latents",
    "musubi_tuner.hv_1_5_cache_text_encoder_outputs",
    "musubi_tuner.hv_1_5_generate_video",
    "musubi_tuner.hv_1_5_train_network",
    "musubi_tuner.hv_generate_video",
    "musubi_tuner.hv_train",
    "musubi_tuner.hv_train_network",
    "musubi_tuner.ideogram4",
    "musubi_tuner.ideogram4_cache_latents",
    "musubi_tuner.ideogram4_cache_text_encoder_outputs",
    "musubi_tuner.ideogram4_generate_image",
    "musubi_tuner.ideogram4_train_network",
    "musubi_tuner.kandinsky5",
    "musubi_tuner.kandinsky5_cache_latents",
    "musubi_tuner.kandinsky5_cache_text_encoder_outputs",
    "musubi_tuner.kandinsky5_generate_video",
    "musubi_tuner.kandinsky5_train_network",
    "musubi_tuner.krea2",
    "musubi_tuner.krea2_cache_latents",
    "musubi_tuner.krea2_cache_text_encoder_outputs",
    "musubi_tuner.krea2_generate_image",
    "musubi_tuner.krea2_train_network",
    "musubi_tuner.lora_post_hoc_ema",
    "musubi_tuner.merge_lora",
    "musubi_tuner.minimax_h3",
    "musubi_tuner.minimax_h3_cache_latents",
    "musubi_tuner.minimax_h3_cache_text_encoder_outputs",
    "musubi_tuner.minimax_h3_generate_video",
    "musubi_tuner.minimax_h3_train_network",
    "musubi_tuner.modules.adafactor_fused",
    "musubi_tuner.modules.attention",
    "musubi_tuner.modules.comfy_quant_utils",
    "musubi_tuner.modules.convrot_int8_kernels",
    "musubi_tuner.modules.convrot_int8_utils",
    "musubi_tuner.modules.nvfp4_utils",
    "musubi_tuner.modules.unet_causal_3d_blocks",
    "musubi_tuner.networks.convert_hunyuan_video_1_5_lora_to_comfy",
    "musubi_tuner.networks.convert_z_image_lora_to_comfy",
    "musubi_tuner.networks.lora_flux",
    "musubi_tuner.networks.lora_flux_2",
    "musubi_tuner.networks.lora_framepack",
    "musubi_tuner.networks.lora_hidream_o1",
    "musubi_tuner.networks.lora_hv_1_5",
    "musubi_tuner.networks.lora_ideogram4",
    "musubi_tuner.networks.lora_kandinsky",
    "musubi_tuner.networks.lora_krea2",
    "musubi_tuner.networks.lora_minimax_h3",
    "musubi_tuner.networks.lora_wan",
    "musubi_tuner.networks.lora_zimage",
    "musubi_tuner.qwen_extract_lora",
    "musubi_tuner.qwen_image_generate_image",
    "musubi_tuner.qwen_image_train",
    "musubi_tuner.training.audio_loss",
    "musubi_tuner.wan",
    "musubi_tuner.wan_cache_latents",
    "musubi_tuner.wan_cache_text_encoder_outputs",
    "musubi_tuner.wan_generate_video",
    "musubi_tuner.wan_train_network",
    "musubi_tuner.zimage",
    "musubi_tuner.zimage_cache_latents",
    "musubi_tuner.zimage_cache_text_encoder_outputs",
    "musubi_tuner.zimage_generate_image",
    "musubi_tuner.zimage_train",
    "musubi_tuner.zimage_train_network",
)
RETAINED = (
    "musubi_tuner",
    "musubi_tuner.cache_latents",
    "musubi_tuner.cache_text_encoder_outputs",
    "musubi_tuner.dataset",
    "musubi_tuner.dataset.architectures",
    "musubi_tuner.dataset.bucket",
    "musubi_tuner.dataset.cache_io",
    "musubi_tuner.dataset.config_utils",
    "musubi_tuner.dataset.datasources",
    "musubi_tuner.dataset.image_video_dataset",
    "musubi_tuner.dataset.media_utils",
    "musubi_tuner.modules",
    "musubi_tuner.modules.custom_offloading_utils",
    "musubi_tuner.modules.fp8_optimization_utils",
    "musubi_tuner.modules.lr_schedulers",
    "musubi_tuner.modules.scheduling_flow_match_discrete",
    "musubi_tuner.networks",
    "musubi_tuner.networks.loha",
    "musubi_tuner.networks.lokr",
    "musubi_tuner.networks.lora",
    "musubi_tuner.networks.lora_qwen_image",
    "musubi_tuner.networks.network_arch",
    "musubi_tuner.qwen_image",
    "musubi_tuner.qwen_image.qwen_image_autoencoder_kl",
    "musubi_tuner.qwen_image.qwen_image_model",
    "musubi_tuner.qwen_image.qwen_image_modules",
    "musubi_tuner.qwen_image.qwen_image_utils",
    "musubi_tuner.qwen_image_cache_latents",
    "musubi_tuner.qwen_image_cache_text_encoder_outputs",
    "musubi_tuner.qwen_image_train_network",
    "musubi_tuner.training",
    "musubi_tuner.training.accelerator_setup",
    "musubi_tuner.training.parser_common",
    "musubi_tuner.training.sampling_prompts",
    "musubi_tuner.training.timesteps",
    "musubi_tuner.training.trainer_base",
    "musubi_tuner.utils",
    "musubi_tuner.utils.device_utils",
    "musubi_tuner.utils.huggingface_utils",
    "musubi_tuner.utils.image_utils",
    "musubi_tuner.utils.lora_utils",
    "musubi_tuner.utils.model_utils",
    "musubi_tuner.utils.safetensors_utils",
    "musubi_tuner.utils.sai_model_spec",
    "musubi_tuner.utils.train_utils",
)


def run_guarded(program):
    guard = """
import importlib.abc
import sys
class ExcludedImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in EXCLUDED):
            raise AssertionError('retained path tried to import excluded owner: ' + fullname)
sys.meta_path.insert(0, ExcludedImportGuard())
import torch
"""
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "CUDA_VISIBLE_DEVICES": "-1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
    }
    result = subprocess.run(
        [sys.executable, "-B", "-c", "EXCLUDED=" + repr(EXCLUDED) + "\n" + guard + program],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize("module", [False, True])
def test_real_help(command, module):
    program = "import runpy\nsys.argv=['" + command + "', '--help']\n"
    program += (
        "runpy.run_module(" + repr("musubi_tuner." + command) + ",run_name='__main__')"
        if module
        else "runpy.run_path(" + repr(str(ROOT / (command + ".py"))) + ",run_name='__main__')"
    )
    output = run_guarded(program)
    assert "--dataset_config" in output and "--model_version" in output
    assert "--network_module" in output if command.endswith("train_network") else "--skip_existing" in output


def test_real_retained_imports_and_short_qualified_factories():
    program = """
import importlib
from pathlib import Path
for name in RETAINED:
    importlib.import_module(name)
import musubi_tuner
sys.path.insert(0, str(Path(musubi_tuner.__file__).parent))
from musubi_tuner.qwen_image.qwen_image_model import QwenImageTransformerBlock
for prefix in ('networks.', 'musubi_tuner.networks.'):
    for name in ('lora_qwen_image', 'loha', 'lokr'):
        module = importlib.import_module(prefix + name)
        model = torch.nn.Sequential(QwenImageTransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4))
        network = module.create_arch_network(1, 2, 2, None, [], model)
        assert network.unet_loras
print('All retained imports and six actual factories passed')
"""
    output = run_guarded("RETAINED=" + repr(RETAINED) + "\n" + program)
    assert "six actual factories passed" in output
