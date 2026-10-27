"""LoRA training for Krea 2 (K2). Phase 4: minimal trainable loop.

Implements the architecture-specific hooks of the shared NetworkTrainer for K2:
build/load the single-stream MMDiT, reuse the Qwen-Image VAE, and run flow-matching
in K2's token space (replicating sampling.prepare batched, with varlen text padding).

Wired: bf16 + gradient checkpointing, sample generation during training (text-to-image,
optional CFG), dynamic scaled fp8 for the DiT (--fp8_base --fp8_scaled), block swap
(--blocks_to_swap, CPU offloading of the main blocks), and torch.compile of the main
blocks (--compile).
"""

import argparse
import ast
import gc
import math
import os
import re
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from safetensors.torch import load_file
from tqdm import tqdm
from einops import rearrange, repeat

from musubi_tuner.dataset.architectures import ARCHITECTURE_KREA2, ARCHITECTURE_KREA2_FULL
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer
from musubi_tuner.training.accelerator_setup import clean_memory_on_device
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.training.parser_common import setup_parser_common, read_config_from_file, config_error
from musubi_tuner.krea2 import krea2_utils
from musubi_tuner.krea2 import krea2_sampling
from musubi_tuner.qwen_image import qwen_image_utils
from musubi_tuner.utils import model_utils

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Krea2NetworkTrainer(NetworkTrainer):
    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_KREA2

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_KREA2_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.bfloat16
        # self.blocks_to_swap is set by the base trainer (handle_model_specific_args runs first).
        # K2 fp8 supports only the scaled (dynamic) path; plain --fp8_base alone would cast the
        # whole DiT (incl. norms) to fp8, which breaks. Require --fp8_scaled with --fp8_base.
        if args.fp8_base and not args.fp8_scaled:
            config_error(args, "fp8_base", "Krea2 requires scaled fp8; also set fp8_scaled=true.")
        # ConvRot int8 is an alternative base-weight quantization; one quantization at a time.
        if args.convrot_int8 and (args.fp8_base or args.fp8_scaled):
            config_error(args, "convrot_int8", "cannot be combined with fp8_base/fp8_scaled; choose one quantization.")
        if args.convrot_int8_bwd == "int8" and not args.convrot_int8:
            config_error(args, "convrot_int8_bwd", "int8 backward requires convrot_int8=true; enable it or use bf16.")

    def validate_training_args(self, args):
        super().validate_training_args(args)
        from musubi_tuner.modules import attention
        from musubi_tuner.modules.convrot_int8_utils import HAS_TRITON

        for key, backend in (
            ("flash_attn", attention.flash_attn),
            ("flash3", attention.flash_attn_interface),
            ("xformers", attention.xops),
        ):
            if getattr(args, key) and backend is None:
                config_error(args, key, "the requested attention dependency is unavailable; install it or choose sdpa.")
        if args.convrot_int8_bwd == "int8" and not HAS_TRITON:
            config_error(args, "convrot_int8_bwd", "int8 backward requires Triton; install it or select bf16 backward.")
        if args.network_module not in ("networks.lora_krea2", "musubi_tuner.networks.lora_krea2"):
            config_error(args, "network_module", "only Krea2 LoRA is supported; use networks.lora_krea2.")
        max_swap = krea2_utils.single_mmdit_large_wide.layers - 2
        if (args.blocks_to_swap or 0) > max_swap:
            config_error(args, "blocks_to_swap", f"the loaded architecture supports at most {max_swap} swapped blocks.")
        options = self._parse_option_pairs(args, "network_args", literal=False)
        known = {
            "conv_dim",
            "conv_alpha",
            "rank_dropout",
            "module_dropout",
            "verbose",
            "exclude_patterns",
            "include_patterns",
            "loraplus_lr_ratio",
        }
        for key, raw in options.items():
            if key not in known:
                config_error(args, "network_args", f"unknown LoRA option {key!r}; remove it or use a supported LoRA option.")
            try:
                if key in ("exclude_patterns", "include_patterns"):
                    patterns = ast.literal_eval(raw)
                    if patterns is not None:
                        if not isinstance(patterns, list) or any(not isinstance(pattern, str) for pattern in patterns):
                            raise ValueError("expected a list of regular-expression strings")
                        for pattern in patterns:
                            re.compile(pattern)
                elif key == "verbose":
                    if raw not in ("True", "False", "None"):
                        raise ValueError("expected True or False")
                else:
                    value = int(raw) if key == "conv_dim" else float(raw)
                    if not math.isfinite(value):
                        raise ValueError("expected a finite number")
                    if key.endswith("dropout") and not 0 <= value <= 1:
                        raise ValueError("expected a value between 0 and 1")
                    if key == "rank_dropout" and value == 1:
                        raise ValueError("rank_dropout must be below 1 to avoid division by zero")
                    if key == "conv_dim" and value <= 0:
                        raise ValueError("expected a positive value")
                    if key in ("conv_alpha", "loraplus_lr_ratio") and value < 0:
                        raise ValueError("expected a nonnegative value")
            except (ValueError, SyntaxError, re.error) as exc:
                config_error(args, "network_args", f"invalid {key}={raw!r}: {exc}; correct this LoRA option.")
        sampling_requested = args.sample_at_first or args.sample_every_n_steps or args.sample_every_n_epochs
        if sampling_requested and not args.sample_prompts:
            config_error(args, "sample_prompts", "required for sample_at_first/sample_every_n_steps/sample_every_n_epochs.")
        if args.sample_prompts:
            for key in ("vae", "text_encoder"):
                if not getattr(args, key):
                    config_error(args, key, "required when sample_prompts is configured.")
            try:
                load_prompts(args.sample_prompts)
            except (OSError, ValueError) as exc:
                config_error(args, "sample_prompts", f"{exc}; provide a valid image prompt file.")
        for key in ("dataset_config", "dit", "network_weights", "log_tracker_config"):
            value = getattr(args, key)
            if value and not os.path.isfile(value):
                config_error(args, key, f"file {str(value)!r} does not exist; set an existing local file.")
        for key in ("vae", "text_encoder"):
            value = getattr(args, key)
            if args.sample_prompts and not os.path.isfile(value):
                config_error(args, key, f"file {value!r} does not exist; set an existing local file.")
        for path in args.base_weights or []:
            if not os.path.isfile(path):
                config_error(args, "base_weights", f"file {path!r} does not exist; use local LoRA files.")
        if args.resume and not args.resume_from_huggingface and not os.path.isdir(args.resume):
            config_error(args, "resume", "must point to an existing local Accelerate state directory.")

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        """Encode the sample prompts with Qwen3-VL up front, cache the embeds, free the encoder.

        Kept deliberately simple (text-to-image only): for each prompt and its optional
        negative prompt we store the varlen selected-layer hidden stack (valid tokens only),
        matching the training cache format, then drop the 4B encoder before training resumes.
        """
        device = accelerator.device

        assert args.text_encoder is not None, "--text_encoder is required for sample generation during training"
        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        encoder = krea2_utils.load_krea2_text_encoder(args.text_encoder, dtype=torch.bfloat16, device=device)

        logger.info("Encoding sample prompts with Qwen3-VL")
        te_outputs = {}  # prompt str -> (valid_len, num_layers, hidden) on cpu
        with torch.no_grad():
            for prompt_dict in prompts:
                for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", None)]:
                    if p is None or p in te_outputs:
                        continue
                    hiddens, mask = krea2_utils.get_krea2_prompt_embeds(encoder, [p])  # (1, seq, L, D), (1, seq)
                    embed = hiddens[0][mask[0]]  # gather valid tokens -> (valid_len, L, D), drops padding
                    te_outputs[p] = embed.to("cpu")

        del encoder
        gc.collect()
        clean_memory_on_device(device)

        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()
            prompt_dict_copy["krea2_vl_embed"] = te_outputs[prompt_dict.get("prompt", "")]
            negative_prompt = prompt_dict.get("negative_prompt", None)
            if negative_prompt is not None:
                prompt_dict_copy["negative_krea2_vl_embed"] = te_outputs[negative_prompt]
            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(device)
        return sample_parameters

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        sample_steps,
        width,
        height,
        generator,
        do_classifier_free_guidance,
        cfg_scale,
    ):
        """Architecture-dependent inference: K2 flow-matching Euler sampler with optional CFG.

        Replicates krea2_sampling.sample's core loop but with embeds taken from the cached
        sample parameters (no encoder) and the trainer's DiT + Qwen-Image VAE. CFG (standard
        uncond + scale*(cond-uncond)) is enabled when a negative prompt is present and cfg_scale > 1
        (musubi convention).
        Resolution-aware mu time-shift uses the K2 Raw defaults (y1=0.5, y2=1.15).
        """
        model = transformer  # SingleStreamDiT
        device = accelerator.device
        patch = model.config.patch
        compression = qwen_image_utils.VAE_SCALE_FACTOR  # Qwen-Image VAE: 8x

        # Standard CFG (uncond + scale*(cond-uncond)), enabled when cfg_scale > 1 — matches the
        # rest of musubi. The official Krea 2 "guidance" value maps as cfg_scale = guidance + 1
        # (official default guidance 4.5 -> cfg_scale 5.5).
        cfg = cfg_scale if cfg_scale is not None else 5.5
        do_cfg = do_classifier_free_guidance and cfg > 1.0

        # The latent grid is patchified in `patch`-sized blocks, so spatial dims must be
        # multiples of compression * patch. The base flow already rounds to 8; align to 16.
        align = compression * patch
        width = krea2_sampling.roundup(width, align, "width")
        height = krea2_sampling.roundup(height, align, "height")
        lat_h, lat_w = height // compression, width // compression

        def build_branch(embed):
            embed = embed.to(device=device, dtype=torch.bfloat16).unsqueeze(0)  # (1, seq, L, D)
            txtmask = torch.ones(1, embed.shape[1], device=device, dtype=torch.bool)
            return embed, txtmask

        txt, txtmask = build_branch(sample_parameter["krea2_vl_embed"])
        if do_cfg:
            untxt, untxtmask = build_branch(sample_parameter["negative_krea2_vl_embed"])

        # Seeded gaussian latent noise (generator already seeded by the base sampler).
        noise = torch.randn(1, model.config.channels, lat_h, lat_w, device=device, dtype=torch.bfloat16, generator=generator)

        img, pos, mask = krea2_sampling.prepare(noise, txt.shape[1], patch, txtmask)
        if do_cfg:
            _, unpos, unmask = krea2_sampling.prepare(noise, untxt.shape[1], patch, untxtmask)

        # mu interpolation endpoints (krea2 sample defaults minres=256, maxres=1280).
        x1 = (256 // align) ** 2
        x2 = (1280 // align) ** 2
        ts = krea2_sampling.timesteps(img.shape[1], sample_steps, x1, x2, y1=0.5, y2=1.15)

        for tcurr, tprev in tqdm(zip(ts[:-1], ts[1:]), total=len(ts) - 1, desc="Denoising steps"):
            t = torch.full((1,), tcurr, dtype=img.dtype, device=device)
            cond = model(img=img, context=txt, t=t, pos=pos, mask=mask)
            if do_cfg:
                uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
                v = uncond + cfg * (cond - uncond)
            else:
                v = cond
            img = img + (tprev - tcurr) * v

        # Unpatchify token sequence back to a latent (1, C, 1, H, W) for the VAE.
        latent = rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=lat_h // patch, w=lat_w // patch)
        latent = latent.unsqueeze(2)  # (1, C, 1, H, W)

        vae.to(device)
        vae.eval()
        with torch.no_grad():
            pixels = vae.decode_to_pixels(latent.to(vae.dtype))  # (1, C, H, W) in [0, 1]
        vae.to("cpu")
        clean_memory_on_device(device)

        pixels = pixels.unsqueeze(2).to(torch.float32).cpu()  # (1, C, 1, H, W) for the grid saver
        return pixels

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        logger.info(f"Loading VAE model from {args.vae}")
        vae = qwen_image_utils.load_vae(args.vae, input_channels=3, device="cpu", disable_mmap=True)
        vae.eval()
        return vae

    def merge_base_weights(self, args, accelerator, transformer, network_module, weight_dtype):
        # Already merged by load_transformer before quantization and block placement.
        pass

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        # For fp8_scaled, dit_weight_dtype is None (the base trainer skips the post-load cast);
        # the fp8 path ignores dtype and keeps non-target weights in their checkpoint dtype.
        # For convrot_int8, dit_weight_dtype equals dit_dtype (bf16), so the post-load cast is
        # a no-op; nn.Module.to would not touch the int8 weights anyway.
        dtype = dit_weight_dtype if dit_weight_dtype is not None else torch.bfloat16
        model = krea2_utils.load_krea2_dit(
            dit_path,
            # device is the calc device for quantization / LoRA merge; under block swap
            # loading_device is "cpu" and the weights are returned there, but the per-tensor
            # computation should still run on the GPU (CPU quantization is extremely slow,
            # especially the ConvRot rotation matmuls).
            device=accelerator.device,
            dtype=dtype,
            fp8_scaled=args.fp8_scaled,
            loading_device=loading_device,
            attn_mode=attn_mode,
            split_attn=split_attn,
            convrot_int8=args.convrot_int8,
            convrot_int8_bwd=args.convrot_int8_bwd,
            lora_weights=[load_file(path) for path in args.base_weights] if args.base_weights else None,
            lora_multipliers=args.base_weights_multiplier,
        )
        return model

    def compile_transformer(self, args, transformer):
        model = transformer  # SingleStreamDiT
        # Compile the per-block SingleStreamBlocks (the heavy, repeated compute). The forward
        # already pads the combined sequence to a multiple of 256 to keep kernel shapes stable.
        # When block swap is on, exclude the swap blocks' Linears from compile (cf. zimage/qwen_image).
        # ConvRot int8 Linears are also excluded: the custom autograd.Function + autotuned Triton
        # kernels are not dynamo-traceable.
        disable_linear = self.blocks_to_swap > 0 or args.convrot_int8
        return model_utils.compile_transformer(args, model, [model.blocks], disable_linear=disable_linear)

    def scale_shift_latents(self, latents):
        # K2 latents are already normalized by the Qwen-Image VAE caching ((raw-mean)/std).
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        **kwargs,
    ) -> DiTOutput:
        model = transformer  # SingleStreamDiT
        device = accelerator.device
        patch = model.config.patch

        latents = batch["latents"]  # (B, C, 1, H, W)
        bsize = latents.shape[0]
        assert latents.shape[2] == 1, f"K2 expects single-frame latents (B,C,1,H,W), got {latents.shape}"

        # --- image tokens / pos / mask (replicates krea2 sampling.prepare) ---
        nmi = noisy_model_input.squeeze(2)  # (B, C, H, W)
        _, _, lat_h, lat_w = nmi.shape
        h_, w_ = lat_h // patch, lat_w // patch

        img_tokens = rearrange(nmi, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

        imgids = torch.zeros((h_, w_, 3), device=device)
        imgids[..., 1] = torch.arange(h_, device=device)[:, None]
        imgids[..., 2] = torch.arange(w_, device=device)[None, :]
        imgpos = repeat(imgids, "h w three -> b (h w) three", b=bsize, three=3)
        imgmask = torch.ones(bsize, h_ * w_, device=device, dtype=torch.bool)

        # --- text tokens / pos / mask (varlen -> padded batch) ---
        vl_embed = batch["krea2_vl_embed"]  # list of (valid_len, num_layers, hidden)
        txt_seq_lens = [x.shape[0] for x in vl_embed]
        max_len = max(txt_seq_lens)
        # pad along the sequence axis (dim 0): F.pad pads last dim first, so (0,0)x2 then (0, pad)
        vl_embed = [F.pad(x, (0, 0, 0, 0, 0, max_len - x.shape[0])) for x in vl_embed]
        context = torch.stack(vl_embed, dim=0).to(device=device, dtype=network_dtype)  # (B, max_len, L, D)

        txtmask = torch.zeros(bsize, max_len, device=device, dtype=torch.bool)
        for i, n in enumerate(txt_seq_lens):
            txtmask[i, :n] = True
        txtpos = torch.zeros(bsize, max_len, 3, device=device)

        # --- combine (image-first: valid tokens form a contiguous prefix per sample) ---
        mask = torch.cat((imgmask, txtmask), dim=1)
        pos = torch.cat((imgpos, txtpos), dim=1)

        img_tokens = img_tokens.to(device=device, dtype=network_dtype)
        t = (timesteps / 1000.0).to(device=device)

        if args.gradient_checkpointing:
            img_tokens.requires_grad_(True)
            context.requires_grad_(True)

        with accelerator.autocast():
            model_pred = model(img=img_tokens, context=context, t=t, pos=pos, mask=mask)  # (B, h*w, c*ph*pw)

        # unpatchify to latent space (B, C, 1, H, W)
        model_pred = rearrange(model_pred, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=h_, w=w_)
        model_pred = model_pred.unsqueeze(2)  # (B, C, 1, H, W)

        # flow matching target (velocity): noise - data
        latents = latents.to(device=device, dtype=network_dtype)
        target = noise - latents
        return DiTOutput(pred=model_pred, target=target)

    # endregion model specific


def krea2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--fp8_scaled",
        action="store_true",
        help="use dynamic scaled fp8 for the DiT (requires --fp8_base). Quantizes per-block Linears at load time.",
    )
    parser.add_argument(
        "--convrot_int8",
        action="store_true",
        help="use ConvRot int8 for the DiT base weights (alternative to fp8; cannot be combined with "
        "--fp8_base/--fp8_scaled). Quantizes per-block Linears at load time with Hadamard rotation + int8; "
        "forward runs fused Triton int8 GEMM (requires triton / triton-windows; falls back to slower "
        "dequantized bf16 matmul without it).",
    )
    parser.add_argument(
        "--convrot_int8_bwd",
        type=str,
        default="bf16",
        choices=["bf16", "int8"],
        help="backward mode for --convrot_int8. bf16 (default): transient dequantized matmul, most accurate. "
        "int8: reuse the fused int8 GEMM for grad_x (faster, quantizes gradients slightly, requires triton).",
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default=None,
        help="Qwen3-VL-4B text encoder safetensors path (only needed for sample generation during training)",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = krea2_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = Krea2NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
