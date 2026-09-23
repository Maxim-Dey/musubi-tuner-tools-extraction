import argparse
import gc
import ast
import importlib
import importlib.util
import inspect
import math
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
import re
from typing import Optional


import numpy as np
import toml
import torch
from tqdm import tqdm
from accelerate import Accelerator

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_QWEN_IMAGE,
    ARCHITECTURE_QWEN_IMAGE_FULL,
)
from musubi_tuner.qwen_image import qwen_image_autoencoder_kl, qwen_image_model, qwen_image_utils
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.training.accelerator_setup import clean_memory_on_device
from musubi_tuner.training.parser_common import setup_parser_common, read_config_from_file
from musubi_tuner.training.experiment_paths import rebase_path, resolve_experiment_root
from musubi_tuner.utils import model_utils

import logging


@contextmanager
def _validation_state(transformer, network):
    """Temporarily evaluate without changing the following training update."""
    modules = {id(module): module for root in (transformer, network) for module in root.modules()}
    modes = {key: module.training for key, module in modules.items()}
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        transformer.eval()
        network.eval()
        with torch.no_grad():
            yield
    finally:
        for key, module in modules.items():
            module.training = modes[key]
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QwenImageNetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()

    def validate_training_inputs(self, args):
        if getattr(args, "experiment_dir", None) is not None:
            _prepare_experiment_training_inputs(args)
        validate_training_args(args)
        if getattr(args, "_experiment_root", None):
            from musubi_tuner.training.validation_inputs import prepare_validation_inputs

            self.validation_manifest = prepare_validation_inputs(args)

    def validate_training_dataset(self, args, dataset):
        source = getattr(args, "_config_source", "CLI")
        if not args.sdpa and not args.split_attn and any(item.batch_size > 1 for item in dataset.datasets):
            raise ValueError(
                f"{source}: split_attn: FlashAttention/xformers require split attention for padded text batches; enable split_attn or use SDPA"
            )
        processes = int(os.environ.get("WORLD_SIZE", "1"))
        steps = args.max_train_steps
        if args.max_train_epochs is not None:
            steps = args.max_train_epochs * math.ceil(len(dataset) / processes / args.gradient_accumulation_steps)
        validate_scheduler_args(args, steps * processes)
        if getattr(args, "val_dataset_config", None) and getattr(self, "validation_manifest", None) is None:
            from musubi_tuner.training.validation_inputs import prepare_validation_inputs

            self.validation_manifest = prepare_validation_inputs(args)

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_QWEN_IMAGE

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_QWEN_IMAGE_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.bfloat16

    def process_sample_prompts(self, args, accelerator, sample_prompts):
        device = accelerator.device
        prompts = load_prompts(sample_prompts)
        vl_dtype = torch.float8_e4m3fn if args.fp8_vl else torch.bfloat16
        tokenizer, text_encoder = qwen_image_utils.load_qwen2_5_vl(args.text_encoder, vl_dtype, device, disable_mmap=True)
        sample_prompts_te_outputs = {}
        with torch.amp.autocast(device_type=device.type, dtype=vl_dtype), torch.no_grad():
            for prompt_dict in prompts:
                if "negative_prompt" not in prompt_dict:
                    prompt_dict["negative_prompt"] = " "
                for prompt in [prompt_dict.get("prompt", ""), prompt_dict["negative_prompt"]]:
                    if prompt in sample_prompts_te_outputs:
                        continue
                    logger.info(f"cache Text Encoder outputs for prompt: {prompt}")
                    embed, mask = qwen_image_utils.get_qwen_prompt_embeds(tokenizer, text_encoder, prompt)
                    txt_len = mask.to(dtype=torch.bool).sum().item()
                    sample_prompts_te_outputs[prompt] = embed[:, :txt_len]
        del tokenizer, text_encoder
        gc.collect()
        clean_memory_on_device(device)
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()
            prompt_dict_copy["vl_embed"] = sample_prompts_te_outputs[prompt_dict.get("prompt", "")]
            prompt_dict_copy["negative_vl_embed"] = sample_prompts_te_outputs[prompt_dict["negative_prompt"]]
            sample_parameters.append(prompt_dict_copy)
        clean_memory_on_device(accelerator.device)
        return sample_parameters

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        generator,
        do_classifier_free_guidance,
        cfg_scale,
    ):
        """architecture dependent inference"""
        model: qwen_image_model.QwenImageTransformer2DModel = transformer
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage = vae

        device = accelerator.device

        if cfg_scale is None:
            cfg_scale = 4.0

        # Get embeddings
        vl_embed = sample_parameter["vl_embed"].to(device=device, dtype=torch.bfloat16)
        txt_seq_lens = [vl_embed.shape[1]]
        negative_vl_embed = sample_parameter["negative_vl_embed"].to(device=device, dtype=torch.bfloat16)
        negative_txt_seq_lens = [negative_vl_embed.shape[1]]

        # 4. Prepare latent variables
        num_channels_latents = model.in_channels // 4
        # latents is packed
        latents = qwen_image_utils.prepare_latents(1, num_channels_latents, height, width, torch.bfloat16, device, generator)
        img_shapes = [(1, height // qwen_image_utils.VAE_SCALE_FACTOR // 2, width // qwen_image_utils.VAE_SCALE_FACTOR // 2)]

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / sample_steps, sample_steps)
        image_seq_len = latents.shape[1]

        mu = qwen_image_utils.calculate_shift_qwen_image(image_seq_len)
        scheduler = qwen_image_utils.get_scheduler(discrete_flow_shift)
        # mu is kwarg for FlowMatchingDiscreteScheduler
        timesteps, n = qwen_image_utils.retrieve_timesteps(scheduler, sample_steps, device, sigmas=sigmas, mu=mu)
        assert n == sample_steps, f"Expected steps={sample_steps}, got {n} from scheduler."

        num_warmup_steps = 0  # because FlowMatchingDiscreteScheduler.order is 1, we don't need warmup steps

        # handle guidance
        guidance = None  # guidance_embeds is false for Qwen-Image

        # 6. Denoising loop
        do_cfg = do_classifier_free_guidance and cfg_scale > 1.0
        scheduler.set_begin_index(0)
        # with progress_bar(total=sample_steps) as pbar:

        with tqdm(total=sample_steps, desc="Denoising steps") as pbar:
            for i, t in enumerate(timesteps):
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                latent_model_input = latents

                with torch.no_grad():
                    noise_pred = model(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=None,
                        encoder_hidden_states=vl_embed,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                    )

                if do_cfg:
                    with torch.no_grad():
                        neg_noise_pred = model(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=None,
                            encoder_hidden_states=negative_vl_embed,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                        )
                    comb_pred = neg_noise_pred + cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    pbar.update()

        # Original image latents use the singleton image axis: B, C, 1, H, W.
        latents = qwen_image_utils.unpack_latents(latents, height, width)

        # Move VAE to the appropriate device for sampling
        vae.to(device)
        vae.eval()

        # Decode latents to an image
        logger.info(f"Decoding image from latents: {latents.shape}")
        pixels_list = []
        with torch.no_grad():
            for i in range(latents.shape[0]):
                latents_i = latents[i : i + 1].to(device)
                pixels_i = vae.decode_to_pixels(latents_i)  # decode to pixels, 0-1
                pixels_list.append(pixels_i.to(torch.float32).cpu())
                del latents_i, pixels_i
        latents = None
        pixels = torch.cat(pixels_list, dim=0)  # L C H W

        logger.info("Decoding complete")
        pixels = pixels.to(torch.float32).cpu()

        vae.to("cpu")
        clean_memory_on_device(device)

        pixels = pixels.unsqueeze(2)  # restore the original singleton image axis: B C H W -> B C 1 H W
        return pixels

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        vae_path = args.vae

        logger.info(f"Loading VAE model from {vae_path}")
        vae = qwen_image_utils.load_vae(args.vae, device="cpu", disable_mmap=True)
        vae.eval()
        return vae

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
        model = qwen_image_model.load_qwen_image_model(
            accelerator.device,
            dit_path,
            attn_mode,
            split_attn,
            loading_device,
            dit_weight_dtype,
            args.fp8_scaled,
            num_layers=args.num_layers,
            disable_numpy_memmap=args.disable_numpy_memmap,
        )
        return model

    def compile_transformer(self, args, transformer):
        transformer: qwen_image_model.QwenImageTransformer2DModel = transformer
        return model_utils.compile_transformer(
            args, transformer, [transformer.transformer_blocks], disable_linear=self.blocks_to_swap > 0
        )

    def scale_shift_latents(self, latents):
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
        model: qwen_image_model.QwenImageTransformer2DModel = transformer

        bsize = latents.shape[0]
        latents = batch["latents"]  # B, C, 1, H, W
        assert latents.shape[2] == 1, "Expected latents shape B, C, 1, H, W for the original image model"

        # pack latents
        lat_h = latents.shape[3]
        lat_w = latents.shape[4]
        noisy_model_input = qwen_image_utils.pack_latents(noisy_model_input)

        # control

        # context
        vl_embed = batch["vl_embed"]  # list of (L, D)
        txt_seq_lens = [x.shape[0] for x in vl_embed]

        max_len = max(txt_seq_lens)
        vl_embed = [torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in vl_embed]
        vl_embed = torch.stack(vl_embed, dim=0)  # B, L, D

        # if not split_attn, we need to make attention mask
        if not args.split_attn and bsize > 1:
            vl_mask = torch.zeros(bsize, max_len, dtype=torch.bool, device=vl_embed[0].device)
            for i, x in enumerate(txt_seq_lens):
                vl_mask[i, :x] = True
        else:
            vl_mask = None  # if split_attn, vl_mask is not used
        # print(f"vl_embed shape: {vl_embed.shape}, vl_mask shape: {vl_mask.shape if vl_mask is not None else None}")

        # ensure the hidden state will require grad
        if args.gradient_checkpointing and torch.is_grad_enabled():
            noisy_model_input.requires_grad_(True)
            vl_embed.requires_grad_(True)

        # call DiT
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        vl_embed = vl_embed.to(device=accelerator.device, dtype=network_dtype)
        if vl_mask is not None:
            vl_mask = vl_mask.to(device=accelerator.device)  # bool

        img_shapes = [(1, lat_h // 2, lat_w // 2)]
        img_shapes = [img_shapes]  # make it a list of list for consistency

        # print(
        #     f"noisy_model_input: {noisy_model_input.shape}, vl_embed: {vl_embed.shape}, vl_mask: {vl_mask.shape if vl_mask is not None else None}, img_shapes: {img_shapes}, txt_seq_lens: {txt_seq_lens}"
        # )

        guidance = None
        timesteps = timesteps / 1000.0
        with accelerator.autocast():
            model_pred = model(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                guidance=guidance,
                encoder_hidden_states_mask=vl_mask,
                encoder_hidden_states=vl_embed,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
            )

        # unpack latents
        model_pred = qwen_image_utils.unpack_latents(
            model_pred,
            lat_h * qwen_image_utils.VAE_SCALE_FACTOR,
            lat_w * qwen_image_utils.VAE_SCALE_FACTOR,
            qwen_image_utils.VAE_SCALE_FACTOR,
        )

        # flow matching loss
        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        target = noise - latents

        # print(model_pred.dtype, target.dtype)
        return DiTOutput(pred=model_pred, target=target)

    def evaluate_validation_event(
        self,
        args,
        accelerator,
        transformer,
        network,
        noise_scheduler,
        dit_dtype,
        network_dtype,
        absolute_step,
    ) -> dict[str, float]:
        """Evaluate the frozen Qwen validation sets and publish one complete event."""
        from musubi_tuner.training.validation_inputs import make_validation_loader, read_validation_cache_pair, iter_noise_checks

        manifest = getattr(self, "validation_manifest", None)
        if manifest is None:
            raise ValueError("val_dataset_config: validation input is not prepared; provide and preflight both roles")

        def finite_mean(values: list[float], expected: int, label: str) -> float:
            if len(values) != expected:
                raise ValueError(f"{label}: expected {expected} validation losses, got {len(values)}; check the fixed input")
            try:
                result = math.fsum(values) / expected
            except OverflowError as error:
                raise ValueError(f"{label}: nonfinite aggregate loss (overflow); correct the validation input") from error
            if not math.isfinite(result):
                raise ValueError(f"{label}: nonfinite aggregate loss; correct the validation input")
            return result

        image_means = {role: {"all": [], "low": [], "high": []} for role in ("val_familiar", "val_unfamiliar")}
        checks_per_image = args.val_level_noise_n * args.val_seed_noise_n
        checks_per_half = checks_per_image // 2
        with _validation_state(transformer, network):
            for item in make_validation_loader(manifest):
                latent, vl_embed = read_validation_cache_pair(item)
                latent = latent.to(device=accelerator.device)
                vl_embed = vl_embed.to(device=accelerator.device)
                batch_latents = latent.unsqueeze(0)
                batch = {"latents": batch_latents, "vl_embed": [vl_embed]}
                losses = {"all": [], "low": [], "high": []}
                for check in iter_noise_checks(
                    item,
                    latent,
                    val_seed_noise=args.val_seed_noise,
                    val_level_noise_n=args.val_level_noise_n,
                    val_seed_noise_n=args.val_seed_noise_n,
                ):
                    timestep = torch.tensor([check.timestep], device=accelerator.device, dtype=torch.float32)
                    output = self.call_dit(
                        args,
                        accelerator,
                        transformer,
                        batch_latents,
                        batch,
                        check.epsilon.unsqueeze(0),
                        check.noisy_latent.unsqueeze(0),
                        timestep,
                        network_dtype,
                    )
                    loss, _ = self.compute_loss(
                        args,
                        output,
                        timestep,
                        noise_scheduler,
                        dit_dtype,
                        network_dtype,
                        absolute_step,
                        exact_sigma=check.t,
                    )
                    value = float(loss.detach().item())
                    if not math.isfinite(value):
                        raise ValueError(
                            f"{item.role}: {item.image_path} check i={check.i} j={check.j}: "
                            "nonfinite validation loss; correct the source/cache or model state"
                        )
                    losses["all"].append(value)
                    losses["low" if check.t < 0.5 else "high"].append(value)
                label = f"{item.role}: {item.image_path}"
                image_means[item.role]["all"].append(finite_mean(losses["all"], checks_per_image, label))
                image_means[item.role]["low"].append(finite_mean(losses["low"], checks_per_half, label + " low-noise"))
                image_means[item.role]["high"].append(finite_mean(losses["high"], checks_per_half, label + " high-noise"))
                del latent, vl_embed, batch_latents, batch, check, output, loss

        payload = {}
        for role, prefix in (("val_familiar", "train_eval_loss"), ("val_unfamiliar", "val_loss")):
            expected = len(manifest.items_by_role[role])
            if expected < 1:
                raise ValueError(f"val_dataset_config: {role} is empty; add a captioned validation item")
            payload[prefix + "_mean"] = finite_mean(image_means[role]["all"], expected, role)
            payload[prefix + "_low_noise"] = finite_mean(image_means[role]["low"], expected, role + " low-noise")
            payload[prefix + "_high_noise"] = finite_mean(image_means[role]["high"], expected, role + " high-noise")
        accelerator.log(payload, step=absolute_step)
        return payload

    # endregion model specific


def _prepare_experiment_training_inputs(args) -> None:
    """Resolve the optional experiment hierarchy before any model is loaded."""
    source = getattr(args, "_config_source", "CLI")
    selected = getattr(args, "_selected_train_config_path", None)
    if not selected:
        raise ValueError(
            f"{source}: experiment_dir requires --config_file <root>/train.toml; "
            "select that file before training"
        )
    root = resolve_experiment_root(args.experiment_dir, selected)
    selected_path = Path(selected).resolve()
    required_train_config = root / "train.toml"
    if selected_path != required_train_config:
        raise ValueError(
            f"{source}: selected --config_file {selected_path} does not match experiment_dir {root}; "
            f"select {required_train_config}"
        )
    if getattr(args, "network_module", None) not in (
        "networks.lora_qwen_image", "musubi_tuner.networks.lora_qwen_image",
    ):
        raise ValueError(
            f"{_setting_source(args, 'network_module')}: network_module={args.network_module!r}: "
            "experiment_dir requires networks.lora_qwen_image"
        )

    def fail(key, value, correction):
        selected_source = _setting_source(args, key)
        raise ValueError(f"{selected_source}: {key}={value!r} conflicts with experiment_dir={root}; {correction}")

    if not getattr(args, "dataset_config", None):
        fail("dataset_config", getattr(args, "dataset_config", None), "set dataset_config to the train dataset TOML")
    if not getattr(args, "val_dataset_config", None):
        fail("val_dataset_config", getattr(args, "val_dataset_config", None), "set val_dataset_config to the two-role val TOML")

    for key in (
        "dataset_config", "val_dataset_config", "sample_prompts", "dit", "vae", "text_encoder",
        "network_weights", "log_tracker_config",
    ):
        value = getattr(args, key, None)
        if value is not None:
            setattr(args, key, rebase_path(value, root))
    if getattr(args, "base_weights", None):
        args.base_weights = [rebase_path(value, root) for value in args.base_weights]
    if getattr(args, "resume", None) and not getattr(args, "resume_from_huggingface", False):
        args.resume = rebase_path(args.resume, root)

    for key in ("dataset_config", "val_dataset_config"):
        value = Path(getattr(args, key))
        required = root / ("train-dataset.toml" if key == "dataset_config" else "val-dataset.toml")
        if value != required:
            fail(key, str(value), f"select {required}")
        if not value.is_file():
            fail(key, str(value), f"provide the existing {key} TOML in the experiment folder")

    expected_output = (root / "output").resolve()
    expected_logging = (expected_output / "tensorboard").resolve()
    for key, expected in (("output_dir", expected_output), ("logging_dir", expected_logging)):
        value = getattr(args, key, None)
        if value is not None and Path(rebase_path(value, root)) != expected:
            fail(key, value, f"set {key} to {expected} or omit it")
        setattr(args, key, str(expected))

    name = getattr(args, "output_name", None)
    invalid_name = (
        not isinstance(name, str) or not name or name in (".", "..")
        or any(char in name for char in '/\\<>:"|?*')
        or name.endswith((".", " "))
        or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", name) is not None
    )
    if invalid_name:
        fail("output_name", name, "use one safe file basename for output_name")
    if getattr(args, "save_precision", None) not in (None, "float", "fp32"):
        fail("save_precision", args.save_precision, "use save_precision=fp32")
    last_steps = getattr(args, "save_last_n_steps", None)
    if last_steps is not None and (type(last_steps) is not int or last_steps < 0):
        fail("save_last_n_steps", last_steps, "set save_last_n_steps to a nonnegative integer")

    for key in ("save_last_n_steps_state", "save_last_n_epochs", "save_last_n_epochs_state"):
        value = getattr(args, key, None)
        if value is not None:
            fail(key, value, f"remove {key}; use only save_last_n_steps for experiment retention")
    selected_config = toml.load(selected_path)
    explicit_keys = {
        key for section, values in selected_config.items()
        for key in (values if isinstance(values, dict) else {section: values})
    }
    if getattr(args, "save_state_to_huggingface", False) or "save_state_to_huggingface" in explicit_keys:
        fail("save_state_to_huggingface", getattr(args, "save_state_to_huggingface", False),
             "remove save_state_to_huggingface; keep complete states in the experiment folder")

    args.experiment_dir = str(root)
    args._experiment_root = str(root)


def _setting_source(args, key: str) -> str:
    """Name the CLI override when it supplied an effective setting."""
    option = f"--{key}"
    if any(token == option or token.startswith(option + "=") for token in sys.argv[1:]):
        return f"CLI {option}"
    return getattr(args, "_config_source", "CLI")


def qwen_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Qwen-Image specific parser setup"""
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument("--text_encoder", type=str, default=None, help="text encoder (Qwen2.5-VL) checkpoint path")
    parser.add_argument("--fp8_vl", action="store_true", help="use fp8 for Text Encoder model")
    parser.add_argument("--num_layers", type=int, default=None, help="Number of layers in the DiT model, default is None (60)")
    parser.add_argument("--experiment_dir", type=str, default=None, help="opt-in portable Qwen-Image experiment root")
    parser.add_argument("--val_dataset_config", type=str, default=None, help="path to the two-role validation dataset TOML")
    parser.add_argument("--val_every_n_steps", type=int, default=200, help="validation interval in optimizer steps")
    parser.add_argument("--val_seed_noise", type=int, default=42, help="common validation noise seed")
    parser.add_argument("--val_level_noise_n", type=int, default=10, help="even number of validation noise levels")
    parser.add_argument("--val_seed_noise_n", type=int, default=1, help="noise realizations per validation level")
    qwen_image_utils.add_model_version_args(parser)
    return parser


def validate_validation_args(args):
    source = getattr(args, "_config_source", "CLI")
    values = {
        "val_every_n_steps": (1, False),
        "val_level_noise_n": (2, True),
        "val_seed_noise_n": (1, False),
    }
    for key, (minimum, even) in values.items():
        value = getattr(args, key)
        if type(value) is not int or value < minimum or (even and value % 2):
            qualifier = "even integer" if even else "integer"
            raise ValueError(f"{source}: {key}={value!r}: expected {qualifier} >= {minimum}; correct this setting")
    value = args.val_seed_noise
    if type(value) is not int:
        raise ValueError(f"{source}: val_seed_noise={value!r}: expected an integer, not a boolean; correct this setting")
    value = args.val_dataset_config
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{source}: val_dataset_config={value!r}: expected a nonempty path; correct this setting")


def validate_training_args(args):
    """Validate original-image consumers before any model or tracker is initialized."""
    validate_validation_args(args)
    source = getattr(args, "_config_source", "CLI")

    def fail(key, reason):
        raise ValueError(f"{_setting_source(args, key)}: {key}: {reason}; correct this setting before training")

    def require_package(key, package):
        if importlib.util.find_spec(package) is None:
            fail(key, f"requires installed {package}; install the selected prerequisite or choose another supported option")

    def pairs(key, literal=False):
        result = {}
        for item in getattr(args, key) or []:
            if item.count("=") != 1:
                fail(key, f"expected key=value, got {item!r}")
            name, value = item.split("=")
            if not name.isidentifier() or not value:
                fail(key, f"malformed argument {item!r}")
            if literal:
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    fail(key, f"{name} needs a Python literal value")
            result[name] = value
        return result

    if args.network_module is None:
        fail("network_module", "select networks.lora_qwen_image, networks.loha or networks.lokr")
    if args.fp8_scaled and not args.fp8_base:
        fail("fp8_scaled", "requires fp8_base=true")
    if args.sage_attn:
        fail("sage_attn", "unsupported for training; use sdpa, flash_attn or xformers")
    backend = (
        "sdpa"
        if args.sdpa
        else "flash_attn"
        if args.flash_attn
        else "xformers"
        if args.xformers
        else "flash3"
        if args.flash3
        else None
    )
    if backend is None:
        fail("attention", "select sdpa, flash_attn or xformers")
    if backend == "flash3":
        fail("flash3", "unsupported by Qwen-Image; select sdpa, flash_attn or xformers")
    if backend in ("flash_attn", "xformers"):
        require_package(backend, backend)
        from musubi_tuner.qwen_image import qwen_image_modules

        available = qwen_image_modules.flash_attn_func if backend == "flash_attn" else qwen_image_modules.xops
        if available is None:
            fail(backend, "the selected backend could not be imported; use a compatible installation or SDPA")
    if args.persistent_data_loader_workers and min(args.max_data_loader_n_workers, os.cpu_count() or 1) == 0:
        fail("persistent_data_loader_workers", "requires nonzero max_data_loader_n_workers; otherwise disable persistence")
    positive = (
        "max_train_steps",
        "max_train_epochs",
        "gradient_accumulation_steps",
        "network_dim",
        "num_layers",
        "sample_every_n_steps",
        "sample_every_n_epochs",
        "save_every_n_steps",
        "save_every_n_epochs",
        "num_timestep_buckets",
        "block_swap_ring_size",
        "ddp_timeout",
    )
    nonnegative = (
        "max_data_loader_n_workers",
        "blocks_to_swap",
        "learning_rate",
        "network_alpha",
        "max_grad_norm",
        "save_last_n_steps",
        "save_last_n_epochs",
        "save_last_n_steps_state",
        "save_last_n_epochs_state",
        "scale_weight_norms",
    )
    for key in positive + nonnegative:
        value = getattr(args, key)
        if key == "max_train_steps" and args.max_train_epochs is not None:
            continue  # validated after the existing epoch-derived count becomes available
        if value is not None and (value <= 0 if key in positive else value < 0):
            fail(key, "must be positive" if key in positive else "must be nonnegative (zero disables where documented)")
    if args.network_dropout is not None and not 0 <= args.network_dropout <= 1:
        fail("network_dropout", "must be between 0 and 1")
    if (args.blocks_to_swap or 0) > (args.num_layers or 60) - 1:
        fail("blocks_to_swap", f"cannot exceed {(args.num_layers or 60) - 1} for the configured model")
    if args.blocks_to_swap and args.block_swap_h2d_only and not args.gradient_checkpointing:
        fail(
            "block_swap_h2d_only",
            f"blocks_to_swap={args.blocks_to_swap} with H2D-only swapping requires gradient_checkpointing=true "
            "to re-read reused weights for backward; enable it, disable block_swap_h2d_only or set blocks_to_swap=0",
        )
    if args.gradient_checkpointing_cpu_offload and not args.gradient_checkpointing:
        fail("gradient_checkpointing_cpu_offload", "requires gradient_checkpointing")
    if args.dim_from_weights and not args.network_weights:
        fail("dim_from_weights", "requires network_weights")
    for key in ("min_timestep", "max_timestep"):
        value = getattr(args, key)
        if value is not None and not 0 <= value <= 1000:
            fail(key, "must be within 0..1000")
    if args.min_timestep is not None and args.max_timestep is not None and args.min_timestep >= args.max_timestep:
        fail("min_timestep", "must be less than max_timestep")
    if args.discrete_flow_shift <= 0:
        fail("discrete_flow_shift", "must be positive")
    if args.logit_std < 0 and (
        args.timestep_sampling in ("logsnr", "qinglong_flux")
        or (args.timestep_sampling == "sigma" and args.weighting_scheme == "logit_normal")
    ):
        fail("logit_std", "the selected timestep distribution requires a nonnegative standard deviation")
    if args.vae_dtype is not None:
        try:
            model_utils.str_to_dtype(args.vae_dtype)
        except (ValueError, KeyError):
            fail("vae_dtype", "unknown dtype; supply a supported floating precision")
    network = pairs("network_args")
    allowed = {
        "conv_dim",
        "conv_alpha",
        "rank_dropout",
        "module_dropout",
        "verbose",
        "exclude_patterns",
        "include_patterns",
        "loraplus_lr_ratio",
    }
    if args.network_module.endswith(".lokr"):
        allowed.add("factor")
    for key, value in network.items():
        if key not in allowed:
            fail("network_args", f"unsupported {key}; correct the name for the selected adapter factory")
        try:
            if key.endswith("patterns"):
                patterns = ast.literal_eval(value)
                if not isinstance(patterns, list) or any(not isinstance(pattern, str) for pattern in patterns):
                    raise ValueError("expected a list of regex strings")
                for pattern in patterns:
                    re.compile(pattern)
            elif key in ("rank_dropout", "module_dropout"):
                number = float(value)
                if not 0 <= number < 1:
                    raise ValueError("expected 0 <= dropout < 1")
            elif key == "conv_dim":
                if int(value) < 0:
                    raise ValueError("expected nonnegative integer (0 disables)")
            elif key == "factor":
                if int(value) != -1 and int(value) <= 0:
                    raise ValueError("expected -1 or positive integer")
            elif key != "verbose":
                number = float(value)
                if not math.isfinite(number) or number < 0:
                    raise ValueError("expected a finite nonnegative number")
        except (ValueError, SyntaxError, TypeError, re.error) as error:
            fail("network_args", f"{key}: {error}")
    optimizer_args = pairs("optimizer_args", literal=True)
    scheduler_args = pairs("lr_scheduler_args", literal=True)
    optimizer_name = args.optimizer_type
    if optimizer_name.lower() == "adamw8bit":
        require_package("optimizer_type", "bitsandbytes")
        optimizer_class = importlib.import_module("bitsandbytes.optim").AdamW8bit
    elif optimizer_name.lower() == "adafactor":
        optimizer_class = importlib.import_module("transformers.optimization").Adafactor
    elif optimizer_name.lower() == "adamw":
        optimizer_class = torch.optim.AdamW
    else:
        module, _, name = optimizer_name.rpartition(".")
        try:
            optimizer_class = getattr(importlib.import_module(module) if module else torch.optim, name)
        except (ImportError, AttributeError) as error:
            fail("optimizer_type", f"cannot resolve {optimizer_name}: {error}")
    try:
        bound = inspect.signature(optimizer_class).bind(params=[], lr=args.learning_rate, **optimizer_args)
    except TypeError as error:
        fail("optimizer_args", str(error))
    if optimizer_class is torch.optim.AdamW or optimizer_name.lower() == "adamw8bit":
        bound.apply_defaults()
        try:
            betas = bound.arguments["betas"]
            if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
                fail("optimizer_args", "betas must contain two values within [0, 1)")
            for key in ("eps", "weight_decay"):
                if not bound.arguments[key] >= 0:
                    fail("optimizer_args", f"{key} must be nonnegative")
        except TypeError as error:
            fail("optimizer_args", f"invalid AdamW betas/eps/weight_decay: {error}")
    elif optimizer_class is torch.optim.SGD:
        try:
            if bound.arguments.get("momentum", 0) < 0:
                fail("optimizer_args", "SGD momentum must be nonnegative; set momentum=0 to disable it")
        except TypeError as error:
            fail("optimizer_args", f"invalid SGD momentum: {error}; supply a nonnegative number")
    if args.lr_scheduler_type and not optimizer_name.lower().endswith("schedulefree"):
        module, _, name = args.lr_scheduler_type.rpartition(".")
        try:
            scheduler_class = getattr(importlib.import_module(module) if module else torch.optim.lr_scheduler, name)
            inspect.signature(scheduler_class).bind(optimizer=None, **scheduler_args)
        except (ImportError, AttributeError, TypeError) as error:
            fail("lr_scheduler_type/lr_scheduler_args", str(error))
    if (
        not args.lr_scheduler_type
        and not optimizer_name.lower().endswith("schedulefree")
        and args.lr_scheduler.startswith("adafactor:")
        and optimizer_class is not importlib.import_module("transformers.optimization").Adafactor
    ):
        fail("lr_scheduler", "the adafactor schedule requires the Adafactor optimizer; select it or another schedule")
    validate_scheduler_args(args)
    if args.log_with in ("tensorboard", "all") and not args.logging_dir:
        fail("logging_dir", "required by the selected logging backend")
    if args.log_with in ("tensorboard", "all") or (args.log_with is None and args.logging_dir):
        require_package("log_with", "tensorboard")
    if args.log_with in ("wandb", "all"):
        require_package("log_with", "wandb")
    if args.compile:
        from torch._dynamo.backends.registry import lookup_backend
        from torch._dynamo.exc import InvalidBackend

        try:
            lookup_backend(args.compile_backend)
        except InvalidBackend as error:
            fail("compile_backend", f"{error}; select an available backend or disable compile")
        if args.compile_backend == "inductor":
            require_package("compile", "triton")
    if args.log_tracker_config is not None:
        try:
            toml.load(args.log_tracker_config)
        except (OSError, ValueError) as error:
            fail("log_tracker_config", f"cannot read tracker TOML {args.log_tracker_config!r}: {error}; supply a valid file")
    used = ["dataset_config", "dit"]
    if args.sample_prompts:
        used += ["sample_prompts", "vae", "text_encoder"]
    for key in used + ["network_weights"]:
        value = getattr(args, key)
        if not value and key == "network_weights":
            continue
        if not value or not Path(value).is_file():
            anchor = (
                f"experiment_dir {args._experiment_root}" if getattr(args, "_experiment_root", None)
                else "process CWD"
            )
            fail(key, f"missing input file {value!r}; supply an existing path relative to {anchor}")
    for value in args.base_weights or []:
        if not Path(value).is_file():
            fail("base_weights", f"missing input file {value!r}")
    if args.resume and not args.resume_from_huggingface and not Path(args.resume).is_dir():
        fail("resume", f"missing Accelerate state directory {args.resume!r}")
    if not args.output_dir or not args.output_name:
        fail("output_dir/output_name", "both are required; output directories may be new")
    output = Path(args.output_dir)
    for path in (output, *output.parents):
        if path.exists() and not path.is_dir():
            fail("output_dir", f"{path} is a file; supply a directory path whose existing parents are directories")
    if args.sample_prompts:
        load_prompts(args.sample_prompts)


def validate_scheduler_args(args, num_training_steps=None):
    """Check only settings consumed by the selected scheduler; never change their meaning."""
    source = getattr(args, "_config_source", "CLI")

    def fail(key, reason):
        raise ValueError(f"{source}: {key}: {reason}; correct the scheduler settings")

    if args.optimizer_type.lower().endswith("schedulefree") or args.lr_scheduler_type:
        return
    name = "rex" if args.lr_scheduler.lower() == "rex" else args.lr_scheduler
    optimizer_kwargs = dict(item.split("=", 1) for item in args.optimizer_args or [])
    if args.optimizer_type.lower() == "adafactor":
        relative = ast.literal_eval(optimizer_kwargs.get("relative_step", "True"))
        warmup_init = ast.literal_eval(optimizer_kwargs.get("warmup_init", "False"))
        if relative or warmup_init:
            name = "adafactor:" + str(args.learning_rate)
    from transformers.optimization import SchedulerType

    if name not in {x.value for x in SchedulerType} | {"rex", "piecewise_constant"} and not name.startswith("adafactor:"):
        fail("lr_scheduler", f"unsupported {name!r}; choose an existing scheduler or lr_scheduler_type")
    kwargs = {}
    for item in args.lr_scheduler_args or []:
        try:
            key, value = item.split("=")
            kwargs[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            fail("lr_scheduler_args", f"malformed {item!r}; use key=literal")
    if name.startswith("adafactor:"):
        try:
            float(name.split(":", 1)[1])
        except ValueError:
            fail("lr_scheduler", "use adafactor:<initial learning rate>")
        if kwargs:
            fail("lr_scheduler_args", "the Adafactor schedule does not consume additional arguments")
    else:
        if name == "rex":
            from musubi_tuner.modules.lr_schedulers import RexLR

            schedule_func = RexLR
            consumed = {"max_lr", "min_lr", "num_steps", "num_warmup_steps"}
        elif name == "piecewise_constant":
            from diffusers.optimization import TYPE_TO_SCHEDULER_FUNCTION, SchedulerType as Kind

            schedule_func = TYPE_TO_SCHEDULER_FUNCTION[Kind(name)]
            consumed = set()
        else:
            from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION

            schedule_func = TYPE_TO_SCHEDULER_FUNCTION[SchedulerType(name)]
            consumed = set() if name == "constant" else {"num_warmup_steps"}
            if name not in ("constant", "constant_with_warmup", "inverse_sqrt"):
                consumed.add("num_training_steps")
            if name in ("cosine_with_restarts", "cosine_with_min_lr", "warmup_stable_decay"):
                consumed.add("num_cycles")
            if name == "polynomial":
                consumed.add("power")
            if name == "inverse_sqrt":
                consumed.add("timescale")
            if name == "cosine_with_min_lr":
                consumed.add("min_lr_rate")
            if name == "warmup_stable_decay":
                consumed = {"num_warmup_steps", "num_stable_steps", "num_decay_steps", "num_cycles", "min_lr_ratio"}
        if consumed & kwargs.keys():
            fail(
                "lr_scheduler_args",
                f"duplicate derived arguments {sorted(consumed & kwargs.keys())}; use the dedicated scheduler options",
            )
        try:
            bound = inspect.signature(schedule_func).bind(optimizer=None, **dict.fromkeys(consumed), **kwargs)
            bound.apply_defaults()
            if name == "polynomial" and not args.learning_rate > bound.arguments["lr_end"]:
                fail(
                    "learning_rate/lr_scheduler_args",
                    f"polynomial requires learning_rate={args.learning_rate} > lr_end={bound.arguments['lr_end']}; "
                    "increase learning_rate or lower lr_end",
                )
            if name == "cosine_with_min_lr" and args.lr_scheduler_min_lr_ratio is None and bound.arguments["min_lr"] is None:
                fail(
                    "lr_scheduler_min_lr_ratio/lr_scheduler_args",
                    "cosine_with_min_lr requires a minimum; set lr_scheduler_min_lr_ratio or min_lr in lr_scheduler_args",
                )
            if name == "inverse_sqrt" and args.lr_scheduler_timescale is not None and args.lr_scheduler_timescale <= 0:
                fail("lr_scheduler_timescale", "inverse_sqrt divides by the timescale; use a positive value or omit it")
            if name == "warmup_stable_decay" and bound.arguments["decay_type"] not in ("linear", "cosine", "1-sqrt"):
                fail(
                    "lr_scheduler_args",
                    f"unsupported decay_type={bound.arguments['decay_type']!r}; use 'linear', 'cosine' or '1-sqrt'",
                )
        except TypeError as error:
            fail("lr_scheduler_args", str(error))
    if name == "piecewise_constant":
        try:
            rules = kwargs["step_rules"].split(",")
            for rule in rules[:-1]:
                multiplier, step = rule.split(":")
                float(multiplier)
                int(step)
            float(rules[-1])
        except (AttributeError, TypeError, ValueError) as error:
            fail("lr_scheduler_args", f"invalid step_rules: {error}; use multipliers and integer steps, e.g. '1:10,0.5'")
        return
    if name == "constant" or name.startswith("adafactor:"):
        if args.lr_warmup_steps != 0:
            fail("lr_warmup_steps", f"{name} requires zero warmup; set 0 or use constant_with_warmup")
        return
    keys = ["lr_warmup_steps"]
    if name == "warmup_stable_decay":
        keys.append("lr_decay_steps")
    for key in keys:
        value = getattr(args, key)
        if value is None or value < 0 or (isinstance(value, float) and value > 1):
            fail(key, "use nonnegative integer steps or a floating ratio within 0..1; 200.0 is a ratio, not 200 steps")
    if num_training_steps is not None and name == "warmup_stable_decay":
        counts = [
            int(getattr(args, key) * num_training_steps) if isinstance(getattr(args, key), float) else getattr(args, key)
            for key in keys
        ]
        if sum(counts) > num_training_steps:
            fail("lr_decay_steps", "warmup plus decay exceeds the derived total steps")


def main():
    parser = setup_parser_common()
    parser = qwen_image_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = "bfloat16"  # DiT dtype is bfloat16
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"  # make bfloat16 as default for VAE, this should be checked

    qwen_image_utils.resolve_model_version_args(args)

    trainer = QwenImageNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
