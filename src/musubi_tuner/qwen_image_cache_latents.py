import argparse
import logging
from typing import List

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_QWEN_IMAGE,
    ItemInfo,
    save_latent_cache_qwen_image,
)
from musubi_tuner.qwen_image import qwen_image_utils
from musubi_tuner.qwen_image import qwen_image_autoencoder_kl
import musubi_tuner.cache_latents as cache_latents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_contents_qwen_image(batch: List[ItemInfo]) -> torch.Tensor:
    contents = torch.stack([torch.from_numpy(item.content) for item in batch], dim=0)
    contents = contents.permute(0, 3, 1, 2).unsqueeze(2)  # B,C,1,H,W
    return contents / 127.5 - 1.0


def encode_and_save_batch(vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage, batch: List[ItemInfo]):
    contents = preprocess_contents_qwen_image(batch)
    with torch.no_grad():
        latents = vae.encode_pixels_to_latents(contents.to(vae.device, dtype=vae.dtype))
    for item, latent in zip(batch, latents):
        save_latent_cache_qwen_image(item_info=item, latent=latent)


def qwen_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    qwen_image_utils.add_model_version_args(parser)
    return parser


def main():
    parser = cache_latents.setup_parser_common()
    parser = qwen_image_setup_parser(parser)

    args = parser.parse_args()
    config_utils.validate_cache_args(args)
    qwen_image_utils.resolve_model_version_args(args)
    if args.model_version != "original":
        raise ValueError("CLI model_version: unsupported selection; use original")

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_QWEN_IMAGE)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    config_utils.validate_dataset_sources(train_dataset_group, args.dataset_config)
    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images)
        return

    assert args.vae is not None, "VAE checkpoint is required"

    logger.info(f"Loading VAE model from {args.vae}")
    vae = qwen_image_utils.load_vae(args.vae, device=device, disable_mmap=True)
    vae.to(device)

    # encoding closure
    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, batch)

    # reuse core loop from cache_latents with no change
    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()
