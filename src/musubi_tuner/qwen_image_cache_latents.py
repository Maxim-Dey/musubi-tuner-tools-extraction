import argparse
import logging
from pathlib import Path
from typing import List, Optional

import torch
import toml

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_QWEN_IMAGE,
    ItemInfo,
    save_latent_cache_qwen_image,
)
from musubi_tuner.dataset.cache_io import source_image_sha256
from musubi_tuner.qwen_image import qwen_image_utils
from musubi_tuner.qwen_image import qwen_image_autoencoder_kl
import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.training.experiment_paths import rebase_path, resolve_experiment_root

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_contents_qwen_image(batch: List[ItemInfo]) -> torch.Tensor:
    contents = torch.stack([torch.from_numpy(item.content) for item in batch], dim=0)
    contents = contents.permute(0, 3, 1, 2).unsqueeze(2)  # B,C,1,H,W
    return contents / 127.5 - 1.0


def encode_and_save_batch(
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    batch: List[ItemInfo],
    roles: Optional[List[Optional[str]]] = None,
):
    contents = preprocess_contents_qwen_image(batch)
    with torch.no_grad():
        latents = vae.encode_pixels_to_latents(contents.to(vae.device, dtype=vae.dtype))
    for item, latent in zip(batch, latents):
        role = roles[item.dataset_index] if roles is not None else None
        digest = source_image_sha256(item.item_key) if role is not None else None
        save_latent_cache_qwen_image(item_info=item, latent=latent, source_image_sha256=digest)


def qwen_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--train_config", type=str, default=None, help="train.toml anchor for an experiment directory")
    parser.add_argument("--experiment_dir", type=str, default=None, help="portable experiment directory")
    qwen_image_utils.add_model_version_args(parser)
    return parser


def main():
    parser = cache_latents.setup_parser_common()
    parser = qwen_image_setup_parser(parser)

    args = parser.parse_args()
    toml_experiment_dir = None
    if args.train_config is not None:
        selected_train_config = Path(args.train_config).resolve()
        if not selected_train_config.is_file():
            raise ValueError(f"--train_config={args.train_config!r}: file not found; select train.toml")
        for section, values in toml.load(selected_train_config).items():
            if isinstance(values, dict):
                if "experiment_dir" in values:
                    toml_experiment_dir = values["experiment_dir"]
            elif section == "experiment_dir":
                toml_experiment_dir = values
    experiment_dir = args.experiment_dir if args.experiment_dir is not None else toml_experiment_dir
    if experiment_dir is not None:
        if not isinstance(experiment_dir, str) or not experiment_dir:
            raise ValueError("experiment_dir: expected a nonempty path; set a directory")
        root = resolve_experiment_root(experiment_dir, args.train_config)
        if args.train_config is not None and selected_train_config != root / "train.toml":
            raise ValueError(
                f"--train_config={args.train_config!r} conflicts with experiment_dir={root}; "
                f"select {root / 'train.toml'}"
            )
        dataset_config = Path(rebase_path(args.dataset_config, root))
        allowed_datasets = (root / "train-dataset.toml", root / "val-dataset.toml")
        if dataset_config not in allowed_datasets:
            raise ValueError(
                f"--dataset_config={args.dataset_config!r} conflicts with experiment_dir={root}; "
                f"select {allowed_datasets[0]} or {allowed_datasets[1]}"
            )
        args.experiment_dir = str(root)
        args.dataset_config = str(dataset_config)
        args.vae = rebase_path(args.vae, root)
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
    user_config = config_utils.load_user_config(args.dataset_config, experiment_root=args.experiment_dir)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_QWEN_IMAGE)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(
        blueprint.dataset_group, experiment_root=args.experiment_dir
    )

    config_utils.validate_dataset_sources(train_dataset_group, args.dataset_config)
    config_utils.validate_role_aware_sources(train_dataset_group, args.dataset_config)
    datasets = train_dataset_group.datasets
    if args.skip_existing and any(dataset.role is not None for dataset in datasets):
        raise ValueError(
            f"{args.dataset_config}: --skip_existing cannot verify validation source bindings; "
            "rebuild role-bearing caches without this flag"
        )

    if args.debug_mode is not None:
        cache_latents.show_datasets(datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images)
        return

    assert args.vae is not None, "VAE checkpoint is required"

    logger.info(f"Loading VAE model from {args.vae}")
    vae = qwen_image_utils.load_vae(args.vae, device=device, disable_mmap=True)
    vae.to(device)

    # encoding closure
    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, batch, [dataset.role for dataset in datasets])

    # reuse core loop from cache_latents with no change
    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()
