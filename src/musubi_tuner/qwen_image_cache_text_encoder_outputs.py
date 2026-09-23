import argparse
from pathlib import Path
from typing import Optional

import torch
import toml
import accelerate
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_QWEN_IMAGE,
    ItemInfo,
    save_text_encoder_output_cache_qwen_image,
)
from musubi_tuner.dataset.cache_io import source_image_sha256

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.training.experiment_paths import rebase_path, resolve_experiment_root
import logging

from musubi_tuner.qwen_image import qwen_image_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(
    tokenizer: Qwen2Tokenizer,
    text_encoder: Qwen2_5_VLForConditionalGeneration,
    batch: list[ItemInfo],
    device: torch.device,
    accelerator: Optional[accelerate.Accelerator],
    roles: Optional[list[Optional[str]]] = None,
):
    prompts = [item.caption for item in batch]
    with torch.no_grad():
        if accelerator is not None:
            with accelerator.autocast():
                embed, mask = qwen_image_utils.get_qwen_prompt_embeds(tokenizer, text_encoder, prompts)
                if embed.dtype == torch.float8_e4m3fn:
                    embed = embed.to(torch.bfloat16)
        else:
            embed, mask = qwen_image_utils.get_qwen_prompt_embeds(tokenizer, text_encoder, prompts)
    for item, (embed_i, mask_i) in zip(batch, zip(embed, mask)):
        txt_len = mask_i.to(dtype=torch.bool).sum().item()
        role = roles[item.dataset_index] if roles is not None else None
        digest = source_image_sha256(item.item_key) if role is not None else None
        save_text_encoder_output_cache_qwen_image(item, embed_i[:txt_len], source_image_sha256=digest)


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
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
        args.text_encoder = rebase_path(args.text_encoder, root)
    config_utils.validate_cache_args(args)
    qwen_image_utils.resolve_model_version_args(args)
    if args.model_version != "original":
        raise ValueError("CLI model_version: unsupported selection; use original")

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
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

    # define accelerator for fp8 inference
    vl_dtype = torch.float8_e4m3fn if args.fp8_vl else torch.bfloat16
    accelerator = None
    if args.fp8_vl:
        accelerator = accelerate.Accelerator(mixed_precision="bf16")

    # prepare cache files and paths: all_cache_files_for_dataset = exisiting cache files, all_cache_paths_for_dataset = all cache paths in the dataset
    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    # Load Qwen2.5-VL
    logger.info(f"Loading Qwen2.5-VL: {args.text_encoder}")
    tokenizer, text_encoder = qwen_image_utils.load_qwen2_5_vl(
        ckpt_path=args.text_encoder, dtype=vl_dtype, device=device, disable_mmap=True
    )

    # Encode with Qwen2.5-VL
    logger.info("Encoding with Qwen2.5-VL")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        nonlocal tokenizer, text_encoder, device, accelerator, args
        encode_and_save_batch(tokenizer, text_encoder, batch, device, accelerator, [dataset.role for dataset in datasets])

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
    )
    del text_encoder

    # remove cache files not in dataset
    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


def qwen_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--train_config", type=str, default=None, help="train.toml anchor for an experiment directory")
    parser.add_argument("--experiment_dir", type=str, default=None, help="portable experiment directory")
    parser.add_argument("--text_encoder", type=str, default=None, required=True, help="Text Encoder (Qwen2.5-VL) checkpoint path")
    parser.add_argument("--fp8_vl", action="store_true", help="use fp8 for Text Encoder model")
    qwen_image_utils.add_model_version_args(parser)
    return parser


if __name__ == "__main__":
    main()
