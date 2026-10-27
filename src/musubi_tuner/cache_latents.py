"""Shared image cache batching and previews."""

import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from musubi_tuner.dataset.image_video_dataset import BaseDataset, ItemInfo

logger = logging.getLogger(__name__)


def show_image(image):
    import cv2

    cv2.imshow("image", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key


def show_console(image, width, back, interactive=False):
    from ascii_magic import from_pillow_image, Back

    background = None if back is None else getattr(Back, back.upper())
    image = image if isinstance(image, Image.Image) else Image.fromarray(image)
    from_pillow_image(image).to_terminal(columns=width, back=background)
    if not interactive:
        return ord(" ")
    response = input("Press q to quit, d to next dataset, other key to next: ")
    return ord(response[0]) if response else ord(" ")


def show_datasets(datasets, debug_mode, console_width, console_back, console_num_images):
    print("d: next dataset, q: quit")
    num_workers = max(1, (os.cpu_count() or 1) - 1)
    for index, dataset in enumerate(datasets):
        print(f"Dataset [{index}]")
        remaining = console_num_images
        key = None
        for resolution, batch in dataset.retrieve_latent_cache_batches(num_workers):
            print(f"bucket resolution: {resolution}, count: {len(batch)}")
            for item in batch:
                print(item)
                if debug_mode == "image":
                    key = show_image(item.content)
                else:
                    key = show_console(item.content, console_width, console_back, remaining is None)
                    if remaining is not None:
                        remaining -= 1
                        if remaining == 0:
                            key = ord("d")
                if key == ord("q"):
                    return
                if key == ord("d"):
                    break
            if key == ord("d"):
                break


def encode_datasets(
    datasets: list[BaseDataset],
    encode: callable,
    args: argparse.Namespace,
):
    """Encode image batches and prune stale cache files using the existing lifecycle."""
    num_workers = args.num_workers if args.num_workers is not None else max(1, (os.cpu_count() or 1) - 1)
    for i, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{i}]")
        all_latent_cache_paths = []
        for _, batch in tqdm(dataset.retrieve_latent_cache_batches(num_workers)):
            batch: list[ItemInfo] = batch
            for item in batch:
                if item.content.shape[-1] == 4:
                    item.content = item.content[..., :3]

            all_latent_cache_paths.extend([item.latent_cache_path for item in batch])

            if args.skip_existing:
                filtered_batch = [item for item in batch if not os.path.exists(item.latent_cache_path)]
                if len(filtered_batch) == 0:
                    continue
                batch = filtered_batch

            bs = args.batch_size if args.batch_size is not None else len(batch)
            for i in range(0, len(batch), bs):
                encode(batch[i : i + bs])

        # normalize paths
        all_latent_cache_paths = [os.path.normpath(p) for p in all_latent_cache_paths]
        all_latent_cache_paths = set(all_latent_cache_paths)

        # remove old cache files not in the dataset
        all_cache_files = dataset.get_all_latent_cache_files()
        for cache_file in all_cache_files:
            if os.path.normpath(cache_file) not in all_latent_cache_paths:
                if args.keep_cache:
                    logger.info(f"Keep cache file not in the dataset: {cache_file}")
                else:
                    os.remove(cache_file)
                    logger.info(f"Removed old cache file: {cache_file}")


def setup_parser_common(*, include_vae: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--dataset_config", type=str, required=True, help="path to dataset config .toml file")
    if include_vae:
        parser.add_argument("--vae", type=str, required=False, default=None, help="path to vae checkpoint")
    parser.add_argument("--device", type=str, default=None, help="device to use, default is cuda if available")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="batch size, override dataset config if dataset batch size > this"
    )
    parser.add_argument("--num_workers", type=int, default=None, help="number of workers for dataset. default is cpu count-1")
    parser.add_argument("--skip_existing", action="store_true", help="skip existing cache files")
    parser.add_argument("--keep_cache", action="store_true", help="keep cache files not in dataset")
    parser.add_argument("--debug_mode", type=str, default=None, choices=["image", "console"], help="debug mode")
    parser.add_argument("--console_width", type=int, default=80, help="debug mode: console width")
    parser.add_argument(
        "--console_back", type=str, default=None, help="debug mode: console background color, one of ascii_magic.Back"
    )
    parser.add_argument(
        "--console_num_images",
        type=int,
        default=None,
        help="debug mode: not interactive, number of images to show for each dataset",
    )
    return parser


def validate_cache_args(args):
    for key in ("batch_size", "num_workers", "console_width", "console_num_images"):
        value = getattr(args, key, None)
        if value is not None and value < 1:
            raise ValueError(f"CLI --{key}: must be a positive integer; set it to 1 or larger")
