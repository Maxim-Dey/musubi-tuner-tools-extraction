import argparse
import os
from typing import Optional, Union

import numpy as np
from tqdm import tqdm

from PIL import Image

import logging

from musubi_tuner.dataset.image_video_dataset import BaseDataset, ItemInfo

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def show_image(
    image: Union[list[Union[Image.Image, np.ndarray], Union[Image.Image, np.ndarray]]],
) -> int:
    import cv2

    imgs = (
        [image]
        if (isinstance(image, np.ndarray) and len(image.shape) == 3) or isinstance(image, Image.Image)
        else [image[0], image[-1]]
    )
    if len(imgs) > 1:
        print(f"Number of images: {len(image)}")

    for i, img in enumerate(imgs):
        if len(imgs) > 1:
            print(f"Image {i + 1} of {len(imgs)}: {img.shape}")
        else:
            print(f"Image: {img.shape}")
        cv2_img = np.array(img) if isinstance(img, Image.Image) else img
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", cv2_img)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if k == ord("q") or k == ord("d"):
            return k
    return k


def show_console(
    image: Union[list[Union[Image.Image, np.ndarray], Union[Image.Image, np.ndarray]]],
    width: int,
    back: str,
    interactive: bool = False,
) -> int:
    from ascii_magic import from_pillow_image, Back

    back = None
    if back is not None:
        back = getattr(Back, back.upper())

    k = None
    imgs = (
        [image]
        if (isinstance(image, np.ndarray) and len(image.shape) == 3) or isinstance(image, Image.Image)
        else [image[0], image[-1]]
    )
    if len(imgs) > 1:
        print(f"Number of images: {len(image)}")

    for i, img in enumerate(imgs):
        if len(imgs) > 1:
            print(f"Image {i + 1} of {len(imgs)}: {img.shape}")
        else:
            print(f"Image: {img.shape}")
        pil_img = img if isinstance(img, Image.Image) else Image.fromarray(img)
        ascii_img = from_pillow_image(pil_img)
        ascii_img.to_terminal(columns=width, back=back)

        if interactive:
            k = input("Press q to quit, d to next dataset, other key to next: ")
            if k == "q" or k == "d":
                return ord(k)

    if not interactive:
        return ord(" ")
    return ord(k) if k else ord(" ")


def show_datasets(
    datasets: list[BaseDataset],
    debug_mode: str,
    console_width: int,
    console_back: str,
    console_num_images: Optional[int],
):
    print("d: next dataset, q: quit")

    num_workers = max(1, os.cpu_count() - 1)
    for i, dataset in enumerate(datasets):
        print(f"Dataset [{i}]")
        batch_index = 0
        num_images_to_show = console_num_images
        k = None
        for key, batch in dataset.retrieve_latent_cache_batches(num_workers):
            print(f"bucket resolution: {key}, count: {len(batch)}")
            for j, item_info in enumerate(batch):
                item_info: ItemInfo
                print(f"{batch_index}-{j}: {item_info}")
                if debug_mode == "image":
                    k = show_image(item_info.content)
                elif debug_mode == "console":
                    k = show_console(item_info.content, console_width, console_back, console_num_images is None)
                    if num_images_to_show is not None:
                        num_images_to_show -= 1
                        if num_images_to_show == 0:
                            k = ord("d")  # next dataset
                if k == ord("q"):
                    return
                elif k == ord("d"):
                    break
            if k == ord("d"):
                break
            batch_index += 1


def encode_datasets(
    datasets: list[BaseDataset],
    encode: callable,
    args: argparse.Namespace,
):
    """Common function to encode datasets. This function is called from multiple architecture scripts.

    With --skip_existing, an item whose latent cache file exists is skipped.
    The check is existence-only and does not detect changed models, captions or settings.
    """
    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1)
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
    parser = argparse.ArgumentParser()

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
    parser.add_argument(
        "--disable_cudnn_backend", action="store_true", help="Disable CUDNN PyTorch backend. May be useful for AMD GPUs."
    )
    return parser
