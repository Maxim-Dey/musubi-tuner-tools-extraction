from __future__ import annotations

import glob
from importlib.util import find_spec
import os
from typing import Union, TYPE_CHECKING

import numpy as np
from PIL import Image
import cv2

if TYPE_CHECKING:
    pass

import logging

logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP", ".avif", ".AVIF"]


if find_spec("jxlpy") is not None:  # JPEG-XL on Linux
    from jxlpy import JXLImagePlugin  # noqa: F401 # type: ignore

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])

if find_spec("pillow_jxl") is not None:  # JPEG-XL on Windows
    import pillow_jxl  # noqa: F401 # type: ignore

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])


def glob_images(directory, base="*", caption_extension=None):
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    img_paths = list(set(img_paths))  # remove duplicates

    # check for caption files and only keep images with captions
    if caption_extension is not None:
        caption_paths = glob.glob(os.path.join(glob.escape(directory), "*" + caption_extension))
        caption_bases = set()
        for caption_path in caption_paths:
            caption_base = os.path.splitext(os.path.basename(caption_path))[0]
            caption_bases.add(caption_base)
        filtered_img_paths = []
        for img_path in img_paths:
            img_base = os.path.splitext(os.path.basename(img_path))[0]
            if img_base in caption_bases:
                filtered_img_paths.append(img_path)
        img_paths = filtered_img_paths

    img_paths.sort()
    return img_paths


def divisible_by(num: int, divisor: int) -> int:
    return num - num % divisor


def resize_image_to_bucket(image: Union[Image.Image, np.ndarray], bucket_reso: tuple[int, int]) -> np.ndarray:
    """
    Resize the image to the bucket resolution.

    bucket_reso: **(width, height)**
    """
    is_pil_image = isinstance(image, Image.Image)
    if is_pil_image:
        image_width, image_height = image.size
    else:
        image_height, image_width = image.shape[:2]

    if bucket_reso == (image_width, image_height):
        return np.array(image) if is_pil_image else image

    bucket_width, bucket_height = bucket_reso

    # resize the image to the bucket resolution to match the short side
    scale_width = bucket_width / image_width
    scale_height = bucket_height / image_height
    scale = max(scale_width, scale_height)
    image_width = int(image_width * scale + 0.5)
    image_height = int(image_height * scale + 0.5)

    if scale > 1:
        image = Image.fromarray(image) if not is_pil_image else image
        image = image.resize((image_width, image_height), Image.LANCZOS)
        image = np.array(image)
    else:
        image = np.array(image) if is_pil_image else image
        image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)

    # crop the image to the bucket resolution
    crop_left = (image_width - bucket_width) // 2
    crop_top = (image_height - bucket_height) // 2
    image = image[crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width]
    return image
