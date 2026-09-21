import os
import torchvision
from einops import rearrange
import numpy as np
import torch
from PIL import Image
from typing import Tuple, Optional

from musubi_tuner.dataset import image_video_dataset


# prepare image
def preprocess_image(
    image: Image, w: int, h: int, handle_alpha: bool = False
) -> Tuple[torch.Tensor, np.ndarray, Optional[np.ndarray]]:
    """
    Preprocess the image for the model.
    Args:
        image (Image): The input image. RGB or RGBA format.
        w (int): The target bucket width.
        h (int): The target bucket height.
        handle_alpha (bool): Whether to handle alpha channel for tensor and numpy array.
    Returns:
        Tuple[torch.Tensor, np.ndarray, Optional[np.ndarray]]:
            - image_tensor: The preprocessed image tensor (NCHW format). -1.0 to 1.0.
            - image_np: The original image as a numpy array (HWC format). 0 to 255.
            - alpha: The alpha channel of the image if present in original size, otherwise None.
    """
    if image.mode == "RGBA":
        alpha = image.split()[-1]
    else:
        alpha = None
    if handle_alpha:
        image = image.convert("RGBA")
    else:
        image = image.convert("RGB")

    image_np = np.array(image)  # PIL to numpy, HWC

    image_np = image_video_dataset.resize_image_to_bucket(image_np, (w, h))  # TODO move this to this file
    image_tensor = torch.from_numpy(image_np).float() / 127.5 - 1.0  # -1 to 1.0, HWC
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW -> NCHW, N=1
    return image_tensor, image_np, alpha


def save_images_grid(
    videos: torch.Tensor, parent_dir: str, image_name: str, rescale: bool = False, n_rows: int = 1, create_subdir=True
) -> list[str]:
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    if create_subdir:
        output_dir = os.path.join(parent_dir, image_name)
    else:
        output_dir = parent_dir

    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    for i, x in enumerate(outputs):
        image_path = os.path.join(output_dir, f"{image_name}_{i:03d}.png")
        image_paths.append(image_path)
        image = Image.fromarray(x)
        image.save(image_path)

    return image_paths
