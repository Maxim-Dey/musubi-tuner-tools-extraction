import os
import torchvision
from einops import rearrange
import numpy as np
import torch
from PIL import Image


# prepare image


# Extracted unchanged from Musubi Tuner hv_generate_video.py.
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
