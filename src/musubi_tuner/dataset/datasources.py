from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Mapping, Optional, TYPE_CHECKING

from PIL import Image

from musubi_tuner.dataset.media_utils import glob_images

if TYPE_CHECKING:
    pass

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ItemExtras:
    """Per-item fields beyond the shared dataset schema, for architecture-specific consumers.

    ``fields`` holds the record's keys that the shared schema does not define (for example the
    extension fields). Only record-based datasources (JSONL)
    can carry such fields; directory datasources always report an empty mapping. Relative
    paths inside ``fields`` resolve from ``base_directory`` (the JSONL's directory, or the
    dataset directory), and ``label`` names the record's origin for error messages.
    """

    fields: Mapping[str, Any]
    base_directory: str
    label: str


class ContentDatasource:
    def __init__(self):
        self.caption_only = False  # set to True to only fetch caption for Text Encoder caching

    def set_caption_only(self, caption_only: bool):
        self.caption_only = caption_only

    def is_indexable(self):
        return False

    def get_caption(self, idx: int) -> tuple[str, str]:
        """
        Returns caption. May not be called if is_indexable() returns False.
        """
        raise NotImplementedError

    def get_item_extras(self, idx: int) -> ItemExtras:
        """
        Returns the item's fields beyond the shared schema (see ItemExtras). Indices align with
        get_caption and with ItemInfo.datasource_index. May not be called if is_indexable() returns False.
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class ImageDatasource(ContentDatasource):
    def __init__(self):
        super().__init__()

    def get_image_data(self, idx: int) -> tuple[str, Image.Image, str]:
        """
        Returns image data as a tuple of image path, image, and caption for the given index.
        Key must be unique and valid as a file name.
        May not be called if is_indexable() returns False.
        """
        raise NotImplementedError


def _create_image_fetcher(datasource: ImageDatasource, index: int):
    def fetch():
        return datasource.get_image_data(index)

    # the datasource record index travels as a fetcher attribute so that ItemInfo can
    # reference the originating record without re-deriving it from item keys
    fetch.datasource_index = index
    return fetch


class ImageDirectoryDatasource(ImageDatasource):
    def __init__(self, image_directory: str, caption_extension: Optional[str] = None):
        super().__init__()
        self.image_directory = image_directory
        self.caption_extension = caption_extension
        self.current_idx = 0
        if not os.path.isdir(image_directory):
            raise ValueError(f"image_directory {image_directory}: directory not found; supply an existing image directory")
        logger.info(f"glob images in {self.image_directory}")
        self.image_paths = glob_images(self.image_directory, caption_extension=self.caption_extension)
        logger.info(f"found {len(self.image_paths)} images")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.image_paths)

    def get_image_data(self, idx: int) -> tuple[str, Image.Image, str]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        _, caption = self.get_caption(idx)
        return image_path, image, caption

    def get_caption(self, idx: int) -> tuple[str, str]:
        image_path = self.image_paths[idx]
        caption_path = os.path.splitext(image_path)[0] + self.caption_extension if self.caption_extension else ""
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        return image_path, caption

    def get_item_extras(self, idx: int) -> ItemExtras:
        # a directory item has no place for fields beyond the shared schema
        return ItemExtras(fields={}, base_directory=self.image_directory, label=self.image_paths[idx])

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> callable:
        """
        Returns a fetcher function that returns image data.
        """
        if self.current_idx >= len(self.image_paths):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)
        else:
            fetcher = _create_image_fetcher(self, self.current_idx)

        self.current_idx += 1
        return fetcher


class ImageJsonlDatasource(ImageDatasource):
    # the shared image JSONL schema (numbered variants included); every other key is an item extra
    SHARED_KEYS = ("image_path", "caption")

    def __init__(self, image_jsonl_file: str, experiment_root: Optional[str] = None):
        super().__init__()
        self.image_jsonl_file = image_jsonl_file
        self.current_idx = 0
        logger.info(f"load image jsonl from {self.image_jsonl_file}")
        self.data = []
        with open(self.image_jsonl_file, "r", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                label = f"{image_jsonl_file}: line {number}"
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{label}: invalid JSON; provide one image_path/caption object per line") from error
                if not isinstance(data, dict):
                    raise ValueError(f"{label}: expected an object; provide image_path and caption")
                unknown = set(data) - set(self.SHARED_KEYS)
                if unknown:
                    raise ValueError(
                        f"{label}: unsupported fields {sorted(unknown)}; keep only image_path and caption for original images"
                    )
                for key in self.SHARED_KEYS:
                    if not isinstance(data.get(key), str):
                        raise ValueError(f"{label}: {key} must be a string; supply an image path and caption")
                if experiment_root is not None and not os.path.isabs(data["image_path"]):
                    data["image_path"] = os.path.normpath(os.path.join(experiment_root, data["image_path"]))
                if not os.path.isfile(data["image_path"]):
                    origin = "the experiment root" if experiment_root is not None else "the working directory"
                    raise ValueError(
                        f"{label}: image_path {data['image_path']!r} not found; supply an existing image path relative to {origin} or absolute"
                    )
                self.data.append(data)
        logger.info(f"loaded {len(self.data)} images")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.data)

    def get_image_data(self, idx: int) -> tuple[str, Image.Image, str]:
        image_path, caption = self.get_caption(idx)
        image = Image.open(image_path)
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        return image_path, image, caption

    def get_caption(self, idx: int) -> tuple[str, str]:
        data = self.data[idx]
        image_path = data["image_path"]
        caption = data["caption"]
        return image_path, caption

    def get_item_extras(self, idx: int) -> ItemExtras:
        return ItemExtras(
            fields={},
            base_directory=os.path.dirname(os.path.abspath(self.image_jsonl_file)),
            label=f"{os.path.basename(self.image_jsonl_file)} line {idx + 1}",
        )

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> callable:
        if self.current_idx >= len(self.data):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)

        else:
            fetcher = _create_image_fetcher(self, self.current_idx)

        self.current_idx += 1
        return fetcher
