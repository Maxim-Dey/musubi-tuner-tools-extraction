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

    ``fields`` holds the record's keys that the shared schema does not define.
    Only record-based datasources (JSONL)
    can carry such fields; directory datasources always report an empty mapping. Relative
    paths inside ``fields`` resolve from ``base_directory`` (the JSONL's directory, or the
    dataset directory), and ``label`` names the record's origin for error messages.
    """

    fields: Mapping[str, Any]
    base_directory: str
    label: str


def _extra_fields(data: Mapping[str, Any], shared_keys: tuple[str, ...]) -> dict[str, Any]:
    """The record's keys outside the shared schema; ``key_N`` counts as its ``key`` (control_path_0, image_path_1, ...)."""
    extras = {}
    for key, value in data.items():
        stem, _, suffix = key.rpartition("_")
        if key in shared_keys or (suffix.isdigit() and stem in shared_keys):
            continue
        extras[key] = value
    return extras


class ContentDatasource:
    def __init__(self):
        self.caption_only = False  # set to True to only fetch caption for Text Encoder caching
        self.has_control = False

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

    def get_image_data(self, idx: int) -> tuple[str, list[Image.Image], str, list[Image.Image]]:
        """
        Returns image data as a tuple of image path, image, and caption for the given index.
        Key must be unique and valid as a file name.
        May not be called if is_indexable() returns False.
        """
        raise NotImplementedError

    def get_control_paths(self) -> dict[str, list[str]]:
        """
        Returns {image_path: [control paths in index order]} for cache fingerprinting.
        Control pixels travel through ItemInfo.control_content; this accessor exists only so
        cache scripts can fingerprint the source files behind them.
        """
        return {}


def _create_image_fetcher(datasource: ImageDatasource, index: int):
    def fetch():
        return datasource.get_image_data(index)

    # the datasource record index travels as a fetcher attribute so that ItemInfo can
    # reference the originating record without re-deriving it from item keys
    fetch.datasource_index = index
    return fetch


class ImageDirectoryDatasource(ImageDatasource):
    def __init__(
        self,
        image_directory: str,
        caption_extension: Optional[str] = None,
        control_directory: Optional[str] = None,
        control_count_per_image: Optional[int] = None,
    ):
        super().__init__()
        self.image_directory = image_directory
        self.caption_extension = caption_extension
        self.control_directory = control_directory
        self.control_count_per_image = control_count_per_image
        self.current_idx = 0

        # glob images
        logger.info(f"glob images in {self.image_directory}")
        self.image_paths = glob_images(self.image_directory, caption_extension=self.caption_extension)
        logger.info(f"found {len(self.image_paths)} images")

        # glob control images if specified
        if self.control_directory is not None:
            logger.info(f"glob control images in {self.control_directory}")
            self.has_control = True
            self.control_paths = {}

            # sort image paths for matching control images properly: longer names first
            image_paths_sorted = sorted(self.image_paths, key=lambda p: len(os.path.basename(p)), reverse=True)

            # glob control images first
            all_control_image_paths = set(glob_images(self.control_directory))

            for image_path in image_paths_sorted:
                image_basename = os.path.basename(image_path)
                image_basename_no_ext = os.path.splitext(image_basename)[0]

                # find matching control images
                potential_paths = [
                    p
                    for p in all_control_image_paths
                    if os.path.basename(p).startswith(image_basename_no_ext + ".")
                    or os.path.basename(p).startswith(image_basename_no_ext + "_")
                ]

                # remove to avoid duplicate matching
                all_control_image_paths.difference_update(potential_paths)

                if potential_paths:
                    # sort by the digits (`_0000`) suffix, prefer the one without the suffix
                    def sort_key(path):
                        basename = os.path.basename(path)
                        basename_no_ext = os.path.splitext(basename)[0]
                        if image_basename_no_ext == basename_no_ext:  # prefer the one without suffix
                            return 0
                        digits_suffix = basename_no_ext.rsplit("_", 1)[-1]
                        if not digits_suffix.isdigit():
                            raise ValueError(f"Invalid digits suffix in {basename_no_ext}")
                        return int(digits_suffix) + 1

                    potential_paths.sort(key=sort_key)
                    if control_count_per_image is not None and len(potential_paths) < control_count_per_image:
                        logger.error(
                            f"Not enough control images for {image_path}: found {len(potential_paths)}, expected {control_count_per_image}"
                        )
                        raise ValueError(
                            f"Not enough control images for {image_path}: found {len(potential_paths)}, expected {control_count_per_image}"
                        )

                    # take the first `control_count_per_image` paths
                    self.control_paths[image_path] = (
                        potential_paths[:control_count_per_image] if control_count_per_image is not None else potential_paths
                    )
            logger.info(
                f"found {len(self.control_paths)} matching control images for {'arbitrary' if control_count_per_image is None else control_count_per_image} images"
            )

            # log the distribution of number of control images
            count_of_num_control_images = {}
            for paths in self.control_paths.values():
                count = len(paths)
                if count not in count_of_num_control_images:
                    count_of_num_control_images[count] = 0
                count_of_num_control_images[count] += 1
            for count, num_images in count_of_num_control_images.items():
                logger.info(f"  {num_images} images have {count} control images")

            missing_controls = len(self.image_paths) - len(self.control_paths)
            if missing_controls > 0:
                missing_control_paths = set(self.image_paths) - set(self.control_paths.keys())
                logger.error(f"Could not find matching control images for {missing_controls} images: {missing_control_paths}")
                raise ValueError(f"Could not find matching control images for {missing_controls} images")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.image_paths)

    def get_image_data(self, idx: int) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]]]:
        image_path = self.image_paths[idx]
        image_paths = [image_path]

        images = []
        for p in image_paths:
            img = Image.open(p)
            if img.mode != "RGB" and img.mode != "RGBA":
                img = img.convert("RGB")
            images.append(img)

        _, caption = self.get_caption(idx)

        controls = None
        if self.has_control:
            controls = []
            for control_path in self.control_paths[image_path]:
                control = Image.open(control_path)
                if control.mode != "RGB" and control.mode != "RGBA":
                    control = control.convert("RGB")
                controls.append(control)

        return image_path, images, caption, controls

    def get_caption(self, idx: int) -> tuple[str, str]:
        image_path = self.image_paths[idx]
        caption_path = os.path.splitext(image_path)[0] + self.caption_extension if self.caption_extension else ""
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        return image_path, caption

    def get_control_paths(self) -> dict[str, list[str]]:
        if not self.has_control:
            return {}
        return {image_path: list(paths) for image_path, paths in self.control_paths.items()}

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
    SHARED_KEYS = ("image_path", "caption", "control_path")

    def __init__(self, image_jsonl_file: str, control_count_per_image: Optional[int] = None):
        super().__init__()
        self.image_jsonl_file = image_jsonl_file
        self.control_count_per_image = control_count_per_image
        self.current_idx = 0

        # load jsonl
        logger.info(f"load image jsonl from {self.image_jsonl_file}")
        self.data = []
        with open(self.image_jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.error(f"failed to load json: {line} @ {self.image_jsonl_file}")
                    raise
                self.data.append(data)
        logger.info(f"loaded {len(self.data)} images")

        base_directory = os.path.dirname(os.path.abspath(self.image_jsonl_file))
        for row, data in enumerate(self.data, 1):
            if not isinstance(data, dict):
                raise ValueError(f"{self.image_jsonl_file}:{row}={data!r}: expected an image record; use a JSON object")
            for key, value in list(data.items()):
                stem, _, suffix = key.rpartition("_")
                if key not in ("image_path", "control_path") and not (suffix.isdigit() and stem in ("image_path", "control_path")):
                    continue
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"{self.image_jsonl_file}:{row}.{key}={value!r}: invalid image path; provide a nonempty path string"
                    )
                if os.path.isabs(value):
                    continue
                candidate = os.path.join(base_directory, value)
                if os.path.exists(value):
                    data[key] = os.path.abspath(value)
                elif os.path.exists(candidate):
                    data[key] = candidate

        # Normalize control paths
        for item in self.data:
            if "control_path" in item:
                item["control_path_0"] = item.pop("control_path")

            # Ensure control paths are named consistently, from control_path_0000 to control_path_0, control_path_1, etc.
            control_path_keys = [key for key in item.keys() if key.startswith("control_path_")]
            control_path_keys.sort(key=lambda x: int(x.split("_")[-1]))
            for i, key in enumerate(control_path_keys):
                if key != f"control_path_{i}":
                    item[f"control_path_{i}"] = item.pop(key)

        # Check if there are control paths in the JSONL
        self.has_control = any("control_path_0" in item for item in self.data)
        if self.has_control:
            if self.control_count_per_image is None:
                logger.info(f"found {len(self.data)} images with arbitrary control images per image in JSONL data")
            else:
                missing_control_images = [
                    item["image_path"]
                    for item in self.data
                    if sum(f"control_path_{i}" not in item for i in range(self.control_count_per_image)) > 0
                ]
                if missing_control_images:
                    logger.error(f"Some images do not have control paths in JSONL data: {missing_control_images}")
                    raise ValueError(f"Some images do not have control paths in JSONL data: {missing_control_images}")
                logger.info(
                    f"found {len(self.data)} images with {self.control_count_per_image} control images per image in JSONL data"
                )

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.data)

    def get_image_data(self, idx: int) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]]]:
        data = self.data[idx]
        image_path = data.get("image_path", data.get("image_path_0"))
        image_paths = [image_path]

        images = []
        for path in image_paths:
            img = Image.open(path)
            if img.mode != "RGB" and img.mode != "RGBA":
                img = img.convert("RGB")
            images.append(img)

        caption = data["caption"]

        controls = None
        if self.has_control:
            controls = []
            for i in range(self.control_count_per_image or 1000):  # arbitrary large number if control_count_per_image is None
                if f"control_path_{i}" not in data:
                    break
                control_path = data[f"control_path_{i}"]
                control = Image.open(control_path)
                if control.mode != "RGB" and control.mode != "RGBA":
                    control = control.convert("RGB")
                controls.append(control)

        return image_path, images, caption, controls

    def get_caption(self, idx: int) -> tuple[str, str]:
        data = self.data[idx]
        image_path = data.get("image_path", data.get("image_path_0"))
        caption = data["caption"]
        return image_path, caption

    def get_control_paths(self) -> dict[str, list[str]]:
        if not self.has_control:
            return {}
        control_paths: dict[str, list[str]] = {}
        for data in self.data:
            image_path = data.get("image_path", data.get("image_path_0"))
            paths = []
            for i in range(self.control_count_per_image or 1000):  # same bound rule as get_image_data
                if f"control_path_{i}" not in data:
                    break
                paths.append(data[f"control_path_{i}"])
            control_paths[image_path] = paths
        return control_paths

    def get_item_extras(self, idx: int) -> ItemExtras:
        return ItemExtras(
            fields=_extra_fields(self.data[idx], ImageJsonlDatasource.SHARED_KEYS),
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
