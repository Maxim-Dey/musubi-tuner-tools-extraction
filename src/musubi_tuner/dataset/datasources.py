"""Directory and JSONL sources for captioned training images."""

import json
import os
from PIL import Image
from musubi_tuner.dataset.media_utils import glob_images


class ContentDatasource:
    def __init__(self):
        self.caption_only = False
        self.current_idx = 0

    def set_caption_only(self, caption_only):
        self.caption_only = caption_only

    def is_indexable(self):
        return True

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self):
            raise StopIteration
        index = self.current_idx
        self.current_idx += 1
        getter = self.get_caption if self.caption_only else self.get_image_data

        def fetch():
            return getter(index)

        fetch.datasource_index = index
        return fetch


class ImageDatasource(ContentDatasource):
    def get_image_data(self, index):
        path, caption = self.get_caption(index)
        with Image.open(path) as source:
            image = source.copy() if source.mode in ("RGB", "RGBA") else source.convert("RGB")
        return path, image, caption


class ImageDirectoryDatasource(ImageDatasource):
    def __init__(self, image_directory, caption_extension=None):
        super().__init__()
        if not os.path.isdir(image_directory):
            raise ValueError(f"image_directory {image_directory}: directory not found; set an existing image directory")
        if not caption_extension:
            raise ValueError("caption_extension: set the image caption sidecar extension, e.g. '.txt'")
        self.image_directory = image_directory
        self.caption_extension = caption_extension
        self.image_paths = glob_images(image_directory, caption_extension=caption_extension)
        if not self.image_paths:
            raise ValueError(
                f"image_directory {image_directory}: no captioned images found; add images with matching {caption_extension} files"
            )

    def __len__(self):
        return len(self.image_paths)

    def get_caption(self, index):
        image_path = self.image_paths[index]
        caption_path = os.path.splitext(image_path)[0] + self.caption_extension
        with open(caption_path, encoding="utf-8") as stream:
            return image_path, stream.read().strip()


class ImageJsonlDatasource(ImageDatasource):
    def __init__(self, image_jsonl_file):
        super().__init__()
        self.image_jsonl_file = image_jsonl_file
        self.data = []
        with open(image_jsonl_file, encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                label = f"{image_jsonl_file}:{line_number}"
                try:
                    record = json.loads(line)
                except ValueError as error:
                    raise ValueError(f"{label}: invalid JSON; correct this image record") from error
                if not isinstance(record, dict):
                    raise ValueError(f"{label}: expected an image record with image_path and caption")
                unknown = set(record) - {"image_path", "image_path_0", "caption"}
                if unknown:
                    raise ValueError(f"{label}: unsupported fields {sorted(unknown)}; use only image_path and caption")
                paths = [record[key] for key in ("image_path", "image_path_0") if key in record]
                if len(paths) != 1 or not isinstance(paths[0], str) or not paths[0]:
                    raise ValueError(f"{label}: specify one nonempty image_path (or image_path_0)")
                if not isinstance(record.get("caption"), str):
                    raise ValueError(f"{label}.caption: expected a string")
                if not os.path.isfile(paths[0]):
                    raise ValueError(f"{label}.image_path: file not found: {paths[0]}; set an existing image path")
                self.data.append({"image_path": paths[0], "caption": record["caption"]})
        if not self.data:
            raise ValueError(f"{image_jsonl_file}: no image records found; add image_path and caption records")

    def __len__(self):
        return len(self.data)

    def get_caption(self, index):
        record = self.data[index]
        return record["image_path"], record["caption"]
