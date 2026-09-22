from concurrent.futures import ThreadPoolExecutor
import glob
import os
import random
import time
from typing import Any, Optional, Sequence, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

SharedEpoch = Optional["Synchronized[int]"]


import numpy as np
import torch


import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from musubi_tuner.dataset.architectures import *  # noqa: F401,F403
from musubi_tuner.dataset.media_utils import *  # noqa: F401,F403
from musubi_tuner.dataset.media_utils import resize_image_to_bucket  # explicit import for local use


class ItemInfo:
    def __init__(
        self,
        item_key: str,
        caption: str,
        original_size: tuple[int, int],
        bucket_size: Optional[tuple[Any]] = None,
        content: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        latent_cache_path: Optional[str] = None,
    ) -> None:
        self.item_key = item_key
        self.caption = caption
        self.original_size = original_size
        self.bucket_size = bucket_size
        self.content = content
        self.latent_cache_path = latent_cache_path
        self.text_encoder_output_cache_path: Optional[str] = None

        # provenance: the index of the originating dataset in its DatasetGroup, the index of the
        # originating image datasource record.
        self.dataset_index: Optional[int] = None
        self.datasource_index: Optional[int] = None

    def __str__(self) -> str:
        return (
            f"ItemInfo(item_key={self.item_key}, caption={self.caption}, "
            + f"original_size={self.original_size}, bucket_size={self.bucket_size}, "
            + f"latent_cache_path={self.latent_cache_path}, "
            + f"content={[c.shape for c in self.content] if isinstance(self.content, list) else (self.content.shape if self.content is not None else None)}), "
        )


from musubi_tuner.dataset.cache_io import *  # noqa: F401,F403


from musubi_tuner.dataset.bucket import BucketSelector, BucketBatchManager  # noqa: F401


from musubi_tuner.dataset.datasources import (  # noqa: F401
    ContentDatasource,
    ImageDatasource,
    ImageDirectoryDatasource,
    ImageJsonlDatasource,
)


# The following classes have been moved to datasources.py but are kept here
# as a comment reference. They are re-imported above for backward compatibility.
# - ContentDatasource, ImageDatasource, ImageDirectoryDatasource, ImageJsonlDatasource


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        resolution: Tuple[int, int] = (960, 544),
        caption_extension: Optional[str] = None,
        batch_size: int = 1,
        num_repeats: int = 1,
        enable_bucket: bool = False,
        bucket_no_upscale: bool = False,
        cache_directory: Optional[str] = None,
        debug_dataset: bool = False,
        architecture: str = "no_default",
    ):
        self.resolution = resolution
        self.caption_extension = caption_extension
        self.batch_size = batch_size
        self.num_repeats = num_repeats
        self.enable_bucket = enable_bucket
        self.bucket_no_upscale = bucket_no_upscale
        self.cache_directory = cache_directory
        self.debug_dataset = debug_dataset
        self.architecture = architecture
        self.seed = None
        self.current_epoch = 0
        self.shared_epoch = None
        # position in the owning DatasetGroup (stamped by the group), carried by every ItemInfo
        # so cache scripts can find the dataset an item came from
        self.dataset_index: Optional[int] = None

        if not self.enable_bucket:
            self.bucket_no_upscale = False

    def get_metadata(self) -> dict:
        metadata = {
            "resolution": self.resolution,
            "caption_extension": self.caption_extension,
            "batch_size_per_device": self.batch_size,
            "num_repeats": self.num_repeats,
            "enable_bucket": bool(self.enable_bucket),
            "bucket_no_upscale": bool(self.bucket_no_upscale),
        }
        return metadata

    def get_all_latent_cache_files(self):
        return glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}.safetensors"))

    def get_all_text_encoder_output_cache_files(self):
        return glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}_te.safetensors"))

    def get_latent_cache_path(self, item_info: ItemInfo) -> str:
        """
        Returns the cache path for the latent tensor.

        item_info: ItemInfo object

        Returns:
            str: cache path

        cache_path is based on the item_key and the resolution.
        """
        w, h = item_info.original_size
        basename = os.path.splitext(os.path.basename(item_info.item_key))[0]
        assert self.cache_directory is not None, "cache_directory is required / cache_directoryは必須です"
        return os.path.join(self.cache_directory, f"{basename}_{w:04d}x{h:04d}_{self.architecture}.safetensors")

    def get_text_encoder_output_cache_path(self, item_info: ItemInfo) -> str:
        basename = os.path.splitext(os.path.basename(item_info.item_key))[0]
        assert self.cache_directory is not None, "cache_directory is required / cache_directoryは必須です"
        return os.path.join(self.cache_directory, f"{basename}_{self.architecture}_te.safetensors")

    def retrieve_latent_cache_batches(self, num_workers: int):
        raise NotImplementedError

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        raise NotImplementedError

    def prepare_for_training(self, num_timestep_buckets: Optional[int] = None):
        pass

    def set_seed(self, seed: int, shared_epoch: SharedEpoch):
        self.seed = seed
        self.shared_epoch = shared_epoch

    def set_current_epoch(self, epoch):
        assert self.shared_epoch is not None, "shared_epoch is None"
        assert self.shared_epoch.value == epoch, "shared_epoch does not match"

    def set_max_train_steps(self, max_train_steps):
        self.max_train_steps = max_train_steps

    def shuffle_buckets(self):
        raise NotImplementedError

    def __len__(self):
        return NotImplementedError

    def __getitem__(self, idx):
        assert self.shared_epoch is not None, "shared_epoch is None"
        epoch = self.shared_epoch.value
        if epoch > self.current_epoch:
            logger.info(f"epoch is incremented. current_epoch: {self.current_epoch}, epoch: {epoch}")
            num_epochs = epoch - self.current_epoch
            for _ in range(num_epochs):
                self.current_epoch += 1
                self.shuffle_buckets()
        elif epoch < self.current_epoch:
            logger.warning(f"epoch is not incremented. current_epoch: {self.current_epoch}, epoch: {epoch}")
            self.current_epoch = epoch

    def _default_retrieve_text_encoder_output_cache_batches(self, datasource: ContentDatasource, batch_size: int, num_workers: int):
        datasource.set_caption_only(True)
        executor = ThreadPoolExecutor(max_workers=num_workers)

        data: list[ItemInfo] = []
        futures = []

        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed_futures = [future for future in futures if future.done()]
                if len(completed_futures) == 0:
                    if len(futures) >= num_workers or consume_all:  # to avoid adding too many futures
                        time.sleep(0.1)
                        continue
                    else:
                        break  # submit batch if possible

                for future in completed_futures:
                    item_key, caption = future.result()
                    item_info = ItemInfo(item_key, caption, (0, 0), (0, 0))
                    item_info.dataset_index = self.dataset_index
                    item_info.text_encoder_output_cache_path = self.get_text_encoder_output_cache_path(item_info)
                    data.append(item_info)

                    futures.remove(future)

        def submit_batch(flush: bool = False):
            nonlocal data
            if len(data) >= batch_size or (len(data) > 0 and flush):
                batch = data[0:batch_size]
                if len(data) > batch_size:
                    data = data[batch_size:]
                else:
                    data = []
                return batch
            return None

        for fetch_op in datasource:
            future = executor.submit(fetch_op)
            futures.append(future)
            aggregate_future()
            while True:
                batch = submit_batch()
                if batch is None:
                    break
                yield batch

        aggregate_future(consume_all=True)
        while True:
            batch = submit_batch(flush=True)
            if batch is None:
                break
            yield batch

        executor.shutdown()


class ImageDataset(BaseDataset):
    def __init__(
        self,
        resolution: Tuple[int, int],
        caption_extension: Optional[str],
        batch_size: int,
        num_repeats: int,
        enable_bucket: bool,
        bucket_no_upscale: bool,
        image_directory: Optional[str] = None,
        image_jsonl_file: Optional[str] = None,
        cache_directory: Optional[str] = None,
        debug_dataset: bool = False,
        architecture: str = "no_default",
    ):
        super().__init__(
            resolution,
            caption_extension,
            batch_size,
            num_repeats,
            enable_bucket,
            bucket_no_upscale,
            cache_directory,
            debug_dataset,
            architecture,
        )
        self.image_directory = image_directory
        self.image_jsonl_file = image_jsonl_file
        if image_directory is not None:
            self.datasource = ImageDirectoryDatasource(image_directory, caption_extension)
        elif image_jsonl_file is not None:
            self.datasource = ImageJsonlDatasource(image_jsonl_file)
        else:
            raise ValueError("image_directory or image_jsonl_file must be specified")
        if self.cache_directory is None:
            self.cache_directory = self.image_directory
        if not self.cache_directory:
            raise ValueError(f"{image_jsonl_file}: cache_directory is required; provide a cache location")
        self.batch_manager = None
        self.num_train_items = 0

    def get_metadata(self):
        metadata = super().get_metadata()
        if self.image_directory is not None:
            metadata["image_directory"] = os.path.basename(self.image_directory)
        if self.image_jsonl_file is not None:
            metadata["image_jsonl_file"] = os.path.basename(self.image_jsonl_file)
        return metadata

    def get_total_image_count(self):
        return len(self.datasource) if self.datasource.is_indexable() else None

    def retrieve_latent_cache_batches(self, num_workers: int):
        bucket_selector = BucketSelector(self.resolution, self.enable_bucket, self.bucket_no_upscale, self.architecture)
        executor = ThreadPoolExecutor(max_workers=num_workers)

        batches: dict[tuple[int, int], list[ItemInfo]] = {}  # (width, height) -> [ItemInfo]
        futures = []

        # aggregate futures and sort by bucket resolution
        def aggregate_future(consume_all: bool = False):
            while len(futures) >= num_workers or (consume_all and len(futures) > 0):
                completed_futures = [future for future in futures if future.done()]
                if len(completed_futures) == 0:
                    if len(futures) >= num_workers or consume_all:  # to avoid adding too many futures
                        time.sleep(0.1)
                        continue
                    else:
                        break  # submit batch if possible

                for future in completed_futures:
                    original_size, item_key, image, caption, datasource_index = future.result()
                    bucket_height, bucket_width = image.shape[:2]
                    bucket_reso = (bucket_width, bucket_height)

                    item_info = ItemInfo(item_key, caption, original_size, bucket_reso, content=image)
                    item_info.dataset_index = self.dataset_index
                    item_info.datasource_index = datasource_index
                    item_info.latent_cache_path = self.get_latent_cache_path(item_info)

                    item_info.text_encoder_output_cache_path = self.get_text_encoder_output_cache_path(item_info)

                    if bucket_reso not in batches:
                        batches[bucket_reso] = []
                    batches[bucket_reso].append(item_info)

                    futures.remove(future)

        # submit batch if some bucket has enough items
        def submit_batch(flush: bool = False):
            for key in batches:
                if len(batches[key]) >= self.batch_size or flush:
                    batch = batches[key][0 : self.batch_size]
                    if len(batches[key]) > self.batch_size:
                        batches[key] = batches[key][self.batch_size :]
                    else:
                        del batches[key]
                    return key, batch
            return None, None

        for fetch_op in self.datasource:
            # fetch and resize image in a separate thread
            def fetch_and_resize(op: callable) -> tuple:
                image_key, image, caption = op()
                image_size = image.size

                bucket_reso = bucket_selector.get_bucket_resolution(image_size)
                image = resize_image_to_bucket(image, bucket_reso)

                return image_size, image_key, image, caption, getattr(op, "datasource_index", None)

            future = executor.submit(fetch_and_resize, fetch_op)
            futures.append(future)
            aggregate_future()
            while True:
                key, batch = submit_batch()
                if key is None:
                    break
                yield key, batch

        aggregate_future(consume_all=True)
        while True:
            key, batch = submit_batch(flush=True)
            if key is None:
                break
            yield key, batch

        executor.shutdown()

    def retrieve_text_encoder_output_cache_batches(self, num_workers: int):
        return self._default_retrieve_text_encoder_output_cache_batches(self.datasource, self.batch_size, num_workers)

    def prepare_for_training(self, num_timestep_buckets: Optional[int] = None):
        bucket_selector = BucketSelector(self.resolution, self.enable_bucket, self.bucket_no_upscale, self.architecture)

        # glob cache files
        latent_cache_files = glob.glob(os.path.join(self.cache_directory, f"*_{self.architecture}.safetensors"))

        # assign cache files to item info
        # (width, height) -> [ItemInfo] or (width, height, other conds...) -> [ItemInfo]
        bucketed_item_info: dict[Union[tuple[int, int], Any], list[ItemInfo]] = {}
        for cache_file in latent_cache_files:
            tokens = os.path.basename(cache_file).split("_")

            image_size = tokens[-2]  # 0000x0000
            image_width, image_height = map(int, image_size.split("x"))
            image_size = (image_width, image_height)

            item_key = "_".join(tokens[:-2])
            text_encoder_output_cache_file = os.path.join(self.cache_directory, f"{item_key}_{self.architecture}_te.safetensors")
            if not os.path.exists(text_encoder_output_cache_file):
                logger.warning(f"Text encoder output cache file not found: {text_encoder_output_cache_file}")
                continue

            bucket_reso = bucket_selector.get_bucket_resolution(image_size)

            item_info = ItemInfo(item_key, "", image_size, bucket_reso, latent_cache_path=cache_file)
            item_info.text_encoder_output_cache_path = text_encoder_output_cache_file

            bucket = bucketed_item_info.get(bucket_reso, [])
            for _ in range(self.num_repeats):
                bucket.append(item_info)
            bucketed_item_info[bucket_reso] = bucket

        # prepare batch manager
        self.batch_manager = BucketBatchManager(bucketed_item_info, self.batch_size, num_timestep_buckets=num_timestep_buckets)
        self.batch_manager.show_bucket_info()

        self.num_train_items = sum([len(bucket) for bucket in bucketed_item_info.values()])

    def shuffle_buckets(self):
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)
        self.batch_manager.shuffle()

    def __len__(self):
        if self.batch_manager is None:
            return 100  # dummy value
        return len(self.batch_manager)

    def __getitem__(self, idx):
        super().__getitem__(idx)
        return self.batch_manager[idx]


class DatasetGroup(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: Sequence[ImageDataset]):
        super().__init__(datasets)
        self.datasets: list[ImageDataset] = datasets
        self.num_train_items = 0
        for index, dataset in enumerate(self.datasets):
            dataset.dataset_index = index
            self.num_train_items += dataset.num_train_items

    def set_current_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_current_epoch(epoch)

    def set_max_train_steps(self, max_train_steps):
        for dataset in self.datasets:
            dataset.set_max_train_steps(max_train_steps)
