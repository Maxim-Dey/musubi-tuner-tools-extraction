import argparse
import os
from typing import Optional

from tqdm import tqdm

from musubi_tuner.dataset.image_video_dataset import BaseDataset

import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def prepare_cache_files_and_paths(datasets: list[BaseDataset]):
    all_cache_files_for_dataset = []  # exisiting cache files
    all_cache_paths_for_dataset = []  # all cache paths in the dataset
    for dataset in datasets:
        all_cache_files = [os.path.normpath(file) for file in dataset.get_all_text_encoder_output_cache_files()]
        all_cache_files = set(all_cache_files)
        all_cache_files_for_dataset.append(all_cache_files)

        all_cache_paths_for_dataset.append(set())
    return all_cache_files_for_dataset, all_cache_paths_for_dataset


def process_text_encoder_batches(
    num_workers: Optional[int],
    skip_existing: bool,
    batch_size: int,
    datasets: list[BaseDataset],
    all_cache_files_for_dataset: list[set],
    all_cache_paths_for_dataset: list[set],
    encode: callable,
):
    """
    Architecture independent processing of text encoder batches.

    With skip_existing, an item whose cache file already exists is skipped by existence only.
    """

    num_workers = num_workers if num_workers is not None else max(1, os.cpu_count() - 1)
    for i, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{i}]")
        all_cache_files = all_cache_files_for_dataset[i]
        all_cache_paths = all_cache_paths_for_dataset[i]

        batches = dataset.retrieve_text_encoder_output_cache_batches(num_workers)

        for batch in tqdm(batches):
            # update cache files (it's ok if we update it multiple times)
            all_cache_paths.update([os.path.normpath(item.text_encoder_output_cache_path) for item in batch])

            # skip existing cache files
            if skip_existing:
                filtered_batch = [
                    item for item in batch if os.path.normpath(item.text_encoder_output_cache_path) not in all_cache_files
                ]
                # print(f"Filtered {len(batch) - len(filtered_batch)} existing cache files")
                if len(filtered_batch) == 0:
                    continue
                batch = filtered_batch

            bs = batch_size if batch_size is not None else len(batch)
            for i in range(0, len(batch), bs):
                encode(batch[i : i + bs])


def post_process_cache_files(
    datasets: list[BaseDataset], all_cache_files_for_dataset: list[set], all_cache_paths_for_dataset: list[set], keep_cache: bool
):
    for i, dataset in enumerate(datasets):
        all_cache_files = all_cache_files_for_dataset[i]
        all_cache_paths = all_cache_paths_for_dataset[i]
        for cache_file in all_cache_files:
            if cache_file not in all_cache_paths:
                if keep_cache:
                    logger.info(f"Keep cache file not in the dataset: {cache_file}")
                else:
                    os.remove(cache_file)
                    logger.info(f"Removed old cache file: {cache_file}")


def setup_parser_common():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_config", type=str, required=True, help="path to dataset config .toml file")
    parser.add_argument("--device", type=str, default=None, help="device to use, default is cuda if available")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="batch size, override dataset config if dataset batch size > this"
    )
    parser.add_argument("--num_workers", type=int, default=None, help="number of workers for dataset. default is cpu count-1")
    parser.add_argument("--skip_existing", action="store_true", help="skip existing cache files")
    parser.add_argument("--keep_cache", action="store_true", help="keep cache files not in dataset")
    return parser
