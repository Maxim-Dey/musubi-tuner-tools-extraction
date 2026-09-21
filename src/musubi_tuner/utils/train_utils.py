import argparse
import logging
import os
import pickle
import random
import shutil
from pathlib import Path
from typing import Callable

import accelerate
import torch

from musubi_tuner.utils import huggingface_utils
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# checkpointファイル名
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"
STEP_DIFFUSERS_DIR_NAME = "{}-step{:08d}"
TRAINING_PROGRESS_NAME = "training_progress.pt"


def load_training_progress(directory):
    """Optional sidecar; Accelerate's existing model/optimizer/RNG files stay unchanged."""
    path = Path(directory) / TRAINING_PROGRESS_NAME
    if not path.exists():
        return None
    try:
        progress = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(progress, dict) or progress["version"] != 1:
            raise ValueError("unsupported progress version")
        for key in ("epoch", "next_batch", "global_step"):
            if type(progress[key]) is not int or progress[key] < 0:
                raise ValueError(f"invalid {key}={progress[key]!r}")
        for key in (
            "finished",
            "epoch_rng",
            "loader_seed_rng",
            "sampler_rng",
            "dataset_seeds",
            "dataset_epochs",
            "loader_config",
            "loss_list",
            "loss_total",
            "timestep_range_pool",
        ):
            if key not in progress:
                raise ValueError(f"missing {key}")
        if type(progress["finished"]) is not bool:
            raise ValueError("invalid finished flag")
        for key in ("epoch_rng", "loader_seed_rng", "sampler_rng"):
            value = progress[key]
            if value is not None and (not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1):
                raise ValueError(f"invalid {key}")
        if progress["next_batch"] and progress["epoch_rng"] is None:
            raise ValueError("missing epoch RNG for an unfinished epoch")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError(f"{path}: resume progress: {error}; use a complete, compatible state directory") from error
    return progress


def restore_dataset_epochs(dataset_group, seeds, epochs):
    """Rebuild the existing deterministic bucket shuffles, without reading caches."""
    if len(dataset_group.datasets) != len(seeds) or len(seeds) != len(epochs):
        raise ValueError("resume: dataset count changed; use the original dataset configuration")
    rng = random.getstate()
    try:
        for dataset, seed, epoch in zip(dataset_group.datasets, seeds, epochs):
            dataset.seed = seed
            while dataset.current_epoch < epoch:
                dataset.current_epoch += 1
                dataset.shuffle_buckets()
    finally:
        random.setstate(rng)


def get_dataloader_sampler(dataloader):
    from accelerate.data_loader import get_sampler

    sampler = get_sampler(dataloader)
    if sampler is None:
        sampler = dataloader.batch_sampler.batch_sampler.sampler
    return sampler


def resume_dataloader(dataloader, progress):
    """Recreate the saved epoch's sampler and skip completed batches without loading them.

    Keep the returned loader for later epochs so persistent workers are reused.
    Cached image datasets only consume Python RNG when their epoch shuffle changes.
    """
    rng = torch.get_rng_state()
    python_rng = random.getstate()
    dataset_epochs = [dataset.current_epoch for dataset in dataloader.dataset.datasets]
    sampler = get_dataloader_sampler(dataloader)
    if progress["sampler_rng"] is not None:
        sampler.generator.set_state(progress["sampler_rng"])
    torch.set_rng_state(progress["epoch_rng"])
    resumed = accelerate.skip_first_batches(dataloader, progress["next_batch"])
    resumed.set_epoch(progress["epoch"])
    # A continuous persistent loader only draws its worker base seed once.
    # Recreating workers must not steal the sampler's epoch-start RNG draw.
    if resumed.persistent_workers and progress["epoch"] > 0:
        resumed.base_dataloader.generator = torch.Generator().set_state(progress["loader_seed_rng"])
    empty = len(resumed) == 0
    # Accelerate 1.6's shard iterator cannot handle an empty epoch. Exhaust
    # only the base loader to recreate sampler/worker state without its prefetch.
    iterator = iter(resumed.base_dataloader) if empty else iter(resumed)
    try:
        first = next(iterator, None)
    finally:
        if progress["next_batch"]:
            torch.set_rng_state(rng)
        # Preserve a newly encountered dataset's shuffle (the normal loader
        # prefetches one batch ahead), otherwise restore the checkpoint RNG.
        if dataset_epochs == [dataset.current_epoch for dataset in dataloader.dataset.datasets]:
            random.setstate(python_rng)
        resumed.base_dataloader.generator = dataloader.generator
    if empty:
        resumed.set_epoch(progress["epoch"] + 1)

    def batches():
        if first is not None:
            yield first
            yield from iterator

    return resumed, batches()


def get_sanitized_config_or_none(args: argparse.Namespace):
    # if `--log_config` is enabled, return args for logging. if not, return None.
    # when `--log_config is enabled, filter out sensitive values from args
    # if wandb is not enabled, the log is not exposed to the public, but it is fine to filter out sensitive values to be safe

    if not args.log_config:
        return None

    sensitive_args = ["wandb_api_key", "huggingface_token"]
    sensitive_path_args = [
        "dit",
        "vae",
        "text_encoder1",
        "text_encoder2",
        "image_encoder",
        "base_weights",
        "network_weights",
        "output_dir",
        "logging_dir",
    ]
    filtered_args = {}
    for k, v in vars(args).items():
        if k.startswith("_"):
            continue  # parser provenance and validated tracker payloads are internal
        # filter out sensitive values and convert to string if necessary
        if k not in sensitive_args + sensitive_path_args:
            # Accelerate values need to have type `bool`,`str`, `float`, `int`, or `None`.
            if v is None or isinstance(v, bool) or isinstance(v, str) or isinstance(v, float) or isinstance(v, int):
                filtered_args[k] = v
            # accelerate does not support lists
            elif isinstance(v, list):
                filtered_args[k] = f"{v}"
            # accelerate does not support objects
            elif isinstance(v, object):
                filtered_args[k] = f"{v}"

    return filtered_args


class LossRecorder:
    def __init__(self):
        self.loss_list: list[float] = []
        self.loss_total: float = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        if epoch == 0:
            self.loss_list.append(loss)
        else:
            while len(self.loss_list) <= step:
                self.loss_list.append(0.0)
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = loss
        self.loss_total += loss

    @property
    def moving_average(self) -> float:
        return self.loss_total / len(self.loss_list)


def get_epoch_ckpt_name(model_name, epoch_no: int):
    return EPOCH_FILE_NAME.format(model_name, epoch_no) + ".safetensors"


def get_step_ckpt_name(model_name, step_no: int):
    return STEP_FILE_NAME.format(model_name, step_no) + ".safetensors"


def get_last_ckpt_name(model_name):
    return model_name + ".safetensors"


def get_remove_epoch_no(args: argparse.Namespace, epoch_no: int):
    if args.save_last_n_epochs is None:
        return None

    remove_epoch_no = epoch_no - args.save_every_n_epochs * args.save_last_n_epochs
    if remove_epoch_no < 0:
        return None
    return remove_epoch_no


def get_remove_step_no(args: argparse.Namespace, step_no: int):
    if args.save_last_n_steps is None:
        return None

    # calculate the step number to remove from the last_n_steps and save_every_n_steps
    # e.g. if save_every_n_steps=10, save_last_n_steps=30, at step 50, keep 30 steps and remove step 10
    remove_step_no = step_no - args.save_last_n_steps - 1
    remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)
    if remove_step_no < 0:
        return None
    return remove_step_no


def save_and_remove_state_on_epoch_end(args: argparse.Namespace, accelerator: accelerate.Accelerator, epoch_no: int):
    model_name = args.output_name

    logger.info("")
    logger.info(f"saving state at epoch {epoch_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no))
    accelerator.save_state(state_dir)
    if args.save_state_to_huggingface:
        logger.info("uploading state to huggingface.")
        huggingface_utils.upload(args, state_dir, "/" + EPOCH_STATE_NAME.format(model_name, epoch_no))

    last_n_epochs = args.save_last_n_epochs_state if args.save_last_n_epochs_state else args.save_last_n_epochs
    if last_n_epochs is not None:
        remove_epoch_no = epoch_no - args.save_every_n_epochs * last_n_epochs
        state_dir_old = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, remove_epoch_no))
        if os.path.exists(state_dir_old):
            logger.info(f"removing old state: {state_dir_old}")
            shutil.rmtree(state_dir_old)


def save_and_remove_state_stepwise(args: argparse.Namespace, accelerator: accelerate.Accelerator, step_no: int):
    model_name = args.output_name

    logger.info("")
    logger.info(f"saving state at step {step_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, step_no))
    accelerator.save_state(state_dir)
    if args.save_state_to_huggingface:
        logger.info("uploading state to huggingface.")
        huggingface_utils.upload(args, state_dir, "/" + STEP_STATE_NAME.format(model_name, step_no))

    last_n_steps = args.save_last_n_steps_state if args.save_last_n_steps_state else args.save_last_n_steps
    if last_n_steps is not None:
        # last_n_steps前のstep_noから、save_every_n_stepsの倍数のstep_noを計算して削除する
        remove_step_no = step_no - last_n_steps - 1
        remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)

        if remove_step_no > 0:
            state_dir_old = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, remove_step_no))
            if os.path.exists(state_dir_old):
                logger.info(f"removing old state: {state_dir_old}")
                shutil.rmtree(state_dir_old)


def save_state_on_train_end(args: argparse.Namespace, accelerator: accelerate.Accelerator):
    model_name = args.output_name

    logger.info("")
    logger.info("saving last state.")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name))
    accelerator.save_state(state_dir)

    if args.save_state_to_huggingface:
        logger.info("uploading last state to huggingface.")
        huggingface_utils.upload(args, state_dir, "/" + LAST_STATE_NAME.format(model_name))


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def resolve_save_dtype(save_precision: str | None, full_fp16: bool = False, full_bf16: bool = False) -> torch.dtype:
    """Resolve the dtype for saving network weights.

    Explicit --save_precision wins; otherwise follow full_fp16/full_bf16 so the
    saved weights match the training precision; otherwise fp32, the precision
    the network weights are actually trained in.
    """
    if save_precision is not None:
        return str_to_dtype(save_precision)
    if full_fp16:
        return torch.float16
    if full_bf16:
        return torch.bfloat16
    return torch.float32
