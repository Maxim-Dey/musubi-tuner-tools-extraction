"""Prepare missing Qwen caches through the existing, isolated encoder commands."""

import logging
import os
from pathlib import Path
import subprocess
import sys

from musubi_tuner.training import validation_inputs
from musubi_tuner.training.validation import preserve_rng_state


logger = logging.getLogger(__name__)

# These are the rank/size markers consumed by Accelerate's supported launchers.
_SIZE_VARIABLES = (
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "PMI_SIZE",
    "OMPI_COMM_WORLD_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "MPI_LOCALNRANKS",
    "OMPI_COMM_WORLD_LOCAL_SIZE",
    "MV2_COMM_WORLD_LOCAL_SIZE",
    "ROLE_WORLD_SIZE",
    "GROUP_WORLD_SIZE",
)
_RANK_VARIABLES = (
    "RANK",
    "LOCAL_RANK",
    "PMI_RANK",
    "OMPI_COMM_WORLD_RANK",
    "MV2_COMM_WORLD_RANK",
    "MPI_LOCALRANKID",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MV2_COMM_WORLD_LOCAL_RANK",
    "GROUP_RANK",
    "ROLE_RANK",
)


def prepare_missing_caches(args):
    """Finish source-validated missing stages before the caller initializes training."""
    enabled = getattr(args, "auto_cache", False)
    if enabled is False:
        return
    source = getattr(args, "_config_source", "CLI")
    if type(enabled) is not bool:
        raise ValueError(f"{source}: auto_cache must be a boolean; use true/false or --auto_cache/--no_auto_cache")
    if getattr(args, "experiment_mode", False) is not True:
        raise ValueError(f"{source}: auto_cache requires experiment_mode=true; enable it or use manual cache preparation")
    for key in (*_SIZE_VARIABLES, *_RANK_VARIABLES):
        if key not in os.environ:
            continue
        try:
            value = int(os.environ[key])
        except ValueError as error:
            raise ValueError(f"{source}: auto_cache: invalid {key}={os.environ[key]!r}; use a single-process launch") from error
        if (key in _SIZE_VARIABLES and value != 1) or (key in _RANK_VARIABLES and value > 0):
            raise ValueError(
                f"{source}: auto_cache requires a single training process ({key}={value}); prepare caches manually or use one process"
            )

    def context(item, stage):
        role = "val" if item.validation else "train"
        return f"{source}: auto_cache {role} {stage} ({item.dataset_config})"

    with preserve_rng_state():
        inventories = validation_inputs.inspect_cache_inputs(args)
        jobs = []
        for item in inventories:
            for stage, missing, module, model_key in (
                ("latent", item.missing_latents, "qwen_image_cache_latents", "vae"),
                ("text", item.missing_text, "qwen_image_cache_text_encoder_outputs", "text_encoder"),
            ):
                if missing:
                    jobs.append((item, stage, module, model_key))
        if not jobs:
            logger.info("Automatic cache preparation: all required caches are valid; reusing them")
            return

        # Check every planned resource before starting even the first encoder.
        models = {}
        for item, stage, _, model_key in jobs:
            value = getattr(args, model_key, None)
            if not value or not Path(value).is_file():
                raise ValueError(
                    f"{context(item, stage)}: {model_key}={value!r} is not an input file; supply the required encoder checkpoint"
                )
            models[model_key] = str(Path(value).resolve())
        text_job = next((job for job in jobs if job[3] == "text_encoder"), None)
        if text_job is not None:
            from musubi_tuner.qwen_image import qwen_image_utils

            try:
                tokenizer = qwen_image_utils.Qwen2Tokenizer.from_pretrained(qwen_image_utils.QWEN_IMAGE_ID, subfolder="tokenizer")
                del tokenizer
            except Exception as error:
                raise ValueError(
                    f"{context(text_job[0], 'text')}: Qwen tokenizer unavailable: {error}; "
                    "make the native Qwen/Qwen-Image tokenizer available in the Hugging Face cache or correct its access settings"
                ) from error

        for item, stage, module, model_key in jobs:
            command = [
                sys.executable,
                "-m",
                f"musubi_tuner.{module}",
                "--dataset_config",
                item.dataset_config,
                "--model_version",
                "original",
                "--experiment_mode",
                "--skip_existing",
                "--keep_cache",
                "--batch_size",
                "1",
                "--num_workers",
                "1",
                f"--{model_key}",
                models[model_key],
            ]
            if item.validation:
                command.append("--validation")
            if model_key == "text_encoder" and getattr(args, "fp8_vl", False):
                command.append("--fp8_vl")
            child_environment = os.environ.copy()
            for key in (*_SIZE_VARIABLES, *_RANK_VARIABLES, "MASTER_ADDR", "MASTER_PORT", "FORK_LAUNCHED"):
                child_environment.pop(key, None)
            logger.info("%s: preparing missing files", context(item, stage))
            try:
                subprocess.run(command, check=True, env=child_environment)
            except (OSError, subprocess.SubprocessError) as error:
                raise ValueError(
                    f"{context(item, stage)}: preparation failed: {error}; correct the reported cache-command error and retry; "
                    "completed cache files are retained"
                ) from error

        try:
            completed = validation_inputs.inspect_cache_inputs(args)
        except ValueError as error:
            raise ValueError(
                f"{source}: auto_cache post-check failed: {error}; correct the reported cache error before retrying training"
            ) from error
        for item in completed:
            for stage, missing in (("latent", item.missing_latents), ("text", item.missing_text)):
                if missing:
                    raise ValueError(
                        f"{context(item, stage)}: preparation left {len(missing)} missing cache file(s), including {missing[0]}; "
                        "inspect the cache-command output and retry before training"
                    )
        logger.info("Automatic cache preparation complete; all required caches passed validation")
