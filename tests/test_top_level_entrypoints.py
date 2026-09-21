import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ("flux_2_cache_latents", "flux_2_cache_text_encoder_outputs", "flux_2_train_network")


@pytest.mark.parametrize("command", COMMANDS)
@pytest.mark.parametrize("module", [False, True])
def test_retained_help_without_weights(command, module):
    args = ["-m", f"musubi_tuner.{command}"] if module else [str(ROOT / f"{command}.py")]
    environment = os.environ | {
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "-1",
        "HF_HUB_OFFLINE": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    result = subprocess.run(
        [sys.executable, "-B", *args, "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert "--dataset_config" in result.stdout
    assert "--model_version {dev}" in result.stdout


def test_only_retained_root_and_package_commands_exist():
    scripts = {f"{name}.py" for name in COMMANDS}
    assert {p.name for p in ROOT.glob("*.py")} == scripts
    assert {p.name for p in (ROOT / "src/musubi_tuner").glob("*.py")} == scripts | {
        "__init__.py",
        "cache_latents.py",
        "cache_text_encoder_outputs.py",
    }
