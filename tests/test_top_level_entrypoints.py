import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script, expected_option",
    [
        ("krea2_cache_latents.py", "--vae"),
        ("krea2_cache_text_encoder_outputs.py", "--text_encoder"),
        ("krea2_train_network.py", "--config_file"),
    ],
)
def test_retained_root_command_imports_and_displays_help(script, expected_option):
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        PYTHONIOENCODING="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout
    assert "--dataset_config" in result.stdout
    assert expected_option in result.stdout
