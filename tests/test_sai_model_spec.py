"""Metadata consumed by saved Qwen adapters."""

import pytest
from musubi_tuner.dataset.architectures import ARCHITECTURE_QWEN_IMAGE
from musubi_tuner.utils import sai_model_spec


def test_qwen_adapter_metadata():
    metadata = sai_model_spec.build_metadata(None, ARCHITECTURE_QWEN_IMAGE, 0, title="qwen_lora_test")
    assert metadata["modelspec.architecture"] == "Qwen-Image/lora"
    assert metadata["modelspec.implementation"] == "https://github.com/QwenLM/Qwen-Image"
    assert metadata["modelspec.resolution"] == "1328x1328"
    assert metadata["modelspec.title"] == "qwen_lora_test"
    assert all(isinstance(value, str) for value in metadata.values())


def test_qwen_metadata_overrides_and_timestep_range():
    metadata = sai_model_spec.build_metadata(
        None,
        ARCHITECTURE_QWEN_IMAGE,
        0,
        custom_arch="custom-qwen",
        reso="512,768",
        timesteps=(100, 900),
        author="fixture",
        tags="test",
    )
    assert metadata["modelspec.architecture"] == "custom-qwen/lora"
    assert metadata["modelspec.resolution"] == "512x768"
    assert metadata["modelspec.timestep_range"] == "100,900"
    assert metadata["modelspec.author"] == "fixture"
    assert metadata["modelspec.tags"] == "test"


def test_foreign_metadata_is_rejected():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        sai_model_spec.build_metadata(None, "wan", 0)
