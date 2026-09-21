import pytest

from musubi_tuner.dataset.architectures import ARCHITECTURE_FLUX_2_DEV
from musubi_tuner.utils import sai_model_spec


def test_dev_lora_metadata_defaults():
    metadata = sai_model_spec.build_metadata(None, ARCHITECTURE_FLUX_2_DEV, 0, title="dev_lora")
    assert metadata["modelspec.architecture"] == "Flux.2-dev/lora"
    assert metadata["modelspec.implementation"] == "https://github.com/black-forest-labs/flux2"
    assert metadata["modelspec.resolution"] == "1280x720"
    assert metadata["modelspec.title"] == "dev_lora"


def test_dev_metadata_overrides_remain_metadata():
    metadata = sai_model_spec.build_metadata(
        None,
        ARCHITECTURE_FLUX_2_DEV,
        0,
        title="custom",
        reso="1024,768",
        author="author",
        tags="cup",
        custom_arch="custom-dev",
        timesteps=(1, 1000),
    )
    assert metadata["modelspec.architecture"] == "custom-dev/lora"
    assert metadata["modelspec.resolution"] == "1024x768"
    assert metadata["modelspec.author"] == "author"
    assert metadata["modelspec.tags"] == "cup"
    assert metadata["modelspec.timestep_range"] == "1,1000"


@pytest.mark.parametrize("architecture", ["hv", "wan", "f2k4b", "f2k9b", "mmh3"])
def test_other_model_metadata_is_not_registered(architecture):
    with pytest.raises(ValueError, match="Unknown architecture"):
        sai_model_spec.build_metadata(None, architecture, 0)
