import unittest

from musubi_tuner.dataset.architectures import ARCHITECTURE_KREA2
from musubi_tuner.utils import sai_model_spec


class SaiModelSpecTest(unittest.TestCase):
    def test_build_metadata_supports_krea2_lora(self):
        metadata = sai_model_spec.build_metadata(None, ARCHITECTURE_KREA2, 0, title="krea2_lora_test")

        self.assertEqual(metadata["modelspec.architecture"], "Krea-2/lora")
        self.assertEqual(metadata["modelspec.implementation"], "https://github.com/krea-ai/krea-2")
        self.assertEqual(metadata["modelspec.resolution"], "1024x1024")
        self.assertEqual(metadata["modelspec.title"], "krea2_lora_test")

    def test_metadata_preserves_custom_training_resolution(self):
        metadata = sai_model_spec.build_metadata(None, ARCHITECTURE_KREA2, 0, reso=(768, 1024))

        self.assertEqual(metadata["modelspec.resolution"], "768x1024")


if __name__ == "__main__":
    unittest.main()
