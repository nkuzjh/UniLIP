import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from unilip.pipeline_edit import CustomEditPipeline


class FakeEncoder:
    def __init__(self):
        self.kwargs = None

    def generate_image(self, **kwargs):
        self.kwargs = kwargs
        return np.zeros((2, 2, 3), dtype=np.float32)


class PipelineGeneratorTests(unittest.TestCase):
    def test_generator_reaches_generate_image(self):
        encoder = FakeEncoder()
        pipeline = CustomEditPipeline(
            tokenizer=object(),
            multimodal_encoder=encoder,
            image_processor=lambda image, return_tensors: SimpleNamespace(
                pixel_values=torch.zeros((1, 3, 448, 448))
            ),
        )
        generator = torch.Generator().manual_seed(42)
        with patch.object(torch.Tensor, "cuda", lambda tensor: tensor):
            image = pipeline(
                ["positive", "negative", Image.new("RGB", (4, 4))],
                generator=generator,
            )
        self.assertIsInstance(image, Image.Image)
        self.assertIs(encoder.kwargs["generator"], generator)


if __name__ == "__main__":
    unittest.main()
