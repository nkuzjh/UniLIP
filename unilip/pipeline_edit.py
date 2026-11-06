from typing import List, Optional
from PIL import Image
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from csgo_datasets.unified_task_dataset import img_resize_transform

IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

class CustomEditPipeline:

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        multimodal_encoder: AutoModelForCausalLM,
        image_processor: AutoProcessor,
        multimodal_encoder_input_img_size: int = 448,
    ):
        super().__init__()
        self.multimodal_encoder = multimodal_encoder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.multimodal_encoder_input_img_size = multimodal_encoder_input_img_size

    @torch.no_grad()
    def __call__(
        self,
        inputs: List[Image.Image | str] | str | Image.Image,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.5,
        crop_info: List[int] = [0, 0],
        original_size: List[int] = [1024, 1024],
        generator = None,
        actions: torch.Tensor | None = None,
    ):
        if not isinstance(inputs, list):
            inputs = [inputs]

        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode input prompt
        np_img = self._prepare_and_encode_inputs(
            inputs,
            do_classifier_free_guidance,
            generator=generator,
            guidance_scale=guidance_scale,
            actions=actions,
        )
        images = self.numpy_to_pil(np_img)

        return images[0]

    def _prepare_and_encode_inputs(
        self,
        inputs: List[str | Image.Image],
        do_classifier_free_guidance: bool = False,
        generator=None,
        guidance_scale: float = 4.5,
        actions: torch.Tensor | None = None,
    ):
        pos_text_prompt, neg_text_prompt, input_image = inputs
        if self.multimodal_encoder_input_img_size==224:
            pos_text_prompt = pos_text_prompt.replace(
                        "<image>",
                        f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * 64}{IMG_END_TOKEN}'
                    )
            neg_text_prompt = neg_text_prompt.replace(
                        "<image>",
                        f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * 64}{IMG_END_TOKEN}'
                    )
        else:
            pos_text_prompt = pos_text_prompt.replace(
                        "<image>",
                        f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * 256}{IMG_END_TOKEN}'
                    )
            neg_text_prompt = neg_text_prompt.replace(
                        "<image>",
                        f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * 256}{IMG_END_TOKEN}'
                    )
        resized_images = input_image.resize((448, 448))
        image_inputs = self.image_processor(resized_images, return_tensors="pt")
        image_prompt = image_inputs.pixel_values
        if self.multimodal_encoder_input_img_size==224:
            image_prompt = img_resize_transform(image_prompt)

        prompt = self.multimodal_encoder.generate_image(
            text=[pos_text_prompt, neg_text_prompt],
            pixel_values=image_prompt.cuda(),
            tokenizer=self.tokenizer,
            guidance_scale=guidance_scale,
            generator=generator,
            actions=actions,
        )
        return prompt

    @torch.no_grad()
    def generate_batch(
        self,
        inputs_batch: List[List[str | Image.Image]],
        generator=None,
        guidance_scale: float = 4.5,
        actions: torch.Tensor | List[torch.Tensor] | None = None,
    ) -> List[Image.Image]:
        """Generate one image per (positive prompt, negative prompt, image) triplet."""
        if not inputs_batch:
            return []
        positives, negatives, resized_images = [], [], []
        context_count = 64 if self.multimodal_encoder_input_img_size == 224 else 256
        image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * context_count}{IMG_END_TOKEN}'
        for sample in inputs_batch:
            if len(sample) != 3 or not isinstance(sample[2], Image.Image):
                raise ValueError("Each batch item must be a (positive text, negative text, PIL image) triplet")
            positive, negative, image = sample
            if not isinstance(positive, str) or not isinstance(negative, str):
                raise ValueError("Positive and negative prompts must be strings")
            positives.append(positive.replace("<image>", image_tokens))
            negatives.append(negative.replace("<image>", image_tokens))
            resized_images.append(image.resize((448, 448)))

        pixel_values = self.image_processor(resized_images, return_tensors="pt").pixel_values
        batch_size = len(inputs_batch)
        if pixel_values.ndim != 4 or pixel_values.shape[0] != batch_size:
            raise ValueError(f"Image processor must return {batch_size} pixel tensors, got {tuple(pixel_values.shape)}")
        if self.multimodal_encoder_input_img_size == 224:
            pixel_values = img_resize_transform(pixel_values)

        if actions is not None:
            if isinstance(actions, torch.Tensor):
                if actions.ndim < 2 or actions.shape[0] != batch_size:
                    raise ValueError(f"Actions must have batch dimension {batch_size}")
                actions = actions.reshape(batch_size, -1)
            else:
                if len(actions) != batch_size:
                    raise ValueError(f"Expected {batch_size} actions, got {len(actions)}")
                actions = torch.stack([torch.as_tensor(action).reshape(-1) for action in actions])
            if actions.shape[1] == 0:
                raise ValueError("Actions must have a nonempty pose dimension")

        model = self.multimodal_encoder
        device = model.get_model().device
        output = model.generate_image_batch(
            text=positives + negatives,
            pixel_values=pixel_values.to(device),
            tokenizer=self.tokenizer,
            guidance_scale=guidance_scale,
            generator=generator,
            actions=actions,
        )
        images = self.numpy_to_pil(output)
        if len(images) != batch_size:
            raise RuntimeError(f"Model returned {len(images)} images for {batch_size} inputs")
        return images

    def numpy_to_pil(self, images: np.ndarray) -> List[Image.Image]:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # Special case for grayscale (single channel) images.
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]
        return pil_images
