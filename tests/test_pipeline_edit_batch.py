import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from diffusers.schedulers import DPMSolverMultistepScheduler

from unilip.pipeline_edit import CustomEditPipeline
from unilip.model.language_model import unified_unilip as model_module


class FakeTokenizer:
    def __call__(self, texts, padding, return_tensors):
        rows = []
        for text in texts:
            row = [int(text.split()[0])]
            if "<IMG_CONTEXT>" in text:
                row.append(model_module.UND_IMAGE_TOKEN_IDX)
            row.append(2)
            if "long" in text:
                row.append(3)
            rows.append(row)
        width = max(map(len, rows))
        ids = torch.tensor([row + [0] * (width - len(row)) for row in rows])
        return SimpleNamespace(input_ids=ids, attention_mask=(ids != 0).long())


class FakeLanguageModel:
    def __init__(self):
        self.inputs = None
        self.mask = None

    def embed_tokens(self, ids):
        return ids.float().unsqueeze(-1).repeat(1, 1, 2)

    def __call__(self, inputs_embeds, attention_mask, **kwargs):
        self.inputs = inputs_embeds.clone()
        self.mask = attention_mask.clone()
        return SimpleNamespace(hidden_states=[inputs_embeds.clone()])


class FakeCore:
    device = torch.device("cpu")

    def __init__(self):
        self.config = model_module.Unified_UniLIP_InternVLConfig()
        self.config.dit_path = "unused"
        self.config.unilip_factor = 1.0
        self.language_model = FakeLanguageModel()
        self.latent_queries = torch.zeros(1, 1, 2)
        self.llm_connector = lambda inputs_embeds, **kwargs: SimpleNamespace(hidden_states=[inputs_embeds])
        self.projector = lambda states: states
        self.vae_decoder = SimpleNamespace(vae_decode=self.decode)
        self.decode_calls = 0

    def get_image_features(self, pixel_values, **kwargs):
        batch = pixel_values.shape[0]
        return torch.tensor([[[10.0 + i, 20.0 + i]] for i in range(batch)])

    def decode(self, latents):
        self.decode_calls += 1
        return torch.stack([torch.full((3, 32, 32), i / 10) for i in range(latents.shape[0])])


class FakeModel:
    def __init__(self):
        self.model = FakeCore()
        self.config = SimpleNamespace(vision_feature_layer=-1, vision_feature_select_strategy="default")
        self.vision_tower = SimpleNamespace(dtype=torch.float32)
        self.actions_seen = None
        self.sample_seen = None

    def get_model(self):
        return self.model

    def get_n_query(self):
        return 1

    def _replace_loc_st_placeholders(self, ids, embeddings, actions):
        self.actions_seen = actions.clone()
        return embeddings

    def sample_image_latents(self, states, scheduler, **kwargs):
        self.sample_seen = (states.clone(), kwargs)
        return torch.zeros(states.shape[0] // 2, 32, 16, 16)

    generate_image = model_module.Unified_UniLIP_InternVLForCausalLM.generate_image
    generate_image_batch = model_module.Unified_UniLIP_InternVLForCausalLM.generate_image_batch


class BatchPipelineTests(unittest.TestCase):
    def make_pipeline(self, model):
        def process(images, return_tensors):
            self.assertIsInstance(images, list)
            self.assertTrue(all(image.size == (448, 448) for image in images))
            return SimpleNamespace(pixel_values=torch.zeros(len(images), 3, 448, 448))
        return CustomEditPipeline(FakeTokenizer(), model, process)

    def test_batch_cfg_masks_actions_and_all_images(self):
        model = FakeModel()
        pipeline = self.make_pipeline(model)
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++", use_flow_sigmas=True, prediction_type="flow_prediction"
        )
        samples = [
            ["11 long <image>", "31 <image>", Image.new("RGB", (5, 7))],
            ["12 <image>", "32 long <image>", Image.new("RGB", (6, 8))],
        ]
        with patch.object(model_module.DPMSolverMultistepScheduler, "from_pretrained", return_value=scheduler):
            images = pipeline.generate_batch(samples, actions=[torch.arange(5), torch.arange(5) + 10])
        self.assertEqual(len(images), 2)
        self.assertEqual(model.model.decode_calls, 1)
        self.assertTrue(all(isinstance(image, Image.Image) and image.size == (28, 28) for image in images))
        self.assertNotEqual(np.asarray(images[0]).mean(), np.asarray(images[1]).mean())
        self.assertEqual(model.actions_seen.tolist(), [list(range(5)), list(range(10, 15))] * 2)
        states, kwargs = model.sample_seen
        self.assertEqual(states[:, 0, 0].tolist(), [31, 32, 11, 12])
        self.assertEqual(states[:, 1, 0].tolist(), [10, 11, 10, 11])
        self.assertEqual(kwargs["encoder_attention_mask"].sum(dim=1).tolist(), [4, 5, 5, 4])
        self.assertTrue(kwargs["legacy_ordered_noise"])
        self.assertEqual(kwargs["num_inference_steps"], 20)

    def test_sixteen_images_with_base_config_and_five_dim_actions(self):
        model = FakeModel()
        self.assertFalse(hasattr(model.model.config, "action_dim"))
        pipeline = self.make_pipeline(model)
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++", use_flow_sigmas=True, prediction_type="flow_prediction"
        )
        samples = [
            [f"{11 + i} <image>", f"{31 + i} <image>", Image.new("RGB", (2, 2))]
            for i in range(16)
        ]
        actions = torch.arange(80).reshape(16, 5)
        with patch.object(model_module.DPMSolverMultistepScheduler, "from_pretrained", return_value=scheduler):
            self.assertEqual(len(pipeline.generate_batch(samples, actions=actions)), 16)
            self.assertTrue(torch.equal(model.actions_seen, torch.cat([actions, actions]).float()))
            with self.assertRaisesRegex(ValueError, r"Actions must have shape \[16, 5\]"):
                pipeline.generate_batch(samples, actions=torch.zeros(16, 4))

    def test_tail_batch_and_patch_mismatch(self):
        model = FakeModel()
        pipeline = self.make_pipeline(model)
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++", use_flow_sigmas=True, prediction_type="flow_prediction"
        )
        sample = [["11 <image>", "31 <image>", Image.new("RGB", (2, 2))]]
        with patch.object(model_module.DPMSolverMultistepScheduler, "from_pretrained", return_value=scheduler):
            self.assertEqual(len(pipeline.generate_batch(sample, actions=[torch.arange(5)])), 1)
            self.assertEqual(model.actions_seen.tolist(), [list(range(5))] * 2)
            with self.assertRaisesRegex(ValueError, "image tokens"):
                pipeline.generate_batch([["11 no image", "31 <image>", sample[0][2]]])
            with self.assertRaisesRegex(ValueError, "Actions must have shape"):
                pipeline.generate_batch(sample, actions=[torch.zeros(4)])

    def test_explicit_action_dim_keeps_strict_batch_shape(self):
        model = FakeModel()
        model.model.config.action_dim = 3
        pipeline = self.make_pipeline(model)
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++", use_flow_sigmas=True, prediction_type="flow_prediction"
        )
        samples = [
            ["11 <image>", "31 <image>", Image.new("RGB", (2, 2))],
            ["12 <image>", "32 <image>", Image.new("RGB", (2, 2))],
        ]
        with patch.object(model_module.DPMSolverMultistepScheduler, "from_pretrained", return_value=scheduler):
            self.assertEqual(len(pipeline.generate_batch(samples, actions=torch.zeros(2, 3))), 2)
            self.assertEqual(model.actions_seen.shape, (4, 3))
            with self.assertRaisesRegex(ValueError, r"Actions must have shape \[2, 3\]"):
                pipeline.generate_batch(samples, actions=torch.zeros(2, 5))

    def test_unvalidated_scheduler_rejected(self):
        model = FakeModel()
        pipeline = self.make_pipeline(model)
        scheduler = DPMSolverMultistepScheduler()
        samples = [["11 <image>", "31 <image>", Image.new("RGB", (2, 2))]] * 2
        with patch.object(model_module.DPMSolverMultistepScheduler, "from_pretrained", return_value=scheduler):
            with self.assertRaisesRegex(ValueError, "validated Sana"):
                pipeline.generate_batch(samples)


class NoiseTests(unittest.TestCase):
    def test_localization_two_images_replace_each_row_with_padding(self):
        class ActionHost:
            def __init__(self):
                self.config = SimpleNamespace(
                    vision_feature_layer=-1, vision_feature_select_strategy="default"
                )
                self.model = SimpleNamespace(
                    config=SimpleNamespace(action_dim=5, text_config=SimpleNamespace(hidden_size=2)),
                    vision_tower=SimpleNamespace(embeddings=SimpleNamespace(
                        patch_embedding=SimpleNamespace(weight=torch.zeros(1))
                    )),
                    language_model=FakeLanguageModel(),
                    action_dit_norm=lambda value: value,
                    action_dit=lambda inputs_embeds, **kwargs: SimpleNamespace(hidden_states=[inputs_embeds]),
                    action_out_proj=lambda value: torch.zeros(value.shape[0], 1, 5),
                )
                self.model.get_image_features = self.get_image_features

            def get_model(self):
                return self.model

            def get_image_features(self, pixel_values, **kwargs):
                values = pixel_values[:, 0, 0, 0]
                return torch.stack([values, values + 0.5], dim=1).unsqueeze(-1).repeat(1, 1, 2)

            def sample_noise(self, shape, device):
                return torch.zeros(shape, device=device, dtype=torch.bfloat16)

            def embed_action_suffix(self, x_t, time, **kwargs):
                return torch.zeros(x_t.shape[0], 1, 2), None

        host = ActionHost()
        image_token = model_module.IMAGE_TOKEN_IDX
        ids = torch.tensor([
            [1, image_token, image_token, image_token, image_token, 2, 0],
            [3, image_token, image_token, image_token, image_token, 2, 4],
        ])
        mask = (ids != 0).long()
        und = torch.tensor([10.0, 11.0]).view(2, 1, 1, 1).expand(2, 3, 2, 2)
        aux = torch.tensor([20.0, 21.0]).view(2, 1, 1, 1).expand(2, 3, 2, 2)
        generate = model_module.Unified_UniLIP_InternVLForCausalLM.generate_action2
        result = generate(host, input_ids=ids, attention_mask=mask, und_image=und,
                          aux_image=aux, num_steps=1, legacy_ordered_noise=True)
        self.assertEqual(result.shape, (2, 1, 5))
        inputs = host.model.language_model.inputs
        self.assertEqual(inputs[:, 1:5, 0].tolist(),
                         [[10.0, 10.5, 20.0, 20.5], [11.0, 11.5, 21.0, 21.5]])
        self.assertEqual(host.model.language_model.mask.sum(dim=1).tolist(), [6, 7])

        # The aggregate number of placeholders is unchanged, but neither row is valid.
        mismatched = ids.clone()
        mismatched[0] = torch.tensor([1, image_token, image_token, 2, 0, 0, 0])
        mismatched[1] = torch.tensor([3, image_token, image_token, image_token,
                                      image_token, image_token, image_token])
        with self.assertRaisesRegex(ValueError, "Each row needs 2 und image tokens"):
            generate(host, input_ids=mismatched, attention_mask=(mismatched != 0).long(),
                     und_image=und, aux_image=aux, num_steps=1, legacy_ordered_noise=True)

        aux_mismatch = ids.clone()
        aux_mismatch[0] = torch.tensor([1, image_token, image_token, image_token,
                                        image_token, image_token, 2])
        with self.assertRaisesRegex(ValueError, "Each row needs 2 aux image tokens"):
            generate(host, input_ids=aux_mismatch, attention_mask=(aux_mismatch != 0).long(),
                     und_image=und, aux_image=aux, num_steps=1, legacy_ordered_noise=True)

        with self.assertRaisesRegex(ValueError, "aux image tokens are present without aux image input"):
            generate(host, input_ids=ids, attention_mask=mask, und_image=und,
                     aux_image=None, num_steps=1, legacy_ordered_noise=True)

        host.model.get_image_features = lambda pixel_values, **kwargs: torch.zeros(1, 2, 2)
        with self.assertRaisesRegex(ValueError, "image features must have batch dimension 2"):
            generate(host, input_ids=ids, attention_mask=mask, und_image=und,
                     aux_image=aux, num_steps=1, legacy_ordered_noise=True)

        host.model.get_image_features = lambda pixel_values, **kwargs: torch.zeros(2, 2, 3)
        with self.assertRaisesRegex(ValueError, "hidden dimensions do not match"):
            generate(host, input_ids=ids, attention_mask=mask, und_image=und,
                     aux_image=aux, num_steps=1, legacy_ordered_noise=True)

    def test_image_sampler_runs_all_twenty_steps_as_one_batch(self):
        class FakeDit:
            def __init__(self):
                self.shapes = []

            def __call__(self, latents, timestep, encoder_hidden_states,
                         encoder_attention_mask, return_dict):
                self.shapes.append((tuple(latents.shape), tuple(encoder_hidden_states.shape),
                                    tuple(encoder_attention_mask.shape)))
                return (torch.zeros_like(latents),)

        dit = FakeDit()
        host = SimpleNamespace(get_model=lambda: SimpleNamespace(dit=dit))
        sampler = model_module.Unified_UniLIP_InternVLForCausalLM.sample_image_latents
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++", use_flow_sigmas=True, prediction_type="flow_prediction"
        )
        result = sampler(
            host, torch.zeros(4, 3, 2), scheduler,
            encoder_attention_mask=torch.ones(4, 3, dtype=torch.bool),
            generator=torch.Generator().manual_seed(1), num_inference_steps=20,
            legacy_ordered_noise=True,
        )
        self.assertEqual(result.shape, (2, 32, 16, 16))
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(len(scheduler.timesteps), 20)
        self.assertEqual(dit.shapes, [((4, 32, 16, 16), (4, 3, 2), (4, 3))] * 20)

    def test_image_noise_matches_sequential_single_calls_and_rng(self):
        class NoiseHost:
            def get_model(self):
                return SimpleNamespace(dit=None)

        host = NoiseHost()
        scheduler = SimpleNamespace(timesteps=[])
        scheduler.set_timesteps = lambda steps: None
        sampler = model_module.Unified_UniLIP_InternVLForCausalLM.sample_image_latents
        states = torch.zeros(6, 2, 2)
        first = torch.Generator().manual_seed(123)
        second = torch.Generator().manual_seed(123)
        actual = sampler(host, states, scheduler, generator=first, legacy_ordered_noise=True)
        expected = torch.cat([
            model_module.randn_tensor((1, 32, 16, 16), generator=second, device="cpu", dtype=torch.float32)
            for _ in range(3)
        ])
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(first.get_state(), second.get_state()))
        with self.assertRaisesRegex(ValueError, "Initial noise shape"):
            sampler(host, states, scheduler, initial_noise=torch.zeros(2, 32, 16, 16))

    def test_localization_noise_matches_legacy_calls_and_global_rng(self):
        class ActionHost:
            def __init__(self):
                self.config = SimpleNamespace(
                    vision_feature_layer=-1, vision_feature_select_strategy="default"
                )
                self.model = SimpleNamespace(
                    config=SimpleNamespace(action_dim=5, text_config=SimpleNamespace(hidden_size=2)),
                    vision_tower=SimpleNamespace(embeddings=SimpleNamespace(
                        patch_embedding=SimpleNamespace(weight=torch.zeros(1))
                    )),
                    language_model=FakeLanguageModel(),
                    action_dit_norm=lambda value: value,
                    action_dit=lambda inputs_embeds, **kwargs: SimpleNamespace(hidden_states=[inputs_embeds]),
                    action_out_proj=lambda value: torch.zeros(value.shape[0], 1, 5),
                )

            def get_model(self):
                return self.model

            def sample_noise(self, shape, device):
                return model_module.Unified_UniLIP_InternVLForCausalLM.sample_noise(self, shape, device)

            def embed_action_suffix(self, x_t, time, **kwargs):
                return torch.zeros(x_t.shape[0], 1, 2), None

        host = ActionHost()
        ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
        mask = (ids != 0).long()
        unused_generator = torch.Generator().manual_seed(999)
        generator_before = unused_generator.get_state().clone()
        torch.manual_seed(123)
        actual = model_module.Unified_UniLIP_InternVLForCausalLM.generate_action2(
            host, input_ids=ids, attention_mask=mask, num_steps=1,
            generator=unused_generator, legacy_ordered_noise=True,
        )
        actual_state = torch.get_rng_state().clone()
        torch.manual_seed(123)
        expected = torch.cat([host.sample_noise((1, 1, 5), device=ids.device) for _ in range(2)])
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_state, torch.get_rng_state()))
        self.assertTrue(torch.equal(generator_before, unused_generator.get_state()))

        torch.manual_seed(456)
        single = model_module.Unified_UniLIP_InternVLForCausalLM.generate_action2(
            host, input_ids=ids[:1], attention_mask=mask[:1], num_steps=1,
        )
        single_state = torch.get_rng_state().clone()
        torch.manual_seed(456)
        expected_single = host.sample_noise((1, 1, 5), device=ids.device)
        self.assertTrue(torch.equal(single, expected_single))
        self.assertTrue(torch.equal(single_state, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
