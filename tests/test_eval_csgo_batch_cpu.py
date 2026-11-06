"""CPU regressions for the real evaluation batch entry points.

Extract the production definitions so the test does not import model weights,
CUDA extensions, or the inference scripts' eager model dependencies.
"""
import ast
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Union
import unittest

from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]


def production_definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(nodes) == len(names), (path, names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace


class BatchEvaluationCpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gen_ns = {"Image": Image, "torch": torch,
                  "add_template_for_inference": lambda text, **kwargs: [text, "negative"]}
        cls.gen = production_definitions(
            "eval_csgo.py", {"_generate_batch_images", "_generation_module_mode_metadata"}, gen_ns
        )
        cls.gen_manifest = staticmethod(production_definitions(
            "eval_csgo.py", {"_write_inference_manifest"},
            {"os": os, "json": json},
        )["_write_inference_manifest"])
        loc_ns = {
            "torch": torch, "Dict": Dict, "Mapping": Mapping, "Sequence": Sequence,
            "Union": Union, "List": List, "IGNORE_INDEX": -100,
            "dataclass": dataclass,
            "transformers": SimpleNamespace(PreTrainedTokenizer=object),
        }
        cls.loc = production_definitions(
            "eval_csgo_loc.py",
            {"DataCollatorForLoc", "_validate_loc_inference_batch", "_validate_loc_head_batch",
             "_generate_flow_locations"}, loc_ns,
        )
        cls.loc_manifest = staticmethod(production_definitions(
            "eval_csgo_loc.py", {"_write_inference_manifest"},
            {"os": os, "json": json},
        )["_write_inference_manifest"])
        coverage_ns = {"Sequence": Sequence, "Mapping": Mapping, "Any": Any,
                       "AggregationError": ValueError}
        cls.coverage = staticmethod(production_definitions(
            "scripts/aggregate_csgo_benchmark_v2_metrics.py",
            {"validate_localization_result_coverage"}, coverage_ns,
        )["validate_localization_result_coverage"])

    def test_generation_uses_one_batch_call_and_preserves_cross_map_tail(self):
        class Pipe:
            def __init__(self):
                self.calls = []

            def generate_batch(self, prompts, **kwargs):
                self.calls.append((prompts, kwargs))
                return [Image.new("RGB", (2, 2), (i, 0, 0)) for i in range(len(prompts))]

        samples = [
            {"map_name": name, "file_frame": frame, "raw_prompt": frame,
             "radar_img": Image.new("RGB", (2, 2)), "actions": torch.ones(1, 5)}
            for name, frame in (("map_a", "a0"), ("map_b", "b0"), ("map_a", "a1"))
        ]
        pipe = Pipe()
        generate = self.gen["_generate_batch_images"]
        first = generate(pipe, samples[:2], {"guidance_scale": 3}, object(), 2)
        tail = generate(pipe, samples[2:], {"guidance_scale": 3}, object(), 2)
        self.assertEqual([len(first), len(tail)], [2, 1])
        self.assertEqual([len(call[0]) for call in pipe.calls], [2, 1])
        self.assertEqual([prompt[0] for call in pipe.calls for prompt in call[0]], ["a0", "b0", "a1"])
        self.assertTrue(all(len(prompt) == 3 for call in pipe.calls for prompt in call[0]))
        self.assertEqual([len(call[1]["actions"]) for call in pipe.calls], [2, 1])

    def test_generation_batch_one_keeps_legacy_call_and_rejects_bad_actions(self):
        class Pipe:
            def __init__(self):
                self.legacy_calls = []

            def __call__(self, prompt, **kwargs):
                self.legacy_calls.append((prompt, kwargs))
                return Image.new("RGB", (2, 2))

            def generate_batch(self, *args, **kwargs):
                raise AssertionError("batch API must not run for batch one")

        sample = {"radar_img": Image.new("RGB", (2, 2)), "raw_prompt": "p",
                  "actions": torch.zeros(1, 5)}
        pipe = Pipe()
        result = self.gen["_generate_batch_images"](pipe, [sample], {"guidance_scale": 2}, object(), 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(pipe.legacy_calls), 1)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.gen["_generate_batch_images"](pipe, [{**sample, "actions": torch.zeros(5)}],
                                                 {"guidance_scale": 2}, object(), 2)
        with self.assertRaisesRegex(ValueError, "mixed action"):
            self.gen["_generate_batch_images"](pipe, [sample, {**sample, "actions": None}],
                                                 {"guidance_scale": 2}, object(), 2)

    def test_generation_mode_metadata_reads_active_dropout_without_side_effects(self):
        model = torch.nn.Sequential(
            torch.nn.Dropout(p=0.25),
            torch.nn.Dropout2d(p=0.1),
            torch.nn.Dropout(p=0.0),
        )
        model.eval()
        model[0].train()
        model[2].train()
        modes_before = [module.training for module in model.modules()]
        rng_before = torch.random.get_rng_state().clone()
        metadata = self.gen["_generation_module_mode_metadata"](model)
        self.assertEqual(metadata["model_mode_policy"], "preserve_loaded_module_modes_v1")
        self.assertEqual(metadata["active_dropout_modules"], [{"name": "0", "p": 0.25}])
        self.assertTrue(metadata["stochastic_layers_batch_dependent"])
        self.assertEqual([module.training for module in model.modules()], modes_before)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        model.eval()
        self.assertFalse(self.gen["_generation_module_mode_metadata"](model)["stochastic_layers_batch_dependent"])

    def test_localization_collator_flow_tail_and_coverage(self):
        tokenizer = SimpleNamespace(model_max_length=512, pad_token_id=0)
        collate = self.loc["DataCollatorForLoc"](tokenizer=tokenizer)
        rows = [
            {"ids": frame, "map_name": map_name, "input_ids": torch.tensor([1, 2]),
             "labels": torch.tensor([-100, -100]), "und_image": torch.full((1, 3, 2, 2), i),
             "aux_image": torch.full((1, 3, 2, 2), i + 10), "actions": torch.zeros(1, 5),
             "pose_dict": {}, "raw_prompt": frame, "map_img": Image.new("RGB", (2, 2)),
             "fps_img": Image.new("RGB", (2, 2)), "z_range": {"min_z": 0, "max_z": 1}}
            for i, (map_name, frame) in enumerate((("map_a", "a0"), ("map_b", "b0"), ("map_a", "a1")))
        ]

        class Model:
            def __init__(self):
                self.calls = []

            def generate_action2(self, ids, mask, labels, fps, maps, **kwargs):
                self.calls.append((fps.clone(), maps.clone(), kwargs))
                return torch.zeros(ids.shape[0], 1, 5)

        model = Model()
        results = []
        for group in (rows[:2], rows[2:]):
            batch = collate(group)
            self.assertEqual(self.loc["_validate_loc_inference_batch"](batch, 2), len(group))
            pred = self.loc["_generate_flow_locations"](
                model, batch["input_ids"], batch["attention_mask"], batch["labels"],
                batch["und_image"], batch["aux_image"], 2,
            )
            results.extend({"map": name, "file_frame": frame} for name, frame in
                           zip(batch["map_name"], batch["ids"]))
            self.assertEqual(pred.shape, (len(group), 5))
        self.assertEqual([call[0].shape[0] for call in model.calls], [2, 1])
        self.assertTrue(all(call[2] == {"num_steps": 10, "legacy_ordered_noise": True}
                            for call in model.calls))
        self.assertEqual(model.calls[0][0][1, 0, 0, 0].item(), 1)
        self.assertEqual(model.calls[0][1][1, 0, 0, 0].item(), 11)
        self.coverage(results, ["map_a", "map_b"],
                      [{"map": r["map_name"], "file_frame": r["ids"]} for r in rows])
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            self.coverage(results[:-1], ["map_a", "map_b"],
                          [{"map": r["map_name"], "file_frame": r["ids"]} for r in rows])

        self.loc["_generate_flow_locations"](
            model, torch.ones(1, 2), torch.ones(1, 2), torch.ones(1, 2),
            torch.zeros(1, 3, 2, 2), torch.zeros(1, 3, 2, 2), 1,
        )
        self.assertEqual(model.calls[-1][2], {"num_steps": 10})

    def test_localization_rejects_token_heads_and_misaligned_images(self):
        for head in ("lm_bin_ce", "lm_st_ext_vocab"):
            with self.assertRaisesRegex(ValueError, "only support"):
                self.loc["_validate_loc_head_batch"](head, 2)
            self.loc["_validate_loc_head_batch"](head, 1)
        batch = {"ids": ["a", "b"], "map_name": ["x", "y"], "pose_dict": [{}, {}],
                 "raw_prompt": ["a", "b"], "map_img": [None, None], "fps_img": [None, None],
                 "z_range": [{}, {}], "input_ids": torch.ones(2, 2),
                 "attention_mask": torch.ones(2, 2), "labels": torch.ones(2, 2),
                 "und_image": [torch.zeros(1, 3, 2, 2)], "aux_image": torch.zeros(2, 3, 2, 2),
                 "actions": torch.zeros(2, 1, 5)}
        with self.assertRaisesRegex(ValueError, "und_image"):
            self.loc["_validate_loc_inference_batch"](batch, 2)

    def test_manifest_keeps_batch_one_schema_and_records_batch_execution(self):
        for writer in (self.gen_manifest, self.loc_manifest):
            with tempfile.TemporaryDirectory() as temporary_dir:
                writer(temporary_dir, "config.yaml", {}, ["map_a"], 1, "ckpt.pt", 42)
                manifest_path = Path(temporary_dir) / "inference_manifest.json"
                legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertNotIn("inference_execution", legacy)
                writer(temporary_dir, "config.yaml", {}, ["map_a"], 2, "ckpt.pt", 42,
                       inference_execution={"batch_size": 2, "rng_policy": "legacy_ordered_initial_noise_draws_v1"})
                batched = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(batched["inference_execution"]["batch_size"], 2)


if __name__ == "__main__":
    unittest.main()
