import json
from pathlib import Path
import tempfile
import unittest

from csgo_datasets.inference_batching import (
    InferenceExecutionTracker, execution_contract, guard_inference_output,
    positive_batch_size, validate_execution_metadata,
)


class InferenceBatchingTests(unittest.TestCase):
    def test_positive_integer_contract(self):
        for value in (0, -1, True, 1.5, None, "1.5", "invalid"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                positive_batch_size(value)
        self.assertEqual(positive_batch_size("16"), 16)

    def test_order_hash_independent_of_batch_partition(self):
        ids = [{"map_name": "a", "file_frame": str(i)} for i in range(5)]
        single = InferenceExecutionTracker("gen", 1)
        batched = InferenceExecutionTracker("gen_continuous", 3)
        for identity in ids:
            single.update([identity])
        batched.update(ids[:3])
        batched.update(ids[3:])
        a, b = single.finish(), batched.finish()
        self.assertEqual(a["sample_order_sha256"], b["sample_order_sha256"])
        self.assertEqual((b["sample_count"], b["batch_count"], b["last_batch_size"]), (5, 2, 2))
        self.assertEqual(b["max_batch_size"], 3)
        with self.assertRaises(ValueError):
            batched.finish(batch_size=16)
        with self.assertRaises(ValueError):
            batched.update([])

    def test_manifest_execution_is_checked_with_legacy_exception(self):
        validate_execution_metadata({}, "loc", 1)
        with self.assertRaises(ValueError):
            validate_execution_metadata({}, "loc", 32)
        payload = {"inference_execution": execution_contract("gen", 16)}
        validate_execution_metadata(payload, "gen_continuous", 16)
        for task, batch in [("gen", 1), ("gen", 8), ("loc", 16)]:
            with self.subTest(task=task, batch=batch), self.assertRaises(ValueError):
                validate_execution_metadata(payload, task, batch)

    def test_partial_run_pins_execution_and_portable_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = dict(split="crossmap_query_test", seed=42, maps=["cs_office"])
            ckpt = "outputs/csgo_1b/exp37_fs_gen_cs_office/shot_100/seed_0/model.safetensors"
            guard_inference_output(root, "gen", 16, checkpoint=ckpt, **common)
            self.assertTrue((root / "inference_execution_plan.json").is_file())
            guard_inference_output(root, "gen", 16, checkpoint="/other/server/" + ckpt, **common)
            with self.assertRaises(ValueError):
                guard_inference_output(root, "gen", 8, checkpoint=ckpt, **common)
            with self.assertRaises(ValueError):
                guard_inference_output(root, "gen", 16, checkpoint=ckpt.replace("shot_100", "shot_50"), **common)

    def test_legacy_results_are_not_reinterpreted_or_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"checkpoint": "model.safetensors", "benchmark_v2_split": "seen_discrete_test",
                        "seed": 42, "maps": ["de_train"]}
            path = root / "inference_manifest.json"
            path.write_text(json.dumps(manifest))
            before = path.read_bytes()
            args = dict(checkpoint="model.safetensors", split="seen_discrete_test", maps=["de_train"])
            guard_inference_output(root, "loc", 1, **args)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((root / "inference_execution_plan.json").exists())
            with self.assertRaises(ValueError):
                guard_inference_output(root, "loc", 32, **args)

    def test_unidentified_partial_images_require_new_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "gen_imgs/cs_office"
            images.mkdir(parents=True)
            (images / "frame.jpg").write_bytes(b"partial image")
            with self.assertRaisesRegex(ValueError, "Unidentified"):
                guard_inference_output(root, "gen", 16)
            self.assertFalse((root / "inference_execution_plan.json").exists())


if __name__ == "__main__":
    unittest.main()
