"""CPU-only checks for the exp32_gen checkpoint batch migration."""

import ast
import json
import logging
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "train_csgo.py"
FUNCTIONS = {
    "_get_current_world_size",
    "_get_current_train_batch_unit",
    "_positive_resume_int",
    "_checkpoint_world_size",
    "_resume_gradient_accumulation",
    "_synchronize_resume_deepspeed_batch",
    "_load_checkpoint_trainer_state",
    "prepare_resume_batch_size_override",
    "validate_resume_total_optimization_steps",
}


def load_resume_functions():
    # Importing train_csgo loads models and initializes GPU-related libraries.
    tree = ast.parse(SOURCE.read_text())
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS)
        or (isinstance(node, ast.ClassDef) and node.name == "ResumeBatchMetadataCallback")
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {
        "os": os, "json": json, "re": re, "logging": logging,
        "Optional": __import__("typing").Optional, "Dict": __import__("typing").Dict,
        "TrainerCallback": object,
        "_RESUME_BATCH_METADATA_FILENAME": "unilip_resume_batch.json",
        "_load_checkpoint_training_args": lambda path: None,
        "_sanitize_trainer_state_for_resume": lambda state, path: False,
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


class ResumeBatchTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkpoint = Path(self.temp.name) / "checkpoint-4000"
        self.checkpoint.mkdir()
        (self.checkpoint / "trainer_state.json").write_text(
            json.dumps({"max_steps": 13650, "global_step": 4000, "train_batch_size": 128})
        )
        self.f = load_resume_functions()

    def args(self, world_size=2, per_device=64, gas=2):
        return SimpleNamespace(
            world_size=world_size, per_device_train_batch_size=per_device,
            gradient_accumulation_steps=gas, resume_preserve_effective_batch_size=True,
            resume_rewrite_trainer_state_batch_size=True,
            resume_target_effective_batch_size=None, resume_checkpoint_train_batch_size=None,
            resume_checkpoint_gradient_accumulation_steps=1,
            resume_checkpoint_world_size=None,
        )

    def prepare(self, args, config=None):
        with patch.dict(os.environ, {"WORLD_SIZE": str(args.world_size)}):
            return self.f["prepare_resume_batch_size_override"](
                args, str(self.checkpoint), config or {}
            )

    def test_single_to_dual_and_subsequent_dual_resume(self):
        (self.checkpoint / "rng_state.pth").touch()
        first = self.args()
        self.prepare(first)
        self.assertEqual(first.gradient_accumulation_steps, 1)
        self.assertEqual(64 * first.world_size * first.gradient_accumulation_steps, 128)

        (self.checkpoint / "rng_state.pth").unlink()
        (self.checkpoint / "rng_state_0.pth").touch()
        (self.checkpoint / "rng_state_1.pth").touch()
        second = self.args(gas=2)
        self.prepare(second)
        self.assertEqual(second.gradient_accumulation_steps, 1)

    def test_metadata_and_explicit_source_override(self):
        metadata = self.checkpoint / "unilip_resume_batch.json"
        metadata.write_text(json.dumps({"version": 1, "world_size": 2}))
        self.assertEqual(self.f["_checkpoint_world_size"](str(self.checkpoint)), 2)
        self.assertEqual(self.f["_checkpoint_world_size"](str(self.checkpoint), 1), 1)

    def test_saved_metadata_supports_later_dual_resume(self):
        args = self.args(gas=1)
        args.output_dir = self.temp.name
        state = SimpleNamespace(is_world_process_zero=True, global_step=4000)
        callback = self.f["ResumeBatchMetadataCallback"]()
        callback.on_save(args, state, None)
        self.assertEqual(self.f["_checkpoint_world_size"](str(self.checkpoint)), 2)
        metadata = json.loads((self.checkpoint / "unilip_resume_batch.json").read_text())
        self.assertEqual(metadata["effective_batch_size"], 128)

    def test_missing_source_world_size_never_uses_current(self):
        with self.assertRaisesRegex(ValueError, "Cannot infer checkpoint world_size safely"):
            self.prepare(self.args())

    def test_explicit_source_override_works_without_rng_or_metadata(self):
        args = self.args()
        self.prepare(args, {"resume_checkpoint_world_size": 1})
        self.assertEqual(args.gradient_accumulation_steps, 1)

    def test_indivisible_effective_batch_rejected(self):
        (self.checkpoint / "rng_state.pth").touch()
        with self.assertRaisesRegex(ValueError, "Cannot preserve resume effective batch size exactly"):
            self.prepare(self.args(per_device=96), {"resume_target_effective_batch_size": 128})

    def test_fresh_training_does_not_change_accumulation(self):
        args = self.args(gas=3)
        self.assertIsNone(self.f["prepare_resume_batch_size_override"](args, None, {}))
        self.assertEqual(args.gradient_accumulation_steps, 3)

    def test_deepspeed_resolved_values_are_updated(self):
        (self.checkpoint / "rng_state.pth").touch()
        config = {
            "train_micro_batch_size_per_gpu": 64,
            "gradient_accumulation_steps": 2,
            "train_batch_size": 256,
        }
        owner = SimpleNamespace(config=config)
        plugin = SimpleNamespace(hf_ds_config=owner, deepspeed_config=config, gradient_accumulation_steps=2)
        args = self.args()
        args.hf_deepspeed_config = owner
        args.deepspeed_plugin = plugin
        self.prepare(args)
        self.assertEqual(config, {
            "train_micro_batch_size_per_gpu": 64,
            "gradient_accumulation_steps": 1,
            "train_batch_size": 128,
        })
        self.assertEqual(plugin.gradient_accumulation_steps, 1)

    def test_max_steps_mismatch_is_strict_and_reports_actual_batch(self):
        args = self.args(gas=1)

        class Trainer:
            def __init__(self, args):
                self.args = args

            def get_train_dataloader(self):
                return object()

            def set_initial_training_values(self, args, dataloader):
                return (50, 137, 50000, 50000, 1, 782, 6860)

            def get_total_train_batch_size(self, args):
                return args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps

        with self.assertRaisesRegex(ValueError, "current_total_train_batch_size=128") as caught:
            self.f["validate_resume_total_optimization_steps"](Trainer(args), str(self.checkpoint))
        self.assertIn("current_total_optimization_steps=6860", str(caught.exception))
        self.assertIn("checkpoint_trainer_state.max_steps=13650", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
