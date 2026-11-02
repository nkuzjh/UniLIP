"""CPU checks for the fixed temporal supplement queue."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_gradient_diagnostics as original
from scripts import run_gradient_temporal_supplement as supplement


class TemporalSupplementTests(unittest.TestCase):
    def test_plan_is_four_fixed_comparable_items(self):
        config, items = supplement.load_plan()
        self.assertEqual([(item.experiment, item.checkpoint_step, item.source_offset)
                          for item in items],
                         [(experiment, 1200, offset)
                          for experiment in supplement.EXPERIMENTS for offset in supplement.OFFSETS])
        self.assertEqual({item.preferred_microbatch for item in items}, {2})
        self.assertEqual({item.source_seed for item in items}, {20261102})
        self.assertEqual({item.noise_seeds for item in items}, {(17, 29)})
        self.assertEqual({item.measurement_mode for item in items}, {"component_sum"})
        command = original.command_for(items[0], python=config["python"], output=original.ROOT / "x.json",
                                       device=config["device"], microbatch=2, gpu_memory_fraction=0.5)
        self.assertEqual(command[command.index("--microbatch-size") + 1], "2")
        self.assertEqual(command[command.index("--accumulations") + 1], "64")

    def test_original_anchors_have_required_identity_and_freeze(self):
        config, items = supplement.load_plan()
        anchors = supplement.validate_anchors(config, items)
        self.assertEqual(set(anchors), {item.name for item in items})
        self.assertTrue(all(anchor["microbatch_size"] == 2 and anchor["accumulations"] == 64
                            for anchor in anchors.values()))

    def test_source_mismatch_and_routed_aux_are_rejected(self):
        config, items = supplement.load_plan()
        item = items[0]
        anchor = supplement.validate_anchors(config, [item])[item.name]
        report = deepcopy(anchor)
        report["metadata"]["saved_tensor_offload"] = {
            "enabled": True, "mechanism": "torch.autograd.graph.save_on_cpu", "pin_memory": True,
            "scope": "model.forward saved tensors, including total.backward replay forwards",
            "entrypoint": config["wrapper"], "entrypoint_sha256": "wrapper-hash",
            "original_probe_sha256": supplement._sha256(
                supplement.ROOT / "scripts/probe_multitask_gradients.py")}
        self.assertEqual(supplement._paired_checks(report, anchor, item, wrapper_hash="wrapper-hash"), [])
        report["source_rows"][0]["file_frame"] = "wrong-frame"
        self.assertTrue(any("source identity" in error for error in
                            supplement._paired_checks(report, anchor, item,
                                                      wrapper_hash="wrapper-hash")))
        report = deepcopy(anchor)
        report["results"]["17"]["aux_freeze"]["details"][0]["routed_parameters"] = 1
        self.assertTrue(any("routed parameter" in error for error in
                            supplement._strict_aux(report, item)))

    def test_default_plan_never_launches_or_writes(self):
        with patch.object(supplement, "run_queue", side_effect=AssertionError("launched")), \
             patch.object(supplement, "_write_json", side_effect=AssertionError("wrote")):
            self.assertEqual(len(supplement.main([])), 4)

    def test_config_cannot_lower_microbatch(self):
        config = supplement.DEFAULT_CONFIG.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(config.replace("microbatch: 2", "microbatch: 1"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "microbatch2"):
                supplement.load_plan(path)

    def test_unsigned_existing_report_cannot_be_overwritten(self):
        config, items = supplement.load_plan()
        with tempfile.TemporaryDirectory() as directory:
            config = dict(config, output_root=directory)
            output = original.report_path(Path(directory), items[0])
            output.parent.mkdir(parents=True)
            output.write_text("{}", encoding="utf-8")
            wrapper = str(original.path_in_repo(config["wrapper"]).resolve())
            with patch.object(supplement, "validate_offload_acceptance", return_value={"accepted": True}), \
                 patch.object(supplement, "validate_anchors", return_value={items[0].name: {}}), \
                 patch.object(supplement, "input_hashes", return_value={
                     wrapper: supplement.EXPECTED_WRAPPER_SHA256}), \
                 patch.object(supplement, "_assert_frozen"), \
                 patch.object(original, "_run_command", side_effect=AssertionError("launched")):
                with self.assertRaisesRegex(RuntimeError, "Unsigned"):
                    supplement.run_queue(supplement.DEFAULT_CONFIG, config, items[:1],
                                         python=config["python"], max_wait_seconds=0, device_index=0)
            self.assertEqual(output.read_text(encoding="utf-8"), "{}")

    def test_execute_requires_gpu_acceptance_file_before_launch(self):
        config, items = supplement.load_plan()
        with tempfile.TemporaryDirectory() as directory:
            config = dict(config, offload_acceptance=str(Path(directory) / "missing.json"))
            with patch.object(original, "_run_command", side_effect=AssertionError("launched")):
                with self.assertRaises(FileNotFoundError):
                    supplement.run_queue(supplement.DEFAULT_CONFIG, config, items[:1],
                                         python=config["python"], max_wait_seconds=0, device_index=0)

    def test_queue_lock_rejects_second_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            with supplement._queue_lock(Path(directory)):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with supplement._queue_lock(Path(directory)):
                        pass


if __name__ == "__main__":
    unittest.main()
