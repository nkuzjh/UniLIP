"""CPU protocol tests for exp32_loc@19500 Unseen-4 localization."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from scripts import run_exp32_loc_unseen4 as runner


def args(*parts):
    return runner.parser().parse_args(parts)


def model_config() -> dict:
    return {"model_type": "unified_unilip", "is_lora": True,
            "llm_train_mode": "lora", "enable_language_model_lora": True,
            "enable_gen_head_lora": False, "enable_loc_head_lora": True,
            "freeze_inactive_head": True, "is_loc_aux_loss": False,
            "is_loc_perception_loss": False, "img_size": 224,
            "action_dim": 5, "mllm_hf_path": "OpenGVLab/InternVL3-1B-hf"}


class Exp32LocTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "outputs/csgo_1b/exp32_loc/checkpoint-19500/model.safetensors"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"small mock tensor header")
        (self.source.parent / "config.json").write_text(json.dumps(model_config()), encoding="utf-8")
        (self.source.parent / "trainer_state.json").write_text(
            json.dumps({"global_step": 19500, "max_steps": 20000}), encoding="utf-8")
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(runner, "ROOT", self.root))
        stack.enter_context(patch.object(runner, "SOURCE", self.source))
        stack.enter_context(patch.object(runner.legacy, "ROOT", self.root))
        stack.enter_context(patch.object(runner.seen37, "tensor_header", return_value={"tensor_count": 1}))

    def source_evidence(self):
        return runner.source_checkpoint(strict=True)[1]

    def save_snapshot(self, map_name="cs_office", shots=20, *, zero=False, cli=None):
        cli = cli or args("infer", "--zero-shot" if zero else "--map",
                          *([] if zero else [map_name]), "--shots" if not zero else "--task",
                          *( [str(shots)] if not zero else ["loc"]))
        config = runner.recipe(cli, None if zero else map_name,
                               None if zero else shots, zero_shot=zero)
        plan = runner.plan_identity(config, self.source_evidence(), cli,
                                    None if zero else map_name, None if zero else shots)
        base = runner.zero_dir() if zero else runner.model_dir(map_name, shots)
        base.mkdir(parents=True)
        (base / ("zero_shot_config.yaml" if zero else "resolved_config.yaml")).write_text(
            runner.yaml_text(config), encoding="utf-8")
        (base / ("zero_shot_plan.json" if zero else "experiment_plan.json")).write_text(
            runner.json_text(plan), encoding="utf-8")
        if not zero:
            (base / "model.safetensors").write_bytes(b"small final model")
            (base / "trainer_state.json").write_text(
                json.dumps({"global_step": 400, "max_steps": 400}), encoding="utf-8")
        return config, plan, base

    def test_validate_all_sixteen_points_and_recipe(self):
        _, writes, details = runner.plan(args("validate"))
        self.assertEqual((writes, details["validated_points"]), ([], 16))
        for map_name in runner.MAPS:
            for shots in runner.SHOTS:
                config = runner.recipe(args("train", "--map", map_name, "--shots", str(shots)),
                                       map_name, shots, zero_shot=False)
                self.assertEqual(config["train_maps"], [map_name])
                self.assertEqual(config["benchmark_v2_shots_per_map"], shots)
                self.assertEqual(config["finetune_init_ckpt_path"], runner.rel(self.source))
                self.assertTrue(all(config[k] == 2e-5 for k in runner.LOW_LR))
                self.assertTrue(all(config[k] == 1e-4 for k in runner.HIGH_LR))
                self.assertEqual(config["lr_scheduler_kwargs"], {"min_lr_rate": 0.1})
                self.assertEqual(config["lora_dropout"], 0.05)
                self.assertFalse(config["disable_lora_dropout"])
                self.assertTrue(config["save_final_model_at_output_root"])
                self.assertNotIn("csgo_global_batch_size", config)
                self.assertNotIn("warmup_steps", config)

    def test_train_cli_batch_optimizer_and_ports(self):
        for shots in runner.SHOTS:
            cli = args("train", "--map", "de_palacio", "--shots", str(shots))
            commands, writes, details = runner.plan(cli)
            command = commands[0]
            self.assertIn("--standalone", command)
            self.assertIn("--nproc_per_node=1", command)
            self.assertIn("--nnodes=1", command)
            self.assertEqual(command[command.index("--per_device_train_batch_size") + 1], str(shots))
            self.assertEqual(command[command.index("--gradient_accumulation_steps") + 1], "1")
            self.assertEqual(command[command.index("--max_steps") + 1], "400")
            self.assertEqual(command[command.index("--save_strategy") + 1], "no")
            self.assertEqual(command[command.index("--eval_strategy") + 1], "no")
            self.assertEqual(command[command.index("--optim") + 1], "adamw_torch")
            self.assertEqual(command[command.index("--max_grad_norm") + 1], "1.0")
            self.assertEqual(details["source"]["verified"], False)
            self.assertTrue(all(runner.FAMILY in str(path) for path, _ in writes))
        fixed = args("train", "--map", "de_palacio", "--master-port", "29638")
        command = runner.plan(fixed)[0][0]
        self.assertIn("--master_port=29638", command)
        self.assertNotIn("--standalone", command)

    def test_source_exact_step_files_head_and_hash(self):
        evidence = self.source_evidence()
        self.assertTrue(evidence["verified"])
        self.assertEqual(evidence["trainer_state"]["max_steps"], 20000)
        self.assertEqual(evidence["step"], 19500)
        self.assertEqual(evidence["header"], {"tensor_count": 1})
        for missing in (self.source, self.source.parent / "config.json",
                        self.source.parent / "trainer_state.json"):
            with self.subTest(missing=missing.name):
                moved = missing.with_suffix(missing.suffix + ".hold")
                missing.rename(moved)
                try:
                    with self.assertRaisesRegex(ValueError, "Missing exact checkpoint-19500"):
                        runner.source_checkpoint(strict=True)
                finally:
                    moved.rename(missing)
        state = self.source.parent / "trainer_state.json"
        original = state.read_text()
        state.write_text(json.dumps({"global_step": 16000, "max_steps": 20000}))
        with self.assertRaisesRegex(ValueError, "global_step"):
            runner.source_checkpoint(strict=True)
        state.write_text(original)
        config_path = self.source.parent / "config.json"
        config_path.write_text(json.dumps({**model_config(), "enable_gen_head_lora": True}))
        with self.assertRaisesRegex(ValueError, "loc-only"):
            runner.source_checkpoint(strict=True)

    def test_zero_shot_counts_no_support_and_batch(self):
        config, plan, _ = self.save_snapshot(zero=True)
        cli = args("infer", "--zero-shot", "--task", "loc")
        commands, writes, details = runner.plan(cli)
        command = commands[0]
        self.assertEqual(command[command.index("--benchmark_v2_maps") + 1:
                                 command.index("--test-loc-batch-size")], list(runner.MAPS))
        self.assertNotIn("--benchmark_v2_support_seed", command)
        self.assertNotIn("--benchmark_v2_shots_per_map", command)
        self.assertEqual(command[command.index("--test-loc-batch-size") + 1], "16")
        self.assertIn("infer_bs16", str(writes[0][0]))
        evaluation_plan = json.loads(next(body for path, body in writes
                                          if path.name == "evaluation_plan.json"))
        self.assertEqual(evaluation_plan["inference_execution"],
                         runner.execution_contract("loc", 16))
        self.assertIn("outputs_loc", str(runner.zero_dir()))
        self.assertIsNone(details["shots"])
        self.assertNotIn("benchmark_v2_support_seed", config)
        self.assertIsNone(plan["shots"])
        with self.assertRaisesRegex(ValueError, "no shots"):
            runner.plan(args("infer", "--zero-shot", "--shots", "20"))

    def test_full_assets_and_saved_plan_changes_rejected(self):
        cli = args("train", "--map", "cs_office", "--asset-mode", "full",
                   "--data-dir", "/data/full", "--model-path", "/model", "--pi05-path", "/pi05")
        config = runner.recipe(cli, "cs_office", 100, zero_shot=False)
        self.assertIsNone(config["benchmark_v2_asset_manifest"])
        self.assertEqual(config["data_dir"], "/data/full")
        self.assertEqual(config["model_name_or_path"], "/model")
        _, _, full_base = self.save_snapshot("de_golden", 50, cli=args(
            "infer", "--map", "de_golden", "--shots", "50", "--asset-mode", "full",
            "--data-dir", "/data/full", "--model-path", "/model", "--pi05-path", "/pi05"))
        self.assertEqual(runner.read_snapshot(args("infer", "--map", "de_golden", "--shots", "50"),
                                              "de_golden", 50, strict=False)[2], full_base)
        with self.assertRaisesRegex(ValueError, "requires --data-dir"):
            runner.plan(args("train", "--map", "cs_office", "--asset-mode", "full"))
        with self.assertRaisesRegex(ValueError, "requires --asset-mode full"):
            runner.plan(args("train", "--map", "cs_office", "--data-dir", "/data"))
        _, _, base = self.save_snapshot(shots=20)
        read = args("infer", "--map", "cs_office", "--shots", "20")
        runner.read_snapshot(read, "cs_office", 20, strict=True)
        with self.assertRaisesRegex(ValueError, "asset mode"):
            runner.read_snapshot(args("infer", "--map", "cs_office", "--shots", "20",
                                      "--asset-mode", "full", "--data-dir", "/data"),
                                 "cs_office", 20, strict=False)
        plan_path = base / "experiment_plan.json"
        payload = json.loads(plan_path.read_text())
        payload["source"]["path"] = "outputs/csgo_1b/exp32_loc/checkpoint-16000/model.safetensors"
        plan_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "source"):
            runner.read_snapshot(read, "cs_office", 20, strict=False)
        payload["source"]["path"] = runner.rel(self.source)
        payload["effective_batch"]["configured_gas"] = 32
        plan_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "effective batch"):
            runner.read_snapshot(read, "cs_office", 20, strict=False)
        payload["effective_batch"]["configured_gas"] = 1
        payload["source"]["source_recipe_sha256"] = "0" * 64
        plan_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "source recipe"):
            runner.read_snapshot(read, "cs_office", 20, strict=False)

    def test_dry_infer_without_snapshot_does_not_hash_source_or_write(self):
        cli = args("infer", "--map", "de_vertigo", "--shots", "10")
        with patch.object(runner, "digest", side_effect=AssertionError("unexpected hash")), \
             patch.object(runner.subprocess, "run") as run:
            commands, writes, details = runner.plan(cli)
            self.assertEqual(len(commands), 1)
            self.assertEqual(details["source"]["verified"], False)
            self.assertFalse(any(path.exists() for path, _ in writes))
            run.assert_not_called()

    def test_status_missing_and_complete(self):
        first = runner.plan(args("status"))[2]["points"][0]
        self.assertFalse(first["snapshot"])
        self.assertFalse(first["model"])
        self.save_snapshot("cs_office", 100)
        first = runner.plan(args("status"))[2]["points"][0]
        self.assertTrue(first["snapshot"])
        self.assertTrue(first["completed_400"])

    def test_metrics_missing_summary_fails_without_gpu(self):
        self.save_snapshot(shots=20)
        with patch.object(runner.legacy, "validate_inference", return_value={}), \
             patch.object(runner.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "missing"):
                runner.plan(args("metrics", "--map", "cs_office", "--shots", "20", "--execute"))
            run.assert_not_called()
            runner.plan(args("metrics", "--map", "cs_office", "--shots", "20"))

    def test_localization_summary_enforces_8000_zero_and_2000_few(self):
        summary_path = self.root / "loc_summary.json"
        common = {"seed": 42, "split": runner.SPLIT, "kind": "localization",
                  "checkpoint": runner.rel(self.source), "manifest": runner.MATRIX["manifest"]}
        with patch.object(runner.legacy.aggregates, "_validate_localization_summary"), \
             patch.object(runner.legacy.aggregates, "_validate_localization_summary_for_map"):
            for maps, shots, support, count in ((runner.MAPS, None, None, 8000),
                                                (("cs_office",), 20, 0, 2000)):
                payload = {**common, "maps": list(maps), "sample_count": count,
                           "shots_per_map": shots, "support_seed": support}
                summary_path.write_text(json.dumps(payload))
                runner.legacy.validate_loc_summary(summary_path, self.source, maps,
                                                   runner.SPLIT, shots, support)
                payload["sample_count"] = count - 1
                summary_path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "protocol/provenance"):
                    runner.legacy.validate_loc_summary(summary_path, self.source, maps,
                                                       runner.SPLIT, shots, support)

    def test_aggregate_four_maps_and_zero_read_only(self):
        few = args("aggregate", "--shots", "20")
        command = runner.plan(few)[0][0]
        self.assertIn("map-models", command)
        self.assertIn(runner.FAMILY + "_{map}", " ".join(command))
        self.save_snapshot(zero=True)
        with patch.object(runner, "digest", side_effect=AssertionError("unexpected hash")):
            runner.plan(args("aggregate", "--zero-shot"))
        with patch.object(runner.legacy, "validate_inference") as inference, \
             patch.object(runner.legacy, "validate_loc_summary") as summary:
            commands, writes, details = runner.plan(args("aggregate", "--zero-shot", "--execute"))
            self.assertEqual((commands, writes), ([], []))
            self.assertIn("benchmark_csgo_v2_loc.json", details["summary"])
            self.assertEqual(inference.call_args.args[3], runner.MAPS)
            self.assertEqual(summary.call_args.args[2], runner.MAPS)

    def test_execute_only_runs_mock_process_and_protects_output(self):
        dry = args("train", "--map", "cs_office", "--shots", "10")
        with patch.object(runner.subprocess, "run") as run:
            runner.plan(dry)
            run.assert_not_called()
        def complete(*_args, **_kwargs):
            base = runner.model_dir("cs_office", 10)
            (base / "model.safetensors").write_bytes(b"final")
            (base / "trainer_state.json").write_text(json.dumps({"global_step": 400, "max_steps": 400}))
        with patch.object(runner.subprocess, "run", side_effect=complete) as run, redirect_stdout(StringIO()):
            self.assertEqual(runner.main(["train", "--map", "cs_office", "--shots", "10", "--execute"]), 0)
            run.assert_called_once()
        base = runner.model_dir("cs_office", 10)
        self.assertTrue((base / "resolved_config.yaml").is_file())
        with patch.object(runner.subprocess, "run") as run, redirect_stderr(StringIO()):
            self.assertEqual(runner.main(["train", "--map", "cs_office", "--shots", "10", "--execute"]), 1)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
