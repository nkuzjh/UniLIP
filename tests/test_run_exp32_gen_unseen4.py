"""CPU checks for exp32_gen@12000 Unseen-4 transfer preparation."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

from scripts import run_exp32_gen_unseen4 as runner


def args(*parts):
    return runner.parser().parse_args(parts)


def fake_source(root: Path) -> Path:
    path = root / "exp32_gen/checkpoint-12000/model.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"small fixture, safetensors header mocked")
    (path.parent / "trainer_state.json").write_text(
        json.dumps({"global_step": 12000, "max_steps": 13650}), encoding="utf-8")
    # The real model config does not persist LoRA r/alpha/dropout or task mix.
    (path.parent / "config.json").write_text(json.dumps({
        "model_type": "unified_unilip", "is_lora": True, "llm_train_mode": "lora",
        "enable_language_model_lora": True, "enable_gen_head_lora": True,
        "enable_loc_head_lora": False, "freeze_inactive_head": True,
        "is_loc_aux_loss": False, "is_loc_perception_loss": False,
        "img_size": 224, "action_dim": 5,
        "mllm_hf_path": "OpenGVLab/InternVL3-1B-hf"}), encoding="utf-8")
    return path


HEADER = {"tensor_count": 2578, "lora_tensor_count": 876,
          "lora_ranks": {"llm_connector": 16, "other_active_adapters": 32}}


class Exp32GenUnseen4Tests(unittest.TestCase):
    def test_validate_and_sixteen_fresh_batch_plans(self):
        commands, writes, details = runner.plan(args("validate"))
        self.assertEqual((commands, writes, details["validated_points"]), ([], [], 16))
        for name in runner.MAPS:
            for shots in runner.SHOTS:
                with self.subTest(map=name, shots=shots):
                    cfg, plan, _ = runner.model_snapshot(
                        args("train", "--map", name, "--shots", str(shots)),
                        name, shots, strict_source=False)
                    self.assertEqual(cfg["train_maps"], [name])
                    self.assertEqual(plan["effective_batch"], {
                        "micro_batch": shots, "configured_gas": 1,
                        "microbatches_per_update": 1, "base_samples_per_update": shots,
                        "base_sample_exposures_400_steps": shots * 400})

    def test_recipe_cleans_source_resume_and_uses_strict_weight_only_init(self):
        with tempfile.TemporaryDirectory() as directory:
            template = runner.seen37.read_yaml(runner.BASE_RECIPE)
            template.update(resume_ckpt_path="old", resume_target_effective_batch_size=128,
                            csgo_global_batch_size=128, training_checkpoint_steps=[12000],
                            training_stop_after_step=4000, ckpt_path="wrong",
                            base_init_ckpt_path="wrong", finetune_init_ckpt_path="wrong",
                            lr_scheduler_kwargs={"min_lr": 1e-5})
            path = Path(directory) / "template.yaml"
            path.write_text(yaml.safe_dump(template), encoding="utf-8")
            with patch.object(runner, "BASE_RECIPE", path):
                cfg = runner.recipe(args("train", "--map", "cs_office", "--shots", "10"),
                                    "cs_office", 10, zero_shot=False)
        self.assertFalse(any(key.startswith(("resume_", "training_stop_", "training_checkpoint_"))
                             for key in cfg))
        for key in ("csgo_global_batch_size", "ckpt_path", "base_init_ckpt_path"):
            self.assertNotIn(key, cfg)
        self.assertEqual(cfg["finetune_init_ckpt_path"], runner.rel(runner.SOURCE))
        self.assertIs(cfg["finetune_init_strict"], True)
        self.assertEqual(cfg["lr_scheduler_kwargs"], {"min_lr_rate": 0.1})
        self.assertIs(cfg["save_final_model_at_output_root"], True)
        self.assertIs(cfg["csgo_loss_per_microbatch_mean"], True)
        self.assertIs(cfg["csgo_forward_sampler_epoch"], True)
        self.assertEqual((cfg["task_mix_ratio"], cfg["enable_loc_head_lora"]), (0.0, False))

    def test_source_readiness_exact_step_allows_later_training_horizon(self):
        with tempfile.TemporaryDirectory() as directory:
            source = fake_source(Path(directory))
            with patch.object(runner, "SOURCE", source), patch.object(
                    runner.seen37, "tensor_header", return_value=HEADER):
                _, evidence = runner.source_checkpoint(strict=True)
                self.assertIs(evidence["verified"], True)
                self.assertEqual(evidence["trainer_state"],
                                 {"global_step": 12000, "max_steps": 13650})
                self.assertEqual(evidence["sha256"], runner.legacy.sha256(source))
                state = source.parent / "trainer_state.json"
                state.write_text(json.dumps({"global_step": 11999, "max_steps": 13650}),
                                 encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "global_step is not 12000"):
                    runner.source_checkpoint(strict=True)
                state.unlink()
                with self.assertRaisesRegex(ValueError, "Missing exact source"):
                    runner.source_checkpoint(strict=True)

    def test_source_checkpoint_config_gating_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            source = fake_source(Path(directory))
            with patch.object(runner, "SOURCE", source), patch.object(
                    runner.seen37, "tensor_header", return_value=HEADER):
                self.assertIs(runner.source_checkpoint(strict=True)[1]["verified"], True)
                config = source.parent / "config.json"
                payload = json.loads(config.read_text())
                payload["enable_loc_head_lora"] = True
                config.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "gen-only"):
                    runner.source_checkpoint(strict=True)

    def test_zero_two_splits_counts_batch_and_no_support(self):
        commands, writes, _ = runner.plan(args("infer", "--zero-shot"))
        self.assertEqual(len(commands), 2)
        plans = {p.parent.name: json.loads(v) for p, v in writes
                 if p.name == "evaluation_plan.json"}
        self.assertEqual(plans["discrete"]["expected_samples"], 8000)
        self.assertEqual(plans["continuous"]["expected_samples"], 5120)
        self.assertEqual((plans["continuous"]["continuous_clips_per_map"],
                          plans["continuous"]["continuous_frames_per_clip"]), (20, 64))
        for command in commands:
            self.assertNotIn("--benchmark_v2_support_seed", command)
            self.assertNotIn("--benchmark_v2_shots_per_map", command)
            self.assertEqual(command[command.index("--test-gen-batch-size") + 1], "32")
        self.assertTrue(any("crossmap_query_test" in command for command in commands))
        self.assertTrue(any("crossmap_continuous" in command for command in commands))
        configs = [yaml.safe_load(v) for p, v in writes if p.name == "evaluation_config.yaml"]
        self.assertEqual(len(configs), 2)
        for cfg in configs:
            self.assertEqual((cfg["test_gen_batch_size"], cfg["batch_size"],
                              cfg["guidance_scale"], cfg["action_dim"],
                              cfg["mllm_hf_path"], cfg["img_size"]),
                             (32, 1, 4.5, 5, "OpenGVLab/InternVL3-1B-hf", 224))
            self.assertIs(cfg["disable_lora_dropout"], False)
            self.assertNotIn("benchmark_v2_support_seed", cfg)

    def test_few_infer_requires_final_400(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with self.assertRaisesRegex(ValueError, "model.safetensors.*trainer_state.json.*400/400.*earlier training log"):
                runner.final_model(out, strict=True)
            (out / "model.safetensors").write_bytes(b"model")
            state = out / "trainer_state.json"
            with self.assertRaisesRegex(ValueError, "trainer_state.json.*400/400"):
                runner.final_model(out, strict=True)
            state.write_text(json.dumps({"global_step": 399, "max_steps": 400}),
                             encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly 400.*399/400"):
                runner.final_model(out, strict=True)
            state.write_text(json.dumps({"global_step": 400, "max_steps": 400}),
                             encoding="utf-8")
            self.assertEqual(runner.final_model(out, strict=True), out / "model.safetensors")

    def test_status_reports_missing_snapshot_incomplete_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def output(map_name, shots):
                return root / map_name / str(shots)

            for shots in (50, 20, 10):
                out = output("cs_office", shots)
                out.mkdir(parents=True)
                (out / "resolved_config.yaml").write_text("fixture", encoding="utf-8")
                (out / "experiment_plan.json").write_text("{}", encoding="utf-8")
            for shots, step in ((20, 399), (10, 400)):
                out = output("cs_office", shots)
                (out / "model.safetensors").write_bytes(b"model")
                (out / "trainer_state.json").write_text(
                    json.dumps({"global_step": step, "max_steps": 400}), encoding="utf-8")
            with patch.object(runner, "model_dir", side_effect=output), patch.object(
                    runner, "source_checkpoint", return_value=(runner.SOURCE, {"verified": True})):
                _, _, details = runner.plan(args("status"))
            points = {(p["map"], p["shots"]): p for p in details["adaptations"]}
            missing = points["cs_office", 100]
            self.assertEqual(missing["training_status"], "missing")
            self.assertEqual(missing["global_step"], None)
            self.assertEqual(set(missing["missing_artifacts"]), {
                "resolved_config.yaml", "experiment_plan.json", "model.safetensors", "trainer_state.json"})
            snapshot = points["cs_office", 50]
            self.assertEqual(snapshot["training_status"], "snapshot_only")
            self.assertEqual(snapshot["missing_artifacts"], ["model.safetensors", "trainer_state.json"])
            incomplete = points["cs_office", 20]
            self.assertEqual((incomplete["training_status"], incomplete["global_step"],
                              incomplete["max_steps"], incomplete["missing_artifacts"]),
                             ("incomplete", 399, 400, []))
            complete = points["cs_office", 10]
            self.assertEqual((complete["training_status"], complete["global_step"],
                              complete["max_steps"], complete["missing_artifacts"]),
                             ("complete", 400, 400, []))

    def test_train_execute_checks_process_and_completed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fake_source(root)
            command = ["train", "--map", "cs_office", "--shots", "10", "--execute"]
            def out_dir(_map, _shots):
                return root / "adapt"

            def completed_run(*_args, **_kwargs):
                (out / "model.safetensors").write_bytes(b"model")
                (out / "trainer_state.json").write_text(
                    json.dumps({"global_step": 400, "max_steps": 400}), encoding="utf-8")
                return subprocess.CompletedProcess([], 0)

            with patch.object(runner, "validate_matrix"), patch.object(runner, "SOURCE", source), patch.object(
                    runner.seen37, "tensor_header", return_value=HEADER), patch.object(
                    runner, "model_dir", side_effect=out_dir), patch.object(
                    runner.subprocess, "run", side_effect=subprocess.CalledProcessError(1, ["train"])) as run, \
                    redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
                self.assertEqual(runner.main(command), 1)
                self.assertEqual(run.call_count, 1, errors.getvalue())
                self.assertIn("returned non-zero exit status", errors.getvalue())

            # A new output directory is required for each fresh adaptation attempt.
            for name, side_effect, expected in (
                    ("missing", None, 1), ("complete", completed_run, 0)):
                with self.subTest(name=name), tempfile.TemporaryDirectory() as output_root:
                    out = Path(output_root) / "adapt"
                    with patch.object(runner, "validate_matrix"), patch.object(runner, "SOURCE", source), patch.object(
                            runner.seen37, "tensor_header", return_value=HEADER), patch.object(
                            runner, "model_dir", return_value=out), patch.object(
                            runner.subprocess, "run", side_effect=side_effect) as run, \
                            redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
                        self.assertEqual(runner.main(command), expected)
                        self.assertEqual(run.call_count, 1)
                        if name == "missing":
                            self.assertIn("Missing completed 400-step adaptation", errors.getvalue())

    def test_torchrun_port_zero_is_standalone_and_fixed_port_preserved(self):
        out = Path("/tmp/exp32-test-only")
        cfg = runner.recipe(args("train", "--map", "cs_office", "--shots", "10"),
                            "cs_office", 10, zero_shot=False)
        for port in (0, 29740):
            with self.subTest(port=port):
                local = args("train", "--map", "cs_office", "--shots", "10",
                             "--master-port", str(port))
                command = runner.train_command(cfg, out, 10, local)
                self.assertEqual(command[command.index("--max_steps") + 1], "400")
                if port == 0:
                    self.assertIn("--standalone", command)
                    self.assertIn("--nnodes=1", command)
                    self.assertFalse(any(part.startswith("--master_port") for part in command))
                else:
                    self.assertIn("--master_port=29740", command)
                    self.assertNotIn("--standalone", command)
        for port in (-1, 65536):
            with self.subTest(invalid_port=port), self.assertRaisesRegex(ValueError, "valid master port"):
                runner.plan(args("train", "--map", "cs_office", "--shots", "10",
                                 "--master-port", str(port)))

    def test_strict_source_identity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fake_source(root)
            out = root / "adapt"
            local = args("train", "--map", "cs_office", "--shots", "10", "--execute")
            with patch.object(runner, "SOURCE", source), patch.object(
                    runner.seen37, "tensor_header", return_value=HEADER), patch.object(
                    runner, "model_dir", return_value=out):
                cfg, saved, _ = runner.model_snapshot(local, "cs_office", 10,
                                                       strict_source=True)
                out.mkdir()
                (out / "resolved_config.yaml").write_text(runner.yaml_text(cfg),
                                                            encoding="utf-8")
                saved["source"]["path"] = "outputs/csgo_1b/exp32_gen/checkpoint-4000/model.safetensors"
                (out / "experiment_plan.json").write_text(runner.json_text(saved),
                                                             encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "recipe/source hash"):
                    runner.read_model_snapshot(local, "cs_office", 10, strict_source=True)
                with patch.object(runner, "validate_matrix"):
                    with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                        runner.plan(local)

    def test_real_legacy_inference_schema_hash_batch_and_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = args("infer", "--zero-shot", "--asset-mode", "full",
                         "--data-dir", str(root / "data"))
            cfg = runner.recipe(local, None, None, zero_shot=True)
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"test-only weights")
            output = root / "eval"
            output.mkdir()
            task, split = "gen_continuous", "crossmap_continuous"
            ecfg = runner.legacy.eval_config(cfg, checkpoint, split, runner.MAPS, task, 32)
            cp = output / "evaluation_config.yaml"
            cp.write_text(runner.yaml_text(ecfg), encoding="utf-8")
            source = {"path": runner.rel(runner.SOURCE), "sha256": "a" * 64,
                      "step": 12000, "verified": True}
            eval_plan = {"checkpoint": runner.rel(checkpoint),
                         "checkpoint_sha256": runner.legacy.sha256(checkpoint),
                         "config_sha256": runner.legacy.sha256(cp), "split": split,
                         "maps": list(runner.MAPS), "seed": 42, "support_seed": None,
                         "shots": None, "asset_manifest": None, "source": source,
                         "inference_execution": runner.execution_contract(task, 32)}
            (output / "evaluation_plan.json").write_text(runner.json_text(eval_plan),
                                                          encoding="utf-8")
            manifest = {"benchmark_v2_split": split, "maps": list(runner.MAPS),
                        "sample_count": 5120, "seed": 42,
                        "benchmark_v2_support_seed": None,
                        "benchmark_v2_shots_per_map": None,
                        "inference_execution": runner.execution_contract(task, 32),
                        "config_path": str(cp),
                        "benchmark_v2_manifest": runner.MATRIX["manifest"],
                        "checkpoint": str(checkpoint), "ckpt_path": str(checkpoint)}
            mp = output / "inference_manifest.json"
            mp.write_text(runner.json_text(manifest), encoding="utf-8")
            self.assertEqual(runner.legacy.validate_inference(
                output, cfg, checkpoint, runner.MAPS, split, task, None, None,
                source, 32)["sample_count"], 5120)
            manifest["sample_count"] = 5119
            mp.write_text(runner.json_text(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split/maps/count"):
                runner.legacy.validate_inference(output, cfg, checkpoint, runner.MAPS,
                                                 split, task, None, None, source, 32)
            manifest["sample_count"] = 5120
            manifest["inference_execution"]["batch_size"] = 16
            mp.write_text(runner.json_text(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "batch size"):
                runner.legacy.validate_inference(output, cfg, checkpoint, runner.MAPS,
                                                 split, task, None, None, source, 32)

    def test_metric_batch_and_equal_map_aggregate_paths(self):
        cmd = runner.legacy.aggregate_command(runner.FAMILY, 10, "gen", False,
                                              batch_size=32)
        self.assertEqual(cmd[2], "map-models")
        self.assertIn("{map}", cmd[cmd.index("--input_pattern") + 1])
        self.assertEqual(cmd[cmd.index("--split") + 1], "crossmap_query_test")
        self.assertIn("infer_bs32", cmd[cmd.index("--output") + 1])
        metric = runner.legacy.metric_command(
            runner.recipe(args("metrics", "--map", "cs_office"),
                          "cs_office", 10, zero_shot=False),
            runner.eval_dir("cs_office", 10, "gen", False, 32),
            "crossmap_query_test", "gen", "cs_office")
        self.assertEqual(metric[1], "benchmark_csgo_v1.py")
        self.assertEqual(metric[metric.index("--batch_size") + 1], "1")

    def test_dry_run_does_not_write_or_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "new-adaptation"
            captured = StringIO()
            with patch.object(runner, "model_dir", return_value=out), patch.object(
                    runner.subprocess, "run", side_effect=AssertionError("executed")), \
                    redirect_stdout(captured):
                self.assertEqual(runner.main(["train", "--map", "cs_office",
                                              "--shots", "10"]), 0)
            self.assertFalse(out.exists())
            payload = json.loads(captured.getvalue())
            self.assertEqual(payload["mode"], "dry-run")
            self.assertEqual(len(payload["commands"]), 1)


if __name__ == "__main__":
    unittest.main()
