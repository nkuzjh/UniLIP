"""CPU-only contract checks for the exp39-derived exp40 Seen10 runner."""
from __future__ import annotations

import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from csgo_datasets.inference_batching import execution_contract

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_exp40_seen10", REPO / "scripts/run_exp40_seen10.py")
assert SPEC and SPEC.loader
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def option(command, flag):
    return command[command.index(flag) + 1]


class Exp40RunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        configs = self.root / "csgo_configs"
        configs.mkdir()
        for name in (*RUN.EXPERIMENTS, "exp39"):
            shutil.copyfile(REPO / "csgo_configs" / f"{name}.yaml", configs / f"{name}.yaml")
        report = self.root / "data/csgo_benchmark_v2/minimal_dataset_report.json"
        report.parent.mkdir(parents=True)
        shutil.copyfile(REPO / "data/csgo_benchmark_v2/minimal_dataset_report.json", report)
        root_patch = patch.object(RUN, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def args(self, command="train", experiment="exp40_s2_P4", *extra):
        return RUN.parser().parse_args([command, "--experiment", experiment, *extra])

    def config(self, experiment="exp40_s2_P4", *extra):
        return RUN.resolve_config(self.args("train", experiment, *extra), for_training=True)

    def source(self):
        source = RUN.path_in_repo(RUN.SOURCE)
        source.parent.mkdir(parents=True)
        source.write_bytes(b"synthetic source header")
        (source.parent / "trainer_state.json").write_text(json.dumps({"global_step": 12000, "max_steps": 19550}))
        out = self.root / "outputs/csgo_1b/exp39"
        recipe = RUN.read_yaml(self.root / "csgo_configs/exp39.yaml")
        (out / "resolved_config.yaml").write_text(yaml.safe_dump(recipe))
        runtime = {"epochs": 50, "source_batch": 128, "checkpoint_steps": [12000, 16000, 19550],
                   "save_total_limit": 3, "max_steps": 19550, "warmup_ratio": .003, "warmup_steps": None}
        (out / "experiment_plan.json").write_text(json.dumps({"experiment": "exp39", "runtime": runtime,
                                                                  "fingerprint": RUN.fingerprint(recipe, runtime),
                                                                  "source_checkpoint": None}))
        return source

    def managed(self, experiment="exp40_s2_P4", step=4000, *, full=False):
        extra = ("--asset-mode", "full", "--data-dir", "alternate/full_assets") if full else ()
        args = self.args("train", experiment, *extra)
        config = RUN.resolve_config(args, for_training=True)
        _, plan, out = RUN.training_plan(args, config)
        out.mkdir(parents=True)
        plan["source_checkpoint"] = {"path": RUN.SOURCE, "sha256": "a" * 64, "step": 12000}
        (out / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        (out / "experiment_plan.json").write_text(json.dumps(plan))
        checkpoint = out / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": 7550}))
        for name in ("model.safetensors", "training_args.bin", "scheduler.pt", "optimizer.pt", "rng_state.pth"):
            (checkpoint / name).touch()
        return checkpoint, config, plan

    def test_three_fixed_recipes_and_horizon(self):
        for arm in RUN.EXPERIMENTS:
            with self.subTest(arm=arm):
                config = self.config(arm)
                RUN.validate_config(config)
                p4, u2 = arm != "exp40_s2_U2", arm != "exp40_s2_P4"
                self.assertEqual(config["alpha_loc_loss"], 4.0)
                self.assertIs(config["is_loc_perception_loss"], p4)
                self.assertIs(config["is_loc_aux_loss"], u2)
                self.assertIs(config["is_aux_loc_combined_em_unc_loss"], u2)
                self.assertEqual(config["alpha_loc_perception_loss"], 0.0)
                self.assertEqual(config["alpha_loc_aux_loss"], 0.0)
                self.assertEqual(config.get("alpha_loc_perception_schedule_steps"), [0, 503] if p4 else None)
                self.assertEqual(config.get("alpha_loc_perception_schedule_values"), [0.0, .05] if p4 else None)
                self.assertEqual(config.get("alpha_loc_aux_schedule_steps"), [0, 503] if u2 else None)
                self.assertEqual(config.get("alpha_loc_aux_schedule_values"), [0.0, 1.0] if u2 else None)
                self.assertFalse(config["loc_perception_use_attention_weight"])
                self.assertEqual(config["aux_loc_combined_num_samples"], 2)
                self.assertEqual(config["aux_loc_combined_candidate_tau"], 1.0)
                self.assertEqual(config["finetune_init_ckpt_path"], RUN.SOURCE)
                self.assertEqual(config["training_stop_after_step"], 4000)
                self.assertEqual(config["training_checkpoint_steps"], [4000, 7550])
                self.assertIs(config["save_final_model_at_output_root"], False)
                self.assertEqual(config["lr_scheduler_kwargs"], {"min_lr_rate": .1})
                for lr in RUN.LR_STAGE2:
                    self.assertEqual(config[lr], RUN.LR_STAGE2[lr])
                commands, plan, _ = RUN.training_plan(self.args("train", arm), config)
                command = commands[0]
                for flag, value in (("--max_steps", "7550"), ("--warmup_steps", "252"),
                                    ("--save_strategy", "no"), ("--save_total_limit", "2"),
                                    ("--save_only_model", "False"), ("--gradient_accumulation_steps", "32")):
                    self.assertEqual(option(command, flag), value)
                self.assertEqual(plan["cumulative_steps"], {"4000": 16000, "7550": 19550})
                self.assertEqual(plan["evaluation_steps"], {"validation": [4000, 7550], "test": [4000, 7550]})
        changed = self.config("exp40_s2_P4")
        changed["alpha_loc_loss"] = 8.0
        with self.assertRaises(ValueError):
            RUN.validate_config(changed)

    def test_stop_resume_fingerprint_batch_and_assets(self):
        first = self.config()
        no_stop = self.config("exp40_s2_P4", "--stop-after-step", "0")
        self.assertNotIn("training_stop_after_step", no_stop)
        self.assertEqual(RUN.fingerprint(first, {}), RUN.fingerprint(no_stop, {}))
        for stop in (1, 3000, 6000, 7551):
            with self.subTest(stop=stop), self.assertRaises(ValueError):
                self.config("exp40_s2_P4", "--stop-after-step", str(stop))
        full = self.config("exp40_s2_P4", "--asset-mode", "full", "--data-dir", "alternate/full_assets")
        self.assertIsNone(full["benchmark_v2_asset_manifest"])
        self.assertEqual(full["data_dir"], "alternate/full_assets")
        args = self.args("train", "exp40_s2_P4", "--cuda-devices", "0,1", "--micro-batch", "8")
        cmd, plan, _ = RUN.training_plan(args, RUN.resolve_config(args, for_training=True))
        self.assertEqual(option(cmd[0], "--gradient_accumulation_steps"), "8")
        self.assertEqual(plan["runtime"]["source_batch"], 128)
        with self.assertRaises(ValueError):
            args = self.args("train", "exp40_s2_P4", "--cuda-devices", "0,1", "--micro-batch", "3")
            RUN.training_plan(args, RUN.resolve_config(args, for_training=True))

    def test_source_identity_exact_exp39_runtime_and_alpha(self):
        source = self.source()
        with patch.object(RUN, "tensor_header", return_value={"tensor_count": 2830}):
            result = RUN.preflight_stage2_source(self.config("exp40_s2_both"))
        self.assertEqual(result["path"], RUN.SOURCE)
        self.assertEqual(result["sha256"], RUN.digest_file(source))
        self.assertEqual(result["source_max_steps"], 19550)
        with patch.object(RUN, "tensor_header", return_value={}):
            state = source.parent / "trainer_state.json"
            state.write_text(json.dumps({"global_step": 16000, "max_steps": 19550}))
            with self.assertRaisesRegex(ValueError, "global_step=12000"):
                RUN.preflight_stage2_source(self.config())
            state.write_text(json.dumps({"global_step": 12000, "max_steps": 19550}))
            plan_path = self.root / "outputs/csgo_1b/exp39/experiment_plan.json"
            plan = json.loads(plan_path.read_text())
            plan["runtime"]["checkpoint_steps"] = [4000, 8000, 12000]
            plan["fingerprint"] = RUN.fingerprint(RUN.read_yaml(self.root / "outputs/csgo_1b/exp39/resolved_config.yaml"), plan["runtime"])
            plan_path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "runtime differs"):
                RUN.preflight_stage2_source(self.config())
            plan["runtime"]["checkpoint_steps"] = [12000, 16000, 19550]
            plan["fingerprint"] = RUN.fingerprint(RUN.read_yaml(self.root / "outputs/csgo_1b/exp39/resolved_config.yaml"), plan["runtime"])
            plan_path.write_text(json.dumps(plan))
            recipe_path = self.root / "outputs/csgo_1b/exp39/resolved_config.yaml"
            recipe = RUN.read_yaml(recipe_path)
            recipe["alpha_loc_loss"] = 8.0
            recipe_path.write_text(yaml.safe_dump(recipe))
            with self.assertRaisesRegex(ValueError, "fixed recipe differs in alpha_loc_loss"):
                RUN.preflight_stage2_source(self.config())
            recipe["alpha_loc_loss"] = 4.0
            recipe["exp37_experiment"] = "exp37"
            recipe_path.write_text(yaml.safe_dump(recipe))
            with self.assertRaisesRegex(ValueError, "managed exp39"):
                RUN.preflight_stage2_source(self.config())
            wrong = self.config()
            wrong["finetune_init_ckpt_path"] = "outputs/csgo_1b/exp37/checkpoint-12000/model.safetensors"
            with self.assertRaisesRegex(ValueError, "exp39@12000"):
                RUN.preflight_stage2_source(wrong)

    def test_source_hash_binding_siblings_and_own_resume(self):
        source = self.source()
        with patch.object(RUN, "tensor_header", return_value={}):
            record = RUN.preflight_stage2_source(self.config())
        sibling = self.root / "outputs/csgo_1b/exp40_s2_U2"
        sibling.mkdir(parents=True)
        (sibling / "experiment_plan.json").write_text(json.dumps({"experiment": "exp40_s2_U2",
                                                               "source_checkpoint": {**record, "sha256": "b" * 64}}))
        with patch.object(RUN, "tensor_header", return_value={}):
            with self.assertRaisesRegex(ValueError, "sibling"):
                RUN.preflight_stage2_source(self.config())
        (sibling / "experiment_plan.json").write_text(json.dumps({"experiment": "exp40_s2_U2",
                                                               "source_checkpoint": record}))
        with patch.object(RUN, "tensor_header", return_value={}):
            RUN.preflight_stage2_source(self.config())
        (sibling / "experiment_plan.json").unlink()
        ckpt, _, plan = self.managed()
        args = self.args("train", "exp40_s2_P4", "--resume", "--stop-after-step", "0")
        resumed = RUN.resolve_config(args, for_training=True)
        source.unlink()
        _, result, _ = RUN.training_plan(args, resumed)
        self.assertEqual(result["source_checkpoint"], plan["source_checkpoint"])
        self.assertEqual(result["resume_checkpoint"], RUN.relative(ckpt))
        (ckpt / "rng_state.pth").unlink()
        with self.assertRaisesRegex(ValueError, "RNG state"):
            RUN.training_plan(args, resumed)
        (ckpt / "rng_state.pth").touch()
        (ckpt / "scheduler.pt").unlink()
        with self.assertRaisesRegex(ValueError, "optimizer/scheduler"):
            RUN.training_plan(args, resumed)
        (ckpt / "scheduler.pt").touch()
        extra = ckpt.parent / "checkpoint-6000"
        extra.mkdir()
        with self.assertRaisesRegex(ValueError, "Unexpected exp40 checkpoint"):
            RUN.training_plan(args, resumed)

    def test_evaluation_routes_and_disabled_heads(self):
        for step in (4000, 7550):
            for split, expected in (("validation", ["eval_csgo_loc.py", "eval_csgo.py"]),
                                    ("test", ["eval_csgo_loc.py", "eval_csgo.py", "eval_csgo.py"])):
                args = self.args("infer", "exp40_s2_both", "--checkpoint-step", str(step), "--split", split)
                commands, path, config = RUN.evaluation_plan(args, RUN.resolve_config(args, for_training=False))
                self.assertEqual([Path(c[1]).name for c in commands], expected)
                self.assertEqual(option(commands[0], "--test-loc-batch-size"), "32")
                self.assertEqual([option(c, "--test-gen-batch-size") for c in commands[1:]], ["16"] * (len(commands)-1))
                self.assertIn(f"checkpoint-{step}/{split}", str(path))
                self.assertEqual(config["ckpt_path"], f"outputs/csgo_1b/exp40_s2_both/checkpoint-{step}/model.safetensors")
                self.assertEqual(config["benchmark_v2_split"], "seen_validation" if split == "validation" else "seen_discrete_test")
                self.assertFalse(config["is_loc_aux_loss"])
                self.assertFalse(config["is_aux_loc_combined_em_unc_loss"])
                self.assertFalse(config["is_loc_perception_loss"])
                snapshots = RUN.evaluation_snapshots(args, path, config)
                self.assertEqual(len(snapshots), len(commands))
        args = self.args("infer", "exp40_s2_U2", "--checkpoint-step", "final", "--split", "validation")
        self.assertIn("checkpoint-7550", str(RUN.evaluation_plan(args, RUN.resolve_config(args, for_training=False))[1]))
        args = self.args("infer", "exp40_s2_U2", "--checkpoint-step", "4000", "--split", "validation", "--task", "gen_continuous")
        with self.assertRaisesRegex(ValueError, "Continuous generation requires"):
            RUN.evaluation_plan(args, RUN.resolve_config(args, for_training=False))
        args = self.args("metrics", "exp40_s2_U2", "--checkpoint-step", "4000", "--split", "validation", "--task", "gen")
        commands, _, _ = RUN.evaluation_plan(args, RUN.resolve_config(args, for_training=False))
        self.assertEqual(len(commands), 11)
        self.assertTrue(all("seen_validation" in cmd for cmd in commands))

    def test_batch_one_configs_are_isolated_by_route(self):
        args = self.args("infer", "exp40_s2_both", "--checkpoint-step", "4000", "--task", "all",
                         "--test-loc-batch-size", "1", "--test-gen-batch-size", "1")
        commands, path, config = RUN.evaluation_plan(args, RUN.resolve_config(args, for_training=False))
        snapshots = RUN.evaluation_snapshots(args, path, config)
        self.assertEqual(len(snapshots), 3)
        config_paths = [RUN.path_in_repo(option(command, "--csgo_config")) for command in commands]
        self.assertEqual(set(config_paths), set(snapshots))
        for config_path, command in zip(config_paths, commands):
            self.assertEqual(config_path.parent, RUN.path_in_repo(option(command, "--output_dir")))
        continuous = [value for key, value in snapshots.items() if key.parent.name == "continuous"]
        self.assertEqual(continuous[0]["benchmark_v2_split"], "seen_continuous")

    def test_execute_evaluation_requires_managed_checkpoint_and_provenance(self):
        ckpt, config, plan = self.managed()
        args = self.args("infer", "exp40_s2_P4", "--checkpoint-step", "4000", "--execute")
        RUN.evaluation_plan(args, RUN.resolve_config(args, for_training=False))
        (ckpt.parent / "experiment_plan.json").write_text(json.dumps({**plan, "source_checkpoint": None}))
        with self.assertRaisesRegex(ValueError, "source provenance"):
            RUN.evaluation_plan(args, RUN.resolve_config(args, for_training=False))
        (ckpt.parent / "experiment_plan.json").write_text(json.dumps(plan))
        args = self.args("infer", "exp40_s2_P4", "--checkpoint-step", "final", "--execute")
        with self.assertRaises(ValueError):
            RUN.evaluation_plan(args, RUN.resolve_config(args, for_training=False))
        final = ckpt.parent / "checkpoint-7550"
        final.mkdir()
        (final / "trainer_state.json").write_text(json.dumps({"global_step": 7550, "max_steps": 7550}))
        (final / "model.safetensors").touch()
        RUN.evaluation_plan(args, RUN.resolve_config(args, for_training=False))

    def test_read_only_default_and_executed_batch_provenance(self):
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        with patch.object(RUN.subprocess, "run") as launch, patch("sys.stdout", io.StringIO()):
            self.assertEqual(RUN.main(["train", "--experiment", "exp40_s2_P4"]), 0)
            self.assertEqual(RUN.main(["infer", "--experiment", "exp40_s2_P4", "--checkpoint-step", "final"]), 0)
            self.assertEqual(RUN.main(["metrics", "--experiment", "exp40_s2_P4", "--checkpoint-step", "4000"]), 0)
            launch.assert_not_called()
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")))
        self.managed()
        manifest_file = self.root / "data/csgo_benchmark_v2/benchmark_manifest.json"
        shutil.copyfile(REPO / "data/csgo_benchmark_v2/benchmark_manifest.json", manifest_file)
        args = ["--experiment", "exp40_s2_P4", "--checkpoint-step", "4000", "--split", "validation",
                "--task", "localization", "--execute"]
        with patch.object(RUN.subprocess, "run") as launch, patch("sys.stdout", io.StringIO()):
            self.assertEqual(RUN.main(["infer", *args]), 0)
            launch.assert_called_once()
        output = self.root / "outputs_loc/benchmark_v2/exp40_s2_P4/checkpoint-4000/validation/seed_42/infer_bs32"
        config_path = output / "evaluation_config.yaml"
        config = RUN.read_yaml(config_path)
        manifest = {"config_path": str(config_path.resolve()), "benchmark_v2_manifest": str(manifest_file),
                    "benchmark_v2_split": "seen_validation", "maps": RUN.MAPS, "sample_count": 5000,
                    "checkpoint": str(RUN.path_in_repo(config["ckpt_path"])),
                    "ckpt_path": str(RUN.path_in_repo(config["ckpt_path"])), "seed": 42,
                    "benchmark_v2_support_seed": None, "benchmark_v2_shots_per_map": None,
                    "inference_execution": execution_contract("loc", 32)}
        manifest_path = output / "inference_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        metric = {"XY_Dist": 1.0, "Z_Dist": 2.0, "Pitch_Dist": 3.0, "Yaw_Dist": 4.0}
        summary = RUN.aggregates.build_localization_summary(
            manifest=manifest_file, split="seen_validation", maps=RUN.MAPS,
            per_map={name: metric for name in RUN.MAPS}, metrics_macro_map=metric,
            inference_provenance={"path": str(manifest_path), "payload": manifest},
            checkpoint=manifest["checkpoint"], seed=42, support_seed=None,
            shots_per_map=None, sample_count=5000)
        (output / "benchmark_csgo_v2_loc.json").write_text(json.dumps(summary))
        with patch.object(RUN.subprocess, "run") as launch, patch("sys.stdout", io.StringIO()):
            self.assertEqual(RUN.main(["metrics", *args]), 0)
            launch.assert_not_called()
        manifest["inference_execution"] = execution_contract("loc", 1)
        manifest_path.write_text(json.dumps(manifest))
        with patch.object(RUN.subprocess, "run") as launch, patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()):
            self.assertEqual(RUN.main(["metrics", *args]), 1)
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
