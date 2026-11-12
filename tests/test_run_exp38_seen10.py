"""CPU-only contract checks for the opt-in exp38 Seen10 runner."""
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
SPEC = importlib.util.spec_from_file_location("run_exp38_seen10", REPO / "scripts/run_exp38_seen10.py")
assert SPEC and SPEC.loader
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def option(command, flag):
    return command[command.index(flag) + 1]


class Exp38RunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        configs = self.root / "csgo_configs"
        configs.mkdir()
        for name in (*RUN.EXPERIMENTS, "exp37", "exp37_s2_both"):
            shutil.copyfile(REPO / "csgo_configs" / f"{name}.yaml", configs / f"{name}.yaml")
        report = self.root / "data/csgo_benchmark_v2/minimal_dataset_report.json"
        report.parent.mkdir(parents=True)
        shutil.copyfile(REPO / "data/csgo_benchmark_v2/minimal_dataset_report.json", report)
        root_patch = patch.object(RUN, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def args(self, command="train", experiment="exp38_s2_joint", *extra):
        return RUN.parser().parse_args([command, "--experiment", experiment, *extra])

    def config(self, experiment="exp38_s2_joint", *extra):
        return RUN.resolve_config(self.args("train", experiment, *extra), for_training=True)

    def source(self):
        source = RUN.path_in_repo(RUN.SOURCE)
        source.parent.mkdir(parents=True)
        source.write_bytes(b"fixture header")
        (source.parent / "trainer_state.json").write_text(json.dumps({"global_step": 12000, "max_steps": 19550}))
        out = self.root / "outputs/csgo_1b/exp37"
        recipe = RUN.read_yaml(self.root / "csgo_configs/exp37.yaml")
        (out / "resolved_config.yaml").write_text(yaml.safe_dump(recipe))
        runtime = {"epochs": 50, "source_batch": 128, "save_steps": 4000, "warmup_ratio": .003}
        (out / "experiment_plan.json").write_text(json.dumps({"experiment": "exp37", "runtime": runtime,
                                                                  "fingerprint": RUN.fingerprint(recipe, runtime)}))
        return source

    def managed(self, experiment="exp38_s2_joint", step=4000, *, full=False):
        extra = ("--asset-mode", "full", "--data-dir", "alternate/full_assets") if full else ()
        args = self.args("train", experiment, *extra)
        config = RUN.resolve_config(args, for_training=True)
        _, plan, out = RUN.training_plan(args, config)
        out.mkdir(parents=True)
        plan["source_checkpoint"] = {"path": RUN.SOURCE, "sha256": "a" * 64, "step": 12000}
        (out / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        (out / "experiment_plan.json").write_text(json.dumps(plan))
        ckpt = out / f"checkpoint-{step}"
        ckpt.mkdir()
        (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": step, "max_steps": 7550}))
        for filename in ("model.safetensors", "training_args.bin", "scheduler.pt", "rng_state.pth", "optimizer.pt"):
            (ckpt / filename).touch()
        return ckpt, config, plan

    def batched_metrics_fixture(self):
        """Saved batch outputs whose on-disk paths differ from their output aliases."""
        (self.root / "outputs").mkdir()
        for alias, target in (("outputs/csgo_1b", "real_train"),
                              ("outputs_loc", "real_loc"),
                              ("outputs_eval", "real_eval")):
            physical = self.root / target
            physical.mkdir()
            (self.root / alias).symlink_to(physical, target_is_directory=True)
        manifest_file = self.root / "data/csgo_benchmark_v2/benchmark_manifest.json"
        shutil.copyfile(REPO / "data/csgo_benchmark_v2/benchmark_manifest.json", manifest_file)
        ckpt, _, _ = self.managed()
        args = self.args("metrics", "exp38_s2_joint", "--checkpoint-step", "4000",
                         "--task", "all", "--test-loc-batch-size", "32",
                         "--test-gen-batch-size", "16", "--execute")
        config = RUN.resolve_config(args, for_training=False)
        _, legacy_path, evaluated = RUN.evaluation_plan(args, config)
        snapshots = RUN.evaluation_snapshots(args, legacy_path, evaluated)
        report = self.root / "data/csgo_benchmark_v2/minimal_dataset_report.json"
        asset_identity = {
            "asset_backend": "minimal",
            "asset_manifest_path": str(report),
            "asset_manifest_sha256": RUN.digest_file(report),
            "selected_images_sha256": json.loads(report.read_text())["selected_images"]["sha256"],
        }
        location = None
        for snapshot_path, snapshot_cfg in snapshots.items():
            snapshot_path.parent.mkdir(parents=True)
            snapshot_path.write_text(yaml.safe_dump(snapshot_cfg, sort_keys=False))
            task = "loc" if "outputs_loc" in snapshot_path.parts else "gen"
            batch_size = 32 if task == "loc" else 16
            count = 12800 if snapshot_cfg["benchmark_v2_split"] == "seen_continuous" else 20000
            payload = {
                "config_path": str(snapshot_path.resolve()),
                "benchmark_v2_manifest": str(manifest_file.resolve()),
                "benchmark_v2_split": snapshot_cfg["benchmark_v2_split"],
                "benchmark_v2_support_seed": snapshot_cfg.get("benchmark_v2_support_seed"),
                "benchmark_v2_shots_per_map": snapshot_cfg.get("benchmark_v2_shots_per_map"),
                "maps": RUN.MAPS,
                "sample_count": count,
                "checkpoint": str((ckpt / "model.safetensors").resolve()),
                "ckpt_path": str((ckpt / "model.safetensors").resolve()),
                "seed": 42,
                "inference_execution": execution_contract(task, batch_size),
                **asset_identity,
            }
            provenance_path = snapshot_path.parent / "inference_manifest.json"
            provenance_path.write_text(json.dumps(payload))
            if task == "loc":
                summary = RUN.aggregates.build_localization_summary(
                    manifest=str(manifest_file.resolve()), split="seen_discrete_test",
                    maps=RUN.MAPS,
                    per_map={name: {"L2_5D": 1.0} for name in RUN.MAPS},
                    metrics_macro_map={"L2_5D": 1.0},
                    inference_provenance={"path": str(provenance_path.resolve()), "payload": payload},
                    checkpoint=payload["checkpoint"], seed=42,
                    support_seed=payload["benchmark_v2_support_seed"],
                    shots_per_map=payload["benchmark_v2_shots_per_map"],
                    sample_count=count,
                )
                summary_path = snapshot_path.parent / "benchmark_csgo_v2_loc.json"
                summary_path.write_text(json.dumps(summary))
                location = (snapshot_path, provenance_path, summary_path)
        assert location is not None
        return location

    def test_fixed_recipes_and_training_horizon(self):
        for name in RUN.EXPERIMENTS:
            with self.subTest(name=name):
                config = self.config(name)
                RUN.validate_config(config)
                self.assertEqual(config["exp38_experiment"], name)
                self.assertEqual(config["training_checkpoint_steps"], [4000, 7550])
                self.assertIs(config["save_final_model_at_output_root"], False)
                self.assertEqual(config["training_stop_after_step"], 4000)
                self.assertEqual(config["finetune_init_ckpt_path"], RUN.SOURCE)
                command, plan, _ = RUN.training_plan(self.args("train", name), config)
                command = command[0]
                for flag, value in (("--max_steps", "7550"), ("--warmup_steps", "252"),
                                    ("--save_strategy", "no"), ("--save_total_limit", "2"),
                                    ("--save_only_model", "False"), ("--gradient_accumulation_steps", "32")):
                    self.assertEqual(option(command, flag), value)
                self.assertNotIn("--save_steps", command)
                self.assertEqual(plan["cumulative_steps"], {"4000": 16000, "7550": 19550})
                self.assertEqual(plan["runtime"]["checkpoint_steps"], [4000, 7550])
                self.assertEqual(config["lr_scheduler_kwargs"], {"min_lr_rate": .1})
                self.assertEqual(config["learning_rate"], 2e-5)
                self.assertEqual(config["action_dit_connector_lr"], 1e-4)
        joint, both = self.config(), self.config("exp38_s2_both")
        self.assertFalse(joint["is_loc_aux_loss"])
        self.assertFalse(joint["is_loc_perception_loss"])
        self.assertTrue(both["is_loc_aux_loss"])
        self.assertTrue(both["is_aux_loc_combined_em_unc_loss"])
        self.assertTrue(both["is_loc_perception_loss"])
        self.assertFalse(both["loc_perception_use_attention_weight"])
        self.assertEqual(both["alpha_loc_aux_schedule_steps"], [0, 503])
        self.assertEqual(both["alpha_loc_perception_schedule_steps"], [0, 503])
        base = RUN.read_yaml(self.root / "csgo_configs/exp37_s2_both.yaml")
        differences = {key for key in both.keys() | base.keys() if both.get(key) != base.get(key)}
        self.assertEqual(differences, {
            "exp37_experiment", "exp38_experiment", "finetune_init_ckpt_path",
            "training_checkpoint_steps", "save_final_model_at_output_root",
            "alpha_loc_aux_schedule_steps", "alpha_loc_perception_schedule_steps",
            "training_stop_after_step", "model_name_or_path",
        })

    def test_stop_and_fingerprint_preserve_full_horizon(self):
        first = self.config()
        explicit = self.config("exp38_s2_joint", "--stop-after-step", "4000")
        final_stop = self.config("exp38_s2_joint", "--stop-after-step", "7550")
        nonstop = self.config("exp38_s2_joint", "--stop-after-step", "0")
        self.assertEqual(explicit["training_stop_after_step"], 4000)
        self.assertEqual(final_stop["training_stop_after_step"], 7550)
        self.assertEqual(first, explicit)
        self.assertNotIn("training_stop_after_step", nonstop)
        a, pa, _ = RUN.training_plan(self.args(), first)
        _, pe, _ = RUN.training_plan(self.args("train", "exp38_s2_joint", "--stop-after-step", "4000"), explicit)
        _, pf, _ = RUN.training_plan(self.args("train", "exp38_s2_joint", "--stop-after-step", "7550"), final_stop)
        b, pb, _ = RUN.training_plan(self.args("train", "exp38_s2_joint", "--stop-after-step", "0"), nonstop)
        self.assertEqual(a, b)
        self.assertEqual(pa["fingerprint"], pe["fingerprint"])
        self.assertEqual(pa["fingerprint"], pf["fingerprint"])
        self.assertEqual(pa["fingerprint"], pb["fingerprint"])
        self.assertEqual(pb["runtime"]["max_steps"], 7550)
        self.assertEqual(pb["runtime"]["warmup_steps"], 252)
        for stop in (1, 3000, 6000, 7551, -1):
            with self.subTest(stop=stop), self.assertRaises(ValueError):
                self.config("exp38_s2_joint", "--stop-after-step", str(stop))

    def test_legacy_checkpoint_plan_is_rejected(self):
        config = self.config()
        config["training_checkpoint_steps"] = [3000, 7550]
        with self.assertRaisesRegex(ValueError, "recipe differs in fixed field training_checkpoint_steps"):
            RUN.validate_config(config)

    def test_source_preflight_accepts_intermediate_and_cross_backend(self):
        self.source()
        config = self.config("exp38_s2_joint", "--asset-mode", "full", "--data-dir", "alternate/full_assets")
        with patch.object(RUN, "tensor_header", return_value={"tensor_count": 2830}):
            result = RUN.preflight_stage2_source(config)
        self.assertEqual(result["step"], 12000)
        self.assertEqual(result["source_max_steps"], 19550)
        self.assertEqual(result["path"], RUN.SOURCE)
        self.assertEqual(len(result["sha256"]), 64)
        self.assertIsNone(config["benchmark_v2_asset_manifest"])

    def test_source_rejects_wrong_state_plan_and_recipe(self):
        source = self.source()
        config = self.config()
        with patch.object(RUN, "tensor_header", return_value={}):
            (source.parent / "trainer_state.json").write_text(json.dumps({"global_step": 12000, "max_steps": 12000}))
            with self.assertRaisesRegex(ValueError, "12000/max_steps=19550"):
                RUN.preflight_stage2_source(config)
            (source.parent / "trainer_state.json").write_text(json.dumps({"global_step": 12000, "max_steps": 19550}))
            path = self.root / "outputs/csgo_1b/exp37/experiment_plan.json"
            plan = json.loads(path.read_text())
            plan["fingerprint"] = "0" * 64
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                RUN.preflight_stage2_source(config)
            plan["fingerprint"] = RUN.fingerprint(RUN.read_yaml(self.root / "outputs/csgo_1b/exp37/resolved_config.yaml"), plan["runtime"])
            path.write_text(json.dumps(plan))
            config["test_maps"] = config["test_maps"][:-1]
            with self.assertRaisesRegex(ValueError, "test_maps"):
                RUN.preflight_stage2_source(config)
            config = self.config()
            recipe_path = self.root / "outputs/csgo_1b/exp37/resolved_config.yaml"
            recipe = RUN.read_yaml(recipe_path)
            recipe["is_loc_aux_loss"] = True
            recipe_path.write_text(yaml.safe_dump(recipe))
            plan["fingerprint"] = RUN.fingerprint(recipe, plan["runtime"])
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "fixed recipe differs in is_loc_aux_loss"):
                RUN.preflight_stage2_source(config)

    def test_resume_uses_own_full_checkpoint_and_rejects_other_steps(self):
        ckpt, _, plan = self.managed()
        args = self.args("train", "exp38_s2_joint", "--resume", "--stop-after-step", "0")
        config = RUN.resolve_config(args, for_training=True)
        self.assertNotIn("training_stop_after_step", config)
        _, resumed, _ = RUN.training_plan(args, config)
        self.assertEqual(resumed["resume_checkpoint"], RUN.relative(ckpt))
        self.assertEqual(resumed["source_checkpoint"], plan["source_checkpoint"])
        self.assertEqual(resumed["runtime"]["max_steps"], 7550)
        self.assertEqual(resumed["fingerprint"], plan["fingerprint"])
        self.assertEqual(resumed["runtime"]["checkpoint_steps"], [4000, 7550])
        self.assertEqual(resumed["cumulative_steps"], {"4000": 16000, "7550": 19550})
        (ckpt / "rng_state.pth").unlink()
        with self.assertRaisesRegex(ValueError, "RNG state"):
            RUN.training_plan(args, config)
        (ckpt / "rng_state.pth").touch()
        (ckpt / "scheduler.pt").unlink()
        (ckpt / "optimizer.pt").unlink()
        with self.assertRaisesRegex(ValueError, "optimizer/scheduler"):
            RUN.training_plan(args, config)
        (ckpt / "scheduler.pt").touch()
        (ckpt / "optimizer.pt").touch()
        extra = ckpt.parent / "checkpoint-6000"
        extra.mkdir()
        with self.assertRaisesRegex(ValueError, "Unexpected exp38 checkpoint"):
            RUN.training_plan(args, config)

    def test_evaluation_two_test_nodes_and_three_routes(self):
        for step in (4000, 7550):
            args = self.args("infer", "exp38_s2_both", "--checkpoint-step", str(step), "--task", "all")
            config = RUN.resolve_config(args, for_training=False)
            commands, path, evaluated = RUN.evaluation_plan(args, config)
            self.assertEqual(len(commands), 3)
            self.assertEqual([Path(c[1]).name for c in commands], ["eval_csgo_loc.py", "eval_csgo.py", "eval_csgo.py"])
            self.assertEqual(option(commands[0], "--test-loc-batch-size"), "32")
            self.assertEqual([option(c, "--test-gen-batch-size") for c in commands[1:]], ["16", "16"])
            self.assertIn(f"checkpoint-{step}/test", str(path))
            self.assertIn(f"checkpoint-{step}/model.safetensors", evaluated["ckpt_path"])
            self.assertFalse(evaluated["is_loc_aux_loss"])
            snapshots = RUN.evaluation_snapshots(args, path, evaluated)
            self.assertEqual(len(snapshots), 3)
        final = self.args("infer", "exp38_s2_joint", "--checkpoint-step", "final")
        self.assertIn("checkpoint-7550", str(RUN.evaluation_plan(final, RUN.resolve_config(final, for_training=False))[1]))
        for step in (2000, 3000, 6000, 7551):
            args = self.args("infer", "exp38_s2_joint", "--checkpoint-step", str(step))
            with self.assertRaises(ValueError):
                RUN.evaluation_plan(args, RUN.resolve_config(args, for_training=False))
        args = self.args("infer", "exp38_s2_joint", "--checkpoint-step", "4000", "--split", "validation")
        with self.assertRaises(ValueError):
            RUN.evaluation_plan(args, RUN.resolve_config(args, for_training=False))

    def test_execute_evaluation_allows_4000_but_checks_final(self):
        self.managed()
        args = self.args("infer", "exp38_s2_joint", "--checkpoint-step", "4000", "--execute")
        RUN.evaluation_plan(args, RUN.resolve_config(args, for_training=False))
        args = self.args("infer", "exp38_s2_joint", "--checkpoint-step", "final", "--execute")
        with self.assertRaises(ValueError):
            RUN.evaluation_plan(args, RUN.resolve_config(args, for_training=False))
        final = self.root / "outputs/csgo_1b/exp38_s2_joint/checkpoint-7550"
        final.mkdir()
        (final / "trainer_state.json").write_text(json.dumps({"global_step": 7550, "max_steps": 7550}))
        (final / "model.safetensors").touch()
        RUN.evaluation_plan(args, RUN.resolve_config(args, for_training=False))

    def test_batched_metrics_accepts_resolved_output_paths_without_reinferring(self):
        snapshot_path, provenance_path, summary_path = self.batched_metrics_fixture()
        summary = json.loads(summary_path.read_text())
        self.assertNotEqual(Path(summary["inference_provenance"]["path"]),
                            provenance_path)
        self.assertEqual(Path(summary["inference_provenance"]["path"]),
                         provenance_path.resolve())
        self.assertNotEqual(Path(summary["checkpoint"]),
                            RUN.path_in_repo("outputs/csgo_1b/exp38_s2_joint/checkpoint-4000/model.safetensors"))
        with patch.object(RUN.subprocess, "run") as launch, \
             patch("sys.stdout", io.StringIO()):
            code = RUN.main(["metrics", "--experiment", "exp38_s2_joint",
                             "--checkpoint-step", "4000", "--task", "all", "--execute",
                             "--test-loc-batch-size", "32", "--test-gen-batch-size", "16"])
        self.assertEqual(code, 0)
        self.assertEqual(launch.call_count, 22)
        commands = [call.args[0] for call in launch.call_args_list]
        self.assertEqual(sum(Path(cmd[1]).name == "benchmark_csgo_v1.py" for cmd in commands), 10)
        self.assertEqual(sum(Path(cmd[1]).name == "benchmark_csgo_v1_conti.py" for cmd in commands), 10)
        self.assertEqual(sum(Path(cmd[1]).name == "aggregate_csgo_benchmark_v2_metrics.py"
                             for cmd in commands), 2)
        self.assertFalse(any(Path(cmd[1]).name.startswith("eval_csgo") for cmd in commands))

    def test_batched_metrics_rejects_other_source_files(self):
        snapshot_path, provenance_path, summary_path = self.batched_metrics_fixture()
        command = ["metrics", "--experiment", "exp38_s2_joint", "--checkpoint-step", "4000",
                   "--task", "all", "--execute", "--test-loc-batch-size", "32",
                   "--test-gen-batch-size", "16"]
        original_payload = json.loads(provenance_path.read_text())
        original_summary = json.loads(summary_path.read_text())

        def assert_rejected(payload, summary, message):
            provenance_path.write_text(json.dumps(payload))
            summary_path.write_text(json.dumps(summary))
            with patch.object(RUN.subprocess, "run") as launch, \
                 patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()) as errors:
                self.assertEqual(RUN.main(command), 1)
                self.assertIn(message, errors.getvalue())
                launch.assert_not_called()

        payload = dict(original_payload, config_path=str(self.root / "other_evaluation_config.yaml"))
        assert_rejected(payload, original_summary, "Inference config path differs")

        alternate = self.root / "other_inference_manifest.json"
        alternate.write_text(json.dumps(original_payload))
        summary = json.loads(json.dumps(original_summary))
        summary["inference_provenance"]["path"] = str(alternate)
        assert_rejected(original_payload, summary, "Localization summary inference provenance differs")

        other_checkpoint = self.root / "other_model.safetensors"
        other_checkpoint.touch()
        payload = dict(original_payload, checkpoint=str(other_checkpoint), ckpt_path=str(other_checkpoint))
        summary = json.loads(json.dumps(original_summary))
        summary["checkpoint"] = str(other_checkpoint)
        summary["inference_provenance"]["payload"] = payload
        assert_rejected(payload, summary, "Localization summary checkpoint/manifest differs")

        other_manifest = self.root / "other_benchmark_manifest.json"
        shutil.copyfile(REPO / "data/csgo_benchmark_v2/benchmark_manifest.json", other_manifest)
        payload = dict(original_payload, benchmark_v2_manifest=str(other_manifest))
        summary = json.loads(json.dumps(original_summary))
        summary["manifest"] = str(other_manifest)
        summary["inference_provenance"]["payload"] = payload
        assert_rejected(payload, summary, "Localization summary checkpoint/manifest differs")

    def test_dry_run_has_no_writes_or_subprocess(self):
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        with patch.object(RUN.subprocess, "run") as launch, patch("sys.stdout", io.StringIO()):
            self.assertEqual(RUN.main(["train", "--experiment", "exp38_s2_joint"]), 0)
            self.assertEqual(RUN.main(["infer", "--experiment", "exp38_s2_joint", "--checkpoint-step", "final"]), 0)
            self.assertEqual(RUN.main(["metrics", "--experiment", "exp38_s2_joint", "--checkpoint-step", "4000"]), 0)
            launch.assert_not_called()
        after = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        self.assertEqual(before, after)

    def test_first_execute_pins_source_and_resume_needs_no_source_file(self):
        source = self.source()
        with patch.object(RUN, "tensor_header", return_value={"tensor_count": 2830}), \
             patch.object(RUN.subprocess, "run") as launch, patch("sys.stdout", io.StringIO()):
            code = RUN.main(["train", "--experiment", "exp38_s2_joint", "--execute",
                             "--asset-mode", "full", "--data-dir", "alternate/full_assets"])
            self.assertEqual(code, 0)
            launch.assert_called_once()
        out = self.root / "outputs/csgo_1b/exp38_s2_joint"
        config = RUN.read_yaml(out / "resolved_config.yaml")
        plan = json.loads((out / "experiment_plan.json").read_text())
        self.assertIsNone(config["benchmark_v2_asset_manifest"])
        self.assertEqual(config["data_dir"], "alternate/full_assets")
        self.assertEqual(plan["source_checkpoint"]["sha256"], RUN.digest_file(source))
        ckpt = out / "checkpoint-4000"
        ckpt.mkdir()
        (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": 4000, "max_steps": 7550}))
        for filename in ("model.safetensors", "training_args.bin", "scheduler.pt", "rng_state.pth", "optimizer.pt"):
            (ckpt / filename).touch()
        source.unlink()
        with patch.object(RUN.subprocess, "run") as launch, patch("sys.stdout", io.StringIO()):
            code = RUN.main(["train", "--experiment", "exp38_s2_joint", "--execute", "--resume",
                             "--stop-after-step", "0"])
            self.assertEqual(code, 0)
            launch.assert_called_once()
        resumed = json.loads((out / "experiment_plan.json").read_text())
        self.assertEqual(resumed["resume_checkpoint"], RUN.relative(ckpt))
        self.assertEqual(resumed["source_checkpoint"], plan["source_checkpoint"])


if __name__ == "__main__":
    unittest.main()
