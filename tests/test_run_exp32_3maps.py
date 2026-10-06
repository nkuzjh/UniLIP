"""CPU-only contract tests for the opt-in Seen-3 experiment runner."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_exp32_3maps", REPO_ROOT / "scripts/run_exp32_3maps.py"
)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


class Exp32ThreeMapsRunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        configs = self.root / "csgo_configs"
        configs.mkdir()
        for experiment in RUNNER.EXPERIMENTS:
            source = REPO_ROOT / "csgo_configs" / f"{experiment}.yaml"
            shutil.copyfile(source, configs / source.name)
        report = self.root / "data/csgo_benchmark_v2/minimal_dataset_report.json"
        report.parent.mkdir(parents=True)
        shutil.copyfile(REPO_ROOT / "data/csgo_benchmark_v2/minimal_dataset_report.json", report)
        self.root_patch = patch.object(RUNNER, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def args(self, command="train", experiment="exp32_3maps_joint_original", *extra):
        return RUNNER.parser().parse_args(
            [command, "--experiment", experiment, *extra]
        )

    def test_matrix_has_seven_first_round_and_four_second_round_arms(self):
        self.assertEqual(len(RUNNER.ROUND1), 7)
        self.assertEqual(len(RUNNER.ROUND2), 4)
        self.assertEqual(
            RUNNER.ROUND2, ("aux_control", "perception", "auxloc", "both")
        )
        self.assertEqual(len(RUNNER.JOINT_BASES), 5)
        for experiment in RUNNER.EXPERIMENTS:
            config = RUNNER.resolve_config(self.args(experiment=experiment), for_training=True)
            RUNNER.validate_config(config)
            self.assertEqual(config["train_maps"], RUNNER.MAPS)
            self.assertEqual(config["val_maps"], RUNNER.MAPS)
            self.assertEqual(config["test_maps"], RUNNER.MAPS)

    def test_dry_run_never_writes_or_launches_a_subprocess(self):
        for command, extra in (
            ("validate", []),
            ("train", []),
            ("infer", ["--checkpoint-step", "2400"]),
            ("metrics", ["--checkpoint-step", "2400"]),
        ):
            with self.subTest(command=command), patch.object(RUNNER.subprocess, "run") as run, patch("sys.stdout", io.StringIO()):
                result = RUNNER.main(
                    [command, "--experiment", "exp32_3maps_joint_original", *extra]
                )
                self.assertEqual(result, 0)
                run.assert_not_called()
        self.assertFalse((self.root / "outputs").exists())
        self.assertFalse((self.root / "outputs_eval").exists())

    def test_staged_train_keeps_full_horizon_and_resumable_state(self):
        for step in (2400, 3600):
            with self.subTest(step=step):
                args = self.args("train", "exp32_3maps_both", "--stop-after-step", str(step))
                config = RUNNER.resolve_config(args, for_training=True)
                commands, metadata, _ = RUNNER.training_plan(args, config)
                command = commands[0]
                self.assertEqual(option(command, "--num_train_epochs"), "50")
                self.assertEqual(option(command, "--save_steps"), "1200")
                self.assertEqual(option(command, "--save_only_model"), "False")
                self.assertEqual(option(command, "--eval_strategy"), "no")
                self.assertNotIn("--max_steps", command)
                self.assertEqual(config["training_stop_after_step"], step)
                self.assertEqual(metadata["runtime"]["gradient_accumulation_steps"], 32)
        full = self.args("train", "exp32_3maps_both", "--stop-after-step", "0")
        full_config = RUNNER.resolve_config(full, for_training=True)
        self.assertNotIn("training_stop_after_step", full_config)
        self.assertEqual(
            RUNNER.fingerprint(config, metadata["runtime"]),
            RUNNER.fingerprint(full_config, metadata["runtime"]),
        )

    def test_resume_requires_exact_recipe_and_complete_checkpoint(self):
        args = self.args("train", "exp32_3maps_joint_original")
        config = RUNNER.resolve_config(args, for_training=True)
        _, metadata, output = RUNNER.training_plan(args, config)
        output.mkdir(parents=True)
        (output / "experiment_plan.json").write_text(json.dumps(metadata))
        checkpoint = output / "checkpoint-2400"
        checkpoint.mkdir()
        state = {"global_step": 2400, "max_steps": 10000}
        (checkpoint / "trainer_state.json").write_text(json.dumps(state))
        for name in ("training_args.bin", "model.safetensors", "rng_state_0.pth", "optimizer.pt"):
            (checkpoint / name).touch()
        resumed = self.args("train", args.experiment, "--resume", "--stop-after-step", "3600")
        resumed_config = RUNNER.resolve_config(resumed, for_training=True)
        with self.assertRaisesRegex(ValueError, "optimizer/scheduler"):
            RUNNER.training_plan(resumed, resumed_config)
        (checkpoint / "scheduler.pt").touch()
        _, resumed_metadata, _ = RUNNER.training_plan(resumed, resumed_config)
        self.assertEqual(resumed_metadata["resume_checkpoint"], RUNNER.relative(checkpoint))
        self.assertEqual(resumed_metadata["fingerprint"], metadata["fingerprint"])
        same_stop = self.args("train", args.experiment, "--resume")
        with self.assertRaisesRegex(ValueError, "later stop"):
            RUNNER.training_plan(
                same_stop, RUNNER.resolve_config(same_stop, for_training=True)
            )
        changed = copy.deepcopy(resumed_config)
        changed["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "recipe/layout changed"):
            RUNNER.training_plan(resumed, changed)

    def test_round_two_base_is_a_recipe_choice_without_checkpoint_init(self):
        for base in RUNNER.JOINT_BASES:
            with self.subTest(base=base):
                args = self.args("train", "exp32_3maps_auxloc", "--joint-base", base)
                config = RUNNER.resolve_config(args, for_training=True)
                base_config = RUNNER.read_yaml(RUNNER.config_path(base))
                self.assertEqual(config["exp32_3maps_joint_base"], base)
                for key in RUNNER.BASE_KEYS:
                    self.assertEqual(config.get(key), base_config.get(key))
                for key in ("resume_ckpt_path", "finetune_init_ckpt_path",
                            "base_init_ckpt_path", "gen_init_ckpt_path", "loc_init_ckpt_path"):
                    self.assertFalse(config.get(key))
                RUNNER.validate_config(config)

    def test_resume_rejects_checkpoint_step_inconsistent_with_trainer_state(self):
        args = self.args("train", "exp32_3maps_joint_original")
        config = RUNNER.resolve_config(args, for_training=True)
        _, metadata, output = RUNNER.training_plan(args, config)
        checkpoint = output / "checkpoint-2400"
        checkpoint.mkdir(parents=True)
        (output / "experiment_plan.json").write_text(json.dumps(metadata))
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": 1200, "max_steps": 10000})
        )
        for name in ("training_args.bin", "model.safetensors", "rng_state_0.pth",
                     "optimizer.pt", "scheduler.pt"):
            (checkpoint / name).touch()
        resumed = self.args("train", args.experiment, "--resume", "--stop-after-step", "3600")
        with self.assertRaisesRegex(ValueError, "global_step|checkpoint step"):
            RUNNER.training_plan(resumed, RUNNER.resolve_config(resumed, for_training=True))

    def test_round_two_resume_inherits_saved_nondefault_joint_base(self):
        experiment = "exp32_3maps_perception"
        selected_base = "exp32_3maps_joint_headlr"
        first = self.args("train", experiment, "--joint-base", selected_base)
        first_config = RUNNER.resolve_config(first, for_training=True)
        _, metadata, output = RUNNER.training_plan(first, first_config)
        output.mkdir(parents=True)
        (output / "resolved_config.yaml").write_text(yaml.safe_dump(first_config))
        (output / "experiment_plan.json").write_text(json.dumps(metadata))
        checkpoint = output / "checkpoint-3600"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": 3600, "max_steps": 10000})
        )
        for name in ("training_args.bin", "model.safetensors", "rng_state_0.pth",
                     "optimizer.pt", "scheduler.pt"):
            (checkpoint / name).touch()
        resume = self.args("train", experiment, "--resume", "--stop-after-step", "4800")
        resumed_config = RUNNER.resolve_config(resume, for_training=True)
        self.assertEqual(resumed_config["exp32_3maps_joint_base"], selected_base)
        _, resumed_metadata, _ = RUNNER.training_plan(resume, resumed_config)
        self.assertEqual(resumed_metadata["fingerprint"], metadata["fingerprint"])
        self.assertEqual(resumed_metadata["resume_checkpoint"], RUNNER.relative(checkpoint))

    def test_full_backend_flows_from_training_snapshot_to_infer_and_metrics(self):
        experiment = "exp32_3maps_joint_original"
        source_data = "/dataset/full_csgo"
        train = self.args("train", experiment, "--asset-mode", "full", "--data-dir", source_data)
        trained = RUNNER.resolve_config(train, for_training=True)
        self.assertIsNone(trained["benchmark_v2_asset_manifest"])
        train_commands, _, output = RUNNER.training_plan(train, trained)
        self.assertEqual(option(train_commands[0], "--csgo_image_folder"), source_data)
        output.mkdir(parents=True)
        (output / "resolved_config.yaml").write_text(yaml.safe_dump(trained))
        for command in ("infer", "metrics"):
            with self.subTest(command=command):
                args = self.args(command, experiment, "--checkpoint-step", "2400")
                saved = RUNNER.resolve_config(args, for_training=False)
                self.assertIsNone(saved["benchmark_v2_asset_manifest"])
                self.assertEqual(saved["data_dir"], source_data)
                commands, _, eval_config = RUNNER.evaluation_plan(args, saved)
                self.assertIsNone(eval_config["benchmark_v2_asset_manifest"])
                self.assertEqual(eval_config["data_dir"], source_data)
                if command == "metrics":
                    for metric_command, map_name in zip(commands[:3], RUNNER.MAPS):
                        self.assertEqual(option(metric_command, "--gt"),
                                         f"{source_data}/{map_name}/imgs")
                        self.assertNotIn("--benchmark_v2_asset_manifest", metric_command)
                        self.assertEqual(option(metric_command, "--data_dir"), source_data)
                else:
                    self.assertEqual(len(commands), 2)
                    self.assertTrue(all(option(item, "--csgo_config") for item in commands))

    def test_single_task_lora_arms_only_plan_active_evaluations(self):
        for suffix, active, inactive in (
            ("loc_single", "localization", "generation"),
            ("gen_single", "generation", "localization"),
        ):
            experiment = f"exp32_3maps_{suffix}"
            config = RUNNER.resolve_config(self.args("train", experiment), for_training=True)
            self.assertTrue(config["enable_language_model_lora"])
            self.assertTrue(config["freeze_inactive_head"])
            self.assertEqual(config["enable_loc_head_lora"], active == "localization")
            self.assertEqual(config["enable_gen_head_lora"], active == "generation")
            infer = self.args("infer", experiment, "--checkpoint-step", "2400")
            infer_commands, _, _ = RUNNER.evaluation_plan(infer, config)
            self.assertEqual(len(infer_commands), 1)
            self.assertEqual(infer_commands[0][1],
                             "eval_csgo_loc.py" if active == "localization" else "eval_csgo.py")
            with self.assertRaisesRegex(ValueError, "inactive"):
                invalid = self.args("infer", experiment, "--checkpoint-step", "2400", "--task", inactive)
                RUNNER.evaluation_plan(invalid, config)
            metrics = self.args("metrics", experiment, "--checkpoint-step", "2400")
            metric_commands, _, _ = RUNNER.evaluation_plan(metrics, config)
            self.assertEqual(len(metric_commands), 0 if active == "localization" else 4)

    def test_eval_uses_saved_recipe_and_minimal_report_ground_truth(self):
        experiment = "exp32_3maps_both"
        report = self.root / "data/csgo_benchmark_v2/minimal_dataset_report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({
            "status": "verified", "images": {
                "status": "verified", "target_template": "images/{map}/{file_frame}.jpg"
            }
        }))
        train_args = self.args("train", experiment, "--joint-base",
                               "exp32_3maps_joint_headlr")
        config = RUNNER.resolve_config(train_args, for_training=True)
        snapshot = self.root / "outputs/csgo_1b" / experiment / "resolved_config.yaml"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_text(yaml.safe_dump(config))
        source = RUNNER.config_path(experiment)
        source_config = RUNNER.read_yaml(source)
        source_config["learning_rate"] = 0.123
        source.write_text(yaml.safe_dump(source_config))
        eval_args = self.args("metrics", experiment, "--checkpoint-step", "2400")
        saved = RUNNER.resolve_config(eval_args, for_training=False)
        self.assertEqual(saved["learning_rate"], config["learning_rate"])
        self.assertEqual(saved["exp32_3maps_joint_base"], "exp32_3maps_joint_headlr")
        commands, _, eval_config = RUNNER.evaluation_plan(eval_args, saved)
        self.assertEqual(eval_config["benchmark_v2_asset_manifest"],
                         config["benchmark_v2_asset_manifest"])
        self.assertEqual(len(commands), 4)
        for command, map_name in zip(commands[:3], RUNNER.MAPS):
            self.assertEqual(option(command, "--gt"),
                             f"data/csgo_benchmark_v2/images/{map_name}")
            self.assertEqual(option(command, "--benchmark_v2_asset_manifest"),
                             config["benchmark_v2_asset_manifest"])
            self.assertEqual(option(command, "--benchmark_v2_split"), "seen_validation")
        self.assertNotIn("benchmark_csgo_v1_conti.py", [part for cmd in commands for part in cmd])
        infer = self.args("infer", experiment, "--checkpoint-step", "2400")
        infer_commands, _, _ = RUNNER.evaluation_plan(infer, RUNNER.resolve_config(infer, for_training=False))
        self.assertEqual(len(infer_commands), 2)
        self.assertTrue(all(option(cmd, "--benchmark_v2_split") == "seen_validation"
                            for cmd in infer_commands))

    def test_eval_rejects_backend_override_different_from_saved_training(self):
        experiment = "exp32_3maps_joint_original"
        train_args = self.args("train", experiment)
        trained = RUNNER.resolve_config(train_args, for_training=True)
        snapshot = self.root / "outputs/csgo_1b" / experiment / "resolved_config.yaml"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_text(yaml.safe_dump(trained))
        for options in (
            ("--asset-mode", "full"),
            ("--asset-manifest", "data/another_report.json"),
            ("--data-dir", "data/another_source"),
        ):
            with self.subTest(options=options), self.assertRaisesRegex(
                ValueError, "saved training recipe|asset|data"
            ):
                RUNNER.resolve_config(
                    self.args("infer", experiment, "--checkpoint-step", "2400", *options),
                    for_training=False,
                )


if __name__ == "__main__":
    unittest.main()
