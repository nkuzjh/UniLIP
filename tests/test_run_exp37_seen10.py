"""CPU-only behavior checks for the opt-in exp37 Seen-10 runner."""

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
    "run_exp37_seen10", REPO_ROOT / "scripts/run_exp37_seen10.py"
)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

EXPERIMENTS = (
    "exp37", "exp37_s2_gen", "exp37_s2_loc", "exp37_s2", "exp37_s2_both",
)
SEEN_MAPS = json.loads(
    (REPO_ROOT / "data/csgo_benchmark_v2/benchmark_manifest.json").read_text()
)["protocol"]["seen_maps"]


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


class Exp37RunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        configs = self.root / "csgo_configs"
        configs.mkdir()
        for experiment in EXPERIMENTS:
            shutil.copyfile(
                REPO_ROOT / "csgo_configs" / f"{experiment}.yaml",
                configs / f"{experiment}.yaml",
            )
        report = self.root / "data/csgo_benchmark_v2/minimal_dataset_report.json"
        report.parent.mkdir(parents=True)
        shutil.copyfile(
            REPO_ROOT / "data/csgo_benchmark_v2/minimal_dataset_report.json", report
        )
        root_patch = patch.object(RUNNER, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def args(self, command="train", experiment="exp37", *extra):
        return RUNNER.parser().parse_args(
            [command, "--experiment", experiment, *extra]
        )

    def config(self, experiment="exp37", *extra, for_training=True):
        return RUNNER.resolve_config(
            self.args("train" if for_training else "infer", experiment, *extra),
            for_training=for_training,
        )

    def completed_checkpoint(self, experiment: str, step: int) -> Path:
        args = self.args("train", experiment)
        config = RUNNER.resolve_config(args, for_training=True)
        _, metadata, out = RUNNER.training_plan(args, config)
        out = self.root / "outputs/csgo_1b" / experiment
        out.mkdir(parents=True, exist_ok=True)
        (out / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        (out / "experiment_plan.json").write_text(json.dumps(metadata))
        state = {"global_step": step, "max_steps": step}
        if experiment == "exp37":
            state.update(num_train_epochs=50, epoch=50.0)
        (out / "trainer_state.json").write_text(
            json.dumps(state)
        )
        checkpoint = out / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(json.dumps(state))
        (checkpoint / "model.safetensors").touch()
        return checkpoint

    def snapshot_recipe(self, experiment: str) -> Path:
        out = self.root / "outputs/csgo_1b" / experiment
        out.mkdir(parents=True, exist_ok=True)
        (out / "resolved_config.yaml").write_text(yaml.safe_dump(self.config(experiment)))
        return out

    def managed_resume_checkpoint(self, experiment: str, *device_args: str) -> tuple[Path, dict]:
        args = self.args("train", experiment, *device_args)
        config = RUNNER.resolve_config(args, for_training=True)
        _, metadata, out = RUNNER.training_plan(args, config)
        out.mkdir(parents=True)
        metadata["source_checkpoint"] = {
            "path": config["finetune_init_ckpt_path"],
            "sha256": "a" * 64,
            "step": 19500,
        }
        (out / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        (out / "experiment_plan.json").write_text(json.dumps(metadata))
        checkpoint = out / "checkpoint-2000"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": 2000, "max_steps": 6000})
        )
        for name in ("training_args.bin", "model.safetensors", "optimizer.pt", "scheduler.pt"):
            (checkpoint / name).touch()
        return checkpoint, metadata

    def legacy_source_fixture(self, experiment: str) -> tuple[Path, dict]:
        legacy = "exp32_gen" if experiment.endswith("gen") else "exp32_loc"
        shutil.copyfile(
            REPO_ROOT / "csgo_configs" / f"{legacy}.yaml",
            self.root / "csgo_configs" / f"{legacy}.yaml",
        )
        config = self.config(experiment)
        checkpoint = self.root / "outputs/csgo_1b" / legacy / "checkpoint-19500"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(b"small header fixture")
        (checkpoint / "training_args.bin").write_bytes(b"mocked trusted arguments")
        config_keys = (
            "enable_language_model_lora", "enable_gen_head_lora", "enable_loc_head_lora",
            "freeze_inactive_head", "is_lora", "llm_train_mode", "alpha_loc_loss",
            "is_action_dit_projector", "use_pi05_action_dit",
        )
        (checkpoint / "config.json").write_text(
            json.dumps({key: config[key] for key in config_keys})
        )
        gen_count = 128 if experiment.endswith("gen") else 0
        state = {
            "global_step": 19500, "max_steps": 19500,
            "num_train_epochs": 50, "epoch": 50.0,
            "log_history": [{"step": 19500, "other_info": {
                "gen_indices": gen_count, "loc_indices": 128 - gen_count,
            }}],
        }
        (checkpoint / "trainer_state.json").write_text(json.dumps(state))
        return checkpoint, config

    def test_fixed_seen10_recipes_and_new_optimization_horizons(self):
        self.assertEqual(len(SEEN_MAPS), 10)
        self.assertEqual(tuple(RUNNER.EXPERIMENTS), EXPERIMENTS)
        for experiment in EXPERIMENTS:
            with self.subTest(experiment=experiment):
                config = self.config(experiment)
                RUNNER.validate_config(config)
                for key in ("train_maps", "val_maps", "test_maps"):
                    self.assertEqual(config[key], SEEN_MAPS)
                self.assertNotIn("benchmark_v2_allow_map_subset_summary", config)
                self.assertEqual(config["benchmark_v2_split"], "seen_train")
                self.assertTrue(config["csgo_loss_per_microbatch_mean"])
                self.assertEqual(config["alpha_loc_loss"],
                                 8.0 if experiment in ("exp37", "exp37_s2", "exp37_s2_both") else 1.0)
                commands, metadata, out = RUNNER.training_plan(
                    self.args("train", experiment), config
                )
                command = commands[0]
                self.assertFalse(out.exists())
                self.assertEqual(option(command, "--num_train_epochs"), "50")
                self.assertEqual(option(command, "--save_only_model"), "False")
                self.assertEqual(option(command, "--save_total_limit"), "6")
                self.assertEqual(metadata["runtime"]["gradient_accumulation_steps"], 32)
                if experiment == "exp37":
                    self.assertNotIn("--max_steps", command)
                    self.assertEqual(option(command, "--save_steps"), "4000")
                    self.assertEqual(config["warmup_ratio"], 0.003)
                    self.assertEqual(config["lr_scheduler_kwargs"], {"min_lr": 1e-5})
                    self.assertNotIn("training_stop_after_step", config)
                else:
                    self.assertEqual(option(command, "--max_steps"), "6000")
                    self.assertEqual(option(command, "--warmup_steps"), "200")
                    self.assertEqual(option(command, "--save_steps"), "2000")
                    self.assertEqual(config["lr_scheduler_kwargs"], {"min_lr_rate": 0.1})
                    self.assertEqual(config["learning_rate"], 2e-5)
                    self.assertEqual(config["action_dit_connector_lr"], 1e-4)

    def test_single_task_and_combined_objectives_remain_scoped(self):
        gen = self.config("exp37_s2_gen")
        loc = self.config("exp37_s2_loc")
        both = self.config("exp37_s2_both")
        control = self.config("exp37_s2")
        for config, ratio, gen_head, loc_head in (
            (gen, 0.0, True, False), (loc, 1.0, False, True),
        ):
            self.assertEqual(config["task_mix_ratio"], ratio)
            self.assertTrue(config["freeze_inactive_head"])
            self.assertEqual(config["enable_gen_head_lora"], gen_head)
            self.assertEqual(config["enable_loc_head_lora"], loc_head)
            self.assertFalse(config["is_loc_aux_loss"])
            self.assertFalse(config["is_loc_perception_loss"])
            self.assertIn("checkpoint-19500/model.safetensors", config["finetune_init_ckpt_path"])
            self.assertNotIn("3maps", config["finetune_init_ckpt_path"])
        self.assertIn("exp32_gen", gen["finetune_init_ckpt_path"])
        self.assertIn("exp32_loc", loc["finetune_init_ckpt_path"])
        for config in (control, both):
            self.assertIn("outputs/csgo_1b/exp37/checkpoint-", config["finetune_init_ckpt_path"])
            self.assertTrue(config["finetune_init_strict"])
        self.assertFalse(control["is_loc_aux_loss"])
        self.assertFalse(control["is_loc_perception_loss"])
        self.assertTrue(both["is_loc_aux_loss"])
        self.assertTrue(both["is_aux_loc_combined_em_unc_loss"])
        self.assertTrue(both["is_loc_perception_loss"])
        self.assertFalse(both["loc_perception_use_attention_weight"])
        self.assertEqual(both["alpha_loc_perception_schedule_steps"], [0, 400])
        self.assertEqual(both["alpha_loc_perception_schedule_values"], [0.0, 0.05])
        self.assertEqual(both["alpha_loc_aux_schedule_steps"], [0, 400])
        self.assertEqual(both["alpha_loc_aux_schedule_values"], [0.0, 1.0])
        self.assertEqual(both["aux_loc_combined_candidate_tau"], 1.0)
        self.assertEqual(both["aux_loc_combined_num_samples"], 2)

    def test_final_selects_completed_state_and_joint_source_is_dynamic(self):
        checkpoint = self.completed_checkpoint("exp37", 19337)
        later = checkpoint.parent / "checkpoint-20000"
        later.mkdir()
        (later / "trainer_state.json").write_text(
            json.dumps({"global_step": 20000, "max_steps": 22000})
        )
        (later / "model.safetensors").touch()
        self.assertEqual(RUNNER.final_checkpoint_step("exp37"), 19337)
        for experiment in ("exp37_s2", "exp37_s2_both"):
            with self.subTest(experiment=experiment):
                config = self.config(experiment)
                self.assertEqual(
                    config["finetune_init_ckpt_path"],
                    "outputs/csgo_1b/exp37/checkpoint-19337/model.safetensors",
                )
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": 19336, "max_steps": 19337})
        )
        with self.assertRaises(ValueError):
            RUNNER.final_checkpoint_step("exp37")

    def test_final_evaluation_nodes_reject_early_stop_and_stage2_requires_6000(self):
        self.completed_checkpoint("exp37", 19500)
        saved = self.config("exp37", for_training=False)
        early = self.args("infer", "exp37", "--checkpoint-step", "2400")
        with self.assertRaises(ValueError):
            RUNNER.evaluation_plan(early, saved)
        final = self.args("infer", "exp37", "--checkpoint-step", "final",
                          "--split", "test")
        _, path, evaluated = RUNNER.evaluation_plan(final, saved)
        self.assertIn("checkpoint-19500/test", str(path))
        self.assertIn("checkpoint-19500/model.safetensors", evaluated["ckpt_path"])
        self.completed_checkpoint("exp37_s2_gen", 6000)
        self.assertEqual(RUNNER.final_checkpoint_step("exp37_s2_gen"), 6000)
        step_4000_test = self.args("infer", "exp37_s2_gen", "--checkpoint-step", "4000",
                                   "--split", "test")
        with self.assertRaises(ValueError):
            RUNNER.evaluation_plan(step_4000_test,
                                   self.config("exp37_s2_gen", for_training=False))
        state_path = RUNNER.output_dir("exp37_s2_gen") / "checkpoint-6000/trainer_state.json"
        state_path.write_text(json.dumps({"global_step": 5000, "max_steps": 6000}))
        with self.assertRaises(ValueError):
            RUNNER.final_checkpoint_step("exp37_s2_gen")

    def test_dry_run_plans_do_not_write_or_launch(self):
        for experiment in EXPERIMENTS:
            for command in ("validate", "train"):
                with self.subTest(experiment=experiment, command=command), \
                     patch.object(RUNNER.subprocess, "run") as run, \
                     patch("sys.stdout", io.StringIO()):
                    self.assertEqual(
                        RUNNER.main([command, "--experiment", experiment]), 0
                    )
                    run.assert_not_called()
        self.assertFalse((self.root / "outputs").exists())
        self.completed_checkpoint("exp37", 19500)
        for experiment in EXPERIMENTS[1:]:
            self.snapshot_recipe(experiment)
        before = {path: path.read_bytes() for path in (self.root / "outputs").rglob("resolved_config.yaml")}
        for experiment in EXPERIMENTS:
            for command in ("infer", "metrics"):
                step = "8000" if experiment == "exp37" else "4000"
                with self.subTest(experiment=experiment, command=command), \
                     patch.object(RUNNER.subprocess, "run") as run, \
                     patch("sys.stdout", io.StringIO()):
                    self.assertEqual(
                        RUNNER.main([command, "--experiment", experiment,
                                     "--checkpoint-step", step]), 0
                    )
                    run.assert_not_called()
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertFalse((self.root / "outputs_eval").exists())
        self.assertFalse((self.root / "outputs_loc").exists())

    def test_stage1_8000_validation_plans_before_run_and_executes_before_final(self):
        command = ["infer", "--experiment", "exp37", "--checkpoint-step", "8000",
                   "--split", "validation"]
        with patch.object(RUNNER.subprocess, "run") as run, \
             patch("sys.stdout", io.StringIO()):
            self.assertEqual(RUNNER.main(command), 0)
            run.assert_not_called()
        self.assertFalse((self.root / "outputs").exists())
        self.assertFalse((self.root / "outputs_eval").exists())

        args = self.args("train", "exp37")
        config = RUNNER.resolve_config(args, for_training=True)
        _, plan, out = RUNNER.training_plan(args, config)
        out.mkdir(parents=True)
        (out / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        (out / "experiment_plan.json").write_text(json.dumps(plan))
        checkpoint = out / "checkpoint-8000"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").touch()
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": 8000, "max_steps": 19500, "epoch": 20.0,
                        "num_train_epochs": 50})
        )
        with patch.object(RUNNER.subprocess, "run") as run, \
             patch("sys.stdout", io.StringIO()):
            self.assertEqual(RUNNER.main([*command, "--execute"]), 0)
            self.assertEqual(run.call_count, 2)
        self.assertTrue((self.root / "outputs_eval/benchmark_v2/exp37/checkpoint-8000/validation/seed_42/evaluation_config.yaml").is_file())

    def test_stage2_test_execute_rejects_wrong_horizon_incomplete_and_corrupt_plan(self):
        checkpoint = self.completed_checkpoint("exp37_s2_gen", 6000)
        out = checkpoint.parent
        state_path = checkpoint / "trainer_state.json"
        original_state = json.loads(state_path.read_text())
        plan_path = out / "experiment_plan.json"
        original_plan = json.loads(plan_path.read_text())
        command = ["infer", "--experiment", "exp37_s2_gen", "--checkpoint-step", "6000",
                   "--split", "test", "--execute"]
        for case, expected in (
            ("wrong_horizon", "step/horizon"),
            ("incomplete", "step/horizon"),
            ("corrupt_plan", "fingerprint"),
        ):
            state = copy.deepcopy(original_state)
            plan = copy.deepcopy(original_plan)
            if case == "wrong_horizon":
                state["max_steps"] = 7000
            elif case == "incomplete":
                state["global_step"] = 5999
            else:
                plan["fingerprint"] = "0" * 64
            state_path.write_text(json.dumps(state))
            plan_path.write_text(json.dumps(plan))
            with self.subTest(case=case), patch.object(RUNNER.subprocess, "run") as run, \
                 patch("sys.stderr", io.StringIO()) as error, \
                 patch("sys.stdout", io.StringIO()):
                self.assertEqual(RUNNER.main(command), 1)
                self.assertIn(expected, error.getvalue())
                run.assert_not_called()
                self.assertFalse((self.root / "outputs_eval").exists())

    def test_saved_and_raw_experiment_identity_must_match_cli_arm(self):
        experiment = "exp37_s2_gen"
        out = self.snapshot_recipe(experiment)
        saved_path = out / "resolved_config.yaml"
        swapped = yaml.safe_load(saved_path.read_text())
        swapped["exp37_experiment"] = "exp37_s2_loc"
        saved_path.write_text(yaml.safe_dump(swapped))
        with self.assertRaisesRegex(ValueError, "another exp37 arm"):
            self.config(experiment, for_training=False)
        saved_path.unlink()
        raw_path = self.root / "csgo_configs/exp37_s2_gen.yaml"
        raw = yaml.safe_load(raw_path.read_text())
        raw["exp37_experiment"] = "exp37_s2_loc"
        raw_path.write_text(yaml.safe_dump(raw))
        with self.assertRaisesRegex(ValueError, "another exp37 arm"):
            self.config(experiment)

    def test_evaluation_routes_only_active_tasks_across_full_seen10(self):
        cases = (
            ("exp37_s2_gen", ["eval_csgo.py", "eval_csgo.py"]),
            ("exp37_s2_loc", ["eval_csgo_loc.py"]),
            ("exp37_s2", ["eval_csgo_loc.py", "eval_csgo.py", "eval_csgo.py"]),
            ("exp37_s2_both", ["eval_csgo_loc.py", "eval_csgo.py", "eval_csgo.py"]),
        )
        for experiment, scripts in cases:
            self.snapshot_recipe(experiment)
            for split, step in (("validation", 4000), ("test", 6000)):
                with self.subTest(experiment=experiment, split=split):
                    args = self.args("infer", experiment, "--checkpoint-step", str(step),
                                     "--split", split)
                    config = self.config(experiment, for_training=False)
                    commands, path, evaluated = RUNNER.evaluation_plan(args, config)
                    expected_scripts = scripts if split == "test" else scripts[:1 if experiment == "exp37_s2_gen" else 2]
                    self.assertEqual([command[1] for command in commands], expected_scripts)
                    self.assertIn(f"checkpoint-{step}/{split}", str(path))
                    self.assertEqual(evaluated["train_maps"], SEEN_MAPS)
                    self.assertEqual(evaluated["val_maps"], SEEN_MAPS)
                    self.assertEqual(evaluated["test_maps"], SEEN_MAPS)
                    self.assertNotIn("benchmark_v2_allow_map_subset_summary", evaluated)
                    self.assertFalse(evaluated["is_loc_aux_loss"])
                    self.assertFalse(evaluated["is_loc_perception_loss"])
                    for command in commands:
                        self.assertEqual(option(command, "--benchmark_v2_maps"), SEEN_MAPS[0])
                    if split == "test" and experiment != "exp37_s2_loc":
                        self.assertIn("seen_continuous", [option(command, "--benchmark_v2_split")
                                                            for command in commands])
        for experiment, wrong in (("exp37_s2_gen", "localization"),
                                  ("exp37_s2_loc", "generation")):
            args = self.args("infer", experiment, "--checkpoint-step", "6000",
                             "--split", "test", "--task", wrong)
            with self.assertRaises(ValueError):
                RUNNER.evaluation_plan(args, self.config(experiment, for_training=False))
        metrics_args = self.args("metrics", "exp37_s2_gen", "--checkpoint-step", "6000",
                                 "--split", "test")
        metric_commands, _, _ = RUNNER.evaluation_plan(
            metrics_args, self.config("exp37_s2_gen", for_training=False)
        )
        self.assertEqual(len(metric_commands), 22)
        summary_commands = [command for command in metric_commands
                            if command[1] == "scripts/aggregate_csgo_benchmark_v2_metrics.py"]
        self.assertEqual(len(summary_commands), 2)
        for command in summary_commands:
            maps = command[command.index("--maps") + 1:command.index("--input_root")]
            self.assertEqual(maps, SEEN_MAPS)
            self.assertNotIn("--allow-map-subset", command)

    def test_resume_requires_own_complete_checkpoint_and_saved_recipe(self):
        experiment = "exp37_s2_gen"
        config = self.config(experiment)
        _, metadata, out = RUNNER.training_plan(self.args("train", experiment), config)
        out.mkdir(parents=True)
        metadata["source_checkpoint"] = {
            "path": config["finetune_init_ckpt_path"],
            "sha256": "f" * 64,
            "step": 19500,
        }
        (out / "experiment_plan.json").write_text(json.dumps(metadata))
        (out / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        checkpoint = out / "checkpoint-2000"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": 2000, "max_steps": 6000})
        )
        for name in ("training_args.bin", "model.safetensors", "rng_state.pth",
                     "optimizer.pt"):
            (checkpoint / name).touch()
        resumed = self.args("train", experiment, "--resume")
        with self.assertRaises(ValueError):
            RUNNER.training_plan(resumed, self.config(experiment))
        (checkpoint / "scheduler.pt").touch()
        commands, resumed_metadata, _ = RUNNER.training_plan(resumed, self.config(experiment))
        self.assertEqual(resumed_metadata["fingerprint"], metadata["fingerprint"])
        self.assertEqual(resumed_metadata["resume_checkpoint"], RUNNER.relative(checkpoint))
        self.assertNotIn("--resume_from_checkpoint", commands[0])
        changed = copy.deepcopy(config)
        changed["learning_rate"] *= 2
        with self.assertRaises(ValueError):
            RUNNER.training_plan(resumed, changed)
        metadata["source_checkpoint"]["sha256"] = "not-a-sha256"
        (out / "experiment_plan.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "pinned source provenance"):
            RUNNER.training_plan(resumed, self.config(experiment))
        self.assertFalse((self.root / "outputs/csgo_1b/exp32_gen").exists())

    def test_resume_requires_exact_rng_names_for_single_and_all_distributed_ranks(self):
        single, _ = self.managed_resume_checkpoint("exp37_s2_gen")
        single_args = self.args("train", "exp37_s2_gen", "--resume")
        with self.assertRaisesRegex(ValueError, "rank RNG state"):
            RUNNER.training_plan(single_args, self.config("exp37_s2_gen"))
        (single / "rng_state_0.pth").touch()
        with self.assertRaisesRegex(ValueError, "rng_state.pth"):
            RUNNER.training_plan(single_args, self.config("exp37_s2_gen"))
        (single / "rng_state.pth").touch()
        RUNNER.training_plan(single_args, self.config("exp37_s2_gen"))

        distributed, _ = self.managed_resume_checkpoint(
            "exp37_s2_loc", "--cuda-devices", "0,1"
        )
        distributed_args = self.args("train", "exp37_s2_loc", "--resume",
                                     "--cuda-devices", "0,1")
        (distributed / "rng_state.pth").touch()
        (distributed / "rng_state_0.pth").touch()
        with self.assertRaisesRegex(ValueError, "rng_state_1.pth"):
            RUNNER.training_plan(distributed_args, self.config("exp37_s2_loc"))
        (distributed / "rng_state_1.pth").touch()
        RUNNER.training_plan(distributed_args, self.config("exp37_s2_loc"))

    def test_evaluation_and_resume_reject_asset_changes_from_saved_recipe(self):
        experiment = "exp37_s2_gen"
        args = self.args("train", experiment, "--asset-mode", "full",
                         "--data-dir", "/dataset/full", "--model-path", "/model/custom",
                         "--pi05-path", "/pi05/custom")
        config = RUNNER.resolve_config(args, for_training=True)
        out = RUNNER.output_dir(experiment)
        out.mkdir(parents=True)
        (out / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        matching = self.args("infer", experiment, "--checkpoint-step", "4000",
                             "--asset-mode", "full", "--data-dir", "/dataset/full",
                             "--model-path", "/model/custom", "--pi05-path", "/pi05/custom")
        self.assertEqual(RUNNER.resolve_config(matching, for_training=False), config)
        for changed in (
            ("--asset-mode", "minimal"),
            ("--data-dir", "/dataset/other"),
            ("--model-path", "/model/other"),
            ("--pi05-path", "/pi05/other"),
        ):
            with self.subTest(changed=changed):
                evaluation = self.args("infer", experiment, "--checkpoint-step", "4000",
                                       *changed)
                with self.assertRaises(ValueError):
                    RUNNER.resolve_config(evaluation, for_training=False)
                resume = self.args("train", experiment, "--resume", *changed)
                with self.assertRaises(ValueError):
                    RUNNER.resolve_config(resume, for_training=True)

    def test_execute_rejects_source_preflight_before_writing_or_launching(self):
        checkpoint, _ = self.legacy_source_fixture("exp37_s2_gen")
        state_path = checkpoint / "trainer_state.json"
        original = json.loads(state_path.read_text())
        model = checkpoint / "model.safetensors"
        observed_args = {
            "num_train_epochs": 50, "max_steps": -1,
            "per_device_train_batch_size": 4, "gradient_accumulation_steps": 32,
            "lora_r": 32, "lora_alpha": 64, "save_only_model": False,
        }
        with patch.object(RUNNER, "legacy_training_args", return_value=observed_args), \
             patch.object(RUNNER, "tensor_header", return_value={
                 "tensor_count": 2578, "lora_tensor_count": 876,
                 "shape_validation": "mocked header",
             }), \
             patch.object(RUNNER, "digest_file", return_value="a" * 64), \
             patch.object(RUNNER.subprocess, "run") as run:
            for case, expected in (
                ("missing", "Missing Stage 2 source"),
                ("incomplete", "completed checkpoint"),
                ("wrong_task", "single-task head"),
            ):
                state = copy.deepcopy(original)
                if case == "missing":
                    model.unlink()
                elif case == "incomplete":
                    state["global_step"] = 19499
                else:
                    state["log_history"][-1]["other_info"] = {
                        "gen_indices": 0, "loc_indices": 128,
                    }
                state_path.write_text(json.dumps(state))
                with self.subTest(case=case), patch("sys.stderr", io.StringIO()) as error, \
                     patch("sys.stdout", io.StringIO()):
                    self.assertEqual(RUNNER.main([
                        "train", "--experiment", "exp37_s2_gen", "--execute",
                    ]), 1)
                    self.assertIn(expected, error.getvalue())
                    run.assert_not_called()
                    self.assertFalse(RUNNER.output_dir("exp37_s2_gen").exists())
                if case == "missing":
                    model.write_bytes(b"small header fixture")

    def test_legacy_sources_need_no_snapshot_but_require_completed_right_task(self):
        observed_args = {
            "num_train_epochs": 50, "max_steps": -1,
            "per_device_train_batch_size": 4, "gradient_accumulation_steps": 32,
            "lora_r": 32, "lora_alpha": 64, "save_only_model": False,
        }
        for experiment in ("exp37_s2_gen", "exp37_s2_loc"):
            with self.subTest(experiment=experiment):
                checkpoint, config = self.legacy_source_fixture(experiment)
                self.assertFalse((checkpoint.parent / "resolved_config.yaml").exists())
                with patch.object(RUNNER, "legacy_training_args", return_value=observed_args), \
                     patch.object(RUNNER, "tensor_header", return_value={
                         "tensor_count": 2578 if experiment.endswith("gen") else 2290,
                         "lora_tensor_count": 876 if experiment.endswith("gen") else 588,
                         "shape_validation": "mocked header",
                     }), \
                     patch.object(RUNNER, "digest_file", return_value="a" * 64):
                    provenance = RUNNER.preflight_stage2_source(config)
                    self.assertEqual(provenance["step"], 19500)
                    self.assertEqual(provenance["sha256"], "a" * 64)
                    self.assertIn(f"csgo_configs/{checkpoint.parent.name}.yaml",
                                  provenance["source_recipe"])
                    state_path = checkpoint / "trainer_state.json"
                    state = json.loads(state_path.read_text())
                    state["global_step"] = 19499
                    state_path.write_text(json.dumps(state))
                    with self.assertRaisesRegex(ValueError, "completed checkpoint"):
                        RUNNER.preflight_stage2_source(config)
                    state["global_step"] = 19500
                    state["log_history"][-1]["other_info"] = {
                        "gen_indices": 0 if experiment.endswith("gen") else 128,
                        "loc_indices": 128 if experiment.endswith("gen") else 0,
                    }
                    state_path.write_text(json.dumps(state))
                    with self.assertRaisesRegex(ValueError, "single-task head"):
                        RUNNER.preflight_stage2_source(config)
                    (checkpoint / "model.safetensors").unlink()
                    with self.assertRaisesRegex(ValueError, "Missing Stage 2 source"):
                        RUNNER.preflight_stage2_source(config)

    def test_joint_arms_pin_same_completed_stage1_source_hash(self):
        checkpoint = self.completed_checkpoint("exp37", 19337)
        source_name = RUNNER.relative(checkpoint / "model.safetensors")
        sibling = self.root / "outputs/csgo_1b/exp37_s2"
        sibling.mkdir(parents=True)
        sibling_plan = sibling / "experiment_plan.json"
        sibling_plan.write_text(json.dumps({"source_checkpoint": {
            "path": source_name, "sha256": "b" * 64,
        }}))
        config = self.config("exp37_s2_both")
        with patch.object(RUNNER, "tensor_header", return_value={
            "tensor_count": 2830, "lora_tensor_count": 1128,
            "shape_validation": "mocked header",
        }), patch.object(RUNNER, "digest_file", return_value="a" * 64):
            with self.assertRaisesRegex(ValueError, "sibling uses a different"):
                RUNNER.preflight_stage2_source(config)
            sibling_plan.write_text(json.dumps({"source_checkpoint": {
                "path": source_name, "sha256": "a" * 64,
            }}))
            provenance = RUNNER.preflight_stage2_source(config)
        self.assertEqual(provenance["path"], source_name)
        self.assertEqual(provenance["sha256"], "a" * 64)


if __name__ == "__main__":
    unittest.main()
