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

from scripts.aggregate_csgo_benchmark_v2_metrics import (
    AggregationError,
    build_localization_summary,
)


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

    def test_matrix_has_seven_original_six_extra_and_seven_second_round_arms(self):
        self.assertEqual(len(RUNNER.ROUND1), 7)
        self.assertEqual(len(RUNNER.ROUND1_EXTRA), 6)
        self.assertEqual(len(RUNNER.ROUND2), 4)
        self.assertEqual(
            RUNNER.ROUND2, ("aux_control", "perception", "auxloc", "both")
        )
        self.assertEqual(RUNNER.ROUND2_J8, ("perception_j8", "auxloc_j8", "both_j8"))
        self.assertEqual(len(RUNNER.EXPERIMENTS), 24)
        self.assertEqual(len(RUNNER.STAGE2_J8), 4)
        self.assertEqual(len(RUNNER.JOINT_BASES), 11)
        for experiment in RUNNER.EXPERIMENTS:
            config = RUNNER.resolve_config(self.args(experiment=experiment), for_training=True)
            RUNNER.validate_config(config)
            self.assertEqual(config["train_maps"], RUNNER.MAPS)
            self.assertEqual(config["val_maps"], RUNNER.MAPS)
            self.assertEqual(config["test_maps"], RUNNER.MAPS)
            if experiment.removeprefix(RUNNER.PREFIX) in RUNNER.ROUND2:
                self.assertEqual(config["exp32_3maps_joint_base"], RUNNER.DEFAULT_BASE)
            elif experiment.removeprefix(RUNNER.PREFIX) in RUNNER.ROUND2_J8:
                self.assertEqual(config["exp32_3maps_joint_base"], RUNNER.J8_BASE)

    def test_j8_recipes_preserve_auxiliary_settings_and_fix_joint_base(self):
        base = RUNNER.read_yaml(RUNNER.config_path(RUNNER.J8_BASE))
        cases = (("perception_j8", "perception", False, True),
                 ("auxloc_j8", "auxloc", True, False),
                 ("both_j8", "both", True, True))
        for suffix, old_suffix, aux, perception in cases:
            with self.subTest(suffix=suffix):
                experiment = RUNNER.PREFIX + suffix
                raw = RUNNER.read_yaml(RUNNER.config_path(experiment))
                old = RUNNER.read_yaml(RUNNER.config_path(RUNNER.PREFIX + old_suffix))
                expected = copy.deepcopy(old)
                for key in RUNNER.BASE_KEYS:
                    expected.pop(key, None)
                    if key in base:
                        expected[key] = copy.deepcopy(base[key])
                expected["exp32_3maps_joint_base"] = RUNNER.J8_BASE
                self.assertEqual(raw, expected)
                self.assertEqual(raw["alpha_loc_loss"], 8.0)
                self.assertEqual(raw["llm_lora_lr"], 1e-4)
                self.assertEqual(raw["training_stop_after_step"], 3600)
                self.assertEqual(raw["is_loc_aux_loss"], aux)
                self.assertEqual(raw["is_aux_loc_combined_em_unc_loss"], aux)
                self.assertEqual(raw["is_loc_perception_loss"], perception)
                self.assertEqual(raw.get("alpha_loc_aux_schedule_steps"),
                                 [0, 1800] if aux else None)
                self.assertEqual(raw.get("alpha_loc_aux_schedule_values"),
                                 [0.0, 2.0] if aux else None)
                self.assertEqual(raw.get("alpha_loc_perception_schedule_steps"),
                                 [0, 599, 600] if perception else None)
                self.assertEqual(raw.get("alpha_loc_perception_schedule_values"),
                                 [0.0, 0.0, 0.1] if perception else None)
                self.assertTrue(RUNNER.is_round2(experiment))
                for mode in ("minimal", "full"):
                    args = self.args("train", experiment, "--asset-mode", mode)
                    resolved = RUNNER.resolve_config(args, for_training=True)
                    expected_resolved = {**raw, "model_name_or_path": "UniLIP-1B"}
                    if mode == "full":
                        expected_resolved["benchmark_v2_asset_manifest"] = None
                    self.assertEqual(resolved, expected_resolved)
                    RUNNER.validate_config(resolved)
                    commands, metadata, output = RUNNER.training_plan(args, resolved)
                    self.assertEqual(metadata["joint_base"], RUNNER.J8_BASE)
                    self.assertEqual(metadata["runtime"]["epochs"], 50)
                    self.assertEqual(metadata["runtime"]["gradient_accumulation_steps"], 2)
                    self.assertEqual(option(commands[0], "--num_train_epochs"), "50")
                    self.assertNotIn("--resume_from_checkpoint", commands[0])
                    self.assertFalse(output.exists())
                for key in ("resume_ckpt_path", "finetune_init_ckpt_path",
                            "base_init_ckpt_path", "gen_init_ckpt_path", "loc_init_ckpt_path"):
                    self.assertFalse(raw.get(key))

    def test_extra_joint_arms_change_only_constant_alpha(self):
        cases = (
            ("joint_original_constloc", "joint_original", 2.0),
            ("joint_constloc1", "joint_constloc", 1.0),
            ("joint_constloc4", "joint_constloc", 4.0),
            ("joint_constloc8", "joint_constloc", 8.0),
            ("joint_constloc12", "joint_constloc8", 12.0),
            ("joint_constloc16", "joint_constloc8", 16.0),
        )
        for suffix, parent, alpha in cases:
            with self.subTest(suffix=suffix):
                experiment = RUNNER.PREFIX + suffix
                child = RUNNER.read_yaml(RUNNER.config_path(experiment))
                expected = RUNNER.read_yaml(RUNNER.config_path(RUNNER.PREFIX + parent))
                expected["alpha_loc_loss"] = alpha
                expected.pop("alpha_loc_schedule_steps", None)
                expected.pop("alpha_loc_schedule_values", None)
                self.assertEqual(child, expected)
                self.assertEqual(child["training_stop_after_step"], 2400)
                self.assertNotIn("alpha_loc_schedule_steps", child)
                self.assertNotIn("alpha_loc_schedule_values", child)
                self.assertIn(experiment, RUNNER.JOINT_BASES)
                for mode in ("minimal", "full"):
                    args = self.args("train", experiment, "--asset-mode", mode)
                    resolved = RUNNER.resolve_config(args, for_training=True)
                    RUNNER.validate_config(resolved)
                    command, metadata, out = RUNNER.training_plan(args, resolved)
                    self.assertEqual(metadata["runtime"]["epochs"], 50)
                    self.assertEqual(metadata["runtime"]["save_steps"], 1200)
                    self.assertEqual(metadata["runtime"]["gradient_accumulation_steps"], 2)
                    self.assertEqual(option(command[0], "--num_train_epochs"), "50")
                    self.assertEqual(option(command[0], "--save_steps"), "1200")
                    self.assertFalse(out.exists())

    def test_eval_subset_summary_opt_in_is_scoped_to_all_evaluation_plans(self):
        flag = "benchmark_v2_allow_map_subset_summary"
        for experiment in RUNNER.EXPERIMENTS:
            train = RUNNER.resolve_config(self.args("train", experiment), for_training=True)
            original = copy.deepcopy(train)
            identity = RUNNER.fingerprint(train, {"seed": 42})
            self.assertNotIn(flag, train)
            for command in ("infer", "metrics"):
                for split, benchmark_split in (
                    ("validation", "seen_validation"),
                    ("test", "seen_discrete_test"),
                ):
                    with self.subTest(experiment=experiment, command=command, split=split):
                        args = self.args(command, experiment, "--checkpoint-step", "2400",
                                         "--split", split)
                        _, _, evaluation = RUNNER.evaluation_plan(args, train)
                        self.assertIs(evaluation[flag], True)
                        self.assertEqual(evaluation["benchmark_v2_split"], benchmark_split)
                        self.assertEqual(evaluation["seed"], 42)
                        self.assertEqual(
                            evaluation["ckpt_path"],
                            f"outputs/csgo_1b/{experiment}/checkpoint-2400/model.safetensors",
                        )
                        self.assertNotIn(flag, train)
            self.assertEqual(train, original)
            self.assertEqual(RUNNER.fingerprint(train, {"seed": 42}), identity)

    def test_seen_ten_protocol_requires_explicit_seen_three_summary_opt_in(self):
        protocol = json.loads((REPO_ROOT / "data/csgo_benchmark_v2/benchmark_manifest.json")
                              .read_text(encoding="utf-8"))["protocol"]["seen_maps"]
        self.assertEqual(len(protocol), 10)
        self.assertEqual([name for name in protocol if name in RUNNER.MAPS], RUNNER.MAPS)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"protocol": {"seen_maps": protocol}}), encoding="utf-8")
            for split in ("validation", "test"):
                args = self.args("infer", "exp32_3maps_joint_original",
                                 "--checkpoint-step", "2400", "--split", split)
                training = RUNNER.resolve_config(self.args("train"), for_training=True)
                _, _, evaluation = RUNNER.evaluation_plan(args, training)
                selected_split = evaluation["benchmark_v2_split"]
                for maps in (RUNNER.MAPS, protocol):
                    inference_manifest = root / f"{split}_{len(maps)}_inference.json"
                    provenance = {
                        "benchmark_v2_manifest": str(manifest),
                        "benchmark_v2_split": selected_split,
                        "benchmark_v2_support_seed": None,
                        "benchmark_v2_shots_per_map": None,
                        "maps": maps,
                        "sample_count": len(maps),
                        "checkpoint": evaluation["ckpt_path"],
                        "ckpt_path": evaluation["ckpt_path"],
                        "seed": evaluation["seed"],
                    }
                    inference_manifest.write_text(json.dumps(provenance), encoding="utf-8")
                    summary_args = dict(
                        manifest=manifest,
                        split=selected_split,
                        maps=maps,
                        per_map={name: {"L2_5D": 1.0} for name in maps},
                        metrics_macro_map={"L2_5D": 1.0},
                        inference_provenance={"path": str(inference_manifest),
                                              "payload": provenance},
                        checkpoint=evaluation["ckpt_path"],
                        seed=evaluation["seed"],
                        support_seed=None,
                        shots_per_map=None,
                        sample_count=len(maps),
                    )
                    if len(maps) == 3:
                        with self.assertRaisesRegex(AggregationError,
                                                    "do not match manifest protocol"):
                            build_localization_summary(**summary_args)
                        with self.assertRaisesRegex(AggregationError,
                                                    "do not match manifest protocol"):
                            build_localization_summary(
                                allow_protocol_map_subset=False, **summary_args
                            )
                        summary = build_localization_summary(
                            allow_protocol_map_subset=evaluation[
                                "benchmark_v2_allow_map_subset_summary"
                            ], **summary_args
                        )
                    else:
                        summary = build_localization_summary(**summary_args)
                    self.assertEqual(summary["maps"], maps)

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
                args = self.args("train", "exp32_3maps_both", "--micro-batch", "4",
                                 "--stop-after-step", str(step))
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

    def test_j8_base_identity_survives_resume_and_rejects_mismatches(self):
        experiment = RUNNER.PREFIX + "both_j8"
        old_base = RUNNER.DEFAULT_BASE
        with self.assertRaisesRegex(ValueError, "J8 experiment requires"):
            RUNNER.resolve_config(
                self.args("train", experiment, "--joint-base", old_base), for_training=True
            )
        first = self.args("train", experiment)
        initial = RUNNER.resolve_config(first, for_training=True)
        _, metadata, output = RUNNER.training_plan(first, initial)
        output.mkdir(parents=True)
        snapshot = output / "resolved_config.yaml"
        snapshot.write_text(yaml.safe_dump(initial))
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
        resumed = RUNNER.resolve_config(resume, for_training=True)
        self.assertEqual(resumed["exp32_3maps_joint_base"], RUNNER.J8_BASE)
        _, resumed_metadata, _ = RUNNER.training_plan(resume, resumed)
        self.assertEqual(resumed_metadata["fingerprint"], metadata["fingerprint"])
        self.assertEqual(resumed_metadata["resume_checkpoint"], RUNNER.relative(checkpoint))
        explicit = self.args("train", experiment, "--resume", "--stop-after-step", "4800",
                             "--joint-base", RUNNER.J8_BASE)
        self.assertEqual(RUNNER.resolve_config(explicit, for_training=True), resumed)
        source_path = RUNNER.config_path(experiment)
        source = RUNNER.read_yaml(source_path)
        source["exp32_3maps_joint_base"] = old_base
        source_path.write_text(yaml.safe_dump(source))
        evaluation = self.args("infer", experiment, "--checkpoint-step", "3600")
        self.assertEqual(RUNNER.resolve_config(evaluation, for_training=False)[
            "exp32_3maps_joint_base"], RUNNER.J8_BASE)
        with self.assertRaisesRegex(ValueError, "J8 experiment requires"):
            RUNNER.resolve_config(first, for_training=True)
        source["exp32_3maps_joint_base"] = RUNNER.J8_BASE
        source_path.write_text(yaml.safe_dump(source))
        full = self.args("train", experiment, "--resume", "--stop-after-step", "0")
        full_config = RUNNER.resolve_config(full, for_training=True)
        self.assertNotIn("training_stop_after_step", full_config)
        self.assertEqual(RUNNER.fingerprint(full_config, metadata["runtime"]),
                         metadata["fingerprint"])
        saved_wrong = copy.deepcopy(initial)
        saved_wrong["exp32_3maps_joint_base"] = old_base
        snapshot.write_text(yaml.safe_dump(saved_wrong))
        for command, extra in (("train", ("--resume",)),
                               ("infer", ("--checkpoint-step", "3600")),
                               ("metrics", ("--checkpoint-step", "3600"))):
            with self.subTest(command=command), self.assertRaisesRegex(
                ValueError, "Saved J8 training recipe"
            ):
                RUNNER.resolve_config(self.args(command, experiment, *extra),
                                      for_training=command == "train")

    def test_j8_inference_routes_both_tasks_and_isolates_outputs(self):
        for suffix in RUNNER.ROUND2_J8:
            experiment = RUNNER.PREFIX + suffix
            config = RUNNER.resolve_config(self.args("train", experiment), for_training=True)
            for split, expected_splits in (("validation", ["seen_validation", "seen_validation"]),
                                           ("test", ["seen_discrete_test", "seen_continuous"])):
                with self.subTest(experiment=experiment, split=split):
                    args = self.args("infer", experiment, "--checkpoint-step", "3600",
                                     "--split", split)
                    commands, path, evaluation = RUNNER.evaluation_plan(args, config)
                    self.assertEqual(len(commands), 2 if split == "validation" else 3)
                    self.assertEqual(commands[0][1], "eval_csgo_loc.py")
                    self.assertTrue(all(item[1] == "eval_csgo.py" for item in commands[1:]))
                    self.assertEqual([option(item, "--benchmark_v2_split")
                                      for item in commands[1:]],
                                     expected_splits[1:] if split == "validation"
                                     else expected_splits)
                    self.assertIn(f"/{experiment}/checkpoint-3600/{split}/", str(path))
                    self.assertIn(f"/{experiment}/checkpoint-3600/{split}/",
                                  option(commands[0], "--output_dir"))
                    self.assertEqual(len({option(item, "--output_dir") for item in commands}),
                                     len(commands))
                    self.assertEqual(evaluation["exp32_3maps_joint_base"], RUNNER.J8_BASE)
                    for task, count in (("localization", 1),
                                        ("generation", 1 if split == "validation" else 2)):
                        selected = self.args("infer", experiment, "--checkpoint-step", "3600",
                                             "--split", split, "--task", task)
                        selected_commands, _, _ = RUNNER.evaluation_plan(selected, config)
                        self.assertEqual(len(selected_commands), count)
                        self.assertEqual([option(item, "--output_dir")
                                          for item in selected_commands],
                                         [option(item, "--output_dir") for item in commands
                                          if (item[1] == "eval_csgo_loc.py")
                                          == (task == "localization")])

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


    def test_stage2_four_arms_share_j8_initialization_and_fixed_schedule(self):
        base = RUNNER.read_yaml(RUNNER.config_path(RUNNER.J8_BASE))
        for suffix, aux, perception in (("control", False, False),
                                        ("perception", False, True),
                                        ("auxloc", True, False),
                                        ("both", True, True)):
            experiment = f"exp32_3maps_stage2_{suffix}_j8"
            for asset_mode in ("minimal", "full"):
                with self.subTest(experiment=experiment, asset_mode=asset_mode):
                    args = self.args("train", experiment, "--asset-mode", asset_mode)
                    config = RUNNER.resolve_config(args, for_training=True)
                    RUNNER.validate_config(config)
                    self.assertEqual(config["finetune_init_ckpt_path"], RUNNER.STAGE2_SOURCE)
                    self.assertTrue(config["finetune_init_strict"])
                    self.assertEqual(config["alpha_loc_loss"], 8)
                    self.assertEqual(config["training_stop_after_step"], 1200)
                    self.assertEqual(config["is_loc_aux_loss"], aux)
                    self.assertEqual(config["is_loc_perception_loss"], perception)
                    self.assertEqual(config.get("alpha_loc_aux_schedule_steps"), [0, 120] if aux else None)
                    self.assertEqual(config.get("alpha_loc_perception_schedule_steps"), [0, 120] if perception else None)
                    self.assertEqual(config.get("alpha_loc_aux_schedule_values"), [0.0, 2.0] if aux else None)
                    self.assertEqual(config.get("alpha_loc_perception_schedule_values"), [0.0, 0.1] if perception else None)
                    self.assertEqual(config["lr_scheduler_kwargs"], {"min_lr_rate": 0.1})
                    self.assertEqual(config["warmup_ratio"], 0.0)
                    if asset_mode == "full":
                        self.assertIsNone(config["benchmark_v2_asset_manifest"])
                    else:
                        self.assertIsNotNone(config["benchmark_v2_asset_manifest"])
                    for key in RUNNER.STAGE2_LR_KEYS:
                        self.assertEqual(config[key], float(base[key]) * 0.2)
                    commands, metadata, _ = RUNNER.training_plan(args, config)
                    command = commands[0]
                    for key, value in (("--max_steps", "1800"), ("--warmup_steps", "60"),
                                       ("--save_steps", "300"), ("--save_total_limit", "6"),
                                       ("--save_only_model", "False"), ("--optim", "adamw_torch"),
                                       ("--adam_beta1", "0.9"), ("--adam_beta2", "0.999"),
                                       ("--adam_epsilon", "1e-8"), ("--max_grad_norm", "1.0")):
                        self.assertEqual(option(command, key), value)
                        self.assertEqual(command.count(key), 1)
                    self.assertEqual(metadata["runtime"]["max_steps"], 1800)
                    self.assertEqual(metadata["runtime"]["gradient_accumulation_steps"], 2)
                    self.assertIsNone(metadata["source_checkpoint"])

    def test_stage2_dry_run_does_not_require_source_or_write(self):
        for suffix in RUNNER.STAGE2_J8:
            for command, extra in (("train", ()),
                                   ("infer", ("--checkpoint-step", "1200")),
                                   ("metrics", ("--checkpoint-step", "1200"))):
                experiment = RUNNER.PREFIX + suffix
                with self.subTest(experiment=experiment, command=command), patch.object(
                    RUNNER.subprocess, "run"
                ) as run, patch("sys.stdout", io.StringIO()):
                    self.assertEqual(RUNNER.main([command, "--experiment", experiment, *extra]), 0)
                    run.assert_not_called()
        self.assertFalse((self.root / "outputs").exists())
        self.assertFalse((self.root / "outputs_eval").exists())

    def test_stage2_rejects_joint_base_changes_and_stop_beyond_horizon(self):
        experiment = "exp32_3maps_stage2_both_j8"
        with self.assertRaisesRegex(ValueError, "--joint-base"):
            RUNNER.resolve_config(self.args("train", experiment, "--joint-base", RUNNER.J8_BASE),
                                  for_training=True)
        config = RUNNER.resolve_config(self.args("train", experiment, "--stop-after-step", "1801"),
                                       for_training=True)
        with self.assertRaisesRegex(ValueError, "horizon"):
            RUNNER.validate_config(config)
        config["finetune_init_ckpt_path"] = "another.safetensors"
        with self.assertRaisesRegex(ValueError, "J8 checkpoint"):
            RUNNER.validate_config(config)

    def test_stage2_resume_uses_own_full_checkpoint_without_j8_source(self):
        experiment = "exp32_3maps_stage2_both_j8"
        args = self.args("train", experiment)
        config = RUNNER.resolve_config(args, for_training=True)
        _, metadata, output = RUNNER.training_plan(args, config)
        output.mkdir(parents=True)
        metadata["source_checkpoint"] = {"path": RUNNER.STAGE2_SOURCE, "sha256": "abc",
                                         "size_bytes": 100, "step": 5850}
        (output / "experiment_plan.json").write_text(json.dumps(metadata))
        (output / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        checkpoint = output / "checkpoint-1200"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1200,
                                                                    "max_steps": 1800}))
        for name in ("training_args.bin", "model.safetensors", "rng_state_0.pth",
                     "optimizer.pt", "scheduler.pt"):
            (checkpoint / name).touch()
        resumed = self.args("train", experiment, "--resume", "--stop-after-step", "1800")
        resumed_config = RUNNER.resolve_config(resumed, for_training=True)
        commands, resumed_meta, _ = RUNNER.training_plan(resumed, resumed_config)
        self.assertEqual(resumed_meta["source_checkpoint"], metadata["source_checkpoint"])
        self.assertEqual(resumed_meta["resume_checkpoint"], RUNNER.relative(checkpoint))
        self.assertEqual(resumed_meta["fingerprint"], metadata["fingerprint"])
        self.assertNotIn("--resume_from_checkpoint", commands[0])
        self.assertFalse(RUNNER.repo_path(RUNNER.STAGE2_SOURCE).exists())
        with self.assertRaisesRegex(ValueError, "later stop"):
            RUNNER.training_plan(self.args("train", experiment, "--resume"), config)
        changed = copy.deepcopy(resumed_config)
        changed["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "recipe/layout changed"):
            RUNNER.training_plan(resumed, changed)
        missing_source = copy.deepcopy(metadata)
        missing_source.pop("source_checkpoint")
        (output / "experiment_plan.json").write_text(json.dumps(missing_source))
        with self.assertRaisesRegex(ValueError, "source provenance"):
            RUNNER.training_plan(resumed, resumed_config)

    def test_stage2_snapshot_evaluation_routes_both_tasks(self):
        experiment = "exp32_3maps_stage2_control_j8"
        config = RUNNER.resolve_config(self.args("train", experiment, "--asset-mode", "full",
                                                "--data-dir", "/dataset/full"), for_training=True)
        out = self.root / "outputs/csgo_1b" / experiment
        out.mkdir(parents=True)
        (out / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        for command in ("infer", "metrics"):
            for split, count in (("validation", 2), ("test", 3)):
                args = self.args(command, experiment, "--checkpoint-step", "1200", "--split", split)
                saved = RUNNER.resolve_config(args, for_training=False)
                planned, _, eval_config = RUNNER.evaluation_plan(args, saved)
                if command == "infer":
                    self.assertEqual(len(planned), count)
                    self.assertEqual(planned[0][1], "eval_csgo_loc.py")
                    self.assertTrue(all(item[1] == "eval_csgo.py" for item in planned[1:]))
                self.assertIn("checkpoint-1200", eval_config["ckpt_path"])
                self.assertIsNone(eval_config["benchmark_v2_asset_manifest"])
                self.assertEqual(eval_config["data_dir"], "/dataset/full")
                self.assertFalse(eval_config["is_loc_aux_loss"])
                self.assertFalse(eval_config["is_loc_perception_loss"])

    def test_stage2_source_preflight_validates_provenance_on_small_fixture(self):
        import torch
        from safetensors.torch import save_file

        source = RUNNER.repo_path(RUNNER.STAGE2_SOURCE)
        source.parent.mkdir(parents=True)
        base = RUNNER.read_yaml(RUNNER.config_path(RUNNER.J8_BASE))
        base["model_name_or_path"] = "UniLIP-1B"
        (source.parent.parent / "resolved_config.yaml").write_text(yaml.safe_dump(base))
        (source.parent / "trainer_state.json").write_text(
            json.dumps({"global_step": 5850, "max_steps": 5850})
        )
        state = {f"x.lora_{index}": torch.zeros(1) for index in range(1124)}
        state.update({
            f"{prefix}layer.lora_A.weight": torch.zeros(1)
            for prefix in ("language_model.model.", "model.llm_connector.",
                           "model.dit.", "model.action_dit.")
        })
        state.update({f"x.weight_{index}": torch.zeros(1) for index in range(1697)})
        state.update({key: torch.zeros(1) for key in (
            "language_model.lm_head.weight", "model.action_dit_norm.weight",
            "model.action_dit_connector.0.bias", "model.action_dit_projector.0.bias",
            "multi_modal_projector.0.bias")})
        save_file(state, str(source))
        config = RUNNER.resolve_config(self.args("train", "exp32_3maps_stage2_control_j8"),
                                       for_training=True)
        provenance = RUNNER.preflight_stage2_source(config)
        self.assertEqual(provenance["step"], 5850)
        self.assertEqual(provenance["tensor_count"], 2830)
        self.assertEqual(provenance["lora_tensor_count"], 1128)
        self.assertEqual(len(provenance["sha256"]), 64)
        config["action_dim"] = 6
        with self.assertRaisesRegex(ValueError, "action_dim"):
            RUNNER.preflight_stage2_source(config)
        config["action_dim"] = 5
        (source.parent / "trainer_state.json").write_text(
            json.dumps({"global_step": 5700, "max_steps": 5850})
        )
        with self.assertRaisesRegex(ValueError, "complete"):
            RUNNER.preflight_stage2_source(config)



if __name__ == "__main__":
    unittest.main()
