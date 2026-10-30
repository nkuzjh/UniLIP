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

    def final_fixture(self, experiment: str, step: int, *, root_state: bool = True) -> Path:
        config = RUNNER.resolve_config(self.args("train", experiment), for_training=True)
        out = self.root / "outputs/csgo_1b" / experiment
        out.mkdir(parents=True)
        (out / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        checkpoint = out / f"checkpoint-{step}"
        checkpoint.mkdir()
        state = json.dumps({"global_step": step, "max_steps": step})
        (checkpoint / "trainer_state.json").write_text(state)
        (checkpoint / "model.safetensors").touch()
        if root_state:
            (out / "trainer_state.json").write_text(state)
        return checkpoint

    def test_matrix_has_seven_original_six_extra_and_seven_second_round_arms(self):
        self.assertEqual(len(RUNNER.ROUND1), 7)
        self.assertEqual(len(RUNNER.ROUND1_EXTRA), 6)
        self.assertEqual(len(RUNNER.ROUND2), 4)
        self.assertEqual(
            RUNNER.ROUND2, ("aux_control", "perception", "auxloc", "both")
        )
        self.assertEqual(RUNNER.ROUND2_J8, ("perception_j8", "auxloc_j8", "both_j8"))
        self.assertEqual(len(RUNNER.LEGACY_EXPERIMENTS), 35)
        self.assertEqual(len(RUNNER.EXPERIMENTS), 38)
        self.assertEqual(RUNNER.EXPERIMENTS[-1], RUNNER.PREFIX + "B2_S3_J8")
        self.assertNotIn(RUNNER.PREFIX + "B2_S3_J8", RUNNER.LEGACY_EXPERIMENTS)
        self.assertEqual(len(RUNNER.STAGE2_J8), 4)
        self.assertEqual(len(RUNNER.STAGE2_NEW_ARMS), 12)
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
                        step = (1200 if split == "validation" else 1800) if experiment in RUNNER.STAGE2_NEW_EXPERIMENTS + RUNNER.STAGE2_SINGLE_EXPERIMENTS else 2400
                        args = self.args(command, experiment, "--checkpoint-step", str(step),
                                         "--split", split)
                        _, _, evaluation = RUNNER.evaluation_plan(args, train)
                        self.assertIs(evaluation[flag], True)
                        self.assertEqual(evaluation["benchmark_v2_split"], benchmark_split)
                        self.assertEqual(evaluation["seed"], 42)
                        self.assertEqual(
                            evaluation["ckpt_path"],
                            f"outputs/csgo_1b/{experiment}/checkpoint-{step}/model.safetensors",
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

    def test_final_test_plans_reuse_numeric_paths_and_active_task_routing(self):
        cases = (
            ("loc_single", 5850, 1, 0),
            ("gen_single", 5850, 2, 8),
            ("joint_constloc8", 5850, 3, 8),
            ("perception_j8", 5900, 3, 8),
            ("stage2_control_j8", 1800, 3, 8),
        )
        for suffix, step, infer_count, metric_count in cases:
            experiment = RUNNER.PREFIX + suffix
            self.final_fixture(experiment, step)
            for command, count in (("infer", infer_count), ("metrics", metric_count)):
                with self.subTest(experiment=experiment, command=command):
                    args = self.args(command, experiment, "--checkpoint-step", "final",
                                     "--split", "test", "--task", "all")
                    config = RUNNER.resolve_config(args, for_training=False)
                    commands, path, evaluated = RUNNER.evaluation_plan(args, config)
                    self.assertEqual(len(commands), count)
                    self.assertIn(f"/{experiment}/checkpoint-{step}/test/", str(path))
                    self.assertEqual(evaluated["ckpt_path"],
                                     f"outputs/csgo_1b/{experiment}/checkpoint-{step}/model.safetensors")
                    self.assertNotIn("checkpoint-final", str(path))
                    numeric = self.args(command, experiment, "--checkpoint-step", str(step),
                                        "--split", "test", "--task", "all")
                    self.assertEqual(RUNNER.evaluation_plan(numeric, config),
                                     (commands, path, evaluated))
                    if command == "infer":
                        self.assertEqual([item[1] for item in commands],
                                         (["eval_csgo_loc.py"] if suffix == "loc_single" else
                                          ["eval_csgo.py", "eval_csgo.py"] if suffix == "gen_single" else
                                          ["eval_csgo_loc.py", "eval_csgo.py", "eval_csgo.py"]))
                        self.assertEqual([option(item, "--benchmark_v2_split") for item in commands],
                                         (["seen_discrete_test"] if suffix == "loc_single" else
                                          ["seen_discrete_test", "seen_continuous"] if suffix == "gen_single" else
                                          ["seen_discrete_test", "seen_discrete_test", "seen_continuous"]))

    def test_final_requires_completed_matching_state_and_model(self):
        experiment = RUNNER.PREFIX + "joint_constloc8"
        checkpoint = self.final_fixture(experiment, 5850)
        out = checkpoint.parent
        args = self.args("infer", experiment, "--checkpoint-step", "final", "--split", "test")
        config = RUNNER.resolve_config(args, for_training=False)
        self.assertEqual(RUNNER.final_checkpoint_step(experiment), 5850)
        (out / "trainer_state.json").write_text(json.dumps({"global_step": 4800, "max_steps": 5850}))
        with self.assertRaisesRegex(ValueError, "not reached max_steps"):
            RUNNER.evaluation_plan(args, config)
        (out / "trainer_state.json").write_text(json.dumps({"global_step": 5850, "max_steps": 5850}))
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 4800, "max_steps": 5850}))
        with self.assertRaisesRegex(ValueError, "step/global_step/max_steps differ"):
            RUNNER.evaluation_plan(args, config)
        (checkpoint / "trainer_state.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing Trainer state"):
            RUNNER.evaluation_plan(args, config)
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 5850, "max_steps": 5850}))
        (checkpoint / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "missing model weights"):
            RUNNER.evaluation_plan(args, config)
        (checkpoint / "model.safetensors").touch()
        (out / "resolved_config.yaml").unlink()
        with self.assertRaisesRegex(ValueError, "saved training recipe"):
            RUNNER.evaluation_plan(args, config)

    def test_final_without_root_state_requires_checkpoint_state(self):
        experiment = RUNNER.PREFIX + "stage2_both_j8"
        checkpoint = self.final_fixture(experiment, 1800, root_state=False)
        self.assertEqual(RUNNER.final_checkpoint_step(experiment), 1800)
        (checkpoint / "trainer_state.json").unlink()
        with self.assertRaisesRegex(ValueError, "Expected one completed final checkpoint"):
            RUNNER.final_checkpoint_step(experiment)
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1200, "max_steps": 1800}))
        with self.assertRaisesRegex(ValueError, "directory and Trainer step differ"):
            RUNNER.final_checkpoint_step(experiment)

    def test_final_dry_run_and_command_scope(self):
        experiment = RUNNER.PREFIX + "joint_constloc8"
        self.final_fixture(experiment, 5850)
        self.assertEqual(self.args("infer", experiment, "--checkpoint-step", "5850").checkpoint_step, 5850)
        for command in ("infer", "metrics"):
            with self.subTest(command=command), patch.object(RUNNER.subprocess, "run") as run, \
                    patch("sys.stdout", io.StringIO()) as output:
                self.assertEqual(RUNNER.main([command, "--experiment", experiment,
                                              "--checkpoint-step", "final", "--split", "test"]), 0)
                self.assertIn("checkpoint-5850/test", output.getvalue())
                run.assert_not_called()
        self.assertFalse((self.root / "outputs_eval").exists())
        for command in ("train", "validate"):
            with self.subTest(command=command), patch("sys.stderr", io.StringIO()) as error:
                self.assertEqual(RUNNER.main([command, "--experiment", experiment,
                                              "--checkpoint-step", "final"]), 1)
                self.assertIn("applies only to infer/metrics", error.getvalue())

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




    def test_single_stage2_recipes_runtime_and_legacy_defaults(self):
        for name, base in (("g0", "gen_single"), ("l0", "loc_single")):
            experiment = RUNNER.PREFIX + "stage2_control_" + name
            args = self.args("train", experiment)
            config = RUNNER.resolve_config(args, for_training=True)
            RUNNER.validate_config(config)
            parent = RUNNER.read_yaml(RUNNER.config_path(RUNNER.PREFIX + base))
            for key in ("is_multi_task", "is_multi_task_balanced", "task_mix_ratio",
                        "alpha_loc_loss", "enable_gen_head_lora", "enable_loc_head_lora",
                        "freeze_inactive_head", "fix_vit", "fix_llm", "fix_connect", "fix_dit",
                        "lora_r", "lora_alpha", "lora_dropout"):
                self.assertEqual(config[key], parent[key])
            for key, peak in RUNNER.STAGE2_LR_PEAKS.items():
                self.assertEqual(config[key], peak)
            self.assertEqual(config["finetune_init_ckpt_path"], RUNNER.stage2_source_path(experiment))
            self.assertTrue(config["finetune_init_strict"])
            self.assertNotIn("training_stop_after_step", config)
            commands, metadata, _ = RUNNER.training_plan(args, config)
            self.assertEqual(metadata["runtime"]["micro_batch"], 4)
            self.assertEqual(metadata["runtime"]["gradient_accumulation_steps"], 32)
            self.assertEqual(metadata["runtime"]["save_steps"], 600)
            self.assertEqual(option(commands[0], "--save_steps"), "600")
            self.assertEqual(option(commands[0], "--max_steps"), "1800")
            self.assertEqual(option(commands[0], "--warmup_steps"), "60")
            self.assertNotIn("--resume_from_checkpoint", commands[0])
            for key, value in (("alpha_loc_loss", 8.0), ("task_mix_ratio", 0.5),
                               ("enable_gen_head_lora", not config["enable_gen_head_lora"]),
                               ("is_loc_aux_loss", True), ("finetune_init_strict", False)):
                invalid = copy.deepcopy(config)
                invalid[key] = value
                with self.assertRaisesRegex(ValueError, "single-task stage 2"):
                    RUNNER.validate_config(invalid)
            for stop in (None, 0, -1, "1200", 1801):
                invalid = copy.deepcopy(config)
                invalid["training_stop_after_step"] = stop
                with self.assertRaisesRegex(ValueError, "stage 2 stop|Stage 2 stop"):
                    RUNNER.validate_config(invalid)
        old_args = self.args("train", RUNNER.PREFIX + "stage2_control_j8")
        old_config = RUNNER.resolve_config(old_args, for_training=True)
        old_commands, old_metadata, _ = RUNNER.training_plan(old_args, old_config)
        self.assertEqual(old_metadata["runtime"]["micro_batch"], 64)
        self.assertEqual(old_metadata["runtime"]["gradient_accumulation_steps"], 2)
        self.assertEqual(option(old_commands[0], "--save_steps"), "300")

    def test_single_stage2_evaluation_routes_only_active_task_at_saved_nodes(self):
        for name, active, inactive in (("g0", "generation", "localization"),
                                       ("l0", "localization", "generation")):
            experiment = RUNNER.PREFIX + "stage2_control_" + name
            config = RUNNER.resolve_config(self.args("train", experiment), for_training=True)
            for split, step in (("validation", 1200), ("validation", 1800), ("test", 1800)):
                for action in ("infer", "metrics"):
                    for task in ("all", active):
                        args = self.args(action, experiment, "--split", split,
                                         "--checkpoint-step", str(step), "--task", task)
                        commands, _, _ = RUNNER.evaluation_plan(args, config)
                        scripts = [command[1] for command in commands]
                        if name == "g0":
                            self.assertEqual(scripts.count("eval_csgo_loc.py"), 0)
                            self.assertEqual(scripts.count("eval_csgo.py"),
                                             (1 if split == "validation" else 2) if action == "infer" else 0)
                            self.assertEqual(len(commands), (1 if split == "validation" else 2)
                                             if action == "infer" else (4 if split == "validation" else 8))
                        else:
                            self.assertEqual(scripts, ["eval_csgo_loc.py"] if action == "infer" else [])
                    with self.assertRaisesRegex(ValueError, "inactive"):
                        RUNNER.evaluation_plan(self.args(action, experiment, "--split", split,
                                                         "--checkpoint-step", str(step),
                                                         "--task", inactive), config)
            for split, step in (("validation", 600), ("test", 600), ("test", 1200)):
                with self.assertRaisesRegex(ValueError, "requires checkpoint step"):
                    RUNNER.evaluation_plan(self.args("infer", experiment, "--split", split,
                                                     "--checkpoint-step", str(step)), config)
            self.final_fixture(experiment, 1800)
            final = self.args("infer", experiment, "--checkpoint-step", "final", "--split", "test")
            commands, _, _ = RUNNER.evaluation_plan(final, config)
            self.assertEqual(len(commands), 2 if name == "g0" else 1)


    def test_single_stage2_source_preflight_checks_identity_state_and_header(self):
        import torch
        from safetensors.torch import save_file

        heads = ("language_model.lm_head.weight", "model.action_dit_norm.weight",
                 "model.action_dit_connector.0.bias", "model.action_dit_projector.0.bias",
                 "multi_modal_projector.0.bias")
        for name, base in (("g0", "gen_single"), ("l0", "loc_single")):
            experiment = RUNNER.PREFIX + "stage2_control_" + name
            source = RUNNER.repo_path(RUNNER.stage2_source_path(experiment))
            source.parent.mkdir(parents=True)
            recipe = RUNNER.read_yaml(RUNNER.config_path(RUNNER.PREFIX + base))
            recipe["model_name_or_path"] = "UniLIP-1B"
            recipe.pop("training_stop_after_step", None)
            recipe_path = source.parent.parent / "resolved_config.yaml"
            recipe_path.write_text(yaml.safe_dump(recipe))
            (source.parent / "trainer_state.json").write_text(
                json.dumps({"global_step": 5850, "max_steps": 5850}))
            _, total, lora_total, prefixes, forbidden = RUNNER.STAGE2_SINGLE_SOURCES[experiment]
            state = {key: torch.zeros(1) for key in heads}
            state.update({f"{prefix}layer.lora_A.weight": torch.zeros(1) for prefix in prefixes})
            state.update({f"x.lora_{i}": torch.zeros(1) for i in range(lora_total - len(prefixes))})
            state.update({f"x.weight_{i}": torch.zeros(1) for i in range(total - lora_total - len(heads))})
            save_file(state, str(source))
            config = RUNNER.resolve_config(self.args("train", experiment), for_training=True)
            provenance = RUNNER.preflight_stage2_source(config)
            self.assertEqual(provenance["path"], RUNNER.stage2_source_path(experiment))
            self.assertEqual((provenance["tensor_count"], provenance["lora_tensor_count"]),
                             (total, lora_total))
            self.assertEqual(len(provenance["sha256"]), 64)
            recipe["task_mix_ratio"] = 0.5
            recipe_path.write_text(yaml.safe_dump(recipe))
            with self.assertRaisesRegex(ValueError, "source recipe differs"):
                RUNNER.preflight_stage2_source(config)
            recipe["task_mix_ratio"] = 0.0 if name == "g0" else 1.0
            recipe_path.write_text(yaml.safe_dump(recipe))
            state_path = source.parent / "trainer_state.json"
            state_path.rename(source.parent.parent / "trainer_state.json")
            RUNNER.preflight_stage2_source(config)
            (source.parent.parent / "trainer_state.json").write_text(
                json.dumps({"global_step": 5700, "max_steps": 5850}))
            with self.assertRaisesRegex(ValueError, "not complete"):
                RUNNER.preflight_stage2_source(config)
            (source.parent.parent / "trainer_state.json").write_text(
                json.dumps({"global_step": 5850, "max_steps": 5850}))
            bad = dict(state)
            bad.pop(next(k for k in bad if k.startswith("x.lora_")))
            save_file(bad, str(source))
            with self.assertRaisesRegex(ValueError, "tensor header"):
                RUNNER.preflight_stage2_source(config)
            bad[f"{forbidden[0]}layer.lora_A.weight"] = torch.zeros(1)
            save_file(bad, str(source))
            with self.assertRaisesRegex(ValueError, "inactive-head LoRA"):
                RUNNER.preflight_stage2_source(config)
            recipe["model_name_or_path"] = "wrong-model"
            recipe_path.write_text(yaml.safe_dump(recipe))
            with self.assertRaisesRegex(ValueError, "model identity"):
                RUNNER.preflight_stage2_source(config)
            recipe["model_name_or_path"] = "UniLIP-1B"
            recipe_path.write_text(yaml.safe_dump(recipe))
            source.unlink()
            with self.assertRaisesRegex(ValueError, "Missing single-task source model"):
                RUNNER.preflight_stage2_source(config)

    def test_single_stage2_resume_retains_provenance_without_source_file(self):
        for name in ("g0", "l0"):
            experiment = RUNNER.PREFIX + "stage2_control_" + name
            args = self.args("train", experiment)
            config = RUNNER.resolve_config(args, for_training=True)
            _, metadata, out = RUNNER.training_plan(args, config)
            out.mkdir(parents=True)
            source = {"path": RUNNER.stage2_source_path(experiment), "sha256": "fixture"}
            metadata["source_checkpoint"] = source
            (out / "experiment_plan.json").write_text(json.dumps(metadata))
            (out / "resolved_config.yaml").write_text(yaml.safe_dump(config))
            checkpoint = out / "checkpoint-600"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"global_step": 600, "max_steps": 1800}))
            for filename in ("training_args.bin", "model.safetensors", "rng_state.pth",
                             "optimizer.pt", "scheduler.pt"):
                (checkpoint / filename).touch()
            resume_args = self.args("train", experiment, "--resume")
            resumed = RUNNER.resolve_config(resume_args, for_training=True)
            commands, result, _ = RUNNER.training_plan(resume_args, resumed)
            self.assertEqual(result["source_checkpoint"], source)
            self.assertEqual(result["resume_checkpoint"], RUNNER.relative(checkpoint))
            self.assertEqual(result["fingerprint"], metadata["fingerprint"])
            self.assertNotIn("--resume_from_checkpoint", commands[0])
            wrong = copy.deepcopy(metadata)
            wrong["source_checkpoint"]["path"] = RUNNER.STAGE2_SOURCE
            (out / "experiment_plan.json").write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, "source provenance"):
                RUNNER.training_plan(resume_args, resumed)

    def test_single_stage2_dry_run_has_no_side_effects(self):
        for experiment in RUNNER.STAGE2_SINGLE_EXPERIMENTS:
            for action, extra in (("train", []),
                                  ("infer", ["--split", "test", "--checkpoint-step", "1800"]),
                                  ("metrics", ["--split", "test", "--checkpoint-step", "1800"])):
                with patch.object(RUNNER.subprocess, "run") as run, patch("sys.stdout", io.StringIO()):
                    self.assertEqual(RUNNER.main([action, "--experiment", experiment, *extra]), 0)
                    run.assert_not_called()
                self.assertFalse((self.root / "outputs/csgo_1b" / experiment).exists())
                self.assertFalse((self.root / "outputs_eval/benchmark_v2" / experiment).exists())

    def test_new_stage2_recipes_change_only_the_declared_factor(self):
        expected_names = (
            "P1_S1_J8", "U1_S2_J8", "B1_S3_J8", "P2_S1_J8", "P3_S1_J8",
            "U2_S2_J8", "U3_S2_J8", "P4_S1_J8", "U4_S2_J8", "U5_S2_J8",
            "P5_S1_J8", "B2_S3_J8",
        )
        self.assertEqual(tuple(RUNNER.STAGE2_NEW_ARMS), expected_names)
        for suffix, (base_suffix, changes, removed) in RUNNER.STAGE2_NEW_ARMS.items():
            with self.subTest(suffix=suffix):
                experiment = RUNNER.PREFIX + suffix
                self.assertTrue(RUNNER.is_stage2(experiment))
                base = RUNNER.read_yaml(RUNNER.config_path(RUNNER.PREFIX + base_suffix))
                expected = copy.deepcopy(base)
                expected.update(changes)
                for key in removed:
                    expected.pop(key)
                expected["exp32_3maps_experiment"] = experiment
                expected["training_stop_after_step"] = 1800
                raw = RUNNER.read_yaml(RUNNER.config_path(experiment))
                self.assertEqual(raw, expected)
                config = RUNNER.resolve_config(self.args(experiment=experiment), for_training=True)
                RUNNER.validate_config(config)
                self.assertEqual(config["benchmark_v2_asset_manifest"],
                                 "data/csgo_benchmark_v2/minimal_dataset_report.json")
                self.assertEqual(config["finetune_init_ckpt_path"], RUNNER.STAGE2_SOURCE)
                for stem in ("alpha_loc_aux", "alpha_loc_perception"):
                    steps = config.get(stem + "_schedule_steps")
                    if steps is None:
                        self.assertNotIn(stem + "_schedule_values", config)
                    else:
                        self.assertEqual(steps, [0, 120])
                        self.assertEqual(config[stem + "_schedule_values"][0], 0.0)
                if suffix == "P1_S1_J8":
                    self.assertEqual(config["alpha_loc_perception_loss"], 0.1)
                    self.assertNotIn("alpha_loc_perception_schedule_steps", config)
                if suffix == "U1_S2_J8":
                    self.assertEqual(config["alpha_loc_aux_loss"], 2.0)
                    self.assertNotIn("alpha_loc_aux_schedule_steps", config)
                commands, metadata, output = RUNNER.training_plan(self.args(experiment=experiment), config)
                self.assertEqual(metadata["runtime"]["save_steps"], 600)
                self.assertEqual(metadata["runtime"]["max_steps"], 1800)
                self.assertEqual(metadata["runtime"]["warmup_steps"], 60)
                self.assertEqual(metadata["runtime"]["save_total_limit"], 6)
                self.assertEqual(option(commands[0], "--save_steps"), "600")
                self.assertEqual(option(commands[0], "--max_steps"), "1800")
                self.assertEqual(option(commands[0], "--optim"), "adamw_torch")
                self.assertFalse(output.exists())
                mutated = copy.deepcopy(config)
                mutated["aux_loc_combined_candidate_tau"] = 0.123
                with self.assertRaisesRegex(ValueError, "single-factor"):
                    RUNNER.validate_config(mutated)

    def test_b1_combines_p3_perception_and_u2_auxloc(self):
        p3 = RUNNER.read_yaml(RUNNER.config_path(RUNNER.PREFIX + "P3_S1_J8"))
        u2 = RUNNER.read_yaml(RUNNER.config_path(RUNNER.PREFIX + "U2_S2_J8"))
        b1 = RUNNER.read_yaml(RUNNER.config_path(RUNNER.PREFIX + "B1_S3_J8"))
        expected = copy.deepcopy(p3)
        expected["exp32_3maps_experiment"] = RUNNER.PREFIX + "B1_S3_J8"
        for key in ("is_loc_aux_loss", "is_aux_loc_combined_em_unc_loss",
                    "alpha_loc_aux_loss", "alpha_loc_aux_schedule_steps",
                    "alpha_loc_aux_schedule_values"):
            expected[key] = copy.deepcopy(u2[key])
        self.assertEqual(b1, expected)
        self.assertEqual(b1["alpha_loc_perception_schedule_values"], [0.0, 0.05])
        self.assertEqual(b1["alpha_loc_aux_schedule_values"], [0.0, 1.0])
        self.assertEqual(b1["alpha_loc_perception_loss"], 0.0)
        self.assertEqual(b1["alpha_loc_aux_loss"], 0.0)
        self.assertTrue(b1["loc_perception_use_attention_weight"])
        self.assertEqual(b1["aux_loc_combined_candidate_tau"], 1.0)

    def test_b2_combines_p4_perception_and_u2_auxloc_from_j8_only(self):
        b1_name = RUNNER.PREFIX + "B1_S3_J8"
        b2_name = RUNNER.PREFIX + "B2_S3_J8"
        b1 = RUNNER.read_yaml(RUNNER.config_path(b1_name))
        b2 = RUNNER.read_yaml(RUNNER.config_path(b2_name))
        p4 = RUNNER.read_yaml(RUNNER.config_path(RUNNER.PREFIX + "P4_S1_J8"))
        u2 = RUNNER.read_yaml(RUNNER.config_path(RUNNER.PREFIX + "U2_S2_J8"))
        expected = copy.deepcopy(b1)
        expected["exp32_3maps_experiment"] = b2_name
        expected["loc_perception_use_attention_weight"] = False
        self.assertEqual(b2, expected)
        for key in ("loc_perception_use_attention_weight", "loc_perception_feature_source",
                    "loc_perception_loss_type", "alpha_loc_perception_loss",
                    "alpha_loc_perception_schedule_steps", "alpha_loc_perception_schedule_values"):
            self.assertEqual(b2[key], p4[key], key)
        for key in ("is_loc_aux_loss", "is_aux_loc_combined_em_unc_loss",
                    "alpha_loc_aux_loss", "alpha_loc_aux_schedule_steps",
                    "alpha_loc_aux_schedule_values", "aux_loc_combined_num_samples",
                    "aux_loc_combined_candidate_tau"):
            self.assertEqual(b2[key], u2[key], key)
        self.assertEqual(b2["finetune_init_ckpt_path"], RUNNER.STAGE2_SOURCE)
        self.assertEqual(b2["exp32_3maps_joint_base"], RUNNER.J8_BASE)
        self.assertNotIn("B1_S3_J8", b2["finetune_init_ckpt_path"])
        self.assertEqual(b2["train_maps"], ["de_ancient", "de_dust2", "de_nuke"])

    def test_b2_guards_attention_ramps_and_source(self):
        name = RUNNER.PREFIX + "B2_S3_J8"
        config = RUNNER.resolve_config(self.args("train", name), for_training=True)
        RUNNER.validate_config(config)
        for key, value in (("loc_perception_use_attention_weight", True),
                           ("alpha_loc_perception_schedule_steps", [0, 180]),
                           ("alpha_loc_perception_schedule_values", [0.0, 0.1]),
                           ("alpha_loc_aux_schedule_values", [0.0, 2.0]),
                           ("finetune_init_ckpt_path", "outputs/csgo_1b/exp32_3maps_B1_S3_J8/checkpoint-1800/model.safetensors")):
            with self.subTest(key=key):
                invalid = copy.deepcopy(config)
                invalid[key] = value
                with self.assertRaisesRegex(ValueError, "single-factor"):
                    RUNNER.validate_config(invalid)

    def test_b2_explicit_microbatch_training_and_eval_plan(self):
        name = RUNNER.PREFIX + "B2_S3_J8"
        args = self.args("train", name, "--micro-batch", "4")
        config = RUNNER.resolve_config(args, for_training=True)
        RUNNER.validate_config(config)
        commands, metadata, output = RUNNER.training_plan(args, config)
        runtime = metadata["runtime"]
        self.assertEqual((runtime["micro_batch"], runtime["gradient_accumulation_steps"]), (4, 32))
        self.assertEqual((runtime["max_steps"], runtime["warmup_steps"], runtime["save_steps"],
                          runtime["save_total_limit"], runtime["optimizer"]),
                         (1800, 60, 600, 6, "adamw_torch"))
        self.assertEqual(option(commands[0], "--per_device_train_batch_size"), "4")
        self.assertEqual(option(commands[0], "--gradient_accumulation_steps"), "32")
        self.assertEqual(option(commands[0], "--max_grad_norm"), "1.0")
        self.assertEqual(config["csgo_loss_per_microbatch_mean"], True)
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["weight_decay"], 0.0)
        self.assertEqual(config["lr_scheduler_type"], "cosine_with_min_lr")
        self.assertEqual(config["lr_scheduler_kwargs"], {"min_lr_rate": 0.1})
        self.assertIsNone(metadata["resume_checkpoint"])
        self.assertFalse(output.exists())
        default_args = self.args("train", name)
        default_config = RUNNER.resolve_config(default_args, for_training=True)
        _, default_metadata, _ = RUNNER.training_plan(default_args, default_config)
        self.assertEqual(default_metadata["runtime"]["micro_batch"], 64)
        for split, step, infer_count, metric_count in (("validation", 1200, 2, 4),
                                                       ("validation", 1800, 2, 4),
                                                       ("test", 1800, 3, 8)):
            for task, count in (("infer", infer_count), ("metrics", metric_count)):
                with self.subTest(split=split, step=step, task=task):
                    eval_args = self.args(task, name, "--split", split, "--checkpoint-step", str(step))
                    routed, path, evaluated = RUNNER.evaluation_plan(eval_args, config)
                    self.assertEqual(len(routed), count)
                    self.assertIn(f"/{name}/checkpoint-{step}/{split}/", str(path))
                    self.assertEqual(evaluated["ckpt_path"],
                                     f"outputs/csgo_1b/{name}/checkpoint-{step}/model.safetensors")
                    self.assertFalse(path.exists())
        for split, step in (("validation", 600), ("test", 1200)):
            with self.assertRaisesRegex(ValueError, "requires checkpoint step"):
                RUNNER.evaluation_plan(self.args("infer", name, "--split", split,
                                                 "--checkpoint-step", str(step)), config)
        with patch.object(RUNNER.subprocess, "run") as run, patch("sys.stdout", io.StringIO()):
            self.assertEqual(RUNNER.main(["train", "--experiment", name, "--micro-batch", "4"]), 0)
            run.assert_not_called()
        self.assertFalse((self.root / "outputs").exists())
        self.assertFalse((self.root / "outputs_eval").exists())

    def test_p4_and_u4_change_only_one_field_from_p3_and_u2(self):
        for source_suffix, target_suffix, key, value in (
            ("P3_S1_J8", "P4_S1_J8", "loc_perception_use_attention_weight", False),
            ("U2_S2_J8", "U4_S2_J8", "aux_loc_combined_candidate_tau", 0.001),
        ):
            with self.subTest(target=target_suffix):
                source = RUNNER.read_yaml(RUNNER.config_path(RUNNER.PREFIX + source_suffix))
                target = RUNNER.read_yaml(RUNNER.config_path(RUNNER.PREFIX + target_suffix))
                expected = copy.deepcopy(source)
                expected["exp32_3maps_experiment"] = RUNNER.PREFIX + target_suffix
                expected[key] = value
                self.assertEqual(target, expected)
                self.assertEqual(target["alpha_loc_perception_schedule_values"]
                                 if target_suffix.startswith("P") else target["alpha_loc_aux_schedule_values"],
                                 [0.0, 0.05] if target_suffix.startswith("P") else [0.0, 1.0])

    def test_revised_arms_reject_old_recipe_and_old_checkpoint_identity(self):
        for suffix in ("B1_S3_J8", "P4_S1_J8", "U4_S2_J8"):
            with self.subTest(suffix=suffix):
                experiment = RUNNER.PREFIX + suffix
                args = self.args("train", experiment)
                new_config = RUNNER.resolve_config(args, for_training=True)
                RUNNER.validate_config(new_config)
                _, metadata, output = RUNNER.training_plan(args, new_config)
                old_config = copy.deepcopy(new_config)
                if suffix == "B1_S3_J8":
                    old_config["alpha_loc_perception_loss"] = 0.1
                    old_config["alpha_loc_aux_loss"] = 2.0
                    for stem in ("alpha_loc_perception", "alpha_loc_aux"):
                        old_config.pop(stem + "_schedule_steps")
                        old_config.pop(stem + "_schedule_values")
                elif suffix == "P4_S1_J8":
                    old_config["alpha_loc_perception_schedule_values"] = [0.0, 0.1]
                else:
                    old_config["alpha_loc_aux_schedule_values"] = [0.0, 2.0]
                with self.assertRaisesRegex(ValueError, "single-factor"):
                    RUNNER.validate_config(old_config)
                old_fingerprint = RUNNER.fingerprint(old_config, metadata["runtime"])
                self.assertNotEqual(old_fingerprint, metadata["fingerprint"])
                output.mkdir(parents=True)
                metadata["fingerprint"] = old_fingerprint
                metadata["source_checkpoint"] = {"path": RUNNER.STAGE2_SOURCE, "sha256": "fixture"}
                plan = output / "experiment_plan.json"
                plan.write_text(json.dumps(metadata))
                checkpoint = output / "checkpoint-600"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps({"global_step": 600, "max_steps": 1800}))
                for name in ("training_args.bin", "model.safetensors", "rng_state_0.pth",
                             "optimizer.pt", "scheduler.pt"):
                    (checkpoint / name).touch()
                resumed_args = self.args("train", experiment, "--resume")
                with self.assertRaisesRegex(ValueError, "recipe/layout changed"):
                    RUNNER.training_plan(resumed_args, new_config)
                metadata["fingerprint"] = RUNNER.fingerprint(new_config, metadata["runtime"])
                plan.write_text(json.dumps(metadata))
                commands, resumed, _ = RUNNER.training_plan(resumed_args, new_config)
                self.assertEqual(resumed["resume_checkpoint"], RUNNER.relative(checkpoint))
                self.assertNotIn("--finetune_init_ckpt_path", commands[0])

    def test_u5_is_u2_with_only_a_lower_aux_ramp_endpoint(self):
        u2_name = RUNNER.PREFIX + "U2_S2_J8"
        u5_name = RUNNER.PREFIX + "U5_S2_J8"
        u2 = RUNNER.read_yaml(RUNNER.config_path(u2_name))
        u5 = RUNNER.read_yaml(RUNNER.config_path(u5_name))
        expected = copy.deepcopy(u2)
        expected["exp32_3maps_experiment"] = u5_name
        expected["alpha_loc_aux_schedule_values"] = [0.0, 0.5]
        self.assertEqual(u5, expected)
        self.assertEqual(u5["alpha_loc_aux_schedule_steps"], [0, 120])
        self.assertEqual(u5["alpha_loc_aux_loss"], 0.0)
        self.assertEqual(u5["alpha_loc_loss"], 8.0)
        self.assertFalse(u5["is_loc_perception_loss"])
        self.assertEqual(u5["finetune_init_ckpt_path"], RUNNER.STAGE2_SOURCE)
        self.assertNotIn(u2_name, u5["finetune_init_ckpt_path"])

        args = self.args("train", u5_name)
        config = RUNNER.resolve_config(args, for_training=True)
        RUNNER.validate_config(config)
        commands, metadata, output = RUNNER.training_plan(args, config)
        self.assertEqual(metadata["runtime"]["max_steps"], 1800)
        self.assertEqual(metadata["runtime"]["save_steps"], 600)
        self.assertEqual(metadata["runtime"]["optimizer"], "adamw_torch")
        self.assertIsNone(metadata["resume_checkpoint"])
        self.assertIsNone(metadata["source_checkpoint"])
        self.assertEqual(option(commands[0], "--max_steps"), "1800")
        self.assertEqual(option(commands[0], "--save_steps"), "600")
        self.assertNotIn("--resume_from_checkpoint", commands[0])
        self.assertFalse(output.exists())

        for split, step in (("validation", 1200), ("validation", 1800), ("test", 1800)):
            for task, expected_count in (("infer", 2 if split == "validation" else 3),
                                         ("metrics", 4 if split == "validation" else 8)):
                with self.subTest(split=split, step=step, task=task):
                    eval_args = self.args(task, u5_name, "--split", split,
                                          "--checkpoint-step", str(step))
                    routed, path, evaluated = RUNNER.evaluation_plan(eval_args, config)
                    self.assertEqual(len(routed), expected_count)
                    self.assertIn(f"/{u5_name}/checkpoint-{step}/{split}/", str(path))
                    self.assertEqual(evaluated["ckpt_path"],
                                     f"outputs/csgo_1b/{u5_name}/checkpoint-{step}/model.safetensors")
                    self.assertEqual(evaluated["benchmark_v2_split"],
                                     "seen_validation" if split == "validation" else "seen_discrete_test")
                    self.assertFalse(evaluated["is_loc_aux_loss"])

    def test_p5_is_p3_with_only_a_lower_perception_ramp_endpoint(self):
        p3_name = RUNNER.PREFIX + "P3_S1_J8"
        p5_name = RUNNER.PREFIX + "P5_S1_J8"
        p3 = RUNNER.read_yaml(RUNNER.config_path(p3_name))
        p5 = RUNNER.read_yaml(RUNNER.config_path(p5_name))
        expected = copy.deepcopy(p3)
        expected["exp32_3maps_experiment"] = p5_name
        expected["alpha_loc_perception_schedule_values"] = [0.0, 0.025]
        self.assertEqual(p5, expected)
        self.assertEqual(p5["alpha_loc_perception_schedule_steps"], [0, 120])
        self.assertEqual(p5["alpha_loc_perception_loss"], 0.0)
        self.assertEqual(p5["alpha_loc_loss"], 8.0)
        self.assertTrue(p5["loc_perception_use_attention_weight"])
        self.assertFalse(p5["is_loc_aux_loss"])
        self.assertFalse(p5["is_aux_loc_combined_em_unc_loss"])
        self.assertEqual(p5["finetune_init_ckpt_path"], RUNNER.STAGE2_SOURCE)
        self.assertNotIn(p3_name, p5["finetune_init_ckpt_path"])

        args = self.args("train", p5_name)
        config = RUNNER.resolve_config(args, for_training=True)
        RUNNER.validate_config(config)
        commands, metadata, output = RUNNER.training_plan(args, config)
        self.assertEqual(metadata["runtime"]["max_steps"], 1800)
        self.assertEqual(metadata["runtime"]["save_steps"], 600)
        self.assertEqual(metadata["runtime"]["save_total_limit"], 6)
        self.assertEqual(metadata["runtime"]["optimizer"], "adamw_torch")
        self.assertIsNone(metadata["resume_checkpoint"])
        self.assertIsNone(metadata["source_checkpoint"])
        self.assertNotIn("--resume_from_checkpoint", commands[0])
        self.assertFalse(output.exists())

        for key, value in (("alpha_loc_perception_schedule_values", [0.0, 0.05]),
                           ("alpha_loc_perception_schedule_steps", [0, 180]),
                           ("loc_perception_use_attention_weight", False),
                           ("is_loc_aux_loss", True)):
            with self.subTest(key=key):
                invalid = copy.deepcopy(config)
                invalid[key] = value
                with self.assertRaisesRegex(ValueError, "single-factor"):
                    RUNNER.validate_config(invalid)

    def test_new_stage2_evaluation_steps_and_existing_task_routing(self):
        for suffix in RUNNER.STAGE2_NEW_ARMS:
            experiment = RUNNER.PREFIX + suffix
            for mode in ("minimal", "full"):
                with self.subTest(experiment=experiment, mode=mode):
                    extra = ("--asset-mode", "full", "--data-dir", "/dataset/full") if mode == "full" else ()
                    config = RUNNER.resolve_config(self.args("train", experiment, *extra), for_training=True)
                    RUNNER.validate_config(config)
                    for split, step, infer_count, metric_count in (
                        ("validation", 1200, 2, 4), ("validation", 1800, 2, 4),
                        ("test", 1800, 3, 8),
                    ):
                        for command, count in (("infer", infer_count), ("metrics", metric_count)):
                            args = self.args(command, experiment, "--split", split,
                                             "--checkpoint-step", str(step))
                            commands, path, evaluated = RUNNER.evaluation_plan(args, config)
                            self.assertEqual(len(commands), count)
                            self.assertIn(f"/{experiment}/checkpoint-{step}/{split}/", str(path))
                            self.assertEqual(evaluated["ckpt_path"],
                                             f"outputs/csgo_1b/{experiment}/checkpoint-{step}/model.safetensors")
                            self.assertFalse(evaluated["is_loc_aux_loss"])
                            self.assertFalse(evaluated["is_loc_perception_loss"])
                            if command == "infer":
                                self.assertEqual(commands[0][1], "eval_csgo_loc.py")
                                self.assertTrue(all(item[1] == "eval_csgo.py" for item in commands[1:]))
                                expected_splits = (["seen_validation", "seen_validation"] if split == "validation"
                                                   else ["seen_discrete_test", "seen_discrete_test", "seen_continuous"])
                                self.assertEqual([option(item, "--benchmark_v2_split") for item in commands],
                                                 expected_splits)
                            else:
                                metric_scripts = [item[1] for item in commands]
                                self.assertEqual(metric_scripts.count("benchmark_csgo_v1.py"), 3)
                                self.assertEqual(metric_scripts.count("benchmark_csgo_v1_conti.py"),
                                                 3 if split == "test" else 0)
                                if mode == "full":
                                    self.assertEqual(option(commands[0], "--gt"),
                                                     "/dataset/full/de_ancient/imgs")
                                    self.assertNotIn("--benchmark_v2_asset_manifest", commands[0])
                                else:
                                    self.assertIn("--benchmark_v2_asset_manifest", commands[0])
                    for split, step in (("validation", 600), ("test", 1200), ("test", 600)):
                        with self.assertRaisesRegex(ValueError, "requires checkpoint step"):
                            RUNNER.evaluation_plan(
                                self.args("infer", experiment, "--split", split,
                                          "--checkpoint-step", str(step)), config)

    def test_new_stage2_resume_keeps_own_recipe_and_legacy_save_interval(self):
        experiment = RUNNER.PREFIX + "P5_S1_J8"
        first = self.args("train", experiment, "--stop-after-step", "1200")
        config = RUNNER.resolve_config(first, for_training=True)
        _, metadata, output = RUNNER.training_plan(first, config)
        output.mkdir(parents=True)
        metadata["source_checkpoint"] = {"path": RUNNER.STAGE2_SOURCE, "sha256": "fixture"}
        (output / "experiment_plan.json").write_text(json.dumps(metadata))
        (output / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        checkpoint = output / "checkpoint-600"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 600, "max_steps": 1800}))
        for name in ("training_args.bin", "model.safetensors", "rng_state_0.pth",
                     "optimizer.pt", "scheduler.pt"):
            (checkpoint / name).touch()
        resume = self.args("train", experiment, "--resume", "--stop-after-step", "1800")
        resumed = RUNNER.resolve_config(resume, for_training=True)
        commands, result, _ = RUNNER.training_plan(resume, resumed)
        self.assertEqual(result["fingerprint"], metadata["fingerprint"])
        self.assertEqual(result["resume_checkpoint"], RUNNER.relative(checkpoint))
        self.assertEqual(result["source_checkpoint"], metadata["source_checkpoint"])
        self.assertEqual(option(commands[0], "--save_steps"), "600")
        self.assertNotIn("--resume_from_checkpoint", commands[0])
        with self.assertRaisesRegex(ValueError, "later stop"):
            RUNNER.training_plan(self.args("train", experiment, "--resume", "--stop-after-step", "600"),
                                 RUNNER.resolve_config(self.args("train", experiment, "--resume",
                                                                 "--stop-after-step", "600"), for_training=True))
        legacy = RUNNER.PREFIX + "stage2_perception_j8"
        legacy_args = self.args("train", legacy)
        legacy_config = RUNNER.resolve_config(legacy_args, for_training=True)
        legacy_commands, legacy_metadata, _ = RUNNER.training_plan(legacy_args, legacy_config)
        self.assertEqual(legacy_metadata["runtime"]["save_steps"], 300)
        self.assertEqual(option(legacy_commands[0], "--save_steps"), "300")

    def test_new_stage2_dry_runs_write_nothing_and_spawn_nothing(self):
        for suffix in RUNNER.STAGE2_NEW_ARMS:
            experiment = RUNNER.PREFIX + suffix
            for command, split, step in (("train", None, None), ("infer", "test", 1800),
                                         ("metrics", "validation", 1200)):
                extra = [] if command == "train" else ["--split", split, "--checkpoint-step", str(step)]
                with self.subTest(experiment=experiment, command=command), patch.object(
                    RUNNER.subprocess, "run"
                ) as run, patch("sys.stdout", io.StringIO()):
                    self.assertEqual(RUNNER.main([command, "--experiment", experiment, *extra]), 0)
                    run.assert_not_called()
        self.assertFalse((self.root / "outputs").exists())
        self.assertFalse((self.root / "outputs_eval").exists())


if __name__ == "__main__":
    unittest.main()
