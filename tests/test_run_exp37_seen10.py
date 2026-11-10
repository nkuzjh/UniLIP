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
from csgo_datasets.inference_batching import execution_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_exp37_seen10", REPO_ROOT / "scripts/run_exp37_seen10.py"
)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

EXPERIMENTS = (
    "exp37", "exp39", "exp39_consloc12", "exp37_s2_gen", "exp37_s2_loc", "exp37_s2", "exp37_s2_both",
    "exp37_s2_both_b1",
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
        if experiment in RUNNER.STAGE1:
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
                                 RUNNER.EXP39_ALPHA[experiment] if experiment in RUNNER.EXP39_EXPERIMENTS else
                                 8.0 if experiment in ("exp37", *RUNNER.JOINT_STAGE2) else 1.0)
                commands, metadata, out = RUNNER.training_plan(
                    self.args("train", experiment), config
                )
                command = commands[0]
                self.assertFalse(out.exists())
                self.assertEqual(option(command, "--num_train_epochs"), "50")
                self.assertEqual(option(command, "--save_only_model"), "False")
                self.assertEqual(option(command, "--save_total_limit"),
                                 "3" if experiment in RUNNER.EXP39_EXPERIMENTS else "6")
                self.assertEqual(metadata["runtime"]["gradient_accumulation_steps"], 32)
                if experiment == "exp37":
                    self.assertNotIn("--max_steps", command)
                    self.assertEqual(option(command, "--save_steps"), "4000")
                    self.assertEqual(config["warmup_ratio"], 0.003)
                    self.assertEqual(config["lr_scheduler_kwargs"], {"min_lr": 1e-5})
                    self.assertNotIn("training_stop_after_step", config)
                elif experiment in RUNNER.EXP39_EXPERIMENTS:
                    self.assertEqual(option(command, "--max_steps"), "19550")
                    self.assertEqual(option(command, "--save_strategy"), "no")
                    self.assertNotIn("--save_steps", command)
                    self.assertNotIn("--warmup_steps", command)
                    self.assertEqual(metadata["runtime"]["checkpoint_steps"], [12000, 16000, 19550])
                    self.assertEqual(config["training_checkpoint_steps"], [12000, 16000, 19550])
                    self.assertIs(config["save_final_model_at_output_root"], False)
                    self.assertEqual(config["warmup_ratio"], 0.003)
                    self.assertEqual(config["lr_scheduler_kwargs"], {"min_lr": 1e-5})
                    self.assertNotIn("training_stop_after_step", config)
                else:
                    self.assertEqual(option(command, "--max_steps"), "6000")
                    self.assertEqual(option(command, "--warmup_steps"), "200")
                    # Existing Stage 2 runner saves every 1000 steps; evaluation batching does not change training.
                    self.assertEqual(option(command, "--save_steps"), "1000")
                    self.assertEqual(config["lr_scheduler_kwargs"], {"min_lr_rate": 0.1})
                    self.assertEqual(config["learning_rate"], 2e-5)
                    self.assertEqual(config["action_dit_connector_lr"], 1e-4)

    def exp39_resume_fixture(self, step: int = 12000, experiment: str = "exp39") -> Path:
        args = self.args("train", experiment, "--stop-after-step", str(step))
        config = RUNNER.resolve_config(args, for_training=True)
        _, plan, out = RUNNER.training_plan(args, config)
        out.mkdir(parents=True)
        (out / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        (out / "experiment_plan.json").write_text(json.dumps(plan))
        checkpoint = out / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(json.dumps({
            "global_step": step, "max_steps": 19550, "num_train_epochs": 50,
            "epoch": 50.0 * step / 19550,
        }))
        for name in ("training_args.bin", "model.safetensors", "optimizer.pt",
                     "scheduler.pt", "rng_state.pth"):
            (checkpoint / name).touch()
        return checkpoint

    def test_exp39_only_changes_loc_alpha_and_checkpoint_controls(self):
        base = yaml.safe_load((self.root / "csgo_configs/exp37.yaml").read_text())
        new = yaml.safe_load((self.root / "csgo_configs/exp39.yaml").read_text())
        differences = {key for key in base.keys() | new.keys() if base.get(key) != new.get(key)}
        self.assertEqual(differences, {
            "exp37_experiment", "alpha_loc_loss", "training_checkpoint_steps",
            "save_final_model_at_output_root",
        })
        self.assertEqual(new["alpha_loc_loss"], 4.0)
        self.assertEqual(base["alpha_loc_loss"], 8.0)
        for key in ("finetune_init_ckpt_path", "resume_ckpt_path", "base_init_ckpt_path"):
            self.assertNotIn(key, new)
        for corrupt in (
            {"training_checkpoint_steps": [12000, 19550]},
            {"save_final_model_at_output_root": True},
            {"alpha_loc_loss": 8.0},
            {"warmup_ratio": 0.004},
            {"finetune_init_ckpt_path": "outputs/csgo_1b/exp37/checkpoint-19550/model.safetensors"},
        ):
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                RUNNER.validate_config({**self.config("exp39"), **corrupt})

    def test_consloc12_recipe_and_stage1_runtime(self):
        base = self.config("exp39")
        variant = self.config("exp39_consloc12")
        differences = {key for key in base.keys() | variant.keys()
                       if base.get(key) != variant.get(key)}
        self.assertEqual(differences, {"exp37_experiment", "alpha_loc_loss"})
        self.assertEqual(variant["alpha_loc_loss"], 12.0)
        self.assertEqual(variant["exp37_experiment"], "exp39_consloc12")
        for key in ("finetune_init_ckpt_path", "resume_ckpt_path", "base_init_ckpt_path",
                    "gen_init_ckpt_path", "loc_init_ckpt_path"):
            self.assertFalse(variant.get(key))
        for stop in (12000, 16000, 19550):
            with self.subTest(stop=stop):
                args = self.args("train", "exp39_consloc12", "--stop-after-step", str(stop))
                config = RUNNER.resolve_config(args, for_training=True)
                RUNNER.validate_config(config)
                commands, plan, out = RUNNER.training_plan(args, config)
                self.assertEqual(out.name, "exp39_consloc12")
                self.assertEqual(option(commands[0], "--max_steps"), "19550")
                self.assertEqual(option(commands[0], "--save_strategy"), "no")
                self.assertEqual(plan["runtime"]["checkpoint_steps"], [12000, 16000, 19550])
                self.assertEqual(plan["runtime"]["save_total_limit"], 3)
                self.assertIsNone(plan["source_checkpoint"])
        for bad in ({"alpha_loc_loss": 4.0},
                    {"finetune_init_ckpt_path": "outputs/csgo_1b/exp39/checkpoint-19550/model.safetensors"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                RUNNER.validate_config({**variant, **bad})

    def test_consloc12_resume_and_evaluation_stay_in_own_arm(self):
        experiment = "exp39_consloc12"
        checkpoint = self.exp39_resume_fixture(12000, experiment)
        args = self.args("train", experiment, "--resume")
        config = RUNNER.resolve_config(args, for_training=True)
        self.assertNotIn("training_stop_after_step", config)
        commands, plan, _ = RUNNER.training_plan(args, config)
        self.assertEqual(option(commands[0], "--max_steps"), "19550")
        self.assertEqual(plan["resume_checkpoint"], RUNNER.relative(checkpoint))
        self.assertIsNone(plan["source_checkpoint"])
        for split, count in (("test", 3), ("validation", 2)):
            with self.subTest(split=split):
                infer = self.args("infer", experiment, "--checkpoint-step", "12000",
                                  "--split", split, "--execute")
                routes, path, evaluated = RUNNER.evaluation_plan(
                    infer, RUNNER.resolve_config(infer, for_training=False))
                self.assertEqual(len(routes), count)
                self.assertIn(f"{experiment}/checkpoint-12000", str(path))
                self.assertEqual(evaluated["alpha_loc_loss"], 12.0)
        saved = checkpoint.parent / "resolved_config.yaml"
        saved.write_text(yaml.safe_dump(self.config("exp39")))
        with self.assertRaisesRegex(ValueError, "belongs to another exp37 arm"):
            RUNNER.resolve_config(args, for_training=True)
        saved.write_text(yaml.safe_dump(config))
        plan_path = checkpoint.parent / "experiment_plan.json"
        original = json.loads(plan_path.read_text())
        plan_path.write_text(json.dumps({**original, "experiment": "exp39"}))
        with self.assertRaisesRegex(ValueError, "identity"):
            RUNNER.training_plan(args, config)
        plan_path.write_text(json.dumps(original))
        checkpoint.rename(checkpoint.parent / "checkpoint-19550")
        with self.assertRaises(ValueError):
            RUNNER.training_plan(args, config)
        shutil.rmtree(checkpoint.parent)
        self.exp39_resume_fixture(19550, experiment)
        self.assertEqual(RUNNER.final_checkpoint_step(experiment), 19550)
        final = self.args("infer", experiment, "--checkpoint-step", "final", "--split", "test", "--execute")
        routes, path, _ = RUNNER.evaluation_plan(final, RUNNER.resolve_config(final, for_training=False))
        self.assertEqual(len(routes), 3)
        self.assertIn(f"{experiment}/checkpoint-19550", str(path))

    def test_exp39_micro_batches_and_full_assets_keep_base_batch(self):
        for experiment in RUNNER.EXP39_EXPERIMENTS:
            for micro, devices, gas in ((64, "0", 2), (32, "0", 4), (16, "0", 8), (8, "0", 16),
                                        (8, "0,1,2,3", 4)):
                with self.subTest(experiment=experiment, micro=micro, devices=devices):
                    args = self.args("train", experiment, "--micro-batch", str(micro),
                                 "--cuda-devices", devices, "--asset-mode", "full",
                                 "--data-dir", "/data/seen10")
                    config = RUNNER.resolve_config(args, for_training=True)
                    commands, plan, _ = RUNNER.training_plan(args, config)
                    self.assertIsNone(config["benchmark_v2_asset_manifest"])
                    self.assertEqual(config["data_dir"], "/data/seen10")
                    self.assertEqual(plan["runtime"]["source_batch"], 128)
                    self.assertEqual(plan["runtime"]["gradient_accumulation_steps"], gas)
                    self.assertEqual(option(commands[0], "--gradient_accumulation_steps"), str(gas))

    def test_exp39_execute_uses_stage1_path_without_source_preflight(self):
        for experiment in RUNNER.EXP39_EXPERIMENTS:
            with self.subTest(experiment=experiment):
                with patch.object(RUNNER, "preflight_stage2_source") as preflight, \
                     patch.object(RUNNER.subprocess, "run") as run, \
                     patch("sys.stdout", io.StringIO()):
                    self.assertEqual(RUNNER.main([
                        "train", "--experiment", experiment, "--execute", "--stop-after-step", "0",
                    ]), 0)
                    preflight.assert_not_called()
                    run.assert_called_once()
                out = RUNNER.output_dir(experiment)
                plan = json.loads((out / "experiment_plan.json").read_text())
                self.assertIsNone(plan["source_checkpoint"])
                self.assertEqual(plan["runtime"]["max_steps"], 19550)
                self.assertFalse((out / "model.safetensors").exists())

    def test_exp39_stops_and_resumes_full_fixed_horizon(self):
        for step in (12000, 16000):
            with self.subTest(step=step):
                checkpoint = self.exp39_resume_fixture(step)
                for suffix in ((), ("--stop-after-step", "0")):
                    args = self.args("train", "exp39", "--resume", *suffix)
                    config = RUNNER.resolve_config(args, for_training=True)
                    self.assertNotIn("training_stop_after_step", config)
                    command, plan, _ = RUNNER.training_plan(args, config)
                    self.assertEqual(option(command[0], "--max_steps"), "19550")
                    self.assertEqual(plan["resume_checkpoint"], RUNNER.relative(checkpoint))
                next_step = 16000 if step == 12000 else 19550
                args = self.args("train", "exp39", "--resume", "--stop-after-step", str(next_step))
                config = RUNNER.resolve_config(args, for_training=True)
                self.assertEqual(config["training_stop_after_step"], next_step)
                RUNNER.training_plan(args, config)
                for invalid in (1, 11999, 12001, 20000):
                    with self.assertRaises(ValueError):
                        RUNNER.validate_config(self.config("exp39", "--stop-after-step", str(invalid)))
                shutil.rmtree(checkpoint.parent)

    def test_exp39_resume_rejects_wrong_horizon_extra_checkpoint_and_incomplete_state(self):
        checkpoint = self.exp39_resume_fixture()
        args = self.args("train", "exp39", "--resume")
        config = RUNNER.resolve_config(args, for_training=True)
        state_path = checkpoint / "trainer_state.json"
        original = json.loads(state_path.read_text())
        state_path.write_text(json.dumps({**original, "max_steps": 12000}))
        with self.assertRaisesRegex(ValueError, "horizon"):
            RUNNER.training_plan(args, config)
        state_path.write_text(json.dumps(original))
        (checkpoint / "rng_state.pth").unlink()
        with self.assertRaisesRegex(ValueError, "RNG"):
            RUNNER.training_plan(args, config)
        (checkpoint / "rng_state.pth").touch()
        extra = checkpoint.parent / "checkpoint-14000"
        extra.mkdir()
        with self.assertRaisesRegex(ValueError, "Unexpected exp39 checkpoint"):
            RUNNER.training_plan(args, config)
        extra.rmdir()
        plan_path = checkpoint.parent / "experiment_plan.json"
        plan = json.loads(plan_path.read_text())
        plan_path.write_text(json.dumps({**plan, "experiment": "exp37"}))
        with self.assertRaisesRegex(ValueError, "identity"):
            RUNNER.training_plan(args, config)
        plan_path.write_text(json.dumps(plan))
        checkpoint.rename(checkpoint.parent / "checkpoint-19550")
        with self.assertRaises(ValueError):
            RUNNER.training_plan(args, config)

    def test_exp39_final_requires_exact_19550_completed_state(self):
        checkpoint = self.exp39_resume_fixture(19550)
        state_path = checkpoint / "trainer_state.json"
        state = json.loads(state_path.read_text())
        (checkpoint / "model.safetensors").unlink()
        with self.assertRaises(ValueError):
            RUNNER.final_checkpoint_step("exp39")
        (checkpoint / "model.safetensors").touch()
        self.assertEqual(RUNNER.final_checkpoint_step("exp39"), 19550)
        state["epoch"] = 49.9
        state_path.write_text(json.dumps(state))
        self.assertEqual(RUNNER.final_checkpoint_step("exp39"), 19550)
        state["max_steps"] = 20000
        state_path.write_text(json.dumps(state))
        with self.assertRaises(ValueError):
            RUNNER.final_checkpoint_step("exp39")

    def test_exp39_12000_and_final_evaluation_nodes(self):
        checkpoint = self.exp39_resume_fixture(12000)
        for split, count in (("test", 3), ("validation", 2)):
            args = self.args("infer", "exp39", "--checkpoint-step", "12000",
                             "--split", split, "--execute")
            commands, _, _ = RUNNER.evaluation_plan(
                args, RUNNER.resolve_config(args, for_training=False))
            self.assertEqual(len(commands), count)
        shutil.rmtree(checkpoint.parent)
        self.exp39_resume_fixture(19550)
        for split, count in (("test", 3), ("validation", 2)):
            args = self.args("infer", "exp39", "--checkpoint-step", "final",
                             "--split", split, "--execute")
            commands, path, _ = RUNNER.evaluation_plan(
                args, RUNNER.resolve_config(args, for_training=False))
            self.assertEqual(len(commands), count)
            self.assertIn("checkpoint-19550", str(path))

    def test_stage1_16000_evaluation_covers_all_test_tasks_before_final(self):
        for experiment in RUNNER.STAGE1:
            with self.subTest(experiment=experiment):
                if experiment in RUNNER.EXP39_EXPERIMENTS:
                    checkpoint = self.exp39_resume_fixture(16000, experiment)
                else:
                    config = self.config("exp37")
                    args = self.args("train", "exp37")
                    _, plan, out = RUNNER.training_plan(args, config)
                    out.mkdir(parents=True)
                    (out / "resolved_config.yaml").write_text(yaml.safe_dump(config))
                    (out / "experiment_plan.json").write_text(json.dumps(plan))
                    checkpoint = out / "checkpoint-16000"
                    checkpoint.mkdir()
                    (checkpoint / "trainer_state.json").write_text(json.dumps({
                        "global_step": 16000, "max_steps": 19550,
                    }))
                    (checkpoint / "model.safetensors").touch()
                for split, count in (("test", 3), ("validation", 2)):
                    args = self.args("infer", experiment, "--checkpoint-step", "16000",
                                     "--split", split, "--execute")
                    config = RUNNER.resolve_config(args, for_training=False)
                    commands, path, _ = RUNNER.evaluation_plan(args, config)
                    self.assertEqual(len(commands), count)
                    self.assertIn("checkpoint-16000", str(path))
                shutil.rmtree(checkpoint.parent)

    def test_single_task_and_combined_objectives_remain_scoped(self):
        gen = self.config("exp37_s2_gen")
        loc = self.config("exp37_s2_loc")
        both = self.config("exp37_s2_both")
        b1 = self.config("exp37_s2_both_b1")
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
        for config in (control, both, b1):
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
        differences = {key for key in both.keys() | b1.keys()
                       if both.get(key) != b1.get(key)}
        self.assertEqual(differences, {"exp37_experiment", "loc_perception_use_attention_weight"})
        self.assertTrue(b1["loc_perception_use_attention_weight"])
        self.assertEqual(b1["alpha_loc_perception_schedule_values"], [0.0, 0.05])
        self.assertEqual(b1["alpha_loc_aux_schedule_values"], [0.0, 1.0])
        self.assertEqual(b1["aux_loc_combined_candidate_tau"], 1.0)

    def test_final_selects_completed_state_and_joint_source_is_dynamic(self):
        checkpoint = self.completed_checkpoint("exp37", 19337)
        later = checkpoint.parent / "checkpoint-20000"
        later.mkdir()
        (later / "trainer_state.json").write_text(
            json.dumps({"global_step": 20000, "max_steps": 22000})
        )
        (later / "model.safetensors").touch()
        self.assertEqual(RUNNER.final_checkpoint_step("exp37"), 19337)
        for experiment in RUNNER.JOINT_STAGE2:
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
                step = "8000" if experiment == "exp37" else "12000" if experiment in RUNNER.EXP39_EXPERIMENTS else "4000"
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
                   "--split", "validation", "--test-loc-batch-size", "1",
                   "--test-gen-batch-size", "1"]
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
            ("exp37_s2_both_b1", ["eval_csgo_loc.py", "eval_csgo.py", "eval_csgo.py"]),
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

    def test_independent_test_generation_routes_and_snapshots(self):
        config = self.config("exp37_s2", for_training=False)
        for task, kind, split, benchmark in (
            ("gen", "discrete", "seen_discrete_test", "benchmark_csgo_v1.py"),
            ("gen_continuous", "continuous", "seen_continuous", "benchmark_csgo_v1_conti.py"),
        ):
            with self.subTest(task=task):
                base = ("--checkpoint-step", "6000", "--split", "test", "--task", task)
                infer_args = self.args("infer", "exp37_s2", *base)
                infer, legacy_path, evaluated = RUNNER.evaluation_plan(infer_args, config)
                self.assertEqual(len(infer), 1)
                self.assertEqual(infer[0][1], "eval_csgo.py")
                self.assertEqual(option(infer[0], "--benchmark_v2_split"), split)
                self.assertEqual(option(infer[0], "--test-gen-batch-size"), "16")
                self.assertEqual(Path(option(infer[0], "--output_dir")).name, kind)
                path = RUNNER.path_in_repo(option(infer[0], "--csgo_config"))
                self.assertEqual(path.parent.name, kind)
                snapshots = RUNNER.evaluation_snapshots(infer_args, legacy_path, evaluated)
                self.assertEqual(set(snapshots), {path})
                self.assertEqual(snapshots[path]["benchmark_v2_split"], split)
                self.assertEqual(snapshots[path]["test_gen_batch_size"], 16)
                self.assertNotIn("test_loc_batch_size", snapshots[path])

                metrics_args = self.args("metrics", "exp37_s2", *base)
                metrics, metrics_path, metrics_config = RUNNER.evaluation_plan(metrics_args, config)
                self.assertEqual(len(metrics), 11)
                self.assertTrue(all(command[1] == benchmark for command in metrics[:10]))
                self.assertTrue(all(option(command, "--benchmark_v2_split") == split
                                    for command in metrics[:10]))
                summary = metrics[-1]
                self.assertEqual(option(summary, "--split"), split)
                self.assertEqual(option(summary, "--kind"), kind)
                self.assertEqual(option(summary, "--input_root"), option(infer[0], "--output_dir"))
                self.assertEqual(set(RUNNER.evaluation_snapshots(metrics_args, metrics_path,
                                                                 metrics_config)), {path})

        old_args = self.args("infer", "exp37_s2", "--checkpoint-step", "6000",
                             "--split", "test", "--task", "generation")
        old_commands, old_path, old_config = RUNNER.evaluation_plan(old_args, config)
        self.assertEqual([Path(option(command, "--output_dir")).name for command in old_commands],
                         ["discrete", "continuous"])
        self.assertEqual(len(RUNNER.evaluation_snapshots(old_args, old_path, old_config)), 2)

        for task in ("gen", "gen_continuous"):
            legacy = self.args("infer", "exp37_s2", "--checkpoint-step", "6000",
                               "--split", "test", "--task", task,
                               "--test-gen-batch-size", "1")
            commands, path, evaluated = RUNNER.evaluation_plan(legacy, config)
            self.assertEqual(option(commands[0], "--csgo_config"), RUNNER.relative(path))
            self.assertEqual(RUNNER.evaluation_snapshots(legacy, path, evaluated), {path: evaluated})

        invalid = self.args("infer", "exp37_s2", "--checkpoint-step", "4000",
                            "--split", "validation", "--task", "gen_continuous")
        with self.assertRaisesRegex(ValueError, "requires --split test"):
            RUNNER.evaluation_plan(invalid, config)
        for task in ("gen", "gen_continuous"):
            wrong = self.args("infer", "exp37_s2_loc", "--checkpoint-step", "6000",
                              "--split", "test", "--task", task)
            with self.assertRaisesRegex(ValueError, "inactive"):
                RUNNER.evaluation_plan(wrong, self.config("exp37_s2_loc", for_training=False))

    def test_independent_metrics_execute_checks_only_selected_batch_snapshot(self):
        self.completed_checkpoint("exp37", 19500)
        for task, kind, split in (("gen", "discrete", "seen_discrete_test"),
                                  ("gen_continuous", "continuous", "seen_continuous")):
            with self.subTest(task=task):
                base = ["--experiment", "exp37", "--checkpoint-step", "final",
                        "--split", "test", "--task", task, "--execute"]
                with patch.object(RUNNER.subprocess, "run") as run, patch("sys.stdout", io.StringIO()):
                    self.assertEqual(RUNNER.main(["infer", *base]), 0)
                    run.assert_called_once()
                output = self.root / f"outputs_eval/benchmark_v2/exp37/checkpoint-19500/test/seed_42/infer_bs16/{kind}"
                config_path = output / "evaluation_config.yaml"
                self.assertEqual(RUNNER.read_yaml(config_path)["benchmark_v2_split"], split)
                (output / "inference_manifest.json").write_text(json.dumps({
                    "config_path": RUNNER.relative(config_path),
                    "inference_execution": execution_contract("gen", 16),
                }))
                with patch.object(RUNNER.subprocess, "run") as run, patch("sys.stdout", io.StringIO()):
                    self.assertEqual(RUNNER.main(["metrics", *base]), 0)
                    self.assertEqual(run.call_count, 11)

    def test_metrics_cpu_changes_only_generation_metric_device(self):
        base = ("--checkpoint-step", "6000", "--split", "test", "--task", "gen",
                "--seed", "42", "--test-gen-batch-size", "8")
        config = self.config("exp37_s2_gen", for_training=False)
        default = self.args("metrics", "exp37_s2_gen", *base)
        explicit_cuda = self.args("metrics", "exp37_s2_gen", *base,
                                  "--metrics-device", "cuda")
        cpu = self.args("metrics", "exp37_s2_gen", *base,
                        "--metrics-device", "cpu")
        default_commands, default_path, default_config = RUNNER.evaluation_plan(default, config)
        cuda_commands, cuda_path, cuda_config = RUNNER.evaluation_plan(explicit_cuda, config)
        cpu_commands, cpu_path, cpu_config = RUNNER.evaluation_plan(cpu, config)
        self.assertEqual(default_commands, cuda_commands)
        self.assertEqual(default_path, cuda_path)
        self.assertEqual(default_config, cuda_config)
        self.assertEqual(default_path, cpu_path)
        self.assertEqual(default_config, cpu_config)
        self.assertEqual(len(cpu_commands), 11)
        self.assertEqual([command[1] for command in cpu_commands[:10]],
                         ["benchmark_csgo_v1.py"] * 10)
        self.assertEqual([option(command, "--map_name") for command in cpu_commands[:10]],
                         SEEN_MAPS)
        self.assertEqual(cpu_commands[-1], default_commands[-1])
        self.assertEqual(option(cpu_commands[-1], "--kind"), "discrete")
        self.assertEqual(option(cpu_commands[-1], "--input_root"),
                         "outputs_eval/benchmark_v2/exp37_s2_gen/checkpoint-6000/test/seed_42/infer_bs8/discrete")
        for default_command, cpu_command in zip(default_commands[:10], cpu_commands[:10]):
            self.assertEqual(option(default_command, "--device"), "cuda")
            self.assertEqual(option(cpu_command, "--device"), "cpu")
            self.assertEqual(cpu_command,
                             ["cpu" if arg == "cuda" and index == default_command.index("--device") + 1
                              else arg for index, arg in enumerate(default_command)])

    def test_lora_dropout_off_routes_to_separate_config_infer_and_metrics(self):
        base = ("--checkpoint-step", "6000", "--split", "test", "--task", "gen",
                "--test-gen-batch-size", "8", "--seed", "42")
        config = self.config("exp37_s2_both", for_training=False)
        on = self.args("infer", "exp37_s2_both", *base)
        off = self.args("infer", "exp37_s2_both", *base, "--disable-lora-dropout")
        on_commands, _, on_config = RUNNER.evaluation_plan(on, config)
        off_commands, legacy_path, off_config = RUNNER.evaluation_plan(off, config)
        on_output = option(on_commands[0], "--output_dir")
        off_output = option(off_commands[0], "--output_dir")
        self.assertEqual(off_output, on_output.replace("/infer_bs8/discrete",
                                                        "/infer_bs8/lora_dropout_off/discrete"))
        self.assertEqual(on_output, "outputs_eval/benchmark_v2/exp37_s2_both/checkpoint-6000/test/seed_42/infer_bs8/discrete")
        self.assertNotIn("--disable-lora-dropout", on_commands[0])
        self.assertIn("--disable-lora-dropout", off_commands[0])
        self.assertNotIn("disable_lora_dropout", on_config)
        self.assertIs(off_config["disable_lora_dropout"], True)
        snapshots = RUNNER.evaluation_snapshots(off, legacy_path, off_config)
        off_path = RUNNER.path_in_repo(option(off_commands[0], "--csgo_config"))
        self.assertEqual(set(snapshots), {off_path})
        self.assertEqual(snapshots[off_path], {**off_config, "test_gen_batch_size": 8})
        self.assertEqual(off_path.parent, self.root / off_output)
        self.assertNotIn(legacy_path, snapshots)

        metric_args = self.args("metrics", "exp37_s2_both", *base, "--disable-lora-dropout")
        metric_commands, metric_legacy, metric_config = RUNNER.evaluation_plan(metric_args, config)
        self.assertEqual(metric_legacy, legacy_path)
        self.assertEqual(RUNNER.evaluation_snapshots(metric_args, metric_legacy, metric_config), snapshots)
        self.assertTrue(all(off_output in option(cmd, "--pred") for cmd in metric_commands[:10]))
        self.assertEqual(option(metric_commands[-1], "--input_root"), off_output)
        self.assertTrue(all("--disable-lora-dropout" not in cmd for cmd in metric_commands))

    def test_lora_dropout_off_rejects_other_scope(self):
        base = ["--checkpoint-step", "6000", "--split", "test", "--task", "gen",
                "--test-gen-batch-size", "8", "--seed", "42"]
        changes = [
            ("--experiment", "exp37_s2_gen"),
            ("--checkpoint-step", "4000"),
            ("--split", "validation"),
            ("--task", "all"),
            ("--test-gen-batch-size", "16"),
            ("--seed", "43"),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                args = ["infer", "--experiment", "exp37_s2_both", *base,
                        key, value, "--disable-lora-dropout"]
                parsed = RUNNER.parser().parse_args(args)
                with self.assertRaisesRegex(ValueError, "requires exp37_s2_both"):
                    RUNNER.evaluation_plan(parsed, self.config("exp37_s2_both", for_training=False))
        for command in ("train", "validate"):
            with self.subTest(command=command), patch("sys.stderr", io.StringIO()) as stderr:
                self.assertEqual(RUNNER.main([command, "--experiment", "exp37_s2_both",
                                              "--disable-lora-dropout"]), 1)
                self.assertIn("applies only to infer/metrics", stderr.getvalue())

    def test_lora_dropout_off_metrics_preflight_checks_mode_and_active_modules(self):
        self.completed_checkpoint("exp37_s2_both", 6000)
        base = ["metrics", "--experiment", "exp37_s2_both", "--checkpoint-step", "6000",
                "--split", "test", "--task", "gen", "--test-gen-batch-size", "8",
                "--disable-lora-dropout"]
        args = RUNNER.parser().parse_args(base)
        config = RUNNER.resolve_config(args, for_training=False)
        _, path, evaluated = RUNNER.evaluation_plan(args, config)
        snapshot_path, snapshot = next(iter(RUNNER.evaluation_snapshots(args, path, evaluated).items()))
        snapshot_path.parent.mkdir(parents=True)
        snapshot_path.write_text(yaml.safe_dump(snapshot))
        manifest_path = snapshot_path.parent / "inference_manifest.json"
        payload = {"config_path": RUNNER.relative(snapshot_path),
                   "inference_execution": execution_contract("gen", 8)}
        disabled = [{"name": f"layer.{i}.lora_dropout.default", "p": 0.05}
                    for i in range(438)]
        for execution in (
            {},
            {"model_mode_policy": "preserve_loaded_module_modes_v1",
             "active_dropout_modules": [], "disabled_lora_dropout_modules": disabled},
            {"model_mode_policy": "disable_lora_dropout_v1",
             "active_dropout_modules": [{"name": "other", "p": 0.2}],
             "disabled_lora_dropout_modules": disabled},
            {"model_mode_policy": "disable_lora_dropout_v1",
             "active_dropout_modules": [], "disabled_lora_dropout_modules": disabled[:-1]},
        ):
            manifest_path.write_text(json.dumps({**payload, "inference_execution":
                                                 {**payload["inference_execution"], **execution}}))
            with patch.object(RUNNER.subprocess, "run") as run, \
                 patch("sys.stderr", io.StringIO()) as stderr:
                self.assertEqual(RUNNER.main([*base, "--execute"]), 1)
                self.assertIn("LoRA Dropout-off inference provenance", stderr.getvalue())
                run.assert_not_called()
        manifest_path.write_text(json.dumps({**payload, "inference_execution":
                                             {**payload["inference_execution"],
                                              "model_mode_policy": "disable_lora_dropout_v1",
                                              "active_dropout_modules": [],
                                              "disabled_lora_dropout_modules": disabled}}))
        with patch.object(RUNNER.subprocess, "run") as run, patch("sys.stdout", io.StringIO()):
            self.assertEqual(RUNNER.main([*base, "--execute"]), 0)
            self.assertEqual(run.call_count, 11)

    def test_metrics_cpu_dry_run_and_execute_hide_cuda_devices(self):
        base = ["metrics", "--experiment", "exp37_s2_gen", "--checkpoint-step", "6000",
                "--split", "test", "--task", "gen", "--seed", "42",
                "--test-gen-batch-size", "8", "--metrics-device", "cpu"]
        self.completed_checkpoint("exp37_s2_gen", 6000)
        with patch.object(RUNNER.subprocess, "run") as run, \
             patch("sys.stdout", io.StringIO()) as stdout:
            self.assertEqual(RUNNER.main(base), 0)
            run.assert_not_called()
            self.assertIn("CUDA_VISIBLE_DEVICES=''", stdout.getvalue())
            self.assertIn("--device cpu", stdout.getvalue())
            self.assertNotIn("benchmark_csgo_v1_conti.py", stdout.getvalue())
        self.assertFalse((self.root / "outputs_eval").exists())

        args = RUNNER.parser().parse_args(base)
        config = RUNNER.resolve_config(args, for_training=False)
        _, legacy_path, evaluated = RUNNER.evaluation_plan(args, config)
        snapshots = RUNNER.evaluation_snapshots(args, legacy_path, evaluated)
        self.assertEqual(len(snapshots), 1)
        snapshot_path, snapshot = next(iter(snapshots.items()))
        snapshot_path.parent.mkdir(parents=True)
        snapshot_path.write_text(yaml.safe_dump(snapshot))
        (snapshot_path.parent / "inference_manifest.json").write_text(json.dumps({
            "config_path": RUNNER.relative(snapshot_path),
            "inference_execution": execution_contract("gen", 8),
        }))
        with patch.object(RUNNER.subprocess, "run") as run, \
             patch("sys.stdout", io.StringIO()) as stdout:
            self.assertEqual(RUNNER.main([*base, "--execute"]), 0)
            self.assertEqual(run.call_count, 11)
            for call in run.call_args_list:
                self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
                self.assertEqual(call.kwargs["cwd"], self.root)
                self.assertTrue(call.kwargs["check"])
            self.assertIn("CUDA_VISIBLE_DEVICES=''", stdout.getvalue())

    def test_metrics_cpu_rejected_for_non_metrics_commands(self):
        for command in ("validate", "train", "infer"):
            with self.subTest(command=command), \
                 patch.object(RUNNER.subprocess, "run") as run, \
                 patch("sys.stderr", io.StringIO()) as stderr:
                self.assertEqual(RUNNER.main([command, "--experiment", "exp37_s2_gen",
                                              "--metrics-device", "cpu"]), 1)
                self.assertIn("applies only to metrics", stderr.getvalue())
                run.assert_not_called()

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

    def test_joint_arms_pin_same_completed_stage1_source_path_and_hash(self):
        checkpoint = self.completed_checkpoint("exp37", 19337)
        source_name = RUNNER.relative(checkpoint / "model.safetensors")
        joint = RUNNER.JOINT_STAGE2
        with patch.object(RUNNER, "tensor_header", return_value={
            "tensor_count": 2830, "lora_tensor_count": 1128,
            "shape_validation": "mocked header",
        }), patch.object(RUNNER, "digest_file", return_value="a" * 64):
            # Every arm can start first, and every existing peer must agree.
            for experiment in joint:
                config = self.config(experiment)
                with self.subTest(first=experiment):
                    provenance = RUNNER.preflight_stage2_source(config)
                    self.assertEqual(provenance["path"], source_name)
                    self.assertEqual(provenance["sha256"], "a" * 64)
                for peer in joint:
                    if peer == experiment:
                        continue
                    plan_path = RUNNER.output_dir(peer) / "experiment_plan.json"
                    plan_path.parent.mkdir(parents=True, exist_ok=True)
                    for case, pinned in (
                        ("missing", None),
                        ("path", {"path": source_name.replace("19337", "19336"), "sha256": "a" * 64}),
                        ("hash", {"path": source_name, "sha256": "b" * 64}),
                    ):
                        plan_path.write_text(json.dumps({"source_checkpoint": pinned}))
                        with self.subTest(experiment=experiment, peer=peer, case=case):
                            with self.assertRaisesRegex(ValueError, "sibling .* different Stage 1 source path/hash"):
                                RUNNER.preflight_stage2_source(config)
                    plan_path.write_text(json.dumps({"source_checkpoint": {
                        "path": source_name, "sha256": "a" * 64,
                    }}))
                    with self.subTest(experiment=experiment, peer=peer, case="matching"):
                        self.assertEqual(RUNNER.preflight_stage2_source(config)["path"], source_name)
                    plan_path.unlink()
                first_peer, second_peer = (peer for peer in joint if peer != experiment)
                first_plan = RUNNER.output_dir(first_peer) / "experiment_plan.json"
                second_plan = RUNNER.output_dir(second_peer) / "experiment_plan.json"
                first_plan.write_text(json.dumps({"source_checkpoint": {
                    "path": source_name, "sha256": "a" * 64,
                }}))
                for case, pinned in (
                    ("path", {"path": source_name.replace("19337", "19336"), "sha256": "a" * 64}),
                    ("hash", {"path": source_name, "sha256": "b" * 64}),
                ):
                    second_plan.write_text(json.dumps({"source_checkpoint": pinned}))
                    with self.subTest(experiment=experiment, first_peer=first_peer,
                                      second_peer=second_peer, case=case):
                        with self.assertRaisesRegex(ValueError, f"sibling {second_peer} .* source path/hash"):
                            RUNNER.preflight_stage2_source(config)
                first_plan.unlink()
                second_plan.unlink()
            (checkpoint / "model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "Final checkpoint is incomplete|Missing Stage 2 source"):
                RUNNER.preflight_stage2_source(self.config("exp37_s2_both_b1"))

    def test_b1_resume_uses_own_checkpoint_after_stage1_source_is_removed(self):
        source = self.completed_checkpoint("exp37", 19337) / "model.safetensors"
        checkpoint, metadata = self.managed_resume_checkpoint("exp37_s2_both_b1")
        self.assertIn("checkpoint-19337/model.safetensors", metadata["source_checkpoint"]["path"])
        (checkpoint / "rng_state.pth").touch()
        source.unlink()
        args = self.args("train", "exp37_s2_both_b1", "--resume")
        config = RUNNER.resolve_config(args, for_training=True)
        RUNNER.validate_config(config)
        commands, resumed, _ = RUNNER.training_plan(args, config)
        self.assertEqual(resumed["source_checkpoint"], metadata["source_checkpoint"])
        self.assertEqual(resumed["resume_checkpoint"], RUNNER.relative(checkpoint))
        self.assertNotIn("--resume_from_checkpoint", commands[0])
        self.assertFalse(RUNNER.output_dir("exp37_s2").exists())
        self.assertFalse(RUNNER.output_dir("exp37_s2_both").exists())

    def test_b1_evaluation_steps_and_recipe_are_frozen(self):
        experiment = "exp37_s2_both_b1"
        self.snapshot_recipe(experiment)
        config = self.config(experiment, for_training=False)
        for step in (4000, 6000):
            args = self.args("infer", experiment, "--checkpoint-step", str(step))
            commands, _, evaluated = RUNNER.evaluation_plan(args, config)
            self.assertEqual([command[1] for command in commands],
                             ["eval_csgo_loc.py", "eval_csgo.py"])
            self.assertTrue(evaluated["loc_perception_use_attention_weight"])
            self.assertFalse(evaluated["is_loc_aux_loss"])
            self.assertFalse(evaluated["is_loc_perception_loss"])
        with self.assertRaisesRegex(ValueError, "requires checkpoint"):
            RUNNER.evaluation_plan(
                self.args("infer", experiment, "--checkpoint-step", "4000", "--split", "test"), config)
        commands, _, _ = RUNNER.evaluation_plan(
            self.args("infer", experiment, "--checkpoint-step", "6000", "--split", "test"), config)
        self.assertEqual([command[1] for command in commands],
                         ["eval_csgo_loc.py", "eval_csgo.py", "eval_csgo.py"])
        metrics, _, _ = RUNNER.evaluation_plan(
            self.args("metrics", experiment, "--checkpoint-step", "6000", "--split", "test"), config)
        self.assertEqual(len(metrics), 22)
        changed = copy.deepcopy(config)
        changed["loc_perception_use_attention_weight"] = False
        with self.assertRaisesRegex(ValueError, "fixed field loc_perception_use_attention_weight"):
            RUNNER.validate_config(changed)

    def test_inference_batch_paths_snapshots_and_legacy_routing(self):
        config = self.config("exp37_s2", for_training=False)
        default = self.args("infer", "exp37_s2", "--checkpoint-step", "6000", "--split", "test")
        commands, legacy_path, evaluated = RUNNER.evaluation_plan(default, config)
        snapshots = RUNNER.evaluation_snapshots(default, legacy_path, evaluated)
        self.assertEqual(len(snapshots), 3)
        self.assertEqual({path.parent.name for path in snapshots}, {"infer_bs32", "discrete", "continuous"})
        for command in commands:
            path = RUNNER.path_in_repo(option(command, "--csgo_config"))
            self.assertIn(path, snapshots)
            self.assertEqual(RUNNER.path_in_repo(option(command, "--output_dir")), path.parent)
            if command[1] == "eval_csgo_loc.py":
                self.assertEqual(option(command, "--test-loc-batch-size"), "32")
                self.assertEqual(snapshots[path]["test_loc_batch_size"], 32)
                self.assertNotIn("test_gen_batch_size", snapshots[path])
            else:
                self.assertEqual(option(command, "--test-gen-batch-size"), "16")
                self.assertEqual(snapshots[path]["test_gen_batch_size"], 16)
                self.assertNotIn("test_loc_batch_size", snapshots[path])
                if path.parent.name == "continuous":
                    self.assertEqual(snapshots[path]["benchmark_v2_split"], "seen_continuous")
        mixed = self.args("infer", "exp37_s2", "--checkpoint-step", "6000", "--split", "test",
                          "--test-loc-batch-size", "7", "--test-gen-batch-size", "11")
        mixed_commands, _, _ = RUNNER.evaluation_plan(mixed, config)
        self.assertIn("infer_bs7", option(mixed_commands[0], "--output_dir"))
        self.assertTrue(all("infer_bs11" in option(cmd, "--output_dir") for cmd in mixed_commands[1:]))
        legacy = self.args("infer", "exp37_s2", "--checkpoint-step", "6000", "--split", "test",
                           "--test-loc-batch-size", "1", "--test-gen-batch-size", "1")
        old_commands, old_path, old_cfg = RUNNER.evaluation_plan(legacy, config)
        self.assertEqual(RUNNER.evaluation_snapshots(legacy, old_path, old_cfg), {old_path: old_cfg})
        self.assertTrue(all("infer_bs" not in option(cmd, "--output_dir") for cmd in old_commands))
        for flag in ("--test-loc-batch-size", "--test-gen-batch-size"):
            for value in ("0", "-2"):
                with self.assertRaises(SystemExit):
                    self.args("infer", "exp37_s2", flag, value)

    def test_new_metrics_reject_missing_or_incompatible_inference_metadata(self):
        self.completed_checkpoint("exp37", 8000)
        base = ["--experiment", "exp37", "--checkpoint-step", "8000", "--split", "validation",
                "--task", "generation", "--execute"]
        with patch.object(RUNNER.subprocess, "run") as run, patch("sys.stdout", io.StringIO()):
            self.assertEqual(RUNNER.main(["infer", *base]), 0)
            run.assert_called_once()
        output = self.root / "outputs_eval/benchmark_v2/exp37/checkpoint-8000/validation/seed_42/infer_bs16/discrete"
        self.assertEqual(RUNNER.read_yaml(output / "evaluation_config.yaml")["test_gen_batch_size"], 16)
        manifest = {"config_path": RUNNER.relative(output / "evaluation_config.yaml")}
        (output / "inference_manifest.json").write_text(json.dumps(manifest))
        with patch.object(RUNNER.subprocess, "run") as run, patch("sys.stderr", io.StringIO()) as error, \
             patch("sys.stdout", io.StringIO()):
            self.assertEqual(RUNNER.main(["metrics", *base]), 1)
            self.assertIn("Inference batch size", error.getvalue())
            run.assert_not_called()

    def test_batched_loc_metrics_require_completed_valid_summary(self):
        self.completed_checkpoint("exp37", 8000)
        manifest_file = self.root / "data/csgo_benchmark_v2/benchmark_manifest.json"
        shutil.copyfile(REPO_ROOT / "data/csgo_benchmark_v2/benchmark_manifest.json", manifest_file)
        base = ["--experiment", "exp37", "--checkpoint-step", "8000", "--split", "validation",
                "--task", "localization", "--execute"]
        with patch.object(RUNNER.subprocess, "run") as run, patch("sys.stdout", io.StringIO()):
            self.assertEqual(RUNNER.main(["infer", *base]), 0)
            run.assert_called_once()
        output = self.root / "outputs_loc/benchmark_v2/exp37/checkpoint-8000/validation/seed_42/infer_bs32"
        config_path = output / "evaluation_config.yaml"
        config = RUNNER.read_yaml(config_path)
        manifest = {"config_path": RUNNER.relative(config_path),
                    "benchmark_v2_manifest": str(manifest_file),
                    "benchmark_v2_split": "seen_validation", "maps": SEEN_MAPS,
                    "sample_count": 5000, "checkpoint": config["ckpt_path"],
                    "ckpt_path": config["ckpt_path"], "seed": 42,
                    "benchmark_v2_support_seed": None,
                    "benchmark_v2_shots_per_map": None,
                    "inference_execution": execution_contract("loc", 32)}
        manifest_path = output / "inference_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        summary_path = output / "benchmark_csgo_v2_loc.json"
        with patch.object(RUNNER.subprocess, "run") as run, patch("sys.stderr", io.StringIO()) as error, \
             patch("sys.stdout", io.StringIO()):
            self.assertEqual(RUNNER.main(["metrics", *base]), 1)
            self.assertIn("Missing localization summary", error.getvalue())
            run.assert_not_called()
        metrics = {"XY_Dist": 1.0, "Z_Dist": 2.0, "Pitch_Dist": 3.0, "Yaw_Dist": 4.0}
        summary = RUNNER.aggregates.build_localization_summary(
            manifest=manifest_file, split="seen_validation", maps=SEEN_MAPS,
            per_map={name: metrics for name in SEEN_MAPS}, metrics_macro_map=metrics,
            inference_provenance={"path": str(manifest_path), "payload": manifest},
            checkpoint=config["ckpt_path"], seed=42, support_seed=None,
            shots_per_map=None, sample_count=5000)
        summary_path.write_text(json.dumps(summary))
        with patch.object(RUNNER.subprocess, "run") as run, patch("sys.stdout", io.StringIO()):
            self.assertEqual(RUNNER.main(["metrics", *base]), 0)
            run.assert_not_called()
        for key, bad in (("checkpoint", "outputs/other/model.safetensors"),
                         ("split", "seen_discrete_test")):
            corrupted = dict(summary, **{key: bad})
            summary_path.write_text(json.dumps(corrupted))
            with self.subTest(key=key), patch.object(RUNNER.subprocess, "run") as run, \
                 patch("sys.stderr", io.StringIO()), patch("sys.stdout", io.StringIO()):
                self.assertEqual(RUNNER.main(["metrics", *base]), 1)
                run.assert_not_called()

    def test_symlinked_loc_summary_all_metrics_preflight(self):
        self.completed_checkpoint("exp37", 19500)
        manifest_file = self.root / "data/csgo_benchmark_v2/benchmark_manifest.json"
        shutil.copyfile(REPO_ROOT / "data/csgo_benchmark_v2/benchmark_manifest.json", manifest_file)
        physical = self.root / "physical_loc"
        physical.mkdir()
        (self.root / "outputs_loc").symlink_to(physical, target_is_directory=True)
        base = ["--experiment", "exp37", "--checkpoint-step", "final", "--split", "test",
                "--task", "all", "--execute"]
        with patch.object(RUNNER.subprocess, "run") as run, patch("sys.stdout", io.StringIO()):
            self.assertEqual(RUNNER.main(["infer", *base]), 0)
            self.assertEqual(run.call_count, 3)
        target = "benchmark_v2/exp37/checkpoint-19500/test/seed_42"
        output = self.root / "outputs_loc" / target / "infer_bs32"
        config_path = output / "evaluation_config.yaml"
        config = RUNNER.read_yaml(config_path)
        manifest = {"config_path": str(config_path.resolve()),
                    "benchmark_v2_manifest": str(manifest_file),
                    "benchmark_v2_split": "seen_discrete_test", "maps": SEEN_MAPS,
                    "sample_count": 20000, "checkpoint": str(RUNNER.path_in_repo(config["ckpt_path"])),
                    "ckpt_path": str(RUNNER.path_in_repo(config["ckpt_path"])), "seed": 42,
                    "benchmark_v2_support_seed": None,
                    "benchmark_v2_shots_per_map": None,
                    "inference_execution": execution_contract("loc", 32)}
        manifest_path = output / "inference_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        metrics = {"XY_Dist": 1.0, "Z_Dist": 2.0, "Pitch_Dist": 3.0, "Yaw_Dist": 4.0}
        summary = RUNNER.aggregates.build_localization_summary(
            manifest=manifest_file, split="seen_discrete_test", maps=SEEN_MAPS,
            per_map={name: metrics for name in SEEN_MAPS}, metrics_macro_map=metrics,
            inference_provenance={"path": str(manifest_path), "payload": manifest},
            checkpoint=manifest["checkpoint"], seed=42, support_seed=None,
            shots_per_map=None, sample_count=20000)
        self.assertEqual(summary["inference_provenance"]["path"], str(manifest_path.resolve()))
        summary_path = output / "benchmark_csgo_v2_loc.json"
        summary_path.write_text(json.dumps(summary))
        for kind in ("discrete", "continuous"):
            gen_output = self.root / "outputs_eval" / target / "infer_bs16" / kind
            gen_config = gen_output / "evaluation_config.yaml"
            (gen_output / "inference_manifest.json").write_text(json.dumps({
                "config_path": str(gen_config.resolve()),
                "inference_execution": execution_contract("gen", 16)}))
        with patch.object(RUNNER.subprocess, "run") as run, patch("sys.stdout", io.StringIO()):
            self.assertEqual(RUNNER.main(["metrics", *base]), 0)
            self.assertEqual(run.call_count, 22)
        with patch.object(RUNNER.subprocess, "run") as run, patch("sys.stdout", io.StringIO()):
            self.assertEqual(RUNNER.main(["metrics", *base, "--metrics-device", "cpu"]), 0)
            self.assertEqual(run.call_count, 22)
            self.assertTrue(all(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
                                for call in run.call_args_list))
            self.assertTrue(all(option(call.args[0], "--device") == "cpu"
                                for call in run.call_args_list if "--device" in call.args[0]))
        alternate = physical / "other" / "inference_manifest.json"
        alternate.parent.mkdir()
        alternate.write_text(json.dumps(manifest))
        corrupted = copy.deepcopy(summary)
        corrupted["inference_provenance"]["path"] = str(alternate)
        summary_path.write_text(json.dumps(corrupted))
        with patch.object(RUNNER.subprocess, "run") as run, patch("sys.stdout", io.StringIO()), \
             patch("sys.stderr", io.StringIO()) as error:
            self.assertEqual(RUNNER.main(["metrics", *base]), 1)
            self.assertIn("inference provenance differs", error.getvalue())
            run.assert_not_called()
        summary_path.write_text(json.dumps(summary))
        manifest["inference_execution"] = execution_contract("loc", 1)
        manifest_path.write_text(json.dumps(manifest))
        with patch.object(RUNNER.subprocess, "run") as run, patch("sys.stdout", io.StringIO()):
            self.assertEqual(RUNNER.main(["metrics", *base]), 1)
            run.assert_not_called()

    def test_eval_batch_flags_do_not_change_training_plan(self):
        config = self.config("exp37")
        default = self.args("train", "exp37")
        legacy = self.args("train", "exp37", "--test-loc-batch-size", "1",
                           "--test-gen-batch-size", "1")
        self.assertEqual(RUNNER.training_plan(default, config),
                         RUNNER.training_plan(legacy, config))


if __name__ == "__main__":
    unittest.main()
