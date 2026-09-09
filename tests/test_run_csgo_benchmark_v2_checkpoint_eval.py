from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts/run_csgo_benchmark_v2_checkpoint_eval.py"
SPEC = importlib.util.spec_from_file_location(
    "run_csgo_benchmark_v2_checkpoint_eval", MODULE_PATH
)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def option(command: list[str], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


class BenchmarkV2CheckpointEvaluationRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.config_path = RUNNER.load_config()

    def test_configuration_and_protocol_files_are_valid(self):
        self.assertEqual(
            self.config_path,
            (REPO_ROOT / "csgo_configs/benchmark_v2_checkpoint_eval.yaml").resolve(),
        )
        RUNNER.validate_configuration(self.config)

    def test_checkpoint_identity_and_training_step_are_frozen_in_matrix(self):
        expected = {
            "exp31": "b56dced04cab9ac35d952a592b6c7b16ff59210c55814605812e5b1996f898c6",
            "exp32": "16642d606c0ae531b99289a44bdfc8b64606c4dbb6a850f935d504ae7bded0d8",
        }
        for experiment, digest in expected.items():
            with self.subTest(experiment=experiment):
                spec = self.config["experiments"][experiment]
                self.assertEqual(spec["expected_sha256"], digest)
                self.assertEqual(spec["expected_global_step"], 6000)
                RUNNER._validate_checkpoint_artifact(self.config, experiment)

    def test_checkpoint_cli_override_is_present_for_all_six_tasks(self):
        for experiment in RUNNER.EXPERIMENTS:
            expected = f"outputs/csgo_1b/{experiment}/checkpoint-6000/model.safetensors"
            for task in RUNNER.TASKS:
                with self.subTest(experiment=experiment, task=task):
                    command = (
                        RUNNER.build_localization_command(self.config, experiment)
                        if task == "localization"
                        else RUNNER.build_generation_inference_command(
                            self.config, experiment, task
                        )
                    )
                    self.assertEqual(option(command, "--ckpt_path"), expected)
                    self.assertEqual(option(command, "--seed"), "42")
                    self.assertEqual(
                        option(command, "--benchmark_v2_split"),
                        RUNNER.SPLITS[task],
                    )
                    maps_index = command.index("--benchmark_v2_maps")
                    self.assertEqual(command[maps_index + 1 :], list(RUNNER.SEEN_MAPS))

    def test_checkpoint_paths_and_outputs_are_isolated(self):
        for experiment in RUNNER.EXPERIMENTS:
            self.assertEqual(
                RUNNER.checkpoint_path(self.config, experiment),
                REPO_ROOT
                / f"outputs/csgo_1b/{experiment}/checkpoint-6000/model.safetensors",
            )
            self.assertEqual(
                RUNNER.localization_dir(self.config, experiment),
                REPO_ROOT
                / f"outputs_loc/benchmark_v2/{experiment}/checkpoint_6000/seen",
            )
            for task in RUNNER.GENERATION_TASKS:
                self.assertEqual(
                    RUNNER.generation_dir(self.config, experiment, task),
                    REPO_ROOT
                    / f"outputs_eval/benchmark_v2/{experiment}/checkpoint_6000/seen/{task}",
                )

    def test_minimal_asset_switch_routes_all_tasks_and_skips_source_data_dir(self):
        config = copy.deepcopy(self.config)
        config["benchmark_v2_asset_manifest"] = (
            "data/csgo_benchmark_v2/minimal_dataset_report.json"
        )
        config["evaluation"].pop("data_dir")
        RUNNER.validate_configuration(config, check_runtime_files=False)
        localization = RUNNER.build_localization_command(config, "exp31")
        generation = RUNNER.build_generation_inference_command(
            config, "exp31", "discrete"
        )
        metric = RUNNER.build_metric_command(config, "exp31", "discrete", "de_train")
        for command in (localization, generation, metric):
            self.assertEqual(
                option(command, "--benchmark_v2_asset_manifest"),
                "data/csgo_benchmark_v2/minimal_dataset_report.json",
            )
        self.assertEqual(
            option(metric, "--gt"), "data/csgo_benchmark_v2/images/de_train"
        )
        self.assertNotIn("--data_dir", metric)

        provenance = RUNNER._asset_expected_provenance(config)
        self.assertIsNotNone(provenance)
        conflicting = dict(provenance)
        conflicting["benchmark_v2_asset"] = {
            "manifest": provenance["benchmark_v2_asset_manifest"],
            "backend": "source",
            "sha256": provenance["benchmark_v2_asset_manifest_sha256"],
            "selected_images_sha256": provenance[
                "benchmark_v2_selected_images_sha256"
            ],
        }
        with self.assertRaisesRegex(RUNNER.PipelineError, "conflicting"):
            RUNNER._validate_asset_provenance(
                config, conflicting, REPO_ROOT / "inference_manifest.json"
            )

    def test_generation_commands_are_inference_then_ten_metrics_then_aggregate(self):
        commands = RUNNER.build_task_commands(self.config, "exp32", "continuous")
        self.assertEqual(commands[0][0], "inference")
        self.assertEqual(commands[-1][0], "aggregate")
        self.assertEqual(
            [stage.removeprefix("metric_") for stage, _ in commands[1:-1]],
            list(RUNNER.SEEN_MAPS),
        )
        self.assertEqual(commands[0][1][1], "eval_csgo.py")
        self.assertEqual(commands[1][1][1], "benchmark_csgo_v1_conti.py")
        self.assertEqual(commands[-1][1][2], "maps")

    def test_schedule_order_is_localization_first_and_is_bounded(self):
        scheduling = RUNNER._scheduling(self.config)
        self.assertEqual(scheduling["max_parallel_pipelines"], 6)
        self.assertGreaterEqual(scheduling["launch_settle_seconds"], 120)
        jobs = RUNNER._default_jobs(self.config)
        self.assertEqual(
            jobs[:2], [("exp31", "localization"), ("exp32", "localization")]
        )
        self.assertEqual(len(jobs), 6)

    def test_inference_manifest_rejects_wrong_sample_count_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                RUNNER.EXPECTED_SAMPLES,
                {"seen_discrete_test": 1, "seen_continuous": 1},
            ):
                config = copy.deepcopy(self.config)
                config["paths"]["generation_root"] = str(
                    Path(temp_dir) / "outputs_eval"
                )
                config["protocol"]["seen_maps"] = ["seen_a"]
                config["protocol"]["expected_samples"] = {
                    "seen_discrete_test": 1,
                    "seen_continuous": 1,
                }
                output = RUNNER.generation_dir(config, "exp31", "discrete")
                (output / "gen_imgs" / "seen_a").mkdir(parents=True)
                (output / "gen_imgs" / "seen_a" / "frame.jpg").touch()
                payload = RUNNER._expected_inference_payload(
                    config, "exp31", "discrete"
                )
                (output / "inference_manifest.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                RUNNER._validate_inference_manifest(config, "exp31", "discrete")

                payload["sample_count"] = 2
                (output / "inference_manifest.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                with self.assertRaises(RUNNER.PipelineError):
                    RUNNER._validate_inference_manifest(config, "exp31", "discrete")

                payload["sample_count"] = 1
                payload["ckpt_path"] = "outputs/csgo_1b/exp31/model.safetensors"
                (output / "inference_manifest.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                with self.assertRaises(RUNNER.PipelineError):
                    RUNNER._validate_inference_manifest(config, "exp31", "discrete")

    def test_localization_manifest_validates_results_instead_of_generation_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                RUNNER.EXPECTED_SAMPLES,
                {"seen_discrete_test": 1, "seen_continuous": 1},
            ):
                config = copy.deepcopy(self.config)
                config["paths"]["localization_root"] = str(
                    Path(temp_dir) / "outputs_loc"
                )
                config["protocol"]["seen_maps"] = ["seen_a"]
                config["protocol"]["expected_samples"] = {
                    "seen_discrete_test": 1,
                    "seen_continuous": 1,
                }
                output = RUNNER.localization_dir(config, "exp31")
                output.mkdir(parents=True)
                payload = RUNNER._expected_inference_payload(
                    config, "exp31", "localization"
                )
                (output / "inference_manifest.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )
                (output / "loc_results.json").write_text(
                    json.dumps([{"map": "seen_a"}]), encoding="utf-8"
                )
                RUNNER._validate_inference_manifest(config, "exp31", "localization")

                (output / "loc_results.json").write_text("[]", encoding="utf-8")
                with self.assertRaises(RUNNER.PipelineError):
                    RUNNER._validate_inference_manifest(config, "exp31", "localization")

    def test_localization_metric_metadata_is_not_treated_as_a_numeric_metric(self):
        checkpoint = "outputs/csgo_1b/exp31/checkpoint-6000/model.safetensors"
        RUNNER._finite_metrics(
            {"L2_5D": 1.25, "ckpt_path": checkpoint},
            "per-map localization metrics",
            expected_metadata={"ckpt_path": checkpoint},
        )
        with self.assertRaises(RUNNER.PipelineError):
            RUNNER._finite_metrics(
                {"L2_5D": 1.25, "ckpt_path": "wrong.safetensors"},
                "per-map localization metrics",
                expected_metadata={"ckpt_path": checkpoint},
            )

    def test_validate_does_not_bind_yaml_checkpoint_to_cli_checkpoint(self):
        config = copy.deepcopy(self.config)
        config_path = REPO_ROOT / config["experiments"]["exp31"]["localization_config"]
        yaml_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertNotEqual(
            yaml_config["ckpt_path"],
            "outputs/csgo_1b/exp31/checkpoint-6000/model.safetensors",
        )
        command = RUNNER.build_localization_command(config, "exp31")
        self.assertEqual(
            option(command, "--ckpt_path"),
            "outputs/csgo_1b/exp31/checkpoint-6000/model.safetensors",
        )


if __name__ == "__main__":
    unittest.main()
