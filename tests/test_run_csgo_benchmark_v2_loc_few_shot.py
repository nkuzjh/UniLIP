"""Contract tests for the Benchmark v2 localization few-shot scheduler."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from csgo_datasets.benchmark_v2 import load_benchmark_v2_selection


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = REPO_ROOT / "csgo_configs/benchmark_v2_loc_few_shot.yaml"
SEEN_MAPS = [
    "cs_agency",
    "cs_italy",
    "de_ancient",
    "de_anubis",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
    "de_overpass",
    "de_train",
]
CROSSMAP_MAPS = ["cs_office", "de_golden", "de_palacio", "de_vertigo"]
SHOTS = (50, 20, 10)


def _option(command: list[str], name: str) -> str:
    for index, token in enumerate(command):
        if token == name:
            return command[index + 1]
        if token.startswith(f"{name}="):
            return token.split("=", 1)[1]
    raise AssertionError(f"missing option {name!r} in command: {command!r}")


def _option_values(command: list[str], name: str) -> list[str]:
    index = command.index(name) + 1
    return command[index:]


class BenchmarkV2LocFewShotSchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scheduler = importlib.import_module(
            "scripts.run_csgo_benchmark_v2_loc_few_shot"
        )
        cls.matrix, cls.config_path = cls.scheduler.load_matrix(MATRIX_PATH)

    def _training_runtime_config(self) -> dict:
        with (REPO_ROOT / "csgo_configs/exp33_loc.yaml").open(
            "r", encoding="utf-8"
        ) as handle:
            config = dict(yaml.safe_load(handle))
        config["benchmark_v2_manifest"] = str(
            REPO_ROOT / "data/csgo_benchmark_v2/benchmark_manifest.json"
        )
        config["data_dir"] = str(REPO_ROOT / "data/preprocessed_data")
        config["benchmark_v2_support_seed"] = 0
        return config

    def test_matrix_configuration_and_parent_checkpoints_are_valid(self):
        self.assertEqual(self.config_path, MATRIX_PATH.resolve())
        self.scheduler.validate_configuration(self.matrix)

        expected_parents = {
            "exp33_loc": "outputs/csgo_1b/exp31_loc/model.safetensors",
            "exp34_loc": "outputs/csgo_1b/exp32_loc/model.safetensors",
        }
        for experiment, parent in expected_parents.items():
            with self.subTest(experiment=experiment):
                spec = self.matrix["experiments"][experiment]
                self.assertEqual(spec["parent_checkpoint"], parent)
                with (REPO_ROOT / spec["train_config"]).open(
                    "r", encoding="utf-8"
                ) as handle:
                    train_config = yaml.safe_load(handle)
                self.assertEqual(train_config["finetune_init_ckpt_path"], parent)
                self.assertTrue((REPO_ROOT / parent).is_file())

    def test_support_counts_and_shots_are_nested_prefixes_of_100_shot(self):
        runtime = self._training_runtime_config()
        runtime["benchmark_v2_split"] = "crossmap_support"
        runtime["benchmark_v2_shots_per_map"] = 100
        full = load_benchmark_v2_selection(runtime)

        self.assertEqual(full.map_names, CROSSMAP_MAPS)
        self.assertEqual(len(full.rows), 400)
        full_by_map = {
            map_name: [row["file_frame"] for row in full.rows if row["map"] == map_name]
            for map_name in CROSSMAP_MAPS
        }
        self.assertTrue(all(len(rows) == 100 for rows in full_by_map.values()))

        for shots in SHOTS:
            with self.subTest(shots=shots):
                runtime["benchmark_v2_shots_per_map"] = shots
                selection = load_benchmark_v2_selection(runtime)
                self.assertEqual(selection.support_seed, 0)
                self.assertEqual(selection.shots_per_map, shots)
                self.assertEqual(len(selection.rows), shots * len(CROSSMAP_MAPS))
                for map_name in CROSSMAP_MAPS:
                    selected = [
                        row["file_frame"]
                        for row in selection.rows
                        if row["map"] == map_name
                    ]
                    self.assertEqual(len(selected), shots)
                    self.assertEqual(selected, full_by_map[map_name][:shots])

    def test_train_commands_preserve_protocol_parent_and_exp_specific_settings(self):
        expected = {
            "exp33_loc": {
                "ports": {50: "29650", 20: "29620", 10: "29610"},
                "parent": "outputs/csgo_1b/exp31_loc/model.safetensors",
                "fix_llm": "True",
                "lora_r": None,
                "batch_sizes": {50: "128", 20: "80", 10: "40"},
            },
            "exp34_loc": {
                "ports": {50: "29750", 20: "29720", 10: "29710"},
                "parent": "outputs/csgo_1b/exp32_loc/model.safetensors",
                "fix_llm": "False",
                "lora_r": "32",
                "batch_sizes": {50: "128", 20: "80", 10: "40"},
            },
        }
        for experiment, details in expected.items():
            with self.subTest(experiment=experiment):
                spec = self.matrix["experiments"][experiment]
                for shots in SHOTS:
                    with self.subTest(shots=shots):
                        command = self.scheduler.build_train_command(
                            self.matrix, experiment, shots
                        )
                        self.assertEqual(command[0], self.scheduler.sys.executable)
                        self.assertEqual(
                            command[1:4],
                            [
                                "-m",
                                "torch.distributed.run",
                                "--nproc_per_node=1",
                            ],
                        )
                        self.assertEqual(
                            _option(command, "--master_port"), details["ports"][shots]
                        )
                        self.assertEqual(
                            _option(command, "--csgo_config"), spec["train_config"]
                        )
                        self.assertEqual(
                            _option(command, "--output_dir"),
                            str(
                                self.scheduler._relative(
                                    self.scheduler.model_dir(
                                        self.matrix, experiment, shots
                                    )
                                )
                            ),
                        )
                        self.assertEqual(_option(command, "--max_steps"), "400")
                        self.assertEqual(
                            _option(command, "--per_device_train_batch_size"),
                            details["batch_sizes"][shots],
                        )
                        self.assertEqual(
                            _option(command, "--fix_llm"), details["fix_llm"]
                        )
                        self.assertEqual(
                            _option(command, "--benchmark_v2_support_seed"), "0"
                        )
                        self.assertEqual(
                            _option(command, "--benchmark_v2_shots_per_map"), str(shots)
                        )
                        self.assertEqual(
                            _option(command, "--finetune_init_ckpt_path")
                            if "--finetune_init_ckpt_path" in command
                            else details["parent"],
                            details["parent"],
                        )
                        if details["lora_r"] is None:
                            self.assertNotIn("--lora_r", command)
                        else:
                            self.assertEqual(
                                _option(command, "--lora_r"), details["lora_r"]
                            )

    def test_eval_commands_use_crossmap_and_seen_retention_contract(self):
        expected = {
            "exp33_loc": "csgo_configs/test/exp33_loc_loc.yaml",
            "exp34_loc": "csgo_configs/test/exp34_loc_loc.yaml",
        }
        for experiment, eval_config in expected.items():
            for shots in SHOTS:
                with self.subTest(experiment=experiment, shots=shots):
                    crossmap = self.scheduler.build_eval_command(
                        self.matrix, experiment, shots, "crossmap_query_test"
                    )
                    seen = self.scheduler.build_eval_command(
                        self.matrix, experiment, shots, "seen_discrete_test"
                    )
                    for command, split, maps, suffix in (
                        (
                            crossmap,
                            "crossmap_query_test",
                            CROSSMAP_MAPS,
                            f"outputs_loc/benchmark_v2/{experiment}/shot_{shots}/seed_0",
                        ),
                        (
                            seen,
                            "seen_discrete_test",
                            SEEN_MAPS,
                            f"outputs_loc/benchmark_v2/{experiment}/shot_{shots}/seed_0/seen_retention",
                        ),
                    ):
                        self.assertEqual(command[0], self.scheduler.sys.executable)
                        self.assertEqual(command[1], "eval_csgo_loc.py")
                        self.assertEqual(_option(command, "--csgo_config"), eval_config)
                        self.assertEqual(_option(command, "--output_dir"), suffix)
                        self.assertEqual(
                            _option(command, "--ckpt_path"),
                            f"outputs/csgo_1b/{experiment}/shot_{shots}/seed_0/model.safetensors",
                        )
                        self.assertEqual(_option(command, "--seed"), "42")
                        self.assertEqual(
                            _option(command, "--benchmark_v2_split"), split
                        )
                        self.assertEqual(
                            _option(command, "--benchmark_v2_support_seed"), "0"
                        )
                        self.assertEqual(
                            _option(command, "--benchmark_v2_shots_per_map"), str(shots)
                        )
                        self.assertEqual(
                            _option_values(command, "--benchmark_v2_maps"), maps
                        )

    def test_minimal_asset_switch_routes_train_and_both_eval_commands(self):
        matrix = dict(self.matrix)
        matrix["benchmark_v2_asset_manifest"] = (
            "data/csgo_benchmark_v2/minimal_dataset_report.json"
        )
        self.scheduler.validate_configuration(matrix)
        train = self.scheduler.build_train_command(matrix, "exp33_loc", 50)
        crossmap = self.scheduler.build_eval_command(
            matrix, "exp33_loc", 50, "crossmap_query_test"
        )
        seen = self.scheduler.build_eval_command(
            matrix, "exp33_loc", 50, "seen_discrete_test"
        )
        for command in (train, crossmap, seen):
            self.assertEqual(
                _option(command, "--benchmark_v2_asset_manifest"),
                "data/csgo_benchmark_v2/minimal_dataset_report.json",
            )

        provenance = self.scheduler._asset_expected_provenance(matrix)
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
        with self.assertRaisesRegex(self.scheduler.PipelineError, "conflicting"):
            self.scheduler._validate_asset_provenance(
                matrix, conflicting, REPO_ROOT / "inference_manifest.json"
            )

    def test_summary_validation_binds_manifest_checkpoint_and_provenance(self):
        experiment = "exp33_loc"
        shots = 50
        split = "crossmap_query_test"
        checkpoint = self.scheduler._relative(
            self.scheduler.checkpoint_path(self.matrix, experiment, shots)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inference_manifest = root / "inference_manifest.json"
            inference_manifest.write_text("{}", encoding="utf-8")
            summary = root / "benchmark_csgo_v2_loc.json"
            payload = {
                "manifest": str(
                    (REPO_ROOT / self.matrix["protocol"]["manifest"]).resolve()
                ),
                "split": split,
                "kind": "localization",
                "maps": CROSSMAP_MAPS,
                "checkpoint": checkpoint,
                "seed": 42,
                "support_seed": 0,
                "shots_per_map": shots,
                "sample_count": 8000,
                "per_map": {map_name: {"L2_XY": 1.0} for map_name in CROSSMAP_MAPS},
                "metrics_macro_map": {"L2_XY": 1.0},
                "inference_provenance": {
                    "path": str(inference_manifest),
                    "payload": {
                        "benchmark_v2_manifest": self.matrix["protocol"]["manifest"],
                        "benchmark_v2_split": split,
                        "benchmark_v2_support_seed": 0,
                        "benchmark_v2_shots_per_map": shots,
                        "maps": CROSSMAP_MAPS,
                        "sample_count": 8000,
                        "checkpoint": checkpoint,
                        "ckpt_path": checkpoint,
                        "seed": 42,
                    },
                },
            }
            summary.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(self.scheduler, "summary_path", return_value=summary):
                self.scheduler._validate_summary(self.matrix, experiment, shots, split)
                payload["inference_provenance"]["payload"]["ckpt_path"] = "wrong"
                summary.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(self.scheduler.PipelineError, "ckpt_path"):
                    self.scheduler._validate_summary(
                        self.matrix, experiment, shots, split
                    )

    def test_dry_run_emits_train_then_two_eval_commands_without_running_them(self):
        for experiment in ("exp33_loc", "exp34_loc"):
            for shots in SHOTS:
                with self.subTest(experiment=experiment, shots=shots):
                    output = io.StringIO()
                    with (
                        patch.object(
                            self.scheduler,
                            "_run_command",
                            side_effect=AssertionError("dry-run started a command"),
                        ) as run_command,
                        contextlib.redirect_stdout(output),
                    ):
                        self.scheduler.run_pipeline(
                            self.matrix,
                            experiment,
                            shots,
                            cuda_device="0",
                            dry_run=True,
                        )

                    lines = output.getvalue().splitlines()
                    self.assertEqual(len(lines), 3)
                    self.assertEqual(
                        [line.split("]", 1)[0] + "]" for line in lines],
                        [
                            f"[{experiment}/{shots}:train]",
                            f"[{experiment}/{shots}:crossmap_query_test]",
                            f"[{experiment}/{shots}:seen_discrete_test]",
                        ],
                    )
                    run_command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
