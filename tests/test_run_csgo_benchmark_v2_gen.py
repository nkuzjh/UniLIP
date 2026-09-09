import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts/run_csgo_benchmark_v2_gen.py"
SPEC = importlib.util.spec_from_file_location("run_csgo_benchmark_v2_gen", MODULE_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class BenchmarkV2GenerationRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix, cls.config_path = RUNNER.load_matrix()

    def test_configuration_and_real_nested_support_are_valid(self):
        RUNNER.validate_configuration(self.matrix)

    def test_scheduler_covers_all_pipelines_and_keeps_memory_gate(self):
        scheduling = RUNNER._scheduling(self.matrix)
        self.assertEqual(
            scheduling["max_parallel_few_shot_pipelines"],
            len(RUNNER.FEW_SHOT_EXPERIMENTS) * len(RUNNER._shots(self.matrix)),
        )
        self.assertEqual(scheduling["minimum_free_memory_mb"], 40000)
        self.assertEqual(scheduling["launch_settle_seconds"], 180)

    def test_scheduler_adopts_external_pipeline_lock_outside_active_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            matrix = copy.deepcopy(self.matrix)
            matrix["paths"]["log_root"] = str(Path(temporary) / "logs")
            external_job = ("exp34_gen", 50)
            pending = [external_job, ("exp33_gen", 20)]
            external = set()
            lock_path = RUNNER._pipeline_lock_path(matrix, *external_job)
            with RUNNER._exclusive_lock(lock_path):
                self.assertTrue(RUNNER._pipeline_lock_is_held(lock_path))
                adopted = RUNNER._adopt_held_pipeline_locks(
                    matrix, pending, external
                )
                self.assertEqual(adopted, [external_job])
                self.assertEqual(pending, [("exp33_gen", 20)])
                self.assertEqual(external, {external_job})
            self.assertFalse(RUNNER._pipeline_lock_is_held(lock_path))

    def test_zero_shot_command_uses_exact_crossmap_protocol(self):
        command = RUNNER.build_inference_command(
            self.matrix, "exp32_gen", None, "discrete"
        )
        rendered = " ".join(command)
        self.assertIn("csgo_configs/test/exp32_gen_gen.yaml", rendered)
        self.assertIn("--benchmark_v2_split crossmap_query_test", rendered)
        self.assertIn("--seed 42", rendered)
        self.assertNotIn("--benchmark_v2_support_seed", command)
        self.assertNotIn("--benchmark_v2_shots_per_map", command)
        self.assertEqual(
            RUNNER.generation_dir(self.matrix, "exp32_gen", None, "discrete"),
            REPO_ROOT
            / "outputs_eval/benchmark_v2/exp32_gen/zero_shot/crossmap/discrete",
        )

    def test_minimal_asset_switch_routes_all_children_and_gt_to_flat_bundle(self):
        matrix = copy.deepcopy(self.matrix)
        matrix["benchmark_v2_asset_manifest"] = (
            "data/csgo_benchmark_v2/minimal_dataset_report.json"
        )
        matrix["evaluation"].pop("data_dir")
        RUNNER.validate_configuration(matrix)

        train = RUNNER.build_train_command(matrix, "exp33_gen", 10)
        inference = RUNNER.build_inference_command(
            matrix, "exp32_gen", None, "discrete"
        )
        metric = RUNNER.build_metric_command(
            matrix, "exp32_gen", None, "discrete", "cs_office"
        )
        for command in (train, inference, metric):
            self.assertEqual(
                RUNNER._command_option(
                    command, "--benchmark_v2_asset_manifest"
                ),
                "data/csgo_benchmark_v2/minimal_dataset_report.json",
            )
        self.assertEqual(
            RUNNER._command_option(metric, "--gt"),
            "data/csgo_benchmark_v2/images/cs_office",
        )
        self.assertNotIn("--data_dir", metric)

        provenance = RUNNER._asset_expected_provenance(matrix)
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
                matrix, conflicting, REPO_ROOT / "inference_manifest.json"
            )

    def test_seen_command_uses_exact_seen_protocol_and_paths(self):
        parser_args = RUNNER._build_parser().parse_args(
            [
                "run-seen",
                "--experiment",
                "exp32_gen",
                "--kind",
                "continuous",
                "--cuda-device",
                "0",
                "--dry-run",
            ]
        )
        self.assertEqual(parser_args.command, "run-seen")
        self.assertEqual(parser_args.experiment, "exp32_gen")
        self.assertEqual(parser_args.kind, "continuous")
        self.assertTrue(parser_args.dry_run)

        expected_maps = list(RUNNER.SEEN_MAPS)
        for kind, split, count in (
            ("discrete", "seen_discrete_test", 20000),
            ("continuous", "seen_continuous", 12800),
        ):
            command = RUNNER.build_inference_command(
                self.matrix, "exp32_gen", None, kind, "seen"
            )
            rendered = " ".join(command)
            self.assertIn(
                f"csgo_configs/test/exp32_gen_gen{'_conti' if kind == 'continuous' else ''}.yaml",
                rendered,
            )
            self.assertIn(f"--benchmark_v2_split {split}", rendered)
            self.assertIn("--seed 42", rendered)
            map_index = command.index("--benchmark_v2_maps")
            self.assertEqual(command[map_index + 1 :], expected_maps)
            self.assertNotIn("--benchmark_v2_support_seed", command)
            self.assertNotIn("--benchmark_v2_shots_per_map", command)
            self.assertEqual(
                RUNNER.generation_dir(self.matrix, "exp32_gen", None, kind, "seen"),
                REPO_ROOT / f"outputs_eval/benchmark_v2/exp32_gen/seen/{kind}",
            )
            self.assertEqual(
                RUNNER.log_dir(self.matrix, "exp32_gen", None, "seen"),
                REPO_ROOT / "logs/benchmark_v2_gen/exp32_gen/seen",
            )
            self.assertEqual(
                RUNNER._expected_for_kind(self.matrix, kind, "seen")[1], count
            )

    def test_seen_metric_and_aggregate_commands_keep_existing_metric_contract(self):
        metric = RUNNER.build_metric_command(
            self.matrix, "exp32_gen", None, "continuous", "de_train", "seen"
        )
        self.assertEqual(metric[1], "benchmark_csgo_v1_conti.py")
        for option, expected in {
            "--map_name": "de_train",
            "--benchmark_v2_manifest": "data/csgo_benchmark_v2/benchmark_manifest.json",
            "--benchmark_v2_split": "seen_continuous",
            "--external_loc_repo_root": "csgosquare",
            "--external_loc_config_path": "configs_reg_newdata/exp5_2.yaml",
            "--external_loc_checkpoint_path": "checkpoints_reg_newdata/exp5_2/20251227_091745/current_model.pth",
        }.items():
            self.assertEqual(RUNNER._command_option(metric, option), expected)
        aggregate = RUNNER.build_aggregate_command(
            self.matrix, "exp32_gen", None, "discrete", "seen"
        )
        self.assertEqual(aggregate[2], "maps")
        self.assertEqual(
            RUNNER._command_option(aggregate, "--split"), "seen_discrete_test"
        )
        self.assertEqual(
            RUNNER._command_option(aggregate, "--input_root"),
            "outputs_eval/benchmark_v2/exp32_gen/seen/discrete",
        )

    def test_few_shot_training_commands_use_fixed_steps_and_safe_batches(self):
        expected_batches = {100: "128", 50: "128", 20: "80", 10: "40"}
        for experiment in RUNNER.FEW_SHOT_EXPERIMENTS:
            for shots, expected_batch in expected_batches.items():
                command = RUNNER.build_train_command(self.matrix, experiment, shots)
                self.assertEqual(command[command.index("--max_steps") + 1], "400")
                self.assertEqual(
                    command[command.index("--per_device_train_batch_size") + 1],
                    expected_batch,
                )
                self.assertEqual(
                    command[command.index("--benchmark_v2_shots_per_map") + 1],
                    str(shots),
                )
                output = command[command.index("--output_dir") + 1]
                self.assertEqual(
                    output,
                    f"outputs/csgo_1b/{experiment}/shot_{shots}/seed_0",
                )

    def test_existing_exp31_zero_shot_artifacts_validate(self):
        for kind in RUNNER.KINDS:
            RUNNER._validate_inference(self.matrix, "exp31_gen", None, kind)
            summary = RUNNER._validate_summary(
                self.matrix, "exp31_gen", None, kind
            )
            self.assertEqual(summary["sample_count"], 8000 if kind == "discrete" else 5120)

    def test_existing_exp31_seen_artifacts_validate(self):
        for kind, count in (("discrete", 20000), ("continuous", 12800)):
            RUNNER._validate_inference(
                self.matrix, "exp31_gen", None, kind, "seen"
            )
            summary = RUNNER._validate_summary(
                self.matrix, "exp31_gen", None, kind, "seen"
            )
            self.assertEqual(summary["sample_count"], count)

    def test_seen_validation_rejects_manifest_count_jpg_and_nonfinite_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            matrix = copy.deepcopy(self.matrix)
            matrix["paths"]["generation_root"] = str(Path(temporary) / "generated")
            matrix["seen"]["maps"] = ["cs_agency"]
            matrix["seen"]["expected_samples"] = {
                "seen_discrete_test": 1,
                "seen_continuous": 1,
            }
            output = RUNNER.generation_dir(
                matrix, "exp32_gen", None, "discrete", "seen"
            )
            map_dir = output / "gen_imgs" / "cs_agency"
            map_dir.mkdir(parents=True)
            (map_dir / "frame.jpg").touch()
            manifest = RUNNER._expected_inference_payload(
                matrix, "exp32_gen", None, "discrete", "seen"
            )
            manifest_path = output / "inference_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            RUNNER._validate_inference(
                matrix, "exp32_gen", None, "discrete", "seen"
            )

            manifest["sample_count"] = 2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(RUNNER.PipelineError):
                RUNNER._validate_inference(
                    matrix, "exp32_gen", None, "discrete", "seen"
                )

            manifest["sample_count"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (map_dir / "frame.jpg").unlink()
            with self.assertRaises(RUNNER.PipelineError):
                RUNNER._validate_inference(
                    matrix, "exp32_gen", None, "discrete", "seen"
                )

            (map_dir / "frame.jpg").touch()
            summary = {
                "manifest": str(
                    (REPO_ROOT / self.matrix["protocol"]["manifest"]).resolve()
                ),
                "split": "seen_discrete_test",
                "kind": "discrete",
                "maps": ["cs_agency"],
                "per_map": {"cs_agency": {"PSNR": 1.0}},
                "metrics_macro_map": {"PSNR": 1.0},
                "inference_provenance": {
                    "path": str(manifest_path),
                    "payload": manifest,
                },
                "checkpoint": "outputs/csgo_1b/exp32_gen/model.safetensors",
                "ckpt_path": "outputs/csgo_1b/exp32_gen/model.safetensors",
                "inference_seed": 42,
                "support_seed": None,
                "shots_per_map": None,
                "sample_count": 1,
            }
            summary_path = output / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            RUNNER._validate_summary(
                matrix, "exp32_gen", None, "discrete", "seen"
            )
            summary["metrics_macro_map"]["PSNR"] = float("nan")
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaises(RUNNER.PipelineError):
                RUNNER._validate_summary(
                    matrix, "exp32_gen", None, "discrete", "seen"
                )

    def test_results_sync_only_updates_existing_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            temp_path = Path(temporary) / "results.md"
            source = REPO_ROOT / "csgo_benchmark_v2_experiments_results.md"
            temp_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            matrix = copy.deepcopy(self.matrix)
            matrix["paths"]["results_file"] = str(temp_path)
            RUNNER.sync_results(matrix)
            rendered = temp_path.read_text(encoding="utf-8")
            headings = [line for line in rendered.splitlines() if line.startswith("#")]
            self.assertEqual(
                headings,
                [
                    "# csgo benchmark v2 实验进度",
                    "# csgo benchmark v2 主表",
                    "## 定位",
                    "## 离散生成",
                    "## 连续生成",
                ],
            )
            for experiment in RUNNER.FEW_SHOT_EXPERIMENTS:
                for shots in (100, 50, 20, 10):
                    self.assertIn(f"| `{experiment}` {shots}-shot |", rendered)
                    self.assertEqual(
                        rendered.count(
                            f"| CrossMap-4 few-shot | Discrete generation | {experiment} | {shots} |"
                        ),
                        1,
                    )
                    self.assertEqual(
                        rendered.count(
                            f"| CrossMap-4 few-shot | Continuous generation | {experiment} | {shots} |"
                        ),
                        1,
                    )
            self.assertIn(
                "| CrossMap-4 zero-shot | Discrete generation | exp31_gen | - | 12.288 |",
                rendered,
            )
            self.assertIn(
                "| Seen-10 | Discrete generation | exp31_gen | - | 14.561 | 0.4268 | 0.5888 | 0.5258 | 28.629 |",
                rendered,
            )
            self.assertIn(
                "| Seen-10 | Continuous generation | exp31_gen | - | 15.164 | 0.4349 | 0.5622 | 31.409 | 38.033 | 737.197 |",
                rendered,
            )


if __name__ == "__main__":
    unittest.main()
