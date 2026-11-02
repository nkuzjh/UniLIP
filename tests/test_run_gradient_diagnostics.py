import json
import hashlib
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_gradient_diagnostics as controller


def valid_report(item):
    checkpoint = controller.checkpoint_path(item)
    config_hash = hashlib.sha256(
        (checkpoint.parent / "resolved_config.yaml").read_bytes()).hexdigest()
    result = {
        "linearity": {"passed": True},
        "scalar_linearity": {"weights_constant": True, "max_abs_difference": 1e-6},
        "total_gradient_norm": 1.0,
        "other_objectives_active": False,
        "aux_freeze": {"status": "skipped", "passed": None},
        "total_backward_check": {"passed": True},
        "repeated_total_vjp": {"passed": True},
    }
    if item.measurement_mode == "component_sum":
        result.update({
            "measurement_mode": "component_sum", "linearity": None,
            "measurement_scope": {
                "components": "direct_vjp_every_microbatch",
                "main_total": "component_sum_of_mean_weighted_vjps",
                "other": "zero_by_verified_weights",
                "direct_reconstruction": "first_microbatch_only",
            },
            "first_micro_direct_checks": {
                "scope": "first_microbatch_direct_vjp",
                "main_vs_components": {"passed": True, "difference_norm": 0.0,
                                       "reference_norm": 1.0, "rtol": 2e-3, "atol": 1e-5},
                "total_vs_components": {"passed": True, "difference_norm": 0.0,
                                        "reference_norm": 1.0, "rtol": 2e-3, "atol": 1e-5},
            },
            "total_gradient_coverage": None,
            "components": {"linearity_residual": {"weighted_norm": None,
                                                   "signed_contribution_to_total": None}},
            "combined": {key: None for key in (
                "main_joint_vjp_norm", "main_joint_vs_component_sum_difference_norm",
                "actual_aux_increment_norm", "actual_aux_increment_vs_component_cosine",
                "actual_aux_increment_vs_component_difference_norm")},
            "groups": {"shared": {"linearity_difference_norm": None,
                                  "linearity_relative_difference": None}},
        })
    if item.stage in ("confirm", "temporal"):
        result["coordinate"] = {"parameter_count": 336, "numel": 17596416}
        result["shared_language_layers"] = {
            "status": "ok", "shared_parameter_count": 336,
            "shared_numel": 17596416, "unassigned": None,
            "global_shared_weighted_gen_loc_dot": 24.0,
            "layers": {str(i): {"parameter_count": 14, "numel": 733184,
                                "weighted_gen_loc_dot": 1.0} for i in range(24)},
            "checks": {
                "named_layer_dot_sum": 24.0,
                "named_layer_dot_matches_global": True,
                "partition_dot_sum": 24.0,
                "partition_dot_difference": 0.0,
                "partition_dot_matches_global": True,
                "covered_parameter_count": 336,
                "covered_numel": 17596416,
                "parameter_count_matches_global": True,
                "numel_matches_global": True,
            },
        }
    count = item.source_count
    return {
        "measurement_mode": item.measurement_mode,
        "stage": item.probe_stage, "checkpoint_dir": str(checkpoint.resolve()),
        "source_seed": item.source_seed, "source_offset": item.source_offset,
        "noise_seeds": list(item.noise_seeds), "coordinate": "shared",
        "tasks": ["gen", "loc", "perception", "aux_loc"],
        "source_indices": list(range(count)),
        "source_rows": [{"map": "de_ancient", "file_frame": str(i)} for i in range(count)],
        "microbatch_size": 1, "accumulations": count,
        "metadata": {
            "experiment": item.experiment, "checkpoint_step": item.checkpoint_step,
            "measurement_mode": item.measurement_mode,
            "gradient_checkpointing": True, "dtype": "float32",
            "precision_reference": True, "deterministic_probe": True,
            "parameter_dtype_numel": {"torch.float32": 1},
            "precision_reference_adjustments": {
                "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                "flash_sdp_enabled": False, "mem_efficient_sdp_enabled": False,
                "math_sdp_enabled": True,
            },
            "deterministic_algorithms": True,
            "flash_attention_deterministic_env": "1",
            "resolved_config_sha256": config_hash,
        },
        "results": {str(seed): copy.deepcopy(result) for seed in item.noise_seeds},
    }


class GradientDiagnosticControllerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = controller.load_config(controller.DEFAULT_CONFIG)

    def test_fixed_queue_and_read_only_plan(self):
        items = controller.build_items(self.config)
        self.assertEqual(len(items), 27)
        self.assertEqual(sum(item.stage == "screen" for item in items), 6)
        self.assertEqual(sum(item.stage == "confirm" for item in items), 18)
        self.assertEqual(sum(item.stage == "temporal" for item in items), 3)
        self.assertEqual({item.source_offset for item in items if item.stage == "confirm"},
                         {0, 128, 256})
        self.assertTrue(all(item.noise_seeds == (17, 29) for item in items))
        self.assertTrue(all(item.precision_reference and item.deterministic_probe for item in items))
        self.assertTrue(all(item.numerical_check for item in items))
        self.assertTrue(all(item.measurement_mode == ("full" if item.stage == "screen" else "component_sum") for item in items))
        self.assertTrue(all(item.check_total_backward for item in items))
        preferred = {item.experiment: item.preferred_microbatch for item in items}
        self.assertEqual(preferred["exp32_3maps_stage2_control_j8"], 8)
        self.assertEqual(preferred["exp32_3maps_joint_constloc8"], 8)
        self.assertEqual(preferred["exp32_3maps_P3_S1_J8"], 2)
        self.assertEqual(preferred["exp32_3maps_B2_S3_J8"], 1)
        with tempfile.TemporaryDirectory() as temporary:
            plan_config = dict(self.config, output_root=str(Path(temporary) / "results"))
            config_path = Path(temporary) / "queue.yaml"
            import yaml
            config_path.write_text(yaml.safe_dump(plan_config))
            controller.main(["--config", str(config_path), "--stage", "screen"])
            self.assertFalse((Path(temporary) / "results").exists())

    def test_gpu_memory_policy_and_command(self):
        with self.assertRaises(ValueError):
            controller.choose_microbatch(33.9, 96, 8)
        self.assertEqual(controller.choose_microbatch(40, 96, 8)[0], 8)
        self.assertEqual(controller.choose_microbatch(40, 96, 2)[0], 2)
        self.assertEqual(controller.choose_microbatch(54.9, 96, 1)[0], 1)
        micro, fraction = controller.choose_microbatch(56, 96, 1)
        self.assertEqual(micro, 2)
        self.assertAlmostEqual(fraction, 50 / 96)
        item = controller.build_items(self.config, ("confirm",))[0]
        command = controller.command_for(item, python="python", output=Path("out.json"),
                                         device="cuda:0", microbatch=2,
                                         gpu_memory_fraction=fraction)
        self.assertEqual(command[command.index("--accumulations") + 1], "64")
        self.assertEqual(command[command.index("--source-offset") + 1], "0")
        self.assertIn("--gpu-memory-fraction", command)
        self.assertIn("--fp32-reference", command)
        self.assertIn("--deterministic-probe", command)
        self.assertIn("--numerical-check", command)
        self.assertEqual(command[command.index("--measurement-mode") + 1], "component_sum")
        self.assertIn("--check-total-backward", command)
        screen_command = controller.command_for(controller.build_items(self.config, ("screen",))[0],
                                                python="python", output=Path("out.json"),
                                                device="cuda:0", microbatch=1,
                                                gpu_memory_fraction=fraction)
        self.assertIn("--numerical-check", screen_command)
        self.assertEqual(screen_command[screen_command.index("--measurement-mode") + 1], "full")
        fast_command = controller.command_for(controller.build_items(self.config, ("screen",))[0],
                                              python="python", output=Path("out.json"),
                                              device="cuda:0", microbatch=8,
                                              gpu_memory_fraction=fraction)
        self.assertEqual(fast_command[fast_command.index("--accumulations") + 1], "12")

    def test_report_checks_require_fp32_determinism_and_small_scalar_residual(self):
        item = controller.build_items(self.config, ("screen",))[0]
        report = valid_report(item)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            path.write_text(json.dumps(report))
            self.assertEqual(controller.report_checks(path, item), (True, []))
            report.pop("measurement_mode")
            report["metadata"].pop("measurement_mode")
            path.write_text(json.dumps(report))
            self.assertEqual(controller.report_checks(path, item), (True, []))
            report["results"]["29"]["scalar_linearity"] = {
                "weights_constant": True, "max_abs_difference": 2e-5,
            }
            path.write_text(json.dumps(report))
            valid, errors = controller.report_checks(path, item)
            self.assertFalse(valid)
            self.assertIn("seed 29 scalar linearity", errors)
            report["results"]["29"]["scalar_linearity"]["max_abs_difference"] = 1e-6
            report["microbatch_size"] = 8
            report["accumulations"] = 12
            path.write_text(json.dumps(report))
            self.assertEqual(controller.report_checks(path, item), (True, []))
            report["accumulations"] = 11
            path.write_text(json.dumps(report))
            self.assertIn("microbatch or accumulation count", controller.report_checks(path, item)[1])

    def test_confirm_and_temporal_require_complete_shared_layer_partition(self):
        for stage in ("confirm", "temporal"):
            item = controller.build_items(self.config, (stage,))[0]
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "report.json"
                report = valid_report(item)
                path.write_text(json.dumps(report))
                self.assertEqual(controller.report_checks(path, item), (True, []))

                cases = (
                    (lambda stats: stats.pop("shared_language_layers"), "shared layer payload"),
                    (lambda stats: stats["shared_language_layers"]["layers"]["23"].update(
                        parameter_count=13), "shared layer coverage"),
                    (lambda stats: stats["shared_language_layers"]["layers"].pop("23"),
                     "shared layer count"),
                    (lambda stats: stats["shared_language_layers"].update(
                        shared_numel=17596415), "shared layer coordinate"),
                    (lambda stats: stats["shared_language_layers"].update(
                        unassigned={"parameter_count": 1}), "shared layer status/unassigned"),
                    (lambda stats: stats["shared_language_layers"]["layers"]["0"].update(
                        weighted_gen_loc_dot=2.0), "shared layer dot"),
                    (lambda stats: stats["shared_language_layers"]["checks"].update(
                        partition_dot_matches_global=False), "shared layer partition checks"),
                )
                for mutate, expected_error in cases:
                    broken = copy.deepcopy(report)
                    mutate(broken["results"]["17"])
                    path.write_text(json.dumps(broken))
                    valid, errors = controller.report_checks(path, item)
                    self.assertFalse(valid)
                    self.assertIn(f"seed 17 {expected_error}", errors)

    def test_component_sum_rejects_failed_or_missing_direct_checks_and_wrong_mode(self):
        item = controller.build_items(self.config, ("confirm",))[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            report = valid_report(item)
            path.write_text(json.dumps(report))
            self.assertEqual(controller.report_checks(path, item), (True, []))
            cases = (
                (lambda r: r["results"]["17"].pop("first_micro_direct_checks"),
                 "seed 17 first micro direct checks"),
                (lambda r: r["results"]["17"]["first_micro_direct_checks"]["total_vs_components"].update(passed=False),
                 "seed 17 first micro total_vs_components"),
                (lambda r: r["results"]["17"]["first_micro_direct_checks"]["main_vs_components"].update(difference_norm=1.),
                 "seed 17 first micro main_vs_components"),
                (lambda r: r["results"]["17"].pop("linearity"),
                 "seed 17 linearity scope"),
                (lambda r: r.update(measurement_mode="full"), "measurement mode"),
                (lambda r: r["results"]["17"].update(measurement_mode="full"),
                 "seed 17 measurement mode"),
                (lambda r: r["results"]["17"]["total_backward_check"].update(passed=False),
                 "seed 17 total backward"),
                (lambda r: r["results"]["17"]["groups"]["shared"].pop("linearity_difference_norm"),
                 "seed 17 group residual scope"),
            )
            for mutate, expected in cases:
                broken = copy.deepcopy(report)
                mutate(broken)
                path.write_text(json.dumps(broken))
                self.assertIn(expected, controller.report_checks(path, item)[1])

    def test_code_hash_participates_in_signature(self):
        item = controller.build_items(self.config, ("screen",))[0]
        left = controller.run_signature(item, code_hash="a", config_hash="z")
        right = controller.run_signature(item, code_hash="b", config_hash="z")
        self.assertNotEqual(left, right)

    def test_existing_result_requires_matching_digest_for_skip(self):
        item = controller.build_items(self.config, ("screen",))[0]
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            config = dict(self.config, output_root=str(output_root))
            output = controller.report_path(output_root, item)
            output.parent.mkdir(parents=True)
            output.write_text("{}")
            with patch.object(controller, "report_checks", return_value=(True, [])):
                with self.assertRaisesRegex(RuntimeError, "matching parameter/code digest"):
                    controller.run_queue(config, [item], python="python", max_wait_seconds=0,
                                         config_hash="cfg")
            code_hash = controller.code_sha256()
            signature = controller.run_signature(item, code_hash=code_hash, config_hash="cfg")
            manifest = {"schema_version": 1, "items": {item.name: {
                "status": "complete", "signature": signature,
            }}}
            (output_root / "manifest.json").write_text(json.dumps(manifest))
            with patch.object(controller, "report_checks", return_value=(True, [])), \
                 patch.object(controller, "gpu_free_gib", side_effect=AssertionError("GPU queried")):
                result = controller.run_queue(config, [item], python="python", max_wait_seconds=0,
                                              config_hash="cfg")
            self.assertEqual(result["items"][item.name]["status"], "complete")
            self.assertTrue(result["items"][item.name]["reused"])

    def test_oom_halves_microbatch_and_restarts_from_full_source_pool(self):
        item = controller.build_items(self.config, ("screen",))[0]
        with tempfile.TemporaryDirectory() as temporary:
            config = dict(self.config, output_root=temporary)
            output = controller.report_path(Path(temporary), item)
            seen = []

            def fake_run(command, log_path):
                micro = int(command[command.index("--microbatch-size") + 1])
                accumulation = int(command[command.index("--accumulations") + 1])
                seen.append((micro, accumulation))
                log_path.parent.mkdir(parents=True, exist_ok=True)
                if micro > 1:
                    log_path.write_text("CUDA out of memory\n")
                    output.with_suffix(output.suffix + ".jsonl").write_text("partial\n")
                    return 1
                log_path.write_text("complete\n")
                return 0

            def fake_check(path, _item):
                return (bool(seen and seen[-1][0] == 1), [])

            with patch.object(controller, "_wait_for_memory", return_value=(40.0, 96.0)), \
                 patch.object(controller, "_run_command", side_effect=fake_run), \
                 patch.object(controller, "report_checks", side_effect=fake_check):
                manifest = controller.run_queue(config, [item], python="python",
                                                max_wait_seconds=0, config_hash="cfg")
            self.assertEqual(seen, [(8, 12), (4, 24), (2, 48), (1, 96)])
            self.assertEqual(manifest["items"][item.name]["status"], "complete")
            for attempt in (1, 2, 3):
                self.assertTrue(output.with_suffix(f".attempt{attempt}.jsonl").is_file())

    def test_old_partial_run_is_archived_before_new_signature(self):
        item = controller.build_items(self.config, ("screen",))[0]
        with tempfile.TemporaryDirectory() as temporary:
            config = dict(self.config, output_root=temporary)
            output = controller.report_path(Path(temporary), item)
            output.parent.mkdir(parents=True)
            stale_jsonl = output.with_suffix(output.suffix + ".jsonl")
            stale_log = output.with_suffix(".attempt1.log")
            stale_jsonl.write_text("old partial result\n")
            stale_log.write_text("old log\n")
            (Path(temporary) / "manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "items": {item.name: {"status": "running", "signature": "old-hash"}},
            }))

            def fake_run(_command, log_path):
                log_path.write_text("new log\n")
                return 0

            with patch.object(controller, "_wait_for_memory", return_value=(40.0, 96.0)), \
                 patch.object(controller, "_run_command", side_effect=fake_run), \
                 patch.object(controller, "report_checks", side_effect=[(False, []), (True, [])]):
                manifest = controller.run_queue(config, [item], python="python",
                                                max_wait_seconds=0, config_hash="new-cfg")
            self.assertEqual(manifest["history"][0]["record"]["signature"], "old-hash")
            self.assertEqual(manifest["items"][item.name]["status"], "complete")
            self.assertEqual(len(list(output.parent.glob(output.name + ".jsonl.backup.*"))), 1)
            self.assertEqual(len(list(output.parent.glob(output.stem + ".attempt1.log.backup.*"))), 1)


if __name__ == "__main__":
    unittest.main()
