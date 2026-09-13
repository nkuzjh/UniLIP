import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts/run_csgo_benchmark_v2_map_specific.py"
SPEC = importlib.util.spec_from_file_location(
    "run_csgo_benchmark_v2_map_specific", MODULE_PATH
)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def option(command, name):
    for index, token in enumerate(command):
        if token == name:
            return command[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    raise ValueError(f"option not found: {name}")


class FakeProcess:
    def __init__(self, experiment, completed):
        self.experiment = experiment
        self.completed = completed
        self.pid = 1000

    def poll(self):
        return 0 if self.completed else None


class MapSpecificRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix, cls.config_path = RUNNER.load_matrix()
        RUNNER.validate_configuration(cls.matrix)

    def temporary_matrix(self, root):
        matrix = copy.deepcopy(self.matrix)
        matrix["paths"] = {
            "model_root": str(root / "models"),
            "generation_root": str(root / "generation"),
            "localization_root": str(root / "localization"),
            "log_root": str(root / "logs"),
            "results_file": str(root / "results.md"),
        }
        return matrix

    def test_default_matrix_order_shots_lock_and_family_route_thresholds(self):
        expected = []
        for family in ("exp35", "exp36", "exp36_1"):
            for map_name in RUNNER.CROSSMAP_MAPS:
                for route in RUNNER.FAMILY_ROUTES[family]:
                    expected.append(
                        f"{family}_{map_name}"
                        if route == "joint"
                        else f"{family}_{route}_{map_name}"
                    )
        self.assertEqual(RUNNER.experiment_names(self.matrix), tuple(expected))
        self.assertEqual(len(expected), 28)
        self.assertEqual(RUNNER.scheduler_shots(self.matrix), [100, 50, 20, 10])
        self.assertEqual(RUNNER._shots(self.matrix), [100, 50, 20, 10])
        self.assertEqual(self.matrix["scheduling"]["max_active_pipelines"], 24)
        self.assertEqual(
            self.matrix["scheduling"]["launch_memory_confirmation_seconds"], 60
        )
        self.assertEqual(
            RUNNER.pipeline_lock_path(self.matrix, "exp35_loc_cs_office", 100),
            REPO_ROOT
            / "logs/benchmark_v2_map_specific/exp35_loc_cs_office/shot_100/seed_0/.pipeline.lock",
        )
        expected_memory = {
            ("exp35", "joint"): 45000,
            ("exp35", "gen"): 32000,
            ("exp35", "loc"): 26000,
            ("exp36", "joint"): 35000,
            ("exp36", "gen"): 22000,
            ("exp36", "loc"): 22000,
            ("exp36_1", "joint"): 45000,
        }
        for key, value in expected_memory.items():
            self.assertEqual(RUNNER._minimum_free_memory(self.matrix, *key), value)
        self.assertEqual(
            RUNNER._launch_reservation_memory(self.matrix, "exp36_1", "joint"),
            6000,
        )

    def test_multi_shot_family_route_filters_exclude_joint_jobs(self):
        parser = RUNNER._build_parser()
        parsed = parser.parse_args(
            [
                "schedule",
                "--shots",
                "50",
                "20",
                "10",
                "--families",
                "exp35",
                "exp36",
                "--routes",
                "gen",
                "loc",
            ]
        )
        self.assertEqual(parsed.shots, [50, 20, 10])
        self.assertEqual(parsed.families, ["exp35", "exp36"])
        self.assertEqual(parsed.routes, ["gen", "loc"])
        legacy = parser.parse_args(["schedule", "--shots", "50"])
        self.assertEqual(legacy.shots, [50])

        jobs = RUNNER._job_order(
            self.matrix,
            parsed.shots,
            families=parsed.families,
            routes=parsed.routes,
        )
        self.assertEqual(len(jobs), 2 * 4 * 2 * 3)
        self.assertTrue(all("_gen_" in name or "_loc_" in name for name, _ in jobs))
        self.assertNotIn(("exp35_cs_office", 50), jobs)
        self.assertEqual({shot for _, shot in jobs}, {50, 20, 10})

        output = io.StringIO()
        with (
            patch.object(RUNNER, "run_family_aggregations") as aggregate,
            patch("sys.stdout", output),
        ):
            RUNNER.schedule(
                self.matrix,
                self.config_path,
                cuda_device="0",
                shots=[50, 20],
                families=["exp35"],
                routes=["gen", "loc"],
                dry_run=True,
                jobs_override=[
                    ("exp35_cs_office", 50),
                    ("exp35_gen_cs_office", 50),
                    ("exp35_loc_cs_office", 20),
                ],
            )
        rendered = output.getvalue()
        self.assertNotIn("[exp35_cs_office/50:", rendered)
        self.assertIn("[exp35_gen_cs_office/50:train]", rendered)
        self.assertIn("[exp35_loc_cs_office/20:train]", rendered)
        self.assertEqual([call.args[1] for call in aggregate.call_args_list], [50, 20])
        self.assertTrue(
            all(
                call.kwargs["routes"] == ("gen", "loc")
                for call in aggregate.call_args_list
            )
        )

        status = parser.parse_args(
            [
                "status",
                "--shots",
                "50",
                "20",
                "10",
                "--families",
                "exp35",
                "--routes",
                "loc",
            ]
        )
        self.assertEqual(status.shots, [50, 20, 10])
        self.assertEqual(status.families, ["exp35"])
        self.assertEqual(status.routes, ["loc"])

    def test_exp36_1_is_joint_only_for_all_four_shots(self):
        jobs = RUNNER._job_order(
            self.matrix,
            [100, 50, 20, 10],
            families=["exp36_1"],
            routes=["joint", "gen", "loc"],
        )
        expected_names = [f"exp36_1_{map_name}" for map_name in RUNNER.CROSSMAP_MAPS]
        self.assertEqual(len(jobs), 16)
        self.assertEqual([name for name, shot in jobs if shot == 100], expected_names)
        self.assertEqual({shot for _, shot in jobs}, {100, 50, 20, 10})
        self.assertTrue(all("_gen_" not in name and "_loc_" not in name for name, _ in jobs))

        command = RUNNER.build_train_command(
            self.matrix, "exp36_1_cs_office", 10
        )
        self.assertEqual(
            option(command, "--csgo_config"),
            "csgo_configs/exp36_1_cs_office.yaml",
        )
        self.assertEqual(
            option(command, "--output_dir"),
            "outputs/csgo_1b/exp36_1_cs_office/shot_10/seed_0",
        )
        self.assertEqual(option(command, "--per_device_train_batch_size"), "4")
        self.assertEqual(option(command, "--per_device_eval_batch_size"), "4")
        self.assertEqual(option(command, "--gradient_accumulation_steps"), "32")
        self.assertEqual(option(command, "--max_steps"), "400")
        self.assertEqual(option(command, "--fix_llm"), "True")
        self.assertEqual(option(command, "--master_port"), "30473")
        with self.assertRaisesRegex(RUNNER.PipelineError, "unsupported family route"):
            RUNNER.build_map_models_command(
                self.matrix, "exp36_1", "gen", 100, "discrete"
            )

    def test_schedule_syncs_each_selected_shot_with_filters(self):
        with tempfile.TemporaryDirectory() as temporary:
            matrix = self.temporary_matrix(Path(temporary))
            with (
                patch.object(RUNNER, "pipeline_complete", return_value=True),
                patch.object(RUNNER, "sync_results") as sync_results,
                patch.object(RUNNER, "run_family_aggregations"),
            ):
                RUNNER.schedule(
                    matrix,
                    Path(temporary) / "matrix.yaml",
                    cuda_device="0",
                    shots=[50, 20],
                    families=["exp35"],
                    routes=["gen", "loc"],
                    jobs_override=[("exp35_gen_cs_office", 50)],
                    launch_settle_seconds=0,
                    poll_seconds=0,
                )
            self.assertEqual(
                [
                    (
                        call.kwargs["shots"],
                        call.kwargs["families"],
                        call.kwargs["routes"],
                    )
                    for call in sync_results.call_args_list
                ],
                [(50, ("exp35",), ("gen", "loc")), (20, ("exp35",), ("gen", "loc"))],
            )

    def test_training_command_uses_current_interpreter_and_recorded_route_args(self):
        command = RUNNER.build_train_command(self.matrix, "exp35_loc_cs_office", 100)
        self.assertEqual(
            command[:4],
            [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1"],
        )
        self.assertEqual(option(command, "--master_port"), "30106")
        self.assertEqual(
            option(command, "--csgo_config"), "csgo_configs/exp35_loc_cs_office.yaml"
        )
        self.assertEqual(
            option(command, "--output_dir"),
            "outputs/csgo_1b/exp35_loc_cs_office/shot_100/seed_0",
        )
        self.assertEqual(option(command, "--per_device_train_batch_size"), "100")
        self.assertEqual(option(command, "--per_device_eval_batch_size"), "128")
        self.assertEqual(option(command, "--gradient_accumulation_steps"), "1")
        self.assertEqual(option(command, "--max_steps"), "400")
        self.assertEqual(option(command, "--eval_strategy"), "no")
        self.assertEqual(option(command, "--save_strategy"), "no")
        self.assertEqual(option(command, "--save_steps"), "4000")
        self.assertEqual(option(command, "--save_total_limit"), "5")

        joint = RUNNER.build_train_command(self.matrix, "exp36_cs_office", 50)
        self.assertEqual(option(joint, "--per_device_train_batch_size"), "4")
        self.assertEqual(option(joint, "--gradient_accumulation_steps"), "32")
        self.assertEqual(option(joint, "--lora_r"), "32")
        self.assertEqual(option(joint, "--lora_alpha"), "64")
        self.assertEqual(option(joint, "--master_port"), "30237")

    def test_minimal_asset_switch_routes_train_infer_loc_and_metric(self):
        matrix = copy.deepcopy(self.matrix)
        matrix["benchmark_v2_asset_manifest"] = (
            "data/csgo_benchmark_v2/minimal_dataset_report.json"
        )
        matrix["evaluation"].pop("data_dir")
        RUNNER.validate_configuration(matrix)
        commands = [
            RUNNER.build_train_command(matrix, "exp35_gen_cs_office", 100),
            RUNNER.build_generation_inference_command(
                matrix, "exp35_gen_cs_office", 100, "discrete"
            ),
            RUNNER.build_localization_inference_command(
                matrix, "exp35_loc_cs_office", 100
            ),
            RUNNER.build_metric_command(
                matrix, "exp35_gen_cs_office", 100, "discrete", "cs_office"
            ),
        ]
        for command in commands:
            self.assertEqual(
                option(command, "--benchmark_v2_asset_manifest"),
                "data/csgo_benchmark_v2/minimal_dataset_report.json",
            )
        self.assertEqual(
            option(commands[-1], "--gt"),
            "data/csgo_benchmark_v2/images/cs_office",
        )
        self.assertNotIn("--data_dir", commands[-1])

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

    def test_commands_keep_exact_map_split_checkpoint_and_provenance_contract(self):
        generation = RUNNER.build_generation_inference_command(
            self.matrix, "exp35_gen_cs_office", 100, "continuous", "crossmap"
        )
        self.assertEqual(generation[0], sys.executable)
        self.assertEqual(
            option(generation, "--csgo_config"),
            "csgo_configs/test/exp35_gen_cs_office_gen_conti.yaml",
        )
        self.assertEqual(
            option(generation, "--output_dir"),
            "outputs_eval/benchmark_v2/exp35_gen_cs_office/shot_100/seed_0/continuous",
        )
        self.assertEqual(
            option(generation, "--ckpt_path"),
            "outputs/csgo_1b/exp35_gen_cs_office/shot_100/seed_0/model.safetensors",
        )
        self.assertEqual(option(generation, "--seed"), "42")
        self.assertEqual(
            option(generation, "--benchmark_v2_split"), "crossmap_continuous"
        )
        self.assertEqual(option(generation, "--benchmark_v2_support_seed"), "0")
        self.assertEqual(option(generation, "--benchmark_v2_shots_per_map"), "100")
        self.assertEqual(
            generation[generation.index("--benchmark_v2_maps") + 1 :], ["cs_office"]
        )

        localization = RUNNER.build_localization_inference_command(
            self.matrix, "exp35_loc_cs_office", 100, "seen"
        )
        self.assertEqual(
            option(localization, "--output_dir"),
            "outputs_loc/benchmark_v2/exp35_loc_cs_office/shot_100/seed_0/seen_retention",
        )
        self.assertEqual(
            option(localization, "--benchmark_v2_split"), "seen_discrete_test"
        )
        maps_index = localization.index("--benchmark_v2_maps")
        self.assertEqual(localization[maps_index + 1 :], list(RUNNER.SEEN_MAPS))

        metric = RUNNER.build_metric_command(
            self.matrix, "exp35_gen_cs_office", 100, "discrete", "cs_office"
        )
        self.assertEqual(metric[1], "benchmark_csgo_v1.py")
        self.assertEqual(
            option(metric, "--pred"),
            "outputs_eval/benchmark_v2/exp35_gen_cs_office/shot_100/seed_0/discrete/gen_imgs/cs_office",
        )
        self.assertEqual(
            option(metric, "--benchmark_v2_manifest"),
            "data/csgo_benchmark_v2/benchmark_manifest.json",
        )
        self.assertEqual(
            option(metric, "--external_loc_checkpoint_path"),
            "checkpoints_reg_newdata/exp5_2/20251227_091745/current_model.pth",
        )

        map_models = RUNNER.build_map_models_command(
            self.matrix, "exp35", "gen", 100, "continuous"
        )
        self.assertEqual(map_models[2], "map-models")
        self.assertIn(
            "outputs_eval/benchmark_v2/exp35_gen_{map}/shot_100/seed_0/continuous/benchmark_csgo_v2_conti_{map}.json",
            option(map_models, "--input_pattern"),
        )
        self.assertEqual(
            option(map_models, "--output"),
            "outputs_eval/benchmark_v2/exp35_gen/shot_100/seed_0/map_models/benchmark_v2_continuous_crossmap.json",
        )

        expected_payload = RUNNER._expected_inference_payload(
            self.matrix, "exp35_gen_cs_office", 100, "discrete", "crossmap"
        )
        self.assertEqual(expected_payload["maps"], ["cs_office"])
        self.assertEqual(expected_payload["sample_count"], 2000)
        self.assertEqual(expected_payload["seed"], 42)
        self.assertEqual(expected_payload["benchmark_v2_support_seed"], 0)

    def test_pipeline_stage_order_serializes_all_applicable_work(self):
        generation_labels = [
            stage["label"]
            for stage in RUNNER.pipeline_stages(self.matrix, "exp35_gen_cs_office", 100)
        ]
        self.assertEqual(generation_labels[0], "train")
        self.assertEqual(
            generation_labels[1:5],
            [
                "generation_discrete_crossmap_inference",
                "generation_continuous_crossmap_inference",
                "generation_discrete_seen_inference",
                "generation_continuous_seen_inference",
            ],
        )
        self.assertEqual(
            generation_labels[5], "generation_discrete_crossmap_metric_cs_office"
        )
        self.assertEqual(
            generation_labels[6], "generation_continuous_crossmap_metric_cs_office"
        )
        self.assertLess(
            generation_labels.index("generation_discrete_seen_aggregate"),
            generation_labels.index("generation_continuous_seen_metric_cs_agency"),
        )
        self.assertEqual(
            [
                stage["label"]
                for stage in RUNNER.pipeline_stages(
                    self.matrix, "exp35_loc_cs_office", 100
                )
            ],
            [
                "train",
                "localization_crossmap_inference_metric",
                "localization_seen_inference_metric",
            ],
        )

    def test_parent_and_memory_scan_skips_pending_joint_and_finds_later_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            matrix = self.temporary_matrix(Path(temporary))
            pending = [
                ("exp35_cs_office", 100),
                ("exp35_gen_cs_office", 100),
            ]
            with (
                patch.object(RUNNER, "_lock_is_held", return_value=False),
                patch.object(
                    RUNNER,
                    "_parent_ready",
                    side_effect=lambda _matrix, experiment: (
                        experiment == "exp35_gen_cs_office"
                    ),
                ),
            ):
                selected = RUNNER.select_launch_candidate(
                    matrix,
                    pending,
                    "0",
                    free_memory_fn=lambda _device: 33000,
                    launch_memory_confirmation_seconds=0,
                )
            self.assertEqual(selected, (("exp35_gen_cs_office", 100), 33000))

    def test_launch_memory_confirmation_skips_when_first_sample_is_below_threshold(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            matrix = self.temporary_matrix(Path(temporary))
            sleeps = []
            with (
                patch.object(RUNNER, "_lock_is_held", return_value=False),
                patch.object(RUNNER, "_parent_ready", return_value=True),
            ):
                selected = RUNNER.select_launch_candidate(
                    matrix,
                    [("exp35_loc_cs_office", 100)],
                    "0",
                    free_memory_fn=lambda _device: 25999,
                    launch_memory_confirmation_seconds=60,
                    sleep_fn=sleeps.append,
                )
            self.assertIsNone(selected)
            self.assertEqual(sleeps, [])

    def test_launch_memory_confirmation_skips_when_second_sample_drops(self):
        with tempfile.TemporaryDirectory() as temporary:
            matrix = self.temporary_matrix(Path(temporary))
            samples = iter((26000, 25999))
            sleeps = []
            with (
                patch.object(RUNNER, "_lock_is_held", return_value=False),
                patch.object(RUNNER, "_parent_ready", return_value=True),
            ):
                selected = RUNNER.select_launch_candidate(
                    matrix,
                    [("exp35_loc_cs_office", 100)],
                    "0",
                    free_memory_fn=lambda _device: next(samples),
                    launch_memory_confirmation_seconds=60,
                    sleep_fn=sleeps.append,
                )
            self.assertIsNone(selected)
            self.assertEqual(sleeps, [60.0])

    def test_launch_memory_confirmation_launches_when_samples_are_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            matrix = self.temporary_matrix(Path(temporary))
            samples = iter((26000, 26000))
            sleeps = []
            with (
                patch.object(RUNNER, "_lock_is_held", return_value=False),
                patch.object(RUNNER, "_parent_ready", return_value=True),
            ):
                selected = RUNNER.select_launch_candidate(
                    matrix,
                    [("exp35_loc_cs_office", 100)],
                    "0",
                    free_memory_fn=lambda _device: next(samples),
                    launch_memory_confirmation_seconds=60,
                    sleep_fn=sleeps.append,
                )
            self.assertEqual(selected, (("exp35_loc_cs_office", 100), 26000))
            self.assertEqual(sleeps, [60.0])

    def test_held_lock_consumes_only_slot_then_requeues_same_job_at_queue_head(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = self.temporary_matrix(root)
            matrix["scheduling"]["max_active_pipelines"] = 1
            held = ("exp35_loc_cs_office", 100)
            later = ("exp35_gen_cs_office", 100)
            lock_path = RUNNER.pipeline_lock_path(matrix, *held)
            with RUNNER._exclusive_lock(lock_path):
                pending, external = RUNNER.partition_pending_jobs(matrix, [held, later])
                self.assertEqual(pending, [later])
                self.assertEqual(external, {held})
                self.assertTrue(RUNNER._lock_is_held(lock_path))
            self.assertFalse(RUNNER._lock_is_held(lock_path))

            held_lock = {held: True}
            completed = {held: False, later: False}
            launches = []
            max_active = 0
            active = 0
            sleep_calls = 0

            def lock_state(path):
                return held_lock.get(held, False) if path == lock_path else False

            def fake_pipeline_complete(_matrix, experiment, shots):
                return completed[(experiment, shots)]

            def spawn(command, **_kwargs):
                nonlocal active, max_active
                experiment = command[command.index("--experiment") + 1]
                job = (experiment, int(command[command.index("--shots") + 1]))
                launches.append(job)
                active += 1
                max_active = max(max_active, active)

                class ImmediateProcess:
                    pid = 2000

                    def poll(self_inner):
                        nonlocal active
                        active -= 1
                        completed[job] = True
                        return 0

                return ImmediateProcess()

            def sleep(_seconds):
                nonlocal sleep_calls
                sleep_calls += 1
                if sleep_calls == 1:
                    held_lock[held] = False

            with (
                patch.object(RUNNER, "_lock_is_held", side_effect=lock_state),
                patch.object(RUNNER, "_parent_ready", return_value=True),
                patch.object(RUNNER, "sync_results") as sync_results,
                patch.object(RUNNER, "run_family_aggregations"),
                patch.object(
                    RUNNER, "pipeline_complete", side_effect=fake_pipeline_complete
                ),
            ):
                RUNNER.schedule(
                    matrix,
                    root / "matrix.yaml",
                    cuda_device="0",
                    shots=[100],
                    launch_settle_seconds=0,
                    poll_seconds=0,
                    jobs_override=[held, later],
                    free_memory_fn=lambda _device: 100000,
                    popen_factory=spawn,
                    sleep_fn=sleep,
                )

            self.assertEqual(launches, [held, later])
            self.assertEqual(max_active, 1)
            self.assertGreaterEqual(sync_results.call_count, 1)
            self.assertTrue(sync_results.call_args_list[0].kwargs["initialize"])

    def test_scheduler_fills_multiple_slots_and_skips_blocked_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = self.temporary_matrix(root)
            matrix["scheduling"]["max_active_pipelines"] = 2
            blocked = ("exp35_cs_office", 100)
            runnable_gen = ("exp35_gen_cs_office", 100)
            runnable_loc = ("exp35_loc_cs_office", 100)
            jobs = [blocked, runnable_gen, runnable_loc]
            launched = []
            processes = []
            completed = {job: False for job in jobs}

            class HeldProcess:
                pid = 3000

                def __init__(self):
                    self.done = False

                def poll(self):
                    return 0 if self.done else None

            def spawn(command, **_kwargs):
                experiment = command[command.index("--experiment") + 1]
                launched.append(experiment)
                process = HeldProcess()
                processes.append(process)
                return process

            def sleep(_seconds):
                for process in processes:
                    process.done = True
                for experiment in launched:
                    completed[(experiment, 100)] = True

            def parent_ready(_matrix, experiment):
                # The first scan must skip the blocked joint job; once the
                # lower-memory jobs have occupied and released both slots,
                # make its parent available so the synthetic queue drains.
                return experiment != blocked[0] or len(launched) >= 2

            with (
                patch.object(RUNNER, "_lock_is_held", return_value=False),
                patch.object(
                    RUNNER,
                    "_parent_ready",
                    side_effect=parent_ready,
                ),
                patch.object(
                    RUNNER,
                    "pipeline_complete",
                    side_effect=lambda _matrix, experiment, shots: completed[
                        (experiment, shots)
                    ],
                ),
                patch.object(RUNNER, "sync_results"),
                patch.object(RUNNER, "run_family_aggregations"),
                patch.object(RUNNER.time, "monotonic", return_value=100.0),
            ):
                RUNNER.schedule(
                    matrix,
                    root / "matrix.yaml",
                    cuda_device="0",
                    shots=[100],
                    launch_settle_seconds=0,
                    poll_seconds=0,
                    jobs_override=jobs,
                    free_memory_fn=lambda _device: 100000,
                    popen_factory=spawn,
                    sleep_fn=sleep,
                )

            self.assertEqual(launched[:2], [runnable_gen[0], runnable_loc[0]])
            self.assertEqual(launched[2:], [blocked[0]])
            self.assertEqual(len(processes), 3)

    def test_positive_max_active_pipeline_validation_and_zero_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            matrix = self.temporary_matrix(Path(temporary))
            matrix["scheduling"]["max_active_pipelines"] = 3
            self.assertEqual(RUNNER._max_active_pipelines(matrix), 3)
            matrix["scheduling"]["max_active_pipelines"] = 0
            with self.assertRaises(RUNNER.PipelineError):
                RUNNER._max_active_pipelines(matrix)

    def test_scheduler_skips_routes_above_live_memory_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = self.temporary_matrix(root)
            matrix["scheduling"]["max_active_pipelines"] = 1
            jobs = [
                ("exp35_cs_office", 100),
                ("exp35_gen_cs_office", 100),
                ("exp35_loc_cs_office", 100),
            ]
            launched = []
            processes = []
            completed = {job: False for job in jobs}

            class HeldProcess:
                pid = 4000

                def __init__(self):
                    self.done = False

                def poll(self):
                    return 0 if self.done else None

            def spawn(command, **_kwargs):
                experiment = command[command.index("--experiment") + 1]
                launched.append(experiment)
                process = HeldProcess()
                processes.append(process)
                return process

            def sleep(_seconds):
                for process in processes:
                    process.done = True
                for experiment in launched:
                    completed[(experiment, 100)] = True

            def free_memory(_device):
                # Only the loc route fits initially; after it finishes, all
                # routes fit and the synthetic queue can drain.
                return 30000 if not completed[jobs[2]] else 50000

            with (
                patch.object(RUNNER, "_lock_is_held", return_value=False),
                patch.object(RUNNER, "_parent_ready", return_value=True),
                patch.object(
                    RUNNER,
                    "pipeline_complete",
                    side_effect=lambda _matrix, experiment, shots: completed[
                        (experiment, shots)
                    ],
                ),
                patch.object(RUNNER, "sync_results"),
                patch.object(RUNNER, "run_family_aggregations"),
                patch.object(RUNNER.time, "monotonic", return_value=100.0),
            ):
                RUNNER.schedule(
                    matrix,
                    root / "matrix.yaml",
                    cuda_device="0",
                    shots=[100],
                    launch_settle_seconds=0,
                    poll_seconds=0,
                    jobs_override=jobs,
                    free_memory_fn=free_memory,
                    popen_factory=spawn,
                    sleep_fn=sleep,
                )

            self.assertEqual(launched[0], "exp35_loc_cs_office")

    def test_train_reservation_blocks_second_train_while_gpu_is_invisible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = self.temporary_matrix(root)
            matrix["scheduling"]["max_active_pipelines"] = 3
            matrix["scheduling"]["launch_memory_confirmation_seconds"] = 0
            jobs = [
                ("exp35_gen_cs_office", 100),
                ("exp35_loc_cs_office", 100),
                ("exp36_gen_cs_office", 100),
            ]
            completed = {job: False for job in jobs}
            processes = []
            launches = []
            max_live = 0

            class StartupProcess:
                next_pid = 5000

                def __init__(self, job):
                    self.job = job
                    self.pid = StartupProcess.next_pid
                    StartupProcess.next_pid += 1
                    self.done = False
                    self.scheduler_turns = 0

                def poll(self):
                    return 0 if self.done else None

            def spawn(command, **_kwargs):
                nonlocal max_live
                job = (
                    command[command.index("--experiment") + 1],
                    int(command[command.index("--shots") + 1]),
                )
                process = StartupProcess(job)
                processes.append(process)
                launches.append(job)
                max_live = max(max_live, sum(not item.done for item in processes))
                return process

            def sleep(_seconds):
                # The first child remains in CPU/W&B startup for two scheduler
                # turns.  No GPU bytes become visible during that interval.
                for process in processes:
                    if not process.done:
                        process.scheduler_turns += 1
                        if process.scheduler_turns >= 2:
                            process.done = True
                            completed[process.job] = True
                        break

            with (
                patch.object(RUNNER, "_parent_ready", return_value=True),
                patch.object(
                    RUNNER,
                    "pipeline_complete",
                    side_effect=lambda _matrix, experiment, shots: completed[
                        (experiment, shots)
                    ],
                ),
                patch.object(RUNNER, "sync_results"),
                patch.object(RUNNER, "run_family_aggregations"),
            ):
                RUNNER.schedule(
                    matrix,
                    root / "matrix.yaml",
                    cuda_device="0",
                    shots=[100],
                    jobs_override=jobs,
                    launch_settle_seconds=0,
                    launch_memory_confirmation_seconds=0,
                    poll_seconds=0,
                    free_memory_fn=lambda _device: 100000,
                    process_gpu_memory_fn=lambda _device, _pid: 0,
                    popen_factory=spawn,
                    sleep_fn=sleep,
                )

            self.assertEqual(launches, jobs)
            self.assertEqual(max_live, 1)

    def test_confirmed_reservation_uses_remaining_target_when_memory_fits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = self.temporary_matrix(root)
            matrix["scheduling"]["max_active_pipelines"] = 2
            matrix["scheduling"]["launch_memory_confirmation_seconds"] = 0
            matrix["scheduling"]["reservation_release_stable_samples"] = 2
            jobs = [
                ("exp35_loc_cs_office", 100),
                ("exp36_loc_cs_office", 100),
            ]
            completed = {job: False for job in jobs}
            processes = []
            launches = []
            memory_samples = []
            max_live = 0

            class ResidentProcess:
                def __init__(self, job):
                    self.job = job
                    self.pid = (
                        RUNNER.os.getpid() if job == jobs[0] else RUNNER.os.getpid() + 1
                    )
                    self.done = False

                def poll(self):
                    return 0 if self.done else None

            def spawn(command, **_kwargs):
                nonlocal max_live
                job = (
                    command[command.index("--experiment") + 1],
                    int(command[command.index("--shots") + 1]),
                )
                process = ResidentProcess(job)
                processes.append(process)
                launches.append(job)
                max_live = max(max_live, sum(not item.done for item in processes))
                return process

            def process_memory(_device, pid):
                memory_samples.append(pid)
                return 2048 if pid == processes[0].pid else 0

            def sleep(_seconds):
                if len(processes) >= 2:
                    for process in processes:
                        if not process.done:
                            process.done = True
                            completed[process.job] = True

            with (
                patch.object(RUNNER, "_parent_ready", return_value=True),
                patch.object(
                    RUNNER,
                    "pipeline_complete",
                    side_effect=lambda _matrix, experiment, shots: completed[
                        (experiment, shots)
                    ],
                ),
                patch.object(RUNNER, "sync_results"),
                patch.object(RUNNER, "run_family_aggregations"),
            ):
                RUNNER.schedule(
                    matrix,
                    root / "matrix.yaml",
                    cuda_device="0",
                    shots=[100],
                    jobs_override=jobs,
                    launch_settle_seconds=0,
                    launch_memory_confirmation_seconds=0,
                    poll_seconds=0,
                    # 50000 - (26000 - 2048) is enough for the second loc
                    # pipeline, while the full startup target would not be.
                    free_memory_fn=lambda _device: 50000,
                    process_gpu_memory_fn=process_memory,
                    popen_factory=spawn,
                    sleep_fn=sleep,
                )

            self.assertEqual(launches, jobs)
            self.assertGreaterEqual(memory_samples.count(processes[0].pid), 2)
            self.assertEqual(max_live, 2)

    def test_pipeline_completion_releases_reservation_for_next_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = self.temporary_matrix(root)
            matrix["scheduling"]["max_active_pipelines"] = 2
            matrix["scheduling"]["launch_memory_confirmation_seconds"] = 0
            matrix["scheduling"]["reservation_release_stable_samples"] = 2
            jobs = [
                ("exp35_loc_cs_office", 100),
                ("exp36_loc_cs_office", 100),
            ]
            completed = {job: False for job in jobs}
            processes = []
            launches = []
            sleep_calls = 0
            max_live = 0

            class MemoryBoundProcess:
                def __init__(self, job):
                    self.job = job
                    self.pid = RUNNER.os.getpid() + len(processes)
                    self.done = False

                def poll(self):
                    return 0 if self.done else None

            def spawn(command, **_kwargs):
                nonlocal max_live
                job = (
                    command[command.index("--experiment") + 1],
                    int(command[command.index("--shots") + 1]),
                )
                process = MemoryBoundProcess(job)
                processes.append(process)
                launches.append(job)
                max_live = max(max_live, sum(not item.done for item in processes))
                return process

            def free_memory(_device):
                if not processes or processes[0].done:
                    return 45000
                return 21000

            def process_memory(_device, pid):
                if processes and pid == processes[0].pid:
                    return 2048
                return 0

            def sleep(_seconds):
                nonlocal sleep_calls
                sleep_calls += 1
                if sleep_calls == 3:
                    processes[0].done = True
                    completed[processes[0].job] = True
                elif len(processes) >= 2:
                    processes[1].done = True
                    completed[processes[1].job] = True

            with (
                patch.object(RUNNER, "_parent_ready", return_value=True),
                patch.object(
                    RUNNER,
                    "pipeline_complete",
                    side_effect=lambda _matrix, experiment, shots: completed[
                        (experiment, shots)
                    ],
                ),
                patch.object(RUNNER, "sync_results"),
                patch.object(RUNNER, "run_family_aggregations"),
            ):
                RUNNER.schedule(
                    matrix,
                    root / "matrix.yaml",
                    cuda_device="0",
                    shots=[100],
                    jobs_override=jobs,
                    launch_settle_seconds=0,
                    launch_memory_confirmation_seconds=0,
                    poll_seconds=0,
                    free_memory_fn=free_memory,
                    process_gpu_memory_fn=process_memory,
                    popen_factory=spawn,
                    sleep_fn=sleep,
                )

            self.assertEqual(launches, jobs)
            self.assertEqual(max_live, 1)

    def test_adopted_inference_and_metric_hold_target_until_release(self):
        for external_phase in (RUNNER.INFERENCE_PHASE, RUNNER.METRIC_PHASE):
            with self.subTest(external_phase=external_phase):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    matrix = self.temporary_matrix(root)
                    matrix["scheduling"]["max_active_pipelines"] = 2
                    held = ("exp36_loc_cs_office", 100)
                    later = ("exp36_gen_cs_office", 100)
                    completed = {held: False, later: False}
                    held_lock = True
                    sleep_calls = 0
                    launches = []
                    lock_state_at_launch = []
                    held_lock_path = RUNNER.pipeline_lock_path(matrix, *held)

                    class AdoptedProcess:
                        pid = 9000

                        def poll(self):
                            return 0

                    def lock_state(path):
                        return held_lock if path == held_lock_path else False

                    def phase_from_artifacts(_matrix, experiment, shots):
                        if (experiment, shots) == held:
                            return external_phase
                        return RUNNER.TRAINING_PHASE

                    def spawn(command, **_kwargs):
                        job = (
                            command[command.index("--experiment") + 1],
                            int(command[command.index("--shots") + 1]),
                        )
                        launches.append(job)
                        lock_state_at_launch.append(held_lock)
                        completed[job] = True
                        return AdoptedProcess()

                    def sleep(_seconds):
                        nonlocal held_lock, sleep_calls
                        sleep_calls += 1
                        if sleep_calls >= 3:
                            held_lock = False
                            completed[held] = True

                    with (
                        patch.object(RUNNER, "_lock_is_held", side_effect=lock_state),
                        patch.object(RUNNER, "_pipeline_owner_pid", return_value=0),
                        patch.object(
                            RUNNER,
                            "_phase_from_next_pending_stage",
                            side_effect=phase_from_artifacts,
                        ),
                        patch.object(RUNNER, "_parent_ready", return_value=True),
                        patch.object(
                            RUNNER,
                            "pipeline_complete",
                            side_effect=lambda _matrix, experiment, shots: completed[
                                (experiment, shots)
                            ],
                        ),
                        patch.object(RUNNER, "sync_results"),
                        patch.object(RUNNER, "run_family_aggregations"),
                    ):
                        RUNNER.schedule(
                            matrix,
                            root / "matrix.yaml",
                            cuda_device="0",
                            shots=[100],
                            jobs_override=[held, later],
                            launch_settle_seconds=0,
                            launch_memory_confirmation_seconds=0,
                            poll_seconds=0,
                            # Exactly enough for the pending gen train only
                            # after the adopted loc target is released.
                            free_memory_fn=lambda _device: 22000,
                            process_gpu_memory_fn=lambda _device, _pid: 0,
                            popen_factory=spawn,
                            sleep_fn=sleep,
                        )

                    self.assertEqual(launches, [later])
                    self.assertEqual(lock_state_at_launch, [False])
                    self.assertGreaterEqual(sleep_calls, 3)

    def test_metric_reservation_waits_for_stable_visibility_before_growth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = self.temporary_matrix(root)
            matrix["scheduling"]["max_active_pipelines"] = 2
            matrix["scheduling"]["reservation_release_stable_samples"] = 2
            held = ("exp36_loc_cs_office", 100)
            later = ("exp36_gen_cs_office", 100)
            completed = {held: False, later: False}
            held_lock = True
            sleep_calls = 0
            launches = []
            launch_turns = []
            observed = iter((1072, 1072, 22000))
            held_lock_path = RUNNER.pipeline_lock_path(matrix, *held)

            class ChildProcess:
                pid = 9100

                def poll(self):
                    return 0

            def lock_state(path):
                return held_lock if path == held_lock_path else False

            def phase_from_artifacts(_matrix, experiment, shots):
                if (experiment, shots) == held:
                    return RUNNER.METRIC_PHASE
                return RUNNER.TRAINING_PHASE

            def process_memory(_device, pid):
                if pid == 4242:
                    return next(observed)
                return 0

            def spawn(command, **_kwargs):
                job = (
                    command[command.index("--experiment") + 1],
                    int(command[command.index("--shots") + 1]),
                )
                launches.append(job)
                launch_turns.append(sleep_calls)
                completed[job] = True
                return ChildProcess()

            def sleep(_seconds):
                nonlocal held_lock, sleep_calls
                sleep_calls += 1
                if later in launches:
                    held_lock = False
                    completed[held] = True

            with (
                patch.object(RUNNER, "_lock_is_held", side_effect=lock_state),
                patch.object(RUNNER, "_pipeline_owner_pid", return_value=4242),
                patch.object(
                    RUNNER,
                    "_phase_from_next_pending_stage",
                    side_effect=phase_from_artifacts,
                ),
                patch.object(RUNNER, "_parent_ready", return_value=True),
                patch.object(
                    RUNNER,
                    "pipeline_complete",
                    side_effect=lambda _matrix, experiment, shots: completed[
                        (experiment, shots)
                    ],
                ),
                patch.object(RUNNER, "sync_results"),
                patch.object(RUNNER, "run_family_aggregations"),
            ):
                RUNNER.schedule(
                    matrix,
                    root / "matrix.yaml",
                    cuda_device="0",
                    shots=[100],
                    jobs_override=[held, later],
                    launch_settle_seconds=0,
                    launch_memory_confirmation_seconds=0,
                    poll_seconds=0,
                    # At 1072 MB the remaining reservation is 20928 MB;
                    # only the later 22000 MB observation permits launch.
                    free_memory_fn=lambda _device: 22000,
                    process_gpu_memory_fn=process_memory,
                    popen_factory=spawn,
                    sleep_fn=sleep,
                )

            self.assertEqual(launches, [later])
            self.assertEqual(launch_turns, [2])
            self.assertGreaterEqual(sleep_calls, 2)

    def test_training_to_metric_memory_drop_restores_remaining_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = self.temporary_matrix(root)
            matrix["scheduling"]["max_active_pipelines"] = 2
            matrix["scheduling"]["reservation_release_stable_samples"] = 2
            held = ("exp36_loc_cs_office", 100)
            later = ("exp36_gen_cs_office", 100)
            completed = {held: False, later: False}
            held_lock = True
            sleep_calls = 0
            phase_calls = 0
            launches = []
            launch_turns = []
            observed = iter((22000, 22000, 0))
            held_lock_path = RUNNER.pipeline_lock_path(matrix, *held)

            class ChildProcess:
                pid = 9200

                def poll(self):
                    return 0

            def lock_state(path):
                return held_lock if path == held_lock_path else False

            def phase_from_artifacts(_matrix, experiment, shots):
                return RUNNER.TRAINING_PHASE

            def pipeline_phase(*_args, **_kwargs):
                nonlocal phase_calls
                phase_calls += 1
                return (
                    RUNNER.TRAINING_PHASE if phase_calls == 1 else RUNNER.METRIC_PHASE
                )

            def process_memory(_device, pid):
                if pid == 4242:
                    return next(observed)
                return 0

            def parent_ready(_matrix, experiment):
                return experiment != later[0] or sleep_calls >= 2

            def spawn(command, **_kwargs):
                job = (
                    command[command.index("--experiment") + 1],
                    int(command[command.index("--shots") + 1]),
                )
                launches.append(job)
                launch_turns.append(sleep_calls)
                completed[job] = True
                return ChildProcess()

            def sleep(_seconds):
                nonlocal held_lock, sleep_calls
                sleep_calls += 1
                if sleep_calls >= 3:
                    held_lock = False
                    completed[held] = True

            with (
                patch.object(RUNNER, "_lock_is_held", side_effect=lock_state),
                patch.object(RUNNER, "_pipeline_owner_pid", return_value=4242),
                patch.object(
                    RUNNER,
                    "_phase_from_next_pending_stage",
                    side_effect=phase_from_artifacts,
                ),
                patch.object(RUNNER, "_pipeline_phase", side_effect=pipeline_phase),
                patch.object(RUNNER, "_parent_ready", side_effect=parent_ready),
                patch.object(
                    RUNNER,
                    "pipeline_complete",
                    side_effect=lambda _matrix, experiment, shots: completed[
                        (experiment, shots)
                    ],
                ),
                patch.object(RUNNER, "sync_results"),
                patch.object(RUNNER, "run_family_aggregations"),
            ):
                RUNNER.schedule(
                    matrix,
                    root / "matrix.yaml",
                    cuda_device="0",
                    shots=[100],
                    jobs_override=[held, later],
                    launch_settle_seconds=0,
                    launch_memory_confirmation_seconds=0,
                    poll_seconds=0,
                    # The 22000 MB reading temporarily removes the remainder;
                    # the following zero reading must restore the full target.
                    free_memory_fn=lambda _device: 22000,
                    process_gpu_memory_fn=process_memory,
                    popen_factory=spawn,
                    sleep_fn=sleep,
                )

            self.assertEqual(launches, [later])
            self.assertEqual(launch_turns, [3])
            self.assertGreaterEqual(phase_calls, 2)
            self.assertGreaterEqual(sleep_calls, 3)

    def test_legacy_training_lock_without_pid_blocks_until_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = self.temporary_matrix(root)
            matrix["scheduling"]["max_active_pipelines"] = 2
            held = ("exp35_loc_cs_office", 100)
            later = ("exp35_gen_cs_office", 100)
            completed = {held: False, later: False}
            held_lock = True
            launches = []
            lock_state_at_launch = []
            held_lock_path = RUNNER.pipeline_lock_path(matrix, *held)

            class AdoptedProcess:
                pid = 9000

                def poll(self):
                    return 0

            def lock_state(path):
                return held_lock if path == held_lock_path else False

            def spawn(command, **_kwargs):
                job = (
                    command[command.index("--experiment") + 1],
                    int(command[command.index("--shots") + 1]),
                )
                launches.append(job)
                lock_state_at_launch.append(held_lock)
                completed[job] = True
                return AdoptedProcess()

            def sleep(_seconds):
                nonlocal held_lock
                held_lock = False
                completed[held] = True

            with (
                patch.object(RUNNER, "_lock_is_held", side_effect=lock_state),
                patch.object(RUNNER, "_pipeline_owner_pid", return_value=0),
                patch.object(
                    RUNNER,
                    "_phase_from_next_pending_stage",
                    return_value=RUNNER.TRAINING_PHASE,
                ),
                patch.object(RUNNER, "_parent_ready", return_value=True),
                patch.object(
                    RUNNER,
                    "pipeline_complete",
                    side_effect=lambda _matrix, experiment, shots: completed[
                        (experiment, shots)
                    ],
                ),
                patch.object(RUNNER, "sync_results"),
                patch.object(RUNNER, "run_family_aggregations"),
            ):
                RUNNER.schedule(
                    matrix,
                    root / "matrix.yaml",
                    cuda_device="0",
                    shots=[100],
                    jobs_override=[held, later],
                    launch_settle_seconds=0,
                    launch_memory_confirmation_seconds=0,
                    poll_seconds=0,
                    free_memory_fn=lambda _device: 100000,
                    process_gpu_memory_fn=lambda _device, _pid: 0,
                    popen_factory=spawn,
                    sleep_fn=sleep,
                )

            self.assertEqual(launches, [later])
            self.assertEqual(lock_state_at_launch, [False])

    def test_dynamic_held_lock_adoption_initializes_startup_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = self.temporary_matrix(root)
            matrix["scheduling"]["max_active_pipelines"] = 2
            matrix["scheduling"]["reservation_release_stable_samples"] = 2
            held = ("exp35_loc_cs_office", 100)
            later = ("exp36_loc_cs_office", 100)
            completed = {held: False, later: False}
            held_lock = True
            sleep_calls = 0
            launches = []
            launch_turns = []
            observed_pids = []
            held_lock_path = RUNNER.pipeline_lock_path(matrix, *held)

            class AdoptedProcess:
                pid = 9000

                def poll(self):
                    return 0

            def lock_state(path):
                return held_lock if path == held_lock_path else False

            def process_memory(_device, pid):
                observed_pids.append(pid)
                return 2048

            def spawn(command, **_kwargs):
                job = (
                    command[command.index("--experiment") + 1],
                    int(command[command.index("--shots") + 1]),
                )
                launches.append(job)
                launch_turns.append(sleep_calls)
                completed[job] = True
                return AdoptedProcess()

            def sleep(_seconds):
                nonlocal held_lock, sleep_calls
                sleep_calls += 1
                if sleep_calls >= 3:
                    held_lock = False
                    completed[held] = True

            with (
                patch.object(
                    RUNNER,
                    "partition_pending_jobs",
                    return_value=([held, later], set()),
                ),
                patch.object(RUNNER, "_lock_is_held", side_effect=lock_state),
                patch.object(RUNNER, "_pipeline_owner_pid", return_value=4242),
                patch.object(
                    RUNNER,
                    "_phase_from_next_pending_stage",
                    return_value=RUNNER.TRAINING_PHASE,
                ),
                patch.object(RUNNER, "_parent_ready", return_value=True),
                patch.object(
                    RUNNER,
                    "pipeline_complete",
                    side_effect=lambda _matrix, experiment, shots: completed[
                        (experiment, shots)
                    ],
                ),
                patch.object(RUNNER, "sync_results"),
                patch.object(RUNNER, "run_family_aggregations"),
            ):
                RUNNER.schedule(
                    matrix,
                    root / "matrix.yaml",
                    cuda_device="0",
                    shots=[100],
                    jobs_override=[held, later],
                    launch_settle_seconds=0,
                    launch_memory_confirmation_seconds=0,
                    poll_seconds=0,
                    free_memory_fn=lambda _device: 100000,
                    process_gpu_memory_fn=process_memory,
                    popen_factory=spawn,
                    sleep_fn=sleep,
                )

            self.assertEqual(launches, [later])
            self.assertEqual(launch_turns, [2])
            self.assertEqual(observed_pids[:2], [4242, 4242])

    def test_scheduler_retries_failed_pipeline_and_resumes_successfully(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = self.temporary_matrix(root)
            matrix["scheduling"]["pipeline_max_attempts"] = 3
            matrix["scheduling"]["pipeline_retry_backoff_seconds"] = [0, 0]
            job = ("exp35_loc_cs_office", 100)
            completed = False
            attempts = 0

            class ExitingProcess:
                pid = 7000

                def __init__(self, returncode):
                    self.returncode = returncode

                def poll(self):
                    return self.returncode

            def spawn(_command, **_kwargs):
                nonlocal attempts, completed
                attempts += 1
                if attempts == 2:
                    completed = True
                return ExitingProcess(1 if attempts == 1 else 0)

            with (
                patch.object(RUNNER, "_parent_ready", return_value=True),
                patch.object(
                    RUNNER,
                    "pipeline_complete",
                    side_effect=lambda *_args: completed,
                ),
                patch.object(RUNNER, "sync_results"),
                patch.object(RUNNER, "run_family_aggregations"),
            ):
                RUNNER.schedule(
                    matrix,
                    root / "matrix.yaml",
                    cuda_device="0",
                    shots=[100],
                    jobs_override=[job],
                    launch_settle_seconds=0,
                    launch_memory_confirmation_seconds=0,
                    poll_seconds=0,
                    free_memory_fn=lambda _device: 100000,
                    process_gpu_memory_fn=lambda _device, _pid: 0,
                    popen_factory=spawn,
                    sleep_fn=lambda _seconds: None,
                )

            self.assertEqual(attempts, 2)

    def test_scheduler_stops_only_after_retry_budget_is_exhausted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = self.temporary_matrix(root)
            matrix["scheduling"]["pipeline_max_attempts"] = 3
            matrix["scheduling"]["pipeline_retry_backoff_seconds"] = [0, 0]
            attempts = 0

            class FailedProcess:
                pid = 8000

                def poll(self):
                    return 7

            def spawn(_command, **_kwargs):
                nonlocal attempts
                attempts += 1
                return FailedProcess()

            with (
                patch.object(RUNNER, "_parent_ready", return_value=True),
                patch.object(RUNNER, "pipeline_complete", return_value=False),
                patch.object(RUNNER, "sync_results"),
                patch.object(RUNNER, "run_family_aggregations"),
                self.assertRaisesRegex(RUNNER.PipelineError, "exhausted attempts=3/3"),
            ):
                RUNNER.schedule(
                    matrix,
                    root / "matrix.yaml",
                    cuda_device="0",
                    shots=[100],
                    jobs_override=[("exp35_loc_cs_office", 100)],
                    launch_settle_seconds=0,
                    launch_memory_confirmation_seconds=0,
                    poll_seconds=0,
                    free_memory_fn=lambda _device: 100000,
                    process_gpu_memory_fn=lambda _device, _pid: 0,
                    popen_factory=spawn,
                    sleep_fn=lambda _seconds: None,
                )

            self.assertEqual(attempts, 3)

    def test_completed_training_resumes_at_next_stage_without_duplicate_train(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = self.temporary_matrix(root)
            model_path = RUNNER.checkpoint_path(matrix, "exp35_gen_cs_office", 100)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with model_path.open("wb") as handle:
                handle.truncate(1024 * 1024)
            self.write_json(
                model_path.parent / "trainer_state.json",
                {"global_step": 400, "max_steps": 400},
            )
            commands = []
            with (
                patch.object(RUNNER, "_parent_ready", return_value=True),
                patch.object(
                    RUNNER,
                    "_run_command",
                    side_effect=lambda command, _log, _cuda: commands.append(command),
                ),
                patch.object(RUNNER, "_validate_stage_artifact"),
                patch.object(RUNNER, "sync_results") as sync_results,
                patch.object(RUNNER, "run_family_aggregations"),
            ):
                RUNNER.run_pipeline(
                    matrix,
                    "exp35_gen_cs_office",
                    100,
                    cuda_device="0",
                    families=["exp35"],
                    routes=["gen"],
                )
            self.assertEqual(len(commands), 28)
            self.assertTrue(all(command[0] == sys.executable for command in commands))
            self.assertNotIn("train_csgo.py", [command[1] for command in commands])
            self.assertEqual(sync_results.call_count, 2)
            self.assertTrue(
                all(
                    call.kwargs["families"] == ["exp35"]
                    and call.kwargs["routes"] == ["gen"]
                    for call in sync_results.call_args_list
                )
            )
            state = json.loads(
                RUNNER.pipeline_state_path(
                    matrix, "exp35_gen_cs_office", 100
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["phase"], RUNNER.COMPLETE_PHASE)

    def test_dry_run_prints_serial_plan_without_subprocess(self):
        output = io.StringIO()
        with (
            patch.object(RUNNER, "_run_command") as run_command,
            patch.object(RUNNER.subprocess, "run") as subprocess_run,
            patch.object(RUNNER.subprocess, "Popen") as popen,
            patch("sys.stdout", output),
        ):
            RUNNER.run_pipeline(
                self.matrix,
                "exp35_loc_cs_office",
                100,
                cuda_device="0",
                dry_run=True,
            )
        rendered = output.getvalue()
        self.assertIn("[exp35_loc_cs_office/100:train]", rendered)
        self.assertIn(
            "[exp35_loc_cs_office/100:localization_seen_inference_metric]", rendered
        )
        run_command.assert_not_called()
        subprocess_run.assert_not_called()
        popen.assert_not_called()

    def write_json(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def write_synthetic_generation_metric(self, matrix, root, shots=100):
        experiment = "exp35_gen_cs_office"
        metric_path = RUNNER._metric_path(
            matrix, experiment, shots, "discrete", "cs_office", "crossmap"
        )
        manifest_path = metric_path.parent / "inference_manifest.json"
        expected = RUNNER._expected_inference_payload(
            matrix, experiment, shots, "discrete", "crossmap"
        )
        self.write_json(manifest_path, expected)
        self.write_json(
            metric_path,
            {
                "map_name": "cs_office",
                "benchmark_v2_manifest": expected["benchmark_v2_manifest"],
                "benchmark_v2_split": "crossmap_query_test",
                "Common_Count": 2000,
                "metrics_ordered": {
                    "PSNR": 13.5,
                    "SSIM": 0.4567,
                    "LPIPS": 0.789,
                    "Boundary_F1": 0.6543,
                    "FID": 22.2,
                },
                "inference_provenance": {
                    "path": str(manifest_path.resolve()),
                    "payload": expected,
                },
            },
        )

    def real_results_document_text(self):
        """Use the repository document while tolerating its pending marker migration."""

        text = (
            REPO_ROOT / "csgo_benchmark_v2_experiments_results.md"
        ).read_text(encoding="utf-8")
        legacy_marker = "# ablation 3 maps表\n"
        setext_marker = "ablation 3 maps表\n====================\n"
        if legacy_marker in text:
            text = text.replace(legacy_marker, setext_marker, 1)
        return text

    def ablation_block(self, text):
        lines = text.splitlines(keepends=True)
        start, end = RUNNER._ablation_section_bounds(lines)
        return "".join(lines[start:end])

    def result_document_lines(self, main_discrete=(), supplement_discrete=()):
        return [
            "# csgo benchmark v2 实验进度\n",
            "ablation 3 maps表\n",
            "====================\n",
            "## 定位\n",
            "| header |\n",
            "## 离散生成\n",
            "| header |\n",
            "## 连续生成\n",
            "| header |\n",
            "# csgo benchmark v2 主表\n",
            "## 定位\n",
            "| header |\n",
            "## 离散生成\n",
            "| header |\n",
            *main_discrete,
            "## 连续生成\n",
            "| header |\n",
            "# csgo benchmark v2 补充表格\n",
            "## 定位\n",
            "| header |\n",
            "## 离散生成\n",
            "| header |\n",
            *supplement_discrete,
            "## 连续生成\n",
            "| header |\n",
        ]

    def test_result_sync_replaces_non_100_shot_row_in_main_without_inserting_supplement(
        self,
    ):
        prefix = (
            "| CrossMap-4 few-shot | Discrete generation | "
            "exp35_gen_cs_office | 50 |"
        )
        old_row = prefix + " old-main |\n"
        new_row = prefix + " new-main |"
        lines = self.result_document_lines(main_discrete=(old_row,))

        self.assertTrue(
            RUNNER._insert_or_replace_result_row(
                lines,
                "## 离散生成",
                prefix,
                new_row,
                shots=50,
                initialize=True,
            )
        )

        main_start, main_end = RUNNER._section_bounds(
            lines,
            "## 离散生成",
            parent_heading=RUNNER.MAIN_RESULTS_HEADING,
        )
        supplement_start, supplement_end = RUNNER._section_bounds(
            lines,
            "## 离散生成",
            parent_heading=RUNNER.SUPPLEMENT_RESULTS_HEADING,
        )
        main_section = lines[main_start:main_end]
        supplement_section = lines[supplement_start:supplement_end]
        self.assertIn(new_row + "\n", main_section)
        self.assertNotIn(old_row, main_section)
        self.assertFalse(any(line.startswith(prefix) for line in supplement_section))

    def test_result_sync_replaces_existing_supplement_row_in_place(self):
        prefix = (
            "| CrossMap-4 few-shot | Discrete generation | "
            "exp35_gen_cs_office | 50 |"
        )
        old_row = prefix + " old-supplement |\n"
        new_row = prefix + " new-supplement |"
        lines = self.result_document_lines(supplement_discrete=(old_row,))

        self.assertTrue(
            RUNNER._insert_or_replace_result_row(
                lines,
                "## 离散生成",
                prefix,
                new_row,
                shots=50,
                initialize=True,
            )
        )

        main_start, main_end = RUNNER._section_bounds(
            lines,
            "## 离散生成",
            parent_heading=RUNNER.MAIN_RESULTS_HEADING,
        )
        supplement_start, supplement_end = RUNNER._section_bounds(
            lines,
            "## 离散生成",
            parent_heading=RUNNER.SUPPLEMENT_RESULTS_HEADING,
        )
        main_section = lines[main_start:main_end]
        supplement_section = lines[supplement_start:supplement_end]
        self.assertFalse(any(line.startswith(prefix) for line in main_section))
        self.assertIn(new_row + "\n", supplement_section)
        self.assertNotIn(old_row, supplement_section)

    def test_result_sync_initializes_missing_non_100_shot_row_in_default_supplement_position(
        self,
    ):
        prefix = (
            "| CrossMap-4 few-shot | Discrete generation | "
            "exp35_gen_cs_office | 20 |"
        )
        new_row = prefix + " initialized |"
        existing_row = "| existing row |\n"
        lines = self.result_document_lines(supplement_discrete=(existing_row,))

        self.assertTrue(
            RUNNER._insert_or_replace_result_row(
                lines,
                "## 离散生成",
                prefix,
                new_row,
                shots=20,
                initialize=True,
            )
        )

        main_start, main_end = RUNNER._section_bounds(
            lines,
            "## 离散生成",
            parent_heading=RUNNER.MAIN_RESULTS_HEADING,
        )
        supplement_start, supplement_end = RUNNER._section_bounds(
            lines,
            "## 离散生成",
            parent_heading=RUNNER.SUPPLEMENT_RESULTS_HEADING,
        )
        supplement_continuous_start, _ = RUNNER._section_bounds(
            lines,
            "## 连续生成",
            parent_heading=RUNNER.SUPPLEMENT_RESULTS_HEADING,
        )
        main_section = lines[main_start:main_end]
        supplement_section = lines[supplement_start:supplement_end]
        inserted_index = lines.index(new_row + "\n")
        self.assertFalse(any(line.startswith(prefix) for line in main_section))
        self.assertIn(new_row + "\n", supplement_section)
        self.assertEqual(lines[inserted_index - 1], existing_row)
        self.assertLess(inserted_index, supplement_continuous_start)

    def test_sync_results_initializes_rows_extracts_metrics_and_preserves_headings(
        self,
    ):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            root = Path(temporary)
            matrix = self.temporary_matrix(root)
            original = (
                self.real_results_document_text()
            )
            ablation_before = self.ablation_block(original)
            maintenance_prefix = original[: original.index(RUNNER.EXPECTED_HEADINGS[0])]
            original = original.replace(
                "| CrossMap-4 few-shot | Discrete generation | exp33_gen | 100 |",
                "| Custom unknown row | keep |\n| CrossMap-4 few-shot | Discrete generation | exp33_gen | 100 |",
            )
            Path(matrix["paths"]["results_file"]).write_text(original, encoding="utf-8")
            self.write_synthetic_generation_metric(matrix, root)

            with patch.object(
                RUNNER, "_exclusive_lock", wraps=RUNNER._exclusive_lock
            ) as exclusive_lock:
                RUNNER.sync_results(matrix, shots=100, initialize=True)
                self.write_synthetic_generation_metric(matrix, root, shots=50)
                RUNNER.sync_results(matrix, shots=50, initialize=True)
            self.assertTrue(exclusive_lock.call_args.kwargs["blocking"])
            rendered = Path(matrix["paths"]["results_file"]).read_text(encoding="utf-8")
            self.assertTrue(rendered.startswith(maintenance_prefix))
            self.assertEqual(self.ablation_block(rendered), ablation_before)
            self.assertEqual(
                RUNNER._result_headings(rendered.splitlines(True)),
                list(RUNNER.EXPECTED_HEADINGS),
            )
            self.assertIn("| Custom unknown row | keep |", rendered)
            self.assertIn(
                "| CrossMap-4 few-shot | Discrete generation | exp35_gen_cs_office | 100 | 13.500 | 0.4567 | 0.7890 | 0.6543 | 22.200 | 400 |",
                rendered,
            )
            self.assertIn(
                "| CrossMap-4 few-shot | Discrete generation | exp35_gen_cs_office | 50 | 13.500 | 0.4567 | 0.7890 | 0.6543 | 22.200 | 400 |",
                rendered,
            )
            self.assertIn(
                "| CrossMap-4 few-shot | Discrete generation | exp35_cs_office | 100 |  |  |  |  |  | - |",
                rendered,
            )
            self.assertEqual(
                rendered.count(
                    "| CrossMap-4 few-shot | Discrete generation | exp35_gen_cs_office | 100 |"
                ),
                1,
            )
            main_start, main_end = RUNNER._section_bounds(
                rendered.splitlines(True),
                "## 离散生成",
                parent_heading=RUNNER.MAIN_RESULTS_HEADING,
            )
            supplement_start, supplement_end = RUNNER._section_bounds(
                rendered.splitlines(True),
                "## 离散生成",
                parent_heading=RUNNER.SUPPLEMENT_RESULTS_HEADING,
            )
            main_section = "".join(rendered.splitlines(True)[main_start:main_end])
            supplement_section = "".join(
                rendered.splitlines(True)[supplement_start:supplement_end]
            )
            self.assertIn("| exp35_gen_cs_office | 100 |", main_section)
            self.assertIn(
                "| exp35_gen_cs_office | 50 | 13.500 | 0.4567 | 0.7890 | 0.6543 | 22.200 | 400 |",
                main_section,
            )
            self.assertNotIn(
                "| CrossMap-4 few-shot | Discrete generation | exp35_gen_cs_office | 50 |",
                supplement_section,
            )
            self.assertEqual(
                rendered.count(
                    "| CrossMap-4 few-shot | Discrete generation | exp35_gen_cs_office | 50 |"
                ),
                1,
            )
            self.assertNotIn(
                "| exp35_gen_cs_office | 50 |", rendered.splitlines(True)[0:main_start]
            )
            self.assertEqual(rendered.count("| `exp35_gen_cs_office` 100-shot |"), 1)
            self.assertEqual(rendered.count("| `exp35_gen_cs_office` 50-shot |"), 1)

    def test_sync_results_rejects_heading_change_without_rewriting(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            root = Path(temporary)
            matrix = self.temporary_matrix(root)
            results = Path(matrix["paths"]["results_file"])
            results.write_text(
                self.real_results_document_text()
                + "\n# unexpected heading\n",
                encoding="utf-8",
            )
            before = results.read_bytes()
            with self.assertRaises(RUNNER.PipelineError):
                RUNNER.sync_results(matrix, shots=100, initialize=True)
            self.assertEqual(results.read_bytes(), before)

    def test_sync_results_rejects_invalid_subheadings_in_every_result_parent(self):
        original = self.real_results_document_text()
        original_lines = original.splitlines(keepends=True)
        parents = (
            RUNNER.ABLATION_RESULTS_HEADING,
            RUNNER.MAIN_RESULTS_HEADING,
            RUNNER.SUPPLEMENT_RESULTS_HEADING,
        )

        for parent in parents:
            if parent == RUNNER.ABLATION_RESULTS_HEADING:
                parent_start, parent_end = RUNNER._ablation_section_bounds(
                    original_lines
                )
                child_start = parent_start + 2
            else:
                parent_start, parent_end = RUNNER._section_bounds(
                    original_lines, parent
                )
                child_start = parent_start + 1
            child_indices = [
                index
                for index in range(child_start, parent_end)
                if RUNNER._heading_level(original_lines[index]) == 2
            ]
            self.assertEqual(
                [original_lines[index].strip() for index in child_indices],
                list(RUNNER.EXPECTED_RESULT_SUBHEADINGS),
            )

            mutations = {
                "unknown": lambda lines: lines.__setitem__(
                    child_indices[0], "## 未知结果分组\n"
                ),
                "missing": lambda lines: lines.pop(child_indices[0]),
                "reordered": lambda lines: lines.__setitem__(
                    slice(child_indices[0], child_indices[1] + 1),
                    [lines[child_indices[1]], lines[child_indices[0]]],
                ),
            }
            for mutation_name, mutate in mutations.items():
                with self.subTest(parent=parent, mutation=mutation_name):
                    with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
                        root = Path(temporary)
                        matrix = self.temporary_matrix(root)
                        results = Path(matrix["paths"]["results_file"])
                        bad_lines = list(original_lines)
                        mutate(bad_lines)
                        results.write_text("".join(bad_lines), encoding="utf-8")
                        before = results.read_bytes()
                        with self.assertRaises(RUNNER.PipelineError):
                            RUNNER.sync_results(matrix, shots=100, initialize=True)
                        self.assertEqual(results.read_bytes(), before)

    def test_sync_results_rejects_invalid_setext_ablation_marker_without_rewriting(
        self,
    ):
        original = self.real_results_document_text()
        original_lines = original.splitlines(keepends=True)
        ablation_start, ablation_end = RUNNER._ablation_section_bounds(original_lines)
        ablation_block = original_lines[ablation_start:ablation_end]

        duplicate = list(original_lines) + ablation_block
        misplaced = original_lines[:ablation_start] + original_lines[ablation_end:]
        supplement_index = next(
            index
            for index, line in enumerate(misplaced)
            if line.rstrip("\r\n") == RUNNER.SUPPLEMENT_RESULTS_HEADING
        )
        misplaced[supplement_index:supplement_index] = ablation_block
        malformed = list(original_lines)
        malformed[ablation_start + 1] = "--------------------\n"

        for name, bad_lines in (
            ("duplicate", duplicate),
            ("misplaced", misplaced),
            ("malformed", malformed),
        ):
            with self.subTest(marker=name):
                with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
                    root = Path(temporary)
                    matrix = self.temporary_matrix(root)
                    results = Path(matrix["paths"]["results_file"])
                    results.write_text("".join(bad_lines), encoding="utf-8")
                    before = results.read_bytes()
                    with self.assertRaises(RUNNER.PipelineError):
                        RUNNER.sync_results(matrix, shots=100, initialize=True)
                    self.assertEqual(results.read_bytes(), before)

    def test_filtered_sync_does_not_initialize_joint_shot_rows(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temporary:
            root = Path(temporary)
            matrix = self.temporary_matrix(root)
            results = Path(matrix["paths"]["results_file"])
            original = (
                self.real_results_document_text()
            )
            maintenance_prefix = original[: original.index(RUNNER.EXPECTED_HEADINGS[0])]
            ablation_before = self.ablation_block(original)
            results.write_text(original, encoding="utf-8")

            joint_names = {
                "exp35",
                *(f"exp35_{map_name}" for map_name in RUNNER.CROSSMAP_MAPS),
            }

            def joint_rows(text):
                rows = []
                for line in text.splitlines():
                    fields = [field.strip() for field in line.split("|")]
                    if (
                        len(fields) >= 6
                        and fields[3] in joint_names
                        and fields[4] == "50"
                    ):
                        rows.append(line)
                return rows

            joint_rows_before = joint_rows(original)
            self.assertTrue(joint_rows_before)

            RUNNER.sync_results(
                matrix,
                shots=50,
                initialize=True,
                families=["exp35"],
                routes=["gen", "loc"],
            )
            rendered = results.read_text(encoding="utf-8")
            self.assertTrue(rendered.startswith(maintenance_prefix))
            self.assertEqual(self.ablation_block(rendered), ablation_before)
            self.assertEqual(joint_rows(rendered), joint_rows_before)
            self.assertIn("| `exp35_gen_cs_office` 50-shot |", rendered)
            self.assertIn("| `exp35_loc_cs_office` 50-shot |", rendered)

    def test_family_map_models_aggregation_waits_for_all_four_maps(self):
        with tempfile.TemporaryDirectory() as temporary:
            matrix = self.temporary_matrix(Path(temporary))
            with (
                patch.object(RUNNER, "_family_complete", return_value=False),
                patch.object(RUNNER, "_run_command") as run_command,
            ):
                self.assertEqual(
                    RUNNER.run_family_aggregations(matrix, 100, cuda_device="0"), []
                )
                run_command.assert_not_called()

            def only_exp35_gen(_matrix, family, route, _shots):
                return family == "exp35" and route == "gen"

            with (
                patch.object(RUNNER, "_family_complete", side_effect=only_exp35_gen),
                patch.object(RUNNER, "_run_command") as run_command,
                patch.object(RUNNER, "_validate_family_summary", return_value={}),
            ):
                outputs = RUNNER.run_family_aggregations(matrix, 100, cuda_device="0")
            self.assertEqual(len(outputs), 2)
            self.assertEqual(run_command.call_count, 2)
            self.assertTrue(
                all("map-models" in call.args[0] for call in run_command.call_args_list)
            )
            self.assertTrue(all("exp35_gen" in str(path) for path in outputs))

    def test_cli_exposes_required_commands_and_family_route_overrides(self):
        parser = RUNNER._build_parser()
        for command in ("validate", "status", "run", "schedule", "sync-results"):
            args = [command]
            if command == "run":
                args += ["--experiment", "exp35_loc_cs_office", "--shots", "100"]
            parsed = parser.parse_args(args)
            self.assertEqual(parsed.command, command)
        parsed = parser.parse_args(
            ["schedule", "--exp35-loc-minimum-free-memory-mb", "27000"]
        )
        self.assertEqual(parsed.exp35_loc_minimum_free_memory_mb, 27000)
        parsed = parser.parse_args(
            ["schedule", "--exp36_1-joint-minimum-free-memory-mb", "46000"]
        )
        self.assertEqual(parsed.exp36_1_joint_minimum_free_memory_mb, 46000)
        parsed = parser.parse_args(
            ["schedule", "--launch-memory-confirmation-seconds", "0"]
        )
        self.assertEqual(parsed.launch_memory_confirmation_seconds, 0)
        parsed = parser.parse_args(
            [
                "sync-results",
                "--shots",
                "50",
                "--families",
                "exp35",
                "--routes",
                "gen",
                "loc",
            ]
        )
        self.assertEqual(parsed.families, ["exp35"])
        self.assertEqual(parsed.routes, ["gen", "loc"])


if __name__ == "__main__":
    unittest.main()
