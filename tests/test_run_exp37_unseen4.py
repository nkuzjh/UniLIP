"""CPU-only protocol checks for the exp37 Unseen-4 planner."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import yaml

from scripts import aggregate_csgo_benchmark_v2_metrics as aggregates
from scripts import run_exp37_unseen4 as runner


def args(*values: str):
    return runner.parser().parse_args(values)


def option(cmd: list[str], key: str) -> str:
    return cmd[cmd.index(key) + 1]


class Exp37Unseen4Tests(unittest.TestCase):
    def test_all_80_points_validate_without_parent_weights(self):
        commands, writes, info = runner.plan(args("validate"))
        self.assertEqual((commands, writes), ([], []))
        self.assertEqual(info["validated_points"], 80)

    def test_train_plan_fixed_protocol(self):
        families = {"exp37_fs_gen": "gen", "exp37_fs_loc": "loc",
                    "exp37_fs_joint": "joint", "exp37_fs_both": "joint",
                    "exp37_fs_both_b1": "joint"}
        batches = {100: (100, 25), 50: (48, 12), 20: (20, 5), 10: (8, 2)}
        for family, route in families.items():
            for shots, (base, micro) in batches.items():
                with self.subTest(family=family, shots=shots):
                    a = args("train", "--experiment", family, "--map", "cs_office", "--shots", str(shots))
                    commands, writes, info = runner.plan(a)
                    cfg = runner.training_recipe(family, "cs_office", shots, a, runner.parent_checkpoint(family, strict=False)[0])
                    self.assertEqual((len(commands), len(writes)), (1, 2))
                    cmd = commands[0]
                    for flag, expected in (("--max_steps", "400"), ("--save_strategy", "no"),
                                           ("--warmup_ratio", "0.003"), ("--optim", "adamw_torch"),
                                           ("--adam_beta1", "0.9"), ("--adam_beta2", "0.999"),
                                           ("--adam_epsilon", "1e-8"), ("--weight_decay", "0"),
                                           ("--max_grad_norm", "1.0"), ("--benchmark_v2_support_seed", "0"),
                                           ("--benchmark_v2_shots_per_map", str(shots))):
                        self.assertEqual(option(cmd, flag), expected)
                    self.assertNotIn("--warmup_steps", cmd)
                    self.assertEqual(cfg["benchmark_v2_split"], "crossmap_support")
                    self.assertEqual(cfg["benchmark_v2_asset_manifest"], runner.MATRIX["minimal_asset_manifest"])
                    self.assertIs(cfg["csgo_forward_sampler_epoch"], True)
                    self.assertIs(cfg["csgo_loss_per_microbatch_mean"], True)
                    self.assertEqual(cfg["lr_scheduler_kwargs"], {"min_lr_rate": 0.1})
                    if route == "joint":
                        self.assertEqual(option(cmd, "--per_device_train_batch_size"), "4")
                        self.assertEqual(option(cmd, "--gradient_accumulation_steps"), "32")
                        plan = json.loads(writes[1][1])
                        self.assertEqual(plan["effective_batch"], {
                            "micro_batch": 4, "configured_gas": 32,
                            "microbatches_per_update": micro,
                            "base_samples_per_update": base,
                            "base_sample_exposures_400_steps": base * 400,
                        })
                    else:
                        self.assertEqual(option(cmd, "--per_device_train_batch_size"), str(shots))
                        self.assertEqual(option(cmd, "--gradient_accumulation_steps"), "1")
                    self.assertTrue(info["source"]["path"].endswith("checkpoint-6000/model.safetensors"))
                    self.assertIs(info["source"]["verified"], False)

    def test_combined_fixed_objective_and_control_gating(self):
        for family in runner.FAMILIES:
            with self.subTest(family=family):
                a = args("train", "--experiment", family, "--map", "de_palacio")
                cfg = runner.training_recipe(family, "de_palacio", 100, a, runner.parent_checkpoint(family, strict=False)[0])
                runner.validate_recipe(cfg, family, "de_palacio", 100)
                self.assertTrue(all(cfg[key] == 2e-5 for key in runner.LR_LOW))
                self.assertTrue(all(cfg[key] == 1e-4 for key in runner.LR_HIGH))
                self.assertFalse(any(key.endswith("_schedule_steps") or key.endswith("_schedule_values") for key in cfg))
                if family.startswith("exp37_fs_both"):
                    self.assertEqual((cfg["alpha_loc_loss"], cfg["alpha_loc_aux_loss"],
                                      cfg["alpha_loc_perception_loss"]), (8, 1, .05))
                    self.assertEqual(cfg["loc_perception_use_attention_weight"], family.endswith("_b1"))
                else:
                    self.assertIs(cfg["is_loc_aux_loss"], False)
                    self.assertIs(cfg["is_loc_perception_loss"], False)
        bad = copy.deepcopy(cfg)
        bad["warmup_steps"] = 200
        with self.assertRaisesRegex(ValueError, "schedule"):
            runner.validate_recipe(bad, family, "de_palacio", 100)

    def test_zero_shot_three_routes_one_inference_per_task_and_null_support(self):
        a = args("infer", "--experiment", "exp37_fs_joint", "--zero-shot")
        commands, writes, _ = runner.plan(a)
        self.assertEqual(len(commands), 3)
        self.assertEqual(len(writes), 8)
        for cmd in commands:
            self.assertEqual(cmd.count("--benchmark_v2_maps"), 1)
            self.assertEqual(cmd[cmd.index("--benchmark_v2_maps") + 1:cmd.index("--benchmark_v2_asset_manifest")], list(runner.MAPS))
            self.assertNotIn("--benchmark_v2_shots_per_map", cmd)
            self.assertNotIn("--benchmark_v2_support_seed", cmd)
        source_cfg = next(text for path, text in writes if path.name == "zero_shot_config.yaml")
        cfg = yaml.safe_load(source_cfg)
        self.assertNotIn("benchmark_v2_shots_per_map", cfg)
        self.assertNotIn("benchmark_v2_support_seed", cfg)
        self.assertEqual([option(cmd, "--benchmark_v2_split") for cmd in commands],
                         ["crossmap_query_test", "crossmap_query_test", "crossmap_continuous"])

    def test_split_counts_and_metric_backend(self):
        a = args("metrics", "--experiment", "exp37_fs_joint", "--map", "cs_office",
                 "--split", "retention", "--task", "gen_continuous")
        commands, _, _ = runner.plan(a)
        self.assertEqual(len(commands), 11)
        for cmd in commands[:-1]:
            self.assertIn("benchmark_csgo_v1_conti.py", cmd)
            self.assertEqual(option(cmd, "--benchmark_v2_split"), "seen_continuous")
            self.assertEqual(option(cmd, "--metric_profile"), "benchmark_v2_core")
            self.assertEqual((option(cmd, "--clip_length"), option(cmd, "--clip_stride")), ("16", "16"))
            self.assertEqual(option(cmd, "--fvd_size"), "224")
        self.assertIn("maps", commands[-1])

    def test_query_adapted_map_model_macro_and_zero_shot_maps_macro(self):
        cmd = runner.aggregate_command("exp37_fs_joint", 50, "gen_continuous", False)
        self.assertIn("map-models", cmd)
        self.assertTrue(option(cmd, "--input_pattern").endswith(
            "exp37_fs_joint_{map}/shot_50/seed_0/continuous/benchmark_csgo_v2_conti_{map}.json"))
        self.assertEqual(option(cmd, "--split"), "crossmap_continuous")
        zs = runner.aggregate_command("exp37_fs_joint", 100, "gen", True)
        self.assertIn("maps", zs)
        self.assertNotIn("map-models", zs)
        self.assertEqual(zs[zs.index("--maps") + 1:zs.index("--input_root")], list(runner.MAPS))
        per_map = {m: {"PSNR": i * 10.0} for i, m in enumerate(runner.MAPS, 1)}
        macro = aggregates._equal_map_macro(per_map, {m: Path(m) for m in runner.MAPS})
        self.assertEqual(macro["PSNR"], 25.0)

    def test_invalid_route_backend_seed_and_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not support"):
            runner.plan(args("infer", "--experiment", "exp37_fs_gen", "--map", "cs_office", "--task", "loc"))
        with self.assertRaisesRegex(ValueError, "requires --data-dir"):
            runner.plan(args("train", "--experiment", "exp37_fs_gen", "--map", "cs_office", "--asset-mode", "full"))
        with self.assertRaisesRegex(ValueError, "Protocol fixes"):
            runner.plan(args("train", "--experiment", "exp37_fs_gen", "--map", "cs_office", "--seed", "7"))
        with patch.object(runner.seen37, "final_checkpoint_step", side_effect=ValueError("missing final")):
            with self.assertRaisesRegex(ValueError, "missing final"):
                runner.parent_checkpoint("exp37_fs_gen", strict=True)

    def test_full_backend_override_and_saved_recipe_rejects_corruption(self):
        a = args("train", "--experiment", "exp37_fs_both", "--map", "de_golden",
                 "--asset-mode", "full", "--data-dir", "/data/full", "--model-path", "/models/main",
                 "--pi05-path", "/models/pi05")
        cfg = runner.training_recipe("exp37_fs_both", "de_golden", 100, a, runner.parent_checkpoint("exp37_fs_both", strict=False)[0])
        self.assertIsNone(cfg["benchmark_v2_asset_manifest"])
        self.assertEqual(cfg["data_dir"], "/data/full")
        self.assertEqual(cfg["model_name_or_path"], "/models/main")
        self.assertEqual(cfg["pi05_pytorch_weight_path"], "/models/pi05")
        broken = copy.deepcopy(cfg)
        broken["alpha_loc_aux_loss"] = 0
        with self.assertRaisesRegex(ValueError, "Combined loss"):
            runner.validate_recipe(broken, "exp37_fs_both", "de_golden", 100)

    def test_matrix_device_and_port_drift_rejected(self):
        with patch.dict(runner.MATRIX["training"], {"warmup_ratio": 0.5}):
            with self.assertRaisesRegex(ValueError, "matrix differs"):
                runner.plan(args("validate"))
        for extra in (("--cuda-devices", "0,1"), ("--cuda-device", ""),
                      ("--master-port", "70000")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                runner.plan(args("train", "--experiment", "exp37_fs_gen", "--map", "cs_office", *extra))

    def test_execute_fixture_snapshot_inference_and_provenance(self):
        original_root = runner.ROOT
        source_yaml = original_root / "csgo_configs/exp37_s2.yaml"
        manifest = original_root / runner.MATRIX["manifest"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "csgo_configs").mkdir(parents=True)
            (root / "data/csgo_benchmark_v2").mkdir(parents=True)
            shutil.copy2(source_yaml, root / "csgo_configs/exp37_s2.yaml")
            shutil.copy2(manifest, root / runner.MATRIX["manifest"])
            source_cfg = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
            parent = root / "outputs/csgo_1b/exp37_s2"
            checkpoint_dir = parent / "checkpoint-6000"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "model.safetensors").write_bytes(b"parent model")
            state = {"global_step": 6000, "max_steps": 6000}
            (checkpoint_dir / "trainer_state.json").write_text(json.dumps(state))
            (parent / "trainer_state.json").write_text(json.dumps(state))
            (parent / "resolved_config.yaml").write_text(yaml.safe_dump(source_cfg))
            runtime = {"max_steps": 6000, "seed": 42, "source_batch": 128, "warmup_steps": 200}
            parent_plan = {"experiment": "exp37_s2", "runtime": runtime,
                           "fingerprint": runner.seen37.fingerprint(source_cfg, runtime)}
            (parent / "experiment_plan.json").write_text(json.dumps(parent_plan))
            with patch.object(runner, "ROOT", root), patch.object(runner.seen37, "ROOT", root), \
                 patch.object(runner.seen37, "tensor_header", return_value={"tensor_count": 1}):
                train_args = args("train", "--experiment", "exp37_fs_joint", "--map", "cs_office",
                                  "--shots", "50", "--asset-mode", "full", "--data-dir", "/data/full",
                                  "--model-path", "/models/main", "--pi05-path", "/models/pi05", "--execute")
                commands, writes, _ = runner.plan(train_args)
                self.assertEqual(len(commands), 1)
                for path, content in writes:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                out = runner.model_dir("exp37_fs_joint", "cs_office", 50)
                (out / "model.safetensors").write_bytes(b"adapted model A")
                (out / "trainer_state.json").write_text(json.dumps({"global_step": 400, "max_steps": 400}))
                infer_args = args("infer", "--experiment", "exp37_fs_joint", "--map", "cs_office",
                                  "--shots", "50", "--task", "gen", "--execute")
                commands, writes, info = runner.plan(infer_args)
                self.assertEqual(len(commands), 1)
                self.assertNotIn("--benchmark_v2_asset_manifest", commands[0])
                for path, content in writes:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                output = runner.eval_dir("exp37_fs_joint", "cs_office", 50, "query", "gen", False)
                self.assertIs(runner.seen37.read_yaml(output / "evaluation_config.yaml")["benchmark_v2_allow_map_subset_summary"], True)
                manifest_payload = {
                    "config_path": str(output / "evaluation_config.yaml"),
                    "benchmark_v2_manifest": str(root / runner.MATRIX["manifest"]),
                    "benchmark_v2_split": "crossmap_query_test", "benchmark_v2_support_seed": 0,
                    "benchmark_v2_shots_per_map": 50, "maps": ["cs_office"],
                    "sample_count": 2000, "checkpoint": str(out / "model.safetensors"),
                    "ckpt_path": str(out / "model.safetensors"), "seed": 42,
                }
                (output / "inference_manifest.json").write_text(json.dumps(manifest_payload))
                runner.validate_inference(output, runner.seen37.read_yaml(out / "resolved_config.yaml"),
                                          out / "model.safetensors", ("cs_office",),
                                          "crossmap_query_test", "gen", 50, 0, info["source"])
                metric_payload = {
                    "map_name": "cs_office", "benchmark_v2_split": "crossmap_query_test",
                    "benchmark_v2_manifest": str(root / runner.MATRIX["manifest"]),
                    "metrics_ordered": {"PSNR": 12.0, "Common_Count": 2000},
                    "inference_provenance": {"path": str(output / "inference_manifest.json"),
                                             "payload": manifest_payload},
                }
                (output / "benchmark_csgo_v2_cs_office.json").write_text(json.dumps(metric_payload))
                runner.validate_generation_result(output, "cs_office", "gen", "crossmap_query_test", manifest_payload)
                aggregates._validate_manifest_provenance(
                    metric_payload, path=output / "benchmark_csgo_v2_cs_office.json",
                    manifest_path=root / runner.MATRIX["manifest"], map_name="cs_office",
                    split="crossmap_query_test", kind="discrete", expected_maps=["cs_office"])
                metrics_args = args("metrics", "--experiment", "exp37_fs_joint", "--map", "cs_office",
                                    "--shots", "50", "--task", "gen", "--execute")
                self.assertEqual(len(runner.plan(metrics_args)[0]), 1)
                loc_args = args("infer", "--experiment", "exp37_fs_joint", "--map", "cs_office",
                                "--shots", "50", "--task", "loc", "--execute")
                _, loc_writes, _ = runner.plan(loc_args)
                for path, content in loc_writes:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                loc_output = runner.eval_dir("exp37_fs_joint", "cs_office", 50, "query", "loc", False)
                loc_provenance = dict(manifest_payload, config_path=str(loc_output / "evaluation_config.yaml"))
                (loc_output / "inference_manifest.json").write_text(json.dumps(loc_provenance))
                loc_metrics = {"XY_Dist": 1.0, "Z_Dist": 2.0, "Pitch_Dist": 3.0, "Yaw_Dist": 4.0}
                summary = aggregates.build_localization_summary(
                    manifest=str(root / runner.MATRIX["manifest"]), split="crossmap_query_test",
                    maps=["cs_office"], per_map={"cs_office": loc_metrics},
                    metrics_macro_map=loc_metrics,
                    inference_provenance={"path": str(loc_output / "inference_manifest.json"),
                                          "payload": loc_provenance},
                    checkpoint=str(out / "model.safetensors"), seed=42,
                    support_seed=0, shots_per_map=50, sample_count=2000,
                    allow_protocol_map_subset=True)
                (loc_output / "benchmark_csgo_v2_loc.json").write_text(json.dumps(summary))
                self.assertEqual(runner.plan(args("metrics", "--experiment", "exp37_fs_joint",
                                                  "--map", "cs_office", "--shots", "50",
                                                  "--task", "loc", "--execute"))[0], [])
                bad_args = args("infer", "--experiment", "exp37_fs_joint", "--map", "cs_office",
                                "--shots", "100", "--task", "gen", "--execute")
                with self.assertRaisesRegex(ValueError, "Missing adaptation snapshot"):
                    runner.plan(bad_args)
                bad_manifest = dict(manifest_payload, checkpoint=str(parent / "checkpoint-6000/model.safetensors"))
                (output / "inference_manifest.json").write_text(json.dumps(bad_manifest))
                with self.assertRaisesRegex(ValueError, "checkpoint differs"):
                    runner.plan(metrics_args)
                (output / "inference_manifest.json").write_text(json.dumps(manifest_payload))
                (out / "model.safetensors").write_bytes(b"adapted model B")
                with self.assertRaisesRegex(ValueError, "weight/config provenance differs"):
                    runner.plan(metrics_args)
                source_plan_path = out / "experiment_plan.json"
                plan_data = json.loads(source_plan_path.read_text())
                plan_data["source"]["sha256"] = "0" * 64
                source_plan_path.write_text(json.dumps(plan_data))
                with self.assertRaisesRegex(ValueError, "Saved source model hash differs"):
                    runner.plan(metrics_args)


if __name__ == "__main__":
    unittest.main()
