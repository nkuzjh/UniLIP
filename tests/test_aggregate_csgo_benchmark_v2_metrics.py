import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_csgo_benchmark_v2_metrics import (
    AggregationError,
    _build_parser,
    aggregate_maps,
    aggregate_seeds,
    build_localization_summary,
    validate_localization_result_coverage,
)


class AggregateCsgoBenchmarkV2MetricsTests(unittest.TestCase):
    def _write_json(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, allow_nan=True, indent=2) + "\n", encoding="utf-8")

    def _make_manifest(self, root: Path) -> Path:
        manifest = root / "benchmark_manifest.json"
        self._write_json(
            manifest,
            {
                "schema_version": 1,
                "benchmark_id": "csgo_benchmark_v2",
                "protocol": {
                    "seen_maps": ["map_a", "map_b"],
                    "crossmap_maps": ["map_c", "map_d"],
                },
            },
        )
        return manifest

    def _write_per_map_results(
        self,
        root: Path,
        manifest: Path,
        split: str = "seen_discrete_test",
        kind: str = "discrete",
        values=(1.0, 3.0),
    ) -> None:
        maps = ["map_a", "map_b"] if split.startswith("seen_") else ["map_c", "map_d"]
        prefix = "benchmark_csgo_v2_" if kind == "discrete" else "benchmark_csgo_v2_conti_"
        inference_manifest = root / "inference_manifest.json"
        inference_payload = {
            "benchmark_v2_manifest": str(manifest),
            "benchmark_v2_split": split,
            "benchmark_v2_support_seed": None,
            "benchmark_v2_shots_per_map": None,
            "maps": maps,
            "sample_count": 4,
            "checkpoint": "checkpoint.safetensors",
            "ckpt_path": "checkpoint.safetensors",
            "seed": 42,
        }
        self._write_json(inference_manifest, inference_payload)
        for map_name, value in zip(maps, values):
            self._write_json(
                root / f"{prefix}{map_name}.json",
                {
                    "map_name": map_name,
                    "benchmark_v2_manifest": str(manifest),
                    "benchmark_v2_split": split,
                    "metrics_ordered": {
                        "PSNR": value,
                        "SSIM": value + 10.0,
                        "Common_Count": int(value),
                        "not_a_metric": None,
                        "bool_metric": True,
                    },
                    "inference_provenance": {
                        "path": str(inference_manifest),
                        "payload": inference_payload,
                    },
                },
            )

    def _write_maps_aggregate(
        self,
        path: Path,
        manifest: Path,
        values,
        *,
        split: str = "crossmap_query_test",
        kind: str = "discrete",
        inference_seed: int = 42,
        support_seed: int | None = None,
        shots_per_map: int | None = None,
        sample_count: int = 8,
    ) -> None:
        maps = ["map_a", "map_b"] if split.startswith("seen_") else ["map_c", "map_d"]
        inference_manifest = path.parent / "inference_manifest.json"
        inference_payload = {
            "benchmark_v2_manifest": str(manifest),
            "benchmark_v2_split": split,
            "benchmark_v2_support_seed": support_seed,
            "benchmark_v2_shots_per_map": shots_per_map,
            "maps": maps,
            "sample_count": sample_count,
            "checkpoint": "checkpoint.safetensors",
            "ckpt_path": "checkpoint.safetensors",
            "seed": inference_seed,
        }
        self._write_json(inference_manifest, inference_payload)
        self._write_json(
            path,
            {
                "manifest": str(manifest),
                "split": split,
                "kind": kind,
                "maps": maps,
                "per_map": {},
                "metrics_macro_map": {
                    "PSNR": values[0],
                    "SSIM": values[1],
                },
                "inference_provenance": {
                    "path": str(inference_manifest),
                    "payload": inference_payload,
                },
                "checkpoint": "checkpoint.safetensors",
                "ckpt_path": "checkpoint.safetensors",
                "inference_seed": inference_seed,
                "support_seed": support_seed,
                "shots_per_map": shots_per_map,
                "sample_count": sample_count,
                "source_files": {},
            },
        )

    def test_maps_equal_map_macro_and_exact_protocol_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            results_root = root / "per_map"
            self._write_per_map_results(results_root, manifest)
            output = root / "nested" / "map_macro.json"

            result = aggregate_maps(
                manifest=manifest,
                split="seen_discrete_test",
                input_root=results_root,
                kind="discrete",
                output=output,
            )

            self.assertEqual(result["maps"], ["map_a", "map_b"])
            self.assertEqual(result["metrics_macro_map"]["PSNR"], 2.0)
            self.assertEqual(result["metrics_macro_map"]["SSIM"], 12.0)
            self.assertEqual(result["metrics_macro_map"]["Common_Count"], 2.0)
            self.assertNotIn("bool_metric", result["metrics_macro_map"])
            self.assertEqual(result["inference_seed"], 42)
            self.assertIsNone(result["support_seed"])
            self.assertIsNone(result["shots_per_map"])
            self.assertEqual(result["sample_count"], 4)
            self.assertEqual(
                result["inference_provenance"]["path"],
                str((results_root / "inference_manifest.json").resolve()),
            )
            self.assertTrue(output.is_file())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), result)

            self._write_json(
                results_root / "benchmark_csgo_v2_extra_map.json",
                {
                    "map_name": "extra_map",
                    "benchmark_v2_manifest": str(manifest),
                    "benchmark_v2_split": "seen_discrete_test",
                    "metrics_ordered": {"PSNR": 1.0},
                },
            )
            with self.assertRaisesRegex(AggregationError, "extra per-map"):
                aggregate_maps(
                    manifest=manifest,
                    split="seen_discrete_test",
                    input_root=results_root,
                    kind="discrete",
                    output=root / "unused.json",
                )

    def test_maps_reject_manifest_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            other_manifest = root / "other_manifest.json"
            self._write_json(other_manifest, {"protocol": {"seen_maps": ["map_a", "map_b"]}})
            results_root = root / "per_map"
            self._write_per_map_results(results_root, manifest)
            payload_path = results_root / "benchmark_csgo_v2_map_b.json"
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            payload["benchmark_v2_manifest"] = str(other_manifest)
            self._write_json(payload_path, payload)

            with self.assertRaisesRegex(AggregationError, "manifest provenance mismatch"):
                aggregate_maps(
                    manifest=manifest,
                    split="seen_discrete_test",
                    input_root=results_root,
                    kind="discrete",
                    output=root / "unused.json",
                )

    def test_maps_reject_nonfinite_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            results_root = root / "per_map"
            self._write_per_map_results(results_root, manifest)
            payload_path = results_root / "benchmark_csgo_v2_map_a.json"
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            payload["metrics_ordered"]["PSNR"] = float("nan")
            self._write_json(payload_path, payload)

            with self.assertRaisesRegex(AggregationError, "NaN or infinite"):
                aggregate_maps(
                    manifest=manifest,
                    split="seen_discrete_test",
                    input_root=results_root,
                    kind="discrete",
                    output=root / "unused.json",
                )

    def test_maps_reject_non_common_inference_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            results_root = root / "per_map"
            self._write_per_map_results(results_root, manifest)
            other_inference_manifest = results_root / "other_inference_manifest.json"
            other_payload = json.loads(
                (results_root / "inference_manifest.json").read_text(encoding="utf-8")
            )
            other_payload["seed"] = 43
            self._write_json(other_inference_manifest, other_payload)
            map_b_path = results_root / "benchmark_csgo_v2_map_b.json"
            map_b = json.loads(map_b_path.read_text(encoding="utf-8"))
            map_b["inference_provenance"] = {
                "path": str(other_inference_manifest),
                "payload": other_payload,
            }
            self._write_json(map_b_path, map_b)

            with self.assertRaisesRegex(AggregationError, "path/payload"):
                aggregate_maps(
                    manifest=manifest,
                    split="seen_discrete_test",
                    input_root=results_root,
                    kind="discrete",
                    output=root / "unused.json",
                )

    def test_five_support_seeds_report_sample_ci(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            seed_root = root / "seed_results"
            for seed in range(5):
                self._write_maps_aggregate(
                    seed_root / f"seed_{seed}" / "map_macro.json",
                    manifest,
                    (float(seed + 1), float(10 + seed)),
                    support_seed=seed,
                    shots_per_map=100,
                )

            output = root / "summary" / "support_selection.json"
            result = aggregate_seeds(
                seed_root_pattern=str(seed_root / "seed_{seed}" / "map_macro.json"),
                output=output,
            )

            psnr = result["metrics_support_selection"]["PSNR"]
            self.assertEqual(result["uncertainty"], "support_selection")
            self.assertEqual(result["seeds"], [0, 1, 2, 3, 4])
            self.assertEqual(result["inference_seed"], 42)
            self.assertEqual(result["shots_per_map"], 100)
            self.assertEqual(psnr["n"], 5)
            self.assertEqual(psnr["mean"], 3.0)
            self.assertAlmostEqual(psnr["sample_std"], math.sqrt(2.5))
            self.assertAlmostEqual(psnr["standard_error"], math.sqrt(0.5))
            self.assertEqual(psnr["t_critical"], 2.7764451051977987)
            expected_margin = 2.7764451051977987 * math.sqrt(0.5)
            self.assertAlmostEqual(psnr["ci95_low"], 3.0 - expected_margin)
            self.assertAlmostEqual(psnr["ci95_high"], 3.0 + expected_margin)
            self.assertTrue(output.is_file())

    def test_generation_seed_metadata_requires_support_and_fixed_inference_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            seed_root = root / "seed_results"
            for seed in range(5):
                self._write_maps_aggregate(
                    seed_root / f"seed_{seed}" / "map_macro.json",
                    manifest,
                    (float(seed + 1), float(10 + seed)),
                    support_seed=seed,
                    shots_per_map=100,
                )

            pattern = str(seed_root / "seed_{seed}" / "map_macro.json")
            bad_path = seed_root / "seed_2" / "map_macro.json"
            bad = json.loads(bad_path.read_text(encoding="utf-8"))
            bad["support_seed"] = 1
            bad["inference_provenance"]["payload"][
                "benchmark_v2_support_seed"
            ] = 1
            bad_inference_path = Path(bad["inference_provenance"]["path"])
            bad_inference = json.loads(
                bad_inference_path.read_text(encoding="utf-8")
            )
            bad_inference["benchmark_v2_support_seed"] = 1
            self._write_json(bad_inference_path, bad_inference)
            self._write_json(bad_path, bad)
            with self.assertRaisesRegex(AggregationError, "support_seed"):
                aggregate_seeds(seed_root_pattern=pattern)

            self._write_maps_aggregate(
                bad_path,
                manifest,
                (3.0, 12.0),
                support_seed=2,
                shots_per_map=100,
            )
            bad = json.loads(bad_path.read_text(encoding="utf-8"))
            bad["inference_seed"] = 43
            bad["inference_provenance"]["payload"]["seed"] = 43
            bad_inference_path = Path(bad["inference_provenance"]["path"])
            bad_inference = json.loads(
                bad_inference_path.read_text(encoding="utf-8")
            )
            bad_inference["seed"] = 43
            self._write_json(bad_inference_path, bad_inference)
            self._write_json(bad_path, bad)
            with self.assertRaisesRegex(AggregationError, "inference seed mismatch"):
                aggregate_seeds(seed_root_pattern=pattern)

    def test_localization_summary_contract_helper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            inference_manifest = root / "inference_manifest.json"
            inference_payload = {
                "benchmark_v2_manifest": str(manifest),
                "benchmark_v2_split": "seen_discrete_test",
                "benchmark_v2_support_seed": None,
                "benchmark_v2_shots_per_map": None,
                "maps": ["map_a", "map_b"],
                "sample_count": 2,
                "checkpoint": "checkpoint.safetensors",
                "ckpt_path": "checkpoint.safetensors",
                "seed": 0,
            }
            self._write_json(inference_manifest, inference_payload)
            summary = build_localization_summary(
                manifest=manifest,
                split="seen_discrete_test",
                maps=["map_a", "map_b"],
                per_map={
                    "map_a": {"L2_5D": 1.0, "XY_Dist": 2.0, "ckpt_path": "model"},
                    "map_b": {"L2_5D": 3.0, "XY_Dist": 4.0, "ckpt_path": "model"},
                },
                metrics_macro_map={"L2_5D": 2.0, "XY_Dist": 3.0},
                inference_provenance={
                    "path": str(inference_manifest),
                    "payload": inference_payload,
                },
                checkpoint="checkpoint.safetensors",
                seed=0,
                support_seed=None,
                shots_per_map=None,
                sample_count=2,
            )

            self.assertEqual(summary["manifest"], str(manifest.resolve()))
            self.assertEqual(summary["split"], "seen_discrete_test")
            self.assertEqual(summary["kind"], "localization")
            self.assertEqual(summary["maps"], ["map_a", "map_b"])
            self.assertEqual(summary["per_map"]["map_a"]["L2_5D"], 1.0)
            self.assertEqual(summary["per_map"]["map_a"]["ckpt_path"], "model")
            self.assertEqual(summary["metrics_macro_map"]["XY_Dist"], 3.0)
            self.assertEqual(
                summary["inference_provenance"]["path"],
                str(inference_manifest.resolve()),
            )
            self.assertEqual(summary["checkpoint"], "checkpoint.safetensors")
            self.assertEqual(summary["seed"], 0)
            self.assertIsNone(summary["support_seed"])
            self.assertIsNone(summary["shots_per_map"])
            self.assertEqual(summary["sample_count"], 2)

            with self.assertRaisesRegex(AggregationError, "non-empty"):
                validate_localization_result_coverage(
                    [],
                    ["map_a", "map_b"],
                    [
                        {"map": "map_a", "file_frame": "frame_a"},
                        {"map": "map_b", "file_frame": "frame_b"},
                    ],
                )
            with self.assertRaisesRegex(AggregationError, "coverage mismatch"):
                validate_localization_result_coverage(
                    [
                        {"map": "map_a", "file_frame": "wrong_frame"},
                        {"map": "map_b", "file_frame": "frame_b"},
                    ],
                    ["map_a", "map_b"],
                    [
                        {"map": "map_a", "file_frame": "frame_a"},
                        {"map": "map_b", "file_frame": "frame_b"},
                    ],
                )

    def test_five_localization_summaries_report_support_selection_ci(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            seed_root = root / "loc_results"
            for seed in range(5):
                value = float(seed + 1)
                checkpoint = f"checkpoint_{seed}.safetensors"
                inference_manifest = (
                    seed_root / f"seed_{seed}" / "inference_manifest.json"
                )
                inference_payload = {
                    "benchmark_v2_manifest": str(manifest),
                    "benchmark_v2_split": "crossmap_query_test",
                    "benchmark_v2_support_seed": seed,
                    "benchmark_v2_shots_per_map": 100,
                    "maps": ["map_c", "map_d"],
                    "sample_count": 8,
                    "checkpoint": checkpoint,
                    "ckpt_path": checkpoint,
                    "seed": 42,
                }
                self._write_json(inference_manifest, inference_payload)
                summary = build_localization_summary(
                    manifest=manifest,
                    split="crossmap_query_test",
                    maps=["map_c", "map_d"],
                    per_map={
                        "map_c": {"L2_5D": value, "XY_Dist": value + 1.0},
                        "map_d": {"L2_5D": value + 2.0, "XY_Dist": value + 3.0},
                    },
                    metrics_macro_map={
                        "L2_5D": value + 1.0,
                        "XY_Dist": value + 2.0,
                    },
                    inference_provenance={
                        "path": str(inference_manifest),
                        "payload": inference_payload,
                    },
                    checkpoint=checkpoint,
                    seed=42,
                    support_seed=seed,
                    shots_per_map=100,
                    sample_count=8,
                )
                self._write_json(
                    seed_root / f"seed_{seed}" / "benchmark_csgo_v2_loc.json",
                    summary,
                )

            result = aggregate_seeds(
                seed_root_pattern=str(
                    seed_root / "seed_{seed}" / "benchmark_csgo_v2_loc.json"
                )
            )

            metric = result["metrics_support_selection"]["L2_5D"]
            self.assertEqual(result["kind"], "localization")
            self.assertEqual(result["split"], "crossmap_query_test")
            self.assertEqual(result["maps"], ["map_c", "map_d"])
            self.assertEqual(result["seeds"], [0, 1, 2, 3, 4])
            self.assertEqual(result["inference_seed"], 42)
            self.assertEqual(result["shots_per_map"], 100)
            self.assertEqual(metric["n"], 5)
            self.assertEqual(metric["mean"], 4.0)
            self.assertAlmostEqual(metric["sample_std"], math.sqrt(2.5))
            self.assertEqual(metric["t_critical"], 2.7764451051977987)

            mislabeled_path = (
                seed_root / "seed_2" / "benchmark_csgo_v2_loc.json"
            )
            mislabeled = json.loads(mislabeled_path.read_text(encoding="utf-8"))
            mislabeled["support_seed"] = 1
            mislabeled["inference_provenance"]["payload"][
                "benchmark_v2_support_seed"
            ] = 1
            self._write_json(mislabeled_path, mislabeled)
            with self.assertRaisesRegex(AggregationError, "support_seed"):
                aggregate_seeds(
                    seed_root_pattern=str(
                        seed_root / "seed_{seed}" / "benchmark_csgo_v2_loc.json"
                    )
                )

    def test_localization_seed_rejects_provenance_or_shot_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            seed_root = root / "loc_results"
            for seed in range(5):
                checkpoint = f"checkpoint_{seed}.safetensors"
                inference_manifest = (
                    seed_root / f"seed_{seed}" / "inference_manifest.json"
                )
                inference_payload = {
                    "benchmark_v2_manifest": str(manifest),
                    "benchmark_v2_split": "crossmap_query_test",
                    "benchmark_v2_support_seed": seed,
                    "benchmark_v2_shots_per_map": 100,
                    "maps": ["map_c", "map_d"],
                    "sample_count": 2,
                    "checkpoint": checkpoint,
                    "ckpt_path": checkpoint,
                    "seed": 42,
                }
                self._write_json(inference_manifest, inference_payload)
                summary = build_localization_summary(
                    manifest=manifest,
                    split="crossmap_query_test",
                    maps=["map_c", "map_d"],
                    per_map={
                        "map_c": {"L2_5D": float(seed + 1)},
                        "map_d": {"L2_5D": float(seed + 3)},
                    },
                    metrics_macro_map={"L2_5D": float(seed + 2)},
                    inference_provenance={
                        "path": str(inference_manifest),
                        "payload": inference_payload,
                    },
                    checkpoint=checkpoint,
                    seed=42,
                    support_seed=seed,
                    shots_per_map=100,
                    sample_count=2,
                )
                self._write_json(
                    seed_root / f"seed_{seed}" / "benchmark_csgo_v2_loc.json",
                    summary,
                )

            bad_path = seed_root / "seed_3" / "benchmark_csgo_v2_loc.json"
            bad = json.loads(bad_path.read_text(encoding="utf-8"))
            bad["inference_provenance"]["payload"]["maps"] = ["map_d", "map_c"]
            self._write_json(bad_path, bad)
            with self.assertRaisesRegex(AggregationError, "provenance maps"):
                aggregate_seeds(
                    seed_root_pattern=str(
                        seed_root / "seed_{seed}" / "benchmark_csgo_v2_loc.json"
                    )
                )

            bad["inference_provenance"]["payload"]["maps"] = ["map_c", "map_d"]
            bad["shots_per_map"] = 50
            bad["inference_provenance"]["payload"]["benchmark_v2_shots_per_map"] = 50
            self._write_json(bad_path, bad)
            with self.assertRaisesRegex(AggregationError, "shots_per_map"):
                aggregate_seeds(
                    seed_root_pattern=str(
                        seed_root / "seed_{seed}" / "benchmark_csgo_v2_loc.json"
                    )
                )

            bad["shots_per_map"] = 100
            bad["inference_provenance"]["payload"][
                "benchmark_v2_shots_per_map"
            ] = 100
            bad["seed"] = 43
            bad["inference_provenance"]["payload"]["seed"] = 43
            self._write_json(bad_path, bad)
            with self.assertRaisesRegex(AggregationError, "inference seed mismatch"):
                aggregate_seeds(
                    seed_root_pattern=str(
                        seed_root / "seed_{seed}" / "benchmark_csgo_v2_loc.json"
                    )
                )

    def test_maps_cli_rejects_localization_kind(self):
        parser = _build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parser.parse_args(
                    [
                        "maps",
                        "--manifest",
                        "manifest.json",
                        "--split",
                        "seen_discrete_test",
                        "--input_root",
                        "results",
                        "--kind",
                        "localization",
                    ]
                )
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
