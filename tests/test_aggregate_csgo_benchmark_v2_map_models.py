import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_csgo_benchmark_v2_metrics import (
    AggregationError,
    _build_parser,
    aggregate_map_models,
)


class AggregateCsgoBenchmarkV2MapModelsTests(unittest.TestCase):
    maps = ["map_c", "map_d"]

    def _write_json(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _make_manifest(self, root: Path) -> Path:
        manifest = root / "benchmark_manifest.json"
        self._write_json(
            manifest,
            {
                "schema_version": 1,
                "benchmark_id": "csgo_benchmark_v2",
                "protocol": {
                    "seen_maps": ["map_a"],
                    "crossmap_maps": self.maps,
                },
            },
        )
        return manifest

    def _write_evaluator_inputs(
        self,
        root: Path,
        manifest: Path,
        *,
        split: str = "crossmap_query_test",
        kind: str = "discrete",
        sample_count: int = 2,
        support_seed: int | None = 0,
        shots_per_map: int | None = 100,
        inference_seed: int = 42,
    ) -> str:
        paths = []
        for index, map_name in enumerate(self.maps):
            inference_path = root / f"inference_{map_name}.json"
            checkpoint = f"checkpoint_{map_name}.safetensors"
            inference_payload = {
                "benchmark_v2_manifest": str(manifest),
                "benchmark_v2_split": split,
                "benchmark_v2_support_seed": support_seed,
                "benchmark_v2_shots_per_map": shots_per_map,
                "maps": [map_name],
                "sample_count": sample_count,
                "checkpoint": checkpoint,
                "ckpt_path": checkpoint,
                "seed": inference_seed,
            }
            self._write_json(inference_path, inference_payload)
            result_path = root / f"result_{map_name}.json"
            self._write_json(
                result_path,
                {
                    "map_name": map_name,
                    "benchmark_v2_manifest": str(manifest),
                    "benchmark_v2_split": split,
                    "kind": kind,
                    "metrics_ordered": {
                        "PSNR": float(index + 1),
                        "SSIM": float(index + 11),
                        "Common_Count": sample_count,
                        "not_a_metric": None,
                    },
                    "inference_provenance": {
                        "path": str(inference_path),
                        "payload": inference_payload,
                    },
                },
            )
            paths.append(result_path)
        return str(root / "result_{map}.json")

    def _write_localization_inputs(
        self,
        root: Path,
        manifest: Path,
        *,
        split: str = "crossmap_query_test",
        kind: str = "localization",
        sample_count: int = 2,
        support_seed: int | None = 0,
        shots_per_map: int | None = 100,
        inference_seed: int = 42,
    ) -> str:
        for index, map_name in enumerate(self.maps):
            inference_path = root / f"loc_inference_{map_name}.json"
            checkpoint = f"loc_checkpoint_{map_name}.safetensors"
            inference_payload = {
                "benchmark_v2_manifest": str(manifest),
                "benchmark_v2_split": split,
                "benchmark_v2_support_seed": support_seed,
                "benchmark_v2_shots_per_map": shots_per_map,
                "maps": [map_name],
                "sample_count": sample_count,
                "checkpoint": checkpoint,
                "ckpt_path": checkpoint,
                "seed": inference_seed,
            }
            self._write_json(inference_path, inference_payload)
            value = float(index + 1)
            self._write_json(
                root / f"loc_{map_name}.json",
                {
                    "manifest": str(manifest),
                    "split": split,
                    "kind": kind,
                    "maps": [map_name],
                    "per_map": {
                        map_name: {
                            "L2_5D": value,
                            "XY_Dist": value + 1.0,
                            "ckpt_path": checkpoint,
                        }
                    },
                    "metrics_macro_map": {
                        "L2_5D": value,
                        "XY_Dist": value + 1.0,
                    },
                    "inference_provenance": {
                        "path": str(inference_path),
                        "payload": inference_payload,
                    },
                    "checkpoint": checkpoint,
                    "seed": inference_seed,
                    "support_seed": support_seed,
                    "shots_per_map": shots_per_map,
                    "sample_count": sample_count,
                },
            )
        return str(root / "loc_{map}.json")

    def test_parser_exposes_map_models_subcommand(self):
        parser = _build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            args = parser.parse_args(
                [
                    "map-models",
                    "--manifest",
                    "manifest.json",
                    "--split",
                    "crossmap_query_test",
                    "--kind",
                    "localization",
                    "--input_pattern",
                    "results/{map}.json",
                    "--output",
                    "summary.json",
                ]
            )
        self.assertEqual(args.command, "map-models")
        self.assertEqual(args.input_pattern, "results/{map}.json")
        self.assertEqual(args.output, "summary.json")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parser.parse_args(
                    [
                        "map-models",
                        "--manifest",
                        "manifest.json",
                        "--split",
                        "crossmap_query_test",
                        "--kind",
                        "discrete",
                        "--input_pattern",
                        "results/{map}.json",
                    ]
                )
        self.assertEqual(raised.exception.code, 2)

    def test_discrete_map_models_allows_distinct_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            pattern = self._write_evaluator_inputs(root, manifest)

            output = root / "summary" / "discrete.json"
            result = aggregate_map_models(
                manifest=manifest,
                split="crossmap_query_test",
                kind="discrete",
                input_pattern=pattern,
                output=output,
            )

            self.assertEqual(result["aggregation"], "map_specific_models")
            self.assertEqual(result["maps"], self.maps)
            self.assertEqual(result["metrics_macro_map"]["PSNR"], 1.5)
            self.assertEqual(result["metrics_macro_map"]["SSIM"], 11.5)
            self.assertEqual(result["inference_seed"], 42)
            self.assertEqual(result["support_seed"], 0)
            self.assertEqual(result["shots_per_map"], 100)
            self.assertEqual(result["sample_count"], 4)
            self.assertNotEqual(
                result["model_context_by_map"]["map_c"]["checkpoint"],
                result["model_context_by_map"]["map_d"]["checkpoint"],
            )
            self.assertEqual(
                result["model_context_by_map"]["map_c"]["inference_seed"], 42
            )
            self.assertTrue(output.is_file())
            self.assertEqual(json.loads(output.read_text()), result)

    def test_continuous_map_models_use_crossmap_continuous_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            pattern = self._write_evaluator_inputs(
                root,
                manifest,
                split="crossmap_continuous",
                kind="continuous",
                sample_count=3,
                support_seed=None,
                shots_per_map=None,
            )

            result = aggregate_map_models(
                manifest=manifest,
                split="crossmap_continuous",
                kind="continuous",
                input_pattern=pattern,
                output=root / "summary" / "continuous.json",
            )

            self.assertEqual(result["sample_count"], 6)
            self.assertEqual(result["inference_seed"], 42)
            self.assertIsNone(result["support_seed"])
            self.assertIsNone(result["shots_per_map"])
            self.assertIsNone(
                result["model_context_by_map"]["map_c"]["support_seed"]
            )

    def test_localization_map_models_validate_single_map_summaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            pattern = self._write_localization_inputs(root, manifest)

            result = aggregate_map_models(
                manifest=manifest,
                split="crossmap_query_test",
                kind="localization",
                input_pattern=pattern,
                output=root / "localization.json",
            )

            self.assertEqual(result["kind"], "localization")
            self.assertEqual(result["metrics_macro_map"]["L2_5D"], 1.5)
            self.assertEqual(result["metrics_macro_map"]["XY_Dist"], 2.5)
            self.assertEqual(result["inference_seed"], 42)
            self.assertEqual(result["support_seed"], 0)
            self.assertEqual(result["shots_per_map"], 100)
            self.assertEqual(result["sample_count"], 4)
            self.assertEqual(
                result["model_context_by_map"]["map_d"]["inference_provenance"][
                    "payload"
                ]["maps"],
                ["map_d"],
            )

    def test_rejects_missing_placeholder_and_missing_rendered_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            with self.assertRaisesRegex(AggregationError, "placeholder"):
                aggregate_map_models(
                    manifest=manifest,
                    split="crossmap_query_test",
                    kind="discrete",
                    input_pattern=str(root / "result.json"),
                    output=root / "unused.json",
                )

            self._write_evaluator_inputs(root / "partial", manifest)
            (root / "partial" / "result_map_d.json").unlink()
            with self.assertRaisesRegex(AggregationError, "map 'map_d'"):
                aggregate_map_models(
                    manifest=manifest,
                    split="crossmap_query_test",
                    kind="discrete",
                    input_pattern=str(root / "partial" / "result_{map}.json"),
                    output=root / "unused.json",
                )

    def test_rejects_wrong_target_map_in_evaluator_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            pattern = self._write_evaluator_inputs(root, manifest)
            result_path = root / "result_map_d.json"
            payload = json.loads(result_path.read_text())
            payload["map_name"] = "map_c"
            self._write_json(result_path, payload)

            with self.assertRaisesRegex(AggregationError, "map_name"):
                aggregate_map_models(
                    manifest=manifest,
                    split="crossmap_query_test",
                    kind="discrete",
                    input_pattern=pattern,
                    output=root / "unused.json",
                )

    def test_rejects_wrong_target_map_in_localization_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            pattern = self._write_localization_inputs(root, manifest)
            result_path = root / "loc_map_d.json"
            payload = json.loads(result_path.read_text())
            payload["maps"] = ["map_c"]
            self._write_json(result_path, payload)

            with self.assertRaisesRegex(AggregationError, "target map"):
                aggregate_map_models(
                    manifest=manifest,
                    split="crossmap_query_test",
                    kind="localization",
                    input_pattern=pattern,
                    output=root / "unused.json",
                )

    def test_rejects_mixed_support_seed_shots_and_inference_seed(self):
        cases = (
            ("benchmark_v2_support_seed", 1, "support_seed mismatch"),
            ("benchmark_v2_shots_per_map", 50, "shots_per_map mismatch"),
            ("seed", 43, "inference_seed mismatch"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    manifest = self._make_manifest(root)
                    pattern = self._write_evaluator_inputs(root, manifest)
                    result_path = root / "result_map_d.json"
                    result = json.loads(result_path.read_text())
                    inference_path = Path(
                        result["inference_provenance"]["path"]
                    )
                    inference = json.loads(inference_path.read_text())
                    result["inference_provenance"]["payload"][field] = value
                    inference[field] = value
                    self._write_json(inference_path, inference)
                    self._write_json(result_path, result)

                    with self.assertRaisesRegex(AggregationError, message):
                        aggregate_map_models(
                            manifest=manifest,
                            split="crossmap_query_test",
                            kind="discrete",
                            input_pattern=pattern,
                            output=root / "unused.json",
                        )

    def test_rejects_wrong_split_and_kind(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            pattern = self._write_evaluator_inputs(root, manifest)
            with self.assertRaisesRegex(AggregationError, "continuous.*requires split"):
                aggregate_map_models(
                    manifest=manifest,
                    split="crossmap_query_test",
                    kind="continuous",
                    input_pattern=pattern,
                    output=root / "unused.json",
                )

            continuous_pattern = self._write_evaluator_inputs(
                root / "continuous",
                manifest,
                split="crossmap_continuous",
                kind="continuous",
            )
            with self.assertRaisesRegex(AggregationError, "discrete.*requires split"):
                aggregate_map_models(
                    manifest=manifest,
                    split="crossmap_continuous",
                    kind="discrete",
                    input_pattern=continuous_pattern,
                    output=root / "unused.json",
                )

            discrete_result = root / "result_map_d.json"
            discrete_payload = json.loads(discrete_result.read_text())
            discrete_payload["kind"] = "continuous"
            self._write_json(discrete_result, discrete_payload)
            with self.assertRaisesRegex(AggregationError, "kind"):
                aggregate_map_models(
                    manifest=manifest,
                    split="crossmap_query_test",
                    kind="discrete",
                    input_pattern=pattern,
                    output=root / "unused.json",
                )

            loc_pattern = self._write_localization_inputs(root / "loc", manifest)
            with self.assertRaisesRegex(AggregationError, "requires split"):
                aggregate_map_models(
                    manifest=manifest,
                    split="crossmap_continuous",
                    kind="localization",
                    input_pattern=loc_pattern,
                    output=root / "unused.json",
                )

    def test_requires_explicit_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            pattern = self._write_evaluator_inputs(root, manifest)

            with self.assertRaisesRegex(AggregationError, "output is required"):
                aggregate_map_models(
                    manifest=manifest,
                    split="crossmap_query_test",
                    kind="discrete",
                    input_pattern=pattern,
                    output=None,
                )

    def test_rejects_malformed_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            pattern = self._write_evaluator_inputs(root, manifest)
            result_path = root / "result_map_c.json"
            result = json.loads(result_path.read_text())
            result.pop("inference_provenance")
            self._write_json(result_path, result)

            with self.assertRaisesRegex(AggregationError, "inference_provenance"):
                aggregate_map_models(
                    manifest=manifest,
                    split="crossmap_query_test",
                    kind="discrete",
                    input_pattern=pattern,
                    output=root / "unused.json",
                )

    def test_rejects_duplicate_resolved_input_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._make_manifest(root)
            shared = root / "shared"
            shared.mkdir()
            self._write_json(shared / "result.json", {})
            (root / "map_c").symlink_to(shared, target_is_directory=True)
            (root / "map_d").symlink_to(shared, target_is_directory=True)

            with self.assertRaisesRegex(AggregationError, "same file"):
                aggregate_map_models(
                    manifest=manifest,
                    split="crossmap_query_test",
                    kind="discrete",
                    input_pattern=str(root / "{map}" / "result.json"),
                    output=root / "unused.json",
                )


if __name__ == "__main__":
    unittest.main()
