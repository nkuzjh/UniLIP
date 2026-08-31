import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.export_csgo_benchmark_v2_z_comparison import (
    ExportError,
    main,
    run_export,
)


class ExportCsgoBenchmarkV2ZComparisonTests(unittest.TestCase):
    def test_top_level_exclusions_compare_extrema_and_record_frame_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._write_workspace(root)
            decisions_path = self._write_decisions(
                root,
                {
                    "exclude_records": {"seen_map": ["2"]},
                    "exclude_file_frames": {
                        "seen_map": ["file_num2_frame_2"],
                        "cross_map": ["file_num3_frame_1"],
                    },
                },
            )
            output_dir = root / "comparison"

            manifest = run_export(
                config_path, decisions_path, output_dir, cwd=root
            )

            comparison = self._read_csv(output_dir / "z_extrema_comparison.csv")
            self.assertEqual(
                comparison[0],
                {
                    "setting": "seen",
                    "map": "seen_map",
                    "source_rows": "7",
                    "excluded_rows": "3",
                    "retained_rows": "4",
                    "raw_z_min": "-5",
                    "raw_z_min_frame_count": "1",
                    "raw_z_max": "30",
                    "raw_z_max_frame_count": "1",
                    "clean_z_min": "-5",
                    "clean_z_min_frame_count": "1",
                    "clean_z_max": "20",
                    "clean_z_max_frame_count": "3",
                    "z_min_changed": "False",
                    "z_max_changed": "True",
                },
            )
            self.assertEqual(
                comparison[1]["excluded_rows"], "1"
            )
            self.assertEqual(comparison[1]["clean_z_min"], "5")
            self.assertEqual(comparison[1]["clean_z_min_frame_count"], "3")

            clean_review = self._read_csv(output_dir / "z_extrema_review.csv")
            self.assertEqual(
                [row["file_frame"] for row in clean_review],
                [
                    "file_num1_frame_1",
                    "file_num1_frame_2",
                    "file_num1_frame_3",
                    "file_num1_frame_4",
                    "file_num3_frame_2",
                    "file_num3_frame_3",
                    "file_num4_frame_1",
                    "file_num4_frame_2",
                ],
            )
            self.assertEqual(
                {"source_image_path", "source_index"} - set(clean_review[0]),
                set(),
            )
            self.assertEqual(clean_review[0]["source_index"], "0")

            raw_review = self._read_csv(
                output_dir / "z_extrema_raw_review.csv"
            )
            self.assertEqual(
                {(row["map"], row["bound"], row["file_frame"]) for row in raw_review},
                {
                    ("seen_map", "min", "file_num1_frame_1"),
                    ("seen_map", "max", "file_num2_frame_3"),
                    ("cross_map", "min", "file_num3_frame_1"),
                    ("cross_map", "max", "file_num4_frame_2"),
                },
            )

            summary = self._read_csv(output_dir / "z_extrema_summary.csv")
            self.assertTrue(
                {
                    "map",
                    "z_min",
                    "z_min_frame_count",
                    "z_max",
                    "z_max_frame_count",
                }.issubset(summary[0])
            )
            self.assertEqual(manifest["exclusions"]["overlap"]["rows"], 1)
            self.assertEqual(
                manifest["exclusions"]["by_map"]["seen_map"]["overlap_rows"],
                1,
            )
            self.assertEqual(manifest["exclusions"]["hits"]["effective_rows"], 4)
            self.assertEqual(
                manifest["inputs"]["source"]["positions_sha256"].keys(),
                {"seen_map", "cross_map"},
            )
            self.assertEqual(
                manifest["outputs"]["z_extrema_review"]["rows"],
                len(clean_review),
            )

    def test_nested_formal_decisions_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._write_workspace(root)
            decisions_path = self._write_decisions(
                root,
                {
                    "coordinate_candidates": {
                        "exclude_records": {"cross_map": ["4"]},
                        "exclude_file_frames": {},
                    }
                },
            )
            output_dir = root / "nested-output"

            exit_code = main(
                [
                    "--config",
                    str(config_path),
                    "--decisions",
                    str(decisions_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            rows = self._read_csv(output_dir / "z_extrema_comparison.csv")
            self.assertEqual(rows[1]["excluded_rows"], "2")
            self.assertEqual(rows[1]["clean_z_max"], "5")
            self.assertEqual(
                json.loads(
                    (output_dir / "z_extrema_comparison_manifest.json").read_text(
                        encoding="ascii"
                    )
                )["inputs"]["decisions"]["format"],
                "coordinate_candidates",
            )

    def test_invalid_exclusions_and_source_values_fail(self):
        cases = (
            (
                "unknown map",
                {"exclude_records": {"unknown_map": ["1"]}},
            ),
            (
                "malformed record",
                {"exclude_records": {"seen_map": ["one"]}},
            ),
            (
                "malformed frame",
                {
                    "exclude_file_frames": {
                        "seen_map": ["file_num1_frame_bad"]
                    }
                },
            ),
            (
                "missing record",
                {"exclude_records": {"seen_map": ["99"]}},
            ),
            (
                "missing frame",
                {
                    "exclude_file_frames": {
                        "seen_map": ["file_num1_frame_99"]
                    }
                },
            ),
            (
                "duplicate frame",
                {
                    "exclude_file_frames": {
                        "seen_map": [
                            "file_num1_frame_1",
                            "file_num1_frame_1",
                        ]
                    }
                },
            ),
        )
        for name, exclusion in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config_path = self._write_workspace(root)
                decisions_path = self._write_decisions(root, exclusion)
                with self.assertRaises(ExportError):
                    run_export(
                        config_path,
                        decisions_path,
                        root / "output",
                        cwd=root,
                    )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._write_workspace(root, nonfinite=True)
            decisions_path = self._write_decisions(root, {})
            with self.assertRaises(ExportError):
                run_export(config_path, decisions_path, root / "output", cwd=root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._write_workspace(root)
            decisions_path = self._write_decisions(
                root,
                {
                    "exclude_records": {
                        "seen_map": ["1", "2"],
                    }
                },
            )
            with self.assertRaises(ExportError):
                run_export(config_path, decisions_path, root / "output", cwd=root)

    def test_existing_output_is_protected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._write_workspace(root)
            decisions_path = self._write_decisions(root, {})
            output_dir = root / "output"
            run_export(config_path, decisions_path, output_dir, cwd=root)
            comparison_path = output_dir / "z_extrema_comparison.csv"
            before = comparison_path.read_bytes()

            with self.assertRaises(ExportError):
                run_export(config_path, decisions_path, output_dir, cwd=root)
            self.assertEqual(comparison_path.read_bytes(), before)

            run_export(
                config_path,
                decisions_path,
                output_dir,
                overwrite=True,
                cwd=root,
            )
            self.assertTrue(comparison_path.is_file())

    @staticmethod
    def _write_workspace(root: Path, *, nonfinite: bool = False) -> Path:
        source_root = root / "source"
        rows_by_map = {
            "seen_map": [
                (1, 1, -5),
                (1, 2, 20),
                (1, 3, 20),
                (1, 4, 20),
                (2, 1, 10),
                (2, 2, 10),
                (2, 3, 30),
            ],
            "cross_map": [
                (3, 1, 0),
                (3, 2, 5),
                (3, 3, 5),
                (4, 1, 5),
                (4, 2, 10),
            ],
        }
        for map_name, values in rows_by_map.items():
            map_root = source_root / map_name
            map_root.mkdir(parents=True)
            rows = []
            for record_id, frame_id, z in values:
                if nonfinite and map_name == "seen_map" and frame_id == 1:
                    z = float("nan")
                rows.append(
                    {
                        "map": map_name,
                        "file_frame": f"file_num{record_id}_frame_{frame_id}",
                        "x": record_id,
                        "y": frame_id,
                        "z": z,
                        "angle_h": 0.1,
                        "angle_v": 0.2,
                    }
                )
            (map_root / "positions.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )

        config = {
            "benchmark": {"id": "test_benchmark", "version": "0"},
            "paths": {"source_root": str(source_root)},
            "source": {
                "positions_file": "positions.json",
                "images_dir": "imgs",
                "image_extension": ".jpg",
                "record_regex": (
                    r"^file_num(?P<record>\d+)_frame_(?P<frame>\d+)$"
                ),
            },
            "maps": {"seen": ["seen_map"], "crossmap": ["cross_map"]},
        }
        config_path = root / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        return config_path

    @staticmethod
    def _write_decisions(root: Path, values: dict) -> Path:
        decisions_path = root / "decisions.yaml"
        decisions_path.write_text(
            yaml.safe_dump(values, sort_keys=False), encoding="utf-8"
        )
        return decisions_path

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="ascii", newline="") as handle:
            return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
