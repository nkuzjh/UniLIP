import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_csgo_benchmark_v2_review import _collect_extrema


class ExportCsgoBenchmarkV2ReviewTests(unittest.TestCase):
    def test_extrema_include_all_ties_after_decision_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory)
            rows_by_map = {
                "seen_map": [
                    {"file_frame": "file_num1_frame_1", "z": -10},
                    {"file_frame": "file_num1_frame_2", "z": 3},
                    {"file_frame": "file_num2_frame_1", "z": 3},
                    {"file_frame": "file_num9_frame_1", "z": -100},
                ],
                "cross_map": [
                    {"file_frame": "file_num4_frame_1", "z": 7},
                    {"file_frame": "file_num4_frame_2", "z": 9},
                    {"file_frame": "file_num5_frame_1", "z": 9},
                    {"file_frame": "file_num5_frame_2", "z": 10},
                    {"file_frame": "file_num6_frame_1", "z": 10},
                ],
            }
            for map_name, rows in rows_by_map.items():
                map_root = source_root / map_name
                map_root.mkdir()
                (map_root / "positions.json").write_text(
                    json.dumps(rows), encoding="utf-8"
                )

            config = {
                "source": {
                    "positions_file": "positions.json",
                    "images_dir": "imgs",
                    "image_extension": ".jpg",
                    "record_regex": r"^file_num(?P<record>\d+)_frame_(?P<frame>\d+)$",
                },
                "maps": {"seen": ["seen_map"], "crossmap": ["cross_map"]},
            }
            decisions = {
                "coordinate_candidates": {
                    "exclude_records": {"seen_map": ["9"]},
                    "exclude_file_frames": {
                        "cross_map": ["file_num4_frame_1"],
                    },
                }
            }

            extrema, summaries = _collect_extrema(config, decisions, source_root)

            self.assertEqual(
                [(row["map"], row["bound"], row["file_frame"]) for row in extrema],
                [
                    ("cross_map", "min", "file_num4_frame_2"),
                    ("cross_map", "min", "file_num5_frame_1"),
                    ("cross_map", "max", "file_num5_frame_2"),
                    ("cross_map", "max", "file_num6_frame_1"),
                    ("seen_map", "min", "file_num1_frame_1"),
                    ("seen_map", "max", "file_num1_frame_2"),
                    ("seen_map", "max", "file_num2_frame_1"),
                ],
            )
            summary_by_map = {row["map"]: row for row in summaries}
            self.assertEqual(summary_by_map["seen_map"]["excluded_rows"], 1)
            self.assertEqual(summary_by_map["seen_map"]["z_min"], -10.0)
            self.assertEqual(summary_by_map["seen_map"]["z_max_frame_count"], 2)
            self.assertEqual(summary_by_map["cross_map"]["excluded_rows"], 1)


if __name__ == "__main__":
    unittest.main()
