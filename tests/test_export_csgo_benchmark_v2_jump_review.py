import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.export_csgo_benchmark_v2_jump_review import (
    collect_jump_edges,
    select_angle_edges,
    select_medium_z_edges,
)


class ExportCsgoBenchmarkV2JumpReviewTests(unittest.TestCase):
    def test_edges_apply_exclusions_and_use_circular_yaw(self):
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory)
            map_root = source_root / "test_map"
            map_root.mkdir()
            rows = [
                self._row(1, 1, z=0, yaw=math.radians(359)),
                self._row(1, 2, z=50, yaw=math.radians(1)),
                self._row(1, 3, z=101, yaw=math.radians(130)),
                self._row(1, 5, z=401, yaw=math.radians(130)),
                self._row(1, 8, z=500, yaw=0),
                self._row(2, 1, z=0, yaw=0),
                self._row(2, 2, z=200, yaw=math.pi),
            ]
            (map_root / "positions.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )
            config = self._config()
            decisions = {
                "coordinate_candidates": {
                    "exclude_records": {"test_map": ["2"]},
                    "exclude_file_frames": {
                        "test_map": ["file_num1_frame_2"]
                    },
                }
            }

            edges = collect_jump_edges(config, decisions, source_root)

            self.assertEqual(
                [(row["frame_start"], row["frame_end"]) for row in edges],
                [(1, 3), (3, 5)],
            )
            self.assertAlmostEqual(edges[0]["z_delta"], 101.0)
            self.assertAlmostEqual(edges[0]["yaw_delta_degrees"], 131.0)
            self.assertAlmostEqual(edges[1]["z_delta"], 300.0)
            self.assertEqual(len(select_medium_z_edges(edges, 50, 300)), 2)
            self.assertEqual(len(select_angle_edges(edges, 120)), 1)

    def test_medium_z_and_angle_boundaries_are_strict(self):
        edges = [
            {"z_delta": 50.0, "angle_delta_degrees": 45.0},
            {"z_delta": 50.1, "angle_delta_degrees": 45.1},
            {"z_delta": 300.0, "angle_delta_degrees": 90.0},
            {"z_delta": 300.1, "angle_delta_degrees": 90.1},
        ]

        self.assertEqual(len(select_medium_z_edges(edges, 50, 300)), 2)
        self.assertEqual(len(select_angle_edges(edges, 45)), 3)
        self.assertEqual(len(select_angle_edges(edges, 90)), 1)

    @staticmethod
    def _row(record, frame, z, yaw):
        return {
            "file_frame": f"file_num{record}_frame_{frame}",
            "x": frame,
            "y": 0,
            "z": z,
            "angle_h": yaw,
            "angle_v": math.pi / 2,
        }

    @staticmethod
    def _config():
        return {
            "source": {
                "positions_file": "positions.json",
                "images_dir": "imgs",
                "image_extension": ".jpg",
                "record_regex": r"^file_num(?P<record>\d+)_frame_(?P<frame>\d+)$",
            },
            "maps": {"seen": ["test_map"], "crossmap": []},
            "counts": {"continuous": {"max_frame_gap": 2}},
        }


if __name__ == "__main__":
    unittest.main()
