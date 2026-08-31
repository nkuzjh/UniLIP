import csv
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from PIL import Image

from scripts.visualize_csgo_benchmark_v2_extrema import (
    TargetRow,
    group_target_rows,
    main,
    select_target_rows,
)


class VisualizeCsgoBenchmarkV2ExtremaTests(unittest.TestCase):
    def test_threshold_is_strictly_less_than(self):
        summary = [
            {
                "map": "map_a",
                "z_min": "-1",
                "z_min_frame_count": "19",
                "z_max": "9",
                "z_max_frame_count": "20",
            }
        ]
        review = [self._review("map_a", "min", -1, 1, frame) for frame in range(1, 20)]
        review.append(self._review("map_a", "max", 9, 1, 2))

        selected = select_target_rows(summary, review, tie_count_threshold=20)

        self.assertEqual({row.bound for row in selected}, {"min"})
        self.assertEqual({row.frame for row in selected}, set(range(1, 20)))

    def test_grouping_splits_on_key_and_nonconsecutive_frames(self):
        rows = [
            self._target("map_a", "min", 0, 1, 1),
            self._target("map_a", "min", 0, 1, 2),
            self._target("map_a", "min", 0, 1, 4),
            self._target("map_a", "min", 0, 1, 5),
            self._target("map_a", "min", 1, 1, 6),
            self._target("map_a", "min", 0, 2, 6),
        ]

        groups = group_target_rows(rows, extra_radius=8)

        self.assertEqual(
            [(group.file_num, group.z, group.target_frames) for group in groups],
            [
                (1, Decimal("0"), (1, 2)),
                (1, Decimal("0"), (4, 5)),
                (1, Decimal("1"), (6,)),
                (2, Decimal("0"), (6,)),
            ],
        )

    def test_even_group_uses_lower_middle_and_radius_formula(self):
        rows = [
            self._target("map_a", "max", 10, 7, frame) for frame in (20, 21, 22, 23)
        ]

        group = group_target_rows(rows, extra_radius=8)[0]

        self.assertEqual(group.start, 20)
        self.assertEqual(group.end, 23)
        self.assertEqual(group.center, 21)
        self.assertEqual(group.radius, max(21 - 20, 23 - 21) + 8)

    def test_dry_run_writes_deterministic_indexes_without_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            map_root = source_root / "map_a"
            image_root = map_root / "imgs"
            image_root.mkdir(parents=True)
            positions = []
            for frame in range(1, 6):
                file_frame = f"file_num3_frame_{frame}"
                positions.append(
                    {
                        "map": "map_a",
                        "file_frame": file_frame,
                        "x": frame,
                        "y": frame + 1,
                        "z": 10 if frame in (2, 3) else 20,
                        "angle_h": 0.5,
                        "angle_v": 1.0,
                    }
                )
                Image.new("RGB", (12, 8), (frame * 20, 10, 5)).save(
                    image_root / f"{file_frame}.jpg"
                )
            (map_root / "positions.json").write_text(
                json.dumps(positions), encoding="utf-8"
            )

            summary_path = root / "z_extrema_summary.csv"
            with summary_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "map",
                        "z_min",
                        "z_min_frame_count",
                        "z_max",
                        "z_max_frame_count",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "map": "map_a",
                        "z_min": 10,
                        "z_min_frame_count": 2,
                        "z_max": 20,
                        "z_max_frame_count": 3,
                    }
                )

            review_path = root / "z_extrema_review.csv"
            with review_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "map",
                        "bound",
                        "z",
                        "file_num",
                        "frame",
                        "file_frame",
                    ],
                )
                writer.writeheader()
                writer.writerow(self._review("map_a", "min", 10, 3, 2))
                writer.writerow(self._review("map_a", "min", 10, 3, 3))
                writer.writerow(self._review("map_a", "max", 20, 3, 1))
                writer.writerow(self._review("map_a", "max", 20, 3, 4))
                writer.writerow(self._review("map_a", "max", 20, 3, 5))

            candidate_path = root / "coordinate_candidates.jsonl"
            candidate = {
                "map": "map_a",
                "file_frame": "file_num3_frame_1",
                "record_id": "3",
                "frame_id": 1,
                "coordinates": {
                    "x": 1,
                    "y": 2,
                    "z": 20,
                    "angle_h": 0.5,
                    "angle_v": 1.0,
                },
                "reasons": ["trajectory_jump_xy"],
            }
            candidate_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
            output_dir = root / "output"

            exit_code = main(
                [
                    "--summary",
                    str(summary_path),
                    "--review",
                    str(review_path),
                    "--candidates",
                    str(candidate_path),
                    "--source-root",
                    str(source_root),
                    "--maps",
                    "map_a",
                    "--dry-run",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "index.csv").is_file())
            self.assertTrue((output_dir / "index.json").is_file())
            self.assertFalse((output_dir / "csgo_benchmark_v2_extrema.pdf").exists())
            self.assertEqual(len(list(output_dir.glob("*.png"))), 0)
            index = json.loads((output_dir / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(index), 3)
            self.assertEqual(index[0]["target_frames"], [2, 3])
            self.assertEqual(index[0]["center"], 2)
            self.assertEqual(index[0]["radius"], 9)
            self.assertEqual(index[0]["context_count"], 5)
            self.assertEqual(index[1]["target_frames"], [1])
            self.assertEqual(index[2]["target_frames"], [4, 5])

    @staticmethod
    def _target(map_name, bound, z, file_num, frame):
        return TargetRow(
            map_name=map_name,
            bound=bound,
            z=Decimal(str(z)),
            file_num=file_num,
            frame=frame,
            file_frame=f"file_num{file_num}_frame_{frame}",
        )

    @staticmethod
    def _review(map_name, bound, z, file_num, frame):
        return {
            "map": map_name,
            "bound": bound,
            "z": str(z),
            "file_num": str(file_num),
            "frame": str(frame),
            "file_frame": f"file_num{file_num}_frame_{frame}",
        }


if __name__ == "__main__":
    unittest.main()
