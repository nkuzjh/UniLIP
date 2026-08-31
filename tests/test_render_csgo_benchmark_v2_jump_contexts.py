import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.render_csgo_benchmark_v2_jump_contexts import (
    Edge,
    _display_types,
    merge_edges_into_groups,
    render_jump_contexts,
)


class RenderCsgoBenchmarkV2JumpContextsTests(unittest.TestCase):
    def test_display_types_compacts_cumulative_thresholds(self):
        self.assertEqual(
            _display_types(
                (
                    "angle_gt_045",
                    "angle_gt_060",
                    "angle_gt_120",
                    "medium_z_50_300",
                )
            ),
            "50<|dZ|<=300, angle>45/60/120deg",
        )

    def test_union_group_center_radius_and_discontinuous_split(self):
        edges = [
            self._edge("test_map", "1", 10, 12, "medium_z_50_300"),
            self._edge("test_map", "1", 12, 13, "angle_gt_045"),
            self._edge("test_map", "1", 12, 13, "angle_gt_060"),
            self._edge("test_map", "1", 15, 16, "angle_gt_090"),
            self._edge("test_map", "1", 30, 31, "angle_gt_120"),
            self._edge("test_map", "1", 33, 34, "angle_gt_150"),
        ]

        groups = merge_edges_into_groups(edges)

        self.assertEqual(len(groups), 2)
        first, second = groups
        self.assertEqual(first.anomalous_frames, [10, 12, 13, 15, 16])
        self.assertEqual(first.center_frame, 13)
        self.assertEqual(first.radius, 11)
        self.assertEqual(first.frame_types[12], {"medium_z_50_300", "angle_gt_045", "angle_gt_060"})
        self.assertEqual(first.types, ["angle_gt_045", "angle_gt_060", "angle_gt_090", "medium_z_50_300"])
        self.assertEqual(second.anomalous_frames, [30, 31, 33, 34])
        self.assertEqual(second.center_frame, 31)
        self.assertEqual(second.radius, 11)

    def test_render_paginates_and_preserves_frame_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            map_root = source_root / "test_map"
            (map_root / "imgs").mkdir(parents=True)
            positions = []
            for frame in range(1, 22):
                file_frame = f"file_num1_frame_{frame}"
                positions.append(
                    {
                        "file_frame": file_frame,
                        "x": frame,
                        "y": frame + 1,
                        "z": 10 + frame,
                        "angle_h": math.radians(frame),
                        "angle_v": math.radians(45 + frame),
                    }
                )
                Image.new("RGB", (40, 30), (frame, 20, 30)).save(
                    map_root / "imgs" / f"{file_frame}.jpg"
                )
            (map_root / "positions.json").write_text(
                json.dumps(positions), encoding="utf-8"
            )

            input_dir = root / "jumps"
            input_dir.mkdir()
            self._write_csv(
                input_dir / "medium_z_jumps_50_300.csv",
                [("test_map", "1", 8, 10, "medium_z")],
            )
            self._write_csv(
                input_dir / "angle_jumps_gt_045.csv",
                [("test_map", "1", 10, 12, "angle")],
            )

            output_dir = root / "rendered"
            manifest = render_jump_contexts(
                source_root=source_root,
                output_dir=output_dir,
                input_dir=input_dir,
                categories=("medium_z", "angle"),
                angle_thresholds=(45,),
                columns=2,
                dpi=80,
                max_rows_per_image=4,
            )

            self.assertEqual(manifest["counts"]["groups_rendered"], 1)
            self.assertEqual(manifest["counts"]["pages_rendered"], 3)
            group = manifest["groups"][0]
            self.assertEqual(group["center_file_frame"], "file_num1_frame_10")
            self.assertEqual(group["radius"], 10)
            self.assertEqual(
                group["frame_types"]["file_num1_frame_10"],
                ["angle_gt_045", "medium_z_50_300"],
            )
            self.assertTrue(group["pages"][0]["path"].endswith("_part_001of003.png"))
            self.assertTrue((output_dir / group["pages"][2]["path"]).is_file())
            image = Image.open(output_dir / group["pages"][0]["path"])
            self.assertLess(image.width, 60_000)
            self.assertLess(image.height, 60_000)
            image.close()

            with (output_dir / "jump_context_groups.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(json.loads(rows[0]["png_files"])), 3)

    def test_missing_context_image_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_root = root / "source" / "test_map"
            (map_root / "imgs").mkdir(parents=True)
            positions = [
                {
                    "file_frame": "file_num1_frame_1",
                    "x": 0,
                    "y": 0,
                    "z": 0,
                    "angle_h": 0,
                    "angle_v": 0,
                },
                {
                    "file_frame": "file_num1_frame_2",
                    "x": 1,
                    "y": 1,
                    "z": 1,
                    "angle_h": 0,
                    "angle_v": 0,
                },
            ]
            (map_root / "positions.json").write_text(json.dumps(positions), encoding="utf-8")
            Image.new("RGB", (20, 20), "white").save(
                map_root / "imgs" / "file_num1_frame_1.jpg"
            )
            input_dir = root / "jumps"
            input_dir.mkdir()
            self._write_csv(
                input_dir / "medium_z_jumps_50_300.csv",
                [("test_map", "1", 1, 2, "medium_z")],
            )

            with self.assertRaises(FileNotFoundError):
                render_jump_contexts(
                    source_root=root / "source",
                    output_dir=root / "out",
                    input_dir=input_dir,
                    categories=("medium_z",),
                )

    @staticmethod
    def _edge(map_name, file_num, start, end, label):
        return Edge(
            map_name=map_name,
            file_num=file_num,
            frame_start=start,
            frame_end=end,
            file_frame_start=f"file_num{file_num}_frame_{start}",
            file_frame_end=f"file_num{file_num}_frame_{end}",
            types={label},
            source_files={"test.csv"},
        )

    @staticmethod
    def _write_csv(path, rows):
        fields = [
            "map",
            "file_num",
            "frame_start",
            "frame_end",
            "frame_gap",
            "file_frame_start",
            "file_frame_end",
            "xy_delta",
            "z_delta",
            "yaw_delta_degrees",
            "pitch_delta_degrees",
            "angle_delta_degrees",
            "source_image_start",
            "source_image_end",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for map_name, file_num, start, end, kind in rows:
                writer.writerow(
                    {
                        "map": map_name,
                        "file_num": file_num,
                        "frame_start": start,
                        "frame_end": end,
                        "frame_gap": end - start,
                        "file_frame_start": f"file_num{file_num}_frame_{start}",
                        "file_frame_end": f"file_num{file_num}_frame_{end}",
                        "xy_delta": 0,
                        "z_delta": 100 if kind == "medium_z" else 0,
                        "yaw_delta_degrees": 0,
                        "pitch_delta_degrees": 0,
                        "angle_delta_degrees": 0,
                        "source_image_start": "unused",
                        "source_image_end": "unused",
                    }
                )


if __name__ == "__main__":
    unittest.main()
