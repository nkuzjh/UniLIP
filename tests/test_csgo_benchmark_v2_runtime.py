import json
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from csgo_datasets.benchmark_v2 import (
    BenchmarkV2Error,
    load_benchmark_v2_selection,
    load_benchmark_v2_selection_from_args,
    benchmark_v2_image_path,
    is_benchmark_v2_config,
)


class BenchmarkV2RuntimeTests(unittest.TestCase):
    def _make_fixture(self, root: Path) -> Path:
        benchmark_root = root / "benchmark"
        source_root = root / "source"
        benchmark_root.mkdir()
        source_root.mkdir()
        seen_maps = ["seen_a", "seen_b"]
        crossmap_maps = ["cs_office", "cross_b"]
        all_maps = seen_maps + crossmap_maps

        radar_files = {}
        radar_sha256 = {}
        for map_name in all_maps:
            if map_name == "cs_office":
                relative = "maps/cs_office_radar.png"
            else:
                relative = f"{map_name}/{map_name}_radar.png"
            radar_path = source_root / relative
            radar_path.parent.mkdir(parents=True, exist_ok=True)
            radar_path.write_bytes(b"radar")
            radar_files[map_name] = relative
            radar_sha256[map_name] = hashlib.sha256(b"radar").hexdigest()

        selected_checksum = "".join(
            f"{'a' * 64}  {map_name}/imgs/file_num0_frame_1.jpg\n"
            for map_name in all_maps
        )

        def make_row(map_name: str, index: int) -> dict:
            return {
                "map": map_name,
                "file_frame": f"file_num{index}_frame_{index + 1}",
                "x": float(index),
                "y": float(index + 1),
                "z": 10 + index % 10,
                "angle_h": 0.1 + index / 100.0,
                "angle_v": 1.0,
            }

        counts = {"seen": {}, "crossmap": {}}
        for map_name in seen_maps:
            train = [make_row(map_name, 0), make_row(map_name, 1)]
            validation = [make_row(map_name, 2)]
            discrete_test = [make_row(map_name, 3), make_row(map_name, 4)]
            self._write_json(
                benchmark_root / "splits" / "seen" / map_name / "train.json", train
            )
            self._write_json(
                benchmark_root / "splits" / "seen" / map_name / "validation.json",
                validation,
            )
            self._write_json(
                benchmark_root
                / "splits"
                / "seen"
                / map_name
                / "discrete_test.json",
                discrete_test,
            )
            counts["seen"][map_name] = {
                "train": 2,
                "validation": 1,
                "discrete_test": 2,
                "continuous_clips": 2,
                "continuous_frames": 6,
            }
            self._write_continuous(
                benchmark_root / "splits" / "seen" / map_name,
                map_name,
            )

        for map_name in crossmap_maps:
            support = [make_row(map_name, 1000 + index) for index in range(100)]
            query_test = [make_row(map_name, 1200), make_row(map_name, 1201)]
            for support_seed in (0, 1):
                self._write_json(
                    benchmark_root
                    / "splits"
                    / "crossmap"
                    / map_name
                    / f"support_seed_{support_seed}.json",
                    support,
                )
            self._write_json(
                benchmark_root
                / "splits"
                / "crossmap"
                / map_name
                / "query_test.json",
                query_test,
            )
            counts["crossmap"][map_name] = {
                "support": {"0": 100, "1": 100},
                "query_test": 2,
                "continuous_clips": 2,
                "continuous_frames": 6,
            }
            self._write_continuous(
                benchmark_root / "splits" / "crossmap" / map_name,
                map_name,
            )

        z_ranges = {
            map_name: {"z_min": 0, "z_max": 100} for map_name in all_maps
        }
        manifest = {
            "schema_version": 1,
            "benchmark_id": "csgo_benchmark_v2",
            "benchmark": {
                "id": "csgo_benchmark_v2",
                "version": "2.0.0",
                "global_seed": 123,
                "strict_protocol": True,
            },
            "protocol": {
                "seen_maps": seen_maps,
                "crossmap_maps": crossmap_maps,
                "seen_splits": ["train", "validation", "discrete_test"],
                "crossmap_splits": ["support", "query_test", "continuous"],
                "support_seeds": [0, 1],
            },
            "calibration": {"z_ranges": z_ranges},
            "source": {
                "root": str(source_root),
                "radar_files": radar_files,
            },
            "radar_sha256": radar_sha256,
            "selected_images": {
                "count": len(all_maps),
                "file": "selected_images.sha256",
                "sha256": hashlib.sha256(selected_checksum.encode()).hexdigest(),
            },
            "counts": counts,
            "continuous_protocol": {
                "frames_per_clip": 3,
                "trajectory_disjoint": True,
            },
        }
        manifest_path = benchmark_root / "benchmark_manifest.json"
        self._write_json(manifest_path, manifest)
        return manifest_path

    def _make_minimal_fixture(
        self, root: Path, manifest_path: Path, source_root: Path
    ) -> Path:
        bundle_root = root / "minimal_bundle"
        images_root = bundle_root / "images"
        radars_root = bundle_root / "radars"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        maps = [
            *manifest["protocol"]["seen_maps"],
            *manifest["protocol"]["crossmap_maps"],
        ]
        image_by_map = {}
        for map_name in maps:
            image_dir = images_root / map_name
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / "file_num0_frame_1.jpg"
            image_path.write_bytes(b"image")
            image_by_map[map_name] = {
                "bytes": image_path.stat().st_size,
                "count": 1,
            }

        radar_entries = []
        radar_by_map = {}
        for map_name in maps:
            source_relative = manifest["source"]["radar_files"][map_name]
            source_path = source_root / source_relative
            target_path = radars_root / map_name / source_path.name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target_path)
            radar_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
            radar_entries.append(
                {
                    "bytes": target_path.stat().st_size,
                    "map": map_name,
                    "sha256": radar_hash,
                    "source": source_relative,
                    "target": f"{map_name}/{target_path.name}",
                }
            )
            radar_by_map[map_name] = {
                "bytes": target_path.stat().st_size,
                "count": 1,
            }

        selected_path = bundle_root / "selected_images.sha256"
        selected_path.write_text(
            "".join(
                f"{'a' * 64}  {map_name}/imgs/file_num0_frame_1.jpg\n"
                for map_name in maps
            ),
            encoding="utf-8",
        )
        report = {
            "benchmark_id": "csgo_benchmark_v2",
            "copy_mode": "copyfile_not_move",
            "images": {
                "by_map": image_by_map,
                "bytes": sum(value["bytes"] for value in image_by_map.values()),
                "count": sum(value["count"] for value in image_by_map.values()),
                "root": "images",
                "status": "verified",
                "target_template": "images/{map}/{file_frame}.jpg",
            },
            "manifest": {
                "path": "benchmark_manifest.json",
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            },
            "maps": maps,
            "radars": {
                "by_map": radar_by_map,
                "bytes": sum(value["bytes"] for value in radar_by_map.values()),
                "count": sum(value["count"] for value in radar_by_map.values()),
                "entries": radar_entries,
                "root": "radars",
                "status": "verified",
            },
            "schema_version": 1,
            "selected_images": {
                "count": len(maps),
                "path": "selected_images.sha256",
                "sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
            },
            "status": "verified",
            "invariants": {
                "radars_are_separate_from_images": True,
                "source_exists_after_copy": True,
                "source_target_samefile": False,
                "split_derived_set_equals_selected_images": True,
                "target_contains_only_declared_maps_and_files": True,
            },
        }
        report_path = bundle_root / "minimal_dataset_report.json"
        self._write_json(report_path, report)
        return report_path

    @staticmethod
    def _write_json(path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _write_continuous(directory: Path, map_name: str) -> None:
        clips = []
        for clip_index in range(2):
            frames = []
            for frame_index in range(3):
                frames.append(
                    {
                        "map": map_name,
                        "file_frame": (
                            f"file_num{200 + clip_index}_frame_{frame_index + 1}"
                        ),
                        "x": clip_index * 10 + frame_index,
                        "y": clip_index * 10 + frame_index + 1,
                        "z": 20 + frame_index,
                        "angle_h": 0.2,
                        "angle_v": 1.1,
                    }
                )
            clips.append(
                {
                    "clip_id": f"{map_name}_continuous_{clip_index:04d}",
                    "map": map_name,
                    "record_id": str(200 + clip_index),
                    "frames": frames,
                }
            )
        BenchmarkV2RuntimeTests._write_json(
            directory / "continuous_clips.json",
            {
                "schema_version": 1,
                "benchmark_id": "csgo_benchmark_v2",
                "map": map_name,
                "split": "continuous",
                "frames_per_clip": 3,
                "clips": clips,
            },
        )

    def test_config_detection_and_all_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = self._make_fixture(Path(temporary))
            self.assertTrue(
                is_benchmark_v2_config(
                    {"benchmark_v2_manifest": str(manifest_path)}
                )
            )
            self.assertFalse(is_benchmark_v2_config({"benchmark_v2_split": "seen_train"}))

            expected = {
                "seen_train": (4, ["seen_a", "seen_b"]),
                "seen_validation": (2, ["seen_a", "seen_b"]),
                "seen_discrete_test": (4, ["seen_a", "seen_b"]),
                "seen_continuous": (12, ["seen_a", "seen_b"]),
                "crossmap_support": (200, ["cs_office", "cross_b"]),
                "crossmap_query_test": (4, ["cs_office", "cross_b"]),
                "crossmap_continuous": (12, ["cs_office", "cross_b"]),
            }
            for split, (row_count, maps) in expected.items():
                selection = load_benchmark_v2_selection(
                    {
                        "benchmark_v2_manifest": str(manifest_path),
                        "benchmark_v2_split": split,
                        "benchmark_v2_image_extension": "png",
                    }
                )
                self.assertEqual(selection.split, split)
                self.assertEqual(selection.map_names, maps)
                self.assertEqual(len(selection.rows), row_count)
                self.assertEqual(selection.image_extension, ".png")
                if "cs_office" in maps:
                    self.assertEqual(
                        selection.radar_paths["cs_office"].name,
                        "cs_office_radar.png",
                    )

    def test_continuous_flatten_keeps_clip_and_frame_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = self._make_fixture(Path(temporary))
            selection = load_benchmark_v2_selection_from_args(
                manifest_path,
                "seen_continuous",
                map_names=["seen_b"],
            )
            self.assertEqual(len(selection.rows), 6)
            self.assertEqual(len(selection.clips_by_map["seen_b"]), 2)
            self.assertEqual(
                selection.clips_by_map["seen_b"][0]["clip_id"],
                "seen_b_continuous_0000",
            )
            self.assertEqual(
                len(selection.clips_by_map["seen_b"][0]["frames"]),
                3,
            )
            self.assertEqual(
                [row["_benchmark_frame_index"] for row in selection.rows],
                [0, 1, 2, 0, 1, 2],
            )
            self.assertEqual(
                [row["_benchmark_clip_id"] for row in selection.rows],
                [
                    "seen_b_continuous_0000",
                    "seen_b_continuous_0000",
                    "seen_b_continuous_0000",
                    "seen_b_continuous_0001",
                    "seen_b_continuous_0001",
                    "seen_b_continuous_0001",
                ],
            )
            self.assertEqual(selection.split_files["seen_b"].name, "continuous_clips.json")

    def test_minimal_asset_backend_uses_flat_bundle_without_source_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._make_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source"]["root"] = "missing-source-root"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report_path = self._make_minimal_fixture(
                root, manifest_path, root / "source"
            )
            shutil.rmtree(root / "source")
            selection = load_benchmark_v2_selection(
                {
                    "benchmark_v2_manifest": str(manifest_path),
                    "benchmark_v2_asset_manifest": str(report_path),
                    "benchmark_v2_split": "seen_train",
                    "data_dir": str(root / "also-missing-source-root"),
                },
                map_names=["seen_a"],
            )
            self.assertEqual(selection.asset_backend, "minimal")
            self.assertEqual(selection.source_root, report_path.parent.resolve())
            self.assertEqual(
                selection.benchmark_v2_image_dir("seen_a"),
                (report_path.parent / "images" / "seen_a").resolve(),
            )
            image_path = benchmark_v2_image_path(
                selection, "seen_a", "file_num0_frame_1"
            )
            self.assertTrue(image_path.is_file())
            self.assertEqual(image_path.parent.name, "seen_a")

    def test_minimal_asset_report_is_bound_to_manifest_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._make_fixture(root)

            report_path = self._make_minimal_fixture(
                root, manifest_path, root / "source"
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["selected_images"]["count"] = 2
            self._write_json(report_path, report)
            with self.assertRaisesRegex(
                BenchmarkV2Error, "selected_images.count"
            ):
                load_benchmark_v2_selection(
                    {
                        "benchmark_v2_manifest": str(manifest_path),
                        "benchmark_v2_asset_manifest": str(report_path),
                        "benchmark_v2_split": "seen_train",
                    },
                    map_names=["seen_a"],
                )

            report_path = self._make_minimal_fixture(
                root, manifest_path, root / "source"
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["radars"]["entries"][0]["source"] = "wrong/radar.png"
            self._write_json(report_path, report)
            with self.assertRaisesRegex(
                BenchmarkV2Error, "radar source does not match"
            ):
                load_benchmark_v2_selection(
                    {
                        "benchmark_v2_manifest": str(manifest_path),
                        "benchmark_v2_asset_manifest": str(report_path),
                        "benchmark_v2_split": "seen_train",
                    },
                    map_names=["seen_a"],
                )

            report_path = self._make_minimal_fixture(
                root, manifest_path, root / "source"
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["images"]["bytes"] += 1
            self._write_json(report_path, report)
            with self.assertRaisesRegex(
                BenchmarkV2Error, "images count/bytes"
            ):
                load_benchmark_v2_selection(
                    {
                        "benchmark_v2_manifest": str(manifest_path),
                        "benchmark_v2_asset_manifest": str(report_path),
                        "benchmark_v2_split": "seen_train",
                    },
                    map_names=["seen_a"],
                )

            report_path = self._make_minimal_fixture(
                root, manifest_path, root / "source"
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["images"]["count"] += 1
            report["images"]["by_map"]["seen_a"]["count"] += 1
            self._write_json(report_path, report)
            with self.assertRaisesRegex(
                BenchmarkV2Error, "images.count does not match"
            ):
                load_benchmark_v2_selection(
                    {
                        "benchmark_v2_manifest": str(manifest_path),
                        "benchmark_v2_asset_manifest": str(report_path),
                        "benchmark_v2_split": "seen_train",
                    },
                    map_names=["seen_a"],
                )

    def test_crossmap_support_is_nested_and_seed_specific(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = self._make_fixture(Path(temporary))
            full = load_benchmark_v2_selection_from_args(
                manifest_path, "crossmap_support", support_seed=0, shots_per_map=100
            )
            half = load_benchmark_v2_selection_from_args(
                manifest_path, "crossmap_support", support_seed=0, shots_per_map=50
            )
            one_map = load_benchmark_v2_selection_from_args(
                manifest_path,
                "crossmap_support",
                map_names=["cross_b"],
                support_seed=1,
                shots_per_map=7,
            )
            for map_name in full.map_names:
                full_rows = [row for row in full.rows if row["map"] == map_name]
                half_rows = [row for row in half.rows if row["map"] == map_name]
                self.assertEqual(len(full_rows), 100)
                self.assertEqual(half_rows, full_rows[:50])
            self.assertEqual(len(one_map.rows), 7)
            self.assertTrue(all(row["map"] == "cross_b" for row in one_map.rows))
            self.assertNotEqual(
                [row["file_frame"] for row in one_map.rows],
                [row["file_frame"] for row in full.rows if row["map"] == "cross_b"][:7],
            )

    def test_data_dir_override_and_invalid_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._make_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source"]["root"] = "missing-relative-source"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            selection = load_benchmark_v2_selection_from_args(
                manifest_path,
                "seen_train",
                data_dir=root / "source",
            )
            self.assertEqual(selection.source_root, (root / "source").resolve())
            self.assertEqual(selection.image_extension, ".jpg")

            with self.assertRaises(BenchmarkV2Error):
                load_benchmark_v2_selection_from_args(
                    manifest_path,
                    "crossmap_support",
                    map_names=["seen_a"],
                )
            with self.assertRaises(BenchmarkV2Error):
                load_benchmark_v2_selection_from_args(
                    manifest_path, "crossmap_support", shots_per_map=0
                )
            with self.assertRaises(BenchmarkV2Error):
                load_benchmark_v2_selection_from_args(
                    manifest_path, "crossmap_support", shots_per_map=101
                )

            missing_calibration = dict(manifest)
            missing_calibration.pop("calibration")
            missing_path = root / "missing_calibration.json"
            missing_path.write_text(json.dumps(missing_calibration), encoding="utf-8")
            with self.assertRaises(BenchmarkV2Error):
                load_benchmark_v2_selection_from_args(
                    missing_path,
                    "seen_train",
                    data_dir=root / "source",
                )

    def test_missing_split_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = self._make_fixture(Path(temporary))
            split_path = (
                manifest_path.parent
                / "splits"
                / "seen"
                / "seen_a"
                / "validation.json"
            )
            split_path.unlink()
            with self.assertRaises(FileNotFoundError):
                load_benchmark_v2_selection_from_args(
                    manifest_path, "seen_validation", map_names=["seen_a"]
                )


if __name__ == "__main__":
    unittest.main()
