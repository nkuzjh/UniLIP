import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.materialize_csgo_benchmark_v2 import (
    MaterializationError,
    _load_protocol_maps,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "materialize_csgo_benchmark_v2.py"


class MaterializeCsgoBenchmarkV2Tests(unittest.TestCase):
    def _make_fixture(self, root: Path) -> dict[str, object]:
        source_root = root / "source"
        output_root = root / "data" / "csgo_benchmark_v2"
        source_root.mkdir(parents=True)
        output_root.mkdir(parents=True)
        seen_maps = ["seen_a"]
        crossmap_maps = ["cross_b"]
        support_seeds = [0, 1]
        all_maps = seen_maps + crossmap_maps

        rows_by_map: dict[str, dict[str, list[dict[str, object]]]] = {
            "seen_a": {
                "train": [self._row("seen_a", "file_num1_frame_1")],
                "validation": [self._row("seen_a", "file_num1_frame_2")],
                "discrete_test": [self._row("seen_a", "file_num2_frame_1")],
                "continuous": [
                    self._row("seen_a", "file_num3_frame_1"),
                    self._row("seen_a", "file_num3_frame_2"),
                ],
            },
            "cross_b": {
                "support_seed_0": [self._row("cross_b", "file_num4_frame_1")],
                # Deliberately repeat one support frame across seeds: it must
                # be copied once, while both split references remain valid.
                "support_seed_1": [self._row("cross_b", "file_num4_frame_1")],
                "query_test": [self._row("cross_b", "file_num5_frame_1")],
                "continuous": [
                    self._row("cross_b", "file_num6_frame_1"),
                    self._row("cross_b", "file_num6_frame_2"),
                ],
            },
        }
        radar_files: dict[str, str] = {}
        for map_name in all_maps:
            map_root = source_root / map_name
            image_root = map_root / "imgs"
            image_root.mkdir(parents=True)
            radar_relative = f"{map_name}/{map_name}_radar.png"
            radar_path = source_root / radar_relative
            radar_path.write_bytes(f"radar:{map_name}".encode("ascii"))
            radar_files[map_name] = radar_relative
            for split_values in rows_by_map[map_name].values():
                for row in split_values:
                    frame = str(row["file_frame"])
                    (image_root / f"{frame}.jpg").write_bytes(
                        f"image:{map_name}:{frame}".encode("ascii")
                    )
            (map_root / "positions.json").write_text("[]\n", encoding="utf-8")

        for map_name in seen_maps:
            prefix = output_root / "splits" / "seen" / map_name
            prefix.mkdir(parents=True)
            for split_name in ("train", "validation", "discrete_test"):
                self._write_json(prefix / f"{split_name}.json", rows_by_map[map_name][split_name])
            self._write_continuous(prefix / "continuous_clips.json", map_name, rows_by_map[map_name]["continuous"])
        for map_name in crossmap_maps:
            prefix = output_root / "splits" / "crossmap" / map_name
            prefix.mkdir(parents=True)
            for seed in support_seeds:
                self._write_json(
                    prefix / f"support_seed_{seed}.json",
                    rows_by_map[map_name][f"support_seed_{seed}"],
                )
            self._write_json(prefix / "query_test.json", rows_by_map[map_name]["query_test"])
            self._write_continuous(prefix / "continuous_clips.json", map_name, rows_by_map[map_name]["continuous"])

        selected: dict[str, str] = {}
        for map_name, split_values in rows_by_map.items():
            for values in split_values.values():
                for row in values:
                    frame = str(row["file_frame"])
                    relative = f"{map_name}/imgs/{frame}.jpg"
                    source_path = source_root / relative
                    selected[relative] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        selected_bytes = "".join(
            f"{selected[relative]}  {relative}\n" for relative in sorted(selected)
        ).encode("utf-8")
        (output_root / "selected_images.sha256").write_bytes(selected_bytes)
        selected_metadata = {
            "file": "selected_images.sha256",
            "sha256": hashlib.sha256(selected_bytes).hexdigest(),
            "count": len(selected),
        }

        manifest = {
            "schema_version": 1,
            "benchmark_id": "csgo_benchmark_v2",
            "benchmark": {"id": "csgo_benchmark_v2"},
            "protocol": {
                "seen_maps": seen_maps,
                "crossmap_maps": crossmap_maps,
                "support_seeds": support_seeds,
            },
            "source": {
                "root": "source",
                "radar_files": radar_files,
                "radar_sha256": {
                    map_name: hashlib.sha256(
                        (source_root / relative).read_bytes()
                    ).hexdigest()
                    for map_name, relative in radar_files.items()
                },
            },
            "selected_images": selected_metadata,
        }
        self._write_json(output_root / "benchmark_manifest.json", manifest)
        self._write_json(output_root / "build_report.json", {"selected_images": selected_metadata})

        config = {
            "paths": {"source_root": "source", "output_root": "data/csgo_benchmark_v2"},
            "source": {
                "images_dir": "imgs",
                "image_extension": ".jpg",
                "record_regex": r"^file_num(?P<record>\d+)_frame_(?P<frame>\d+)$",
            },
        }
        (root / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
        return {
            "output_root": output_root,
            "source_root": source_root,
            "selected_count": len(selected),
        }

    @staticmethod
    def _row(map_name: str, file_frame: str) -> dict[str, object]:
        return {"map": map_name, "file_frame": file_frame}

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _write_continuous(path: Path, map_name: str, rows: list[dict[str, object]]) -> None:
        MaterializeCsgoBenchmarkV2Tests._write_json(
            path,
            {
                "schema_version": 1,
                "benchmark_id": "csgo_benchmark_v2",
                "map": map_name,
                "split": "continuous",
                "clips": [
                    {
                        "map": map_name,
                        "clip_id": f"{map_name}_continuous_0000",
                        "frames": rows,
                    }
                ],
            },
        )

    @staticmethod
    def _run(root: Path, command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                command,
                "--repo-root",
                str(root),
                "--manifest",
                "data/csgo_benchmark_v2/benchmark_manifest.json",
                "--config",
                "config.yaml",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_materialize_verify_and_existing_tree_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._make_fixture(root)
            output_root = fixture["output_root"]
            first = self._run(root, "materialize")
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn("images: 10%", first.stdout)
            self.assertIn("radars: 100%", first.stdout)

            image_files = sorted((output_root / "images").rglob("*.jpg"))
            self.assertEqual(len(image_files), fixture["selected_count"])
            self.assertEqual(len(list((output_root / "images").rglob("*.png"))), 0)
            radar_files = sorted((output_root / "radars").rglob("*.png"))
            self.assertEqual(len(radar_files), 2)
            report = json.loads(
                (output_root / "minimal_dataset_report.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("entries", report["images"])
            self.assertEqual(report["bundle_paths"]["source_root"], "<source_root>")
            self.assertEqual(len(report["radars"]["entries"]), 2)
            self.assertTrue(
                all(
                    (fixture["source_root"] / relative).is_file()
                    for relative in (
                        "seen_a/imgs/file_num1_frame_1.jpg",
                        "cross_b/imgs/file_num4_frame_1.jpg",
                    )
                )
            )

            verified = self._run(root, "verify")
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertIn("VERIFY OK", verified.stdout)

            report_before = (output_root / "minimal_dataset_report.json").read_bytes()
            second = self._run(root, "materialize")
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("image_status=already_materialized", second.stdout)
            self.assertIn("radar_status=already_materialized", second.stdout)
            self.assertEqual(
                report_before,
                (output_root / "minimal_dataset_report.json").read_bytes(),
            )

            extra = output_root / "images" / "seen_a" / "extra.jpg"
            extra.write_bytes(b"must-not-be-deleted")
            refused = self._run(root, "materialize")
            self.assertNotEqual(refused.returncode, 0)
            self.assertTrue(extra.exists())

    def test_split_derived_set_must_match_selected_checksum_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._make_fixture(root)
            selected_path = fixture["output_root"] / "selected_images.sha256"
            lines = selected_path.read_text(encoding="utf-8").splitlines()
            selected_path.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
            refused = self._run(root, "materialize")
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("split-derived frame set", refused.stderr)
            self.assertFalse((fixture["output_root"] / "images").exists())

    def test_verify_target_succeeds_after_source_corpus_is_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._make_fixture(root)
            materialized = self._run(root, "materialize")
            self.assertEqual(materialized.returncode, 0, materialized.stderr)

            shutil.rmtree(fixture["source_root"])
            verified = self._run(root, "verify-target")
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertIn("VERIFY TARGET OK", verified.stdout)
            self.assertIn("source_access=not_required", verified.stdout)

            first_image = next((fixture["output_root"] / "images").rglob("*.jpg"))
            first_image.write_bytes(b"tampered")
            refused = self._run(root, "verify-target")
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("target hash mismatch", refused.stderr)

    def test_verify_target_rejects_target_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._make_fixture(root)
            materialized = self._run(root, "materialize")
            self.assertEqual(materialized.returncode, 0, materialized.stderr)
            shutil.rmtree(fixture["source_root"])

            image = next((fixture["output_root"] / "images").rglob("*.jpg"))
            hardlink = root / "image-hardlink.jpg"
            os.link(image, hardlink)
            refused = self._run(root, "verify-target")
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("must not be hard-linked", refused.stderr)

            hardlink.unlink()
            image.unlink()
            image.symlink_to(root / "replacement.jpg")
            (root / "replacement.jpg").write_bytes(b"replacement")
            refused = self._run(root, "verify-target")
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("non-file entry", refused.stderr)

    def test_verify_target_rejects_metadata_and_split_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._make_fixture(root)
            materialized = self._run(root, "materialize")
            self.assertEqual(materialized.returncode, 0, materialized.stderr)
            shutil.rmtree(fixture["source_root"])

            manifest = fixture["output_root"] / "benchmark_manifest.json"
            manifest_copy = root / "manifest-copy.json"
            shutil.copy2(manifest, manifest_copy)
            manifest.unlink()
            manifest.symlink_to(manifest_copy)
            refused = self._run(root, "verify-target")
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("benchmark manifest must be a regular non-symlink", refused.stderr)

            manifest.unlink()
            shutil.copy2(manifest_copy, manifest)
            split = fixture["output_root"] / "splits" / "seen" / "seen_a" / "train.json"
            split_copy = root / "train-copy.json"
            shutil.copy2(split, split_copy)
            split.unlink()
            split.symlink_to(split_copy)
            refused = self._run(root, "verify-target")
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("seen train split must be a regular non-symlink", refused.stderr)

            split_dir = fixture["output_root"] / "splits" / "seen" / "seen_a"
            split_dir_copy = root / "seen_a-copy"
            shutil.copytree(split_dir, split_dir_copy)
            shutil.rmtree(split_dir)
            split_dir.symlink_to(split_dir_copy, target_is_directory=True)
            refused = self._run(root, "verify-target")
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("seen seen_a splits directory must be a real directory", refused.stderr)

    def test_manifest_map_names_are_single_safe_posix_components(self):
        for bad_name in ("", ".", "..", "seen/a", r"seen\a", "seen\nmap"):
            with self.subTest(bad_name=bad_name):
                with self.assertRaises(MaterializationError):
                    _load_protocol_maps(
                        {
                            "protocol": {
                                "seen_maps": [bad_name],
                                "crossmap_maps": ["cross_b"],
                                "support_seeds": [0],
                            }
                        }
                    )

    def test_report_and_build_report_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._make_fixture(root)
            output_root = fixture["output_root"]
            build_report = output_root / "build_report.json"
            build_report.unlink()
            build_report.symlink_to(output_root / "selected_images.sha256")
            refused = self._run(root, "materialize")
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("build report must be a regular non-symlink", refused.stderr)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._make_fixture(root)
            output_root = fixture["output_root"]
            materialized = self._run(root, "materialize")
            self.assertEqual(materialized.returncode, 0, materialized.stderr)
            report = output_root / "minimal_dataset_report.json"
            report.unlink()
            report.symlink_to(output_root / "selected_images.sha256")
            refused = self._run(root, "verify")
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("materialization report must be a regular non-symlink", refused.stderr)


if __name__ == "__main__":
    unittest.main()
