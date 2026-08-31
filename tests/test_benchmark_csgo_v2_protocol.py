import sys
import types
import unittest
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def _install_lightweight_import_stubs() -> None:
    """Keep protocol tests independent of metric models and optional packages."""

    try:
        __import__("torchmetrics.image.lpip")
    except ModuleNotFoundError:
        torchmetrics = types.ModuleType("torchmetrics")
        image = types.ModuleType("torchmetrics.image")
        fid = types.ModuleType("torchmetrics.image.fid")
        inception = types.ModuleType("torchmetrics.image.inception")
        lpips = types.ModuleType("torchmetrics.image.lpip")
        torchmetrics.__path__ = []
        image.__path__ = []

        class MetricStub:
            def __init__(self, *args, **kwargs):
                pass

        image.PeakSignalNoiseRatio = MetricStub
        image.StructuralSimilarityIndexMeasure = MetricStub
        fid.FrechetInceptionDistance = MetricStub
        inception.InceptionScore = MetricStub
        lpips.LearnedPerceptualImagePatchSimilarity = MetricStub
        sys.modules.update({
            "torchmetrics": torchmetrics,
            "torchmetrics.image": image,
            "torchmetrics.image.fid": fid,
            "torchmetrics.image.inception": inception,
            "torchmetrics.image.lpip": lpips,
        })

    external_loader = "unilip.model.external_loc_model_loader"
    if external_loader not in sys.modules:
        module = types.ModuleType(external_loader)
        module.build_frozen_external_loc_model = lambda *args, **kwargs: None
        sys.modules[external_loader] = module

    fvd_module = "fvd_metric"
    if not Path("third_party/PyTorch-Frechet-Video-Distance/fvd_metric.py").is_file():
        module = types.ModuleType(fvd_module)
        module.compute_fvd = lambda *args, **kwargs: 0.0
        sys.modules[fvd_module] = module


_install_lightweight_import_stubs()

from benchmark_csgo_v1 import (  # noqa: E402
    benchmark_v2_radar_path,
    collect_benchmark_v2_coverage,
    load_benchmark_v2_inference_provenance,
    load_pose_index,
    validate_benchmark_v2_coverage,
)
from benchmark_csgo_v1_conti import (  # noqa: E402
    build_fvd_clips,
    build_tracks_from_benchmark_v2_clips,
)


class BenchmarkV2ProtocolTest(unittest.TestCase):
    def test_coverage_uses_expected_stems_and_reports_pred_extras(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gt_dir = root / "gt"
            pred_dir = root / "pred"
            gt_dir.mkdir()
            pred_dir.mkdir()
            for filename in ("a.jpg", "b.jpg", "gt_only.jpg"):
                (gt_dir / filename).touch()
            for filename in ("a.jpg", "pred_extra.jpg"):
                (pred_dir / filename).touch()

            coverage = collect_benchmark_v2_coverage(
                str(gt_dir),
                str(pred_dir),
                [{"file_frame": "a"}, {"file_frame": "b"}],
                ".jpg",
            )

            self.assertEqual(coverage["GT_Count"], 2)
            self.assertEqual(coverage["Pred_Count"], 2)
            self.assertEqual(coverage["Common_Count"], 1)
            self.assertEqual(coverage["Coverage_GT"], 0.5)
            self.assertEqual(coverage["missing_pred_files"], ["b.jpg"])
            self.assertEqual(coverage["unmatched_pred_files"], ["pred_extra.jpg"])

    def test_coverage_rejects_duplicate_expected_gt_stem(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gt_dir = root / "gt"
            pred_dir = root / "pred"
            gt_dir.mkdir()
            pred_dir.mkdir()
            (gt_dir / "a.jpg").touch()
            (gt_dir / "a.png").touch()
            (pred_dir / "a.jpg").touch()

            with self.assertRaisesRegex(ValueError, "exactly once"):
                collect_benchmark_v2_coverage(
                    str(gt_dir),
                    str(pred_dir),
                    [{"file_frame": "a"}],
                    ".jpg",
                )

    def test_incomplete_coverage_is_strict_by_default(self):
        coverage = {
            "missing_pred_files": ["missing.jpg"],
            "unmatched_pred_files": [],
        }
        with self.assertRaisesRegex(ValueError, "requires exactly one prediction set"):
            validate_benchmark_v2_coverage(coverage, allow_incomplete=False)
        validate_benchmark_v2_coverage(coverage, allow_incomplete=True)

    def test_extra_predictions_are_strict_by_default(self):
        coverage = {
            "missing_pred_files": [],
            "unmatched_pred_files": ["stale_frame.jpg"],
        }
        with self.assertRaisesRegex(ValueError, "extra=1"):
            validate_benchmark_v2_coverage(coverage, allow_incomplete=False)
        validate_benchmark_v2_coverage(coverage, allow_incomplete=True)

    def test_manifest_radar_path_supports_shared_maps_directory(self):
        with TemporaryDirectory() as temp_dir:
            radar_path = Path(temp_dir) / "maps" / "cs_office_radar.png"
            radar_path.parent.mkdir(parents=True)
            radar_path.touch()
            selection = {"radar_paths": {"cs_office": radar_path}}
            self.assertEqual(
                benchmark_v2_radar_path(selection, "cs_office"),
                str(radar_path.resolve()),
            )

    def test_inference_provenance_must_match_manifest_and_split(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "benchmark_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            payload = {
                "benchmark_v2_manifest": str(manifest),
                "benchmark_v2_split": "seen_discrete_test",
                "maps": ["map_a"],
                "ckpt_path": "model.safetensors",
            }
            (root / "inference_manifest.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            args = argparse.Namespace(
                benchmark_v2_manifest=str(manifest),
                benchmark_v2_split="seen_discrete_test",
                allow_missing_inference_manifest=False,
            )
            provenance = load_benchmark_v2_inference_provenance(root, args, "map_a")
            self.assertEqual(provenance["payload"]["ckpt_path"], "model.safetensors")
            args.benchmark_v2_split = "crossmap_query_test"
            with self.assertRaisesRegex(ValueError, "split mismatch"):
                load_benchmark_v2_inference_provenance(root, args, "map_a")

    def test_continuous_tracks_preserve_manifest_clip_order(self):
        clips = {
            "map_a": [
                {
                    "clip_id": "clip_b",
                    "frames": [
                        {"file_frame": "file_num2_frame_9"},
                        {"file_frame": "file_num2_frame_7"},
                    ],
                },
                {
                    "clip_id": "clip_a",
                    "frames": [
                        {"file_frame": "file_num1_frame_3"},
                        {"file_frame": "file_num1_frame_4"},
                    ],
                },
            ]
        }

        tracks, details = build_tracks_from_benchmark_v2_clips(
            clips,
            map_name="map_a",
            image_extension=".jpg",
            available_filenames=[
                "file_num1_frame_3.jpg",
                "file_num1_frame_4.jpg",
                "file_num2_frame_7.jpg",
                "file_num2_frame_9.jpg",
            ],
        )

        self.assertEqual(details["source"], "benchmark_v2_manifest")
        self.assertEqual(details["manifest_clip_count"], 2)
        self.assertEqual([record.filename for record in tracks[0]], [
            "file_num2_frame_9.jpg",
            "file_num2_frame_7.jpg",
        ])
        self.assertEqual([record.filename for record in tracks[1]], [
            "file_num1_frame_3.jpg",
            "file_num1_frame_4.jpg",
        ])

    def test_fvd_windows_stay_within_manifest_clip_boundaries(self):
        clips = {
            "map_a": [
                {
                    "clip_id": f"clip_{clip_index}",
                    "frames": [
                        {
                            "file_frame": (
                                f"file_num{clip_index}_frame_{frame_index}"
                            )
                        }
                        for frame_index in range(64)
                    ],
                }
                for clip_index in range(2)
            ]
        }

        tracks, _ = build_tracks_from_benchmark_v2_clips(clips, map_name="map_a")
        fvd_clips = build_fvd_clips(tracks, clip_length=16, clip_stride=16)

        self.assertEqual(len(fvd_clips), 8)
        self.assertTrue(
            all(len({filename.split("_frame_")[0] for filename in clip}) == 1
                for clip in fvd_clips)
        )

    def test_locator_uses_manifest_global_z_range(self):
        rows = [{
            "file_frame": "file_num1_frame_0",
            "map": "map_a",
            "x": 10,
            "y": 20,
            "z": 3,
            "angle_v": 1.5,
            "angle_h": 2.5,
        }]

        pose_index, z_min, z_max, loaded_paths = load_pose_index(
            "unused",
            "map_a",
            ["file_num1_frame_0.jpg"],
            "auto",
            benchmark_v2_rows=rows,
            benchmark_v2_z_range={"z_min": -100, "z_max": 500},
            benchmark_v2_manifest="manifest.json",
        )

        self.assertEqual(pose_index["file_num1_frame_0"]["z"], 3)
        self.assertEqual((z_min, z_max), (-100.0, 500.0))
        self.assertEqual(loaded_paths, [str(Path("manifest.json").resolve())])


if __name__ == "__main__":
    unittest.main()
