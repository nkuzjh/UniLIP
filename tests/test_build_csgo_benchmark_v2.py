import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - the builder reports this clearly
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_csgo_benchmark_v2.py"


@unittest.skipUnless(yaml is not None, "PyYAML is required")
class BuildCsgoBenchmarkV2Tests(unittest.TestCase):
    def _make_workspace(self, root: Path) -> Path:
        source_root = root / "source"
        source_root.mkdir(parents=True)
        map_names = ["seen_map", "cross_map"]
        for map_name in map_names:
            map_root = source_root / map_name
            image_root = map_root / "imgs"
            image_root.mkdir(parents=True)
            rows = []
            for record_id in range(8):
                for frame_id in range(8):
                    x = record_id * 100.0 + frame_id
                    if map_name == "seen_map" and record_id == 0 and frame_id == 0:
                        x = -5.0
                    row = {
                        "map": map_name,
                        "file_frame": f"file_num{record_id}_frame_{frame_id}",
                        "x": x,
                        "y": record_id * 100.0 + frame_id * 0.1,
                        "z": 20.0 + record_id,
                        "angle_h": 0.1,
                        "angle_v": 0.2,
                        "extra": {"source_row": record_id * 8 + frame_id},
                    }
                    rows.append(row)
                    (image_root / f"{row['file_frame']}.jpg").touch()
            (map_root / "positions.json").write_text(
                json.dumps(rows, indent=2) + "\n", encoding="utf-8"
            )
            (map_root / f"{map_name}_radar.png").write_bytes(
                f"radar-{map_name}".encode("ascii")
            )

        config = {
            "schema_version": 1,
            "benchmark": {
                "id": "csgo_benchmark_v2",
                "version": "2.0.0",
                "global_seed": 20260827,
                "strict_protocol": False,
            },
            "paths": {"source_root": "source", "output_root": "output"},
            "calibration": {
                "z": {
                    "source": "approved_full_corpus",
                    "method": "exact_min_max",
                    "timing": "before_split",
                    "normalize_to": [0.0, 1.0],
                    "clamp": False,
                    "fallback": "error",
                }
            },
            "source": {
                "positions_file": "positions.json",
                "images_dir": "imgs",
                "image_extension": ".jpg",
                "radar_files": {
                    "seen_map": "seen_map/seen_map_radar.png",
                    "cross_map": "cross_map/cross_map_radar.png",
                },
                "record_regex": r"^file_num(?P<record>\d+)_frame_(?P<frame>\d+)$",
                "forbidden_map_dirs": ["forbidden_alias"],
            },
            "maps": {"seen": ["seen_map"], "crossmap": ["cross_map"]},
            "counts": {
                "seen": {"train": 2, "validation": 1, "discrete_test": 2},
                "crossmap": {
                    "support": 2,
                    "query_test": 2,
                    "support_seeds": [0, 1, 2, 3, 4],
                },
                "continuous": {
                    "clips_per_map": 2,
                    "frames_per_clip": 4,
                    "max_frame_gap": 2,
                    "max_clips_per_record": 1,
                },
            },
            "record_pools": {
                "seen": {
                    "train": 0.25,
                    "validation": 0.125,
                    "discrete_test": 0.25,
                    "continuous": 0.375,
                },
                "crossmap": {
                    "support": 0.25,
                    "query_test": 0.375,
                    "continuous": 0.375,
                },
            },
            "sampling": {
                "min_frame_gap": 2,
                "spatial_bins": {"x": 2, "y": 2, "z": 1, "yaw": 2, "pitch": 1},
                "near_pose_filter": {
                    "enabled": True,
                    "xy_tolerance": 2.0,
                    "z_tolerance": 2.0,
                    "angle_tolerance_degrees": 5.0,
                },
            },
            "audit": {
                "coordinate_bounds": {"x": [0, 1000], "y": [0, 1000]},
                "angle_bounds": {
                    "angle_h": [0.0, 6.283185307179586],
                    "angle_v": [0.0, 3.141592653589793],
                },
                "robust_z_mad_threshold": 12.0,
                "z_mad_candidate_enabled": False,
                "jump_thresholds": {"xy": 200.0, "z": 300.0, "angle_degrees": 45.0},
                "require_images": True,
                "candidate_preview_limit": 50,
            },
            "output": {"indent": 2},
        }
        (root / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        return root / "config.yaml"

    def _run(self, workspace: Path, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )

    def _approve_decisions(
        self,
        workspace: Path,
        *,
        default_action: str = "keep",
        exclude_records: dict[str, list[str]] | None = None,
        exclude_file_frames: dict[str, list[str]] | None = None,
        output_name: str = "decisions.yaml",
    ) -> Path:
        template_path = workspace / "output/audit/anomaly_decisions.template.yaml"
        decisions = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        decisions["review"].update(
            {
                "status": "approved",
                "reviewer": "unit-test",
                "reviewed_at": "2026-08-27T00:00:00Z",
                "references": ["synthetic audit"],
            }
        )
        decisions["coordinate_candidates"]["default_action"] = default_action
        decisions["coordinate_candidates"]["exclude_records"] = exclude_records or {}
        decisions["coordinate_candidates"]["exclude_file_frames"] = (
            exclude_file_frames or {}
        )
        path = workspace / output_name
        path.write_text(yaml.safe_dump(decisions, sort_keys=False), encoding="utf-8")
        return path

    def _calibrate_and_approve(
        self,
        workspace: Path,
        decisions_path: Path,
        *,
        output_name: str = "calibration_approval.yaml",
    ) -> Path:
        calibrated = self._run(
            workspace,
            "calibrate",
            "--config",
            "config.yaml",
            "--decisions",
            str(decisions_path.relative_to(workspace)),
        )
        self.assertEqual(calibrated.returncode, 0, calibrated.stderr)
        template_path = (
            workspace / "output/calibration/z_calibration_approval.template.yaml"
        )
        approval = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        approval["review"].update(
            {
                "status": "approved",
                "reviewer": "unit-test",
                "reviewed_at": "2026-08-27T00:00:00Z",
                "references": ["synthetic extrema review"],
            }
        )
        path = workspace / output_name
        path.write_text(yaml.safe_dump(approval, sort_keys=False), encoding="utf-8")
        return path

    def test_audit_gating_determinism_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            self._make_workspace(first)
            self._make_workspace(second)

            audit = self._run(first, "audit", "--config", "config.yaml")
            self.assertEqual(audit.returncode, 0, audit.stderr)
            report = json.loads(
                (first / "output/audit/audit_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["coordinate_candidates"]["total_rows"], 1)
            self.assertEqual(report["coordinate_stats"]["seen_map"]["x"]["count"], 64)
            self.assertIn(
                "coordinate_candidate_reason_counts",
                report["counts"]["by_map"]["seen_map"],
            )
            self.assertTrue(report["source"]["radar_sha256"]["seen_map"])
            candidate_lines = [
                json.loads(line)
                for line in (first / "output/audit/coordinate_candidates.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(candidate_lines), 1)
            self.assertIn("x_out_of_bounds", candidate_lines[0]["reasons"])
            self.assertEqual(candidate_lines[0]["source_index"], 0)
            self.assertEqual(candidate_lines[0]["integrity_reasons"], [])
            self.assertIn("x_out_of_bounds", candidate_lines[0]["coordinate_reasons"])
            self.assertIn(
                report["audit_report_sha256"],
                (first / "output/audit/anomaly_decisions.template.yaml").read_text(
                    encoding="utf-8"
                ),
            )
            candidate_csv_lines = (
                (first / "output/audit/coordinate_candidates.csv")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertEqual(
                candidate_csv_lines[0].split(","),
                [
                    "map",
                    "record_id",
                    "file_num",
                    "frame_id",
                    "frame",
                    "file_frame",
                    "reasons",
                    "z",
                    "source_image_path",
                ],
            )
            self.assertIn("seen_map,0,file_num0,0,frame_0", candidate_csv_lines[1])

            pending = self._run(
                first,
                "build",
                "--config",
                "config.yaml",
                "--decisions",
                "output/audit/anomaly_decisions.template.yaml",
                "--calibration-approval",
                "missing-calibration.yaml",
            )
            self.assertNotEqual(pending.returncode, 0)
            self.assertIn("review.status=approved", pending.stderr)

            decisions = yaml.safe_load(
                (first / "output/audit/anomaly_decisions.template.yaml").read_text(
                    encoding="utf-8"
                )
            )
            decisions["review"]["status"] = "approved"
            decisions["review"]["reviewer"] = "unit-test"
            decisions["review"]["reviewed_at"] = "2026-08-27T00:00:00Z"
            decisions["review"]["references"] = ["synthetic audit"]
            decisions["coordinate_candidates"]["default_action"] = "exclude"
            decisions["coordinate_candidates"]["exclude_file_frames"] = {
                candidate_lines[0]["map"]: [candidate_lines[0]["file_frame"]]
            }
            decisions_bytes = yaml.safe_dump(decisions, sort_keys=False).encode("utf-8")
            (first / "decisions.yaml").write_bytes(decisions_bytes)
            (second / "decisions.yaml").write_bytes(decisions_bytes)
            first_calibration_approval = self._calibrate_and_approve(
                first, first / "decisions.yaml"
            )

            stale = dict(decisions)
            stale["review"] = dict(decisions["review"])
            stale["review"]["audit_report_sha256"] = "0" * 64
            (first / "stale.yaml").write_text(
                yaml.safe_dump(stale, sort_keys=False), encoding="utf-8"
            )
            stale_result = self._run(
                first,
                "build",
                "--config",
                "config.yaml",
                "--decisions",
                "stale.yaml",
                "--calibration-approval",
                "missing-calibration.yaml",
            )
            self.assertNotEqual(stale_result.returncode, 0)
            self.assertIn("does not match current audit", stale_result.stderr)

            first_build = self._run(
                first,
                "build",
                "--config",
                "config.yaml",
                "--decisions",
                "decisions.yaml",
                "--calibration-approval",
                str(first_calibration_approval.relative_to(first)),
            )
            second_audit = self._run(second, "audit", "--config", "config.yaml")
            second_calibration_approval = self._calibrate_and_approve(
                second, second / "decisions.yaml"
            )
            second_build = self._run(
                second,
                "build",
                "--config",
                "config.yaml",
                "--decisions",
                "decisions.yaml",
                "--calibration-approval",
                str(second_calibration_approval.relative_to(second)),
            )
            self.assertEqual(first_build.returncode, 0, first_build.stderr)
            self.assertEqual(second_audit.returncode, 0, second_audit.stderr)
            self.assertEqual(second_build.returncode, 0, second_build.stderr)

            first_checksums = (first / "output/checksums.sha256").read_bytes()
            second_checksums = (second / "output/checksums.sha256").read_bytes()
            self.assertEqual(first_checksums, second_checksums)
            first_manifest = json.loads(
                (first / "output/benchmark_manifest.json").read_text(encoding="utf-8")
            )
            first_build_report = json.loads(
                (first / "output/build_report.json").read_text(encoding="utf-8")
            )
            selected_image_data = (first / "output/selected_images.sha256").read_bytes()
            selected_image_lines = selected_image_data.decode("utf-8").splitlines()
            self.assertEqual(len(selected_image_lines), len(set(selected_image_lines)))
            self.assertEqual(
                first_manifest["selected_images"]["count"], len(selected_image_lines)
            )
            self.assertEqual(
                first_manifest["selected_images"]["sha256"],
                hashlib.sha256(selected_image_data).hexdigest(),
            )
            self.assertEqual(
                first_manifest["selected_images"], first_build_report["selected_images"]
            )
            calibration_manifest = first_manifest["calibration"]
            self.assertEqual(calibration_manifest, first_build_report["calibration"])
            calibration_json = json.loads(
                (first / "output/calibration/z_calibration.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                calibration_manifest["fingerprint"],
                calibration_json["calibration_sha256"],
            )
            self.assertIn("seen_map", calibration_manifest["z_ranges"])
            self.assertIn("selected_images.sha256", first_checksums.decode("utf-8"))
            self.assertIn(
                "calibration/z_calibration.json", first_checksums.decode("utf-8")
            )
            self.assertIn(
                "calibration/z_extrema_rows.jsonl", first_checksums.decode("utf-8")
            )
            for relative in (
                "benchmark_manifest.json",
                "build_report.json",
                "calibration/z_calibration.json",
                "calibration/z_extrema_rows.jsonl",
                "splits/seen/seen_map/train.json",
                "splits/crossmap/cross_map/query_test.json",
            ):
                self.assertEqual(
                    (first / "output" / relative).read_bytes(),
                    (second / "output" / relative).read_bytes(),
                    relative,
                )

            valid = self._run(
                first,
                "validate",
                "--config",
                "config.yaml",
                "--decisions",
                "decisions.yaml",
                "--calibration-approval",
                str(first_calibration_approval.relative_to(first)),
            )
            self.assertEqual(valid.returncode, 0, valid.stderr)
            valid_without_decisions = self._run(
                first, "validate", "--config", "config.yaml"
            )
            self.assertNotEqual(valid_without_decisions.returncode, 0)
            self.assertIn("--decisions", valid_without_decisions.stderr)

            build_report = json.loads(
                (first / "output/build_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(build_report["decision_summary"]["excluded_rows"], 1)
            self.assertEqual(
                build_report["decision_summary"]["accepted_coordinate_anomalies"], 0
            )
            self.assertEqual(
                len(
                    json.loads(
                        (first / "output/splits/seen/seen_map/train.json").read_text()
                    )
                ),
                2,
            )
            continuous = json.loads(
                (
                    first / "output/splits/crossmap/cross_map/continuous_clips.json"
                ).read_text()
            )
            self.assertEqual(len(continuous["clips"]), 2)
            self.assertTrue(
                all(len(clip["frames"]) == 4 for clip in continuous["clips"])
            )
            frame_keys = [
                (row["map"], row["file_frame"])
                for clip in continuous["clips"]
                for row in clip["frames"]
            ]
            self.assertEqual(len(frame_keys), len(set(frame_keys)))

            _, selected_image_relative_path = selected_image_lines[0].split("  ", 1)
            (first / "source" / selected_image_relative_path).write_bytes(
                b"tampered-image"
            )
            tampered = self._run(
                first,
                "validate",
                "--config",
                "config.yaml",
                "--decisions",
                "decisions.yaml",
                "--calibration-approval",
                str(first_calibration_approval.relative_to(first)),
            )
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("selected_images.sha256", tampered.stderr)

    def test_calibration_is_deterministic_and_keeps_all_extrema_ties(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            self._make_workspace(first)
            self._make_workspace(second)
            self.assertEqual(
                self._run(first, "audit", "--config", "config.yaml").returncode, 0
            )
            self.assertEqual(
                self._run(second, "audit", "--config", "config.yaml").returncode, 0
            )
            first_decisions = self._approve_decisions(first)
            second_decisions = self._approve_decisions(second)
            self._calibrate_and_approve(first, first_decisions)
            self._calibrate_and_approve(second, second_decisions)

            first_calibration = (
                first / "output/calibration/z_calibration.json"
            ).read_bytes()
            second_calibration = (
                second / "output/calibration/z_calibration.json"
            ).read_bytes()
            first_extrema = (
                first / "output/calibration/z_extrema_rows.jsonl"
            ).read_bytes()
            second_extrema = (
                second / "output/calibration/z_extrema_rows.jsonl"
            ).read_bytes()
            self.assertEqual(first_calibration, second_calibration)
            self.assertEqual(first_extrema, second_extrema)

            calibration = json.loads(first_calibration)
            self.assertEqual(calibration["retained_rows"]["total"], 128)
            self.assertEqual(calibration["retained_rows"]["by_map"]["seen_map"], 64)
            self.assertEqual(calibration["z_ranges"]["seen_map"]["z_min"], 20.0)
            self.assertEqual(calibration["z_ranges"]["seen_map"]["z_max"], 27.0)
            extrema_rows = [
                json.loads(line)
                for line in first_extrema.decode("utf-8").splitlines()
                if line.strip()
            ]
            seen_extrema_rows = [
                row for row in extrema_rows if row["map"] == "seen_map"
            ]
            self.assertEqual(len(seen_extrema_rows), 16)
            self.assertEqual(
                sum(row["extrema"] == ["min"] for row in seen_extrema_rows), 8
            )
            self.assertEqual(
                sum(row["extrema"] == ["max"] for row in seen_extrema_rows), 8
            )
            self.assertEqual(
                {
                    row["record_id"]
                    for row in seen_extrema_rows
                    if row["extrema"] == ["min"]
                },
                {"0"},
            )
            self.assertEqual(
                {
                    row["record_id"]
                    for row in seen_extrema_rows
                    if row["extrema"] == ["max"]
                },
                {"7"},
            )

    def test_exclusions_are_applied_before_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            self._make_workspace(workspace)
            audit = self._run(workspace, "audit", "--config", "config.yaml")
            self.assertEqual(audit.returncode, 0, audit.stderr)
            decisions = self._approve_decisions(
                workspace,
                exclude_records={"seen_map": ["0", "7"]},
            )
            self._calibrate_and_approve(workspace, decisions)
            calibration = json.loads(
                (workspace / "output/calibration/z_calibration.json").read_text(
                    encoding="utf-8"
                )
            )
            seen_range = calibration["z_ranges"]["seen_map"]
            self.assertEqual(seen_range["z_min"], 21.0)
            self.assertEqual(seen_range["z_max"], 26.0)
            self.assertEqual(seen_range["retained_row_count"], 48)
            extrema_rows = [
                json.loads(line)
                for line in (workspace / "output/calibration/z_extrema_rows.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(
                {row["record_id"] for row in extrema_rows if row["map"] == "seen_map"},
                {"1", "6"},
            )

    def test_build_requires_calibration_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            self._make_workspace(workspace)
            audit = self._run(workspace, "audit", "--config", "config.yaml")
            self.assertEqual(audit.returncode, 0, audit.stderr)
            decisions = self._approve_decisions(workspace)
            calibrated = self._run(
                workspace,
                "calibrate",
                "--config",
                "config.yaml",
                "--decisions",
                "decisions.yaml",
            )
            self.assertEqual(calibrated.returncode, 0, calibrated.stderr)
            result = self._run(
                workspace,
                "build",
                "--config",
                "config.yaml",
                "--decisions",
                str(decisions.relative_to(workspace)),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--calibration-approval", result.stderr)

    def test_stale_or_tampered_calibration_approval_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            self._make_workspace(workspace)
            audit = self._run(workspace, "audit", "--config", "config.yaml")
            self.assertEqual(audit.returncode, 0, audit.stderr)
            decisions = self._approve_decisions(workspace)
            approval = self._calibrate_and_approve(workspace, decisions)

            stale = yaml.safe_load(approval.read_text(encoding="utf-8"))
            stale["review"]["calibration_sha256"] = "0" * 64
            stale_path = workspace / "stale_calibration_approval.yaml"
            stale_path.write_text(
                yaml.safe_dump(stale, sort_keys=False), encoding="utf-8"
            )
            stale_result = self._run(
                workspace,
                "build",
                "--config",
                "config.yaml",
                "--decisions",
                "decisions.yaml",
                "--calibration-approval",
                str(stale_path.relative_to(workspace)),
            )
            self.assertNotEqual(stale_result.returncode, 0)
            self.assertIn("calibration_sha256", stale_result.stderr)

            valid_build = self._run(
                workspace,
                "build",
                "--config",
                "config.yaml",
                "--decisions",
                "decisions.yaml",
                "--calibration-approval",
                str(approval.relative_to(workspace)),
            )
            self.assertEqual(valid_build.returncode, 0, valid_build.stderr)
            calibration_path = workspace / "output/calibration/z_calibration.json"
            calibration_path.write_bytes(calibration_path.read_bytes() + b" \n")
            tampered = self._run(
                workspace,
                "validate",
                "--config",
                "config.yaml",
                "--decisions",
                "decisions.yaml",
                "--calibration-approval",
                str(approval.relative_to(workspace)),
            )
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("independent full-corpus recomputation", tampered.stderr)

    def test_overwrite_replaces_known_files_without_requiring_data_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            self._make_workspace(workspace)
            result = self._run(workspace, "audit", "--config", "config.yaml")
            self.assertEqual(result.returncode, 0, result.stderr)
            unrelated = workspace / "output" / "unrelated.txt"
            unrelated.write_text("keep me", encoding="utf-8")
            second = self._run(
                workspace, "audit", "--config", "config.yaml", "--overwrite"
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep me")

    def test_radar_drift_invalidates_approved_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            self._make_workspace(workspace)
            audit = self._run(workspace, "audit", "--config", "config.yaml")
            self.assertEqual(audit.returncode, 0, audit.stderr)
            report = json.loads(
                (workspace / "output/audit/audit_report.json").read_text(
                    encoding="utf-8"
                )
            )
            decisions = yaml.safe_load(
                (workspace / "output/audit/anomaly_decisions.template.yaml").read_text(
                    encoding="utf-8"
                )
            )
            decisions["review"]["status"] = "approved"
            decisions["review"]["reviewer"] = "unit-test"
            decisions["review"]["reviewed_at"] = "2026-08-27T00:00:00Z"
            decisions["review"]["references"] = ["synthetic audit"]
            decisions["coordinate_candidates"]["default_action"] = "exclude"
            decisions["coordinate_candidates"]["exclude_file_frames"] = {
                "seen_map": ["file_num0_frame_0"]
            }
            decisions["review"]["audit_report_sha256"] = report["audit_report_sha256"]
            (workspace / "decisions.yaml").write_text(
                yaml.safe_dump(decisions, sort_keys=False), encoding="utf-8"
            )
            calibration_approval = self._calibrate_and_approve(
                workspace, workspace / "decisions.yaml"
            )
            radar = workspace / "source" / "seen_map" / "seen_map_radar.png"
            radar.write_bytes(b"changed-radar")
            result = self._run(
                workspace,
                "build",
                "--config",
                "config.yaml",
                "--decisions",
                "decisions.yaml",
                "--calibration-approval",
                str(calibration_approval.relative_to(workspace)),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("radar asset hashes", result.stderr)

    def test_approval_metadata_and_candidate_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            self._make_workspace(workspace)
            audit = self._run(workspace, "audit", "--config", "config.yaml")
            self.assertEqual(audit.returncode, 0, audit.stderr)
            template_path = workspace / "output/audit/anomaly_decisions.template.yaml"
            decisions = yaml.safe_load(template_path.read_text(encoding="utf-8"))
            decisions["review"]["status"] = "approved"
            decisions["coordinate_candidates"]["default_action"] = "exclude"
            decisions["coordinate_candidates"]["exclude_file_frames"] = {
                "seen_map": ["file_num0_frame_0"]
            }
            decisions_path = workspace / "decisions.yaml"

            decisions_path.write_text(
                yaml.safe_dump(decisions, sort_keys=False), encoding="utf-8"
            )
            missing_metadata = self._run(
                workspace,
                "build",
                "--config",
                "config.yaml",
                "--decisions",
                "decisions.yaml",
                "--calibration-approval",
                "missing-calibration.yaml",
            )
            self.assertNotEqual(missing_metadata.returncode, 0)
            self.assertIn("reviewer", missing_metadata.stderr)

            decisions["review"]["reviewer"] = "unit-test"
            decisions["review"]["reviewed_at"] = "2026-08-27T00:00:00Z"
            decisions["review"]["references"] = ["synthetic audit"]
            decisions["review"]["candidate_file_sha256"] = "0" * 64
            decisions_path.write_text(
                yaml.safe_dump(decisions, sort_keys=False), encoding="utf-8"
            )
            stale_candidate_binding = self._run(
                workspace,
                "build",
                "--config",
                "config.yaml",
                "--decisions",
                "decisions.yaml",
                "--calibration-approval",
                "missing-calibration.yaml",
            )
            self.assertNotEqual(stale_candidate_binding.returncode, 0)
            self.assertIn("candidate_file_sha256", stale_candidate_binding.stderr)

    def test_unknown_decision_file_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            self._make_workspace(workspace)
            audit = self._run(workspace, "audit", "--config", "config.yaml")
            self.assertEqual(audit.returncode, 0, audit.stderr)
            decisions = yaml.safe_load(
                (workspace / "output/audit/anomaly_decisions.template.yaml").read_text(
                    encoding="utf-8"
                )
            )
            decisions["review"].update(
                {
                    "status": "approved",
                    "reviewer": "unit-test",
                    "reviewed_at": "2026-08-27T00:00:00Z",
                    "references": ["synthetic audit"],
                }
            )
            decisions["coordinate_candidates"]["default_action"] = "exclude"
            decisions["coordinate_candidates"]["exclude_file_frames"] = {
                "seen_map": ["file_num999_frame_999"]
            }
            (workspace / "decisions.yaml").write_text(
                yaml.safe_dump(decisions, sort_keys=False), encoding="utf-8"
            )
            result = self._run(
                workspace,
                "build",
                "--config",
                "config.yaml",
                "--decisions",
                "decisions.yaml",
                "--calibration-approval",
                "missing-calibration.yaml",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unknown file_frame IDs", result.stderr)


if __name__ == "__main__":
    unittest.main()
