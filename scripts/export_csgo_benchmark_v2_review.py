#!/usr/bin/env python3
"""Export compact, reproducible review tables for CSGO Benchmark v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


CANDIDATE_COLUMNS = (
    "map",
    "file_num",
    "frame",
    "file_frame",
    "reasons",
    "x",
    "y",
    "z",
    "angle_h",
    "angle_v",
    "source_image_path",
    "source_index",
)

EXTREMA_COLUMNS = (
    "setting",
    "map",
    "bound",
    "z",
    "file_num",
    "frame",
    "file_frame",
    "source_image_path",
    "source_index",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML object: {path}")
    return value


def _atomic_csv(
    path: Path, columns: Iterable[str], rows: Iterable[dict[str, Any]]
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)
    return count


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _candidate_rows(
    candidate_path: Path, source_root: Path
) -> Iterable[dict[str, Any]]:
    with candidate_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {candidate_path}:{line_number}"
                ) from exc
            coordinates = row.get("coordinates", {})
            record_id = str(row["record_id"])
            frame_id = int(row["frame_id"])
            relative_image = Path(str(row["source_image_relative_path"]))
            yield {
                "map": row["map"],
                "file_num": record_id,
                "frame": frame_id,
                "file_frame": row["file_frame"],
                "reasons": ";".join(row.get("reasons", [])),
                "x": coordinates.get("x"),
                "y": coordinates.get("y"),
                "z": coordinates.get("z"),
                "angle_h": coordinates.get("angle_h"),
                "angle_v": coordinates.get("angle_v"),
                "source_image_path": str(source_root / relative_image),
                "source_index": row.get("source_index"),
            }


def _parse_exclusions(
    decisions: dict[str, Any],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    coordinate = decisions.get("coordinate_candidates", {})
    excluded_records = {
        str(map_name): {str(value) for value in values}
        for map_name, values in coordinate.get("exclude_records", {}).items()
    }
    excluded_frames = {
        str(map_name): {str(value) for value in values}
        for map_name, values in coordinate.get("exclude_file_frames", {}).items()
    }
    return excluded_records, excluded_frames


def _collect_extrema(
    config: dict[str, Any],
    decisions: dict[str, Any],
    source_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = config["source"]
    positions_name = str(source["positions_file"])
    images_dir = str(source["images_dir"])
    image_extension = str(source["image_extension"])
    record_pattern = re.compile(str(source["record_regex"]))
    excluded_records, excluded_frames = _parse_exclusions(decisions)

    setting_maps = {
        "seen": [str(value) for value in config["maps"]["seen"]],
        "crossmap": [str(value) for value in config["maps"]["crossmap"]],
    }
    extrema_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for setting, map_names in setting_maps.items():
        for map_name in map_names:
            positions_path = source_root / map_name / positions_name
            with positions_path.open("r", encoding="utf-8") as handle:
                source_rows = json.load(handle)
            if not isinstance(source_rows, list):
                raise ValueError(f"expected a JSON list: {positions_path}")

            retained: list[tuple[int, dict[str, Any], str, int]] = []
            map_excluded_records = excluded_records.get(map_name, set())
            map_excluded_frames = excluded_frames.get(map_name, set())
            for source_index, row in enumerate(source_rows):
                file_frame = str(row["file_frame"])
                match = record_pattern.fullmatch(file_frame)
                if match is None:
                    raise ValueError(
                        f"invalid file_frame {file_frame!r} in {positions_path}"
                    )
                record_id = str(match.group("record"))
                frame_id = int(match.group("frame"))
                if (
                    record_id in map_excluded_records
                    or file_frame in map_excluded_frames
                ):
                    continue
                retained.append((source_index, row, record_id, frame_id))

            if not retained:
                raise ValueError(f"no retained rows for map {map_name}")
            z_min = min(float(row[1]["z"]) for row in retained)
            z_max = max(float(row[1]["z"]) for row in retained)
            if not z_max > z_min:
                raise ValueError(
                    f"non-positive Z span for map {map_name}: [{z_min}, {z_max}]"
                )

            bound_counts: dict[str, int] = defaultdict(int)
            for source_index, row, record_id, frame_id in retained:
                z_value = float(row["z"])
                bounds: list[str] = []
                if z_value == z_min:
                    bounds.append("min")
                if z_value == z_max:
                    bounds.append("max")
                for bound in bounds:
                    bound_counts[bound] += 1
                    file_frame = str(row["file_frame"])
                    extrema_rows.append(
                        {
                            "setting": setting,
                            "map": map_name,
                            "bound": bound,
                            "z": row["z"],
                            "file_num": record_id,
                            "frame": frame_id,
                            "file_frame": file_frame,
                            "source_image_path": str(
                                source_root
                                / map_name
                                / images_dir
                                / f"{file_frame}{image_extension}"
                            ),
                            "source_index": source_index,
                        }
                    )

            summaries.append(
                {
                    "setting": setting,
                    "map": map_name,
                    "source_rows": len(source_rows),
                    "retained_rows": len(retained),
                    "excluded_rows": len(source_rows) - len(retained),
                    "z_min": z_min,
                    "z_min_frame_count": bound_counts["min"],
                    "z_max": z_max,
                    "z_max_frame_count": bound_counts["max"],
                }
            )

    extrema_rows.sort(
        key=lambda row: (
            row["setting"],
            row["map"],
            0 if row["bound"] == "min" else 1,
            int(row["file_num"]),
            int(row["frame"]),
            int(row["source_index"]),
        )
    )
    return extrema_rows, summaries


def _check_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"review output already exists; pass --overwrite: {joined}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cwd = Path.cwd()
    config_path = args.config.resolve()
    decisions_path = args.decisions.resolve()
    config = _load_yaml(config_path)
    decisions = _load_yaml(decisions_path)
    source_root = _resolve(cwd, str(config["paths"]["source_root"])).resolve()
    benchmark_root = _resolve(cwd, str(config["paths"]["output_root"])).resolve()
    output_dir = (
        args.output_dir or benchmark_root / "audit" / "review_exports"
    ).resolve()
    candidate_path = benchmark_root / "audit" / "coordinate_candidates.jsonl"

    candidate_csv = output_dir / "coordinate_candidates_review.csv"
    extrema_csv = output_dir / "z_extrema_review.csv"
    extrema_summary_csv = output_dir / "z_extrema_summary.csv"
    manifest_path = output_dir / "review_export_manifest.json"
    outputs = (candidate_csv, extrema_csv, extrema_summary_csv, manifest_path)
    _check_outputs(outputs, args.overwrite)

    candidate_count = _atomic_csv(
        candidate_csv,
        CANDIDATE_COLUMNS,
        _candidate_rows(candidate_path, source_root),
    )
    extrema_rows, summaries = _collect_extrema(config, decisions, source_root)
    extrema_count = _atomic_csv(extrema_csv, EXTREMA_COLUMNS, extrema_rows)
    summary_columns = tuple(summaries[0].keys())
    _atomic_csv(extrema_summary_csv, summary_columns, summaries)

    manifest = {
        "schema_version": 1,
        "benchmark_id": config["benchmark"]["id"],
        "benchmark_version": config["benchmark"]["version"],
        "semantics": {
            "coordinate_candidates": "one row per existing audit candidate",
            "z_extrema": (
                "all retained rows tied at exact per-map min or max after current "
                "decision exclusions; this is provisional and not a frozen calibration"
            ),
        },
        "inputs": {
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "decisions": str(decisions_path),
            "decisions_sha256": _sha256(decisions_path),
            "coordinate_candidates": str(candidate_path),
            "coordinate_candidates_sha256": _sha256(candidate_path),
        },
        "outputs": {
            "coordinate_candidates_review": {
                "path": str(candidate_csv),
                "rows": candidate_count,
                "sha256": _sha256(candidate_csv),
            },
            "z_extrema_review": {
                "path": str(extrema_csv),
                "rows": extrema_count,
                "sha256": _sha256(extrema_csv),
            },
            "z_extrema_summary": {
                "path": str(extrema_summary_csv),
                "rows": len(summaries),
                "sha256": _sha256(extrema_summary_csv),
            },
        },
    }
    _atomic_json(manifest_path, manifest)
    print(json.dumps(manifest["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
