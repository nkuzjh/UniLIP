#!/usr/bin/env python3
"""Export deterministic medium-Z and angle-jump review lists."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


JUMP_COLUMNS = (
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


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _parse_exclusions(
    decisions: Mapping[str, Any],
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


def _circular_delta(first: float, second: float) -> float:
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def _record_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def collect_jump_edges(
    config: Mapping[str, Any],
    decisions: Mapping[str, Any],
    source_root: Path,
) -> list[dict[str, Any]]:
    """Return each retained neighboring-frame edge exactly once."""

    source = config["source"]
    positions_name = str(source["positions_file"])
    images_dir = str(source["images_dir"])
    image_extension = str(source["image_extension"])
    record_pattern = re.compile(str(source["record_regex"]))
    max_frame_gap = int(config["counts"]["continuous"]["max_frame_gap"])
    excluded_records, excluded_frames = _parse_exclusions(decisions)
    map_names = [
        str(value)
        for value in (*config["maps"]["seen"], *config["maps"]["crossmap"])
    ]

    edges: list[dict[str, Any]] = []
    for map_name in map_names:
        positions_path = source_root / map_name / positions_name
        with positions_path.open("r", encoding="utf-8") as handle:
            source_rows = json.load(handle)
        if not isinstance(source_rows, list):
            raise ValueError(f"expected a JSON list: {positions_path}")

        by_record: dict[str, list[tuple[int, str, dict[str, float]]]] = defaultdict(
            list
        )
        map_excluded_records = excluded_records.get(map_name, set())
        map_excluded_frames = excluded_frames.get(map_name, set())
        for row in source_rows:
            file_frame = str(row["file_frame"])
            match = record_pattern.fullmatch(file_frame)
            if match is None:
                raise ValueError(f"invalid file_frame {file_frame!r} in {positions_path}")
            record_id = str(match.group("record"))
            frame_id = int(match.group("frame"))
            if (
                record_id in map_excluded_records
                or file_frame in map_excluded_frames
            ):
                continue
            coordinates = {
                name: float(row[name])
                for name in ("x", "y", "z", "angle_h", "angle_v")
            }
            by_record[record_id].append((frame_id, file_frame, coordinates))

        for record_id in sorted(by_record, key=_record_sort_key):
            rows = sorted(by_record[record_id], key=lambda value: value[0])
            for previous, current in zip(rows, rows[1:]):
                frame_gap = current[0] - previous[0]
                if not 1 <= frame_gap <= max_frame_gap:
                    continue
                previous_coords = previous[2]
                current_coords = current[2]
                xy_delta = math.hypot(
                    current_coords["x"] - previous_coords["x"],
                    current_coords["y"] - previous_coords["y"],
                )
                z_delta = abs(current_coords["z"] - previous_coords["z"])
                yaw_delta = math.degrees(
                    _circular_delta(
                        current_coords["angle_h"], previous_coords["angle_h"]
                    )
                )
                pitch_delta = math.degrees(
                    abs(current_coords["angle_v"] - previous_coords["angle_v"])
                )
                edges.append(
                    {
                        "map": map_name,
                        "file_num": record_id,
                        "frame_start": previous[0],
                        "frame_end": current[0],
                        "frame_gap": frame_gap,
                        "file_frame_start": previous[1],
                        "file_frame_end": current[1],
                        "xy_delta": xy_delta,
                        "z_delta": z_delta,
                        "yaw_delta_degrees": yaw_delta,
                        "pitch_delta_degrees": pitch_delta,
                        "angle_delta_degrees": max(yaw_delta, pitch_delta),
                        "source_image_start": str(
                            source_root
                            / map_name
                            / images_dir
                            / f"{previous[1]}{image_extension}"
                        ),
                        "source_image_end": str(
                            source_root
                            / map_name
                            / images_dir
                            / f"{current[1]}{image_extension}"
                        ),
                    }
                )
    return edges


def select_medium_z_edges(
    edges: Sequence[dict[str, Any]], minimum: float, maximum: float
) -> list[dict[str, Any]]:
    return [row for row in edges if minimum < row["z_delta"] <= maximum]


def select_angle_edges(
    edges: Sequence[dict[str, Any]], threshold_degrees: float
) -> list[dict[str, Any]]:
    return [
        row for row in edges if row["angle_delta_degrees"] > threshold_degrees
    ]


def _atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=JUMP_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)
    return count


def _atomic_text(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        for row in rows:
            handle.write(
                f"{row['map']}/{row['file_num']}: "
                f"frame_{row['frame_start']} -> frame_{row['frame_end']}\n"
            )
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


def _record_count(rows: Sequence[dict[str, Any]]) -> int:
    return len({(row["map"], row["file_num"]) for row in rows})


def _check_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"jump review output exists; pass --overwrite: {joined}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--medium-z-min", type=float, default=50.0)
    parser.add_argument("--medium-z-max", type=float, default=300.0)
    parser.add_argument(
        "--angle-thresholds",
        type=float,
        nargs="+",
        default=(45.0, 60.0, 90.0, 120.0, 150.0),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.medium_z_min < 0 or args.medium_z_max <= args.medium_z_min:
        parser.error("medium-Z bounds must satisfy 0 <= min < max")
    angle_thresholds = sorted(set(args.angle_thresholds))
    if not angle_thresholds or angle_thresholds[0] < 0:
        parser.error("angle thresholds must be non-negative")

    cwd = Path.cwd()
    config_path = args.config.resolve()
    decisions_path = args.decisions.resolve()
    config = _load_yaml(config_path)
    decisions = _load_yaml(decisions_path)
    source_root = _resolve(cwd, str(config["paths"]["source_root"])).resolve()
    benchmark_root = _resolve(cwd, str(config["paths"]["output_root"])).resolve()
    output_dir = (
        args.output_dir or benchmark_root / "audit" / "jump_review"
    ).resolve()

    medium_stem = (
        f"medium_z_jumps_{args.medium_z_min:g}_{args.medium_z_max:g}"
    )
    output_stems = [medium_stem] + [
        f"angle_jumps_gt_{threshold:03g}" for threshold in angle_thresholds
    ]
    output_paths = [
        output_dir / f"{stem}.{suffix}"
        for stem in output_stems
        for suffix in ("csv", "txt")
    ]
    manifest_path = output_dir / "jump_review_manifest.json"
    _check_outputs((*output_paths, manifest_path), args.overwrite)

    edges = collect_jump_edges(config, decisions, source_root)
    selections: list[tuple[str, str, float | list[float], list[dict[str, Any]]]] = [
        (
            medium_stem,
            "medium_z",
            [args.medium_z_min, args.medium_z_max],
            select_medium_z_edges(edges, args.medium_z_min, args.medium_z_max),
        )
    ]
    selections.extend(
        (
            f"angle_jumps_gt_{threshold:03g}",
            "angle",
            threshold,
            select_angle_edges(edges, threshold),
        )
        for threshold in angle_thresholds
    )

    outputs: dict[str, Any] = {}
    for stem, kind, threshold, rows in selections:
        csv_path = output_dir / f"{stem}.csv"
        text_path = output_dir / f"{stem}.txt"
        csv_count = _atomic_csv(csv_path, rows)
        text_count = _atomic_text(text_path, rows)
        assert csv_count == text_count
        outputs[stem] = {
            "kind": kind,
            "threshold": threshold,
            "edges": csv_count,
            "records": _record_count(rows),
            "csv": {"path": str(csv_path), "sha256": _sha256(csv_path)},
            "text": {"path": str(text_path), "sha256": _sha256(text_path)},
        }

    manifest = {
        "schema_version": 1,
        "benchmark_id": config["benchmark"]["id"],
        "benchmark_version": config["benchmark"]["version"],
        "semantics": {
            "edge": (
                "one retained neighboring-frame pair within max_frame_gap; "
                "record and frame exclusions are applied before adjacency"
            ),
            "medium_z": "minimum < abs(delta_z) <= maximum",
            "angle": (
                "max(circular abs delta_yaw, abs delta_pitch) > threshold; "
                "threshold files are cumulative"
            ),
        },
        "inputs": {
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "decisions": str(decisions_path),
            "decisions_sha256": _sha256(decisions_path),
        },
        "outputs": outputs,
    }
    _atomic_json(manifest_path, manifest)
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
