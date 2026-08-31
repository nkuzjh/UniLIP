#!/usr/bin/env python3
"""Compare raw and cleaned per-map Z extrema for CSGO Benchmark v2.

The command consumes the benchmark configuration and an exclusion YAML.  The
exclusion YAML may contain ``exclude_records`` and ``exclude_file_frames`` at
the top level, or under ``coordinate_candidates`` as in the formal approval
file.  It writes clean extrema CSVs compatible with
``visualize_csgo_benchmark_v2_extrema.py`` and a manifest describing every
input, exclusion hit, overlap, and output hash.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml


COMPARISON_COLUMNS = (
    "setting",
    "map",
    "source_rows",
    "excluded_rows",
    "retained_rows",
    "raw_z_min",
    "raw_z_min_frame_count",
    "raw_z_max",
    "raw_z_max_frame_count",
    "clean_z_min",
    "clean_z_min_frame_count",
    "clean_z_max",
    "clean_z_max_frame_count",
    "z_min_changed",
    "z_max_changed",
)

SUMMARY_COLUMNS = (
    "setting",
    "map",
    "source_rows",
    "excluded_rows",
    "retained_rows",
    "z_min",
    "z_min_frame_count",
    "z_max",
    "z_max_frame_count",
)

REVIEW_COLUMNS = (
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

BOUND_ORDER = {"min": 0, "max": 1}
SETTING_ORDER = {"seen": 0, "crossmap": 1}


class ExportError(ValueError):
    """Raised when the config, exclusions, or source data is invalid."""


@dataclass(frozen=True)
class MapSpec:
    setting: str
    map_name: str
    positions_path: Path


@dataclass(frozen=True)
class PositionRow:
    map_name: str
    source_index: int
    file_num: int
    frame: int
    file_frame: str
    z: Decimal


@dataclass(frozen=True)
class ConfigInfo:
    benchmark_id: str
    benchmark_version: Any
    source_root: Path
    positions_file: str
    images_dir: str
    image_extension: str
    record_regex: str
    record_pattern: re.Pattern[str]
    maps: tuple[MapSpec, ...]


@dataclass(frozen=True)
class Exclusions:
    source_format: str
    records: dict[str, frozenset[int]]
    file_frames: dict[str, frozenset[str]]
    declared_record_items: int
    declared_record_unique_items: int
    declared_file_frame_items: int


@dataclass(frozen=True)
class MapResult:
    spec: MapSpec
    source_rows: tuple[PositionRow, ...]
    clean_rows: tuple[PositionRow, ...]
    raw_z_min: Decimal
    raw_z_max: Decimal
    clean_z_min: Decimal
    clean_z_max: Decimal
    raw_min_rows: tuple[PositionRow, ...]
    raw_max_rows: tuple[PositionRow, ...]
    clean_min_rows: tuple[PositionRow, ...]
    clean_max_rows: tuple[PositionRow, ...]
    record_rows_hit: int
    file_frame_rows_hit: int
    overlap_rows: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ExportError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExportError(f"missing YAML file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ExportError(f"cannot read YAML file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExportError(f"expected a YAML object: {path}")
    return value


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExportError(f"{field} must be a non-empty string")
    return value.strip()


def _resolve_path(value: Any, field: str, cwd: Path) -> Path:
    if isinstance(value, Path):
        path = value
    else:
        text = _require_nonempty_string(value, field)
        path = Path(text)
    return (path if path.is_absolute() else cwd / path).resolve()


def _parse_config(config: Mapping[str, Any], cwd: Path) -> ConfigInfo:
    paths = config.get("paths", {})
    if paths is None:
        paths = {}
    if not isinstance(paths, Mapping):
        raise ExportError("config.paths must be an object")
    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ExportError("config.source must be an object")

    source_root_value = paths.get("source_root")
    if source_root_value is None:
        source_root_value = config.get("source_root", source.get("source_root"))
    source_root = _resolve_path(source_root_value, "source_root", cwd)

    positions_file = _require_nonempty_string(
        source.get("positions_file", config.get("positions_file")),
        "source.positions_file",
    )
    images_dir = _require_nonempty_string(
        source.get("images_dir", config.get("images_dir", "imgs")),
        "source.images_dir",
    )
    image_extension = _require_nonempty_string(
        source.get("image_extension", config.get("image_extension", ".jpg")),
        "source.image_extension",
    )
    record_regex = _require_nonempty_string(
        source.get("record_regex", config.get("record_regex")),
        "source.record_regex",
    )
    try:
        record_pattern = re.compile(record_regex)
    except re.error as exc:
        raise ExportError(f"source.record_regex is invalid: {exc}") from exc
    if not {"record", "frame"}.issubset(record_pattern.groupindex):
        raise ExportError(
            "source.record_regex must define named groups 'record' and 'frame'"
        )

    maps = config.get("maps")
    if not isinstance(maps, Mapping):
        raise ExportError("config.maps must be an object")
    map_specs: list[MapSpec] = []
    seen_names: set[str] = set()
    for setting in ("seen", "crossmap"):
        values = maps.get(setting)
        if not isinstance(values, list) or not values:
            raise ExportError(f"maps.{setting} must be a non-empty list")
        for index, value in enumerate(values):
            map_name = _require_nonempty_string(value, f"maps.{setting}[{index}]")
            if map_name in seen_names:
                raise ExportError(f"map appears more than once: {map_name}")
            seen_names.add(map_name)
            map_specs.append(
                MapSpec(
                    setting=setting,
                    map_name=map_name,
                    positions_path=(
                        source_root / map_name / positions_file
                    ).resolve(),
                )
            )

    benchmark = config.get("benchmark", {})
    if benchmark is None:
        benchmark = {}
    if not isinstance(benchmark, Mapping):
        raise ExportError("config.benchmark must be an object")
    benchmark_id = str(benchmark.get("id", "csgo_benchmark_v2"))
    return ConfigInfo(
        benchmark_id=benchmark_id,
        benchmark_version=benchmark.get("version"),
        source_root=source_root,
        positions_file=positions_file,
        images_dir=images_dir,
        image_extension=image_extension,
        record_regex=record_regex,
        record_pattern=record_pattern,
        maps=tuple(map_specs),
    )


def _parse_record_id(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ExportError(f"{field} must be a non-negative integer record ID")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        number = int(value.strip())
    else:
        raise ExportError(f"{field} must be a non-negative integer record ID")
    if number < 0:
        raise ExportError(f"{field} must be a non-negative integer record ID")
    return number


def _parse_file_frame(
    value: Any,
    pattern: re.Pattern[str],
    field: str,
) -> tuple[int, int]:
    if not isinstance(value, str) or not value:
        raise ExportError(f"{field} must be a non-empty file_frame string")
    match = pattern.fullmatch(value)
    if match is None:
        raise ExportError(
            f"{field} has malformed file_frame {value!r}; it does not match "
            "source.record_regex"
        )
    record_text = match.group("record")
    frame_text = match.group("frame")
    if not isinstance(record_text, str) or not isinstance(frame_text, str):
        raise ExportError(
            f"{field} must contain non-optional record/frame components: "
            f"{value!r}"
        )
    if not re.fullmatch(r"\d+", record_text) or not re.fullmatch(
        r"\d+", frame_text
    ):
        raise ExportError(
            f"{field} must contain non-negative integer record/frame components: "
            f"{value!r}"
        )
    return int(record_text), int(frame_text)


def _finite_decimal(value: Any, field: str, source: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ExportError(f"{source}: {field} must be a finite number")
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError, AttributeError) as exc:
        raise ExportError(f"{source}: invalid {field}={value!r}") from exc
    if not number.is_finite():
        raise ExportError(f"{source}: {field} must be finite, got {value!r}")
    return number


def _load_positions(config: ConfigInfo, spec: MapSpec) -> tuple[PositionRow, ...]:
    if not spec.positions_path.is_file():
        raise ExportError(
            f"missing positions file for {spec.map_name}: {spec.positions_path}"
        )
    try:
        with spec.positions_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"cannot read positions file {spec.positions_path}: {exc}") from exc
    if not isinstance(value, list):
        raise ExportError(f"expected a JSON list: {spec.positions_path}")

    rows: list[PositionRow] = []
    seen_file_frames: set[str] = set()
    for source_index, row in enumerate(value):
        source = f"{spec.positions_path}:{source_index + 1}"
        if not isinstance(row, dict):
            raise ExportError(f"{source}: expected a JSON object")
        if "file_frame" not in row:
            raise ExportError(f"{source}: missing file_frame")
        file_frame = row["file_frame"]
        file_num, frame = _parse_file_frame(
            file_frame, config.record_pattern, f"{source}.file_frame"
        )
        if file_frame in seen_file_frames:
            raise ExportError(f"{source}: duplicate file_frame {file_frame!r}")
        seen_file_frames.add(file_frame)
        if "map" in row and row["map"] != spec.map_name:
            raise ExportError(
                f"{source}: map {row['map']!r} disagrees with directory "
                f"{spec.map_name!r}"
            )
        z = _finite_decimal(row.get("z"), "z", source)
        # Validate optional numeric pose fields too when present. This catches
        # NaN/Infinity before a source can silently enter a benchmark split.
        for field in ("x", "y", "angle_h", "angle_v"):
            if field in row:
                _finite_decimal(row[field], field, source)
        rows.append(
            PositionRow(
                map_name=spec.map_name,
                source_index=source_index,
                file_num=file_num,
                frame=frame,
                file_frame=file_frame,
                z=z,
            )
        )
    if not rows:
        raise ExportError(f"empty positions source for map {spec.map_name}")
    return tuple(rows)


def _parse_exclusion_map(
    value: Any,
    field: str,
    known_maps: set[str],
    *,
    parse_value: Callable[[Any, str], Any],
    reject_duplicates: bool = False,
) -> tuple[dict[str, set[Any]], int]:
    if value is None:
        raise ExportError(f"{field} must be an object")
    if not isinstance(value, Mapping):
        raise ExportError(f"{field} must be an object mapping map names to lists")
    result: dict[str, set[Any]] = {}
    raw_count = 0
    for map_name, values in value.items():
        if not isinstance(map_name, str) or not map_name.strip():
            raise ExportError(f"{field} contains an invalid map name")
        map_name = map_name.strip()
        if map_name not in known_maps:
            raise ExportError(f"{field}.{map_name} refers to an unknown map")
        if not isinstance(values, list):
            raise ExportError(f"{field}.{map_name} must be a list")
        parsed_items: list[Any] = []
        for index, item in enumerate(values):
            raw_count += 1
            parsed_items.append(parse_value(item, f"{field}.{map_name}[{index}]"))
        if reject_duplicates and len(parsed_items) != len(set(parsed_items)):
            raise ExportError(f"{field}.{map_name} contains duplicate file_frame")
        result[map_name] = set(parsed_items)
    return result, raw_count


def _parse_exclusions(
    decisions: Mapping[str, Any],
    config: ConfigInfo,
) -> Exclusions:
    known_maps = {spec.map_name for spec in config.maps}
    has_direct = (
        "exclude_records" in decisions or "exclude_file_frames" in decisions
    )
    nested = decisions.get("coordinate_candidates")
    if has_direct and "coordinate_candidates" in decisions:
        raise ExportError(
            "decisions must use either top-level exclusions or "
            "coordinate_candidates exclusions, not both"
        )
    if has_direct:
        section: Mapping[str, Any] = decisions
        source_format = "top_level"
        records_field = "exclude_records"
        frames_field = "exclude_file_frames"
    elif "coordinate_candidates" in decisions:
        if not isinstance(nested, Mapping):
            raise ExportError("decisions.coordinate_candidates must be an object")
        section = nested
        source_format = "coordinate_candidates"
        records_field = "coordinate_candidates.exclude_records"
        frames_field = "coordinate_candidates.exclude_file_frames"
    else:
        section = {}
        source_format = "empty"
        records_field = "exclude_records"
        frames_field = "exclude_file_frames"

    records_raw, record_count = _parse_exclusion_map(
        section.get("exclude_records", {}),
        records_field,
        known_maps,
        parse_value=_parse_record_id,
    )
    def parse_frame(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value:
            raise ExportError(f"{field} must be a non-empty file_frame string")
        _parse_file_frame(value, config.record_pattern, field)
        return value

    frames_raw, frame_count = _parse_exclusion_map(
        section.get("exclude_file_frames", {}),
        frames_field,
        known_maps,
        parse_value=parse_frame,
        reject_duplicates=True,
    )

    file_frames: dict[str, set[str]] = {}
    for map_name, values in frames_raw.items():
        parsed: set[str] = set()
        for index, file_frame in enumerate(sorted(values)):
            if not file_frame:
                raise ExportError(
                    f"{frames_field}.{map_name}[{index}] must be non-empty"
                )
            _parse_file_frame(
                file_frame,
                config.record_pattern,
                f"{frames_field}.{map_name}[{index}]",
            )
            parsed.add(file_frame)
        file_frames[map_name] = parsed

    return Exclusions(
        source_format=source_format,
        records={
            map_name: frozenset(values) for map_name, values in records_raw.items()
        },
        file_frames={
            map_name: frozenset(values) for map_name, values in file_frames.items()
        },
        declared_record_items=record_count,
        declared_record_unique_items=sum(len(values) for values in records_raw.values()),
        declared_file_frame_items=frame_count,
    )


def _extreme_rows(
    rows: Sequence[PositionRow],
) -> tuple[Decimal, Decimal, tuple[PositionRow, ...], tuple[PositionRow, ...]]:
    if not rows:
        raise ExportError("cannot compute extrema for an empty retained map")
    z_min = min(row.z for row in rows)
    z_max = max(row.z for row in rows)
    min_rows = tuple(row for row in rows if row.z == z_min)
    max_rows = tuple(row for row in rows if row.z == z_max)
    return z_min, z_max, min_rows, max_rows


def _build_map_result(
    config: ConfigInfo,
    spec: MapSpec,
    exclusions: Exclusions,
) -> MapResult:
    source_rows = _load_positions(config, spec)
    known_records = {row.file_num for row in source_rows}
    known_file_frames = {row.file_frame for row in source_rows}
    excluded_records = exclusions.records.get(spec.map_name, frozenset())
    excluded_file_frames = exclusions.file_frames.get(spec.map_name, frozenset())

    unknown_records = sorted(excluded_records - known_records)
    if unknown_records:
        raise ExportError(
            f"exclude_records.{spec.map_name} contains non-existent record IDs: "
            + ", ".join(str(value) for value in unknown_records)
        )
    unknown_frames = sorted(excluded_file_frames - known_file_frames)
    if unknown_frames:
        raise ExportError(
            f"exclude_file_frames.{spec.map_name} contains non-existent file_frame: "
            + ", ".join(unknown_frames)
        )

    record_hit_rows = tuple(row for row in source_rows if row.file_num in excluded_records)
    frame_hit_rows = tuple(
        row for row in source_rows if row.file_frame in excluded_file_frames
    )
    record_hit_keys = {row.file_frame for row in record_hit_rows}
    frame_hit_keys = {row.file_frame for row in frame_hit_rows}
    overlap_rows = len(record_hit_keys & frame_hit_keys)
    clean_rows = tuple(
        row
        for row in source_rows
        if row.file_frame not in record_hit_keys | frame_hit_keys
    )
    if not clean_rows:
        raise ExportError(f"no retained rows for map {spec.map_name}")

    raw_z_min, raw_z_max, raw_min_rows, raw_max_rows = _extreme_rows(source_rows)
    clean_z_min, clean_z_max, clean_min_rows, clean_max_rows = _extreme_rows(clean_rows)
    return MapResult(
        spec=spec,
        source_rows=source_rows,
        clean_rows=clean_rows,
        raw_z_min=raw_z_min,
        raw_z_max=raw_z_max,
        clean_z_min=clean_z_min,
        clean_z_max=clean_z_max,
        raw_min_rows=raw_min_rows,
        raw_max_rows=raw_max_rows,
        clean_min_rows=clean_min_rows,
        clean_max_rows=clean_max_rows,
        record_rows_hit=len(record_hit_keys),
        file_frame_rows_hit=len(frame_hit_keys),
        overlap_rows=overlap_rows,
    )


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _image_path(config: ConfigInfo, row: PositionRow) -> Path:
    return (
        config.source_root
        / row.map_name
        / config.images_dir
        / f"{row.file_frame}{config.image_extension}"
    ).resolve()


def _review_row(config: ConfigInfo, row: PositionRow, bound: str) -> dict[str, Any]:
    return {
        "setting": next(
            spec.setting for spec in config.maps if spec.map_name == row.map_name
        ),
        "map": row.map_name,
        "bound": bound,
        "z": _decimal_text(row.z),
        "file_num": row.file_num,
        "frame": row.frame,
        "file_frame": row.file_frame,
        "source_image_path": str(_image_path(config, row)),
        "source_index": row.source_index,
    }


def _sort_review(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            SETTING_ORDER.get(row["setting"], len(SETTING_ORDER)),
            row["setting"],
            row["map"],
            BOUND_ORDER[row["bound"]],
            Decimal(str(row["z"])),
            int(row["file_num"]),
            int(row["frame"]),
            int(row["source_index"]),
        ),
    )


def _map_rows(config: ConfigInfo, result: MapResult, *, clean: bool) -> list[dict[str, Any]]:
    if clean:
        extrema = (("min", result.clean_min_rows), ("max", result.clean_max_rows))
    else:
        extrema = (("min", result.raw_min_rows), ("max", result.raw_max_rows))
    rows: list[dict[str, Any]] = []
    for bound, tied_rows in extrema:
        rows.extend(_review_row(config, row, bound) for row in tied_rows)
    return rows


def _comparison_row(result: MapResult) -> dict[str, Any]:
    source_count = len(result.source_rows)
    retained_count = len(result.clean_rows)
    return {
        "setting": result.spec.setting,
        "map": result.spec.map_name,
        "source_rows": source_count,
        "excluded_rows": source_count - retained_count,
        "retained_rows": retained_count,
        "raw_z_min": _decimal_text(result.raw_z_min),
        "raw_z_min_frame_count": len(result.raw_min_rows),
        "raw_z_max": _decimal_text(result.raw_z_max),
        "raw_z_max_frame_count": len(result.raw_max_rows),
        "clean_z_min": _decimal_text(result.clean_z_min),
        "clean_z_min_frame_count": len(result.clean_min_rows),
        "clean_z_max": _decimal_text(result.clean_z_max),
        "clean_z_max_frame_count": len(result.clean_max_rows),
        "z_min_changed": result.raw_z_min != result.clean_z_min,
        "z_max_changed": result.raw_z_max != result.clean_z_max,
    }


def _summary_row(result: MapResult) -> dict[str, Any]:
    row = _comparison_row(result)
    return {
        "setting": row["setting"],
        "map": row["map"],
        "source_rows": row["source_rows"],
        "excluded_rows": row["excluded_rows"],
        "retained_rows": row["retained_rows"],
        "z_min": row["clean_z_min"],
        "z_min_frame_count": row["clean_z_min_frame_count"],
        "z_max": row["clean_z_max"],
        "z_max_frame_count": row["clean_z_max_frame_count"],
    }


def _atomic_csv(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="ascii",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=list(columns),
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return count


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="ascii",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _check_output_paths(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise ExportError(
            "output already exists; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def _map_exclusion_manifest(
    result: MapResult,
    exclusions: Exclusions,
) -> dict[str, int]:
    map_name = result.spec.map_name
    records = exclusions.records.get(map_name, frozenset())
    frames = exclusions.file_frames.get(map_name, frozenset())
    return {
        "declared_record_ids": len(records),
        "declared_file_frames": len(frames),
        "matched_record_ids": len(records),
        "matched_file_frames": len(frames),
        "record_rows_hit": result.record_rows_hit,
        "file_frame_rows_hit": result.file_frame_rows_hit,
        "overlap_rows": result.overlap_rows,
        "effective_excluded_rows": len(result.source_rows) - len(result.clean_rows),
    }


def _build_manifest(
    config: ConfigInfo,
    config_path: Path,
    decisions_path: Path,
    exclusions: Exclusions,
    results: Sequence[MapResult],
    output_paths: Mapping[str, Path],
    output_rows: Mapping[str, int],
) -> dict[str, Any]:
    map_manifest = []
    for result in results:
        map_manifest.append(
            {
                "setting": result.spec.setting,
                "map": result.spec.map_name,
                "positions_path": str(result.spec.positions_path),
                "positions_sha256": _sha256(result.spec.positions_path),
                "source_rows": len(result.source_rows),
                "excluded_rows": len(result.source_rows) - len(result.clean_rows),
                "retained_rows": len(result.clean_rows),
                "raw_z_min": _decimal_text(result.raw_z_min),
                "raw_z_min_frame_count": len(result.raw_min_rows),
                "raw_z_max": _decimal_text(result.raw_z_max),
                "raw_z_max_frame_count": len(result.raw_max_rows),
                "clean_z_min": _decimal_text(result.clean_z_min),
                "clean_z_min_frame_count": len(result.clean_min_rows),
                "clean_z_max": _decimal_text(result.clean_z_max),
                "clean_z_max_frame_count": len(result.clean_max_rows),
                "z_min_changed": result.raw_z_min != result.clean_z_min,
                "z_max_changed": result.raw_z_max != result.clean_z_max,
                "exclusions": _map_exclusion_manifest(result, exclusions),
            }
        )

    declared_record_unique = sum(
        len(exclusions.records.get(result.spec.map_name, frozenset()))
        for result in results
    )
    declared_frames = sum(
        len(exclusions.file_frames.get(result.spec.map_name, frozenset()))
        for result in results
    )
    record_rows_hit = sum(result.record_rows_hit for result in results)
    frame_rows_hit = sum(result.file_frame_rows_hit for result in results)
    overlap_rows = sum(result.overlap_rows for result in results)
    effective_rows = sum(
        len(result.source_rows) - len(result.clean_rows) for result in results
    )

    outputs = {}
    for name, path in output_paths.items():
        if name == "manifest":
            continue
        outputs[name] = {
            "path": str(path),
            "rows": output_rows[name],
            "sha256": _sha256(path),
        }
    return {
        "schema": {
            "name": "csgo_benchmark_v2_z_extrema_comparison",
            "version": 1,
            "comparison_columns": list(COMPARISON_COLUMNS),
            "summary_columns": list(SUMMARY_COLUMNS),
            "review_columns": list(REVIEW_COLUMNS),
        },
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "benchmark": {
            "id": config.benchmark_id,
            "version": config.benchmark_version,
        },
        "inputs": {
            "config": {
                "path": str(config_path),
                "sha256": _sha256(config_path),
            },
            "decisions": {
                "path": str(decisions_path),
                "sha256": _sha256(decisions_path),
                "format": exclusions.source_format,
            },
            "source": {
                "source_root": str(config.source_root),
                "positions_file": config.positions_file,
                "record_regex": config.record_regex,
                "positions_sha256": {
                    result.spec.map_name: _sha256(result.spec.positions_path)
                    for result in results
                },
            },
        },
        "exclusions": {
            "semantics": "map-specific record ID OR exact file_frame; set union",
            "declared": {
                "record_items": exclusions.declared_record_items,
                "unique_record_ids": declared_record_unique,
                "file_frame_items": exclusions.declared_file_frame_items,
                "unique_file_frames": declared_frames,
            },
            "hits": {
                "record_ids": declared_record_unique,
                "file_frames": declared_frames,
                "record_rows": record_rows_hit,
                "file_frame_rows": frame_rows_hit,
                "effective_rows": effective_rows,
            },
            "overlap": {
                "file_frames_inside_excluded_records": overlap_rows,
                "rows": overlap_rows,
            },
            "by_map": {
                result.spec.map_name: _map_exclusion_manifest(result, exclusions)
                for result in results
            },
        },
        "maps": map_manifest,
        "outputs": outputs,
    }


def run_export(
    config_path: Path,
    decisions_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Run the comparison and return the written manifest."""

    working_dir = (cwd or Path.cwd()).resolve()
    config_path = _resolve_path(config_path, "config", working_dir)
    decisions_path = _resolve_path(decisions_path, "decisions", working_dir)
    output_dir = _resolve_path(output_dir, "output-dir", working_dir)
    config_data = _load_yaml(config_path)
    config = _parse_config(config_data, working_dir)
    decisions = _load_yaml(decisions_path)
    exclusions = _parse_exclusions(decisions, config)

    output_paths = {
        "z_extrema_comparison": output_dir / "z_extrema_comparison.csv",
        "z_extrema_summary": output_dir / "z_extrema_summary.csv",
        "z_extrema_review": output_dir / "z_extrema_review.csv",
        "z_extrema_raw_review": output_dir / "z_extrema_raw_review.csv",
        "manifest": output_dir / "z_extrema_comparison_manifest.json",
    }
    _check_output_paths(tuple(output_paths.values()), overwrite)

    results = tuple(
        _build_map_result(config, spec, exclusions) for spec in config.maps
    )
    comparison_rows = [_comparison_row(result) for result in results]
    summary_rows = [_summary_row(result) for result in results]
    clean_review_rows = _sort_review(
        row
        for result in results
        for row in _map_rows(config, result, clean=True)
    )
    raw_review_rows = _sort_review(
        row
        for result in results
        for row in _map_rows(config, result, clean=False)
    )

    output_rows = {
        "z_extrema_comparison": _atomic_csv(
            output_paths["z_extrema_comparison"], COMPARISON_COLUMNS, comparison_rows
        ),
        "z_extrema_summary": _atomic_csv(
            output_paths["z_extrema_summary"], SUMMARY_COLUMNS, summary_rows
        ),
        "z_extrema_review": _atomic_csv(
            output_paths["z_extrema_review"], REVIEW_COLUMNS, clean_review_rows
        ),
        "z_extrema_raw_review": _atomic_csv(
            output_paths["z_extrema_raw_review"], REVIEW_COLUMNS, raw_review_rows
        ),
    }
    manifest = _build_manifest(
        config,
        config_path,
        decisions_path,
        exclusions,
        results,
        output_paths,
        output_rows,
    )
    _atomic_json(output_paths["manifest"], manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        manifest = run_export(
            args.config,
            args.decisions,
            args.output_dir,
            overwrite=args.overwrite,
        )
    except (ExportError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "maps": len(manifest["maps"]),
                "outputs": manifest["outputs"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
