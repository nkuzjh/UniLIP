"""Pure-Python runtime selection for the CSGO Benchmark v2 artifacts.

The builder intentionally writes one split file per map.  This module keeps
that layout visible to callers and does not import torch, PIL, or the training
dataset implementations.  A caller can therefore use the same selection
object from training, inference, or an offline evaluator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any


EXPECTED_BENCHMARK_ID = "csgo_benchmark_v2"
EXPECTED_SCHEMA_VERSION = 1

SUPPORTED_SPLITS = (
    "seen_train",
    "seen_validation",
    "seen_discrete_test",
    "seen_continuous",
    "crossmap_support",
    "crossmap_query_test",
    "crossmap_continuous",
)

_SPLIT_SETTING = {
    "seen_train": "seen",
    "seen_validation": "seen",
    "seen_discrete_test": "seen",
    "seen_continuous": "seen",
    "crossmap_support": "crossmap",
    "crossmap_query_test": "crossmap",
    "crossmap_continuous": "crossmap",
}

_SPLIT_FILE_NAME = {
    "seen_train": "train.json",
    "seen_validation": "validation.json",
    "seen_discrete_test": "discrete_test.json",
    "seen_continuous": "continuous_clips.json",
    "crossmap_support": "support_seed_{support_seed}.json",
    "crossmap_query_test": "query_test.json",
    "crossmap_continuous": "continuous_clips.json",
}

_REQUIRED_ROW_FIELDS = ("map", "file_frame", "x", "y", "z", "angle_h", "angle_v")


class BenchmarkV2Error(ValueError):
    """Raised when a Benchmark v2 artifact is invalid or inconsistent."""


@dataclass
class BenchmarkV2Selection:
    """Rows and provenance for one Benchmark v2 runtime selection.

    ``split_files`` maps every requested map to the file that supplied its
    rows.  ``clips_by_map`` is empty for discrete selections; for continuous
    selections it maps a map to an ordered list of clip objects whose
    ``frames`` lists are ordered and annotated.  Paths are absolute, resolved
    paths so downstream code cannot accidentally resolve
    ``maps/cs_office/...`` relative to the wrong directory.
    """

    manifest_path: Path
    benchmark_root: Path
    manifest: dict[str, Any]
    split: str
    map_names: list[str]
    rows: list[dict[str, Any]]
    z_ranges: dict[str, dict[str, Any]]
    radar_paths: dict[str, Path]
    source_root: Path
    image_extension: str
    clips_by_map: dict[str, list[dict[str, Any]]]
    split_files: dict[str, Path]
    support_seed: int
    shots_per_map: int


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def is_benchmark_v2_config(config: Any) -> bool:
    """Return whether a config opts into the Benchmark v2 runtime.

    The training configs use a flat ``benchmark_v2_manifest`` field.  A
    non-empty manifest path is the opt-in marker; the split is validated when
    a selection is actually loaded.
    """

    value = _config_value(config, "benchmark_v2_manifest")
    return isinstance(value, (str, os.PathLike)) and bool(os.fspath(value))


def load_benchmark_v2_selection(
    config: Any,
    map_names: Sequence[str] | str | None = None,
    split: str | None = None,
) -> BenchmarkV2Selection:
    """Load a selection using flat Benchmark v2 config fields.

    Recognized fields are ``benchmark_v2_manifest``, ``benchmark_v2_split``,
    ``benchmark_v2_support_seed``, ``benchmark_v2_shots_per_map``,
    ``benchmark_v2_image_extension``, and the existing optional ``data_dir``.
    An explicit function argument takes precedence over the config value.
    """

    manifest_path = _config_value(config, "benchmark_v2_manifest")
    if manifest_path in (None, ""):
        raise BenchmarkV2Error(
            "benchmark_v2_manifest is required for Benchmark v2 selection"
        )
    selected_split = split
    if selected_split is None:
        selected_split = _config_value(config, "benchmark_v2_split")
    if not isinstance(selected_split, str) or not selected_split:
        raise BenchmarkV2Error(
            "benchmark_v2_split is required when split is not provided"
        )

    selection = load_benchmark_v2_selection_from_args(
        manifest_path,
        selected_split,
        map_names=map_names,
        support_seed=_config_value(config, "benchmark_v2_support_seed", 0),
        shots_per_map=_config_value(config, "benchmark_v2_shots_per_map", 100),
        data_dir=_config_value(config, "data_dir"),
    )
    extension = _normalise_image_extension(
        _config_value(config, "benchmark_v2_image_extension")
    )
    if extension is not None:
        selection.image_extension = extension
    return selection


def load_benchmark_v2_selection_from_args(
    manifest_path: str | os.PathLike[str],
    split: str,
    map_names: Sequence[str] | str | None = None,
    support_seed: int = 0,
    shots_per_map: int = 100,
    data_dir: str | os.PathLike[str] | None = None,
) -> BenchmarkV2Selection:
    """Load and validate one Benchmark v2 split without ML dependencies.

    ``data_dir`` overrides ``manifest.source.root``.  Relative paths follow
    the repository working directory convention used by the builder; if that
    path does not exist, a relative source path is also tried below the
    manifest directory, which is useful for self-contained test bundles.
    """

    if split not in SUPPORTED_SPLITS:
        raise BenchmarkV2Error(
            f"unsupported Benchmark v2 split {split!r}; expected one of "
            f"{', '.join(SUPPORTED_SPLITS)}"
        )
    support_seed = _require_int(support_seed, "support_seed")
    shots_per_map = _require_int(shots_per_map, "shots_per_map")
    if split == "crossmap_support" and not 1 <= shots_per_map <= 100:
        raise BenchmarkV2Error(
            "crossmap support shots_per_map must be between 1 and 100"
        )

    manifest_path = _resolve_manifest_path(manifest_path)
    manifest = _load_json_object(manifest_path, "Benchmark v2 manifest")
    protocol = _validate_manifest(manifest)
    setting = _SPLIT_SETTING[split]
    allowed_maps = protocol[f"{setting}_maps"]
    if split == "crossmap_support" and support_seed not in protocol["support_seeds"]:
        raise BenchmarkV2Error(
            f"support_seed {support_seed} is not declared by the manifest; "
            f"expected one of {protocol['support_seeds']}"
        )
    requested_maps = _normalise_map_names(map_names)
    if requested_maps is None:
        requested_maps = list(allowed_maps)
    _validate_requested_maps(requested_maps, allowed_maps, split)

    source = manifest["source"]
    source_root_value = data_dir if data_dir is not None else source["root"]
    source_root = _resolve_source_root(source_root_value, manifest_path)
    z_ranges = _validate_z_ranges(manifest, protocol["all_maps"])
    selected_z_ranges = {name: z_ranges[name] for name in requested_maps}
    radar_paths = _resolve_radar_paths(
        source["radar_files"], source_root, requested_maps
    )
    image_extension = ".jpg"

    split_files = _resolve_split_files(
        manifest,
        manifest_path.parent,
        split,
        requested_maps,
        support_seed,
    )

    if split in {"seen_continuous", "crossmap_continuous"}:
        rows, clips_by_map = _load_continuous_selection(
            manifest,
            split,
            requested_maps,
            split_files,
            selected_z_ranges,
        )
    else:
        rows, clips_by_map = _load_discrete_selection(
            manifest,
            split,
            requested_maps,
            split_files,
            selected_z_ranges,
            support_seed,
            shots_per_map,
        )

    return BenchmarkV2Selection(
        manifest_path=manifest_path,
        benchmark_root=manifest_path.parent,
        manifest=manifest,
        split=split,
        map_names=requested_maps,
        rows=rows,
        z_ranges=selected_z_ranges,
        radar_paths=radar_paths,
        source_root=source_root,
        image_extension=image_extension,
        clips_by_map=clips_by_map,
        split_files=split_files,
        support_seed=support_seed,
        shots_per_map=shots_per_map,
    )


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkV2Error(f"{field_name} must be an integer, got {value!r}")
    return value


def _resolve_manifest_path(value: str | os.PathLike[str]) -> Path:
    try:
        path = Path(value).expanduser()
    except (TypeError, ValueError) as exc:
        raise BenchmarkV2Error(f"invalid benchmark_v2_manifest path: {value!r}") from exc
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark v2 manifest does not exist: {path}")
    return path


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkV2Error(f"could not read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkV2Error(f"{description} must be a JSON object: {path}")
    return value


def _validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != EXPECTED_SCHEMA_VERSION
    ):
        raise BenchmarkV2Error(
            f"manifest schema_version must be {EXPECTED_SCHEMA_VERSION}"
        )
    if manifest.get("benchmark_id") != EXPECTED_BENCHMARK_ID:
        raise BenchmarkV2Error(
            f"manifest benchmark_id must be {EXPECTED_BENCHMARK_ID!r}"
        )
    benchmark = manifest.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise BenchmarkV2Error("manifest.benchmark must be an object")
    if benchmark.get("id") != EXPECTED_BENCHMARK_ID:
        raise BenchmarkV2Error("manifest.benchmark.id does not match benchmark_id")
    _require_int(benchmark.get("global_seed"), "manifest.benchmark.global_seed")

    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise BenchmarkV2Error("manifest.protocol must be an object")
    seen_maps = _protocol_map_list(protocol, "seen_maps")
    crossmap_maps = _protocol_map_list(protocol, "crossmap_maps")
    overlap = set(seen_maps).intersection(crossmap_maps)
    if overlap:
        raise BenchmarkV2Error(
            f"manifest protocol maps overlap between seen and crossmap: "
            f"{sorted(overlap)}"
        )
    seen_splits = _protocol_name_list(protocol, "seen_splits")
    crossmap_splits = _protocol_name_list(protocol, "crossmap_splits")
    if set(seen_splits) != {"train", "validation", "discrete_test"}:
        raise BenchmarkV2Error(
            "manifest.protocol.seen_splits must contain train, validation, "
            "and discrete_test"
        )
    if set(crossmap_splits) != {"support", "query_test", "continuous"}:
        raise BenchmarkV2Error(
            "manifest.protocol.crossmap_splits must contain support, "
            "query_test, and continuous"
        )
    support_seeds = protocol.get("support_seeds")
    if not isinstance(support_seeds, list) or not support_seeds:
        raise BenchmarkV2Error("manifest.protocol.support_seeds must be a non-empty list")
    checked_seeds = [_require_int(seed, "manifest.protocol.support_seeds item") for seed in support_seeds]
    if len(set(checked_seeds)) != len(checked_seeds):
        raise BenchmarkV2Error("manifest.protocol.support_seeds contains duplicates")

    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise BenchmarkV2Error("manifest.source must be an object")
    if not isinstance(source.get("root"), (str, os.PathLike)) or not source.get("root"):
        raise BenchmarkV2Error("manifest.source.root must be a non-empty path")
    radar_files = source.get("radar_files")
    if not isinstance(radar_files, Mapping):
        raise BenchmarkV2Error("manifest.source.radar_files must be an object")
    all_maps = [*seen_maps, *crossmap_maps]
    missing_radars = [name for name in all_maps if name not in radar_files]
    if missing_radars:
        raise BenchmarkV2Error(
            f"manifest.source.radar_files is missing maps: {missing_radars}"
        )

    return {
        "seen_maps": seen_maps,
        "crossmap_maps": crossmap_maps,
        "support_seeds": checked_seeds,
        "all_maps": all_maps,
        "benchmark_global_seed": benchmark["global_seed"],
    }


def _protocol_map_list(protocol: Mapping[str, Any], key: str) -> list[str]:
    values = _protocol_name_list(protocol, key)
    if not values:
        raise BenchmarkV2Error(f"manifest.protocol.{key} must not be empty")
    return values


def _protocol_name_list(protocol: Mapping[str, Any], key: str) -> list[str]:
    values = protocol.get(key)
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise BenchmarkV2Error(f"manifest.protocol.{key} must be a list of strings")
    if len(set(values)) != len(values):
        raise BenchmarkV2Error(f"manifest.protocol.{key} contains duplicates")
    return list(values)


def _normalise_map_names(value: Sequence[str] | str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = [value]
    else:
        if isinstance(value, (bytes, bytearray)):
            raise BenchmarkV2Error("map_names must contain map name strings")
        try:
            values = list(value)
        except TypeError as exc:
            raise BenchmarkV2Error("map_names must be a string or sequence") from exc
    if not values or any(not isinstance(name, str) or not name for name in values):
        raise BenchmarkV2Error("map_names must be a non-empty sequence of strings")
    if len(set(values)) != len(values):
        raise BenchmarkV2Error("map_names contains duplicate map names")
    return values


def _validate_requested_maps(
    requested_maps: Sequence[str], allowed_maps: Sequence[str], split: str
) -> None:
    allowed = set(allowed_maps)
    invalid = [name for name in requested_maps if name not in allowed]
    if invalid:
        raise BenchmarkV2Error(
            f"maps {invalid} are not valid for {split}; expected maps from "
            f"{list(allowed_maps)}"
        )


def _resolve_source_root(value: str | os.PathLike[str], manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        cwd_candidate = (Path.cwd() / path).resolve()
        bundle_candidate = (manifest_path.parent / path).resolve()
        resolved = cwd_candidate if cwd_candidate.is_dir() else bundle_candidate
    if not resolved.is_dir():
        raise FileNotFoundError(f"Benchmark v2 source root does not exist: {resolved}")
    return resolved


def _validate_z_ranges(
    manifest: Mapping[str, Any], all_maps: Sequence[str]
) -> dict[str, dict[str, Any]]:
    calibration = manifest.get("calibration")
    if not isinstance(calibration, Mapping):
        raise BenchmarkV2Error("manifest.calibration must be an object")
    ranges = calibration.get("z_ranges")
    if not isinstance(ranges, Mapping):
        raise BenchmarkV2Error("manifest.calibration.z_ranges must be an object")
    missing = [name for name in all_maps if name not in ranges]
    if missing:
        raise BenchmarkV2Error(
            f"manifest.calibration.z_ranges is missing maps: {missing}"
        )
    result: dict[str, dict[str, Any]] = {}
    for map_name in all_maps:
        value = ranges[map_name]
        if not isinstance(value, Mapping):
            raise BenchmarkV2Error(
                f"manifest.calibration.z_ranges.{map_name} must be an object"
            )
        z_min = value.get("z_min")
        z_max = value.get("z_max")
        _require_finite_number(z_min, f"{map_name}.z_min")
        _require_finite_number(z_max, f"{map_name}.z_max")
        if float(z_min) >= float(z_max):
            raise BenchmarkV2Error(
                f"invalid Z calibration range for {map_name}: {z_min!r}, {z_max!r}"
            )
        result[map_name] = dict(value)
    return result


def _resolve_radar_paths(
    radar_files: Mapping[str, Any], source_root: Path, map_names: Sequence[str]
) -> dict[str, Path]:
    source_root = source_root.resolve()
    result: dict[str, Path] = {}
    for map_name in map_names:
        raw_path = radar_files.get(map_name)
        if not isinstance(raw_path, (str, os.PathLike)) or not os.fspath(raw_path):
            raise BenchmarkV2Error(
                f"manifest.source.radar_files.{map_name} must be a path"
            )
        relative = Path(raw_path)
        if relative.is_absolute():
            raise BenchmarkV2Error(
                f"manifest.source.radar_files.{map_name} must be relative to source root"
            )
        path = (source_root / relative).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise BenchmarkV2Error(
                f"radar path escapes source root for {map_name}: {raw_path!r}"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(
                f"radar file for {map_name} does not exist: {path}"
            )
        result[map_name] = path
    return result


def _normalise_image_extension(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise BenchmarkV2Error("benchmark_v2_image_extension must be a string")
    extension = value.strip()
    if not extension:
        return None
    if not extension.startswith("."):
        extension = "." + extension
    if "/" in extension or "\\" in extension or extension in {".", ".."}:
        raise BenchmarkV2Error("benchmark_v2_image_extension must be a file extension")
    return extension


def _resolve_split_files(
    manifest: Mapping[str, Any],
    benchmark_root: Path,
    split: str,
    map_names: Sequence[str],
    support_seed: int,
) -> dict[str, Path]:
    setting = _SPLIT_SETTING[split]
    file_name = _SPLIT_FILE_NAME[split].format(support_seed=support_seed)
    result: dict[str, Path] = {}
    for map_name in map_names:
        path = benchmark_root / "splits" / setting / map_name / file_name
        # The builder emits per-map files.  Aggregate discrete files are a
        # useful compatibility fallback for early/self-contained manifests.
        if not path.is_file() and split != "seen_continuous" and split != "crossmap_continuous":
            aggregate_name = {
                "seen_train": "seen_train.json",
                "seen_validation": "seen_validation.json",
                "seen_discrete_test": "seen_discrete_test.json",
                "crossmap_support": f"crossmap_support_seed_{support_seed}.json",
                "crossmap_query_test": "crossmap_query_test.json",
            }[split]
            aggregate_path = benchmark_root / "aggregate" / aggregate_name
            if aggregate_path.is_file():
                path = aggregate_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Benchmark v2 split file does not exist for {map_name}/{split}: {path}"
            )
        result[map_name] = path.resolve()
    return result


def _load_discrete_selection(
    manifest: Mapping[str, Any],
    split: str,
    map_names: Sequence[str],
    split_files: Mapping[str, Path],
    z_ranges: Mapping[str, Mapping[str, Any]],
    support_seed: int,
    shots_per_map: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    setting = _SPLIT_SETTING[split]
    allowed_maps = set(manifest["protocol"][f"{setting}_maps"])
    rows_by_map: dict[str, list[dict[str, Any]]] = {name: [] for name in map_names}
    seen_by_map: dict[str, set[str]] = {name: set() for name in map_names}
    maps_by_path: dict[Path, list[str]] = {}
    for map_name in map_names:
        maps_by_path.setdefault(split_files[map_name], []).append(map_name)

    for path, path_maps in maps_by_path.items():
        loaded = _load_json(path, f"{split} split")
        if not isinstance(loaded, list):
            raise BenchmarkV2Error(f"discrete split file must contain a JSON list: {path}")
        is_aggregate = path.parent.name not in path_maps
        for raw_row in loaded:
            if not isinstance(raw_row, Mapping):
                raise BenchmarkV2Error(f"{split} row must be a JSON object: {path}")
            row_map = raw_row.get("map")
            if row_map not in allowed_maps:
                raise BenchmarkV2Error(
                    f"{split} row has map {row_map!r} outside its protocol setting"
                )
            # Aggregate files contain several maps; per-map files must contain
            # only the map selected by their path.
            if is_aggregate:
                if row_map not in path_maps:
                    continue
            elif row_map != path_maps[0]:
                raise BenchmarkV2Error(
                    f"{split} row map mismatch for {path_maps[0]}: {row_map!r}"
                )
            if row_map not in rows_by_map:
                continue
            row = _validate_row(raw_row, row_map, z_ranges[row_map])
            file_frame = row["file_frame"]
            if file_frame in seen_by_map[row_map]:
                raise BenchmarkV2Error(
                    f"duplicate row in {split}: {row_map}:{file_frame}"
                )
            seen_by_map[row_map].add(file_frame)
            rows_by_map[row_map].append(row)

    _validate_expected_discrete_counts(manifest, split, rows_by_map, support_seed)
    if split == "crossmap_support":
        for map_name in map_names:
            rows = rows_by_map[map_name]
            if len(rows) != 100:
                raise BenchmarkV2Error(
                    f"crossmap support for {map_name} must contain exactly 100 rows; "
                    f"got {len(rows)}"
                )
            rows_by_map[map_name] = sorted(
                rows,
                key=lambda row: (
                    _support_rank(
                        manifest["benchmark"]["global_seed"],
                        support_seed,
                        map_name,
                        row["file_frame"],
                    ),
                    row["file_frame"],
                ),
            )[:shots_per_map]

    rows = [row for map_name in map_names for row in rows_by_map[map_name]]
    return rows, {}


def _load_continuous_selection(
    manifest: Mapping[str, Any],
    split: str,
    map_names: Sequence[str],
    split_files: Mapping[str, Path],
    z_ranges: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    protocol = manifest.get("continuous_protocol")
    if not isinstance(protocol, Mapping):
        raise BenchmarkV2Error("manifest.continuous_protocol must be an object")
    expected_frames = protocol.get("frames_per_clip")
    if expected_frames is not None:
        expected_frames = _require_int(
            expected_frames, "manifest.continuous_protocol.frames_per_clip"
        )
        if expected_frames <= 0:
            raise BenchmarkV2Error("frames_per_clip must be positive")

    clips_by_map: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for map_name in map_names:
        payload = _load_json_object(split_files[map_name], f"{split} split")
        if payload.get("schema_version") != EXPECTED_SCHEMA_VERSION:
            raise BenchmarkV2Error(
                f"continuous split schema mismatch for {map_name}"
            )
        if payload.get("benchmark_id") != EXPECTED_BENCHMARK_ID:
            raise BenchmarkV2Error(
                f"continuous split benchmark_id mismatch for {map_name}"
            )
        if payload.get("map") != map_name or payload.get("split") != "continuous":
            raise BenchmarkV2Error(
                f"continuous split metadata mismatch for {map_name}"
            )
        if (
            expected_frames is not None
            and payload.get("frames_per_clip") != expected_frames
        ):
            raise BenchmarkV2Error(
                f"continuous frames_per_clip metadata mismatch for {map_name}"
            )
        clips = payload.get("clips")
        if not isinstance(clips, list):
            raise BenchmarkV2Error(f"continuous clips must be a list for {map_name}")
        _validate_expected_continuous_clip_count(manifest, split, map_name, clips)
        map_clips: list[dict[str, Any]] = []
        seen_file_frames: set[str] = set()
        seen_clip_ids: set[str] = set()
        for clip in clips:
            if not isinstance(clip, Mapping):
                raise BenchmarkV2Error(f"continuous clip must be an object for {map_name}")
            clip_id = clip.get("clip_id")
            if not isinstance(clip_id, str) or not clip_id:
                raise BenchmarkV2Error(f"continuous clip_id is invalid for {map_name}")
            if clip_id in seen_clip_ids:
                raise BenchmarkV2Error(
                    f"duplicate continuous clip_id for {map_name}: {clip_id}"
                )
            seen_clip_ids.add(clip_id)
            if clip.get("map", map_name) != map_name:
                raise BenchmarkV2Error(f"continuous clip map mismatch for {map_name}")
            frames = clip.get("frames")
            if not isinstance(frames, list):
                raise BenchmarkV2Error(
                    f"continuous clip frames must be a list for {map_name}:{clip_id}"
                )
            if expected_frames is not None and len(frames) != expected_frames:
                raise BenchmarkV2Error(
                    f"continuous clip length mismatch for {map_name}:{clip_id}; "
                    f"expected {expected_frames}, got {len(frames)}"
                )
            clip_rows: list[dict[str, Any]] = []
            for frame_index, raw_row in enumerate(frames):
                row = _validate_row(raw_row, map_name, z_ranges[map_name])
                file_frame = row["file_frame"]
                if file_frame in seen_file_frames:
                    raise BenchmarkV2Error(
                        f"duplicate continuous frame for {map_name}: {file_frame}"
                    )
                seen_file_frames.add(file_frame)
                row["_benchmark_clip_id"] = clip_id
                row["_benchmark_frame_index"] = frame_index
                clip_rows.append(row)
                rows.append(row)
            clip_copy = dict(clip)
            clip_copy["frames"] = clip_rows
            map_clips.append(clip_copy)
        clips_by_map[map_name] = map_clips
    return rows, clips_by_map


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkV2Error(f"could not read {description} {path}: {exc}") from exc


def _validate_row(
    raw_row: Mapping[str, Any],
    expected_map: str,
    z_range: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [field for field in _REQUIRED_ROW_FIELDS if field not in raw_row]
    if missing:
        raise BenchmarkV2Error(
            f"row for {expected_map} is missing required fields: {missing}"
        )
    if raw_row.get("map") != expected_map:
        raise BenchmarkV2Error(
            f"row map mismatch: expected {expected_map!r}, got {raw_row.get('map')!r}"
        )
    if not isinstance(raw_row.get("file_frame"), str) or not raw_row["file_frame"]:
        raise BenchmarkV2Error(f"row file_frame is invalid for {expected_map}")
    for field in ("x", "y", "z", "angle_h", "angle_v"):
        _require_finite_number(raw_row[field], f"{expected_map}.{field}")
    z = float(raw_row["z"])
    if z < float(z_range["z_min"]) or z > float(z_range["z_max"]):
        raise BenchmarkV2Error(
            f"row Z is outside calibration range for {expected_map}: "
            f"{raw_row['z']!r} not in [{z_range['z_min']!r}, {z_range['z_max']!r}]"
        )
    return dict(raw_row)


def _require_finite_number(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BenchmarkV2Error(f"{field_name} must be numeric, got {value!r}")
    if not math.isfinite(float(value)):
        raise BenchmarkV2Error(f"{field_name} must be finite, got {value!r}")


def _manifest_counts(manifest: Mapping[str, Any], map_name: str) -> Mapping[str, Any] | None:
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        return None
    if map_name in counts.get("seen", {}):
        value = counts["seen"][map_name]
    elif map_name in counts.get("crossmap", {}):
        value = counts["crossmap"][map_name]
    else:
        return None
    return value if isinstance(value, Mapping) else None


def _validate_expected_discrete_counts(
    manifest: Mapping[str, Any],
    split: str,
    rows_by_map: Mapping[str, Sequence[Mapping[str, Any]]],
    support_seed: int,
) -> None:
    key = {
        "seen_train": "train",
        "seen_validation": "validation",
        "seen_discrete_test": "discrete_test",
        "crossmap_support": "support",
        "crossmap_query_test": "query_test",
    }[split]
    for map_name, rows in rows_by_map.items():
        count_spec = _manifest_counts(manifest, map_name)
        if count_spec is None or key not in count_spec:
            continue
        expected = count_spec[key]
        if key == "support" and isinstance(expected, Mapping):
            expected = expected.get(str(support_seed), expected.get(support_seed))
        if expected is not None and len(rows) != expected:
            raise BenchmarkV2Error(
                f"{split} count mismatch for {map_name}: expected {expected}, got {len(rows)}"
            )


def _validate_expected_continuous_clip_count(
    manifest: Mapping[str, Any],
    split: str,
    map_name: str,
    clips: Sequence[Any],
) -> None:
    count_spec = _manifest_counts(manifest, map_name)
    if count_spec is None or "continuous_clips" not in count_spec:
        return
    expected = count_spec["continuous_clips"]
    if len(clips) != expected:
        raise BenchmarkV2Error(
            f"{split} clip count mismatch for {map_name}: expected {expected}, got {len(clips)}"
        )


def _support_rank(
    global_seed: int, support_seed: int, map_name: str, file_frame: str
) -> str:
    value = f"{global_seed}|{support_seed}|{map_name}|{file_frame}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "BenchmarkV2Error",
    "BenchmarkV2Selection",
    "SUPPORTED_SPLITS",
    "is_benchmark_v2_config",
    "load_benchmark_v2_selection",
    "load_benchmark_v2_selection_from_args",
]
