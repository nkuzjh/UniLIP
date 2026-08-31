#!/usr/bin/env python3
"""Build and validate the deterministic CS2 benchmark-v2 splits.

The command intentionally has no dependency on the rest of the training
package.  It reads the source rows as opaque JSON objects, adds no fields to
discrete manifests, and keeps all generated paths below ``paths.output_root``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - exercised by environments without PyYAML
    yaml = None


EXPECTED_SCHEMA_VERSION = 1
EXPECTED_BENCHMARK_ID = "csgo_benchmark_v2"
REQUIRED_COORDINATES = ("x", "y", "z", "angle_h", "angle_v")
CALIBRATION_JSON = "calibration/z_calibration.json"
CALIBRATION_EXTREMA_JSONL = "calibration/z_extrema_rows.jsonl"
CALIBRATION_APPROVAL_TEMPLATE = "calibration/z_calibration_approval.template.yaml"
CALIBRATION_ARTIFACTS = (CALIBRATION_JSON, CALIBRATION_EXTREMA_JSONL)
COORDINATE_CANDIDATE_CSV = "audit/coordinate_candidates.csv"
COORDINATE_CANDIDATE_CSV_COLUMNS = (
    "map",
    "record_id",
    "file_num",
    "frame_id",
    "frame",
    "file_frame",
    "reasons",
    "z",
    "source_image_path",
)
SEEN_POOL_ORDER = ("train", "validation", "discrete_test", "continuous")
CROSSMAP_POOL_ORDER = ("support", "query_test", "continuous")
DISCRETE_SPLITS = ("train", "validation", "discrete_test")
PRODUCTION_SEEN_MAPS = (
    "cs_agency",
    "cs_italy",
    "de_ancient",
    "de_anubis",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
    "de_overpass",
    "de_train",
)
PRODUCTION_CROSSMAP_MAPS = ("cs_office", "de_golden", "de_palacio", "de_vertigo")
PRODUCTION_SEEN_RATIOS = {
    "train": 0.60,
    "validation": 0.10,
    "discrete_test": 0.20,
    "continuous": 0.10,
}
PRODUCTION_CROSSMAP_RATIOS = {
    "support": 0.20,
    "query_test": 0.60,
    "continuous": 0.20,
}
PRODUCTION_SPATIAL_BINS = {"x": 4, "y": 4, "z": 3, "yaw": 8, "pitch": 3}
PRODUCTION_NEAR_POSE = {
    "enabled": True,
    "xy_tolerance": 8.0,
    "z_tolerance": 8.0,
    "angle_tolerance_degrees": 5.0,
}


class BenchmarkError(RuntimeError):
    """A user-actionable configuration, audit, build, or validation error."""


def require_yaml() -> Any:
    if yaml is None:
        raise BenchmarkError(
            "PyYAML is required for this builder. Install it with 'pip install PyYAML'."
        )
    return yaml


def canonical_json_bytes(value: Any) -> bytes:
    """Return the stable JSON representation used by all fingerprints."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any, indent: int) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _path_from_cwd(value: str, cwd: Path | None = None) -> Path:
    base = cwd or Path.cwd()
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkError(f"{name} must be a mapping")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise BenchmarkError(f"{name} must be a list")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkError(f"{name} must be a non-empty string")
    return value


def _require_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    result = _require_nonnegative_int(value, name)
    if result == 0:
        raise BenchmarkError(f"{name} must be positive")
    return result


def _require_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive " if positive else "finite "
        raise BenchmarkError(f"{name} must be a {qualifier}number")
    return result


def _validate_ratio_mapping(
    value: Any, name: str, required: Sequence[str]
) -> dict[str, float]:
    mapping = _require_mapping(value, name)
    result: dict[str, float] = {}
    for key in required:
        if key not in mapping:
            raise BenchmarkError(f"{name}.{key} is required")
        result[key] = _require_number(mapping[key], f"{name}.{key}")
        if result[key] < 0:
            raise BenchmarkError(f"{name}.{key} must be non-negative")
    extra = sorted(set(mapping) - set(required))
    if extra:
        raise BenchmarkError(f"{name} has unknown keys: {', '.join(extra)}")
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise BenchmarkError(f"{name} ratios must sum to 1.0")
    return result


def _validate_bounds(value: Any, name: str) -> tuple[float, float]:
    bounds = _require_list(value, name)
    if len(bounds) != 2:
        raise BenchmarkError(f"{name} must contain exactly two numbers")
    low = _require_number(bounds[0], f"{name}[0]")
    high = _require_number(bounds[1], f"{name}[1]")
    if high < low:
        raise BenchmarkError(f"{name} upper bound must not be below lower bound")
    return low, high


def _validate_strict_protocol(config: Mapping[str, Any]) -> None:
    benchmark = config["benchmark"]
    if not benchmark.get("strict_protocol", False):
        return
    if benchmark.get("version") != "2.0.0":
        raise BenchmarkError("strict_protocol requires benchmark.version=2.0.0")
    if benchmark.get("global_seed") != 20260827:
        raise BenchmarkError("strict_protocol requires benchmark.global_seed=20260827")
    maps = config["maps"]
    if tuple(maps["seen"]) != PRODUCTION_SEEN_MAPS:
        raise BenchmarkError(
            "strict_protocol requires the documented ordered Seen-10 map list"
        )
    if tuple(maps["crossmap"]) != PRODUCTION_CROSSMAP_MAPS:
        raise BenchmarkError(
            "strict_protocol requires the documented ordered CrossMap-4 map list"
        )
    counts = config["counts"]
    if {key: counts["seen"][key] for key in DISCRETE_SPLITS} != {
        "train": 5000,
        "validation": 500,
        "discrete_test": 2000,
    }:
        raise BenchmarkError("strict_protocol requires Seen counts 5000/500/2000")
    if counts["crossmap"]["support"] != 100 or counts["crossmap"]["query_test"] != 2000:
        raise BenchmarkError(
            "strict_protocol requires CrossMap support/query counts 100/2000"
        )
    if list(counts["crossmap"]["support_seeds"]) != [0, 1, 2, 3, 4]:
        raise BenchmarkError("strict_protocol requires support_seeds [0, 1, 2, 3, 4]")
    continuous = counts["continuous"]
    if {
        "clips_per_map": continuous["clips_per_map"],
        "frames_per_clip": continuous["frames_per_clip"],
        "max_frame_gap": continuous["max_frame_gap"],
        "max_clips_per_record": continuous["max_clips_per_record"],
    } != {
        "clips_per_map": 20,
        "frames_per_clip": 64,
        "max_frame_gap": 2,
        "max_clips_per_record": 1,
    }:
        raise BenchmarkError(
            "strict_protocol requires continuous settings 20x64/gap2/one clip per record"
        )
    for section_name, expected in (
        ("seen", PRODUCTION_SEEN_RATIOS),
        ("crossmap", PRODUCTION_CROSSMAP_RATIOS),
    ):
        actual = config["record_pools"][section_name]
        if set(actual) != set(expected) or any(
            not math.isclose(
                float(actual[key]), expected[key], rel_tol=0.0, abs_tol=1e-12
            )
            for key in expected
        ):
            raise BenchmarkError(
                f"strict_protocol requires documented {section_name} record pool ratios"
            )
    if config["sampling"]["min_frame_gap"] != 5:
        raise BenchmarkError("strict_protocol requires sampling.min_frame_gap=5")
    if config["sampling"]["spatial_bins"] != PRODUCTION_SPATIAL_BINS:
        raise BenchmarkError("strict_protocol requires the documented spatial bins")
    near_pose = config["sampling"]["near_pose_filter"]
    if set(near_pose) != set(PRODUCTION_NEAR_POSE) or any(
        near_pose[key] != expected for key, expected in PRODUCTION_NEAR_POSE.items()
    ):
        raise BenchmarkError("strict_protocol requires the documented near-pose filter")
    z_calibration = config["calibration"]["z"]
    if z_calibration != {
        "source": "approved_full_corpus",
        "method": "exact_min_max",
        "timing": "before_split",
        "normalize_to": [0.0, 1.0],
        "clamp": False,
        "fallback": "error",
    }:
        raise BenchmarkError(
            "strict_protocol requires frozen exact full-corpus Z calibration"
        )
    if config["audit"].get("z_mad_candidate_enabled", False):
        raise BenchmarkError("strict_protocol requires z MAD to remain diagnostic-only")


def validate_config(config: Any) -> dict[str, Any]:
    """Validate the public YAML contract and return it as a mutable dict."""

    root = _require_mapping(config, "config")
    if root.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise BenchmarkError(f"schema_version must be {EXPECTED_SCHEMA_VERSION}")

    benchmark = _require_mapping(root.get("benchmark"), "benchmark")
    if benchmark.get("id") != EXPECTED_BENCHMARK_ID:
        raise BenchmarkError(f"benchmark.id must be {EXPECTED_BENCHMARK_ID}")
    _require_string(benchmark.get("version"), "benchmark.version")
    _require_nonnegative_int(benchmark.get("global_seed"), "benchmark.global_seed")
    if not isinstance(benchmark.get("strict_protocol", False), bool):
        raise BenchmarkError("benchmark.strict_protocol must be boolean")

    paths = _require_mapping(root.get("paths"), "paths")
    _require_string(paths.get("source_root"), "paths.source_root")
    _require_string(paths.get("output_root"), "paths.output_root")

    calibration = _require_mapping(root.get("calibration"), "calibration")
    z_calibration = _require_mapping(calibration.get("z"), "calibration.z")
    if z_calibration.get("source") != "approved_full_corpus":
        raise BenchmarkError("calibration.z.source must be approved_full_corpus")
    if z_calibration.get("method") != "exact_min_max":
        raise BenchmarkError("calibration.z.method must be exact_min_max")
    if z_calibration.get("timing") != "before_split":
        raise BenchmarkError("calibration.z.timing must be before_split")
    normalize_to = _require_list(
        z_calibration.get("normalize_to"), "calibration.z.normalize_to"
    )
    if (
        len(normalize_to) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in normalize_to
        )
        or [float(value) for value in normalize_to] != [0.0, 1.0]
    ):
        raise BenchmarkError("calibration.z.normalize_to must be [0.0, 1.0]")
    if z_calibration.get("clamp") is not False:
        raise BenchmarkError("calibration.z.clamp must be false")
    if z_calibration.get("fallback") != "error":
        raise BenchmarkError("calibration.z.fallback must be error")

    source = _require_mapping(root.get("source"), "source")
    _require_string(source.get("positions_file"), "source.positions_file")
    _require_string(source.get("images_dir"), "source.images_dir")
    _require_string(source.get("image_extension"), "source.image_extension")
    radar_files = _require_mapping(source.get("radar_files"), "source.radar_files")
    for map_name, radar_file in radar_files.items():
        radar_file = _require_string(radar_file, f"source.radar_files.{map_name}")
        radar_path = Path(radar_file)
        if radar_path.is_absolute() or ".." in radar_path.parts:
            raise BenchmarkError(
                f"source.radar_files.{map_name} must be relative to source_root"
            )
    regex_text = _require_string(source.get("record_regex"), "source.record_regex")
    try:
        record_regex = re.compile(regex_text)
    except re.error as exc:
        raise BenchmarkError(f"source.record_regex is invalid: {exc}") from exc
    if (
        "record" not in record_regex.groupindex
        or "frame" not in record_regex.groupindex
    ):
        raise BenchmarkError(
            "source.record_regex must define named groups 'record' and 'frame'"
        )
    forbidden = source.get("forbidden_map_dirs", [])
    _require_list(forbidden, "source.forbidden_map_dirs")
    if any(not isinstance(item, str) or not item for item in forbidden):
        raise BenchmarkError("source.forbidden_map_dirs must contain non-empty strings")
    if len(set(forbidden)) != len(forbidden):
        raise BenchmarkError("source.forbidden_map_dirs must be unique")

    maps = _require_mapping(root.get("maps"), "maps")
    seen = _require_list(maps.get("seen"), "maps.seen")
    crossmap = _require_list(maps.get("crossmap"), "maps.crossmap")
    for name, values in (("maps.seen", seen), ("maps.crossmap", crossmap)):
        if not values or any(not isinstance(item, str) or not item for item in values):
            raise BenchmarkError(f"{name} must contain non-empty strings")
        if len(set(values)) != len(values):
            raise BenchmarkError(f"{name} must be unique")
    overlap = sorted(set(seen) & set(crossmap))
    if overlap:
        raise BenchmarkError(
            f"maps.seen and maps.crossmap overlap: {', '.join(overlap)}"
        )
    selected_maps = set(seen) | set(crossmap)
    missing_radar_files = sorted(selected_maps - set(radar_files))
    extra_radar_files = sorted(set(radar_files) - selected_maps)
    if missing_radar_files:
        raise BenchmarkError(
            "source.radar_files is missing selected maps: "
            + ", ".join(missing_radar_files)
        )
    if extra_radar_files:
        raise BenchmarkError(
            "source.radar_files contains unselected maps: "
            + ", ".join(extra_radar_files)
        )

    counts = _require_mapping(root.get("counts"), "counts")
    seen_counts = _require_mapping(counts.get("seen"), "counts.seen")
    for key in DISCRETE_SPLITS:
        _require_nonnegative_int(seen_counts.get(key), f"counts.seen.{key}")
    cross_counts = _require_mapping(counts.get("crossmap"), "counts.crossmap")
    _require_nonnegative_int(cross_counts.get("support"), "counts.crossmap.support")
    _require_nonnegative_int(
        cross_counts.get("query_test"), "counts.crossmap.query_test"
    )
    seeds = _require_list(
        cross_counts.get("support_seeds"), "counts.crossmap.support_seeds"
    )
    if not seeds or any(
        isinstance(item, bool) or not isinstance(item, int) for item in seeds
    ):
        raise BenchmarkError("counts.crossmap.support_seeds must contain integers")
    if len(set(seeds)) != len(seeds):
        raise BenchmarkError("counts.crossmap.support_seeds must be unique")
    continuous_counts = _require_mapping(counts.get("continuous"), "counts.continuous")
    _require_nonnegative_int(
        continuous_counts.get("clips_per_map"), "counts.continuous.clips_per_map"
    )
    _require_positive_int(
        continuous_counts.get("frames_per_clip"), "counts.continuous.frames_per_clip"
    )
    _require_positive_int(
        continuous_counts.get("max_frame_gap"), "counts.continuous.max_frame_gap"
    )
    _require_positive_int(
        continuous_counts.get("max_clips_per_record"),
        "counts.continuous.max_clips_per_record",
    )

    record_pools = _require_mapping(root.get("record_pools"), "record_pools")
    _validate_ratio_mapping(
        record_pools.get("seen"), "record_pools.seen", SEEN_POOL_ORDER
    )
    _validate_ratio_mapping(
        record_pools.get("crossmap"), "record_pools.crossmap", CROSSMAP_POOL_ORDER
    )

    sampling = _require_mapping(root.get("sampling"), "sampling")
    _require_nonnegative_int(sampling.get("min_frame_gap"), "sampling.min_frame_gap")
    spatial_bins = _require_mapping(
        sampling.get("spatial_bins"), "sampling.spatial_bins"
    )
    for key in ("x", "y", "z", "yaw", "pitch"):
        _require_positive_int(spatial_bins.get(key), f"sampling.spatial_bins.{key}")
    near_pose = _require_mapping(
        sampling.get("near_pose_filter"), "sampling.near_pose_filter"
    )
    if not isinstance(near_pose.get("enabled"), bool):
        raise BenchmarkError("sampling.near_pose_filter.enabled must be boolean")
    for key in ("xy_tolerance", "z_tolerance", "angle_tolerance_degrees"):
        _require_number(near_pose.get(key), f"sampling.near_pose_filter.{key}")
        if float(near_pose[key]) < 0:
            raise BenchmarkError(
                f"sampling.near_pose_filter.{key} must be non-negative"
            )

    audit = _require_mapping(root.get("audit"), "audit")
    coordinate_bounds = _require_mapping(
        audit.get("coordinate_bounds"), "audit.coordinate_bounds"
    )
    _validate_bounds(coordinate_bounds.get("x"), "audit.coordinate_bounds.x")
    _validate_bounds(coordinate_bounds.get("y"), "audit.coordinate_bounds.y")
    _require_number(
        audit.get("robust_z_mad_threshold"),
        "audit.robust_z_mad_threshold",
        positive=True,
    )
    if not isinstance(audit.get("z_mad_candidate_enabled", False), bool):
        raise BenchmarkError("audit.z_mad_candidate_enabled must be boolean")
    jump_thresholds = _require_mapping(
        audit.get("jump_thresholds"), "audit.jump_thresholds"
    )
    for key in ("xy", "z", "angle_degrees"):
        _require_number(
            jump_thresholds.get(key), f"audit.jump_thresholds.{key}", positive=True
        )
    if not isinstance(audit.get("require_images"), bool):
        raise BenchmarkError("audit.require_images must be boolean")
    _require_nonnegative_int(
        audit.get("candidate_preview_limit"), "audit.candidate_preview_limit"
    )
    angle_bounds = _require_mapping(audit.get("angle_bounds"), "audit.angle_bounds")
    for key in ("angle_h", "angle_v"):
        if key not in angle_bounds:
            raise BenchmarkError(f"audit.angle_bounds.{key} is required")
        _validate_bounds(angle_bounds[key], f"audit.angle_bounds.{key}")
    extra_angle_bounds = sorted(set(angle_bounds) - {"angle_h", "angle_v"})
    if extra_angle_bounds:
        raise BenchmarkError(
            "audit.angle_bounds has unknown keys: " + ", ".join(extra_angle_bounds)
        )

    output = _require_mapping(root.get("output"), "output")
    indent = output.get("indent")
    if isinstance(indent, bool) or not isinstance(indent, int) or indent < 0:
        raise BenchmarkError("output.indent must be a non-negative integer")

    validated = copy.deepcopy(dict(root))
    _validate_strict_protocol(validated)
    return validated


def load_config(config_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate a YAML config.

    Relative paths in the YAML are resolved from the current working
    directory, which is normally the repository root.  The config file itself
    is also located relative to the current working directory.
    """

    require_yaml()
    path = _path_from_cwd(str(config_path))
    if not path.is_file():
        raise BenchmarkError(f"config file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise BenchmarkError(f"could not parse config YAML {path}: {exc}") from exc
    return validate_config(loaded)


@dataclass(frozen=True)
class ConfigContext:
    config: dict[str, Any]
    config_path: Path
    source_root: Path
    output_root: Path

    @property
    def indent(self) -> int:
        return int(self.config["output"]["indent"])

    @property
    def benchmark(self) -> Mapping[str, Any]:
        return self.config["benchmark"]


def make_context(config_path: str | os.PathLike[str]) -> ConfigContext:
    path = _path_from_cwd(str(config_path))
    config = load_config(path)
    return ConfigContext(
        config=config,
        config_path=path,
        source_root=_path_from_cwd(config["paths"]["source_root"]),
        output_root=_path_from_cwd(config["paths"]["output_root"]),
    )


def parse_file_frame(
    file_frame: Any, record_regex: str | re.Pattern[str]
) -> tuple[str, int] | None:
    """Parse an exact ``file_frame`` value into record id and integer frame."""

    if not isinstance(file_frame, str):
        return None
    pattern = (
        re.compile(record_regex) if isinstance(record_regex, str) else record_regex
    )
    match = pattern.fullmatch(file_frame)
    if match is None:
        return None
    try:
        record = match.group("record")
        frame = int(match.group("frame"))
    except (IndexError, TypeError, ValueError):
        return None
    return record, frame


def _record_sort_key(record_id: str) -> tuple[int, Any, str]:
    if record_id.isdigit():
        return (0, int(record_id), record_id)
    return (1, record_id, record_id)


def stable_rank(global_seed: int, *parts: Any) -> int:
    """Return a reproducible SHA-256-derived rank; never uses Python hash()."""

    payload = canonical_json_bytes([int(global_seed), *parts])
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _stable_order(items: Iterable[Any], seed: int, *context: Any) -> list[Any]:
    return sorted(
        items,
        key=lambda item: (
            stable_rank(seed, *context, item),
            str(item),
        ),
    )


def largest_remainder_counts(
    total: int, ratios: Mapping[str, float], order: Sequence[str] | None = None
) -> dict[str, int]:
    """Allocate whole records using the largest-remainder method."""

    if total < 0:
        raise ValueError("total must be non-negative")
    names = list(order or ratios.keys())
    if set(names) != set(ratios):
        raise ValueError("order and ratios must contain the same names")
    raw = {name: total * float(ratios[name]) for name in names}
    result = {name: math.floor(raw[name]) for name in names}
    remaining = total - sum(result.values())
    ranked = sorted(
        names,
        key=lambda name: (-(raw[name] - result[name]), names.index(name)),
    )
    for name in ranked[:remaining]:
        result[name] += 1
    return result


@dataclass
class SourceRow:
    map_name: str
    source_index: int
    raw: Any
    file_frame: str | None
    record_id: str | None
    frame_id: int | None
    coords: dict[str, float] = field(default_factory=dict)
    integrity_reasons: list[str] = field(default_factory=list)
    coordinate_reasons: list[str] = field(default_factory=list)
    neighbor_deltas: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str | None]:
        return self.map_name, self.file_frame

    @property
    def integrity_invalid(self) -> bool:
        return bool(self.integrity_reasons)

    @property
    def coordinate_candidate(self) -> bool:
        return bool(self.coordinate_reasons)

    @property
    def candidate(self) -> bool:
        return self.integrity_invalid or self.coordinate_candidate


@dataclass
class SourceSnapshot:
    rows_by_map: dict[str, list[SourceRow]]
    positions_sha256: dict[str, str]
    radar_sha256: dict[str, str | None]
    source_sha256: str
    radar_exists: dict[str, bool]


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _angle_bounds(config: Mapping[str, Any], field_name: str) -> tuple[float, float]:
    configured = config["audit"]["angle_bounds"][field_name]
    return float(configured[0]), float(configured[1])


def _circular_delta(
    first: float, second: float, period: float = 2.0 * math.pi
) -> float:
    return abs((first - second + period / 2.0) % period - period / 2.0)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _source_image_relative_path(
    config: Mapping[str, Any], map_name: str, file_frame: Any
) -> str:
    images_dir = str(config["source"]["images_dir"]).replace(os.sep, "/").strip("/")
    extension = str(config["source"]["image_extension"])
    name = "" if file_frame is None else str(file_frame)
    return "/".join(part for part in (map_name, images_dir, name + extension) if part)


def _load_map_rows(
    ctx: ConfigContext, map_name: str
) -> tuple[list[SourceRow], str, bool]:
    config = ctx.config
    map_root = ctx.source_root / map_name
    positions_path = map_root / str(config["source"]["positions_file"])
    if not positions_path.is_file():
        raise BenchmarkError(
            f"positions file does not exist for {map_name}: {positions_path}"
        )
    positions_bytes = positions_path.read_bytes()
    try:
        positions = json.loads(positions_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(
            f"invalid positions JSON for {map_name}: {positions_path}: {exc}"
        ) from exc
    if not isinstance(positions, list):
        raise BenchmarkError(f"positions JSON for {map_name} must be a list")

    record_regex = re.compile(str(config["source"]["record_regex"]))
    image_dir = map_root / str(config["source"]["images_dir"])
    radar_path = ctx.source_root / str(config["source"]["radar_files"][map_name])
    radar_exists = radar_path.is_file()
    rows: list[SourceRow] = []
    for index, raw in enumerate(positions):
        file_frame = raw.get("file_frame") if isinstance(raw, Mapping) else None
        parsed = parse_file_frame(file_frame, record_regex)
        row = SourceRow(
            map_name=map_name,
            source_index=index,
            raw=raw,
            file_frame=file_frame if isinstance(file_frame, str) else None,
            record_id=parsed[0] if parsed else None,
            frame_id=parsed[1] if parsed else None,
        )
        if not isinstance(raw, Mapping):
            _add_reason(row.integrity_reasons, "row_not_object")
        else:
            if raw.get("map") != map_name:
                _add_reason(row.integrity_reasons, "map_directory_mismatch")
            if not isinstance(file_frame, str):
                _add_reason(row.integrity_reasons, "file_frame_not_string")
            elif parsed is None:
                _add_reason(row.integrity_reasons, "file_frame_regex_mismatch")
            for field_name in REQUIRED_COORDINATES:
                value = raw.get(field_name)
                if not _is_finite_number(value):
                    _add_reason(row.integrity_reasons, f"non_finite_{field_name}")
                else:
                    row.coords[field_name] = float(value)
            if config["audit"].get("require_images", False) and parsed is not None:
                image_path = image_dir / (
                    file_frame + str(config["source"]["image_extension"])
                )
                if not image_path.is_file():
                    _add_reason(row.integrity_reasons, "missing_image")
        if not radar_exists:
            _add_reason(row.integrity_reasons, "missing_radar")
        rows.append(row)

    by_file_frame: dict[str, list[SourceRow]] = defaultdict(list)
    for row in rows:
        if row.file_frame is not None:
            by_file_frame[row.file_frame].append(row)
    for duplicate_rows in by_file_frame.values():
        if len(duplicate_rows) > 1:
            for row in duplicate_rows:
                _add_reason(row.integrity_reasons, "duplicate_file_frame")

    if config["audit"].get("z_mad_candidate_enabled", False):
        valid_z = [row.coords["z"] for row in rows if "z" in row.coords]
        median_z = _median(valid_z)
        if median_z is not None:
            deviations = [abs(value - median_z) for value in valid_z]
            mad = _median(deviations)
            assert mad is not None
            threshold = float(config["audit"]["robust_z_mad_threshold"])
            for row in rows:
                if "z" not in row.coords:
                    continue
                deviation = abs(row.coords["z"] - median_z)
                if mad > 0:
                    robust_score = 0.67448975 * deviation / mad
                    is_outlier = robust_score > threshold
                else:
                    robust_score = 0.0 if deviation == 0 else math.inf
                    is_outlier = deviation > 0
                if is_outlier:
                    _add_reason(row.coordinate_reasons, "z_robust_outlier")

    coordinate_bounds = config["audit"]["coordinate_bounds"]
    x_bounds = tuple(float(item) for item in coordinate_bounds["x"])
    y_bounds = tuple(float(item) for item in coordinate_bounds["y"])
    for row in rows:
        if "x" in row.coords and not x_bounds[0] <= row.coords["x"] <= x_bounds[1]:
            _add_reason(row.coordinate_reasons, "x_out_of_bounds")
        if "y" in row.coords and not y_bounds[0] <= row.coords["y"] <= y_bounds[1]:
            _add_reason(row.coordinate_reasons, "y_out_of_bounds")
        for field_name in ("angle_h", "angle_v"):
            if field_name in row.coords:
                lower, upper = _angle_bounds(config, field_name)
                in_range = (
                    lower <= row.coords[field_name] < upper
                    if field_name == "angle_h"
                    else lower <= row.coords[field_name] <= upper
                )
                if not in_range:
                    _add_reason(row.coordinate_reasons, f"{field_name}_out_of_range")

    jump_thresholds = config["audit"]["jump_thresholds"]
    by_record: dict[str, list[SourceRow]] = defaultdict(list)
    for row in rows:
        if (
            row.record_id is not None
            and row.frame_id is not None
            and len(row.coords) == 5
        ):
            by_record[row.record_id].append(row)
    max_local_gap = int(config["counts"]["continuous"]["max_frame_gap"])
    for record_id, record_rows in by_record.items():
        record_rows.sort(key=lambda item: (item.frame_id, item.source_index))
        for previous, current in zip(record_rows, record_rows[1:]):
            frame_delta = current.frame_id - previous.frame_id
            if not 1 <= frame_delta <= max_local_gap:
                continue
            xy_delta = math.hypot(
                current.coords["x"] - previous.coords["x"],
                current.coords["y"] - previous.coords["y"],
            )
            z_delta = abs(current.coords["z"] - previous.coords["z"])
            angle_h_delta = _circular_delta(
                current.coords["angle_h"], previous.coords["angle_h"]
            )
            angle_v_delta = abs(current.coords["angle_v"] - previous.coords["angle_v"])
            delta = {
                "frame_delta": frame_delta,
                "xy": xy_delta,
                "z": z_delta,
                "angle_h": angle_h_delta,
                "angle_v": angle_v_delta,
            }
            previous.neighbor_deltas["next"] = delta
            current.neighbor_deltas["previous"] = delta
            if xy_delta > float(jump_thresholds["xy"]):
                _add_reason(previous.coordinate_reasons, "trajectory_jump_xy")
                _add_reason(current.coordinate_reasons, "trajectory_jump_xy")
            if z_delta > float(jump_thresholds["z"]):
                _add_reason(previous.coordinate_reasons, "trajectory_jump_z")
                _add_reason(current.coordinate_reasons, "trajectory_jump_z")
            if max(angle_h_delta, angle_v_delta) > math.radians(
                float(jump_thresholds["angle_degrees"])
            ):
                _add_reason(previous.coordinate_reasons, "trajectory_jump_angle")
                _add_reason(current.coordinate_reasons, "trajectory_jump_angle")

    return rows, sha256_bytes(positions_bytes), radar_exists


def _source_snapshot(ctx: ConfigContext) -> SourceSnapshot:
    selected_maps = list(ctx.config["maps"]["seen"]) + list(
        ctx.config["maps"]["crossmap"]
    )
    source_root = ctx.source_root
    if not source_root.is_dir():
        raise BenchmarkError(f"source_root does not exist: {source_root}")
    forbidden = list(ctx.config["source"].get("forbidden_map_dirs", []))
    selected = set(selected_maps)
    selected_forbidden = sorted(selected & set(forbidden))
    if selected_forbidden:
        raise BenchmarkError(
            "configured forbidden map directories are selected: "
            + ", ".join(selected_forbidden)
        )
    for map_name in selected_maps:
        map_root = source_root / map_name
        if not map_root.is_dir():
            raise BenchmarkError(
                f"selected source map directory does not exist: {map_root}"
            )

    rows_by_map: dict[str, list[SourceRow]] = {}
    positions_sha256: dict[str, str] = {}
    radar_exists: dict[str, bool] = {}
    radar_sha256: dict[str, str | None] = {}
    for map_name in selected_maps:
        rows, positions_hash, has_radar = _load_map_rows(ctx, map_name)
        rows_by_map[map_name] = rows
        positions_sha256[map_name] = positions_hash
        radar_exists[map_name] = has_radar
        radar_path = ctx.source_root / str(
            ctx.config["source"]["radar_files"][map_name]
        )
        radar_sha256[map_name] = sha256_file(radar_path) if has_radar else None
    source_sha256 = sha256_bytes(
        canonical_json_bytes(
            {"positions_sha256": positions_sha256, "radar_sha256": radar_sha256}
        )
    )
    return SourceSnapshot(
        rows_by_map, positions_sha256, radar_sha256, source_sha256, radar_exists
    )


def _selected_image_manifest_bytes(
    ctx: ConfigContext, rows: Iterable[SourceRow]
) -> tuple[bytes, int]:
    """Hash exactly the unique images referenced by selected output rows."""

    image_paths: set[str] = set()
    for row in rows:
        if row.file_frame is None:
            raise BenchmarkError(
                f"selected row has no file_frame: {row.map_name}:{row.source_index}"
            )
        relative = _source_image_relative_path(ctx.config, row.map_name, row.file_frame)
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise BenchmarkError(
                f"selected image path is not source-root-relative: {relative}"
            )
        image_paths.add(relative)

    entries: list[tuple[str, str]] = []
    for relative in sorted(image_paths):
        image_path = ctx.source_root / Path(relative)
        if not image_path.is_file():
            raise BenchmarkError(f"selected image does not exist: {image_path}")
        entries.append((sha256_file(image_path), relative))
    data = "".join(f"{digest}  {relative}\n" for digest, relative in entries).encode(
        "utf-8"
    )
    return data, len(entries)


def _candidate_line(row: SourceRow, config: Mapping[str, Any]) -> dict[str, Any]:
    coordinates = {
        name: row.coords[name] for name in REQUIRED_COORDINATES if name in row.coords
    }
    categories: list[str] = []
    if row.coordinate_candidate:
        categories.append("coordinate")
    if row.integrity_invalid:
        categories.append("integrity")
    return {
        "map": row.map_name,
        "source_index": row.source_index,
        "file_frame": row.file_frame,
        "record_id": row.record_id,
        "frame_id": row.frame_id,
        "category": "coordinate" if row.coordinate_candidate else "integrity",
        "categories": categories,
        "reasons": list(row.integrity_reasons) + list(row.coordinate_reasons),
        "integrity_reasons": list(row.integrity_reasons),
        "coordinate_reasons": list(row.coordinate_reasons),
        "coordinates": coordinates,
        "source_image_relative_path": _source_image_relative_path(
            config, row.map_name, row.file_frame
        ),
        "neighbor_deltas": row.neighbor_deltas,
    }


def _candidate_bytes(snapshot: SourceSnapshot, config: Mapping[str, Any]) -> bytes:
    lines: list[bytes] = []
    for map_name in list(config["maps"]["seen"]) + list(config["maps"]["crossmap"]):
        for row in snapshot.rows_by_map[map_name]:
            if row.candidate:
                lines.append(canonical_json_bytes(_candidate_line(row, config)) + b"\n")
    return b"".join(lines)


def _candidate_csv_bytes(snapshot: SourceSnapshot, config: Mapping[str, Any]) -> bytes:
    """Return a review-friendly, deterministic CSV of coordinate candidates."""

    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(COORDINATE_CANDIDATE_CSV_COLUMNS)
    for map_name in list(config["maps"]["seen"]) + list(config["maps"]["crossmap"]):
        for row in snapshot.rows_by_map[map_name]:
            if not row.coordinate_candidate:
                continue
            record_id = "" if row.record_id is None else row.record_id
            frame_id = "" if row.frame_id is None else str(row.frame_id)
            writer.writerow(
                [
                    row.map_name,
                    record_id,
                    f"file_num{record_id}" if record_id else "",
                    frame_id,
                    f"frame_{frame_id}" if frame_id else "",
                    row.file_frame or "",
                    "|".join(row.integrity_reasons + row.coordinate_reasons),
                    "" if "z" not in row.coords else str(row.coords["z"]),
                    _source_image_relative_path(config, row.map_name, row.file_frame),
                ]
            )
    return stream.getvalue().encode("utf-8")


def _coordinate_stats(
    rows: Sequence[SourceRow], config: Mapping[str, Any]
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for field_name in REQUIRED_COORDINATES:
        values = [row.coords[field_name] for row in rows if field_name in row.coords]
        stats[field_name] = {
            "count": len(values),
            "min": min(values) if values else None,
            "p01": _percentile(values, 0.01),
            "median": _median(values),
            "p99": _percentile(values, 0.99),
            "max": max(values) if values else None,
        }
    z_values = [row.coords["z"] for row in rows if "z" in row.coords]
    z_median = _median(z_values)
    z_mad = (
        _median([abs(value - z_median) for value in z_values])
        if z_median is not None
        else None
    )
    stats["z"]["mad"] = z_mad
    stats["z"]["robust_mad_threshold"] = float(
        config["audit"]["robust_z_mad_threshold"]
    )
    stats["z"]["mad_candidate_enabled"] = bool(
        config["audit"].get("z_mad_candidate_enabled", False)
    )
    stats["z_mad"] = z_mad
    stats["robust_z_mad_threshold"] = float(config["audit"]["robust_z_mad_threshold"])
    stats["z_mad_candidate_enabled"] = bool(
        config["audit"].get("z_mad_candidate_enabled", False)
    )
    return stats


def _audit_report_payload(
    ctx: ConfigContext,
    snapshot: SourceSnapshot,
    candidate_bytes: bytes,
    candidate_csv_bytes: bytes,
) -> dict[str, Any]:
    config = ctx.config
    all_rows = [row for rows in snapshot.rows_by_map.values() for row in rows]
    integrity_rows = [row for row in all_rows if row.integrity_invalid]
    coordinate_rows = [row for row in all_rows if row.coordinate_candidate]
    candidate_rows = [row for row in all_rows if row.candidate]
    integrity_by_reason = Counter(
        reason for row in integrity_rows for reason in row.integrity_reasons
    )
    coordinate_by_reason = Counter(
        reason for row in coordinate_rows for reason in row.coordinate_reasons
    )
    coordinate_category_counts = {
        "coordinate": len(coordinate_rows),
        "coordinate_and_integrity": sum(
            row.coordinate_candidate and row.integrity_invalid for row in all_rows
        ),
    }
    map_counts: dict[str, Any] = {}
    record_counts: dict[str, Any] = {}
    for map_name in list(config["maps"]["seen"]) + list(config["maps"]["crossmap"]):
        rows = snapshot.rows_by_map[map_name]
        valid_records = sorted(
            {
                row.record_id
                for row in rows
                if row.record_id is not None and not row.integrity_invalid
            },
            key=_record_sort_key,
        )
        all_records = sorted(
            {row.record_id for row in rows if row.record_id is not None},
            key=_record_sort_key,
        )
        map_counts[map_name] = {
            "rows_total": len(rows),
            "rows_integrity_invalid": sum(row.integrity_invalid for row in rows),
            "rows_coordinate_candidates": sum(row.coordinate_candidate for row in rows),
            "candidate_rows": sum(row.candidate for row in rows),
            "records_total": len(all_records),
            "records_with_usable_rows": len(valid_records),
            "radar_exists": snapshot.radar_exists[map_name],
            "coordinate_candidate_reason_counts": dict(
                sorted(
                    Counter(
                        reason for row in rows for reason in row.coordinate_reasons
                    ).items()
                )
            ),
            "integrity_candidate_reason_counts": dict(
                sorted(
                    Counter(
                        reason for row in rows for reason in row.integrity_reasons
                    ).items()
                )
            ),
        }
        record_counts[map_name] = {
            "total": len(all_records),
            "with_usable_rows": len(valid_records),
            "usable_record_ids": valid_records,
        }

    forbidden = []
    for map_name in config["source"].get("forbidden_map_dirs", []):
        forbidden.append(
            {
                "map": map_name,
                "path": str(Path(config["paths"]["source_root"]) / map_name),
                "exists": (ctx.source_root / map_name).is_dir(),
                "selected": map_name
                in set(config["maps"]["seen"]) | set(config["maps"]["crossmap"]),
            }
        )

    payload: dict[str, Any] = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "benchmark_id": config["benchmark"]["id"],
        "benchmark_version": config["benchmark"]["version"],
        "config_sha256": sha256_file(ctx.config_path),
        "source": {
            "root": str(config["paths"]["source_root"]),
            "positions_file": str(config["source"]["positions_file"]),
            "positions_sha256": snapshot.positions_sha256,
            "positions_assets": {
                map_name: {
                    "path": str(
                        Path(map_name) / str(config["source"]["positions_file"])
                    ),
                    "sha256": snapshot.positions_sha256[map_name],
                }
                for map_name in snapshot.positions_sha256
            },
            "radar_sha256": snapshot.radar_sha256,
            "radar_files": dict(config["source"]["radar_files"]),
            "radar_assets": {
                map_name: {
                    "path": str(Path(config["source"]["radar_files"][map_name])),
                    "sha256": snapshot.radar_sha256[map_name],
                }
                for map_name in snapshot.radar_sha256
            },
            "source_sha256": snapshot.source_sha256,
        },
        "positions_sha256": snapshot.positions_sha256,
        "radar_sha256": snapshot.radar_sha256,
        "forbidden_map_dirs": forbidden,
        "counts": {
            "rows_total": len(all_rows),
            "rows_integrity_invalid": len(integrity_rows),
            "rows_coordinate_candidates": len(coordinate_rows),
            "candidate_rows": len(candidate_rows),
            "by_map": map_counts,
        },
        "record_counts": record_counts,
        "coordinate_stats": {
            map_name: _coordinate_stats(snapshot.rows_by_map[map_name], config)
            for map_name in list(config["maps"]["seen"])
            + list(config["maps"]["crossmap"])
        },
        "integrity_anomalies": {
            "total": len(integrity_rows),
            "total_rows": len(integrity_rows),
            "by_reason": dict(sorted(integrity_by_reason.items())),
            "items": [
                {
                    "map": row.map_name,
                    "file_frame": row.file_frame,
                    "source_index": row.source_index,
                    "reasons": list(row.integrity_reasons),
                }
                for row in all_rows
                if row.integrity_invalid
            ],
        },
        "coordinate_candidates": {
            "count": len(coordinate_rows),
            "total": len(coordinate_rows),
            "total_rows": len(coordinate_rows),
            "categories": coordinate_category_counts,
            "by_category": coordinate_category_counts,
            "by_reason": dict(sorted(coordinate_by_reason.items())),
            "preview_limit": int(config["audit"]["candidate_preview_limit"]),
            "preview": [
                _candidate_line(row, config)
                for row in all_rows
                if row.coordinate_candidate
            ][: int(config["audit"]["candidate_preview_limit"])],
        },
        "output": {
            "candidate_file": "audit/coordinate_candidates.jsonl",
            "candidate_file_sha256": sha256_bytes(candidate_bytes),
            "candidate_file_line_count": len(candidate_bytes.splitlines()),
            "candidate_csv_file": COORDINATE_CANDIDATE_CSV,
            "candidate_csv_file_sha256": sha256_bytes(candidate_csv_bytes),
            "candidate_csv_file_line_count": len(candidate_csv_bytes.splitlines()),
        },
    }
    return payload


def audit_fingerprint(report: Mapping[str, Any]) -> str:
    """Hash the report payload without its self-referential fingerprint field."""

    payload = {
        key: value for key, value in report.items() if key != "audit_report_sha256"
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _decision_template(
    fingerprint: str, candidate_file_sha256: str, candidate_file_line_count: int
) -> str:
    return (
        "schema_version: 1\n"
        "benchmark_id: csgo_benchmark_v2\n"
        "review:\n"
        "  status: pending\n"
        "  reviewer: ''\n"
        "  reviewed_at: ''\n"
        f"  audit_report_sha256: {fingerprint}\n"
        f"  candidate_file_sha256: {candidate_file_sha256}\n"
        f"  candidate_file_line_count: {candidate_file_line_count}\n"
        "  references: []\n"
        "coordinate_candidates:\n"
        "  default_action: undecided\n"
        "  exclude_records: {}\n"
        "  exclude_file_frames: {}\n"
        "  keep_file_frames: {}\n"
        "notes: ''\n"
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_artifacts(
    output_root: Path,
    artifacts: Mapping[str, bytes],
    overwrite: bool,
) -> None:
    paths = {relative: output_root / relative for relative in artifacts}
    if not overwrite:
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise BenchmarkError(
                "refusing to overwrite generated artifacts; use --overwrite: "
                + ", ".join(existing[:5])
            )
    for relative, data in artifacts.items():
        path = paths[relative]
        if path.exists() and not path.is_file():
            raise BenchmarkError(f"generated artifact path is not a file: {path}")
        _atomic_write(path, data)


def run_audit(
    config_path: str | os.PathLike[str], overwrite: bool = False
) -> dict[str, Any]:
    ctx = make_context(config_path)
    snapshot = _source_snapshot(ctx)
    candidate_data = _candidate_bytes(snapshot, ctx.config)
    candidate_csv_data = _candidate_csv_bytes(snapshot, ctx.config)
    payload = _audit_report_payload(ctx, snapshot, candidate_data, candidate_csv_data)
    fingerprint = audit_fingerprint(payload)
    report = dict(payload)
    report["audit_report_sha256"] = fingerprint
    artifacts = {
        "audit/audit_report.json": json_bytes(report, ctx.indent),
        "audit/coordinate_candidates.jsonl": candidate_data,
        COORDINATE_CANDIDATE_CSV: candidate_csv_data,
        "audit/anomaly_decisions.template.yaml": _decision_template(
            fingerprint,
            payload["output"]["candidate_file_sha256"],
            payload["output"]["candidate_file_line_count"],
        ).encode("utf-8"),
    }
    _write_artifacts(ctx.output_root, artifacts, overwrite)
    return report


def audit(
    config_path: str | os.PathLike[str], overwrite: bool = False
) -> dict[str, Any]:
    """Public alias for :func:`run_audit`."""

    return run_audit(config_path, overwrite=overwrite)


@dataclass
class DecisionSet:
    status: str
    audit_report_sha256: str
    candidate_file_sha256: str
    candidate_file_line_count: int
    default_action: str
    exclude_records: dict[str, set[str]]
    exclude_file_frames: dict[str, set[str]]
    keep_file_frames: dict[str, set[str]]
    raw_sha256: str


def _decision_map(
    value: Any,
    name: str,
    selected_maps: set[str],
) -> dict[str, set[str]]:
    mapping = _require_mapping(value, name)
    unknown = sorted(set(mapping) - selected_maps)
    if unknown:
        raise BenchmarkError(f"{name} contains unknown maps: {', '.join(unknown)}")
    result: dict[str, set[str]] = {}
    for map_name, ids in mapping.items():
        values = _require_list(ids, f"{name}.{map_name}")
        if any(
            isinstance(item, (dict, list, tuple, set)) or item is None
            for item in values
        ):
            raise BenchmarkError(f"{name}.{map_name} must contain scalar identifiers")
        result[map_name] = {str(item) for item in values}
    return result


def load_decisions(
    decisions_path: str | os.PathLike[str],
    config: Mapping[str, Any],
    *,
    expected_audit_report_sha256: str | None = None,
    expected_candidate_file_sha256: str | None = None,
    expected_candidate_file_line_count: int | None = None,
) -> DecisionSet:
    require_yaml()
    path = _path_from_cwd(str(decisions_path))
    if not path.is_file():
        raise BenchmarkError(f"decisions file does not exist: {path}")
    raw_bytes = path.read_bytes()
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise BenchmarkError(f"could not parse decisions YAML {path}: {exc}") from exc
    root = _require_mapping(loaded, "decisions")
    if root.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise BenchmarkError(
            f"decisions.schema_version must be {EXPECTED_SCHEMA_VERSION}"
        )
    if root.get("benchmark_id") != config["benchmark"]["id"]:
        raise BenchmarkError(
            "decisions.benchmark_id does not match config benchmark.id"
        )
    review = _require_mapping(root.get("review"), "decisions.review")
    status = review.get("status")
    if status not in ("pending", "approved"):
        raise BenchmarkError("decisions.review.status must be pending or approved")
    fingerprint = review.get("audit_report_sha256")
    if not isinstance(fingerprint, str) or not re.fullmatch(
        r"[0-9a-f]{64}", fingerprint
    ):
        raise BenchmarkError(
            "decisions.review.audit_report_sha256 must be a SHA-256 hex string"
        )
    candidate_file_sha256 = review.get("candidate_file_sha256")
    if not isinstance(candidate_file_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", candidate_file_sha256
    ):
        raise BenchmarkError(
            "decisions.review.candidate_file_sha256 must be a SHA-256 hex string"
        )
    candidate_file_line_count = review.get("candidate_file_line_count")
    if (
        isinstance(candidate_file_line_count, bool)
        or not isinstance(candidate_file_line_count, int)
        or candidate_file_line_count < 0
    ):
        raise BenchmarkError(
            "decisions.review.candidate_file_line_count must be a non-negative integer"
        )
    if status == "approved":
        reviewer = review.get("reviewer")
        reviewed_at = review.get("reviewed_at")
        references = review.get("references")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise BenchmarkError(
                "approved decisions require a nonempty review.reviewer"
            )
        if not isinstance(reviewed_at, str) or not reviewed_at.strip():
            raise BenchmarkError(
                "approved decisions require a nonempty review.reviewed_at"
            )
        if not isinstance(references, list) or not any(
            isinstance(reference, str) and reference.strip() for reference in references
        ):
            raise BenchmarkError(
                "approved decisions require at least one nonempty review.references string"
            )
    if (
        expected_audit_report_sha256 is not None
        and fingerprint != expected_audit_report_sha256
    ):
        raise BenchmarkError(
            "decisions audit_report_sha256 does not match current audit"
        )
    if (
        expected_candidate_file_sha256 is not None
        and candidate_file_sha256 != expected_candidate_file_sha256
    ):
        raise BenchmarkError(
            "decisions candidate_file_sha256 does not match current audit"
        )
    if (
        expected_candidate_file_line_count is not None
        and candidate_file_line_count != expected_candidate_file_line_count
    ):
        raise BenchmarkError(
            "decisions candidate_file_line_count does not match current audit"
        )
    configured_output_root = _path_from_cwd(str(config["paths"]["output_root"]))
    candidate_path = configured_output_root / "audit" / "coordinate_candidates.jsonl"
    if candidate_path.is_file():
        actual_candidate_bytes = candidate_path.read_bytes()
        actual_candidate_sha256 = sha256_bytes(actual_candidate_bytes)
        actual_candidate_line_count = len(actual_candidate_bytes.splitlines())
        if candidate_file_sha256 != actual_candidate_sha256:
            raise BenchmarkError(
                "decisions candidate_file_sha256 does not match current audit"
            )
        if candidate_file_line_count != actual_candidate_line_count:
            raise BenchmarkError(
                "decisions candidate_file_line_count does not match current audit"
            )
    audit_path = configured_output_root / "audit" / "audit_report.json"
    if expected_audit_report_sha256 is None and audit_path.is_file():
        try:
            current_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BenchmarkError(
                f"could not read current audit report: {audit_path}: {exc}"
            ) from exc
        current_audit_sha256 = current_audit.get("audit_report_sha256")
        if fingerprint != current_audit_sha256:
            raise BenchmarkError(
                "decisions audit_report_sha256 does not match current audit"
            )
    candidate_section = _require_mapping(
        root.get("coordinate_candidates"), "decisions.coordinate_candidates"
    )
    default_action = candidate_section.get("default_action")
    if default_action not in ("undecided", "keep", "exclude"):
        raise BenchmarkError(
            "coordinate_candidates.default_action must be undecided, keep, or exclude"
        )
    selected_maps = set(config["maps"]["seen"]) | set(config["maps"]["crossmap"])
    exclude_records = _decision_map(
        candidate_section.get("exclude_records", {}),
        "coordinate_candidates.exclude_records",
        selected_maps,
    )
    exclude_file_frames = _decision_map(
        candidate_section.get("exclude_file_frames", {}),
        "coordinate_candidates.exclude_file_frames",
        selected_maps,
    )
    keep_file_frames = _decision_map(
        candidate_section.get("keep_file_frames", {}),
        "coordinate_candidates.keep_file_frames",
        selected_maps,
    )
    return DecisionSet(
        status=str(status),
        audit_report_sha256=fingerprint,
        candidate_file_sha256=candidate_file_sha256,
        candidate_file_line_count=candidate_file_line_count,
        default_action=str(default_action),
        exclude_records=exclude_records,
        exclude_file_frames=exclude_file_frames,
        keep_file_frames=keep_file_frames,
        raw_sha256=sha256_bytes(raw_bytes),
    )


def _verify_audit_snapshot(
    ctx: ConfigContext,
    snapshot: SourceSnapshot,
    *,
    require_files: bool = True,
) -> tuple[dict[str, Any], str, bytes]:
    report_path = ctx.output_root / "audit" / "audit_report.json"
    candidate_path = ctx.output_root / "audit" / "coordinate_candidates.jsonl"
    candidate_csv_path = ctx.output_root / COORDINATE_CANDIDATE_CSV
    if (
        not report_path.is_file()
        or not candidate_path.is_file()
        or not candidate_csv_path.is_file()
    ):
        raise BenchmarkError(
            "audit artifacts are missing; run audit first: "
            f"{report_path}, {candidate_path}, and {candidate_csv_path}"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(
            f"could not read audit report: {report_path}: {exc}"
        ) from exc
    if not isinstance(report, Mapping):
        raise BenchmarkError("audit report must be a JSON object")
    recorded_fingerprint = report.get("audit_report_sha256")
    if (
        not isinstance(recorded_fingerprint, str)
        or audit_fingerprint(report) != recorded_fingerprint
    ):
        raise BenchmarkError("audit report fingerprint is invalid or stale")
    if report.get("config_sha256") != sha256_file(ctx.config_path):
        raise BenchmarkError("current config hash does not match audit report")
    source_section = _require_mapping(report.get("source"), "audit report source")
    if source_section.get("positions_sha256") != snapshot.positions_sha256:
        raise BenchmarkError("current positions.json hashes do not match audit report")
    if source_section.get("radar_files") != ctx.config["source"]["radar_files"]:
        raise BenchmarkError("audit radar file mapping does not match current config")
    if source_section.get("radar_sha256") != snapshot.radar_sha256:
        raise BenchmarkError("current radar asset hashes do not match audit report")
    if source_section.get("source_sha256") != snapshot.source_sha256:
        raise BenchmarkError("current source hash does not match audit report")
    missing_radars = sorted(
        map_name for map_name, exists in snapshot.radar_exists.items() if not exists
    )
    if missing_radars:
        raise BenchmarkError(
            "map-level radar assets are missing; build/validate cannot proceed: "
            + ", ".join(missing_radars)
        )
    candidate_data = candidate_path.read_bytes()
    candidate_hash = sha256_bytes(candidate_data)
    if report.get("output", {}).get("candidate_file_sha256") != candidate_hash:
        raise BenchmarkError("audit candidate file hash does not match audit report")
    if report.get("output", {}).get("candidate_file_line_count") != len(
        candidate_data.splitlines()
    ):
        raise BenchmarkError(
            "audit candidate file line count does not match audit report"
        )
    regenerated = _candidate_bytes(snapshot, ctx.config)
    if regenerated != candidate_data:
        raise BenchmarkError(
            "audit candidate file does not describe the current source"
        )
    candidate_csv_data = candidate_csv_path.read_bytes()
    if report.get("output", {}).get("candidate_csv_file") != COORDINATE_CANDIDATE_CSV:
        raise BenchmarkError("audit candidate CSV path does not match audit report")
    if report.get("output", {}).get("candidate_csv_file_sha256") != sha256_bytes(
        candidate_csv_data
    ):
        raise BenchmarkError("audit candidate CSV hash does not match audit report")
    if report.get("output", {}).get("candidate_csv_file_line_count") != len(
        candidate_csv_data.splitlines()
    ):
        raise BenchmarkError(
            "audit candidate CSV line count does not match audit report"
        )
    regenerated_csv = _candidate_csv_bytes(snapshot, ctx.config)
    if regenerated_csv != candidate_csv_data:
        raise BenchmarkError("audit candidate CSV does not describe the current source")
    if require_files:
        template_path = ctx.output_root / "audit" / "anomaly_decisions.template.yaml"
        if not template_path.is_file():
            raise BenchmarkError(f"audit decision template is missing: {template_path}")
    return dict(report), recorded_fingerprint, candidate_data


def _record_ids(rows: Iterable[SourceRow]) -> list[str]:
    return sorted(
        {row.record_id for row in rows if row.record_id is not None},
        key=_record_sort_key,
    )


def _validate_decision_ids(decisions: DecisionSet, snapshot: SourceSnapshot) -> None:
    for map_name, rows in snapshot.rows_by_map.items():
        known_file_frames = {
            row.file_frame for row in rows if row.file_frame is not None
        }
        known_records = {row.record_id for row in rows if row.record_id is not None}
        coordinate_file_frames = {
            row.file_frame
            for row in rows
            if row.file_frame is not None and row.coordinate_candidate
        }
        integrity_file_frames = {
            row.file_frame
            for row in rows
            if row.file_frame is not None and row.integrity_invalid
        }
        for label, values, known in (
            (
                "exclude_file_frames",
                decisions.exclude_file_frames.get(map_name, set()),
                known_file_frames,
            ),
            (
                "keep_file_frames",
                decisions.keep_file_frames.get(map_name, set()),
                known_file_frames,
            ),
        ):
            unknown = sorted(values - known)
            if unknown:
                raise BenchmarkError(
                    f"{label}.{map_name} contains unknown file_frame IDs: {', '.join(unknown)}"
                )
        unknown_records = sorted(
            decisions.exclude_records.get(map_name, set()) - known_records
        )
        if unknown_records:
            raise BenchmarkError(
                f"exclude_records.{map_name} contains unknown record IDs: "
                + ", ".join(unknown_records)
            )
        invalid_keeps = sorted(
            decisions.keep_file_frames.get(map_name, set()) & integrity_file_frames
        )
        if invalid_keeps:
            raise BenchmarkError(
                "keep_file_frames cannot rescue integrity-invalid rows: "
                + ", ".join(f"{map_name}:{item}" for item in invalid_keeps)
            )
        non_coordinate_keeps = sorted(
            decisions.keep_file_frames.get(map_name, set()) - coordinate_file_frames
        )
        if non_coordinate_keeps:
            raise BenchmarkError(
                f"keep_file_frames.{map_name} must name coordinate candidates: "
                + ", ".join(non_coordinate_keeps)
            )


def assign_record_pools(
    record_ids: Sequence[str],
    ratios: Mapping[str, float],
    global_seed: int,
    map_name: str,
    order: Sequence[str],
) -> dict[str, list[str]]:
    """Assign each record exactly once, with pool-specific SHA-256 ranks."""

    counts = largest_remainder_counts(len(record_ids), ratios, order)
    remaining = set(record_ids)
    pools: dict[str, list[str]] = {}
    for pool_name in order:
        ordered = sorted(
            remaining,
            key=lambda record_id: (
                stable_rank(global_seed, map_name, "record_pool", pool_name, record_id),
                _record_sort_key(record_id),
            ),
        )
        selected = ordered[: counts[pool_name]]
        pools[pool_name] = selected
        remaining.difference_update(selected)
    if remaining:
        raise BenchmarkError(f"internal record pool allocation error for {map_name}")
    return pools


def temporal_thin(rows: Sequence[SourceRow], min_frame_gap: int) -> list[SourceRow]:
    """Greedily retain source frames at least ``min_frame_gap`` apart per record."""

    if min_frame_gap < 0:
        raise ValueError("min_frame_gap must be non-negative")
    grouped: dict[str, list[SourceRow]] = defaultdict(list)
    for row in rows:
        if row.record_id is None or row.frame_id is None:
            continue
        grouped[row.record_id].append(row)
    result: list[SourceRow] = []
    for record_id in sorted(grouped, key=_record_sort_key):
        last_frame: int | None = None
        for row in sorted(
            grouped[record_id], key=lambda item: (item.frame_id, item.source_index)
        ):
            if last_frame is None or row.frame_id - last_frame >= min_frame_gap:
                result.append(row)
                last_frame = row.frame_id
    return result


def _quantile_bins(
    rows: Sequence[SourceRow], field_name: str, bin_count: int
) -> dict[int, int]:
    ordered = sorted(
        range(len(rows)),
        key=lambda index: (
            rows[index].coords[field_name],
            rows[index].record_id or "",
            rows[index].frame_id if rows[index].frame_id is not None else -1,
            rows[index].source_index,
        ),
    )
    result: dict[int, int] = {}
    for rank, index in enumerate(ordered):
        result[index] = min(bin_count - 1, (rank * bin_count) // len(rows))
    return result


def stratified_sample(
    rows: Sequence[SourceRow],
    quota: int,
    spatial_bins: Mapping[str, int],
    global_seed: int,
    context: Sequence[Any] = (),
) -> list[SourceRow]:
    """Select exactly ``quota`` rows by deterministic quantile-bin round robin."""

    if quota < 0:
        raise ValueError("quota must be non-negative")
    if quota > len(rows):
        raise BenchmarkError(
            f"quota {quota} exceeds temporally eligible candidate count {len(rows)}"
        )
    if quota == 0:
        return []
    dimensions = (
        ("x", int(spatial_bins["x"])),
        ("y", int(spatial_bins["y"])),
        ("z", int(spatial_bins["z"])),
        ("angle_h", int(spatial_bins["yaw"])),
        ("angle_v", int(spatial_bins["pitch"])),
    )
    per_dimension = {
        field_name: _quantile_bins(rows, field_name, max(1, count))
        for field_name, count in dimensions
    }
    groups: dict[tuple[int, ...], list[SourceRow]] = defaultdict(list)
    for index, row in enumerate(rows):
        group_key = tuple(
            per_dimension[field_name][index] for field_name, _ in dimensions
        )
        groups[group_key].append(row)
    group_keys = sorted(
        groups,
        key=lambda group: (stable_rank(global_seed, *context, "bin", group), group),
    )
    for group in group_keys:
        groups[group].sort(
            key=lambda row: (
                stable_rank(global_seed, *context, "row", row.file_frame),
                row.source_index,
            )
        )
    selected: list[SourceRow] = []
    offsets = {group: 0 for group in group_keys}
    while len(selected) < quota:
        progressed = False
        for group in group_keys:
            offset = offsets[group]
            if offset < len(groups[group]):
                selected.append(groups[group][offset])
                offsets[group] += 1
                progressed = True
                if len(selected) == quota:
                    break
        if not progressed:
            raise BenchmarkError("internal stratified sampling error")
    return selected


class _PoseGrid:
    def __init__(
        self,
        references: Sequence[SourceRow],
        xy_tolerance: float,
        z_tolerance: float,
        angle_tolerance_degrees: float,
    ) -> None:
        self.references = list(references)
        self.xy_tolerance = float(xy_tolerance)
        self.z_tolerance = float(z_tolerance)
        self.angle_tolerance = math.radians(float(angle_tolerance_degrees))
        self.cell_size = self.xy_tolerance if self.xy_tolerance > 0 else 1.0
        self.cells: dict[tuple[int, int], list[SourceRow]] = defaultdict(list)
        for row in self.references:
            if len(row.coords) == 5:
                self.cells[self._cell(row.coords["x"], row.coords["y"])].append(row)

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return math.floor(x / self.cell_size), math.floor(y / self.cell_size)

    def is_near(self, row: SourceRow) -> bool:
        if len(row.coords) != 5:
            return False
        cell_x, cell_y = self._cell(row.coords["x"], row.coords["y"])
        for delta_x in (-1, 0, 1):
            for delta_y in (-1, 0, 1):
                for reference in self.cells.get(
                    (cell_x + delta_x, cell_y + delta_y), []
                ):
                    xy_delta = math.hypot(
                        row.coords["x"] - reference.coords["x"],
                        row.coords["y"] - reference.coords["y"],
                    )
                    if xy_delta > self.xy_tolerance:
                        continue
                    if abs(row.coords["z"] - reference.coords["z"]) > self.z_tolerance:
                        continue
                    if (
                        _circular_delta(
                            row.coords["angle_h"], reference.coords["angle_h"]
                        )
                        > self.angle_tolerance
                    ):
                        continue
                    if (
                        abs(row.coords["angle_v"] - reference.coords["angle_v"])
                        > self.angle_tolerance
                    ):
                        continue
                    return True
        return False


def near_pose_filter(
    candidates: Sequence[SourceRow],
    references: Sequence[SourceRow],
    xy_tolerance: float,
    z_tolerance: float,
    angle_tolerance_degrees: float,
) -> list[SourceRow]:
    """Filter candidates near any reference using an XY grid and circular angles."""

    grid = _PoseGrid(references, xy_tolerance, z_tolerance, angle_tolerance_degrees)
    return [row for row in candidates if not grid.is_near(row)]


def _rows_for_pool(
    rows: Sequence[SourceRow], record_ids: Sequence[str]
) -> list[SourceRow]:
    allowed = set(record_ids)
    return [row for row in rows if row.record_id in allowed]


def _pool_context_name(split_name: str) -> str:
    return split_name.replace("/", "_")


def _select_discrete(
    rows: Sequence[SourceRow],
    quota: int,
    config: Mapping[str, Any],
    seed: int,
    context: Sequence[Any],
    references: Sequence[SourceRow] = (),
) -> list[SourceRow]:
    if quota == 0:
        return []
    filtered = list(rows)
    near_pose = config["sampling"]["near_pose_filter"]
    if references and near_pose.get("enabled", False):
        filtered = near_pose_filter(
            filtered,
            references,
            float(near_pose["xy_tolerance"]),
            float(near_pose["z_tolerance"]),
            float(near_pose["angle_tolerance_degrees"]),
        )
    thinned = temporal_thin(filtered, int(config["sampling"]["min_frame_gap"]))
    if len(thinned) < quota:
        raise BenchmarkError(
            f"cannot meet {context} quota {quota}: {len(thinned)} candidates remain "
            "after near-pose filtering and temporal thinning"
        )
    return stratified_sample(
        thinned,
        quota,
        config["sampling"]["spatial_bins"],
        seed,
        context,
    )


def _continuous_candidates(
    rows: Sequence[SourceRow], frames_per_clip: int, max_frame_gap: int
) -> list[list[SourceRow]]:
    grouped: dict[str, list[SourceRow]] = defaultdict(list)
    for row in rows:
        if row.record_id is not None and row.frame_id is not None:
            grouped[row.record_id].append(row)
    candidates: list[list[SourceRow]] = []
    for record_id in sorted(grouped, key=_record_sort_key):
        ordered = sorted(
            grouped[record_id], key=lambda row: (row.frame_id, row.source_index)
        )
        run: list[SourceRow] = []
        runs: list[list[SourceRow]] = []
        for row in ordered:
            if not run or 1 <= row.frame_id - run[-1].frame_id <= max_frame_gap:
                run.append(row)
            else:
                runs.append(run)
                run = [row]
        if run:
            runs.append(run)
        for contiguous_run in runs:
            if len(contiguous_run) < frames_per_clip:
                continue
            for start in range(len(contiguous_run) - frames_per_clip + 1):
                candidates.append(contiguous_run[start : start + frames_per_clip])
    return candidates


def select_continuous_clips(
    rows: Sequence[SourceRow],
    clips_per_map: int,
    frames_per_clip: int,
    max_frame_gap: int,
    max_clips_per_record: int,
    global_seed: int,
    map_name: str,
) -> list[list[SourceRow]]:
    candidates = _continuous_candidates(rows, frames_per_clip, max_frame_gap)
    candidates.sort(
        key=lambda clip: (
            stable_rank(
                global_seed,
                map_name,
                "continuous_clip",
                clip[0].record_id,
                clip[0].frame_id,
                clip[-1].frame_id,
            ),
            _record_sort_key(clip[0].record_id or ""),
            clip[0].frame_id,
        )
    )
    selected: list[list[SourceRow]] = []
    used_frames: set[tuple[str, str]] = set()
    record_clip_counts: Counter[str] = Counter()
    for clip in candidates:
        record_id = clip[0].record_id
        assert record_id is not None
        if record_clip_counts[record_id] >= max_clips_per_record:
            continue
        keys = {(map_name, row.file_frame or "") for row in clip}
        if used_frames & keys:
            continue
        selected.append(clip)
        used_frames.update(keys)
        record_clip_counts[record_id] += 1
        if len(selected) == clips_per_map:
            break
    if len(selected) != clips_per_map:
        raise BenchmarkError(
            f"cannot meet continuous clip quota for {map_name}: requested "
            f"{clips_per_map}, found {len(selected)} non-overlapping clips"
        )
    return selected


def _split_file_bytes(value: Any, indent: int) -> bytes:
    return json_bytes(value, indent)


def _relative_artifact_paths(config: Mapping[str, Any]) -> list[str]:
    paths = [
        CALIBRATION_JSON,
        CALIBRATION_EXTREMA_JSONL,
        "benchmark_manifest.json",
        "build_report.json",
        "selected_images.sha256",
    ]
    for map_name in config["maps"]["seen"]:
        prefix = f"splits/seen/{map_name}"
        paths.extend(
            f"{prefix}/{name}.json" for name in (*DISCRETE_SPLITS, "continuous_clips")
        )
    seeds = config["counts"]["crossmap"]["support_seeds"]
    for map_name in config["maps"]["crossmap"]:
        prefix = f"splits/crossmap/{map_name}"
        paths.extend(f"{prefix}/support_seed_{seed}.json" for seed in seeds)
        paths.extend((f"{prefix}/query_test.json", f"{prefix}/continuous_clips.json"))
    paths.extend(
        [
            "aggregate/seen_train.json",
            "aggregate/seen_validation.json",
            "aggregate/seen_discrete_test.json",
            "aggregate/crossmap_query_test.json",
        ]
    )
    paths.extend(f"aggregate/crossmap_support_seed_{seed}.json" for seed in seeds)
    return paths


def _row_map(rows: Sequence[SourceRow]) -> dict[tuple[str, str], SourceRow]:
    result: dict[tuple[str, str], SourceRow] = {}
    for row in rows:
        if row.file_frame is not None and row.key not in result:
            result[row.key] = row
    return result


def _apply_decisions(
    snapshot: SourceSnapshot,
    config: Mapping[str, Any],
    decisions: DecisionSet,
) -> tuple[dict[str, list[SourceRow]], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: dict[str, list[SourceRow]] = defaultdict(list)
    exclusions: list[dict[str, Any]] = []
    accepted_anomalies: list[dict[str, Any]] = []
    integrity_without_exclusion: list[str] = []
    selected_maps = list(config["maps"]["seen"]) + list(config["maps"]["crossmap"])
    for map_name in selected_maps:
        map_excluded_records = decisions.exclude_records.get(map_name, set())
        map_excluded_frames = decisions.exclude_file_frames.get(map_name, set())
        map_kept_frames = decisions.keep_file_frames.get(map_name, set())
        for row in snapshot.rows_by_map[map_name]:
            file_frame = row.file_frame
            explicit_record = (
                row.record_id is not None and row.record_id in map_excluded_records
            )
            explicit_frame = (
                file_frame is not None and file_frame in map_excluded_frames
            )
            explicit_exclusion = explicit_record or explicit_frame
            keep_override = file_frame is not None and file_frame in map_kept_frames
            if row.integrity_invalid:
                if not explicit_exclusion:
                    integrity_without_exclusion.append(
                        f"{map_name}:{file_frame if file_frame is not None else row.source_index}"
                    )
                    continue
                exclusions.append(
                    {
                        "map": map_name,
                        "file_frame": file_frame,
                        "source_index": row.source_index,
                        "record_id": row.record_id,
                        "reasons": list(row.integrity_reasons),
                        "coordinate_reasons": list(row.coordinate_reasons),
                        "action": "excluded_integrity_invalid",
                        "decision_source": "exclude_records"
                        if explicit_record
                        else "exclude_file_frames",
                    }
                )
                continue

            if keep_override:
                excluded = False
                decision_source = "keep_file_frames"
            elif explicit_exclusion:
                excluded = True
                decision_source = (
                    "exclude_records" if explicit_record else "exclude_file_frames"
                )
            elif row.coordinate_candidate and decisions.default_action == "exclude":
                excluded = True
                decision_source = "coordinate_candidates.default_action"
            else:
                excluded = False
                decision_source = (
                    "coordinate_candidates.default_action"
                    if row.coordinate_candidate
                    else "implicit_keep"
                )

            if excluded:
                exclusions.append(
                    {
                        "map": map_name,
                        "file_frame": file_frame,
                        "source_index": row.source_index,
                        "record_id": row.record_id,
                        "reasons": list(row.coordinate_reasons),
                        "action": "excluded",
                        "decision_source": decision_source,
                    }
                )
            else:
                accepted[map_name].append(row)
                if row.coordinate_candidate:
                    accepted_anomalies.append(
                        {
                            "map": map_name,
                            "file_frame": file_frame,
                            "source_index": row.source_index,
                            "record_id": row.record_id,
                            "reasons": list(row.coordinate_reasons),
                            "action": "accepted_coordinate_anomaly",
                            "decision_source": decision_source,
                        }
                    )
    if integrity_without_exclusion:
        preview = ", ".join(integrity_without_exclusion[:10])
        suffix = "" if len(integrity_without_exclusion) <= 10 else " ..."
        raise BenchmarkError(
            "integrity-invalid rows require explicit exclude_records or "
            f"exclude_file_frames decisions: {preview}{suffix}"
        )
    return dict(accepted), exclusions, accepted_anomalies


@dataclass(frozen=True)
class CalibrationBundle:
    payload: dict[str, Any]
    calibration_bytes: bytes
    extrema_bytes: bytes
    calibration_sha256: str
    calibration_file_sha256: str
    extrema_file_sha256: str
    extrema_file_line_count: int
    approval_sha256: str | None = None


def calibration_fingerprint(payload: Mapping[str, Any]) -> str:
    """Hash calibration semantics without the self-referential fingerprint."""

    fingerprint_payload = {
        key: value for key, value in payload.items() if key != "calibration_sha256"
    }
    return sha256_bytes(canonical_json_bytes(fingerprint_payload))


def _calibration_extrema_line(
    row: SourceRow, extrema: Sequence[str], config: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "map": row.map_name,
        "source_index": row.source_index,
        "file_frame": row.file_frame,
        "record_id": row.record_id,
        "frame_id": row.frame_id,
        "z": row.coords["z"],
        "extrema": list(extrema),
        "source_image_path": _source_image_relative_path(
            config, row.map_name, row.file_frame
        ),
    }


def _calibration_bundle_from_accepted(
    ctx: ConfigContext,
    snapshot: SourceSnapshot,
    audit_hash: str,
    decisions: DecisionSet,
    accepted: Mapping[str, Sequence[SourceRow]],
) -> CalibrationBundle:
    """Build deterministic exact min/max calibration from approved full-corpus rows."""

    selected_maps = list(ctx.config["maps"]["seen"]) + list(
        ctx.config["maps"]["crossmap"]
    )
    z_ranges: dict[str, dict[str, Any]] = {}
    extrema_lines: list[bytes] = []
    retained_rows_by_map: dict[str, int] = {}
    retained_records_by_map: dict[str, int] = {}
    retained_total = 0
    for map_name in selected_maps:
        rows = [row for row in accepted.get(map_name, []) if "z" in row.coords]
        if not rows:
            raise BenchmarkError(
                f"cannot calibrate {map_name}: no retained finite Z rows"
            )
        z_values = [row.coords["z"] for row in rows]
        z_min = min(z_values)
        z_max = max(z_values)
        if not z_max > z_min:
            raise BenchmarkError(
                f"cannot calibrate {map_name}: z_max must be greater than z_min"
            )
        min_rows = [row for row in rows if row.coords["z"] == z_min]
        max_rows = [row for row in rows if row.coords["z"] == z_max]
        for row in rows:
            extrema: list[str] = []
            if row.coords["z"] == z_min:
                extrema.append("min")
            if row.coords["z"] == z_max:
                extrema.append("max")
            if extrema:
                extrema_lines.append(
                    canonical_json_bytes(
                        _calibration_extrema_line(row, extrema, ctx.config)
                    )
                    + b"\n"
                )
        retained_rows_by_map[map_name] = len(rows)
        retained_records_by_map[map_name] = len(
            {row.record_id for row in rows if row.record_id is not None}
        )
        retained_total += len(rows)
        z_ranges[map_name] = {
            "z_min": z_min,
            "z_max": z_max,
            "span": z_max - z_min,
            "retained_row_count": len(rows),
            "retained_record_count": retained_records_by_map[map_name],
            "min_row_count": len(min_rows),
            "max_row_count": len(max_rows),
        }

    extrema_bytes = b"".join(extrema_lines)
    payload: dict[str, Any] = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "benchmark_id": ctx.config["benchmark"]["id"],
        "benchmark_version": ctx.config["benchmark"]["version"],
        "calibration": {
            "source": "approved_full_corpus",
            "method": "exact_min_max",
            "timing": "before_split",
            "normalize_to": [0.0, 1.0],
            "clamp": False,
            "fallback": "error",
        },
        "config_sha256": sha256_file(ctx.config_path),
        "source": {
            "positions_sha256": snapshot.positions_sha256,
            "radar_sha256": snapshot.radar_sha256,
            "source_sha256": snapshot.source_sha256,
        },
        "audit_report_sha256": audit_hash,
        "decisions_sha256": decisions.raw_sha256,
        "decision_review_status": decisions.status,
        "retained_rows": {
            "total": retained_total,
            "by_map": retained_rows_by_map,
        },
        "retained_records": retained_records_by_map,
        "z_ranges": z_ranges,
        "extrema_rows": {
            "file": CALIBRATION_EXTREMA_JSONL,
            "sha256": sha256_bytes(extrema_bytes),
            "line_count": len(extrema_bytes.splitlines()),
        },
    }
    calibration_sha256 = calibration_fingerprint(payload)
    payload["calibration_sha256"] = calibration_sha256
    calibration_bytes = json_bytes(payload, ctx.indent)
    return CalibrationBundle(
        payload=payload,
        calibration_bytes=calibration_bytes,
        extrema_bytes=extrema_bytes,
        calibration_sha256=calibration_sha256,
        calibration_file_sha256=sha256_bytes(calibration_bytes),
        extrema_file_sha256=sha256_bytes(extrema_bytes),
        extrema_file_line_count=len(extrema_bytes.splitlines()),
    )


def _calibration_approval_template(
    ctx: ConfigContext,
    snapshot: SourceSnapshot,
    audit_hash: str,
    decisions: DecisionSet,
    bundle: CalibrationBundle,
) -> bytes:
    require_yaml()
    value = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "benchmark_id": ctx.config["benchmark"]["id"],
        "review": {
            "status": "pending",
            "reviewer": "",
            "reviewed_at": "",
            "references": [],
            "config_sha256": sha256_file(ctx.config_path),
            "source_sha256": snapshot.source_sha256,
            "audit_report_sha256": audit_hash,
            "decisions_sha256": decisions.raw_sha256,
            "calibration_sha256": bundle.calibration_sha256,
            "calibration_file_sha256": bundle.calibration_file_sha256,
            "extrema_file_sha256": bundle.extrema_file_sha256,
            "extrema_file_line_count": bundle.extrema_file_line_count,
        },
        "z_ranges": bundle.payload["z_ranges"],
        "notes": "Review the extrema rows, then set review.status=approved.",
    }
    return yaml.safe_dump(value, sort_keys=False).encode("utf-8")


def _read_calibration_bundle(ctx: ConfigContext) -> CalibrationBundle:
    calibration_path = ctx.output_root / CALIBRATION_JSON
    extrema_path = ctx.output_root / CALIBRATION_EXTREMA_JSONL
    if not calibration_path.is_file() or not extrema_path.is_file():
        raise BenchmarkError(
            "calibration artifacts are missing; run calibrate first: "
            f"{calibration_path} and {extrema_path}"
        )
    calibration_bytes = calibration_path.read_bytes()
    try:
        payload = json.loads(calibration_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(
            f"could not read calibration artifact {calibration_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise BenchmarkError("calibration artifact must be a JSON object")
    recorded_fingerprint = payload.get("calibration_sha256")
    if not isinstance(recorded_fingerprint, str) or not re.fullmatch(
        r"[0-9a-f]{64}", recorded_fingerprint
    ):
        raise BenchmarkError(
            "calibration.calibration_sha256 must be a SHA-256 hex string"
        )
    if calibration_fingerprint(payload) != recorded_fingerprint:
        raise BenchmarkError("calibration fingerprint is invalid or stale")
    extrema_bytes = extrema_path.read_bytes()
    extrema_section = _require_mapping(
        payload.get("extrema_rows"), "calibration.extrema_rows"
    )
    if extrema_section.get("file") != CALIBRATION_EXTREMA_JSONL:
        raise BenchmarkError(
            "calibration extrema file path does not match the protocol"
        )
    if extrema_section.get("sha256") != sha256_bytes(extrema_bytes):
        raise BenchmarkError("calibration extrema file hash does not match calibration")
    if extrema_section.get("line_count") != len(extrema_bytes.splitlines()):
        raise BenchmarkError(
            "calibration extrema file line count does not match calibration"
        )
    return CalibrationBundle(
        payload=dict(payload),
        calibration_bytes=calibration_bytes,
        extrema_bytes=extrema_bytes,
        calibration_sha256=recorded_fingerprint,
        calibration_file_sha256=sha256_bytes(calibration_bytes),
        extrema_file_sha256=sha256_bytes(extrema_bytes),
        extrema_file_line_count=len(extrema_bytes.splitlines()),
    )


def load_calibration_approval(
    approval_path: str | os.PathLike[str],
    config: Mapping[str, Any],
    *,
    expected_config_sha256: str,
    expected_source_sha256: str,
    expected_audit_report_sha256: str,
    expected_decisions_sha256: str,
    expected_calibration_sha256: str,
    expected_calibration_file_sha256: str,
    expected_extrema_file_sha256: str,
    expected_extrema_file_line_count: int,
) -> str:
    """Load and strictly bind a human-approved calibration contract."""

    require_yaml()
    path = _path_from_cwd(str(approval_path))
    if not path.is_file():
        raise BenchmarkError(f"calibration approval file does not exist: {path}")
    raw_bytes = path.read_bytes()
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise BenchmarkError(
            f"could not parse calibration approval YAML {path}: {exc}"
        ) from exc
    root = _require_mapping(loaded, "calibration approval")
    if root.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise BenchmarkError(
            f"calibration approval schema_version must be {EXPECTED_SCHEMA_VERSION}"
        )
    if root.get("benchmark_id") != config["benchmark"]["id"]:
        raise BenchmarkError("calibration approval benchmark_id does not match config")
    review = _require_mapping(root.get("review"), "calibration approval.review")
    if review.get("status") != "approved":
        raise BenchmarkError("calibration approval requires review.status=approved")
    reviewer = review.get("reviewer")
    reviewed_at = review.get("reviewed_at")
    references = review.get("references")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise BenchmarkError("approved calibration requires a nonempty review.reviewer")
    if not isinstance(reviewed_at, str) or not reviewed_at.strip():
        raise BenchmarkError(
            "approved calibration requires a nonempty review.reviewed_at"
        )
    if not isinstance(references, list) or not any(
        isinstance(reference, str) and reference.strip() for reference in references
    ):
        raise BenchmarkError(
            "approved calibration requires at least one nonempty review.references string"
        )
    expected_hashes = {
        "config_sha256": expected_config_sha256,
        "source_sha256": expected_source_sha256,
        "audit_report_sha256": expected_audit_report_sha256,
        "decisions_sha256": expected_decisions_sha256,
        "calibration_sha256": expected_calibration_sha256,
        "calibration_file_sha256": expected_calibration_file_sha256,
        "extrema_file_sha256": expected_extrema_file_sha256,
    }
    for field_name, expected in expected_hashes.items():
        actual = review.get(field_name)
        if not isinstance(actual, str) or not re.fullmatch(r"[0-9a-f]{64}", actual):
            raise BenchmarkError(
                f"calibration approval review.{field_name} must be a SHA-256 hex string"
            )
        if actual != expected:
            raise BenchmarkError(
                f"calibration approval {field_name} does not match current calibration inputs"
            )
    line_count = review.get("extrema_file_line_count")
    if (
        isinstance(line_count, bool)
        or not isinstance(line_count, int)
        or line_count != expected_extrema_file_line_count
    ):
        raise BenchmarkError(
            "calibration approval extrema_file_line_count does not match current calibration"
        )
    return sha256_bytes(raw_bytes)


def _verify_calibration(
    ctx: ConfigContext,
    snapshot: SourceSnapshot,
    audit_hash: str,
    audit_report: Mapping[str, Any],
    decisions: DecisionSet,
    accepted: Mapping[str, Sequence[SourceRow]],
    approval_path: str | os.PathLike[str] | None,
) -> CalibrationBundle:
    if approval_path is None:
        raise BenchmarkError("build and validate require --calibration-approval")
    expected = _calibration_bundle_from_accepted(
        ctx, snapshot, audit_hash, decisions, accepted
    )
    actual = _read_calibration_bundle(ctx)
    if actual.calibration_bytes != expected.calibration_bytes:
        raise BenchmarkError(
            "calibration artifact does not match an independent full-corpus recomputation"
        )
    if actual.extrema_bytes != expected.extrema_bytes:
        raise BenchmarkError(
            "calibration extrema rows do not match an independent full-corpus recomputation"
        )
    source_section = _require_mapping(audit_report.get("source"), "audit report source")
    approval_sha256 = load_calibration_approval(
        approval_path,
        ctx.config,
        expected_config_sha256=sha256_file(ctx.config_path),
        expected_source_sha256=source_section["source_sha256"],
        expected_audit_report_sha256=audit_hash,
        expected_decisions_sha256=decisions.raw_sha256,
        expected_calibration_sha256=actual.calibration_sha256,
        expected_calibration_file_sha256=actual.calibration_file_sha256,
        expected_extrema_file_sha256=actual.extrema_file_sha256,
        expected_extrema_file_line_count=actual.extrema_file_line_count,
    )
    return CalibrationBundle(
        payload=actual.payload,
        calibration_bytes=actual.calibration_bytes,
        extrema_bytes=actual.extrema_bytes,
        calibration_sha256=actual.calibration_sha256,
        calibration_file_sha256=actual.calibration_file_sha256,
        extrema_file_sha256=actual.extrema_file_sha256,
        extrema_file_line_count=actual.extrema_file_line_count,
        approval_sha256=approval_sha256,
    )


def _calibration_metadata(bundle: CalibrationBundle) -> dict[str, Any]:
    return {
        "file": CALIBRATION_JSON,
        "sha256": bundle.calibration_file_sha256,
        "fingerprint": bundle.calibration_sha256,
        "extrema_file": {
            "file": CALIBRATION_EXTREMA_JSONL,
            "sha256": bundle.extrema_file_sha256,
            "line_count": bundle.extrema_file_line_count,
        },
        "z_ranges": bundle.payload["z_ranges"],
    }


def run_calibrate(
    config_path: str | os.PathLike[str],
    decisions_path: str | os.PathLike[str],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create exact full-corpus Z calibration after approved anomaly review."""

    ctx = make_context(config_path)
    snapshot = _source_snapshot(ctx)
    audit_report, audit_hash, _ = _verify_audit_snapshot(ctx, snapshot)
    decisions = load_decisions(
        decisions_path,
        ctx.config,
        expected_audit_report_sha256=audit_hash,
        expected_candidate_file_sha256=audit_report["output"]["candidate_file_sha256"],
        expected_candidate_file_line_count=audit_report["output"][
            "candidate_file_line_count"
        ],
    )
    _validate_decision_ids(decisions, snapshot)
    if decisions.status != "approved":
        raise BenchmarkError("calibrate requires review.status=approved in decisions")
    if decisions.default_action == "undecided":
        raise BenchmarkError(
            "calibrate requires coordinate_candidates.default_action=keep or exclude"
        )
    accepted, _, _ = _apply_decisions(snapshot, ctx.config, decisions)
    bundle = _calibration_bundle_from_accepted(
        ctx, snapshot, audit_hash, decisions, accepted
    )
    artifacts = {
        CALIBRATION_JSON: bundle.calibration_bytes,
        CALIBRATION_EXTREMA_JSONL: bundle.extrema_bytes,
        CALIBRATION_APPROVAL_TEMPLATE: _calibration_approval_template(
            ctx, snapshot, audit_hash, decisions, bundle
        ),
    }
    _write_artifacts(ctx.output_root, artifacts, overwrite)
    return bundle.payload


def calibrate(
    config_path: str | os.PathLike[str],
    decisions_path: str | os.PathLike[str],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Public alias for :func:`run_calibrate`."""

    return run_calibrate(config_path, decisions_path, overwrite=overwrite)


def _continuous_json(
    map_name: str, clips: Sequence[Sequence[SourceRow]], config: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "benchmark_id": config["benchmark"]["id"],
        "map": map_name,
        "split": "continuous",
        "frames_per_clip": int(config["counts"]["continuous"]["frames_per_clip"]),
        "clips": [
            {
                "map": map_name,
                "clip_id": f"{map_name}_continuous_{index:04d}",
                "record_id": clip[0].record_id,
                "frames": [copy.deepcopy(row.raw) for row in clip],
            }
            for index, clip in enumerate(clips)
        ],
    }


def _selection_keys(
    map_name: str,
    result: Mapping[str, Any],
    support_seeds: Sequence[int],
) -> dict[tuple[str, str], list[str]]:
    selected: dict[tuple[str, str], list[str]] = defaultdict(list)

    def add_rows(split_name: str, rows: Iterable[SourceRow]) -> None:
        for row in rows:
            if row.file_frame is not None:
                selected[(map_name, row.file_frame)].append(split_name)

    for split_name in DISCRETE_SPLITS:
        add_rows(split_name, result.get(split_name, []))
    for seed in support_seeds:
        add_rows(f"support_seed_{seed}", result.get(f"support_seed_{seed}", []))
    add_rows("query_test", result.get("query_test", []))
    for clip in result.get("continuous", []):
        add_rows("continuous", clip)
    return selected


def _ensure_pool_disjointness(
    map_name: str,
    results: Mapping[str, Any],
    support_seeds: Sequence[int],
) -> None:
    pool_names = list(DISCRETE_SPLITS) + ["continuous"]
    if support_seeds:
        pool_names = [f"support_seed_{seed}" for seed in support_seeds] + [
            "query_test",
            "continuous",
        ]
    record_sets: dict[str, set[str]] = {}
    for pool_name in pool_names:
        if pool_name == "continuous":
            record_sets[pool_name] = {
                row.record_id
                for clip in results.get("continuous", [])
                for row in clip
                if row.record_id is not None
            }
        else:
            record_sets[pool_name] = {
                row.record_id
                for row in results.get(pool_name, [])
                if row.record_id is not None
            }
    if support_seeds:
        exclusive = ["query_test", "continuous"]
        for left_index, left in enumerate(exclusive):
            for right in exclusive[left_index + 1 :]:
                if record_sets[left] & record_sets[right]:
                    raise BenchmarkError(
                        f"record pools overlap for {map_name}: {left} and {right}"
                    )
        for support_name in (f"support_seed_{seed}" for seed in support_seeds):
            for exclusive_name in exclusive:
                if record_sets[support_name] & record_sets[exclusive_name]:
                    raise BenchmarkError(
                        f"support and exclusive pools overlap for {map_name}: "
                        f"{support_name} and {exclusive_name}"
                    )
    else:
        for left_index, left in enumerate(pool_names):
            for right in pool_names[left_index + 1 :]:
                if record_sets[left] & record_sets[right]:
                    raise BenchmarkError(
                        f"record pools overlap for {map_name}: {left} and {right}"
                    )


def _selected_rows_for_image_hash(
    config: Mapping[str, Any], map_results: Mapping[str, Mapping[str, Any]]
) -> list[SourceRow]:
    rows: list[SourceRow] = []
    for map_name in config["maps"]["seen"]:
        result = map_results[map_name]
        for split_name in DISCRETE_SPLITS:
            rows.extend(result[split_name])
        for clip in result["continuous"]:
            rows.extend(clip)
    for map_name in config["maps"]["crossmap"]:
        result = map_results[map_name]
        for support_seed in config["counts"]["crossmap"]["support_seeds"]:
            rows.extend(result[f"support_seed_{support_seed}"])
        rows.extend(result["query_test"])
        for clip in result["continuous"]:
            rows.extend(clip)
    return rows


def _build_manifest(
    ctx: ConfigContext,
    snapshot: SourceSnapshot,
    audit_hash: str,
    decisions: DecisionSet,
    record_pools: Mapping[str, Mapping[str, Sequence[str]]],
    map_results: Mapping[str, Mapping[str, Any]],
    selected_images_artifact_sha256: str,
    selected_image_count: int,
    calibration: CalibrationBundle,
) -> dict[str, Any]:
    config = ctx.config
    counts: dict[str, Any] = {"seen": {}, "crossmap": {}}
    for map_name in config["maps"]["seen"]:
        result = map_results[map_name]
        counts["seen"][map_name] = {
            "train": len(result["train"]),
            "validation": len(result["validation"]),
            "discrete_test": len(result["discrete_test"]),
            "continuous_clips": len(result["continuous"]),
            "continuous_frames": sum(len(clip) for clip in result["continuous"]),
        }
    for map_name in config["maps"]["crossmap"]:
        result = map_results[map_name]
        counts["crossmap"][map_name] = {
            "query_test": len(result["query_test"]),
            "continuous_clips": len(result["continuous"]),
            "continuous_frames": sum(len(clip) for clip in result["continuous"]),
            "support": {
                str(seed): len(result[f"support_seed_{seed}"])
                for seed in config["counts"]["crossmap"]["support_seeds"]
            },
        }
    return {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "benchmark_id": config["benchmark"]["id"],
        "benchmark": {
            "id": config["benchmark"]["id"],
            "version": config["benchmark"]["version"],
            "global_seed": config["benchmark"]["global_seed"],
            "strict_protocol": bool(config["benchmark"].get("strict_protocol", False)),
        },
        "protocol": {
            "seen_maps": list(config["maps"]["seen"]),
            "crossmap_maps": list(config["maps"]["crossmap"]),
            "seen_splits": list(DISCRETE_SPLITS),
            "crossmap_splits": [
                "support",
                "query_test",
                "continuous",
            ],
            "support_seeds": list(config["counts"]["crossmap"]["support_seeds"]),
        },
        "counts": counts,
        "seeds": {
            "global_seed": config["benchmark"]["global_seed"],
            "support_seeds": list(config["counts"]["crossmap"]["support_seeds"]),
        },
        "record_pools": {
            map_name: {pool: list(ids) for pool, ids in pools.items()}
            for map_name, pools in record_pools.items()
        },
        "source": {
            "root": str(config["paths"]["source_root"]),
            "radar_files": dict(config["source"]["radar_files"]),
            "positions_sha256": snapshot.positions_sha256,
            "radar_sha256": snapshot.radar_sha256,
            "source_sha256": snapshot.source_sha256,
        },
        "positions_sha256": snapshot.positions_sha256,
        "radar_sha256": snapshot.radar_sha256,
        "config_sha256": sha256_file(ctx.config_path),
        "audit_report_sha256": audit_hash,
        "decisions_sha256": decisions.raw_sha256,
        "calibration": _calibration_metadata(calibration),
        "calibration_approval_sha256": calibration.approval_sha256,
        "selected_images": {
            "file": "selected_images.sha256",
            "sha256": selected_images_artifact_sha256,
            "count": selected_image_count,
        },
        "selected_anomaly_policy": {
            "coordinate_default_action": decisions.default_action,
            "keep_file_frames_override": True,
            "precedence": [
                "keep_file_frames",
                "exclude_file_frames_or_records",
                "coordinate_candidates.default_action",
            ],
            "integrity_invalid_requires_explicit_exclusion": True,
        },
        "continuous_protocol": {
            "trajectory_disjoint": True,
            "pose_disjoint": False,
            "statement": "Continuous clips are trajectory-disjoint but are not guaranteed pose-disjoint.",
            "frames_per_clip": config["counts"]["continuous"]["frames_per_clip"],
            "max_frame_gap": config["counts"]["continuous"]["max_frame_gap"],
        },
    }


def run_build(
    config_path: str | os.PathLike[str],
    decisions_path: str | os.PathLike[str],
    calibration_approval_path: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    ctx = make_context(config_path)
    snapshot = _source_snapshot(ctx)
    audit_report, audit_hash, _ = _verify_audit_snapshot(ctx, snapshot)
    decisions = load_decisions(
        decisions_path,
        ctx.config,
        expected_audit_report_sha256=audit_hash,
        expected_candidate_file_sha256=audit_report["output"]["candidate_file_sha256"],
        expected_candidate_file_line_count=audit_report["output"][
            "candidate_file_line_count"
        ],
    )
    _validate_decision_ids(decisions, snapshot)
    if decisions.status != "approved":
        raise BenchmarkError("build requires review.status=approved in decisions")
    if audit_report.get("benchmark_id") != ctx.config["benchmark"]["id"]:
        raise BenchmarkError("audit report benchmark id does not match config")
    if decisions.default_action == "undecided":
        raise BenchmarkError(
            "build requires coordinate_candidates.default_action=keep or exclude"
        )

    accepted, exclusions, accepted_anomalies = _apply_decisions(
        snapshot, ctx.config, decisions
    )
    calibration = _verify_calibration(
        ctx,
        snapshot,
        audit_hash,
        audit_report,
        decisions,
        accepted,
        calibration_approval_path,
    )
    global_seed = int(ctx.config["benchmark"]["global_seed"])
    map_results: dict[str, dict[str, Any]] = {}
    all_record_pools: dict[str, dict[str, list[str]]] = {}

    for group_name, map_names, pool_order in (
        ("seen", ctx.config["maps"]["seen"], SEEN_POOL_ORDER),
        ("crossmap", ctx.config["maps"]["crossmap"], CROSSMAP_POOL_ORDER),
    ):
        ratios = ctx.config["record_pools"][group_name]
        for map_name in map_names:
            rows = accepted.get(map_name, [])
            record_ids = _record_ids(rows)
            pools = assign_record_pools(
                record_ids,
                ratios,
                global_seed,
                map_name,
                pool_order,
            )
            all_record_pools[map_name] = pools
            pool_rows = {
                pool_name: _rows_for_pool(rows, pools[pool_name])
                for pool_name in pool_order
            }
            if group_name == "seen":
                train = _select_discrete(
                    pool_rows["train"],
                    int(ctx.config["counts"]["seen"]["train"]),
                    ctx.config,
                    global_seed,
                    (group_name, map_name, "train"),
                )
                validation = _select_discrete(
                    pool_rows["validation"],
                    int(ctx.config["counts"]["seen"]["validation"]),
                    ctx.config,
                    global_seed,
                    (group_name, map_name, "validation"),
                )
                discrete_test = _select_discrete(
                    pool_rows["discrete_test"],
                    int(ctx.config["counts"]["seen"]["discrete_test"]),
                    ctx.config,
                    global_seed,
                    (group_name, map_name, "discrete_test"),
                    references=train + validation,
                )
                continuous = select_continuous_clips(
                    pool_rows["continuous"],
                    int(ctx.config["counts"]["continuous"]["clips_per_map"]),
                    int(ctx.config["counts"]["continuous"]["frames_per_clip"]),
                    int(ctx.config["counts"]["continuous"]["max_frame_gap"]),
                    int(ctx.config["counts"]["continuous"]["max_clips_per_record"]),
                    global_seed,
                    map_name,
                )
                result = {
                    "train": train,
                    "validation": validation,
                    "discrete_test": discrete_test,
                    "continuous": continuous,
                }
            else:
                support_seeds = list(ctx.config["counts"]["crossmap"]["support_seeds"])
                result = {}
                for support_seed in support_seeds:
                    result[f"support_seed_{support_seed}"] = _select_discrete(
                        pool_rows["support"],
                        int(ctx.config["counts"]["crossmap"]["support"]),
                        ctx.config,
                        global_seed,
                        (group_name, map_name, "support", support_seed),
                    )
                support_union: list[SourceRow] = []
                for support_seed in support_seeds:
                    support_union.extend(result[f"support_seed_{support_seed}"])
                result["query_test"] = _select_discrete(
                    pool_rows["query_test"],
                    int(ctx.config["counts"]["crossmap"]["query_test"]),
                    ctx.config,
                    global_seed,
                    (group_name, map_name, "query_test"),
                    references=support_union,
                )
                result["continuous"] = select_continuous_clips(
                    pool_rows["continuous"],
                    int(ctx.config["counts"]["continuous"]["clips_per_map"]),
                    int(ctx.config["counts"]["continuous"]["frames_per_clip"]),
                    int(ctx.config["counts"]["continuous"]["max_frame_gap"]),
                    int(ctx.config["counts"]["continuous"]["max_clips_per_record"]),
                    global_seed,
                    map_name,
                )
            _ensure_pool_disjointness(
                map_name,
                result,
                list(ctx.config["counts"]["crossmap"]["support_seeds"])
                if group_name == "crossmap"
                else [],
            )
            map_results[map_name] = result

    selected_image_data, selected_image_count = _selected_image_manifest_bytes(
        ctx, _selected_rows_for_image_hash(ctx.config, map_results)
    )
    selected_images_artifact_sha256 = sha256_bytes(selected_image_data)
    support_seeds = list(ctx.config["counts"]["crossmap"]["support_seeds"])
    for anomaly in accepted_anomalies:
        selected = _selection_keys(
            anomaly["map"], map_results[anomaly["map"]], support_seeds
        )
        anomaly["selected_in"] = selected.get(
            (anomaly["map"], anomaly["file_frame"]), []
        )

    manifest = _build_manifest(
        ctx,
        snapshot,
        audit_hash,
        decisions,
        all_record_pools,
        map_results,
        selected_images_artifact_sha256,
        selected_image_count,
        calibration,
    )
    build_report: dict[str, Any] = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "benchmark_id": ctx.config["benchmark"]["id"],
        "benchmark_version": ctx.config["benchmark"]["version"],
        "config_sha256": sha256_file(ctx.config_path),
        "source": {
            "radar_files": dict(ctx.config["source"]["radar_files"]),
            "positions_sha256": snapshot.positions_sha256,
            "radar_sha256": snapshot.radar_sha256,
            "source_sha256": snapshot.source_sha256,
        },
        "positions_sha256": snapshot.positions_sha256,
        "radar_sha256": snapshot.radar_sha256,
        "audit_report_sha256": audit_hash,
        "decisions_sha256": decisions.raw_sha256,
        "calibration": _calibration_metadata(calibration),
        "calibration_approval_sha256": calibration.approval_sha256,
        "selected_images": {
            "file": "selected_images.sha256",
            "sha256": selected_images_artifact_sha256,
            "count": selected_image_count,
        },
        "decision_summary": {
            "review_status": decisions.status,
            "coordinate_default_action": decisions.default_action,
            "excluded_rows": len(exclusions),
            "accepted_coordinate_anomalies": len(accepted_anomalies),
        },
        "exclusions": exclusions,
        "accepted_coordinate_anomalies": accepted_anomalies,
        "record_pools": {
            map_name: {pool: len(ids) for pool, ids in pools.items()}
            for map_name, pools in all_record_pools.items()
        },
        "selected_counts": manifest["counts"],
        "continuous_protocol": manifest["continuous_protocol"],
    }

    artifacts: dict[str, bytes] = {
        "benchmark_manifest.json": json_bytes(manifest, ctx.indent),
        "build_report.json": json_bytes(build_report, ctx.indent),
        "selected_images.sha256": selected_image_data,
    }
    for map_name in ctx.config["maps"]["seen"]:
        result = map_results[map_name]
        prefix = f"splits/seen/{map_name}"
        for split_name in DISCRETE_SPLITS:
            artifacts[f"{prefix}/{split_name}.json"] = _split_file_bytes(
                [copy.deepcopy(row.raw) for row in result[split_name]], ctx.indent
            )
        artifacts[f"{prefix}/continuous_clips.json"] = _split_file_bytes(
            _continuous_json(map_name, result["continuous"], ctx.config), ctx.indent
        )
    for map_name in ctx.config["maps"]["crossmap"]:
        result = map_results[map_name]
        prefix = f"splits/crossmap/{map_name}"
        for support_seed in support_seeds:
            artifacts[f"{prefix}/support_seed_{support_seed}.json"] = _split_file_bytes(
                [
                    copy.deepcopy(row.raw)
                    for row in result[f"support_seed_{support_seed}"]
                ],
                ctx.indent,
            )
        artifacts[f"{prefix}/query_test.json"] = _split_file_bytes(
            [copy.deepcopy(row.raw) for row in result["query_test"]], ctx.indent
        )
        artifacts[f"{prefix}/continuous_clips.json"] = _split_file_bytes(
            _continuous_json(map_name, result["continuous"], ctx.config), ctx.indent
        )

    aggregate_seen = {
        "seen_train": "train",
        "seen_validation": "validation",
        "seen_discrete_test": "discrete_test",
    }
    for aggregate_name, split_name in aggregate_seen.items():
        rows = [
            copy.deepcopy(row.raw)
            for map_name in ctx.config["maps"]["seen"]
            for row in map_results[map_name][split_name]
        ]
        artifacts[f"aggregate/{aggregate_name}.json"] = _split_file_bytes(
            rows, ctx.indent
        )
    query_rows = [
        copy.deepcopy(row.raw)
        for map_name in ctx.config["maps"]["crossmap"]
        for row in map_results[map_name]["query_test"]
    ]
    artifacts["aggregate/crossmap_query_test.json"] = _split_file_bytes(
        query_rows, ctx.indent
    )
    for support_seed in support_seeds:
        support_rows = [
            copy.deepcopy(row.raw)
            for map_name in ctx.config["maps"]["crossmap"]
            for row in map_results[map_name][f"support_seed_{support_seed}"]
        ]
        artifacts[f"aggregate/crossmap_support_seed_{support_seed}.json"] = (
            _split_file_bytes(support_rows, ctx.indent)
        )

    checksum_paths = set(_relative_artifact_paths(ctx.config))
    generated_paths = checksum_paths - set(CALIBRATION_ARTIFACTS)
    if set(artifacts) != generated_paths:
        missing = sorted(generated_paths - set(artifacts))
        extra = sorted(set(artifacts) - generated_paths)
        raise BenchmarkError(
            f"internal artifact layout mismatch; missing={missing}, extra={extra}"
        )
    checksum_data: dict[str, bytes] = dict(artifacts)
    for relative in CALIBRATION_ARTIFACTS:
        calibration_path = ctx.output_root / relative
        if not calibration_path.is_file():
            raise BenchmarkError(
                f"missing calibration artifact for checksums: {calibration_path}"
            )
        checksum_data[relative] = calibration_path.read_bytes()
    checksum_lines = [
        f"{sha256_bytes(checksum_data[relative])}  {relative}"
        for relative in sorted(checksum_paths)
    ]
    artifacts["checksums.sha256"] = ("\n".join(checksum_lines) + "\n").encode("utf-8")
    _write_artifacts(ctx.output_root, artifacts, overwrite)
    return manifest


def build(
    config_path: str | os.PathLike[str],
    decisions_path: str | os.PathLike[str],
    calibration_approval_path: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Public alias for :func:`run_build`."""

    return run_build(
        config_path,
        decisions_path,
        calibration_approval_path=calibration_approval_path,
        overwrite=overwrite,
    )


def _read_json_file(path: Path, description: str) -> Any:
    if not path.is_file():
        raise BenchmarkError(f"missing {description}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"could not read {description} {path}: {exc}") from exc


def _source_row_for_output(
    row: Any,
    map_name: str,
    source_lookup: Mapping[tuple[str, str], SourceRow],
    seen: set[tuple[str, str]],
) -> SourceRow:
    if not isinstance(row, Mapping):
        raise BenchmarkError(f"{map_name} manifest contains a non-object source row")
    if row.get("map") != map_name:
        raise BenchmarkError(
            f"source row map label mismatch: expected {map_name}, got {row.get('map')!r}"
        )
    file_frame = row.get("file_frame")
    if not isinstance(file_frame, str):
        raise BenchmarkError(f"{map_name} manifest contains a non-string file_frame")
    key = (map_name, file_frame)
    if key in seen:
        raise BenchmarkError(
            f"duplicate file_frame in manifest: {map_name}:{file_frame}"
        )
    seen.add(key)
    source_row = source_lookup.get(key)
    if source_row is None:
        raise BenchmarkError(
            f"manifest row is not present in source: {map_name}:{file_frame}"
        )
    if source_row.integrity_invalid:
        raise BenchmarkError(
            f"manifest includes an integrity-invalid row: {map_name}:{file_frame}"
        )
    if row != source_row.raw:
        raise BenchmarkError(
            f"manifest row differs from source: {map_name}:{file_frame}"
        )
    return source_row


def _verify_discrete_rows(
    values: Any,
    map_name: str,
    source_lookup: Mapping[tuple[str, str], SourceRow],
    record_regex: re.Pattern[str],
    min_frame_gap: int,
) -> tuple[list[SourceRow], set[str]]:
    if not isinstance(values, list):
        raise BenchmarkError(f"{map_name} discrete split must be a JSON list")
    source_rows: list[SourceRow] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        source_rows.append(_source_row_for_output(value, map_name, source_lookup, seen))
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in source_rows:
        if row.record_id is None or row.frame_id is None:
            raise BenchmarkError(
                f"manifest row has an unparseable file_frame: {row.file_frame}"
            )
        grouped[row.record_id].append(row.frame_id)
    for record_id, frames in grouped.items():
        ordered = sorted(frames)
        if len(ordered) != len(set(ordered)):
            raise BenchmarkError(
                f"duplicate frame id in manifest record {map_name}:{record_id}"
            )
        if any(
            current - previous < min_frame_gap
            for previous, current in zip(ordered, ordered[1:])
        ):
            raise BenchmarkError(
                f"temporal min_frame_gap violated in {map_name}:{record_id}"
            )
    return source_rows, set(grouped)


def _verify_continuous_payload(
    payload: Any,
    map_name: str,
    expected_clip_count: int,
    frames_per_clip: int,
    max_frame_gap: int,
    max_clips_per_record: int,
    source_lookup: Mapping[tuple[str, str], SourceRow],
    record_regex: re.Pattern[str],
) -> tuple[list[list[SourceRow]], set[str]]:
    if not isinstance(payload, Mapping):
        raise BenchmarkError(f"{map_name} continuous_clips.json must be an object")
    if payload.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise BenchmarkError(f"continuous schema version mismatch for {map_name}")
    if payload.get("benchmark_id") != EXPECTED_BENCHMARK_ID:
        raise BenchmarkError(f"continuous benchmark id mismatch for {map_name}")
    if payload.get("map") != map_name or payload.get("split") != "continuous":
        raise BenchmarkError(f"continuous manifest metadata mismatch for {map_name}")
    clips = payload.get("clips")
    if not isinstance(clips, list) or len(clips) != expected_clip_count:
        raise BenchmarkError(
            f"continuous clip count mismatch for {map_name}: expected {expected_clip_count}"
        )
    if payload.get("frames_per_clip") != frames_per_clip:
        raise BenchmarkError(f"continuous frames_per_clip mismatch for {map_name}")
    seen_clip_ids: set[str] = set()
    seen_frames: set[tuple[str, str]] = set()
    record_counts: Counter[str] = Counter()
    result: list[list[SourceRow]] = []
    for clip in clips:
        if not isinstance(clip, Mapping):
            raise BenchmarkError(f"continuous clip for {map_name} is not an object")
        clip_id = clip.get("clip_id")
        if not isinstance(clip_id, str) or clip_id in seen_clip_ids:
            raise BenchmarkError(
                f"duplicate or invalid continuous clip_id for {map_name}"
            )
        seen_clip_ids.add(clip_id)
        if clip.get("map") != map_name:
            raise BenchmarkError(f"continuous clip map mismatch for {map_name}")
        frames = clip.get("frames")
        if not isinstance(frames, list) or len(frames) != frames_per_clip:
            raise BenchmarkError(
                f"continuous clip length mismatch for {map_name}:{clip_id}"
            )
        clip_rows: list[SourceRow] = []
        local_seen: set[tuple[str, str]] = set()
        for frame in frames:
            row = _source_row_for_output(frame, map_name, source_lookup, local_seen)
            key = (map_name, row.file_frame or "")
            if key in seen_frames:
                raise BenchmarkError(
                    f"continuous clips overlap for {map_name}:{key[1]}"
                )
            seen_frames.add(key)
            clip_rows.append(row)
        record_id = clip.get("record_id")
        if not isinstance(record_id, str) or any(
            row.record_id != record_id for row in clip_rows
        ):
            raise BenchmarkError(
                f"continuous record_id mismatch for {map_name}:{clip_id}"
            )
        parsed_frames = [row.frame_id for row in clip_rows]
        if any(frame is None for frame in parsed_frames):
            raise BenchmarkError(
                f"continuous clip has an unparseable frame id: {map_name}:{clip_id}"
            )
        if any(
            current <= previous
            for previous, current in zip(parsed_frames, parsed_frames[1:])
        ):
            raise BenchmarkError(
                f"continuous clip is not strictly ordered: {map_name}:{clip_id}"
            )
        if any(
            not 1 <= current - previous <= max_frame_gap
            for previous, current in zip(parsed_frames, parsed_frames[1:])
        ):
            raise BenchmarkError(f"continuous frame gaps invalid: {map_name}:{clip_id}")
        record_counts[record_id] += 1
        result.append(clip_rows)
    if any(count > max_clips_per_record for count in record_counts.values()):
        raise BenchmarkError(f"continuous max_clips_per_record violated for {map_name}")
    return result, set(record_counts)


def _verify_checksum_file(output_root: Path, expected_paths: Sequence[str]) -> None:
    checksum_path = output_root / "checksums.sha256"
    if not checksum_path.is_file():
        raise BenchmarkError(f"missing checksum file: {checksum_path}")
    entries: dict[str, str] = {}
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise BenchmarkError(
            f"could not read checksum file: {checksum_path}: {exc}"
        ) from exc
    for line in lines:
        if not line.strip():
            continue
        if "  " not in line:
            raise BenchmarkError(f"invalid checksum line: {line!r}")
        digest, relative = line.split("  ", 1)
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or not relative:
            raise BenchmarkError(f"invalid checksum entry: {line!r}")
        if relative in entries:
            raise BenchmarkError(f"duplicate checksum entry: {relative}")
        entries[relative] = digest
    expected = set(expected_paths)
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        extra = sorted(set(entries) - expected)
        raise BenchmarkError(
            f"checksum file path set mismatch; missing={missing}, extra={extra}"
        )
    for relative, expected_digest in entries.items():
        path = output_root / relative
        if not path.is_file():
            raise BenchmarkError(f"checksum target is missing: {path}")
        actual = sha256_file(path)
        if actual != expected_digest:
            raise BenchmarkError(f"checksum mismatch for {relative}")


def _check_manifest_count(actual: Any, expected: int, label: str) -> None:
    if actual != expected:
        raise BenchmarkError(
            f"count mismatch for {label}: expected {expected}, got {actual}"
        )


def run_validate(
    config_path: str | os.PathLike[str],
    decisions_path: str | os.PathLike[str] | None = None,
    calibration_approval_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    ctx = make_context(config_path)
    snapshot = _source_snapshot(ctx)
    audit_report, audit_hash, _ = _verify_audit_snapshot(ctx, snapshot)
    manifest = _read_json_file(
        ctx.output_root / "benchmark_manifest.json", "benchmark manifest"
    )
    build_report = _read_json_file(
        ctx.output_root / "build_report.json", "build report"
    )
    if not isinstance(manifest, Mapping) or not isinstance(build_report, Mapping):
        raise BenchmarkError("benchmark manifest and build report must be JSON objects")
    if manifest.get("benchmark_id") != ctx.config["benchmark"]["id"]:
        raise BenchmarkError("benchmark manifest id does not match config")
    if build_report.get("benchmark_id") != ctx.config["benchmark"]["id"]:
        raise BenchmarkError("build report id does not match config")
    if manifest.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise BenchmarkError("benchmark manifest schema version mismatch")
    if build_report.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise BenchmarkError("build report schema version mismatch")
    expected_benchmark = {
        "id": ctx.config["benchmark"]["id"],
        "version": ctx.config["benchmark"]["version"],
        "global_seed": ctx.config["benchmark"]["global_seed"],
        "strict_protocol": bool(ctx.config["benchmark"].get("strict_protocol", False)),
    }
    if manifest.get("benchmark") != expected_benchmark:
        raise BenchmarkError("benchmark manifest protocol metadata mismatch")
    expected_protocol = {
        "seen_maps": list(ctx.config["maps"]["seen"]),
        "crossmap_maps": list(ctx.config["maps"]["crossmap"]),
        "seen_splits": list(DISCRETE_SPLITS),
        "crossmap_splits": ["support", "query_test", "continuous"],
        "support_seeds": list(ctx.config["counts"]["crossmap"]["support_seeds"]),
    }
    if manifest.get("protocol") != expected_protocol:
        raise BenchmarkError("benchmark manifest map protocol mismatch")
    expected_seeds = {
        "global_seed": ctx.config["benchmark"]["global_seed"],
        "support_seeds": list(ctx.config["counts"]["crossmap"]["support_seeds"]),
    }
    if manifest.get("seeds") != expected_seeds:
        raise BenchmarkError("benchmark manifest seed metadata mismatch")
    expected_continuous_protocol = {
        "trajectory_disjoint": True,
        "pose_disjoint": False,
        "statement": "Continuous clips are trajectory-disjoint but are not guaranteed pose-disjoint.",
        "frames_per_clip": ctx.config["counts"]["continuous"]["frames_per_clip"],
        "max_frame_gap": ctx.config["counts"]["continuous"]["max_frame_gap"],
    }
    if manifest.get("continuous_protocol") != expected_continuous_protocol:
        raise BenchmarkError("continuous protocol metadata mismatch")
    config_hash = sha256_file(ctx.config_path)
    if (
        manifest.get("config_sha256") != config_hash
        or build_report.get("config_sha256") != config_hash
    ):
        raise BenchmarkError("build/config fingerprint mismatch")
    expected_source = {
        "radar_files": dict(ctx.config["source"]["radar_files"]),
        "positions_sha256": snapshot.positions_sha256,
        "radar_sha256": snapshot.radar_sha256,
        "source_sha256": snapshot.source_sha256,
    }
    if manifest.get("source") != {
        "root": str(ctx.config["paths"]["source_root"]),
        **expected_source,
    }:
        raise BenchmarkError("benchmark manifest source provenance mismatch")
    if build_report.get("source") != expected_source:
        raise BenchmarkError("build report source provenance mismatch")
    if (
        manifest.get("audit_report_sha256") != audit_hash
        or build_report.get("audit_report_sha256") != audit_hash
    ):
        raise BenchmarkError("build audit fingerprint mismatch")
    if audit_report.get("benchmark_id") != ctx.config["benchmark"]["id"]:
        raise BenchmarkError("audit benchmark id does not match config")
    if decisions_path is None:
        raise BenchmarkError(
            "validate requires --decisions to recompute full-corpus calibration"
        )
    decisions = load_decisions(
        decisions_path,
        ctx.config,
        expected_audit_report_sha256=audit_hash,
        expected_candidate_file_sha256=audit_report["output"]["candidate_file_sha256"],
        expected_candidate_file_line_count=audit_report["output"][
            "candidate_file_line_count"
        ],
    )
    _validate_decision_ids(decisions, snapshot)
    if decisions.status != "approved":
        raise BenchmarkError("validate --decisions requires review.status=approved")
    if decisions.default_action == "undecided":
        raise BenchmarkError(
            "validate --decisions requires coordinate_candidates.default_action=keep or exclude"
        )
    accepted_rows, _, _ = _apply_decisions(snapshot, ctx.config, decisions)
    accepted_keys: set[tuple[str, str]] = {
        (map_name, row.file_frame)
        for map_name, rows in accepted_rows.items()
        for row in rows
        if row.file_frame is not None
    }
    if decisions.raw_sha256 != manifest.get("decisions_sha256"):
        raise BenchmarkError("provided decisions hash does not match build manifest")
    if decisions.raw_sha256 != build_report.get("decisions_sha256"):
        raise BenchmarkError("provided decisions hash does not match build report")
    calibration = _verify_calibration(
        ctx,
        snapshot,
        audit_hash,
        audit_report,
        decisions,
        accepted_rows,
        calibration_approval_path,
    )
    expected_calibration = _calibration_metadata(calibration)
    if manifest.get("calibration") != expected_calibration:
        raise BenchmarkError("benchmark manifest calibration provenance mismatch")
    if build_report.get("calibration") != expected_calibration:
        raise BenchmarkError("build report calibration provenance mismatch")
    if manifest.get("calibration_approval_sha256") != calibration.approval_sha256:
        raise BenchmarkError(
            "benchmark manifest calibration approval fingerprint mismatch"
        )
    if build_report.get("calibration_approval_sha256") != calibration.approval_sha256:
        raise BenchmarkError("build report calibration approval fingerprint mismatch")

    expected_paths = _relative_artifact_paths(ctx.config)
    _verify_checksum_file(ctx.output_root, expected_paths)

    source_lookup: dict[tuple[str, str], SourceRow] = {}
    for map_name, rows in snapshot.rows_by_map.items():
        for row in rows:
            if row.file_frame is not None and not row.integrity_invalid:
                source_lookup[(map_name, row.file_frame)] = row
    record_regex = re.compile(str(ctx.config["source"]["record_regex"]))
    min_frame_gap = int(ctx.config["sampling"]["min_frame_gap"])
    frames_per_clip = int(ctx.config["counts"]["continuous"]["frames_per_clip"])
    max_frame_gap = int(ctx.config["counts"]["continuous"]["max_frame_gap"])
    max_clips_per_record = int(
        ctx.config["counts"]["continuous"]["max_clips_per_record"]
    )
    map_results: dict[str, dict[str, Any]] = {}
    seen_pool_records: dict[str, dict[str, set[str]]] = {}
    cross_support_records: dict[str, dict[int, set[str]]] = {}
    selected_image_rows: list[SourceRow] = []

    for map_name in ctx.config["maps"]["seen"]:
        prefix = ctx.output_root / "splits" / "seen" / map_name
        results: dict[str, Any] = {}
        records: dict[str, set[str]] = {}
        for split_name in DISCRETE_SPLITS:
            values = _read_json_file(
                prefix / f"{split_name}.json", f"seen {split_name}"
            )
            rows, row_records = _verify_discrete_rows(
                values, map_name, source_lookup, record_regex, min_frame_gap
            )
            _check_manifest_count(
                len(rows),
                int(ctx.config["counts"]["seen"][split_name]),
                f"{map_name}:{split_name}",
            )
            results[split_name] = values
            records[split_name] = row_records
            selected_image_rows.extend(rows)
        continuous = _read_json_file(
            prefix / "continuous_clips.json", "seen continuous clips"
        )
        clips, clip_records = _verify_continuous_payload(
            continuous,
            map_name,
            int(ctx.config["counts"]["continuous"]["clips_per_map"]),
            frames_per_clip,
            max_frame_gap,
            max_clips_per_record,
            source_lookup,
            record_regex,
        )
        results["continuous"] = continuous
        results["continuous_rows"] = clips
        records["continuous"] = clip_records
        for clip in clips:
            selected_image_rows.extend(clip)
        pool_names = list(DISCRETE_SPLITS) + ["continuous"]
        for left_index, left in enumerate(pool_names):
            for right in pool_names[left_index + 1 :]:
                if records[left] & records[right]:
                    raise BenchmarkError(
                        f"seen record overlap for {map_name}: {left} and {right}"
                    )
        seen_pool_records[map_name] = records
        map_results[map_name] = results

    for map_name in ctx.config["maps"]["crossmap"]:
        prefix = ctx.output_root / "splits" / "crossmap" / map_name
        results = {}
        records: dict[str, set[str]] = {}
        support_records: dict[int, set[str]] = {}
        for support_seed in ctx.config["counts"]["crossmap"]["support_seeds"]:
            values = _read_json_file(
                prefix / f"support_seed_{support_seed}.json",
                f"crossmap support seed {support_seed}",
            )
            rows, row_records = _verify_discrete_rows(
                values, map_name, source_lookup, record_regex, min_frame_gap
            )
            _check_manifest_count(
                len(rows),
                int(ctx.config["counts"]["crossmap"]["support"]),
                f"{map_name}:support_seed_{support_seed}",
            )
            results[f"support_seed_{support_seed}"] = values
            support_records[int(support_seed)] = row_records
            selected_image_rows.extend(rows)
        query_values = _read_json_file(
            prefix / "query_test.json", "crossmap query test"
        )
        query_rows, query_record_ids = _verify_discrete_rows(
            query_values, map_name, source_lookup, record_regex, min_frame_gap
        )
        _check_manifest_count(
            len(query_rows),
            int(ctx.config["counts"]["crossmap"]["query_test"]),
            f"{map_name}:query_test",
        )
        results["query_test"] = query_values
        records["query_test"] = query_record_ids
        selected_image_rows.extend(query_rows)
        continuous = _read_json_file(
            prefix / "continuous_clips.json", "crossmap continuous clips"
        )
        clips, clip_records = _verify_continuous_payload(
            continuous,
            map_name,
            int(ctx.config["counts"]["continuous"]["clips_per_map"]),
            frames_per_clip,
            max_frame_gap,
            max_clips_per_record,
            source_lookup,
            record_regex,
        )
        results["continuous"] = continuous
        results["continuous_rows"] = clips
        records["continuous"] = clip_records
        for clip in clips:
            selected_image_rows.extend(clip)
        for support_seed, support_record_ids in support_records.items():
            if (
                support_record_ids & records["query_test"]
                or support_record_ids & records["continuous"]
            ):
                raise BenchmarkError(
                    f"crossmap support record overlap for {map_name}: seed {support_seed}"
                )
        if records["query_test"] & records["continuous"]:
            raise BenchmarkError(
                f"crossmap query/continuous record overlap for {map_name}"
            )
        support_union_rows: list[SourceRow] = []
        for support_seed in ctx.config["counts"]["crossmap"]["support_seeds"]:
            support_values = results[f"support_seed_{support_seed}"]
            support_rows, _ = _verify_discrete_rows(
                support_values, map_name, source_lookup, record_regex, min_frame_gap
            )
            support_union_rows.extend(support_rows)
        near_pose = ctx.config["sampling"]["near_pose_filter"]
        if near_pose.get("enabled", False):
            non_near = near_pose_filter(
                query_rows,
                support_union_rows,
                float(near_pose["xy_tolerance"]),
                float(near_pose["z_tolerance"]),
                float(near_pose["angle_tolerance_degrees"]),
            )
            if len(non_near) != len(query_rows):
                raise BenchmarkError(
                    f"crossmap query is near-pose to support for {map_name}"
                )
        records_for_manifest = dict(records)
        for support_seed, support_record_ids in support_records.items():
            records_for_manifest[f"support_seed_{support_seed}"] = support_record_ids
        cross_support_records[map_name] = support_records
        seen_pool_records[map_name] = records_for_manifest
        map_results[map_name] = results

    if accepted_keys is not None:

        def verify_accepted_output_row(map_name: str, row: Any, location: str) -> None:
            if not isinstance(row, Mapping) or not isinstance(
                row.get("file_frame"), str
            ):
                raise BenchmarkError(
                    f"invalid output row while checking decisions: {location}"
                )
            if (map_name, row["file_frame"]) not in accepted_keys:
                raise BenchmarkError(
                    f"{location} contains a row outside the approved accepted set: "
                    f"{map_name}:{row['file_frame']}"
                )

        for map_name in ctx.config["maps"]["seen"]:
            result = map_results[map_name]
            for split_name in DISCRETE_SPLITS:
                for row in result[split_name]:
                    verify_accepted_output_row(
                        map_name, row, f"seen/{map_name}/{split_name}"
                    )
            for clip in result["continuous"]["clips"]:
                for row in clip["frames"]:
                    verify_accepted_output_row(
                        map_name, row, f"seen/{map_name}/continuous"
                    )
        for map_name in ctx.config["maps"]["crossmap"]:
            result = map_results[map_name]
            for support_seed in ctx.config["counts"]["crossmap"]["support_seeds"]:
                for row in result[f"support_seed_{support_seed}"]:
                    verify_accepted_output_row(
                        map_name,
                        row,
                        f"crossmap/{map_name}/support_seed_{support_seed}",
                    )
            for row in result["query_test"]:
                verify_accepted_output_row(
                    map_name, row, f"crossmap/{map_name}/query_test"
                )
            for clip in result["continuous"]["clips"]:
                for row in clip["frames"]:
                    verify_accepted_output_row(
                        map_name, row, f"crossmap/{map_name}/continuous"
                    )

    selected_image_data, selected_image_count = _selected_image_manifest_bytes(
        ctx, selected_image_rows
    )
    selected_images_path = ctx.output_root / "selected_images.sha256"
    try:
        actual_selected_image_data = selected_images_path.read_bytes()
    except (OSError, UnicodeDecodeError) as exc:
        raise BenchmarkError(
            f"could not read selected image checksum artifact: {selected_images_path}: {exc}"
        ) from exc
    if actual_selected_image_data != selected_image_data:
        raise BenchmarkError(
            "selected_images.sha256 does not match the selected output image paths or current image digests"
        )
    selected_images_metadata = {
        "file": "selected_images.sha256",
        "sha256": sha256_bytes(actual_selected_image_data),
        "count": selected_image_count,
    }
    if manifest.get("selected_images") != selected_images_metadata:
        raise BenchmarkError("benchmark manifest selected image provenance mismatch")
    if build_report.get("selected_images") != selected_images_metadata:
        raise BenchmarkError("build report selected image provenance mismatch")

    actual_counts: dict[str, Any] = {"seen": {}, "crossmap": {}}
    for map_name in ctx.config["maps"]["seen"]:
        result = map_results[map_name]
        actual_counts["seen"][map_name] = {
            "train": len(result["train"]),
            "validation": len(result["validation"]),
            "discrete_test": len(result["discrete_test"]),
            "continuous_clips": len(result["continuous"]["clips"]),
            "continuous_frames": sum(
                len(clip["frames"]) for clip in result["continuous"]["clips"]
            ),
        }
    for map_name in ctx.config["maps"]["crossmap"]:
        result = map_results[map_name]
        actual_counts["crossmap"][map_name] = {
            "query_test": len(result["query_test"]),
            "continuous_clips": len(result["continuous"]["clips"]),
            "continuous_frames": sum(
                len(clip["frames"]) for clip in result["continuous"]["clips"]
            ),
            "support": {
                str(seed): len(result[f"support_seed_{seed}"])
                for seed in ctx.config["counts"]["crossmap"]["support_seeds"]
            },
        }
    if manifest.get("counts") != actual_counts:
        raise BenchmarkError("benchmark manifest counts do not match split artifacts")
    if build_report.get("selected_counts") != actual_counts:
        raise BenchmarkError("build report counts do not match split artifacts")

    manifest_pools = manifest.get("record_pools")
    if not isinstance(manifest_pools, Mapping):
        raise BenchmarkError("benchmark manifest is missing record_pools")
    for map_name in list(ctx.config["maps"]["seen"]) + list(
        ctx.config["maps"]["crossmap"]
    ):
        pool_values = manifest_pools.get(map_name)
        if not isinstance(pool_values, Mapping):
            raise BenchmarkError(
                f"benchmark manifest is missing record pools for {map_name}"
            )
        expected_pool_names = (
            SEEN_POOL_ORDER
            if map_name in ctx.config["maps"]["seen"]
            else CROSSMAP_POOL_ORDER
        )
        if set(pool_values) != set(expected_pool_names):
            raise BenchmarkError(f"record pool names mismatch for {map_name}")
        pool_sets: dict[str, set[str]] = {}
        for pool_name in expected_pool_names:
            values = pool_values.get(pool_name)
            if not isinstance(values, list) or any(
                not isinstance(item, str) for item in values
            ):
                raise BenchmarkError(
                    f"record pool ids are invalid for {map_name}:{pool_name}"
                )
            pool_sets[pool_name] = set(values)
        if len(set().union(*pool_sets.values())) != sum(
            len(values) for values in pool_sets.values()
        ):
            raise BenchmarkError(
                f"record pools overlap in benchmark manifest for {map_name}"
            )
        output_record_sets = seen_pool_records[map_name]
        if map_name in ctx.config["maps"]["seen"]:
            output_to_pool = {
                "train": "train",
                "validation": "validation",
                "discrete_test": "discrete_test",
                "continuous": "continuous",
            }
        else:
            output_to_pool = {
                "query_test": "query_test",
                "continuous": "continuous",
            }
            for seed in ctx.config["counts"]["crossmap"]["support_seeds"]:
                output_to_pool[f"support_seed_{seed}"] = "support"
        for output_name, pool_name in output_to_pool.items():
            if not output_record_sets.get(output_name, set()) <= pool_sets[pool_name]:
                raise BenchmarkError(
                    f"split records are outside declared pool for {map_name}:{output_name}"
                )

    for map_name in ctx.config["maps"]["seen"]:
        train_rows, _ = _verify_discrete_rows(
            map_results[map_name]["train"],
            map_name,
            source_lookup,
            record_regex,
            min_frame_gap,
        )
        validation_rows, _ = _verify_discrete_rows(
            map_results[map_name]["validation"],
            map_name,
            source_lookup,
            record_regex,
            min_frame_gap,
        )
        test_rows, _ = _verify_discrete_rows(
            map_results[map_name]["discrete_test"],
            map_name,
            source_lookup,
            record_regex,
            min_frame_gap,
        )
        near_pose = ctx.config["sampling"]["near_pose_filter"]
        if near_pose.get("enabled", False):
            non_near = near_pose_filter(
                test_rows,
                train_rows + validation_rows,
                float(near_pose["xy_tolerance"]),
                float(near_pose["z_tolerance"]),
                float(near_pose["angle_tolerance_degrees"]),
            )
            if len(non_near) != len(test_rows):
                raise BenchmarkError(
                    f"seen discrete_test is near-pose to train/validation for {map_name}"
                )

    aggregate_specs = {
        "aggregate/seen_train.json": [
            map_results[map_name]["train"] for map_name in ctx.config["maps"]["seen"]
        ],
        "aggregate/seen_validation.json": [
            map_results[map_name]["validation"]
            for map_name in ctx.config["maps"]["seen"]
        ],
        "aggregate/seen_discrete_test.json": [
            map_results[map_name]["discrete_test"]
            for map_name in ctx.config["maps"]["seen"]
        ],
        "aggregate/crossmap_query_test.json": [
            map_results[map_name]["query_test"]
            for map_name in ctx.config["maps"]["crossmap"]
        ],
    }
    for support_seed in ctx.config["counts"]["crossmap"]["support_seeds"]:
        aggregate_specs[f"aggregate/crossmap_support_seed_{support_seed}.json"] = [
            map_results[map_name][f"support_seed_{support_seed}"]
            for map_name in ctx.config["maps"]["crossmap"]
        ]
    for relative, pieces in aggregate_specs.items():
        actual = _read_json_file(ctx.output_root / relative, relative)
        expected: list[Any] = []
        for piece in pieces:
            expected.extend(piece)
        if actual != expected:
            raise BenchmarkError(
                f"aggregate does not equal per-map concatenation: {relative}"
            )

    summary = {
        "seen_maps": len(ctx.config["maps"]["seen"]),
        "crossmap_maps": len(ctx.config["maps"]["crossmap"]),
        "audit_report_sha256": audit_hash,
        "calibration_sha256": calibration.calibration_sha256,
        "status": "valid",
    }
    return summary


def validate(
    config_path: str | os.PathLike[str],
    decisions_path: str | os.PathLike[str] | None = None,
    calibration_approval_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Public alias for :func:`run_validate`."""

    return run_validate(
        config_path,
        decisions_path=decisions_path,
        calibration_approval_path=calibration_approval_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit, calibrate, build, and validate deterministic CS2 "
            "benchmark-v2 splits."
        ),
        epilog=(
            "Relative CONFIG paths and relative paths in paths.source_root and "
            "paths.output_root resolve from the current working directory "
            "(normally the repository root). Generated artifacts are "
            "written atomically; --overwrite is required to replace known artifacts."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser(
        "audit", help="inspect source rows and write audit artifacts"
    )
    audit_parser.add_argument("--config", required=True, help="benchmark YAML config")
    audit_parser.add_argument(
        "--overwrite", action="store_true", help="replace known audit artifacts"
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="freeze exact per-map Z calibration from approved full-corpus rows",
    )
    calibrate_parser.add_argument(
        "--config", required=True, help="benchmark YAML config"
    )
    calibrate_parser.add_argument(
        "--decisions", required=True, help="approved decisions YAML"
    )
    calibrate_parser.add_argument(
        "--overwrite", action="store_true", help="replace known calibration artifacts"
    )

    build_parser_command = subparsers.add_parser(
        "build", help="build approved benchmark manifests"
    )
    build_parser_command.add_argument(
        "--config", required=True, help="benchmark YAML config"
    )
    build_parser_command.add_argument(
        "--decisions", required=True, help="approved decisions YAML"
    )
    build_parser_command.add_argument(
        "--calibration-approval", required=True, help="approved calibration YAML"
    )
    build_parser_command.add_argument(
        "--overwrite", action="store_true", help="replace known build artifacts"
    )

    validate_parser = subparsers.add_parser(
        "validate", help="independently validate build artifacts"
    )
    validate_parser.add_argument(
        "--config", required=True, help="benchmark YAML config"
    )
    validate_parser.add_argument(
        "--decisions", required=True, help="approved decisions YAML"
    )
    validate_parser.add_argument(
        "--calibration-approval", required=True, help="approved calibration YAML"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            report = run_audit(args.config, overwrite=args.overwrite)
            print(
                "AUDIT OK: "
                f"{report['counts']['candidate_rows']} candidate rows; "
                f"audit_report_sha256={report['audit_report_sha256']}"
            )
        elif args.command == "calibrate":
            calibration = run_calibrate(
                args.config, args.decisions, overwrite=args.overwrite
            )
            print(
                f"CALIBRATE OK: calibration_sha256={calibration['calibration_sha256']}"
            )
        elif args.command == "build":
            manifest = run_build(
                args.config,
                args.decisions,
                calibration_approval_path=args.calibration_approval,
                overwrite=args.overwrite,
            )
            print(
                "BUILD OK: "
                f"{manifest['benchmark_id']} audit_report_sha256="
                f"{manifest['audit_report_sha256']} calibration_sha256="
                f"{manifest['calibration']['fingerprint']}"
            )
        elif args.command == "validate":
            summary = run_validate(
                args.config,
                decisions_path=args.decisions,
                calibration_approval_path=args.calibration_approval,
            )
            print(
                "VALIDATE OK: "
                f"{summary['seen_maps']} seen maps, "
                f"{summary['crossmap_maps']} crossmap maps"
            )
        else:  # pragma: no cover - argparse enforces the command choices
            parser.error(f"unknown command: {args.command}")
    except (BenchmarkError, OSError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests
    raise SystemExit(main())
