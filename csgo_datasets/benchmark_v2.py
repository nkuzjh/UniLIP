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
import string
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
    # ``asset_backend`` is ``source`` for the historical layout and
    # ``minimal`` for the self-contained flat bundle.  Keeping this explicit
    # avoids silently selecting a layout based on which directories happen to
    # exist on a machine.
    asset_backend: str = "source"
    asset_manifest_path: Path | None = None
    asset_manifest_sha256: str | None = None
    benchmark_manifest_sha256: str | None = None
    selected_images_sha256: str | None = None
    asset_root: str | None = None
    image_dirs: dict[str, Path] | None = None
    asset_provenance: dict[str, Any] | None = None

    def benchmark_v2_image_dir(self, map_name: str) -> Path:
        """Return the directory containing v2 FPV frames for ``map_name``."""

        return benchmark_v2_image_dir(self, map_name)

    def benchmark_v2_image_path(self, map_name: str, file_frame: str) -> Path:
        """Return the v2 FPV frame path for ``map_name`` and ``file_frame``."""

        return benchmark_v2_image_path(self, map_name, file_frame)


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

    Recognized fields are ``benchmark_v2_manifest``,
    ``benchmark_v2_asset_manifest``, ``benchmark_v2_split``,
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
        asset_manifest_path=_config_value(config, "benchmark_v2_asset_manifest"),
    )
    extension = _normalise_image_extension(
        _config_value(config, "benchmark_v2_image_extension")
    )
    if extension is not None:
        if (
            selection.asset_backend == "minimal"
            and extension != selection.image_extension
        ):
            raise BenchmarkV2Error(
                "benchmark_v2_image_extension does not match the extension in "
                "benchmark_v2_asset_manifest"
            )
        selection.image_extension = extension
    return selection


def load_benchmark_v2_selection_from_args(
    manifest_path: str | os.PathLike[str],
    split: str,
    map_names: Sequence[str] | str | None = None,
    support_seed: int = 0,
    shots_per_map: int = 100,
    data_dir: str | os.PathLike[str] | None = None,
    asset_manifest_path: str | os.PathLike[str] | None = None,
) -> BenchmarkV2Selection:
    """Load and validate one Benchmark v2 split without ML dependencies.

    ``data_dir`` overrides ``manifest.source.root`` when the historical source
    backend is selected.  Relative paths follow
    the repository working directory convention used by the builder; if that
    path does not exist, a relative source path is also tried below the
    manifest directory, which is useful for self-contained test bundles.
    When ``asset_manifest_path`` is supplied, the validated minimal bundle is
    used instead and ``data_dir``/``manifest.source.root`` are not accessed.
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
    asset_manifest_path_resolved = None
    asset_manifest_sha256 = None
    benchmark_manifest_sha256 = _sha256_file(manifest_path)
    selected_images_sha256 = None
    asset_backend = "source"
    asset_provenance = None
    if asset_manifest_path not in (None, ""):
        asset_backend = "minimal"
        asset_manifest_path_resolved = _resolve_asset_manifest_path(
            asset_manifest_path
        )
        asset_manifest = _load_json_object(
            asset_manifest_path_resolved, "Benchmark v2 asset manifest"
        )
        (
            asset_root,
            image_dirs,
            radar_paths,
            image_extension,
            asset_provenance,
        ) = _resolve_minimal_assets(
            asset_manifest,
            asset_manifest_path_resolved,
            manifest_path,
            benchmark_manifest_sha256,
            protocol["all_maps"],
            source["radar_files"],
            manifest.get("radar_sha256", source.get("radar_sha256")),
            manifest.get("selected_images"),
        )
        source_root = asset_root
        asset_manifest_sha256 = _sha256_file(asset_manifest_path_resolved)
        selected_images_sha256 = asset_provenance["selected_images_sha256"]
    else:
        source_root_value = data_dir if data_dir is not None else source["root"]
        source_root = _resolve_source_root(source_root_value, manifest_path)
        radar_paths = _resolve_radar_paths(
            source["radar_files"], source_root, requested_maps
        )
        image_extension = ".jpg"
        image_dirs = {
            name: (source_root / name / "imgs").resolve()
            for name in requested_maps
        }
    z_ranges = _validate_z_ranges(manifest, protocol["all_maps"])
    selected_z_ranges = {name: z_ranges[name] for name in requested_maps}
    if asset_backend == "minimal":
        radar_paths = {name: radar_paths[name] for name in requested_maps}
        image_dirs = {name: image_dirs[name] for name in requested_maps}

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
        asset_backend=asset_backend,
        asset_manifest_path=asset_manifest_path_resolved,
        asset_manifest_sha256=asset_manifest_sha256,
        benchmark_manifest_sha256=benchmark_manifest_sha256,
        selected_images_sha256=selected_images_sha256,
        asset_root=(asset_provenance.get("asset_root") if asset_provenance else None),
        image_dirs=image_dirs,
        asset_provenance=asset_provenance,
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


def _resolve_asset_manifest_path(value: str | os.PathLike[str]) -> Path:
    """Resolve the explicit flat-bundle report without consulting source data."""

    try:
        path = Path(value).expanduser()
    except (TypeError, ValueError) as exc:
        raise BenchmarkV2Error(
            f"invalid benchmark_v2_asset_manifest path: {value!r}"
        ) from exc
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Benchmark v2 asset manifest does not exist: {path}"
        )
    return path


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkV2Error(f"could not read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkV2Error(f"{description} must be a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BenchmarkV2Error(f"could not hash file {path}: {exc}") from exc
    return digest.hexdigest()


def _require_sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise BenchmarkV2Error(
            f"{field_name} must be a lowercase SHA-256 digest"
        )
    return value


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise BenchmarkV2Error(f"{field_name} must be a non-negative integer")
    return value


def _count_selected_image_checksums(path: Path) -> int:
    count = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                line = raw_line.rstrip("\r\n")
                if not line:
                    raise BenchmarkV2Error(
                        "selected image checksum file contains an empty line at "
                        f"line {line_number}"
                    )
                fields = line.split(maxsplit=1)
                if len(fields) != 2:
                    raise BenchmarkV2Error(
                        "selected image checksum file has an invalid line at "
                        f"line {line_number}"
                    )
                _require_sha256(
                    fields[0],
                    f"selected image checksum line {line_number} digest",
                )
                _safe_relative_path(
                    fields[1],
                    f"selected image checksum line {line_number} path",
                )
                count += 1
    except OSError as exc:
        raise BenchmarkV2Error(
            f"could not read selected image checksum file {path}: {exc}"
        ) from exc
    return count


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


def _safe_relative_path(value: Any, field_name: str) -> Path:
    """Validate a bundle-relative path before joining it to the bundle root."""

    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value):
        raise BenchmarkV2Error(f"{field_name} must be a non-empty relative path")
    raw_value = os.fspath(value)
    if not isinstance(raw_value, str):
        raise BenchmarkV2Error(f"{field_name} must be a text relative path")
    if "\\" in raw_value:
        raise BenchmarkV2Error(
            f"{field_name} must use POSIX-style relative paths, got {value!r}"
        )
    path = Path(raw_value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise BenchmarkV2Error(
            f"{field_name} must be a safe relative path, got {value!r}"
        )
    return path


def _safe_asset_path(asset_root: Path, relative: Path, field_name: str) -> Path:
    """Resolve a bundle target and reject traversal or symlink escapes."""

    path = (asset_root / relative).resolve()
    try:
        path.relative_to(asset_root)
    except ValueError as exc:
        raise BenchmarkV2Error(
            f"{field_name} escapes the benchmark v2 asset bundle: {relative}"
        ) from exc
    return path


def _validate_asset_template(template: Any, field_name: str) -> str:
    if not isinstance(template, str) or not template:
        raise BenchmarkV2Error(f"{field_name} must be a non-empty path template")
    if Path(template).is_absolute():
        raise BenchmarkV2Error(f"{field_name} must be relative")
    fields: list[str] = []
    try:
        for _, field_name_value, format_spec, conversion in string.Formatter().parse(
            template
        ):
            if field_name_value is None:
                continue
            if field_name_value not in {"map", "file_frame"}:
                raise BenchmarkV2Error(
                    f"{field_name} may only use {{map}} and {{file_frame}}"
                )
            if format_spec or conversion:
                raise BenchmarkV2Error(
                    f"{field_name} does not allow format specs or conversions"
                )
            fields.append(field_name_value)
    except ValueError as exc:
        raise BenchmarkV2Error(f"invalid {field_name}: {template!r}") from exc
    if fields.count("map") != 1 or fields.count("file_frame") != 1:
        raise BenchmarkV2Error(
            f"{field_name} must contain exactly one {{map}} and one {{file_frame}}"
        )
    return template


def _resolve_minimal_assets(
    asset_manifest: Mapping[str, Any],
    asset_manifest_path: Path,
    benchmark_manifest_path: Path,
    benchmark_manifest_sha256: str,
    manifest_maps: Sequence[str],
    manifest_radar_files: Mapping[str, Any],
    manifest_radar_sha256: Any,
    manifest_selected_images: Any,
) -> tuple[
    Path,
    dict[str, Path],
    dict[str, Path],
    str,
    dict[str, Any],
]:
    """Validate and resolve the copied flat Benchmark v2 bundle.

    The materializer's ``minimal_dataset_report.json`` is intentionally used
    as the asset contract.  It binds the bundle to the exact benchmark
    manifest and checksum-list file, while its target paths remain portable
    relative to the report's directory on another server.
    """

    expected_maps = list(manifest_maps)
    if asset_manifest.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise BenchmarkV2Error(
            f"asset manifest schema_version must be {EXPECTED_SCHEMA_VERSION}"
        )
    if asset_manifest.get("benchmark_id") != EXPECTED_BENCHMARK_ID:
        raise BenchmarkV2Error(
            f"asset manifest benchmark_id must be {EXPECTED_BENCHMARK_ID!r}"
        )
    if asset_manifest.get("status") != "verified":
        raise BenchmarkV2Error("asset manifest status must be 'verified'")
    if asset_manifest.get("copy_mode") != "copyfile_not_move":
        raise BenchmarkV2Error(
            "asset manifest.copy_mode must be 'copyfile_not_move'"
        )
    invariants = asset_manifest.get("invariants")
    if not isinstance(invariants, Mapping):
        raise BenchmarkV2Error("asset manifest.invariants must be an object")
    required_invariants = {
        "radars_are_separate_from_images": True,
        "source_exists_after_copy": True,
        "source_target_samefile": False,
        "split_derived_set_equals_selected_images": True,
        "target_contains_only_declared_maps_and_files": True,
    }
    for key, expected in required_invariants.items():
        if type(invariants.get(key)) is not type(expected) or invariants.get(key) != expected:
            raise BenchmarkV2Error(
                f"asset manifest.invariants.{key} must be {expected!r}"
            )

    manifest_info = asset_manifest.get("manifest")
    if not isinstance(manifest_info, Mapping):
        raise BenchmarkV2Error("asset manifest.manifest must be an object")
    expected_manifest_hash = manifest_info.get("sha256")
    if (
        not isinstance(expected_manifest_hash, str)
        or len(expected_manifest_hash) != 64
        or expected_manifest_hash.lower() != expected_manifest_hash
        or any(char not in "0123456789abcdef" for char in expected_manifest_hash)
    ):
        raise BenchmarkV2Error(
            "asset manifest.manifest.sha256 must be a lowercase SHA-256 digest"
        )
    if expected_manifest_hash != benchmark_manifest_sha256:
        raise BenchmarkV2Error(
            "asset manifest is bound to a different benchmark manifest: "
            f"expected {expected_manifest_hash}, got {benchmark_manifest_sha256}"
        )
    manifest_reference = manifest_info.get("path")
    if manifest_reference is not None:
        _safe_relative_path(manifest_reference, "asset manifest.manifest.path")

    if not isinstance(manifest_selected_images, Mapping):
        raise BenchmarkV2Error(
            "benchmark manifest.selected_images must be an object for the minimal backend"
        )
    manifest_selected_file = _safe_relative_path(
        manifest_selected_images.get("file"),
        "benchmark manifest.selected_images.file",
    )
    manifest_selected_hash = _require_sha256(
        manifest_selected_images.get("sha256"),
        "benchmark manifest.selected_images.sha256",
    )
    manifest_selected_count = _require_nonnegative_int(
        manifest_selected_images.get("count"),
        "benchmark manifest.selected_images.count",
    )

    report_maps = asset_manifest.get("maps")
    if not isinstance(report_maps, list) or any(
        not isinstance(map_name, str) or not map_name for map_name in report_maps
    ):
        raise BenchmarkV2Error("asset manifest.maps must be a list of map names")
    if len(set(report_maps)) != len(report_maps) or set(report_maps) != set(expected_maps):
        raise BenchmarkV2Error(
            "asset manifest.maps must exactly match benchmark manifest maps"
        )
    if any(
        Path(map_name).name != map_name
        or "\\" in map_name
        or map_name in {".", ".."}
        for map_name in report_maps
    ):
        raise BenchmarkV2Error("asset manifest map names must be single path components")

    asset_root = asset_manifest_path.parent.resolve()
    images = asset_manifest.get("images")
    if not isinstance(images, Mapping):
        raise BenchmarkV2Error("asset manifest.images must be an object")
    if images.get("status") != "verified":
        raise BenchmarkV2Error("asset manifest.images.status must be 'verified'")
    images_root_relative = _safe_relative_path(
        images.get("root"), "asset manifest.images.root"
    )
    images_root = _safe_asset_path(
        asset_root, images_root_relative, "asset manifest.images.root"
    )
    if not images_root.is_dir():
        raise FileNotFoundError(f"Benchmark v2 image root does not exist: {images_root}")
    image_by_map = images.get("by_map")
    if not isinstance(image_by_map, Mapping) or set(image_by_map) != set(expected_maps):
        raise BenchmarkV2Error(
            "asset manifest.images.by_map must exactly match benchmark maps"
        )
    image_count = _require_nonnegative_int(
        images.get("count"), "asset manifest.images.count"
    )
    image_bytes = _require_nonnegative_int(
        images.get("bytes"), "asset manifest.images.bytes"
    )
    image_count_by_map = 0
    image_bytes_by_map = 0
    for map_name in expected_maps:
        map_summary = image_by_map[map_name]
        if not isinstance(map_summary, Mapping):
            raise BenchmarkV2Error(
                f"asset manifest.images.by_map.{map_name} must be an object"
            )
        map_count = _require_nonnegative_int(
            map_summary.get("count"),
            f"asset manifest.images.by_map.{map_name}.count",
        )
        map_bytes = _require_nonnegative_int(
            map_summary.get("bytes"),
            f"asset manifest.images.by_map.{map_name}.bytes",
        )
        image_count_by_map += map_count
        image_bytes_by_map += map_bytes
    if image_count != image_count_by_map or image_bytes != image_bytes_by_map:
        raise BenchmarkV2Error(
            "asset manifest.images count/bytes do not match images.by_map totals"
        )
    image_template = _validate_asset_template(
        images.get("target_template"), "asset manifest.images.target_template"
    )
    image_dirs: dict[str, Path] = {}
    image_extension: str | None = None
    for map_name in expected_maps:
        try:
            rendered = image_template.format(
                map=map_name, file_frame="__benchmark_v2_frame__"
            )
        except (KeyError, ValueError) as exc:
            raise BenchmarkV2Error(
                f"could not render image target for map {map_name}"
            ) from exc
        rendered_relative = _safe_relative_path(
            rendered, f"asset manifest image target for {map_name}"
        )
        target = _safe_asset_path(
            asset_root,
            rendered_relative,
            f"asset manifest image target for {map_name}",
        )
        try:
            target.relative_to(images_root)
        except ValueError as exc:
            raise BenchmarkV2Error(
                f"image target for {map_name} is outside asset manifest.images.root"
            ) from exc
        if target.name != "__benchmark_v2_frame__" + target.suffix:
            raise BenchmarkV2Error(
                "asset manifest.images.target_template must end with "
                "{file_frame} plus an extension"
            )
        if image_extension is None:
            image_extension = target.suffix
        elif image_extension != target.suffix:
            raise BenchmarkV2Error(
                "asset manifest.images.target_template must use one extension"
            )
        image_dir = target.parent
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Benchmark v2 image directory does not exist for {map_name}: "
                f"{image_dir}"
            )
        image_dirs[map_name] = image_dir
    if image_extension is None:
        raise BenchmarkV2Error("asset manifest image extension is empty")

    selected_images = asset_manifest.get("selected_images")
    if not isinstance(selected_images, Mapping):
        raise BenchmarkV2Error("asset manifest.selected_images must be an object")
    checksum_path_relative = _safe_relative_path(
        selected_images.get("path"), "asset manifest.selected_images.path"
    )
    checksum_path = _safe_asset_path(
        asset_root,
        checksum_path_relative,
        "asset manifest.selected_images.path",
    )
    if not checksum_path.is_file():
        raise FileNotFoundError(
            f"Benchmark v2 selected image checksum file does not exist: {checksum_path}"
        )
    checksum_hash = selected_images.get("sha256")
    checksum_hash = _require_sha256(
        checksum_hash, "asset manifest.selected_images.sha256"
    )
    if checksum_path_relative != manifest_selected_file:
        raise BenchmarkV2Error(
            "asset manifest.selected_images.path does not match "
            "benchmark manifest.selected_images.file"
        )
    if checksum_hash != manifest_selected_hash:
        raise BenchmarkV2Error(
            "asset manifest.selected_images.sha256 does not match "
            "benchmark manifest.selected_images.sha256"
        )
    selected_count = _require_nonnegative_int(
        selected_images.get("count"), "asset manifest.selected_images.count"
    )
    if selected_count != manifest_selected_count:
        raise BenchmarkV2Error(
            "asset manifest.selected_images.count does not match "
            "benchmark manifest.selected_images.count"
        )
    if image_count != selected_count:
        raise BenchmarkV2Error(
            "asset manifest.images.count does not match "
            "asset manifest.selected_images.count"
        )
    actual_checksum_hash = _sha256_file(checksum_path)
    if actual_checksum_hash != checksum_hash:
        raise BenchmarkV2Error(
            "asset manifest.selected_images.sha256 does not match checksum file: "
            f"expected {checksum_hash}, got {actual_checksum_hash}"
        )
    actual_selected_count = _count_selected_image_checksums(checksum_path)
    if actual_selected_count != selected_count:
        raise BenchmarkV2Error(
            "selected image checksum entry count does not match "
            "asset manifest.selected_images.count: "
            f"expected {selected_count}, got {actual_selected_count}"
        )

    radars = asset_manifest.get("radars")
    if not isinstance(radars, Mapping):
        raise BenchmarkV2Error("asset manifest.radars must be an object")
    if radars.get("status") != "verified":
        raise BenchmarkV2Error("asset manifest.radars.status must be 'verified'")
    radars_root_relative = _safe_relative_path(
        radars.get("root"), "asset manifest.radars.root"
    )
    radars_root = _safe_asset_path(
        asset_root, radars_root_relative, "asset manifest.radars.root"
    )
    if not radars_root.is_dir():
        raise FileNotFoundError(f"Benchmark v2 radar root does not exist: {radars_root}")
    radar_by_map = radars.get("by_map")
    if not isinstance(radar_by_map, Mapping) or set(radar_by_map) != set(expected_maps):
        raise BenchmarkV2Error(
            "asset manifest.radars.by_map must exactly match benchmark maps"
        )
    radar_count = _require_nonnegative_int(
        radars.get("count"), "asset manifest.radars.count"
    )
    radar_bytes = _require_nonnegative_int(
        radars.get("bytes"), "asset manifest.radars.bytes"
    )
    if not isinstance(manifest_radar_files, Mapping) or set(manifest_radar_files) != set(
        expected_maps
    ):
        raise BenchmarkV2Error(
            "benchmark manifest.source.radar_files must name exactly every protocol map"
        )
    if not isinstance(manifest_radar_sha256, Mapping) or set(manifest_radar_sha256) != set(
        expected_maps
    ):
        raise BenchmarkV2Error(
            "benchmark manifest.radar_sha256 must name exactly every protocol map"
        )
    entries = radars.get("entries")
    if not isinstance(entries, list) or len(entries) != len(expected_maps):
        raise BenchmarkV2Error(
            "asset manifest.radars.entries must contain exactly one entry per map"
        )
    radar_paths: dict[str, Path] = {}
    radar_count_by_map = 0
    radar_bytes_by_map = 0
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise BenchmarkV2Error("asset manifest radar entry must be an object")
        map_name = entry.get("map")
        if (
            not isinstance(map_name, str)
            or map_name in radar_paths
            or map_name not in expected_maps
        ):
            raise BenchmarkV2Error(
                f"asset manifest radar entry has invalid or duplicate map: {map_name!r}"
            )
        manifest_source = _safe_relative_path(
            manifest_radar_files[map_name],
            f"benchmark manifest.source.radar_files.{map_name}",
        )
        entry_source = _safe_relative_path(
            entry.get("source"), f"asset manifest radar source for {map_name}"
        )
        if entry_source != manifest_source:
            raise BenchmarkV2Error(
                f"asset manifest radar source does not match benchmark manifest for "
                f"{map_name}"
            )
        target_relative = _safe_relative_path(
            entry.get("target"), f"asset manifest radar target for {map_name}"
        )
        if (
            len(target_relative.parts) < 2
            or target_relative.parts[0] != map_name
            or target_relative.name != manifest_source.name
        ):
            raise BenchmarkV2Error(
                f"asset manifest radar target is inconsistent with map/source for "
                f"{map_name}"
            )
        target = _safe_asset_path(
            radars_root,
            target_relative,
            f"asset manifest radar target for {map_name}",
        )
        try:
            target.relative_to(radars_root)
        except ValueError as exc:
            raise BenchmarkV2Error(
                f"asset manifest radar target for {map_name} escapes radars.root"
            ) from exc
        if not target.is_file():
            raise FileNotFoundError(
                f"Benchmark v2 radar file does not exist for {map_name}: {target}"
            )
        expected_size = _require_nonnegative_int(
            entry.get("bytes"), f"asset manifest radar bytes for {map_name}"
        )
        actual_size = target.stat().st_size
        if actual_size != expected_size:
            raise BenchmarkV2Error(
                f"Benchmark v2 radar size mismatch for {map_name}: "
                f"expected {expected_size}, got {actual_size}"
            )
        expected_hash = _require_sha256(
            entry.get("sha256"), f"asset manifest radar sha256 for {map_name}"
        )
        manifest_hash = _require_sha256(
            manifest_radar_sha256[map_name],
            f"benchmark manifest.radar_sha256.{map_name}",
        )
        if expected_hash != manifest_hash:
            raise BenchmarkV2Error(
                f"asset manifest radar sha256 does not match benchmark manifest for "
                f"{map_name}"
            )
        actual_hash = _sha256_file(target)
        if actual_hash != expected_hash:
            raise BenchmarkV2Error(
                f"Benchmark v2 radar hash mismatch for {map_name}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        map_summary = radar_by_map[map_name]
        if not isinstance(map_summary, Mapping):
            raise BenchmarkV2Error(
                f"asset manifest.radars.by_map.{map_name} must be an object"
            )
        map_count = _require_nonnegative_int(
            map_summary.get("count"),
            f"asset manifest.radars.by_map.{map_name}.count",
        )
        map_bytes = _require_nonnegative_int(
            map_summary.get("bytes"),
            f"asset manifest.radars.by_map.{map_name}.bytes",
        )
        if map_count != 1 or map_bytes != expected_size:
            raise BenchmarkV2Error(
                f"asset manifest.radars.by_map.{map_name} does not match radar entry"
            )
        radar_count_by_map += map_count
        radar_bytes_by_map += map_bytes
        radar_paths[map_name] = target
    if set(radar_paths) != set(expected_maps):
        missing = sorted(set(expected_maps).difference(radar_paths))
        extra = sorted(set(radar_paths).difference(expected_maps))
        raise BenchmarkV2Error(
            "asset manifest radar maps do not match benchmark maps: "
            f"missing={missing}, extra={extra}"
        )
    if radar_count != radar_count_by_map or radar_bytes != radar_bytes_by_map:
        raise BenchmarkV2Error(
            "asset manifest.radars count/bytes do not match radars.by_map totals"
        )

    return (
        asset_root,
        image_dirs,
        radar_paths,
        image_extension,
        {
            "backend": "minimal",
            "manifest_path": str(asset_manifest_path),
            "manifest_sha256": _sha256_file(asset_manifest_path),
            "root": images_root_relative.as_posix(),
            "asset_root": images_root_relative.as_posix(),
            "benchmark_manifest_sha256": benchmark_manifest_sha256,
            "schema_version": asset_manifest["schema_version"],
            "asset_manifest_path": str(asset_manifest_path),
            "asset_manifest_sha256": _sha256_file(asset_manifest_path),
            "benchmark_manifest_path": str(benchmark_manifest_path),
            "selected_images_path": str(checksum_path),
            "selected_images_sha256": checksum_hash,
            "images_root": str(images_root),
            "image_target_template": image_template,
            "radars_root": str(radars_root),
        },
    )


def benchmark_v2_image_dir(selection: Any, map_name: str) -> Path:
    """Return the resolved v2 image directory for a selected map.

    Both the historical ``<root>/<map>/imgs`` layout and the flat minimal
    bundle are represented by ``selection.image_dirs``.  The fallback keeps
    this helper compatible with older selection-like mappings used by tests.
    """

    if not isinstance(map_name, str) or not map_name:
        raise BenchmarkV2Error("map_name must be a non-empty string")
    image_dirs = (
        selection.get("image_dirs")
        if isinstance(selection, Mapping)
        else getattr(selection, "image_dirs", None)
    )
    if image_dirs is not None and map_name in image_dirs:
        return Path(image_dirs[map_name])
    source_root = (
        selection.get("source_root")
        if isinstance(selection, Mapping)
        else getattr(selection, "source_root", None)
    )
    if source_root is None:
        raise BenchmarkV2Error(f"no Benchmark v2 image directory for map {map_name}")
    return Path(source_root) / map_name / "imgs"


def benchmark_v2_image_path(selection: Any, map_name: str, file_frame: str) -> Path:
    """Return a safe v2 FPV image path for a selected frame."""

    if not isinstance(file_frame, str) or not file_frame:
        raise BenchmarkV2Error("file_frame must be a non-empty string")
    asset_backend = (
        selection.get("asset_backend")
        if isinstance(selection, Mapping)
        else getattr(selection, "asset_backend", "source")
    )
    extension = (
        selection.get("image_extension")
        if isinstance(selection, Mapping)
        else getattr(selection, "image_extension", ".jpg")
    )
    extension = str(extension)
    if extension and not extension.startswith("."):
        extension = "." + extension
    if asset_backend == "minimal":
        provenance = (
            selection.get("asset_provenance")
            if isinstance(selection, Mapping)
            else getattr(selection, "asset_provenance", None)
        )
        template = provenance.get("image_target_template") if provenance else None
        asset_root = (
            selection.get("source_root")
            if isinstance(selection, Mapping)
            else getattr(selection, "source_root", None)
        )
        if not isinstance(template, str) or asset_root is None:
            raise BenchmarkV2Error(
                "minimal Benchmark v2 selection is missing image asset metadata"
            )
        try:
            rendered = template.format(map=map_name, file_frame=file_frame)
        except (KeyError, ValueError) as exc:
            raise BenchmarkV2Error(
                f"could not render minimal image target for {map_name}:{file_frame}"
            ) from exc
        relative = _safe_relative_path(
            rendered, f"minimal image target for {map_name}:{file_frame}"
        )
        path = _safe_asset_path(
            Path(asset_root).resolve(),
            relative,
            f"minimal image target for {map_name}:{file_frame}",
        )
        if path.suffix != extension:
            raise BenchmarkV2Error(
                f"minimal image target extension mismatch for {map_name}:{file_frame}"
            )
        return path
    return benchmark_v2_image_dir(selection, map_name) / (file_frame + extension)


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
    "benchmark_v2_image_dir",
    "benchmark_v2_image_path",
    "is_benchmark_v2_config",
    "load_benchmark_v2_selection",
    "load_benchmark_v2_selection_from_args",
]
