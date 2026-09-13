#!/usr/bin/env python3
"""Materialize the exact image assets selected by Benchmark v2.

The benchmark builder deliberately records image provenance instead of copying
the (very large) source corpus.  This command turns that provenance into a
self-contained asset tree without changing the source corpus:

``source/<map>/imgs/<file_frame>.jpg`` -> ``images/<map>/<file_frame>.jpg``

The expected frame set is rebuilt from every formal split, including all
CrossMap support seeds, and is then compared with ``selected_images.sha256``.
Images and the explicitly declared radar assets are copied into private stage
directories and committed with directory renames.  Existing final trees are
never overwritten, deleted, or repaired; they are only verified.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - the supported runtime is POSIX
    fcntl = None


EXPECTED_BENCHMARK_ID = "csgo_benchmark_v2"
EXPECTED_SCHEMA_VERSION = 1
IMAGE_DIR_NAME = "images"
RADAR_DIR_NAME = "radars"
REPORT_NAME = "minimal_dataset_report.json"
DEFAULT_MANIFEST = Path("data/csgo_benchmark_v2/benchmark_manifest.json")
DEFAULT_CONFIG = Path("csgo_configs/benchmark_v2.yaml")
DEFAULT_IMAGE_DIR = "imgs"
DEFAULT_IMAGE_EXTENSION = ".jpg"
HASH_BLOCK_SIZE = 1024 * 1024


class MaterializationError(RuntimeError):
    """Raised when the release inputs or materialized assets are unsafe."""


@dataclass(frozen=True)
class ImageEntry:
    map_name: str
    file_frame: str
    source_relative: str
    target_relative: str
    source_path: Path
    digest: str
    size: int


@dataclass(frozen=True)
class RadarEntry:
    map_name: str
    source_relative: str
    target_relative: str
    source_path: Path
    digest: str
    size: int


@dataclass(frozen=True)
class TargetEntry:
    """One independently verifiable file in an already migrated bundle."""

    map_name: str
    target_relative: str
    digest: str
    source_relative: str | None = None


@dataclass
class MaterializationContext:
    repo_root: Path
    manifest_path: Path
    config_path: Path
    output_root: Path
    source_root: Path
    manifest: dict[str, Any]
    config: dict[str, Any]
    maps: list[str]
    image_entries: list[ImageEntry]
    radar_entries: list[RadarEntry]
    selected_images_bytes: bytes
    selected_images_digest: str
    manifest_digest: str
    config_digest: str
    build_report_digest: str | None


@dataclass
class TargetVerificationContext:
    repo_root: Path
    manifest_path: Path
    config_path: Path
    output_root: Path
    manifest: dict[str, Any]
    report: dict[str, Any]
    maps: list[str]
    image_entries: list[TargetEntry]
    radar_entries: list[TargetEntry]
    selected_images_digest: str
    manifest_digest: str
    config_digest: str
    build_report_digest: str | None


def _fail(message: str) -> MaterializationError:
    return MaterializationError(message)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(HASH_BLOCK_SIZE), b""):
                digest.update(block)
    except OSError as exc:
        raise _fail(f"could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def _json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail(f"could not read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise _fail(f"{description} must be a JSON object: {path}")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _fail(f"could not read config {path}: {exc}") from exc
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    try:
        value = yaml.safe_load(raw) if yaml is not None else json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise _fail(f"could not parse config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise _fail(f"config must be a mapping: {path}")
    return value


def _resolve_from_repo(repo_root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _relative_path(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(f"{field_name} must be a non-empty relative path")
    path = PurePosixPath(value.replace(os.sep, "/"))
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise _fail(f"{field_name} must not be absolute or contain dot segments: {value!r}")
    if "\\" in value:
        raise _fail(f"{field_name} must use POSIX separators: {value!r}")
    return "/".join(path.parts)


def _safe_source_path(source_root: Path, relative: str, description: str) -> Path:
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise _fail(f"{description} escapes source root: {relative!r}")
    path = source_root / Path(*relative_path.parts)
    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(source_root)
    except ValueError as exc:
        raise _fail(f"{description} escapes source root: {relative!r}") from exc
    return path


def _require_regular_file(path: Path, description: str) -> None:
    try:
        stat_result = path.lstat()
    except OSError as exc:
        raise _fail(f"{description} is missing: {path}") from exc
    if not stat_result:
        raise _fail(f"{description} is missing: {path}")
    if not path.is_file() or path.is_symlink():
        raise _fail(f"{description} must be a regular non-symlink file: {path}")


def _require_private_regular_file(path: Path, description: str) -> None:
    """Require a standalone regular file for a source-free bundle check."""

    _require_regular_file(path, description)
    try:
        link_count = path.stat().st_nlink
    except OSError as exc:
        raise _fail(f"could not stat {description}: {path}: {exc}") from exc
    if link_count != 1:
        raise _fail(
            f"{description} must not be hard-linked "
            f"(link count {link_count}): {path}"
        )


def _load_protocol_maps(manifest: Mapping[str, Any]) -> tuple[list[str], list[int]]:
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise _fail("manifest.protocol must be an object")
    seen = protocol.get("seen_maps")
    crossmap = protocol.get("crossmap_maps")
    seeds = protocol.get("support_seeds")
    _validate_map_names(seen, "manifest.protocol.seen_maps")
    _validate_map_names(crossmap, "manifest.protocol.crossmap_maps")
    if set(seen) & set(crossmap):
        raise _fail("manifest protocol maps overlap")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(item, bool) or not isinstance(item, int) for item in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise _fail("manifest.protocol.support_seeds must be a unique integer list")
    return [*seen, *crossmap], list(seeds)


def _validate_map_names(value: Any, field_name: str) -> None:
    if not isinstance(value, list) or not value:
        raise _fail(f"{field_name} must be a non-empty list")
    for item in value:
        if (
            not isinstance(item, str)
            or not item
            or item in {".", ".."}
            or "/" in item
            or "\\" in item
            or "\x00" in item
            or any(ord(char) < 32 or ord(char) == 127 for char in item)
        ):
            raise _fail(
                f"{field_name} entries must be safe single POSIX path components: {item!r}"
            )
    if len(set(value)) != len(value):
        raise _fail(f"{field_name} must not contain duplicates")


def _require_row_map_frame(raw: Any, map_name: str, location: str) -> str:
    if not isinstance(raw, Mapping):
        raise _fail(f"{location} must be an object")
    if raw.get("map") != map_name:
        raise _fail(
            f"{location} map mismatch: expected {map_name!r}, got {raw.get('map')!r}"
        )
    file_frame = raw.get("file_frame")
    if not isinstance(file_frame, str) or not file_frame:
        raise _fail(f"{location}.file_frame must be a non-empty string")
    return file_frame


def _load_json(
    path: Path,
    description: str,
    *,
    require_private: bool = False,
) -> Any:
    if require_private:
        _require_private_regular_file(path, description)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail(f"could not read {description} {path}: {exc}") from exc


def _add_split_rows(
    expected: dict[tuple[str, str], str],
    values: Any,
    map_name: str,
    location: str,
) -> None:
    if not isinstance(values, list):
        raise _fail(f"{location} must be a JSON list")
    for index, raw in enumerate(values):
        file_frame = _require_row_map_frame(raw, map_name, f"{location}[{index}]")
        key = (map_name, file_frame)
        expected.setdefault(key, f"{location}[{index}]")


def _add_continuous_rows(
    expected: dict[tuple[str, str], str],
    payload: Any,
    map_name: str,
    location: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise _fail(f"{location} must be a JSON object")
    if payload.get("map") != map_name or payload.get("split") != "continuous":
        raise _fail(f"{location} metadata does not match {map_name}/continuous")
    clips = payload.get("clips")
    if not isinstance(clips, list):
        raise _fail(f"{location}.clips must be a JSON list")
    for clip_index, clip in enumerate(clips):
        if not isinstance(clip, Mapping):
            raise _fail(f"{location}.clips[{clip_index}] must be an object")
        frames = clip.get("frames")
        if not isinstance(frames, list):
            raise _fail(f"{location}.clips[{clip_index}].frames must be a JSON list")
        for frame_index, raw in enumerate(frames):
            file_frame = _require_row_map_frame(
                raw,
                map_name,
                f"{location}.clips[{clip_index}].frames[{frame_index}]",
            )
            key = (map_name, file_frame)
            expected.setdefault(
                key,
                f"{location}.clips[{clip_index}].frames[{frame_index}]",
            )


def _rebuild_expected_keys(
    benchmark_root: Path,
    seen_maps: Sequence[str],
    crossmap_maps: Sequence[str],
    support_seeds: Sequence[int],
    *,
    require_private_files: bool = False,
) -> dict[tuple[str, str], str]:
    expected: dict[tuple[str, str], str] = {}

    if require_private_files:
        _require_directory(benchmark_root / "splits", "splits directory")
        _require_directory(benchmark_root / "splits" / "seen", "seen splits directory")
        _require_directory(
            benchmark_root / "splits" / "crossmap", "crossmap splits directory"
        )

    for map_name in seen_maps:
        prefix = benchmark_root / "splits" / "seen" / map_name
        if require_private_files:
            _require_directory(prefix, f"seen {map_name} splits directory")
        for split_name in ("train", "validation", "discrete_test"):
            path = prefix / f"{split_name}.json"
            _add_split_rows(
                expected,
                _load_json(
                    path,
                    f"seen {split_name} split",
                    require_private=require_private_files,
                ),
                map_name,
                str(path),
            )
        path = prefix / "continuous_clips.json"
        _add_continuous_rows(
            expected,
            _load_json(
                path,
                "seen continuous split",
                require_private=require_private_files,
            ),
            map_name,
            str(path),
        )

    for map_name in crossmap_maps:
        prefix = benchmark_root / "splits" / "crossmap" / map_name
        if require_private_files:
            _require_directory(prefix, f"crossmap {map_name} splits directory")
        for seed in support_seeds:
            path = prefix / f"support_seed_{seed}.json"
            _add_split_rows(
                expected,
                _load_json(
                    path,
                    f"crossmap support seed {seed} split",
                    require_private=require_private_files,
                ),
                map_name,
                str(path),
            )
        path = prefix / "query_test.json"
        _add_split_rows(
            expected,
            _load_json(
                path,
                "crossmap query split",
                require_private=require_private_files,
            ),
            map_name,
            str(path),
        )
        path = prefix / "continuous_clips.json"
        _add_continuous_rows(
            expected,
            _load_json(
                path,
                "crossmap continuous split",
                require_private=require_private_files,
            ),
            map_name,
            str(path),
        )
    return expected


def _parse_selected_images(
    selected_path: Path,
    maps: Sequence[str],
    source_root: Path,
    image_dir_name: str,
    image_extension: str,
    record_regex: re.Pattern[str],
) -> tuple[dict[tuple[str, str], tuple[str, str]], bytes, str]:
    try:
        raw_bytes = selected_path.read_bytes()
    except OSError as exc:
        raise _fail(f"could not read selected image checksum file {selected_path}: {exc}") from exc
    lines = raw_bytes.decode("utf-8").splitlines()
    allowed = set(maps)
    parsed: dict[tuple[str, str], tuple[str, str]] = {}
    previous_path: str | None = None
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise _fail(f"empty line in selected image checksum file at line {line_number}")
        pieces = line.split("  ", 1)
        if len(pieces) != 2 or not re.fullmatch(r"[0-9a-f]{64}", pieces[0]):
            raise _fail(f"invalid selected image checksum line {line_number}: {line!r}")
        digest, relative = pieces
        relative = _relative_path(relative, f"selected image path at line {line_number}")
        path = PurePosixPath(relative)
        if len(path.parts) != 3 or path.parts[0] not in allowed:
            raise _fail(
                "selected image path must be <declared-map>/<images-dir>/<file>.jpg: "
                f"{relative!r}"
            )
        map_name, image_dir, filename = path.parts
        if image_dir != image_dir_name or not filename.endswith(image_extension):
            raise _fail(f"selected image path has unexpected layout: {relative!r}")
        file_frame = filename[: -len(image_extension)]
        if record_regex.fullmatch(file_frame) is None:
            raise _fail(f"selected image filename does not match record regex: {relative!r}")
        if previous_path is not None and relative <= previous_path:
            raise _fail("selected_images.sha256 paths must be strictly sorted")
        previous_path = relative
        key = (map_name, file_frame)
        if key in parsed:
            raise _fail(f"duplicate selected image path/frame: {relative}")
        _safe_source_path(source_root, relative, "selected image")
        parsed[key] = (digest, relative)
    return parsed, raw_bytes, _sha256_bytes(raw_bytes)


def _validate_selected_metadata(
    manifest: Mapping[str, Any],
    selected_bytes: bytes,
    selected_digest: str,
    selected_count: int,
    build_report: Mapping[str, Any] | None,
) -> None:
    metadata = manifest.get("selected_images")
    if not isinstance(metadata, Mapping):
        raise _fail("manifest.selected_images must be an object")
    if metadata.get("file") != "selected_images.sha256":
        raise _fail("manifest.selected_images.file must be selected_images.sha256")
    if metadata.get("count") != selected_count:
        raise _fail(
            f"selected image count mismatch: manifest={metadata.get('count')}, "
            f"file={selected_count}"
        )
    if metadata.get("sha256") != selected_digest:
        raise _fail("selected_images.sha256 does not match manifest.selected_images.sha256")
    if build_report is not None:
        report_metadata = build_report.get("selected_images")
        if report_metadata != metadata:
            raise _fail("build_report.selected_images does not match manifest")
    # Reading the argument here makes the intent explicit and keeps callers
    # from accidentally treating only the file hash as the frame set.
    if not selected_bytes:
        raise _fail("selected_images.sha256 is empty")


def _config_value(config: Mapping[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _build_context(
    repo_root: Path,
    manifest_path: Path,
    config_path: Path,
    output_root_override: Path | None = None,
    source_root_override: Path | None = None,
) -> MaterializationContext:
    manifest_path = manifest_path.resolve()
    config_path = config_path.resolve()
    manifest = _json_object(manifest_path, "benchmark manifest")
    config = _load_config(config_path)
    if manifest.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise _fail("manifest schema_version mismatch")
    if manifest.get("benchmark_id") != EXPECTED_BENCHMARK_ID:
        raise _fail("manifest benchmark_id mismatch")

    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise _fail("manifest.protocol must be an object")
    seen = protocol.get("seen_maps")
    crossmap = protocol.get("crossmap_maps")
    maps, support_seeds = _load_protocol_maps(manifest)
    if not isinstance(seen, list) or not isinstance(crossmap, list):
        raise _fail("manifest protocol map lists are missing")

    configured_image_dir = _config_value(config, "source", "images_dir")
    configured_extension = _config_value(config, "source", "image_extension")
    image_dir_name = configured_image_dir or DEFAULT_IMAGE_DIR
    image_extension = configured_extension or DEFAULT_IMAGE_EXTENSION
    if image_dir_name != DEFAULT_IMAGE_DIR:
        raise _fail(f"materializer requires source.images_dir={DEFAULT_IMAGE_DIR!r}")
    if image_extension != DEFAULT_IMAGE_EXTENSION:
        raise _fail(f"materializer requires source.image_extension={DEFAULT_IMAGE_EXTENSION!r}")
    record_regex_text = _config_value(config, "source", "record_regex")
    if not isinstance(record_regex_text, str) or not record_regex_text:
        raise _fail("config.source.record_regex is required")
    try:
        record_regex = re.compile(record_regex_text)
    except re.error as exc:
        raise _fail(f"invalid config.source.record_regex: {exc}") from exc

    manifest_source_root_value = _config_value(config, "paths", "source_root")
    if manifest_source_root_value is None:
        manifest_source_root_value = manifest.get("source", {}).get("root")
    if source_root_override is not None:
        source_root = source_root_override.resolve()
    else:
        source_root = _resolve_from_repo(repo_root, manifest_source_root_value)
    if not source_root.is_dir():
        raise _fail(f"source root does not exist: {source_root}")
    output_root = (
        output_root_override.resolve()
        if output_root_override is not None
        else manifest_path.parent.resolve()
    )
    if not output_root.is_dir():
        raise _fail(f"output root must already be an existing directory: {output_root}")

    configured_output_root = _config_value(config, "paths", "output_root")
    if configured_output_root is not None:
        configured_output_path = _resolve_from_repo(repo_root, configured_output_root)
        if configured_output_path != output_root:
            raise _fail(
                f"output root mismatch: config={configured_output_path}, selected={output_root}"
            )

    source_section = manifest.get("source")
    if not isinstance(source_section, Mapping):
        raise _fail("manifest.source must be an object")
    manifest_root_value = source_section.get("root")
    if source_root_override is None and manifest_root_value is not None:
        manifest_root = _resolve_from_repo(repo_root, manifest_root_value)
        if manifest_root != source_root:
            raise _fail(
                f"source root mismatch: config={source_root}, manifest={manifest_root}"
            )
    radar_files = source_section.get("radar_files")
    if not isinstance(radar_files, Mapping) or set(radar_files) != set(maps):
        raise _fail("manifest.source.radar_files must name exactly every protocol map")

    expected_keys = _rebuild_expected_keys(
        manifest_path.parent, seen, crossmap, support_seeds
    )
    selected_path = output_root / "selected_images.sha256"
    selected, selected_bytes, selected_digest = _parse_selected_images(
        selected_path,
        maps,
        source_root,
        image_dir_name,
        image_extension,
        record_regex,
    )
    expected_key_set = set(expected_keys)
    selected_key_set = set(selected)
    if expected_key_set != selected_key_set:
        missing = sorted(expected_key_set - selected_key_set)
        extra = sorted(selected_key_set - expected_key_set)
        raise _fail(
            "split-derived frame set does not exactly match selected_images.sha256; "
            f"missing={missing[:5]} (total {len(missing)}), "
            f"extra={extra[:5]} (total {len(extra)})"
        )
    _validate_selected_metadata(
        manifest,
        selected_bytes,
        selected_digest,
        len(selected),
        _load_optional_build_report(output_root),
    )

    image_entries: list[ImageEntry] = []
    for map_name, file_frame in sorted(selected):
        digest, source_relative = selected[(map_name, file_frame)]
        source_path = _safe_source_path(source_root, source_relative, "selected image")
        target_relative = f"{map_name}/{file_frame}{image_extension}"
        _require_regular_file(source_path, "selected source image")
        size = source_path.stat().st_size
        image_entries.append(
            ImageEntry(
                map_name,
                file_frame,
                source_relative,
                target_relative,
                source_path,
                digest,
                size,
            )
        )

    radar_hashes = manifest.get("radar_sha256", source_section.get("radar_sha256"))
    if not isinstance(radar_hashes, Mapping) or set(radar_hashes) != set(maps):
        raise _fail("manifest.radar_sha256 must name exactly every protocol map")
    radar_entries: list[RadarEntry] = []
    for map_name in maps:
        source_relative = _relative_path(radar_files[map_name], f"radar_files.{map_name}")
        source_path = _safe_source_path(source_root, source_relative, "manifest radar")
        _require_regular_file(source_path, "manifest radar")
        expected_digest = radar_hashes.get(map_name)
        if not isinstance(expected_digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_digest
        ):
            raise _fail(f"manifest.radar_sha256.{map_name} is invalid")
        target_relative = f"{map_name}/{Path(source_relative).name}"
        radar_entries.append(
            RadarEntry(
                map_name,
                source_relative,
                target_relative,
                source_path,
                expected_digest,
                source_path.stat().st_size,
            )
        )

    build_report_path = output_root / "build_report.json"
    build_report = _load_optional_build_report(output_root)
    build_report_digest = (
        sha256_file(build_report_path) if build_report is not None else None
    )
    return MaterializationContext(
        repo_root=repo_root,
        manifest_path=manifest_path,
        config_path=config_path,
        output_root=output_root,
        source_root=source_root,
        manifest=manifest,
        config=config,
        maps=maps,
        image_entries=image_entries,
        radar_entries=radar_entries,
        selected_images_bytes=selected_bytes,
        selected_images_digest=selected_digest,
        manifest_digest=sha256_file(manifest_path),
        config_digest=sha256_file(config_path),
        build_report_digest=build_report_digest,
    )


def _load_optional_build_report(
    output_root: Path,
    *,
    require_private: bool = False,
) -> dict[str, Any] | None:
    path = output_root / "build_report.json"
    if not path.exists() and not path.is_symlink():
        return None
    if require_private:
        _require_private_regular_file(path, "build report")
    else:
        _require_regular_file(path, "build report")
    return _json_object(path, "build report")


def _build_target_verification_context(
    repo_root: Path,
    manifest_path: Path,
    config_path: Path,
    output_root_override: Path | None = None,
) -> TargetVerificationContext:
    """Build verification state without resolving or reading the source corpus."""

    # Keep the lexical paths long enough to reject direct symlinks.  Calling
    # Path.resolve() first would erase that evidence and make a symlink look
    # like an ordinary bundle file.
    manifest_candidate = manifest_path.absolute()
    config_candidate = config_path.absolute()
    _require_private_regular_file(manifest_candidate, "benchmark manifest")
    _require_private_regular_file(config_candidate, "benchmark config")
    if output_root_override is not None:
        output_candidate = output_root_override.absolute()
        _require_directory(output_candidate, "output root")
        output_root = output_candidate.resolve()
    else:
        output_root = manifest_candidate.parent.resolve()
    manifest_path = manifest_candidate.resolve()
    config_path = config_candidate.resolve()
    _require_directory(output_root, "output root")
    manifest = _json_object(manifest_path, "benchmark manifest")
    config = _load_config(config_path)
    if manifest.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise _fail("manifest schema_version mismatch")
    if manifest.get("benchmark_id") != EXPECTED_BENCHMARK_ID:
        raise _fail("manifest benchmark_id mismatch")

    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise _fail("manifest.protocol must be an object")
    seen = protocol.get("seen_maps")
    crossmap = protocol.get("crossmap_maps")
    maps, support_seeds = _load_protocol_maps(manifest)
    if not isinstance(seen, list) or not isinstance(crossmap, list):
        raise _fail("manifest protocol map lists are missing")

    configured_image_dir = _config_value(config, "source", "images_dir")
    configured_extension = _config_value(config, "source", "image_extension")
    image_dir_name = configured_image_dir or DEFAULT_IMAGE_DIR
    image_extension = configured_extension or DEFAULT_IMAGE_EXTENSION
    if image_dir_name != DEFAULT_IMAGE_DIR:
        raise _fail(f"target verifier requires source.images_dir={DEFAULT_IMAGE_DIR!r}")
    if image_extension != DEFAULT_IMAGE_EXTENSION:
        raise _fail(
            f"target verifier requires source.image_extension={DEFAULT_IMAGE_EXTENSION!r}"
        )
    record_regex_text = _config_value(config, "source", "record_regex")
    if not isinstance(record_regex_text, str) or not record_regex_text:
        raise _fail("config.source.record_regex is required")
    try:
        record_regex = re.compile(record_regex_text)
    except re.error as exc:
        raise _fail(f"invalid config.source.record_regex: {exc}") from exc

    configured_output_root = _config_value(config, "paths", "output_root")
    if configured_output_root is not None and output_root_override is None:
        configured_output_path = _resolve_from_repo(repo_root, configured_output_root)
        if configured_output_path != output_root:
            raise _fail(
                f"output root mismatch: config={configured_output_path}, selected={output_root}"
            )

    expected_keys = _rebuild_expected_keys(
        output_root,
        seen,
        crossmap,
        support_seeds,
        require_private_files=True,
    )
    selected_path = output_root / "selected_images.sha256"
    _require_private_regular_file(selected_path, "selected image checksum file")
    # _parse_selected_images only uses this root for lexical containment.  It
    # deliberately does not read any source image in target-only mode.
    unavailable_source_root = output_root / ".source-unavailable"
    selected, selected_bytes, selected_digest = _parse_selected_images(
        selected_path,
        maps,
        unavailable_source_root,
        image_dir_name,
        image_extension,
        record_regex,
    )
    expected_key_set = set(expected_keys)
    selected_key_set = set(selected)
    if expected_key_set != selected_key_set:
        missing = sorted(expected_key_set - selected_key_set)
        extra = sorted(selected_key_set - expected_key_set)
        raise _fail(
            "split-derived frame set does not exactly match selected_images.sha256; "
            f"missing={missing[:5]} (total {len(missing)}), "
            f"extra={extra[:5]} (total {len(extra)})"
        )
    build_report = _load_optional_build_report(output_root, require_private=True)
    _validate_selected_metadata(
        manifest,
        selected_bytes,
        selected_digest,
        len(selected),
        build_report,
    )
    image_entries = [
        TargetEntry(
            map_name=map_name,
            target_relative=f"{map_name}/{file_frame}{image_extension}",
            digest=selected[(map_name, file_frame)][0],
            source_relative=selected[(map_name, file_frame)][1],
        )
        for map_name, file_frame in sorted(selected)
    ]

    source_section = manifest.get("source")
    if not isinstance(source_section, Mapping):
        raise _fail("manifest.source must be an object")
    radar_files = source_section.get("radar_files")
    if not isinstance(radar_files, Mapping) or set(radar_files) != set(maps):
        raise _fail("manifest.source.radar_files must name exactly every protocol map")
    radar_hashes = manifest.get("radar_sha256", source_section.get("radar_sha256"))
    if not isinstance(radar_hashes, Mapping) or set(radar_hashes) != set(maps):
        raise _fail("manifest.radar_sha256 must name exactly every protocol map")
    radar_entries: list[TargetEntry] = []
    for map_name in maps:
        source_relative = _relative_path(
            radar_files[map_name], f"radar_files.{map_name}"
        )
        digest = radar_hashes.get(map_name)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise _fail(f"manifest.radar_sha256.{map_name} is invalid")
        radar_entries.append(
            TargetEntry(
                map_name=map_name,
                target_relative=f"{map_name}/{Path(source_relative).name}",
                digest=digest,
                source_relative=source_relative,
            )
        )

    report_path = output_root / REPORT_NAME
    _require_private_regular_file(report_path, "materialization report")
    report = _json_object(report_path, "materialization report")
    build_report_path = output_root / "build_report.json"
    build_report_digest = (
        sha256_file(build_report_path) if build_report is not None else None
    )
    return TargetVerificationContext(
        repo_root=repo_root,
        manifest_path=manifest_path,
        config_path=config_path,
        output_root=output_root,
        manifest=manifest,
        report=report,
        maps=maps,
        image_entries=image_entries,
        radar_entries=radar_entries,
        selected_images_digest=selected_digest,
        manifest_digest=sha256_file(manifest_path),
        config_digest=sha256_file(config_path),
        build_report_digest=build_report_digest,
    )


def _expected_relative_paths(
    entries: Iterable[ImageEntry | RadarEntry | TargetEntry],
) -> set[str]:
    return {entry.target_relative for entry in entries}


def _scan_tree(root: Path, maps: Sequence[str], expected: set[str], description: str) -> None:
    _require_directory(root, description)
    allowed_maps = set(maps)
    observed: set[str] = set()
    observed_maps: set[str] = set()
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if child.is_symlink() or not child.is_dir() or child.name not in allowed_maps:
            raise _fail(f"{description} contains an unexpected map entry: {child}")
        observed_maps.add(child.name)
        for nested in sorted(child.iterdir(), key=lambda item: item.name):
            if nested.is_symlink() or not nested.is_file():
                raise _fail(f"{description} contains a non-file entry: {nested}")
            relative = f"{child.name}/{nested.name}"
            if relative in observed:
                raise _fail(f"duplicate target path in {description}: {relative}")
            observed.add(relative)
    if observed_maps != allowed_maps:
        missing_maps = sorted(allowed_maps - observed_maps)
        raise _fail(f"{description} map directory set mismatch; missing={missing_maps}")
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise _fail(
            f"{description} file set mismatch; missing={missing[:5]} (total {len(missing)}), "
            f"extra={extra[:5]} (total {len(extra)})"
        )


def _require_directory(path: Path, description: str) -> None:
    try:
        stat_result = path.lstat()
    except OSError as exc:
        raise _fail(f"{description} is missing: {path}") from exc
    if not stat_result or path.is_symlink() or not path.is_dir():
        raise _fail(f"{description} must be a real directory: {path}")


def _progress(label: str, processed: int, total: int, state: dict[str, int]) -> None:
    if total <= 0:
        return
    while state["next_percent"] <= 100 and processed * 100 >= state["next_percent"] * total:
        print(f"{label}: {state['next_percent']}% ({processed}/{total})", flush=True)
        state["next_percent"] += 10


def _verify_entry(
    source: Path,
    target: Path,
    expected_digest: str,
    description: str,
) -> int:
    _require_regular_file(source, f"{description} source")
    source_digest = sha256_file(source)
    if source_digest != expected_digest:
        raise _fail(
            f"{description} source hash mismatch: expected {expected_digest}, got {source_digest}"
        )
    _require_regular_file(target, f"{description} target")
    target_digest = sha256_file(target)
    if target_digest != expected_digest:
        raise _fail(
            f"{description} target hash mismatch: expected {expected_digest}, got {target_digest}"
        )
    if os.path.samefile(source, target):
        raise _fail(f"{description} source and target are the same file: {source}")
    return target.stat().st_size


def _copy_and_verify_entry(
    source: Path,
    target: Path,
    expected_digest: str,
    description: str,
) -> int:
    _require_regular_file(source, f"{description} source")
    source_digest = sha256_file(source)
    if source_digest != expected_digest:
        raise _fail(
            f"{description} source hash mismatch: expected {expected_digest}, got {source_digest}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise _fail(f"staging target unexpectedly exists: {target}")
    try:
        shutil.copyfile(source, target)
    except OSError as exc:
        raise _fail(f"could not copy {source} to {target}: {exc}") from exc
    target_digest = sha256_file(target)
    if target_digest != expected_digest:
        raise _fail(
            f"{description} staged target hash mismatch: expected {expected_digest}, got {target_digest}"
        )
    _require_regular_file(source, f"{description} source after copy")
    if os.path.samefile(source, target):
        raise _fail(f"{description} source and target unexpectedly share an inode")
    return target.stat().st_size


def _verify_tree_entries(
    root: Path,
    entries: Sequence[ImageEntry | RadarEntry],
    maps: Sequence[str],
    description: str,
    progress_label: str,
) -> dict[str, Any]:
    expected = _expected_relative_paths(entries)
    _scan_tree(root, maps, expected, description)
    by_map: Counter[str] = Counter()
    bytes_by_map: Counter[str] = Counter()
    progress_state = {"next_percent": 10}
    for index, entry in enumerate(entries, 1):
        target = root / Path(*PurePosixPath(entry.target_relative).parts)
        size = _verify_entry(entry.source_path, target, entry.digest, entry.target_relative)
        by_map[entry.map_name] += 1
        bytes_by_map[entry.map_name] += size
        _progress(progress_label, index, len(entries), progress_state)
    return {
        "count": len(entries),
        "bytes": sum(bytes_by_map.values()),
        "by_map": {
            map_name: {"count": by_map[map_name], "bytes": bytes_by_map[map_name]}
            for map_name in maps
        },
    }


def _verify_target_tree_entries(
    root: Path,
    entries: Sequence[TargetEntry],
    maps: Sequence[str],
    description: str,
    progress_label: str,
) -> dict[str, Any]:
    """Verify a migrated tree without consulting source files."""

    _scan_tree(root, maps, _expected_relative_paths(entries), description)
    by_map: Counter[str] = Counter()
    bytes_by_map: Counter[str] = Counter()
    progress_state = {"next_percent": 10}
    for index, entry in enumerate(entries, 1):
        target = root / Path(*PurePosixPath(entry.target_relative).parts)
        _require_regular_file(target, f"{description} target")
        stat_result = target.stat()
        if stat_result.st_nlink != 1:
            raise _fail(
                f"{description} target must not be hard-linked "
                f"(link count {stat_result.st_nlink}): {target}"
            )
        target_digest = sha256_file(target)
        if target_digest != entry.digest:
            raise _fail(
                f"{description} target hash mismatch for {entry.target_relative}: "
                f"expected {entry.digest}, got {target_digest}"
            )
        by_map[entry.map_name] += 1
        bytes_by_map[entry.map_name] += stat_result.st_size
        _progress(progress_label, index, len(entries), progress_state)
    return {
        "count": len(entries),
        "bytes": sum(bytes_by_map.values()),
        "by_map": {
            map_name: {"count": by_map[map_name], "bytes": bytes_by_map[map_name]}
            for map_name in maps
        },
    }


def _copy_tree_entries(
    stage_root: Path,
    entries: Sequence[ImageEntry | RadarEntry],
    maps: Sequence[str],
    description: str,
    progress_label: str,
) -> dict[str, Any]:
    _require_directory(stage_root, f"staging {description}")
    for map_name in maps:
        (stage_root / map_name).mkdir(parents=True, exist_ok=False)
    by_map: Counter[str] = Counter()
    bytes_by_map: Counter[str] = Counter()
    progress_state = {"next_percent": 10}
    for index, entry in enumerate(entries, 1):
        target = stage_root / Path(*PurePosixPath(entry.target_relative).parts)
        size = _copy_and_verify_entry(
            entry.source_path,
            target,
            entry.digest,
            f"{description} {entry.target_relative}",
        )
        by_map[entry.map_name] += 1
        bytes_by_map[entry.map_name] += size
        _progress(progress_label, index, len(entries), progress_state)
    _scan_tree(stage_root, maps, _expected_relative_paths(entries), description)
    return {
        "count": len(entries),
        "bytes": sum(bytes_by_map.values()),
        "by_map": {
            map_name: {"count": by_map[map_name], "bytes": bytes_by_map[map_name]}
            for map_name in maps
        },
    }


def _stale_stage_paths(output_root: Path, prefix: str) -> list[Path]:
    return sorted(output_root.glob(f".{prefix}.stage-*"), key=lambda path: path.name)


def _existing_or_copy_tree(
    output_root: Path,
    directory_name: str,
    entries: Sequence[ImageEntry | RadarEntry],
    maps: Sequence[str],
    description: str,
    progress_label: str,
) -> tuple[dict[str, Any], str]:
    final_root = output_root / directory_name
    if final_root.exists() or final_root.is_symlink():
        stats = _verify_tree_entries(
            final_root, entries, maps, description, f"{progress_label} (existing)"
        )
        return stats, "already_materialized"
    if _stale_stage_paths(output_root, directory_name):
        raise _fail(
            f"stale {directory_name} stage directory exists; inspect/remove it manually: "
            f"{_stale_stage_paths(output_root, directory_name)}"
        )
    stage = Path(
        tempfile.mkdtemp(prefix=f".{directory_name}.stage-", dir=output_root)
    )
    committed = False
    try:
        stats = _copy_tree_entries(
            stage, entries, maps, f"staged {description}", progress_label
        )
        # Ensure the final name did not appear while copying.  The caller
        # holds the output-root directory lock, so this is also a useful
        # explicit guard against accidental replacement.
        if final_root.exists() or final_root.is_symlink():
            raise _fail(f"{final_root} appeared while staging; refusing to replace it")
        os.replace(stage, final_root)
        committed = True
        return stats, "materialized"
    finally:
        if not committed and stage.exists():
            shutil.rmtree(stage)


def _entry_records(entries: Sequence[ImageEntry | RadarEntry]) -> list[dict[str, Any]]:
    return [
        {
            "map": entry.map_name,
            "source": entry.source_relative,
            "target": entry.target_relative,
            "sha256": entry.digest,
            "bytes": entry.size,
        }
        for entry in entries
    ]


def _bundle_relative_to_repo(repo_root: Path, path: Path) -> str:
    """Return a stable repo-relative path for migration provenance."""

    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        # Custom fixture/source paths may intentionally live outside the repo;
        # keep the report portable rather than embedding an absolute path.
        return path.name


def _bundle_relative(ctx: MaterializationContext, path: Path) -> str:
    return _bundle_relative_to_repo(ctx.repo_root, path)


def _report_payload(
    ctx: MaterializationContext,
    image_stats: Mapping[str, Any],
    radar_stats: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark_id": EXPECTED_BENCHMARK_ID,
        "status": "verified",
        "copy_mode": "copyfile_not_move",
        "bundle_paths": {
            "manifest": _bundle_relative(ctx, ctx.manifest_path),
            "config": _bundle_relative(ctx, ctx.config_path),
            "output_root": _bundle_relative(ctx, ctx.output_root),
            "source_root": "<source_root>",
        },
        "manifest": {
            "path": _bundle_relative(ctx, ctx.manifest_path),
            "sha256": ctx.manifest_digest,
        },
        "config": {
            "path": _bundle_relative(ctx, ctx.config_path),
            "sha256": ctx.config_digest,
        },
        "build_report_sha256": ctx.build_report_digest,
        "selected_images": {
            "path": "selected_images.sha256",
            "sha256": ctx.selected_images_digest,
            "count": len(ctx.image_entries),
        },
        "maps": list(ctx.maps),
        "images": {
            "root": IMAGE_DIR_NAME,
            "status": "verified",
            "count": image_stats["count"],
            "bytes": image_stats["bytes"],
            "by_map": image_stats["by_map"],
            "source_template": "<source_root>/{map}/imgs/{file_frame}.jpg",
            "target_template": "images/{map}/{file_frame}.jpg",
        },
        "radars": {
            "root": RADAR_DIR_NAME,
            "status": "verified",
            "count": radar_stats["count"],
            "bytes": radar_stats["bytes"],
            "by_map": radar_stats["by_map"],
            "source_template": "<source_root>/<manifest.source.radar_files[map]>",
            "target_template": "radars/{map}/{basename(source_radar)}",
            "entries": _entry_records(ctx.radar_entries),
        },
        "invariants": {
            "split_derived_set_equals_selected_images": True,
            "source_exists_after_copy": True,
            "source_target_samefile": False,
            "target_contains_only_declared_maps_and_files": True,
            "radars_are_separate_from_images": True,
        },
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise _fail(f"refusing to replace symlink report: {path}")
    data = (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise _fail(f"could not atomically write report {path}: {exc}") from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@contextlib.contextmanager
def _directory_lock(output_root: Path, exclusive: bool) -> Iterator[None]:
    """Serialize materialization without leaving a lock file in the bundle."""

    if fcntl is None:  # pragma: no cover - Linux is the supported runtime
        raise _fail("directory locking requires POSIX fcntl")
    try:
        fd = os.open(output_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        raise _fail(f"could not open output-root lock directory {output_root}: {exc}") from exc
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    except OSError as exc:
        raise _fail(f"could not lock output root {output_root}: {exc}") from exc
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def materialize(ctx: MaterializationContext) -> dict[str, Any]:
    with _directory_lock(ctx.output_root, exclusive=True):
        image_stats, image_status = _existing_or_copy_tree(
            ctx.output_root,
            IMAGE_DIR_NAME,
            ctx.image_entries,
            ctx.maps,
            "image tree",
            "images",
        )
        radar_stats, radar_status = _existing_or_copy_tree(
            ctx.output_root,
            RADAR_DIR_NAME,
            ctx.radar_entries,
            ctx.maps,
            "radar tree",
            "radars",
        )
        payload = _report_payload(ctx, image_stats, radar_stats)
        _atomic_write_json(ctx.output_root / REPORT_NAME, payload)
    print(
        "MATERIALIZE OK: "
        f"images={image_stats['count']} ({image_stats['bytes']} bytes), "
        f"radars={radar_stats['count']} ({radar_stats['bytes']} bytes), "
        f"image_status={image_status}, radar_status={radar_status}"
    )
    return payload


def verify(ctx: MaterializationContext) -> dict[str, Any]:
    with _directory_lock(ctx.output_root, exclusive=False):
        image_stats = _verify_tree_entries(
            ctx.output_root / IMAGE_DIR_NAME,
            ctx.image_entries,
            ctx.maps,
            "image tree",
            "verify images",
        )
        radar_stats = _verify_tree_entries(
            ctx.output_root / RADAR_DIR_NAME,
            ctx.radar_entries,
            ctx.maps,
            "radar tree",
            "verify radars",
        )
        report_path = ctx.output_root / REPORT_NAME
        _require_regular_file(report_path, "materialization report")
        report = _json_object(report_path, "materialization report")
        expected = _report_payload(ctx, image_stats, radar_stats)
        for key in (
            "schema_version",
            "benchmark_id",
            "copy_mode",
            "bundle_paths",
            "manifest",
            "config",
            "build_report_sha256",
            "selected_images",
            "maps",
            "invariants",
        ):
            if report.get(key) != expected.get(key):
                raise _fail(f"materialization report provenance mismatch at {key}")
        for key, stats in (("images", image_stats), ("radars", radar_stats)):
            section = report.get(key)
            if not isinstance(section, Mapping):
                raise _fail(f"materialization report is missing {key}")
            for field in ("count", "bytes", "by_map"):
                if section.get(field) != stats[field]:
                    raise _fail(f"materialization report {key}.{field} mismatch")
            for field in ("root", "source_template", "target_template"):
                if section.get(field) != expected[key][field]:
                    raise _fail(f"materialization report {key}.{field} mismatch")
            if key == "radars" and section.get("entries") != expected[key]["entries"]:
                raise _fail("materialization report radars.entries mismatch")
    print(
        "VERIFY OK: "
        f"images={image_stats['count']} ({image_stats['bytes']} bytes), "
        f"radars={radar_stats['count']} ({radar_stats['bytes']} bytes)"
    )
    return report


def verify_target(ctx: TargetVerificationContext) -> dict[str, Any]:
    """Verify the self-contained migration bundle with no source corpus."""

    with _directory_lock(ctx.output_root, exclusive=False):
        image_stats = _verify_target_tree_entries(
            ctx.output_root / IMAGE_DIR_NAME,
            ctx.image_entries,
            ctx.maps,
            "image tree",
            "verify target images",
        )
        radar_stats = _verify_target_tree_entries(
            ctx.output_root / RADAR_DIR_NAME,
            ctx.radar_entries,
            ctx.maps,
            "radar tree",
            "verify target radars",
        )

        expected_radar_records = []
        for entry in ctx.radar_entries:
            target = ctx.output_root / RADAR_DIR_NAME / Path(
                *PurePosixPath(entry.target_relative).parts
            )
            expected_radar_records.append(
                {
                    "map": entry.map_name,
                    "source": entry.source_relative,
                    "target": entry.target_relative,
                    "sha256": entry.digest,
                    "bytes": target.stat().st_size,
                }
            )

        expected: dict[str, Any] = {
            "schema_version": EXPECTED_SCHEMA_VERSION,
            "benchmark_id": EXPECTED_BENCHMARK_ID,
            "status": "verified",
            "copy_mode": "copyfile_not_move",
            "bundle_paths": {
                "manifest": _bundle_relative_to_repo(ctx.repo_root, ctx.manifest_path),
                "config": _bundle_relative_to_repo(ctx.repo_root, ctx.config_path),
                "output_root": _bundle_relative_to_repo(ctx.repo_root, ctx.output_root),
                "source_root": "<source_root>",
            },
            "manifest": {
                "path": _bundle_relative_to_repo(ctx.repo_root, ctx.manifest_path),
                "sha256": ctx.manifest_digest,
            },
            "config": {
                "path": _bundle_relative_to_repo(ctx.repo_root, ctx.config_path),
                "sha256": ctx.config_digest,
            },
            "build_report_sha256": ctx.build_report_digest,
            "selected_images": {
                "path": "selected_images.sha256",
                "sha256": ctx.selected_images_digest,
                "count": len(ctx.image_entries),
            },
            "maps": list(ctx.maps),
            "images": {
                "root": IMAGE_DIR_NAME,
                "status": "verified",
                "count": image_stats["count"],
                "bytes": image_stats["bytes"],
                "by_map": image_stats["by_map"],
                "source_template": "<source_root>/{map}/imgs/{file_frame}.jpg",
                "target_template": "images/{map}/{file_frame}.jpg",
            },
            "radars": {
                "root": RADAR_DIR_NAME,
                "status": "verified",
                "count": radar_stats["count"],
                "bytes": radar_stats["bytes"],
                "by_map": radar_stats["by_map"],
                "source_template": "<source_root>/<manifest.source.radar_files[map]>",
                "target_template": "radars/{map}/{basename(source_radar)}",
                "entries": expected_radar_records,
            },
            "invariants": {
                "split_derived_set_equals_selected_images": True,
                "source_exists_after_copy": True,
                "source_target_samefile": False,
                "target_contains_only_declared_maps_and_files": True,
                "radars_are_separate_from_images": True,
            },
        }
        for key, value in expected.items():
            if ctx.report.get(key) != value:
                raise _fail(f"materialization report target provenance mismatch at {key}")

    print(
        "VERIFY TARGET OK: "
        f"images={image_stats['count']} ({image_stats['bytes']} bytes), "
        f"radars={radar_stats['count']} ({radar_stats['bytes']} bytes), "
        "source_access=not_required"
    )
    return ctx.report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize and verify exact CSGO Benchmark v2 image assets."
    )
    parser.add_argument("command", choices=("materialize", "verify", "verify-target"))
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = (args.repo_root or Path(__file__).resolve().parents[1]).resolve()
    preserve_links = args.command == "verify-target"

    def _argument_path(value: Path) -> Path:
        path = value if value.is_absolute() else repo_root / value
        return path.absolute() if preserve_links else path.resolve()

    manifest_path = _argument_path(args.manifest)
    config_path = _argument_path(args.config)
    output_root_override = (
        _argument_path(args.output_root) if args.output_root is not None else None
    )
    source_root_override = (
        _argument_path(args.source_root) if args.source_root is not None else None
    )
    try:
        if args.command == "verify-target":
            target_ctx = _build_target_verification_context(
                repo_root,
                manifest_path,
                config_path,
                output_root_override=output_root_override,
            )
            verify_target(target_ctx)
        else:
            ctx = _build_context(
                repo_root,
                manifest_path,
                config_path,
                output_root_override=output_root_override,
                source_root_override=source_root_override,
            )
            if args.command == "materialize":
                materialize(ctx)
            else:
                verify(ctx)
    except (MaterializationError, OSError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
