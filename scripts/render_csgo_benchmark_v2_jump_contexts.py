#!/usr/bin/env python3
"""Render batched image contexts for benchmark-v2 jump-review edges.

The input CSVs contain edges, while this tool renders the two edge endpoints
as anomaly frames and the surrounding frames from the corresponding
``positions.json``.  Edges from different review CSVs are deduplicated by
``(map, file_num, frame_start, frame_end)`` and their labels are unioned.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

# This script is intended for cluster/headless execution.  Set the backend
# before importing pyplot so importing the module does not require DISPLAY.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from PIL import Image


DEFAULT_INPUT_DIR = Path("data/csgo_benchmark_v2/audit/jump_review")
DEFAULT_SOURCE_ROOT = Path("data/preprocessed_data")
DEFAULT_MEDIUM_Z_NAME = "medium_z_jumps_50_300.csv"
DEFAULT_ANGLE_THRESHOLDS = (45.0, 60.0, 90.0, 120.0, 150.0)
DEFAULT_ANGLE_NAMES = {
    threshold: f"angle_jumps_gt_{threshold:03g}.csv"
    for threshold in DEFAULT_ANGLE_THRESHOLDS
}
DEFAULT_POSITIONS_NAME = "positions.json"
DEFAULT_IMAGES_DIR = "imgs"
DEFAULT_IMAGE_EXTENSION = ".jpg"
DEFAULT_RECORD_PATTERN = re.compile(
    r"^file_num(?P<record>\d+)_frame_(?P<frame>\d+)$"
)

EDGE_COLUMNS = (
    "map",
    "file_num",
    "frame_start",
    "frame_end",
    "frame_gap",
    "file_frame_start",
    "file_frame_end",
)
GROUP_COLUMNS = (
    "group_id",
    "map",
    "file_num",
    "center_file_frame",
    "center_frame",
    "radius",
    "group_min_frame",
    "group_max_frame",
    "anomalous_frames",
    "types",
    "edge_count",
    "context_frame_count",
    "page_count",
    "png_files",
)

# Matplotlib's Agg backend refuses figures at or above roughly 2**16 pixels
# on one side.  Keep a margin so metadata/backend rounding cannot cross it.
MAX_PNG_DIMENSION = 60_000
CELL_WIDTH_INCHES = 4.0
CELL_HEIGHT_INCHES = 4.2


@dataclass
class Edge:
    map_name: str
    file_num: str
    frame_start: int
    frame_end: int
    file_frame_start: str
    file_frame_end: str
    types: set[str] = field(default_factory=set)
    source_files: set[str] = field(default_factory=set)

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.map_name, self.file_num, self.frame_start, self.frame_end)


@dataclass
class JumpGroup:
    map_name: str
    file_num: str
    edges: list[Edge]
    frame_types: dict[int, set[str]]
    center_frame: int
    radius: int

    @property
    def anomalous_frames(self) -> list[int]:
        return sorted(self.frame_types)

    @property
    def min_frame(self) -> int:
        return self.anomalous_frames[0]

    @property
    def max_frame(self) -> int:
        return self.anomalous_frames[-1]

    @property
    def types(self) -> list[str]:
        return sorted({label for labels in self.frame_types.values() for label in labels})


@dataclass(frozen=True)
class Position:
    map_name: str
    file_num: str
    frame_id: int
    file_frame: str
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    image_path: Path


@dataclass
class ContextRow:
    position: Position
    types: tuple[str, ...]


def _record_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _normalize_record(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("file_num must not be empty")
    return str(int(text)) if text.isdigit() else text


def _parse_int(value: Any, field_name: str, path: Path, line_number: int) -> int:
    text = str(value).strip()
    try:
        return int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}:{line_number}: {field_name} must be an integer: {value!r}"
        ) from exc


def _parse_file_frame(value: Any, path: Path, line_number: int) -> tuple[str, int]:
    file_frame = str(value).strip()
    match = DEFAULT_RECORD_PATTERN.fullmatch(file_frame)
    if match is None:
        raise ValueError(
            f"{path}:{line_number}: invalid file_frame {file_frame!r}; "
            "expected file_num<record>_frame_<frame>"
        )
    return _normalize_record(match.group("record")), int(match.group("frame"))


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return token or "unnamed"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_replace(path: Path, writer: Any) -> None:
    """Write one output to a same-directory temporary file, then replace it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=path.suffix,
            dir=path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    def write(handle: Any) -> None:
        text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
        handle.write(text.encode("utf-8"))

    _atomic_replace(path, write)


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    def write(handle: Any) -> None:
        text_handle = os.fdopen(os.dup(handle.fileno()), "w", encoding="utf-8", newline="")
        try:
            writer = csv.DictWriter(
                text_handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            text_handle.flush()
        finally:
            text_handle.close()

    _atomic_replace(path, write)


def _read_csv_edges(path: Path, label: str) -> list[Edge]:
    if not path.is_file():
        raise FileNotFoundError(f"jump edge CSV does not exist: {path}")

    result: list[Edge] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"jump edge CSV has no header: {path}")
        missing = [column for column in EDGE_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")

        for line_number, row in enumerate(reader, start=2):
            map_name = str(row["map"]).strip()
            if not map_name:
                raise ValueError(f"{path}:{line_number}: map must not be empty")
            file_num = _normalize_record(row["file_num"])
            frame_start = _parse_int(row["frame_start"], "frame_start", path, line_number)
            frame_end = _parse_int(row["frame_end"], "frame_end", path, line_number)
            frame_gap = _parse_int(row["frame_gap"], "frame_gap", path, line_number)
            if frame_end <= frame_start or frame_gap != frame_end - frame_start:
                raise ValueError(
                    f"{path}:{line_number}: edge frames/gap are inconsistent: "
                    f"{frame_start}->{frame_end}, gap={frame_gap}"
                )
            if not 1 <= frame_gap <= 2:
                raise ValueError(
                    f"{path}:{line_number}: edge frame_gap must be in [1, 2], got {frame_gap}"
                )

            file_frame_start = str(row["file_frame_start"]).strip()
            file_frame_end = str(row["file_frame_end"]).strip()
            record_start, parsed_start = _parse_file_frame(
                file_frame_start, path, line_number
            )
            record_end, parsed_end = _parse_file_frame(file_frame_end, path, line_number)
            if (
                record_start != file_num
                or record_end != file_num
                or parsed_start != frame_start
                or parsed_end != frame_end
            ):
                raise ValueError(
                    f"{path}:{line_number}: file_frame fields do not match the edge: "
                    f"{file_frame_start!r}, {file_frame_end!r}"
                )
            result.append(
                Edge(
                    map_name=map_name,
                    file_num=file_num,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    file_frame_start=file_frame_start,
                    file_frame_end=file_frame_end,
                    types={label},
                    source_files={str(path)},
                )
            )
    return result


def deduplicate_edges(edges: Iterable[Edge]) -> list[Edge]:
    """Union labels for duplicate edges while preserving one logical edge."""

    by_key: dict[tuple[str, str, int, int], Edge] = {}
    for edge in edges:
        existing = by_key.get(edge.key)
        if existing is None:
            by_key[edge.key] = Edge(
                map_name=edge.map_name,
                file_num=edge.file_num,
                frame_start=edge.frame_start,
                frame_end=edge.frame_end,
                file_frame_start=edge.file_frame_start,
                file_frame_end=edge.file_frame_end,
                types=set(edge.types),
                source_files=set(edge.source_files),
            )
            continue
        if (
            existing.file_frame_start != edge.file_frame_start
            or existing.file_frame_end != edge.file_frame_end
        ):
            raise ValueError(f"duplicate edge has conflicting file_frame values: {edge.key}")
        existing.types.update(edge.types)
        existing.source_files.update(edge.source_files)

    return sorted(
        by_key.values(),
        key=lambda edge: (
            edge.map_name,
            _record_sort_key(edge.file_num),
            edge.frame_start,
            edge.frame_end,
        ),
    )


def _finalize_group(map_name: str, file_num: str, edges: list[Edge]) -> JumpGroup:
    frame_types: dict[int, set[str]] = defaultdict(set)
    for edge in edges:
        frame_types[edge.frame_start].update(edge.types)
        frame_types[edge.frame_end].update(edge.types)
    anomaly_frames = sorted(frame_types)
    if not anomaly_frames:
        raise ValueError("cannot finalize a group with no anomalous endpoints")
    minimum = anomaly_frames[0]
    maximum = anomaly_frames[-1]
    midpoint = (minimum + maximum) / 2.0
    center = min(anomaly_frames, key=lambda frame: (abs(frame - midpoint), frame))
    radius = max(center - minimum, maximum - center) + 8
    return JumpGroup(
        map_name=map_name,
        file_num=file_num,
        edges=edges,
        frame_types=dict(frame_types),
        center_frame=center,
        radius=radius,
    )


def merge_edges_into_groups(edges: Iterable[Edge]) -> list[JumpGroup]:
    """Merge same-record edges whose intervals overlap or are at most 2 apart."""

    unique_edges = deduplicate_edges(edges)
    by_record: dict[tuple[str, str], list[Edge]] = defaultdict(list)
    for edge in unique_edges:
        by_record[(edge.map_name, edge.file_num)].append(edge)

    groups: list[JumpGroup] = []
    for (map_name, file_num), record_edges in sorted(
        by_record.items(), key=lambda item: (item[0][0], _record_sort_key(item[0][1]))
    ):
        ordered = sorted(record_edges, key=lambda edge: (edge.frame_start, edge.frame_end))
        current: list[Edge] = []
        current_end: int | None = None
        for edge in ordered:
            if current_end is None or edge.frame_start > current_end + 2:
                if current:
                    groups.append(_finalize_group(map_name, file_num, current))
                current = [edge]
                current_end = edge.frame_end
                continue
            current.append(edge)
            current_end = max(current_end, edge.frame_end)
        if current:
            groups.append(_finalize_group(map_name, file_num, current))

    return sorted(
        groups,
        key=lambda group: (
            group.map_name,
            _record_sort_key(group.file_num),
            group.min_frame,
            group.max_frame,
        ),
    )


def _load_positions(
    source_root: Path,
    map_name: str,
    positions_name: str,
    images_dir: str,
    image_extension: str,
) -> dict[tuple[str, int], Position]:
    positions_path = source_root / map_name / positions_name
    if not positions_path.is_file():
        raise FileNotFoundError(f"positions.json does not exist for {map_name}: {positions_path}")
    with positions_path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, list):
        raise ValueError(f"expected a JSON list in {positions_path}")

    positions: dict[tuple[str, int], Position] = {}
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ValueError(f"{positions_path}: row {index} is not an object")
        if "file_frame" not in value:
            raise ValueError(f"{positions_path}: row {index} has no file_frame")
        file_frame = str(value["file_frame"])
        file_num, frame_id = _parse_file_frame(file_frame, positions_path, index + 1)
        try:
            coordinates = {
                name: float(value[name])
                for name in ("x", "y", "z", "angle_h", "angle_v")
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{positions_path}: invalid coordinates at row {index}") from exc
        key = (file_num, frame_id)
        if key in positions:
            raise ValueError(f"{positions_path}: duplicate file_frame {file_frame}")
        positions[key] = Position(
            map_name=map_name,
            file_num=file_num,
            frame_id=frame_id,
            file_frame=file_frame,
            x=coordinates["x"],
            y=coordinates["y"],
            z=coordinates["z"],
            yaw=coordinates["angle_h"],
            pitch=coordinates["angle_v"],
            image_path=(
                source_root / map_name / images_dir / f"{file_frame}{image_extension}"
            ),
        )
    return positions


def _context_rows(
    group: JumpGroup,
    positions: Mapping[tuple[str, int], Position],
) -> list[ContextRow]:
    center = group.center_frame
    rows = [
        ContextRow(position=position, types=tuple(sorted(group.frame_types.get(position.frame_id, ()))))
        for (file_num, frame_id), position in positions.items()
        if file_num == group.file_num and abs(frame_id - center) <= group.radius
    ]
    rows.sort(key=lambda row: row.position.frame_id)
    if not any(row.position.frame_id == center for row in rows):
        raise ValueError(
            f"center frame missing from positions: {group.map_name}/"
            f"file_num{group.file_num}_frame_{center}"
        )
    for frame_id in group.anomalous_frames:
        if (group.file_num, frame_id) not in positions:
            raise ValueError(
                f"anomalous endpoint missing from positions: {group.map_name}/"
                f"file_num{group.file_num}_frame_{frame_id}"
            )
    return rows


def _validate_image(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"context image does not exist: {path}")
    try:
        with Image.open(path) as image:
            image.convert("RGB").load()
    except Exception as exc:
        raise OSError(f"context image cannot be decoded: {path}") from exc


def _render_shape(columns: int, dpi: int, max_rows_per_image: int) -> tuple[int, int]:
    if columns < 1:
        raise ValueError("columns must be >= 1")
    if dpi < 1:
        raise ValueError("dpi must be >= 1")
    if max_rows_per_image < 1:
        raise ValueError("max_rows_per_image must be >= 1")
    max_columns = int(MAX_PNG_DIMENSION // (CELL_WIDTH_INCHES * dpi))
    max_rows = int(MAX_PNG_DIMENSION // (CELL_HEIGHT_INCHES * dpi))
    if max_columns < 1 or max_rows < 1:
        raise ValueError(
            f"dpi={dpi} is too high for the PNG dimension limit; lower --dpi"
        )
    return min(columns, max_columns), min(max_rows_per_image, max_rows)


def _partition_rows(
    rows: Sequence[ContextRow], columns: int, max_rows_per_image: int
) -> list[list[ContextRow]]:
    capacity = columns * max_rows_per_image
    return [list(rows[index:index + capacity]) for index in range(0, len(rows), capacity)] or [[]]


def _format_coordinate(value: float) -> str:
    return f"{value:.3f}"


def _display_types(types: Sequence[str]) -> str:
    angle_thresholds: list[int | float] = []
    labels: list[str] = []
    for label in types:
        angle_match = re.fullmatch(r"angle_gt_0*([0-9]+(?:\.[0-9]+)?)", label)
        if angle_match:
            value = float(angle_match.group(1))
            angle_thresholds.append(int(value) if value.is_integer() else value)
            continue
        medium_z_match = re.fullmatch(
            r"medium_z_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)",
            label,
        )
        if medium_z_match:
            labels.append(
                f"{medium_z_match.group(1)}<|dZ|<={medium_z_match.group(2)}"
            )
            continue
        labels.append(label)
    if angle_thresholds:
        labels.append(
            "angle>" + "/".join(str(value) for value in angle_thresholds) + "deg"
        )
    return ", ".join(labels)


def _render_page(
    group: JumpGroup,
    rows: Sequence[ContextRow],
    output_path: Path,
    columns: int,
    dpi: int,
    page_number: int,
    page_count: int,
    group_id: int,
) -> None:
    ncols = min(columns, max(1, len(rows)))
    nrows = max(1, math.ceil(len(rows) / ncols))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(CELL_WIDTH_INCHES * ncols, CELL_HEIGHT_INCHES * nrows),
        squeeze=False,
    )
    axes_flat = list(axes.flat)
    for axis in axes_flat:
        axis.axis("off")

    for axis, context_row in zip(axes_flat, rows):
        position = context_row.position
        try:
            with Image.open(position.image_path) as image:
                axis.imshow(image.convert("RGB"))
        except FileNotFoundError:
            plt.close(figure)
            raise
        except Exception as exc:
            plt.close(figure)
            raise OSError(f"context image cannot be rendered: {position.image_path}") from exc

        axis.set_axis_on()
        labels = _display_types(context_row.types) if context_row.types else "normal context"
        labels = "\n".join(textwrap.wrap(labels, width=42))
        yaw_degrees = math.degrees(position.yaw)
        pitch_degrees = math.degrees(position.pitch)
        axis.set_title(
            f"{position.file_frame}\n{labels}\n"
            f"xyz=({_format_coordinate(position.x)}, {_format_coordinate(position.y)}, "
            f"{_format_coordinate(position.z)})\n"
            f"yaw={position.yaw:.6f}rad/{yaw_degrees:.2f}deg\n"
            f"pitch={position.pitch:.6f}rad/{pitch_degrees:.2f}deg",
            fontsize=7,
            pad=3,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        anomaly = bool(context_row.types)
        color = "crimson" if anomaly else "dimgray"
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(2.5 if anomaly else 1.5)

    title_types = _display_types(group.types)
    page_suffix = f" | part={page_number}/{page_count}" if page_count > 1 else ""
    figure.suptitle(
        f"{group.map_name} | center=file_num{group.file_num}_frame_{group.center_frame} "
        f"| radius={group.radius} | group_range={group.min_frame}-{group.max_frame} "
        f"| types={title_types} | group={group_id}{page_suffix}",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{output_path.name}.", suffix=".png",
            dir=output_path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        figure.savefig(temporary_path, format="png", dpi=dpi)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        plt.close(figure)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _input_label(path: Path, kind: str) -> str:
    stem = path.stem
    if kind == "medium_z" and stem.startswith("medium_z_jumps_"):
        return stem.replace("medium_z_jumps_", "medium_z_", 1)
    if kind == "angle":
        match = re.fullmatch(r"angle_jumps_gt_(.+)", stem)
        if match:
            return f"angle_gt_{match.group(1)}"
    return _safe_token(stem)


def _infer_angle_threshold(path: Path) -> float | None:
    match = re.search(r"angle_jumps_gt_([0-9]+(?:\.[0-9]+)?)", path.stem)
    if match is None:
        return None
    return float(match.group(1))


def _resolve_angle_inputs(
    input_dir: Path,
    custom_paths: Sequence[Path] | None,
    requested_thresholds: Sequence[float] | None,
) -> list[Path]:
    if custom_paths:
        paths = [path.resolve() for path in custom_paths]
    else:
        thresholds = (
            list(DEFAULT_ANGLE_THRESHOLDS)
            if requested_thresholds is None
            else list(requested_thresholds)
        )
        paths = [input_dir / DEFAULT_ANGLE_NAMES.get(threshold, f"angle_jumps_gt_{threshold:03g}.csv")
                 for threshold in thresholds]
    if requested_thresholds is None:
        return paths
    wanted = {float(value) for value in requested_thresholds}
    selected: list[Path] = []
    for path in paths:
        threshold = _infer_angle_threshold(path)
        if threshold is None:
            if len(paths) == 1 and len(wanted) == 1:
                selected.append(path)
                continue
            raise ValueError(
                f"cannot infer angle threshold from custom CSV name {path}; "
                "use a name like angle_jumps_gt_120.csv"
            )
        if threshold in wanted:
            selected.append(path)
    return selected


def load_selected_edges(
    input_dir: Path = DEFAULT_INPUT_DIR,
    medium_z_csv: Path | None = None,
    angle_csvs: Sequence[Path] | None = None,
    categories: Sequence[str] = ("medium_z", "angle"),
    angle_thresholds: Sequence[float] | None = None,
) -> tuple[list[Edge], list[dict[str, Any]]]:
    """Load selected CSVs and return deduplicated edges plus input metadata."""

    selected_categories = set(categories)
    unknown = selected_categories.difference({"medium_z", "angle"})
    if unknown:
        raise ValueError(f"unknown categories: {', '.join(sorted(unknown))}")
    input_dir = input_dir.resolve()
    paths_and_kinds: list[tuple[Path, str]] = []
    if "medium_z" in selected_categories:
        medium_path = (medium_z_csv or input_dir / DEFAULT_MEDIUM_Z_NAME).resolve()
        paths_and_kinds.append((medium_path, "medium_z"))
    if "angle" in selected_categories:
        for path in _resolve_angle_inputs(input_dir, angle_csvs, angle_thresholds):
            paths_and_kinds.append((path, "angle"))
    if not paths_and_kinds:
        raise ValueError("at least one category must be selected")

    all_edges: list[Edge] = []
    input_metadata: list[dict[str, Any]] = []
    for path, kind in paths_and_kinds:
        label = _input_label(path, kind)
        edges = _read_csv_edges(path, label)
        all_edges.extend(edges)
        input_metadata.append(
            {
                "kind": kind,
                "label": label,
                "path": str(path),
                "sha256": _sha256(path),
                "edges": len(edges),
                "angle_threshold": _infer_angle_threshold(path) if kind == "angle" else None,
            }
        )
    return deduplicate_edges(all_edges), input_metadata


def _group_stem(group_id: int, group: JumpGroup, page: int, page_count: int) -> str:
    stem = (
        f"group_{group_id:06d}_{_safe_token(group.map_name)}_"
        f"file_num{_safe_token(group.file_num)}_frames_{group.min_frame}_{group.max_frame}"
    )
    if page_count > 1:
        stem += f"_part_{page:03d}of{page_count:03d}"
    return stem


def render_jump_contexts(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_dir: Path = Path("data/csgo_benchmark_v2/audit/jump_contexts"),
    input_dir: Path = DEFAULT_INPUT_DIR,
    medium_z_csv: Path | None = None,
    angle_csvs: Sequence[Path] | None = None,
    categories: Sequence[str] = ("medium_z", "angle"),
    angle_thresholds: Sequence[float] | None = None,
    positions_name: str = DEFAULT_POSITIONS_NAME,
    images_dir: str = DEFAULT_IMAGES_DIR,
    image_extension: str = DEFAULT_IMAGE_EXTENSION,
    columns: int = 4,
    dpi: int = 100,
    max_rows_per_image: int = 20,
    max_groups: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render selected jump groups and return the written manifest object."""

    if max_groups is not None and max_groups < 0:
        raise ValueError("max_groups must be >= 0")
    effective_columns, effective_max_rows = _render_shape(columns, dpi, max_rows_per_image)
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    edges, input_metadata = load_selected_edges(
        input_dir=input_dir,
        medium_z_csv=medium_z_csv,
        angle_csvs=angle_csvs,
        categories=categories,
        angle_thresholds=angle_thresholds,
    )
    all_groups = merge_edges_into_groups(edges)
    groups = all_groups if max_groups is None else all_groups[:max_groups]

    current_map: str | None = None
    current_positions: dict[tuple[str, int], Position] = {}
    contexts: list[tuple[int, JumpGroup, list[ContextRow], list[list[ContextRow]], list[Path]]] = []
    for group_id, group in enumerate(groups, start=1):
        if group.map_name != current_map:
            current_map = group.map_name
            current_positions = _load_positions(
                source_root,
                group.map_name,
                positions_name,
                images_dir,
                image_extension,
            )
        context = _context_rows(group, current_positions)
        pages = _partition_rows(context, effective_columns, effective_max_rows)
        page_paths = [
            output_dir / f"{_group_stem(group_id, group, page, len(pages))}.png"
            for page in range(1, len(pages) + 1)
        ]
        contexts.append((group_id, group, context, pages, page_paths))

    manifest_path = output_dir / "jump_context_manifest.json"
    groups_path = output_dir / "jump_context_groups.csv"
    output_paths = [path for _, _, _, _, paths in contexts for path in paths]
    existing = [path for path in [*output_paths, manifest_path, groups_path] if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "jump context output exists; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    # Validate every image before creating any PNG, so missing context images
    # cannot turn into a partially successful run with silently missing cells.
    validated_images: set[Path] = set()
    for _, _, context, _, _ in contexts:
        for context_row in context:
            image_path = context_row.position.image_path
            if image_path not in validated_images:
                _validate_image(image_path)
                validated_images.add(image_path)

    page_details: dict[int, list[dict[str, Any]]] = {}
    for group_id, group, context, pages, page_paths in contexts:
        for page_number, (page_rows, page_path) in enumerate(zip(pages, page_paths), start=1):
            _render_page(
                group,
                page_rows,
                page_path,
                effective_columns,
                dpi,
                page_number,
                len(pages),
                group_id,
            )
        page_details[group_id] = [
            {
                "part": page_number,
                "path": str(path.relative_to(output_dir)),
                "rows": len(page_rows),
                "sha256": _sha256(path),
            }
            for page_number, (page_rows, path) in enumerate(zip(pages, page_paths), start=1)
        ]

    group_rows: list[dict[str, Any]] = []
    manifest_groups: list[dict[str, Any]] = []
    for group_id, group, context, pages, page_paths in contexts:
        pages_manifest = page_details[group_id]
        group_rows.append(
            {
                "group_id": group_id,
                "map": group.map_name,
                "file_num": group.file_num,
                "center_file_frame": f"file_num{group.file_num}_frame_{group.center_frame}",
                "center_frame": group.center_frame,
                "radius": group.radius,
                "group_min_frame": group.min_frame,
                "group_max_frame": group.max_frame,
                "anomalous_frames": json.dumps(group.anomalous_frames),
                "types": json.dumps(group.types),
                "edge_count": len(group.edges),
                "context_frame_count": len(context),
                "page_count": len(page_paths),
                "png_files": json.dumps([item["path"] for item in pages_manifest]),
            }
        )
        manifest_groups.append(
            {
                "group_id": group_id,
                "map": group.map_name,
                "file_num": group.file_num,
                "center_file_frame": f"file_num{group.file_num}_frame_{group.center_frame}",
                "center_frame": group.center_frame,
                "radius": group.radius,
                "group_range": [group.min_frame, group.max_frame],
                "anomalous_frames": group.anomalous_frames,
                "types": group.types,
                "frame_types": {
                    f"file_num{group.file_num}_frame_{frame_id}": sorted(labels)
                    for frame_id, labels in sorted(group.frame_types.items())
                },
                "edge_count": len(group.edges),
                "context_frame_count": len(context),
                "pages": pages_manifest,
            }
        )

    _atomic_csv(groups_path, GROUP_COLUMNS, group_rows)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "tool": "render_csgo_benchmark_v2_jump_contexts",
        "semantics": {
            "endpoint_frames": "both endpoints of every selected edge are anomalous frames",
            "edge_union": "duplicate map/file_num/frame_start/frame_end edges are unioned across inputs",
            "grouping": "same map and file_num; sorted edge intervals overlap or next start minus current end is <= 2",
            "center": "anomalous frame nearest (group_min_frame + group_max_frame) / 2; ties choose smaller frame",
            "radius": "max(center - group_min_frame, group_max_frame - center) + 8",
            "angle_units": (
                "positions angle_h and angle_v are rendered in source radians "
                "and converted degrees"
            ),
        },
        "inputs": {
            "source_root": str(source_root),
            "positions_name": positions_name,
            "images_dir": images_dir,
            "image_extension": image_extension,
            "input_dir": str(input_dir.resolve()),
            "categories": list(categories),
            "angle_thresholds": list(angle_thresholds) if angle_thresholds is not None else None,
            "files": input_metadata,
        },
        "render": {
            "columns_requested": columns,
            "columns_effective": effective_columns,
            "dpi": dpi,
            "max_rows_per_image_requested": max_rows_per_image,
            "max_rows_per_image_effective": effective_max_rows,
            "max_groups": max_groups,
            "headless_backend": matplotlib.get_backend(),
        },
        "counts": {
            "input_edges_after_union": len(edges),
            "all_groups_before_max_groups": len(all_groups),
            "groups_rendered": len(groups),
            "pages_rendered": len(output_paths),
            "context_frames_rendered": sum(len(context) for _, _, context, _, _ in contexts),
            "anomalous_frames_rendered": sum(
                len(group.anomalous_frames) for _, group, _, _, _ in contexts
            ),
        },
        "groups_csv": {
            "path": str(groups_path.relative_to(output_dir)),
            "sha256": _sha256(groups_path),
        },
        "groups": manifest_groups,
    }
    _atomic_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--medium-z-csv", type=Path)
    parser.add_argument("--angle-csv", type=Path, action="append", dest="angle_csvs")
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=("medium_z", "angle"),
        default=["medium_z", "angle"],
        help="categories to render; default: medium_z angle",
    )
    parser.add_argument(
        "--angle-thresholds",
        type=float,
        nargs="+",
        help="angle thresholds to select, e.g. 90 120; default: all five files",
    )
    parser.add_argument("--positions-file", default=DEFAULT_POSITIONS_NAME)
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--image-extension", default=DEFAULT_IMAGE_EXTENSION)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--max-rows-per-image", type=int, default=20)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/csgo_benchmark_v2/audit/jump_contexts"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    manifest = render_jump_contexts(
        source_root=args.source_root,
        output_dir=args.output_dir,
        input_dir=args.input_dir,
        medium_z_csv=args.medium_z_csv,
        angle_csvs=args.angle_csvs,
        categories=args.categories,
        angle_thresholds=args.angle_thresholds,
        positions_name=args.positions_file,
        images_dir=args.images_dir,
        image_extension=args.image_extension,
        columns=args.columns,
        dpi=args.dpi,
        max_rows_per_image=args.max_rows_per_image,
        max_groups=args.max_groups,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(f"manifest: {manifest['manifest_path']}")


if __name__ == "__main__":
    main()
