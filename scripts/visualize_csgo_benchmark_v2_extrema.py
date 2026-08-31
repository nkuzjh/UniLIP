#!/usr/bin/env python3
"""Render review context for low-count per-map Z extrema.

The script consumes the review exports produced for CSGO Benchmark v2.  It
does not infer extrema from source data: the summary selects the extrema and
the review CSV supplies the tied target rows.  Source positions are then used
to recover every available frame in the requested same-record context.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, UnidentifiedImageError


FILE_FRAME_RE = re.compile(r"^file_num(?P<file_num>\d+)_frame_(?P<frame>\d+)$")
BOUND_ORDER = {"min": 0, "max": 1}
SUMMARY_FIELDS = {
    "map",
    "z_min",
    "z_min_frame_count",
    "z_max",
    "z_max_frame_count",
}
REVIEW_FIELDS = {"map", "bound", "z", "file_num", "frame", "file_frame"}
CANDIDATE_FIELDS = {
    "map",
    "file_frame",
    "record_id",
    "frame_id",
    "coordinates",
    "reasons",
}
POSITION_FIELDS = {"file_frame", "x", "y", "z", "angle_h", "angle_v"}

TARGET_COLOR = "crimson"
CANDIDATE_COLOR = "darkorange"
NORMAL_COLOR = "slategray"


class VisualizationError(RuntimeError):
    """Raised for invalid inputs or an unrecoverable visualization error."""


@dataclass(frozen=True)
class TargetRow:
    map_name: str
    bound: str
    z: Decimal
    file_num: int
    frame: int
    file_frame: str


@dataclass(frozen=True)
class TargetGroup:
    map_name: str
    bound: str
    z: Decimal
    file_num: int
    target_frames: tuple[int, ...]
    start: int
    end: int
    center: int
    radius: int


@dataclass(frozen=True)
class CandidateInfo:
    map_name: str
    file_num: int
    frame: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PositionFrame:
    map_name: str
    file_num: int
    frame: int
    file_frame: str
    x: Decimal
    y: Decimal
    z: Decimal
    angle_h: Decimal
    angle_v: Decimal
    image_path: Path


@dataclass(frozen=True)
class ContextFrame:
    position: PositionFrame
    target: TargetRow | None
    candidate: CandidateInfo | None


@dataclass(frozen=True)
class GroupPlan:
    group: TargetGroup
    context: tuple[ContextFrame, ...]
    output: str


def parse_file_frame(value: Any, source: str = "file_frame") -> tuple[int, int]:
    """Parse the canonical ``file_numN_frame_M`` identifier."""

    if not isinstance(value, str):
        raise VisualizationError(
            f"{source} must be a string, got {type(value).__name__}"
        )
    match = FILE_FRAME_RE.fullmatch(value)
    if match is None:
        raise VisualizationError(
            f"{source} has malformed file_frame {value!r}; expected file_numN_frame_M"
        )
    return int(match.group("file_num")), int(match.group("frame"))


def _decimal(value: Any, field: str, source: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise VisualizationError(f"{source}: {field} must be a finite number")
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise VisualizationError(f"{source}: invalid {field}={value!r}") from exc
    if not number.is_finite():
        raise VisualizationError(f"{source}: {field} must be finite, got {value!r}")
    return number


def _int(value: Any, field: str, source: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise VisualizationError(f"{source}: {field} must be an integer")
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?\d+", text):
        raise VisualizationError(f"{source}: invalid integer {field}={value!r}")
    number = int(text)
    if minimum is not None and number < minimum:
        raise VisualizationError(
            f"{source}: {field} must be >= {minimum}, got {number}"
        )
    return number


def _load_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise VisualizationError(f"missing CSV file: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing = sorted(required - fields)
            if missing:
                raise VisualizationError(
                    f"{path}: missing CSV columns: {', '.join(missing)}"
                )
            rows: list[dict[str, str]] = []
            for line_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise VisualizationError(
                        f"{path}:{line_number}: malformed CSV row with missing fields"
                    )
                if not any(str(value).strip() for value in row.values()):
                    raise VisualizationError(f"{path}:{line_number}: blank CSV row")
                rows.append({key: str(value) for key, value in row.items()})
    except OSError as exc:
        raise VisualizationError(f"cannot read CSV file {path}: {exc}") from exc
    return rows


def _normalize_maps(values: Sequence[str] | None) -> set[str] | None:
    if values is None:
        return None
    maps: set[str] = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                raise VisualizationError("--maps contains an empty map name")
            maps.add(item)
    if not maps:
        raise VisualizationError("--maps must contain at least one map")
    return maps


def _summary_targets(
    summary_rows: Iterable[Mapping[str, str]],
    tie_count_threshold: int,
    maps: set[str] | None,
) -> dict[tuple[str, str, Decimal], int]:
    if tie_count_threshold <= 0:
        raise VisualizationError("tie-count threshold must be a positive integer")

    selected: dict[tuple[str, str, Decimal], int] = {}
    seen_summary: set[tuple[str, str]] = set()
    available_maps: set[str] = set()
    for row_number, row in enumerate(summary_rows, start=2):
        source = f"z_extrema_summary.csv:{row_number}"
        map_name = row["map"].strip()
        if not map_name:
            raise VisualizationError(f"{source}: map is empty")
        available_maps.add(map_name)
        bound_values = (
            ("min", "z_min", "z_min_frame_count"),
            ("max", "z_max", "z_max_frame_count"),
        )
        for bound, z_field, count_field in bound_values:
            key_without_z = (map_name, bound)
            if key_without_z in seen_summary:
                raise VisualizationError(
                    f"{source}: duplicate summary entry for {map_name}/{bound}"
                )
            z = _decimal(row.get(z_field), z_field, source)
            count = _int(row.get(count_field), count_field, source, minimum=0)
            seen_summary.add(key_without_z)
            if maps is None or map_name in maps:
                if count < tie_count_threshold:
                    selected[(map_name, bound, z)] = count

    if maps is not None:
        unknown = sorted(maps - available_maps)
        if unknown:
            raise VisualizationError(
                "--maps contains names absent from summary: " + ", ".join(unknown)
            )
    return selected


def select_target_rows(
    summary_rows: Iterable[Mapping[str, str]],
    review_rows: Iterable[Mapping[str, str]],
    tie_count_threshold: int = 20,
    maps: set[str] | None = None,
) -> list[TargetRow]:
    """Select review rows whose summary tie count is strictly below threshold."""

    selected_counts = _summary_targets(summary_rows, tie_count_threshold, maps)
    selected_rows: dict[tuple[str, str, Decimal, int, int], TargetRow] = {}
    review_seen: set[tuple[str, str, Decimal, int, int]] = set()

    for row_number, row in enumerate(review_rows, start=2):
        source = f"z_extrema_review.csv:{row_number}"
        map_name = row["map"].strip()
        bound = row["bound"].strip()
        if bound not in BOUND_ORDER:
            raise VisualizationError(
                f"{source}: bound must be min or max, got {bound!r}"
            )
        z = _decimal(row.get("z"), "z", source)
        file_num = _int(row.get("file_num"), "file_num", source, minimum=0)
        frame = _int(row.get("frame"), "frame", source, minimum=0)
        parsed_file_num, parsed_frame = parse_file_frame(row.get("file_frame"), source)
        if (file_num, frame) != (parsed_file_num, parsed_frame):
            raise VisualizationError(
                f"{source}: file_num/frame disagree with file_frame "
                f"{row.get('file_frame')!r}"
            )

        full_key = (map_name, bound, z, file_num, frame)
        if full_key in review_seen:
            raise VisualizationError(f"{source}: duplicate target row {full_key!r}")
        review_seen.add(full_key)
        selected_key = (map_name, bound, z)
        if selected_key in selected_counts:
            selected_rows[full_key] = TargetRow(
                map_name=map_name,
                bound=bound,
                z=z,
                file_num=file_num,
                frame=frame,
                file_frame=str(row["file_frame"]),
            )

    selected_by_key: dict[tuple[str, str, Decimal], list[TargetRow]] = defaultdict(list)
    for target in selected_rows.values():
        selected_by_key[(target.map_name, target.bound, target.z)].append(target)
    for key, expected_count in selected_counts.items():
        actual_count = len(selected_by_key.get(key, []))
        if actual_count != expected_count:
            raise VisualizationError(
                f"review CSV has {actual_count} rows for selected extrema {key!r}; "
                f"summary says {expected_count}"
            )

    return sorted(
        selected_rows.values(),
        key=lambda row: (
            row.map_name,
            BOUND_ORDER[row.bound],
            row.z,
            row.file_num,
            row.frame,
        ),
    )


def group_target_rows(
    target_rows: Iterable[TargetRow], extra_radius: int = 8
) -> list[TargetGroup]:
    """Group only same-key consecutive target frames and compute context bounds."""

    if extra_radius < 0:
        raise VisualizationError("extra radius must be non-negative")
    rows = sorted(
        target_rows,
        key=lambda row: (
            row.map_name,
            row.file_num,
            BOUND_ORDER[row.bound],
            row.z,
            row.frame,
        ),
    )
    groups: list[TargetGroup] = []
    current: list[TargetRow] = []

    def flush() -> None:
        if not current:
            return
        start = current[0].frame
        end = current[-1].frame
        center = (start + end) // 2
        radius = max(center - start, end - center) + extra_radius
        first = current[0]
        groups.append(
            TargetGroup(
                map_name=first.map_name,
                bound=first.bound,
                z=first.z,
                file_num=first.file_num,
                target_frames=tuple(row.frame for row in current),
                start=start,
                end=end,
                center=center,
                radius=radius,
            )
        )

    for row in rows:
        if not current:
            current = [row]
            continue
        previous = current[-1]
        same_key = (
            row.map_name,
            row.file_num,
            row.bound,
            row.z,
        ) == (
            previous.map_name,
            previous.file_num,
            previous.bound,
            previous.z,
        )
        if same_key and row.frame == previous.frame + 1:
            current.append(row)
        else:
            flush()
            current = [row]
    flush()
    return groups


def _load_candidate_lookup(path: Path) -> dict[tuple[str, str], CandidateInfo]:
    if not path.is_file():
        raise VisualizationError(f"missing candidate JSONL file: {path}")
    candidates: dict[tuple[str, str], CandidateInfo] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise VisualizationError(f"{path}:{line_number}: blank JSONL row")
                source = f"{path}:{line_number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise VisualizationError(f"{source}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise VisualizationError(f"{source}: expected a JSON object")
                missing = sorted(CANDIDATE_FIELDS - set(row))
                if missing:
                    raise VisualizationError(
                        f"{source}: missing candidate fields: {', '.join(missing)}"
                    )
                map_name = row["map"]
                if not isinstance(map_name, str) or not map_name.strip():
                    raise VisualizationError(
                        f"{source}: map must be a non-empty string"
                    )
                file_frame = row["file_frame"]
                parsed_file_num, parsed_frame = parse_file_frame(file_frame, source)
                record_id = _int(row["record_id"], "record_id", source, minimum=0)
                frame_id = _int(row["frame_id"], "frame_id", source, minimum=0)
                if (record_id, frame_id) != (parsed_file_num, parsed_frame):
                    raise VisualizationError(
                        f"{source}: record_id/frame_id disagree with file_frame"
                    )
                coordinates = row["coordinates"]
                if not isinstance(coordinates, dict):
                    raise VisualizationError(f"{source}: coordinates must be an object")
                for field in ("x", "y", "z", "angle_h", "angle_v"):
                    _decimal(coordinates.get(field), field, source)
                reasons = row["reasons"]
                if not isinstance(reasons, list) or not all(
                    isinstance(reason, str) and reason for reason in reasons
                ):
                    raise VisualizationError(
                        f"{source}: reasons must be a list of non-empty strings"
                    )
                key = (map_name.strip(), str(file_frame))
                if key in candidates:
                    raise VisualizationError(f"{source}: duplicate candidate {key!r}")
                candidates[key] = CandidateInfo(
                    map_name=map_name.strip(),
                    file_num=record_id,
                    frame=frame_id,
                    reasons=tuple(reasons),
                )
    except OSError as exc:
        raise VisualizationError(f"cannot read candidate JSONL {path}: {exc}") from exc
    return candidates


def _load_positions(
    source_root: Path, map_name: str
) -> dict[tuple[int, int], PositionFrame]:
    positions_path = source_root / map_name / "positions.json"
    if not positions_path.is_file():
        raise VisualizationError(
            f"missing positions file for {map_name}: {positions_path}"
        )
    try:
        with positions_path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise VisualizationError(
            f"cannot read positions file {positions_path}: {exc}"
        ) from exc
    if not isinstance(rows, list):
        raise VisualizationError(f"{positions_path}: expected a JSON list")

    positions: dict[tuple[int, int], PositionFrame] = {}
    for row_number, row in enumerate(rows, start=1):
        source = f"{positions_path}:{row_number}"
        if not isinstance(row, dict):
            raise VisualizationError(f"{source}: expected a JSON object")
        missing = sorted(POSITION_FIELDS - set(row))
        if missing:
            raise VisualizationError(
                f"{source}: missing position fields: {', '.join(missing)}"
            )
        file_frame = row["file_frame"]
        file_num, frame = parse_file_frame(file_frame, source)
        if "map" in row and row["map"] != map_name:
            raise VisualizationError(
                f"{source}: map {row['map']!r} disagrees with directory {map_name!r}"
            )
        values = {
            field: _decimal(row[field], field, source)
            for field in ("x", "y", "z", "angle_h", "angle_v")
        }
        key = (file_num, frame)
        if key in positions:
            raise VisualizationError(f"{source}: duplicate file_frame {file_frame!r}")
        positions[key] = PositionFrame(
            map_name=map_name,
            file_num=file_num,
            frame=frame,
            file_frame=str(file_frame),
            x=values["x"],
            y=values["y"],
            z=values["z"],
            angle_h=values["angle_h"],
            angle_v=values["angle_v"],
            image_path=source_root / map_name / "imgs" / f"{file_frame}.jpg",
        )
    return positions


def _target_lookup(
    target_rows: Iterable[TargetRow],
) -> dict[tuple[str, str], TargetRow]:
    result: dict[tuple[str, str], TargetRow] = {}
    for target in target_rows:
        key = (target.map_name, target.file_frame)
        existing = result.get(key)
        if existing is not None and existing != target:
            raise VisualizationError(f"conflicting extrema targets for {key!r}")
        result[key] = target
    return result


def _context_for_group(
    group: TargetGroup,
    positions: Mapping[tuple[int, int], PositionFrame],
    candidates: Mapping[tuple[str, str], CandidateInfo],
    targets: Mapping[tuple[str, str], TargetRow],
) -> tuple[ContextFrame, ...]:
    context_positions = sorted(
        (
            position
            for (file_num, frame), position in positions.items()
            if file_num == group.file_num
            and group.center - group.radius <= frame <= group.center + group.radius
        ),
        key=lambda position: position.frame,
    )
    if not context_positions:
        raise VisualizationError(
            f"no context positions for {group.map_name}: file_num{group.file_num} "
            f"center={group.center} radius={group.radius}"
        )
    target_frames = set(group.target_frames)
    current_target_keys = {
        (group.map_name, f"file_num{group.file_num}_frame_{frame}")
        for frame in target_frames
    }
    result: list[ContextFrame] = []
    for position in context_positions:
        target = targets.get((group.map_name, position.file_frame))
        is_current_target = (
            position.map_name,
            position.file_frame,
        ) in current_target_keys
        if is_current_target and (
            target is None or target.bound != group.bound or target.z != group.z
        ):
            raise VisualizationError(
                f"target frame missing from review lookup: {group.map_name}/"
                f"{position.file_frame}"
            )
        if target is not None and target.z != position.z:
            raise VisualizationError(
                f"target z disagrees with positions for {group.map_name}/"
                f"{position.file_frame}: target={target.z} position={position.z}"
            )
        if not position.image_path.is_file():
            raise VisualizationError(f"missing context image: {position.image_path}")
        result.append(
            ContextFrame(
                position=position,
                target=target,
                candidate=candidates.get((group.map_name, position.file_frame)),
            )
        )
    actual_target_frames = {
        item.position.frame
        for item in result
        if (item.position.map_name, item.position.file_frame) in current_target_keys
        and item.target is not None
        and item.target.bound == group.bound
        and item.target.z == group.z
    }
    if actual_target_frames != target_frames:
        missing = sorted(target_frames - actual_target_frames)
        raise VisualizationError(
            f"target frames are absent from context for {group.map_name}/"
            f"file_num{group.file_num}: {missing}"
        )
    return tuple(result)


def _z_token(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def _output_name(index: int, group: TargetGroup) -> str:
    return (
        f"{index:04d}_{group.map_name}_{group.bound}_z{_z_token(group.z)}"
        f"_file_num{group.file_num}_frames_{group.start}-{group.end}.png"
    )


def build_plan(
    summary_path: Path,
    review_path: Path,
    candidate_path: Path,
    source_root: Path,
    tie_count_threshold: int = 20,
    extra_radius: int = 8,
    maps: set[str] | None = None,
) -> list[GroupPlan]:
    """Load and validate all inputs, returning the deterministic render plan."""

    summary_rows = _load_csv(summary_path, SUMMARY_FIELDS)
    review_rows = _load_csv(review_path, REVIEW_FIELDS)
    target_rows = select_target_rows(
        summary_rows,
        review_rows,
        tie_count_threshold=tie_count_threshold,
        maps=maps,
    )
    groups = group_target_rows(target_rows, extra_radius=extra_radius)
    candidates = _load_candidate_lookup(candidate_path)
    targets = _target_lookup(target_rows)
    positions_by_map: dict[str, dict[tuple[int, int], PositionFrame]] = {}
    plans: list[GroupPlan] = []

    for index, group in enumerate(groups, start=1):
        if group.map_name not in positions_by_map:
            positions_by_map[group.map_name] = _load_positions(
                source_root, group.map_name
            )
        context = _context_for_group(
            group,
            positions_by_map[group.map_name],
            candidates,
            targets,
        )
        plans.append(
            GroupPlan(group=group, context=context, output=_output_name(index, group))
        )
    return plans


def _number_text(value: Decimal, places: int = 0) -> str:
    if places == 0 and value == value.to_integral_value():
        return str(int(value))
    return f"{float(value):.{places}f}"


def _frame_title(item: ContextFrame) -> tuple[str, str]:
    position = item.position
    if item.target is not None:
        label = (
            f"TARGET EXTREMA | {item.target.bound} | z={_number_text(item.target.z)}"
        )
        color = TARGET_COLOR
        if item.candidate is not None:
            label += " | AUDIT CANDIDATE: " + ", ".join(item.candidate.reasons)
    elif item.candidate is not None:
        label = "AUDIT CANDIDATE | " + ", ".join(item.candidate.reasons)
        color = CANDIDATE_COLOR
    else:
        label = "NORMAL CONTEXT"
        color = NORMAL_COLOR
    title = (
        f"{position.file_frame}\n"
        f"{label}\n"
        f"xyz=({_number_text(position.x)}, {_number_text(position.y)}, "
        f"{_number_text(position.z)})\n"
        f"yaw={float(position.angle_h):.5f} rad / "
        f"{math.degrees(float(position.angle_h)):.2f} deg; "
        f"pitch={float(position.angle_v):.5f} rad / "
        f"{math.degrees(float(position.angle_v)):.2f} deg"
    )
    return title, color


def plot_context(
    plan: GroupPlan,
    columns: int = 4,
    *,
    dpi: int = 200,
) -> plt.Figure:
    """Create a notebook-style context figure for one target group."""

    if columns <= 0:
        raise VisualizationError("columns must be a positive integer")
    if dpi <= 0:
        raise VisualizationError("dpi must be a positive integer")
    if not plan.context:
        raise VisualizationError("cannot plot an empty context")
    ncols = min(columns, len(plan.context))
    nrows = math.ceil(len(plan.context) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 3.6 * nrows),
        squeeze=False,
    )
    for ax in axes.flat:
        ax.axis("off")

    for ax, item in zip(axes.flat, plan.context):
        try:
            with Image.open(item.position.image_path) as image:
                ax.imshow(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as exc:
            raise VisualizationError(
                f"cannot read context image {item.position.image_path}: {exc}"
            ) from exc
        title, color = _frame_title(item)
        ax.set_title(title, fontsize=7.5)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(2.5)

    group = plan.group
    fig.suptitle(
        f"{group.map_name} | {group.bound} extrema z={_number_text(group.z)} | "
        f"file_num{group.file_num} | targets={list(group.target_frames)} | "
        f"center={group.center} radius={group.radius}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


INDEX_FIELDS = (
    "map",
    "bound",
    "z",
    "file_num",
    "target_frames",
    "start",
    "end",
    "center",
    "radius",
    "context_count",
    "output",
)


def _index_rows(plans: Iterable[GroupPlan]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plan in plans:
        group = plan.group
        rows.append(
            {
                "map": group.map_name,
                "bound": group.bound,
                "z": int(group.z)
                if group.z == group.z.to_integral_value()
                else float(group.z),
                "file_num": group.file_num,
                "target_frames": list(group.target_frames),
                "start": group.start,
                "end": group.end,
                "center": group.center,
                "radius": group.radius,
                "context_count": len(plan.context),
                "output": plan.output,
            }
        )
    return rows


def _write_indexes(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = output_dir / "index.csv"
    json_path = output_dir / "index.json"
    try:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=INDEX_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            for row in rows:
                csv_row = dict(row)
                csv_row["target_frames"] = json.dumps(
                    row["target_frames"], separators=(",", ":")
                )
                writer.writerow(csv_row)
        json_path.write_text(
            json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise VisualizationError(
            f"cannot write index files in {output_dir}: {exc}"
        ) from exc


def _check_outputs(
    output_dir: Path, plans: Sequence[GroupPlan], make_pdf: bool, overwrite: bool
) -> None:
    paths = [output_dir / "index.csv", output_dir / "index.json"]
    if make_pdf and plans:
        paths.append(output_dir / "csgo_benchmark_v2_extrema.pdf")
    paths.extend(output_dir / plan.output for plan in plans)
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        preview = ", ".join(str(path) for path in existing[:5])
        suffix = " ..." if len(existing) > 5 else ""
        raise VisualizationError(f"output exists; pass --overwrite: {preview}{suffix}")


def render_plans(
    plans: Sequence[GroupPlan],
    output_dir: Path,
    columns: int,
    dpi: int,
    *,
    dry_run: bool = False,
    no_pdf: bool = False,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Write PNG/PDF outputs and deterministic indexes."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _check_outputs(
        output_dir, plans, make_pdf=not no_pdf and not dry_run, overwrite=overwrite
    )
    rows = _index_rows(plans)
    _write_indexes(output_dir, rows)
    if dry_run or not plans:
        return rows

    pdf_path = output_dir / "csgo_benchmark_v2_extrema.pdf"
    pdf = None if no_pdf else PdfPages(pdf_path)
    try:
        for plan in plans:
            fig = plot_context(plan, columns=columns, dpi=dpi)
            try:
                fig.savefig(output_dir / plan.output, dpi=dpi, bbox_inches="tight")
                if pdf is not None:
                    pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
            finally:
                plt.close(fig)
    except VisualizationError:
        raise
    except (OSError, ValueError) as exc:
        raise VisualizationError(
            f"failed while writing visualization outputs: {exc}"
        ) from exc
    finally:
        if pdf is not None:
            pdf.close()
    return rows


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        "--summary",
        dest="summary_csv",
        type=Path,
        default=root
        / "data/csgo_benchmark_v2/audit/review_exports/z_extrema_summary.csv",
    )
    parser.add_argument(
        "--review-csv",
        "--review",
        dest="review_csv",
        type=Path,
        default=root
        / "data/csgo_benchmark_v2/audit/review_exports/z_extrema_review.csv",
    )
    parser.add_argument(
        "--candidates-jsonl",
        "--candidates",
        dest="candidates_jsonl",
        type=Path,
        default=root / "data/csgo_benchmark_v2/audit/coordinate_candidates.jsonl",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=root / "data/preprocessed_data",
        help="root containing one map directory with positions.json and imgs/",
    )
    parser.add_argument(
        "--tie-count-threshold",
        "--threshold",
        dest="tie_count_threshold",
        type=int,
        default=20,
        help="select extrema with tie count strictly less than this value (default: 20)",
    )
    parser.add_argument("--extra-radius", type=int, default=8)
    parser.add_argument("--maps", nargs="+", metavar="MAP", help="optional map filter")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data/csgo_benchmark_v2/audit/extrema_visualizations",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    return parser


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        maps = _normalize_maps(args.maps)
        if args.columns <= 0:
            raise VisualizationError("columns must be a positive integer")
        if args.dpi <= 0:
            raise VisualizationError("dpi must be a positive integer")
        if args.extra_radius < 0:
            raise VisualizationError("extra radius must be non-negative")
        plans = build_plan(
            summary_path=_resolve_path(args.summary_csv),
            review_path=_resolve_path(args.review_csv),
            candidate_path=_resolve_path(args.candidates_jsonl),
            source_root=_resolve_path(args.source_root),
            tie_count_threshold=args.tie_count_threshold,
            extra_radius=args.extra_radius,
            maps=maps,
        )
        output_dir = _resolve_path(args.output_dir)
        rows = render_plans(
            plans,
            output_dir,
            columns=args.columns,
            dpi=args.dpi,
            dry_run=args.dry_run,
            no_pdf=args.no_pdf,
            overwrite=args.overwrite,
        )
    except VisualizationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"planned {len(rows)} extrema groups")
    if args.dry_run:
        print("dry-run: indexes written; PNG/PDF rendering skipped")
    else:
        print(f"outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
