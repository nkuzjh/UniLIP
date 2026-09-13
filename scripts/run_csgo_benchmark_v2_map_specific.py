#!/usr/bin/env python3
"""Run the reproducible, map-specific Benchmark v2 exp35/exp36/exp36_1 matrix.

The scheduler owns a bounded set of map-specific pipeline slots. Each pipeline
owns its experiment lock and runs every stage serially; the scheduler combines
the configured free-VRAM gates with launch reservations, lifecycle markers,
and bounded pipeline retries before filling another slot.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, TextIO

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "csgo_configs/benchmark_v2_map_specific.yaml"

FAMILIES = ("exp35", "exp36", "exp36_1")
ROUTES = ("joint", "gen", "loc")
FAMILY_ROUTES = {
    "exp35": ("joint", "gen", "loc"),
    "exp36": ("joint", "gen", "loc"),
    "exp36_1": ("joint",),
}
GENERATION_ROUTES = ("joint", "gen")
LOCALIZATION_ROUTES = ("joint", "loc")
GENERATION_KINDS = ("discrete", "continuous")
PIPELINE_STATE_FILENAME = "pipeline_state.json"
TRAINING_PHASE = "train"
INFERENCE_PHASE = "inference"
METRIC_PHASE = "metric"
COMPLETE_PHASE = "complete"
TRAINING_PHASES = frozenset((TRAINING_PHASE, "startup"))
SEEN_MAPS = (
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
CROSSMAP_MAPS = ("cs_office", "de_golden", "de_palacio", "de_vertigo")
EXPECTED_HEADINGS = (
    "# csgo benchmark v2 实验进度",
    "# csgo benchmark v2 主表",
    "# csgo benchmark v2 补充表格",
)
EXPECTED_RESULT_SUBHEADINGS = (
    "## 定位",
    "## 离散生成",
    "## 连续生成",
)
ABLATION_RESULTS_HEADING = "ablation 3 maps表"
ABLATION_RESULTS_UNDERLINE = "===================="
MAIN_RESULTS_HEADING = "# csgo benchmark v2 主表"
SUPPLEMENT_RESULTS_HEADING = "# csgo benchmark v2 补充表格"
RESULT_PARENT_HEADINGS = (MAIN_RESULTS_HEADING, SUPPLEMENT_RESULTS_HEADING)
MUTABLE_RESULT_PARENT_HEADINGS = (
    MAIN_RESULTS_HEADING,
    SUPPLEMENT_RESULTS_HEADING,
)
RESULT_MAIN_SHOT = 100
RESULT_SECTION_BY_KIND = {
    "localization": "## 定位",
    "discrete": "## 离散生成",
    "continuous": "## 连续生成",
}
RESULT_TASK_BY_KIND = {
    "localization": "Localization",
    "discrete": "Discrete generation",
    "continuous": "Continuous generation",
}
RESULT_METRICS = {
    "localization": (("XY_Dist", 3), ("Z_Dist", 3), ("Pitch_Dist", 3), ("Yaw_Dist", 3)),
    "discrete": (
        ("PSNR", 3),
        ("SSIM", 4),
        ("LPIPS", 4),
        ("Boundary_F1", 4),
        ("FID", 3),
    ),
    "continuous": (
        ("PSNR", 3),
        ("SSIM", 4),
        ("LPIPS", 4),
        ("Temporal_Warping_Error", 3),
        ("Temporal_Difference_Error", 3),
        ("FVD", 3),
    ),
}


class PipelineError(RuntimeError):
    """Raised when a configuration, artifact, lock, or child stage is invalid."""


class LockBusy(PipelineError):
    """Raised when a non-blocking pipeline or scheduler lock is already held."""


def _repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _relative(path: Path) -> str:
    absolute = path if path.is_absolute() else REPO_ROOT / path
    absolute = absolute.absolute()
    try:
        return absolute.relative_to(REPO_ROOT.absolute()).as_posix()
    except ValueError:
        return str(absolute)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineError(f"{name} must be a mapping")
    return dict(value)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PipelineError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise PipelineError(f"{name} must be a finite number")
    return number


def load_matrix(
    path: str | os.PathLike[str] = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], Path]:
    config_path = _repo_path(path).resolve()
    if not config_path.is_file():
        raise PipelineError(f"matrix config does not exist: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            matrix = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise PipelineError(f"cannot read matrix config {config_path}: {exc}") from exc
    return _mapping(matrix, "matrix config"), config_path


def _protocol(matrix: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(matrix.get("protocol"), "protocol")


def _paths(matrix: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(matrix.get("paths"), "paths")


def _evaluation(matrix: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(matrix.get("evaluation"), "evaluation")


def _asset_report(matrix: Mapping[str, Any]) -> tuple[Path, dict[str, Any]] | None:
    value = matrix.get("benchmark_v2_asset_manifest")
    if value in (None, ""):
        return None
    if not isinstance(value, (str, os.PathLike)):
        raise PipelineError("benchmark_v2_asset_manifest must be a path or null")
    path = _repo_path(os.fspath(value)).resolve()
    if not path.is_file():
        raise PipelineError(f"benchmark_v2_asset_manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PipelineError(f"invalid benchmark_v2_asset_manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("status") != "verified":
        raise PipelineError(f"benchmark_v2_asset_manifest is not verified: {path}")
    images = payload.get("images")
    if not isinstance(images, Mapping) or images.get("status") != "verified":
        raise PipelineError(f"{path}: verified images section is missing")
    root_value = images.get("root")
    if not isinstance(root_value, str) or not root_value:
        raise PipelineError(f"{path}: images.root must be a non-empty path")
    root = (path.parent / root_value).resolve()
    try:
        root.relative_to(path.parent.resolve())
    except ValueError as exc:
        raise PipelineError(f"{path}: images.root escapes the report directory") from exc
    if not root.is_dir():
        raise PipelineError(f"minimal image root does not exist: {root}")
    return path, dict(payload)


def _asset_expected_provenance(matrix: Mapping[str, Any]) -> dict[str, str] | None:
    report = _asset_report(matrix)
    if report is None:
        return None
    path, payload = report
    selected = payload.get("selected_images")
    if not isinstance(selected, Mapping) or not isinstance(selected.get("sha256"), str):
        raise PipelineError(f"{path}: selected_images.sha256 is missing")
    return {
        "benchmark_v2_asset_manifest": str(path),
        "benchmark_v2_asset_backend": "minimal",
        "benchmark_v2_asset_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "benchmark_v2_selected_images_sha256": str(selected["sha256"]),
    }


def _validate_task_asset_config(matrix: Mapping[str, Any], config_path: Path) -> None:
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PipelineError(f"cannot read task config {config_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PipelineError(f"task config must be a mapping: {config_path}")
    task_value = payload.get("benchmark_v2_asset_manifest")
    matrix_value = matrix.get("benchmark_v2_asset_manifest")
    if task_value in (None, ""):
        return
    if not isinstance(task_value, (str, os.PathLike)):
        raise PipelineError(
            f"{config_path}: benchmark_v2_asset_manifest must be a path or null"
        )
    if matrix_value not in (None, "") and not isinstance(matrix_value, (str, os.PathLike)):
        raise PipelineError(
            "matrix benchmark_v2_asset_manifest must be a path or null"
        )
    if matrix_value in (None, "") or _repo_path(os.fspath(task_value)).resolve() != _repo_path(
        os.fspath(matrix_value)
    ).resolve():
        raise PipelineError(
            f"{config_path}: benchmark_v2_asset_manifest conflicts with matrix switch"
        )


def _append_asset_cli(command: list[str], matrix: Mapping[str, Any]) -> None:
    value = matrix.get("benchmark_v2_asset_manifest")
    if value not in (None, ""):
        command.extend(("--benchmark_v2_asset_manifest", str(value)))


def _asset_provenance_value(
    payload: Mapping[str, Any], keys: Sequence[str], nested_keys: Sequence[str] = ()
) -> Any:
    values: list[Any] = []
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            values.append(value)
    for container_key in ("benchmark_v2_asset", "asset_provenance"):
        nested = payload.get(container_key)
        if not isinstance(nested, Mapping):
            continue
        for key in (*keys, *nested_keys):
            value = nested.get(key)
            if value not in (None, ""):
                values.append(value)
    if len({str(value) for value in values}) > 1:
        raise PipelineError("conflicting asset provenance aliases")
    return values[0] if values else None


def _metric_gt_dir(matrix: Mapping[str, Any], map_name: str) -> str:
    report = _asset_report(matrix)
    if report is None:
        evaluation = _evaluation(matrix)
        return f"{evaluation['data_dir']}/{map_name}/imgs"
    path, payload = report
    images = payload["images"]
    assert isinstance(images, Mapping)
    root = images["root"]
    assert isinstance(root, str)
    return _relative((path.parent / root / map_name).resolve())


def _validate_asset_provenance(
    matrix: Mapping[str, Any], payload: Mapping[str, Any], source_path: Path
) -> None:
    expected = _asset_expected_provenance(matrix)
    if expected is None:
        actual_manifest = _asset_provenance_value(
            payload,
            ("benchmark_v2_asset_manifest", "asset_manifest_path", "asset_manifest"),
            ("manifest", "path"),
        )
        actual_backend = _asset_provenance_value(
            payload,
            ("benchmark_v2_asset_backend", "asset_backend"),
            ("backend",),
        )
        actual_manifest_hash = _asset_provenance_value(
            payload,
            (
                "benchmark_v2_asset_manifest_sha256",
                "asset_manifest_sha256",
                "asset_manifest_hash",
            ),
            ("sha256", "manifest_sha256", "hash"),
        )
        actual_selected_hash = _asset_provenance_value(
            payload,
            (
                "benchmark_v2_selected_images_sha256",
                "selected_images_sha256",
                "selected_image_sha256",
                "selected_images_hash",
            ),
            ("selected_images_sha256", "selected_image_sha256", "selected_images_hash"),
        )
        actual_root = _asset_provenance_value(
            payload,
            ("benchmark_v2_asset_root", "asset_root"),
            ("root",),
        )
        if (
            actual_backend in (None, "")
            and actual_manifest in (None, "")
            and actual_manifest_hash in (None, "")
            and actual_selected_hash in (None, "")
            and actual_root in (None, "")
        ):
            return
        if (
            actual_backend == "source"
            and actual_manifest in (None, "")
            and actual_manifest_hash in (None, "")
            and actual_selected_hash in (None, "")
            and actual_root in (None, "")
        ):
            return
        raise PipelineError(
            f"{source_path}: source backend cannot use minimal asset provenance"
        )
    actual = _asset_provenance_value(
        payload,
        ("benchmark_v2_asset_manifest", "asset_manifest_path", "asset_manifest"),
        ("manifest", "path"),
    )
    if not isinstance(actual, str) or _repo_path(actual).resolve() != Path(
        expected["benchmark_v2_asset_manifest"]
    ).resolve():
        raise PipelineError(f"{source_path}: asset manifest provenance does not match")
    actual_backend = _asset_provenance_value(
        payload,
        ("benchmark_v2_asset_backend", "asset_backend"),
        ("backend",),
    )
    if actual_backend != expected["benchmark_v2_asset_backend"]:
        raise PipelineError(f"{source_path}: asset backend provenance does not match")
    actual_manifest_hash = _asset_provenance_value(
        payload,
        (
            "benchmark_v2_asset_manifest_sha256",
            "asset_manifest_sha256",
            "asset_manifest_hash",
        ),
        ("sha256", "manifest_sha256", "hash"),
    )
    if actual_manifest_hash != expected["benchmark_v2_asset_manifest_sha256"]:
        raise PipelineError(f"{source_path}: asset manifest hash provenance does not match")
    selected_hash = _asset_provenance_value(
        payload,
        (
            "benchmark_v2_selected_images_sha256",
            "selected_images_sha256",
            "selected_image_sha256",
            "selected_images_hash",
        ),
        ("selected_images_sha256", "selected_image_sha256", "selected_images_hash"),
    )
    if selected_hash != expected["benchmark_v2_selected_images_sha256"]:
        raise PipelineError(f"{source_path}: selected image checksum provenance does not match")


def _scheduling(matrix: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(matrix.get("scheduling"), "scheduling")


def _max_active_pipelines(matrix: Mapping[str, Any]) -> int:
    return _integer(
        _scheduling(matrix).get("max_active_pipelines"),
        "max_active_pipelines",
        1,
    )


def _training(matrix: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(matrix.get("training"), "training")


def _spec_list(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = matrix.get("experiments")
    if not isinstance(raw, list):
        raise PipelineError("experiments must be an ordered list")
    result: list[dict[str, Any]] = []
    for index, value in enumerate(raw):
        result.append(_mapping(value, f"experiments[{index}]"))
    return result


def _spec_map(matrix: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    specs = _spec_list(matrix)
    result: dict[str, dict[str, Any]] = {}
    for spec in specs:
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise PipelineError("every experiment needs a non-empty name")
        if name in result:
            raise PipelineError(f"duplicate experiment: {name}")
        result[name] = spec
    return result


def experiment_names(matrix: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(spec["name"]) for spec in _spec_list(matrix))


def _spec(matrix: Mapping[str, Any], experiment: str) -> dict[str, Any]:
    specs = _spec_map(matrix)
    if experiment not in specs:
        raise PipelineError(
            f"unknown map-specific experiment {experiment!r}; expected one of "
            f"{tuple(specs)!r}"
        )
    return specs[experiment]


def _shots(matrix: Mapping[str, Any]) -> list[int]:
    raw = _protocol(matrix).get("shots_per_map")
    if not isinstance(raw, list) or not raw:
        raise PipelineError("protocol.shots_per_map must be a non-empty list")
    values = [_integer(value, "shots_per_map item", 1) for value in raw]
    if values != [100, 50, 20, 10]:
        raise PipelineError("protocol.shots_per_map must be [100, 50, 20, 10]")
    return values


def scheduler_shots(matrix: Mapping[str, Any]) -> list[int]:
    raw = _protocol(matrix).get("scheduler_shots")
    if not isinstance(raw, list) or not raw:
        raise PipelineError("protocol.scheduler_shots must be a non-empty list")
    values = [_integer(value, "scheduler_shots item", 1) for value in raw]
    if any(value not in _shots(matrix) for value in values):
        raise PipelineError("scheduler_shots must be declared by shots_per_map")
    return values


def _route(spec: Mapping[str, Any]) -> str:
    route = spec.get("route")
    if route not in ROUTES:
        raise PipelineError(f"unsupported experiment route: {route!r}")
    return str(route)


def _family(spec: Mapping[str, Any]) -> str:
    family = spec.get("family")
    if family not in FAMILIES:
        raise PipelineError(f"unsupported experiment family: {family!r}")
    return str(family)


def _family_route_supported(family: str, route: str) -> bool:
    return family in FAMILIES and route in FAMILY_ROUTES[family]


def _family_routes(family: str) -> tuple[str, ...]:
    if family not in FAMILIES:
        raise PipelineError(f"unsupported experiment family: {family!r}")
    return FAMILY_ROUTES[family]


def _map_name(spec: Mapping[str, Any]) -> str:
    map_name = spec.get("map")
    if map_name not in CROSSMAP_MAPS:
        raise PipelineError(f"unsupported CrossMap map: {map_name!r}")
    return str(map_name)


def _selected_values(
    values: Sequence[str] | None,
    allowed: Sequence[str],
    name: str,
) -> tuple[str, ...]:
    """Validate an optional CLI filter while preserving its declared order."""

    if values is None:
        return tuple(allowed)
    selected = tuple(values)
    if not selected:
        raise PipelineError(f"{name} needs at least one value")
    unknown = [value for value in selected if value not in allowed]
    if unknown:
        raise PipelineError(
            f"unsupported {name}: {unknown!r}; expected values from {tuple(allowed)!r}"
        )
    if len(set(selected)) != len(selected):
        raise PipelineError(f"{name} contains duplicate values: {selected!r}")
    return selected


def _selected_families(families: Sequence[str] | None) -> tuple[str, ...]:
    return _selected_values(families, FAMILIES, "families")


def _selected_routes(routes: Sequence[str] | None) -> tuple[str, ...]:
    return _selected_values(routes, ROUTES, "routes")


def _shot_index(matrix: Mapping[str, Any], shots: int) -> int:
    values = _shots(matrix)
    if shots not in values:
        raise PipelineError(f"unsupported shot count: {shots}")
    return values.index(shots)


def _artifact_min_bytes(matrix: Mapping[str, Any]) -> int:
    validation = _mapping(matrix.get("artifact_validation"), "artifact_validation")
    return _integer(validation.get("min_model_bytes"), "min_model_bytes", 0)


def _is_nonempty_file(path: Path, minimum_bytes: int = 1) -> bool:
    return path.is_file() and path.stat().st_size >= minimum_bytes


def model_dir(matrix: Mapping[str, Any], experiment: str, shots: int) -> Path:
    seed = _integer(_protocol(matrix).get("support_seed"), "support_seed", 0)
    return (
        _repo_path(str(_paths(matrix)["model_root"]))
        / experiment
        / f"shot_{shots}"
        / f"seed_{seed}"
    )


def checkpoint_path(matrix: Mapping[str, Any], experiment: str, shots: int) -> Path:
    return model_dir(matrix, experiment, shots) / "model.safetensors"


def generation_dir(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    kind: str,
    setting: str = "crossmap",
) -> Path:
    if kind not in GENERATION_KINDS:
        raise PipelineError(f"unsupported generation kind: {kind}")
    if setting not in ("crossmap", "seen"):
        raise PipelineError(f"unsupported generation setting: {setting}")
    seed = _integer(_protocol(matrix).get("support_seed"), "support_seed", 0)
    root = (
        _repo_path(str(_paths(matrix)["generation_root"]))
        / experiment
        / f"shot_{shots}"
        / f"seed_{seed}"
    )
    if setting == "seen":
        return root / f"seen_retention_{kind}"
    return root / kind


def localization_dir(
    matrix: Mapping[str, Any], experiment: str, shots: int, setting: str = "crossmap"
) -> Path:
    if setting not in ("crossmap", "seen"):
        raise PipelineError(f"unsupported localization setting: {setting}")
    seed = _integer(_protocol(matrix).get("support_seed"), "support_seed", 0)
    root = (
        _repo_path(str(_paths(matrix)["localization_root"]))
        / experiment
        / f"shot_{shots}"
        / f"seed_{seed}"
    )
    return root if setting == "crossmap" else root / "seen_retention"


def localization_summary_path(
    matrix: Mapping[str, Any], experiment: str, shots: int, setting: str = "crossmap"
) -> Path:
    return (
        localization_dir(matrix, experiment, shots, setting)
        / "benchmark_csgo_v2_loc.json"
    )


def log_dir(matrix: Mapping[str, Any], experiment: str, shots: int) -> Path:
    seed = _integer(_protocol(matrix).get("support_seed"), "support_seed", 0)
    return (
        _repo_path(str(_paths(matrix)["log_root"]))
        / experiment
        / f"shot_{shots}"
        / f"seed_{seed}"
    )


def pipeline_state_path(matrix: Mapping[str, Any], experiment: str, shots: int) -> Path:
    """Return the atomic lifecycle marker used by the scheduler.

    The marker is deliberately separate from the stage logs.  Logs only show
    that a stage was started, while this file gives the scheduler a current,
    machine-readable phase for training-startup admission control.
    """

    return log_dir(matrix, experiment, shots) / PIPELINE_STATE_FILENAME


def pipeline_lock_path(matrix: Mapping[str, Any], experiment: str, shots: int) -> Path:
    """Return the stable hand-off lock path used by every map-specific pipeline."""

    return log_dir(matrix, experiment, shots) / ".pipeline.lock"


def scheduler_lock_path(matrix: Mapping[str, Any]) -> Path:
    return _repo_path(str(_paths(matrix)["log_root"])) / ".scheduler.lock"


def _stage_phase(stage_type: str) -> str:
    if stage_type == "train":
        return TRAINING_PHASE
    if stage_type.endswith("inference"):
        return INFERENCE_PHASE
    if stage_type.endswith("metric") or stage_type.endswith("aggregate"):
        return METRIC_PHASE
    raise PipelineError(f"unknown stage type for lifecycle state: {stage_type}")


def _write_pipeline_state(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    *,
    status: str,
    phase: str,
    stage: str | None = None,
    error: str | None = None,
) -> None:
    """Atomically publish a pipeline lifecycle state for scheduler adoption."""

    path = pipeline_state_path(matrix, experiment, shots)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment": experiment,
        "shots": shots,
        "pid": os.getpid(),
        "status": status,
        "phase": phase,
        "stage": stage,
        "updated_at": time.time(),
    }
    if error:
        payload["error"] = error[-2000:]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_pipeline_state(
    matrix: Mapping[str, Any], experiment: str, shots: int
) -> dict[str, Any] | None:
    path = pipeline_state_path(matrix, experiment, shots)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _config_value(spec: Mapping[str, Any], key: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value:
        raise PipelineError(f"{spec.get('name')}: missing {key}")
    return value


def _append_cli_mapping(command: list[str], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            command.append(f"--{key}")
            command.extend(str(item) for item in value)
            continue
        if isinstance(value, bool):
            value = "True" if value else "False"
        command.extend((f"--{key}", str(value)))


def _family_args(matrix: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    family_args = _mapping(_training(matrix).get("family_args"), "training.family_args")
    family = _family(spec)
    route = _route(spec)
    family_mapping = _mapping(family_args.get(family), f"training.family_args.{family}")
    return _mapping(family_mapping.get(route), f"training.family_args.{family}.{route}")


def _route_args(matrix: Mapping[str, Any], route: str) -> dict[str, Any]:
    routes = _mapping(_training(matrix).get("routes"), "training.routes")
    return _mapping(routes.get(route), f"training.routes.{route}")


def _port(matrix: Mapping[str, Any], spec: Mapping[str, Any], shots: int) -> int:
    base = _integer(
        spec.get("master_port_base"), f"{spec.get('name')}.master_port_base", 1
    )
    # Keep the recorded 100-shot ports unchanged while reserving a separate
    # four-port block for each supported shot count.
    return base + (_shot_index(matrix, shots) * 100)


def build_train_command(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    *,
    launcher: str | None = None,
) -> list[str]:
    spec = _spec(matrix, experiment)
    route = _route(spec)
    _shot_index(matrix, shots)
    training = _training(matrix)
    common = _mapping(training.get("common_args"), "training.common_args")
    route_args = _route_args(matrix, route)
    family_args = _family_args(matrix, spec)
    train_batch = route_args.get("train_batch_size")
    if train_batch == "shots":
        batch_size = shots
    else:
        batch_size = _integer(train_batch, f"{route}.train_batch_size", 1)
    eval_batch = _integer(
        route_args.get("per_device_eval_batch_size"),
        f"{route}.per_device_eval_batch_size",
        1,
    )
    accumulation = _integer(
        route_args.get("gradient_accumulation_steps"),
        f"{route}.gradient_accumulation_steps",
        1,
    )
    # Always use the interpreter that launched this runner.  This keeps the
    # torch/distributed package and CUDA environment identical across hosts.
    if launcher not in (None, "python_module", "torch.distributed.run"):
        raise PipelineError(
            "training launcher must use the current interpreter via "
            "python -m torch.distributed.run"
        )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        f"--master_port={_port(matrix, spec, shots)}",
        "train_csgo.py",
        "--csgo_config",
        _config_value(spec, "train_config"),
    ]

    # Keep the recorded command order: common data/model arguments, output,
    # step controls, scheduler arguments, route-specific overrides.
    post_output = {
        "num_train_epochs",
        "eval_strategy",
        "save_strategy",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "lr_scheduler_type",
        "lr_scheduler_kwargs",
        "model_max_length",
        "logging_steps",
        "tf32",
        "gradient_checkpointing",
        "dataloader_num_workers",
        "lazy_preprocess",
        "n_query",
        "n_und_query",
        "report_to",
    }
    for key, value in common.items():
        if key not in post_output:
            _append_cli_mapping(command, {key: value})
    command.extend(
        (
            "--output_dir",
            _relative(model_dir(matrix, experiment, shots)),
        )
    )
    if "num_train_epochs" in common:
        _append_cli_mapping(command, {"num_train_epochs": common["num_train_epochs"]})
    command.extend(
        (
            "--per_device_train_batch_size",
            str(batch_size),
            "--per_device_eval_batch_size",
            str(eval_batch),
            "--gradient_accumulation_steps",
            str(accumulation),
            "--max_steps",
            str(_integer(_protocol(matrix).get("max_steps"), "max_steps", 1)),
        )
    )
    for key, value in common.items():
        if key in post_output and key != "num_train_epochs":
            _append_cli_mapping(command, {key: value})
    _append_cli_mapping(command, family_args)
    command.extend(
        (
            "--benchmark_v2_support_seed",
            str(_integer(_protocol(matrix).get("support_seed"), "support_seed", 0)),
            "--benchmark_v2_shots_per_map",
            str(shots),
        )
    )
    _append_asset_cli(command, matrix)
    return command


def _generation_config_key(kind: str) -> str:
    if kind == "discrete":
        return "generation_discrete_config"
    if kind == "continuous":
        return "generation_continuous_config"
    raise PipelineError(f"unsupported generation kind: {kind}")


def _split_for_generation(setting: str, kind: str) -> str:
    if setting == "crossmap":
        return "crossmap_query_test" if kind == "discrete" else "crossmap_continuous"
    if setting == "seen":
        return "seen_discrete_test" if kind == "discrete" else "seen_continuous"
    raise PipelineError(f"unsupported generation setting: {setting}")


def _maps_for_setting(spec: Mapping[str, Any], setting: str) -> list[str]:
    if setting == "crossmap":
        return [_map_name(spec)]
    if setting == "seen":
        return list(SEEN_MAPS)
    raise PipelineError(f"unsupported setting: {setting}")


def build_generation_inference_command(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    kind: str,
    setting: str = "crossmap",
    *,
    python_executable: str | None = None,
) -> list[str]:
    spec = _spec(matrix, experiment)
    if _route(spec) not in GENERATION_ROUTES:
        raise PipelineError(f"{experiment} has no generation route")
    if kind not in GENERATION_KINDS:
        raise PipelineError(f"unsupported generation kind: {kind}")
    config_key = _generation_config_key(kind)
    command = [
        python_executable or sys.executable,
        "eval_csgo.py",
        "--csgo_config",
        _config_value(spec, config_key),
        "--output_dir",
        _relative(generation_dir(matrix, experiment, shots, kind, setting)),
        "--ckpt_path",
        _relative(checkpoint_path(matrix, experiment, shots)),
        "--seed",
        str(_integer(_protocol(matrix).get("inference_seed"), "inference_seed", 0)),
        "--benchmark_v2_split",
        _split_for_generation(setting, kind),
        "--benchmark_v2_support_seed",
        str(_integer(_protocol(matrix).get("support_seed"), "support_seed", 0)),
        "--benchmark_v2_shots_per_map",
        str(shots),
        "--benchmark_v2_maps",
        *_maps_for_setting(spec, setting),
    ]
    _append_asset_cli(command, matrix)
    return command


def build_localization_inference_command(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    setting: str = "crossmap",
    *,
    python_executable: str | None = None,
) -> list[str]:
    spec = _spec(matrix, experiment)
    if _route(spec) not in LOCALIZATION_ROUTES:
        raise PipelineError(f"{experiment} has no localization route")
    split = "crossmap_query_test" if setting == "crossmap" else "seen_discrete_test"
    command = [
        python_executable or sys.executable,
        "eval_csgo_loc.py",
        "--csgo_config",
        _config_value(spec, "localization_config"),
        "--output_dir",
        _relative(localization_dir(matrix, experiment, shots, setting)),
        "--ckpt_path",
        _relative(checkpoint_path(matrix, experiment, shots)),
        "--seed",
        str(_integer(_protocol(matrix).get("inference_seed"), "inference_seed", 0)),
        "--benchmark_v2_split",
        split,
        "--benchmark_v2_support_seed",
        str(_integer(_protocol(matrix).get("support_seed"), "support_seed", 0)),
        "--benchmark_v2_shots_per_map",
        str(shots),
        "--benchmark_v2_maps",
        *_maps_for_setting(spec, setting),
    ]
    _append_asset_cli(command, matrix)
    return command


def build_metric_command(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    kind: str,
    map_name: str,
    setting: str = "crossmap",
    *,
    python_executable: str | None = None,
) -> list[str]:
    spec = _spec(matrix, experiment)
    if _route(spec) not in GENERATION_ROUTES:
        raise PipelineError(f"{experiment} has no generation metric route")
    if kind not in GENERATION_KINDS:
        raise PipelineError(f"unsupported generation kind: {kind}")
    expected_maps = _maps_for_setting(spec, setting)
    if map_name not in expected_maps:
        raise PipelineError(f"{map_name} is not in {setting} maps for {experiment}")
    evaluation = _evaluation(matrix)
    script = (
        "benchmark_csgo_v1.py" if kind == "discrete" else "benchmark_csgo_v1_conti.py"
    )
    output = generation_dir(matrix, experiment, shots, kind, setting)
    command = [
        python_executable or sys.executable,
        script,
        "--gt",
        _metric_gt_dir(matrix, map_name),
        "--pred",
        f"{_relative(output)}/gen_imgs/{map_name}",
        "--batch_size",
        str(_integer(evaluation.get("batch_size"), "evaluation.batch_size", 1)),
        "--device",
        "cuda",
        "--paired_size",
        str(_integer(evaluation.get("paired_size"), "evaluation.paired_size", 1)),
    ]
    if _asset_report(matrix) is None:
        command.extend(("--data_dir", str(evaluation["data_dir"])))
    command.extend(("--map_name", map_name))
    if kind == "continuous":
        command.extend(
            (
                "--frame_diff_threshold",
                str(
                    _integer(
                        evaluation.get("frame_diff_threshold"),
                        "frame_diff_threshold",
                        0,
                    )
                ),
                "--min_track_len",
                str(_integer(evaluation.get("min_track_len"), "min_track_len", 1)),
                "--clip_length",
                str(_integer(evaluation.get("clip_length"), "clip_length", 1)),
                "--clip_stride",
                str(_integer(evaluation.get("clip_stride"), "clip_stride", 1)),
                "--fvd_size",
                str(_integer(evaluation.get("fvd_size"), "fvd_size", 1)),
            )
        )
    command.extend(
        (
            "--benchmark_v2_manifest",
            str(_protocol(matrix)["manifest"]),
            "--benchmark_v2_split",
            _split_for_generation(setting, kind),
            "--external_loc_repo_root",
            str(evaluation["external_loc_repo_root"]),
            "--external_loc_config_path",
            str(evaluation["external_loc_config_path"]),
            "--external_loc_checkpoint_path",
            str(evaluation["external_loc_checkpoint_path"]),
        )
    )
    _append_asset_cli(command, matrix)
    return command


def _metric_path(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    kind: str,
    map_name: str,
    setting: str,
) -> Path:
    stem = "benchmark_csgo_v2" if kind == "discrete" else "benchmark_csgo_v2_conti"
    return (
        generation_dir(matrix, experiment, shots, kind, setting)
        / f"{stem}_{map_name}.json"
    )


def build_maps_aggregate_command(
    matrix: Mapping[str, Any], experiment: str, shots: int, kind: str
) -> list[str]:
    if kind not in GENERATION_KINDS:
        raise PipelineError(f"unsupported generation kind: {kind}")
    output = generation_dir(matrix, experiment, shots, kind, "seen")
    return [
        sys.executable,
        "scripts/aggregate_csgo_benchmark_v2_metrics.py",
        "maps",
        "--manifest",
        str(_protocol(matrix)["manifest"]),
        "--split",
        _split_for_generation("seen", kind),
        "--input_root",
        _relative(output),
        "--kind",
        kind,
        "--output",
        f"{_relative(output)}/summary.json",
    ]


def _family_route_name(family: str, route: str) -> str:
    if route == "joint":
        return family
    return f"{family}_{route}"


def _family_specs(
    matrix: Mapping[str, Any], family: str, route: str
) -> list[dict[str, Any]]:
    return [
        spec
        for spec in _spec_list(matrix)
        if _family(spec) == family and _route(spec) == route
    ]


def family_aggregation_output(
    matrix: Mapping[str, Any], family: str, route: str, shots: int, kind: str
) -> Path:
    family_name = _family_route_name(family, route)
    if kind == "localization":
        root = _repo_path(str(_paths(matrix)["localization_root"]))
        filename = "benchmark_v2_localization_crossmap_query_test.json"
    elif kind in GENERATION_KINDS:
        root = _repo_path(str(_paths(matrix)["generation_root"]))
        filename = f"benchmark_v2_{kind}_crossmap.json"
    else:
        raise PipelineError(f"unsupported family aggregation kind: {kind}")
    seed = _integer(_protocol(matrix).get("support_seed"), "support_seed", 0)
    return (
        root / family_name / f"shot_{shots}" / f"seed_{seed}" / "map_models" / filename
    )


def build_map_models_command(
    matrix: Mapping[str, Any], family: str, route: str, shots: int, kind: str
) -> list[str]:
    if family not in FAMILIES or route not in ROUTES:
        raise PipelineError(f"invalid family route: {family}/{route}")
    if not _family_route_supported(family, route):
        raise PipelineError(f"unsupported family route: {family}/{route}")
    family_name = _family_route_name(family, route)
    if kind == "localization":
        if route not in LOCALIZATION_ROUTES:
            raise PipelineError(f"{family_name} is not localization-capable")
        input_pattern = (
            f"{_relative(_repo_path(str(_paths(matrix)['localization_root'])))}"
            f"/{family_name}_{{map}}/shot_{shots}/seed_{_protocol(matrix)['support_seed']}"
            f"/benchmark_csgo_v2_loc.json"
        )
        split = "crossmap_query_test"
    elif kind in GENERATION_KINDS:
        if route not in GENERATION_ROUTES:
            raise PipelineError(f"{family_name} is not generation-capable")
        subdir = kind
        result_name = (
            "benchmark_csgo_v2_{map}.json"
            if kind == "discrete"
            else "benchmark_csgo_v2_conti_{map}.json"
        )
        input_pattern = (
            f"{_relative(_repo_path(str(_paths(matrix)['generation_root'])))}"
            f"/{family_name}_{{map}}/shot_{shots}/seed_{_protocol(matrix)['support_seed']}"
            f"/{subdir}/{result_name}"
        )
        split = _split_for_generation("crossmap", kind)
    else:
        raise PipelineError(f"unsupported family aggregation kind: {kind}")
    return [
        sys.executable,
        "scripts/aggregate_csgo_benchmark_v2_metrics.py",
        "map-models",
        "--manifest",
        str(_protocol(matrix)["manifest"]),
        "--split",
        split,
        "--kind",
        kind,
        "--input_pattern",
        input_pattern,
        "--output",
        _relative(family_aggregation_output(matrix, family, route, shots, kind)),
    ]


def _expected_sample_count(matrix: Mapping[str, Any], setting: str, kind: str) -> int:
    values = _mapping(_protocol(matrix).get("expected_samples"), "expected_samples")
    split = (
        "crossmap_query_test"
        if setting == "crossmap" and kind in ("discrete", "localization")
        else "crossmap_continuous"
        if setting == "crossmap"
        else "seen_discrete_test"
        if kind in ("discrete", "localization")
        else "seen_continuous"
    )
    return _integer(values.get(split), f"expected_samples.{split}", 1)


def _expected_maps(spec: Mapping[str, Any], setting: str) -> list[str]:
    return _maps_for_setting(spec, setting)


def _path_matches(actual: Any, expected: Path | str) -> bool:
    if not isinstance(actual, str) or not actual:
        return False
    return Path(actual).expanduser().resolve() == _repo_path(str(expected)).resolve()


def _expected_checkpoint_string(
    matrix: Mapping[str, Any], experiment: str, shots: int
) -> str:
    return _relative(checkpoint_path(matrix, experiment, shots))


def _expected_inference_payload(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    kind: str,
    setting: str,
) -> dict[str, Any]:
    spec = _spec(matrix, experiment)
    if kind == "localization":
        config_path = _config_value(spec, "localization_config")
        split = "crossmap_query_test" if setting == "crossmap" else "seen_discrete_test"
    else:
        config_path = _config_value(spec, _generation_config_key(kind))
        split = _split_for_generation(setting, kind)
    return {
        "config_path": config_path,
        "benchmark_v2_manifest": str(_protocol(matrix)["manifest"]),
        "benchmark_v2_split": split,
        "benchmark_v2_support_seed": _integer(
            _protocol(matrix)["support_seed"], "support_seed", 0
        ),
        "benchmark_v2_shots_per_map": shots,
        "maps": _expected_maps(spec, setting),
        "sample_count": _expected_sample_count(matrix, setting, kind),
        "checkpoint": _expected_checkpoint_string(matrix, experiment, shots),
        "ckpt_path": _expected_checkpoint_string(matrix, experiment, shots),
        "seed": _integer(_protocol(matrix)["inference_seed"], "inference_seed", 0),
    }


def _validate_inference_manifest(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    kind: str,
    setting: str,
    path: Path,
) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"invalid inference manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PipelineError(f"inference manifest must be an object: {path}")
    expected = _expected_inference_payload(matrix, experiment, shots, kind, setting)
    for key, value in expected.items():
        actual = payload.get(key)
        if key == "benchmark_v2_manifest":
            matches = _path_matches(actual, _repo_path(str(value)))
        elif key in ("checkpoint", "ckpt_path"):
            matches = _path_matches(actual, _repo_path(str(value)))
        else:
            matches = actual == value
        if not matches:
            raise PipelineError(
                f"{path}: provenance {key}={actual!r}, expected {value!r}"
            )
    _validate_asset_provenance(matrix, payload, path)
    return dict(payload)


def _image_count(path: Path) -> int:
    return sum(1 for suffix in ("*.jpg", "*.jpeg", "*.png") for _ in path.glob(suffix))


def _validate_generation_inference(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    kind: str,
    setting: str,
) -> None:
    output = generation_dir(matrix, experiment, shots, kind, setting)
    manifest_path = output / "inference_manifest.json"
    if not manifest_path.is_file():
        raise PipelineError(f"missing inference manifest: {manifest_path}")
    _validate_inference_manifest(
        matrix, experiment, shots, kind, setting, manifest_path
    )
    per_map_count = _expected_sample_count(matrix, setting, kind)
    maps = _expected_maps(_spec(matrix, experiment), setting)
    expected_per_map = (
        per_map_count // len(maps) if setting == "seen" else per_map_count
    )
    for map_name in maps:
        map_dir = output / "gen_imgs" / map_name
        if not map_dir.is_dir():
            raise PipelineError(f"missing generated image directory: {map_dir}")
        count = _image_count(map_dir)
        if count != expected_per_map:
            raise PipelineError(
                f"{map_dir}: found {count} images, expected {expected_per_map}"
            )


def _validate_provenance_reference(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    kind: str,
    setting: str,
    provenance: Any,
    source_path: Path,
) -> None:
    validation = _mapping(matrix.get("artifact_validation"), "artifact_validation")
    if not validation.get("require_metric_provenance", True):
        return
    if not isinstance(provenance, Mapping):
        raise PipelineError(f"{source_path}: inference_provenance is missing")
    raw_path = provenance.get("path", provenance.get("manifest_path"))
    if not isinstance(raw_path, str) or not raw_path:
        raise PipelineError(f"{source_path}: inference_provenance.path is missing")
    provenance_path = Path(raw_path).expanduser().resolve()
    if not provenance_path.is_file():
        raise PipelineError(
            f"{source_path}: missing referenced inference manifest {provenance_path}"
        )
    embedded = provenance.get("payload")
    if not isinstance(embedded, Mapping):
        raise PipelineError(f"{source_path}: inference_provenance.payload is missing")
    expected = _expected_inference_payload(matrix, experiment, shots, kind, setting)
    for key, value in expected.items():
        actual = embedded.get(key)
        if key == "benchmark_v2_manifest" or key in ("checkpoint", "ckpt_path"):
            matches = _path_matches(actual, _repo_path(str(value)))
        else:
            matches = actual == value
        if not matches:
            raise PipelineError(
                f"{source_path}: embedded provenance {key}={actual!r}, expected {value!r}"
            )
    _validate_asset_provenance(matrix, embedded, source_path)
    _validate_inference_manifest(
        matrix, experiment, shots, kind, setting, provenance_path
    )
    try:
        with provenance_path.open("r", encoding="utf-8") as handle:
            referenced = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(
            f"invalid referenced provenance {provenance_path}: {exc}"
        ) from exc
    if referenced != dict(embedded):
        raise PipelineError(f"{source_path}: embedded and referenced provenance differ")


def _metric_mapping(payload: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    for key in ("metrics_macro_map", "metrics_ordered", "metrics"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    raise PipelineError(f"{path}: no metric mapping found")


def _common_count(payload: Mapping[str, Any]) -> Any:
    for key in ("common_count", "Common_Count"):
        if key in payload:
            return payload[key]
    metrics = payload.get("metrics_ordered", payload.get("metrics"))
    if isinstance(metrics, Mapping):
        return metrics.get("Common_Count", metrics.get("common_count"))
    return None


def _validate_generation_metric(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    kind: str,
    map_name: str,
    setting: str,
) -> dict[str, Any]:
    path = _metric_path(matrix, experiment, shots, kind, map_name, setting)
    if not path.is_file():
        raise PipelineError(f"missing generation metric: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"invalid generation metric {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PipelineError(f"generation metric must be an object: {path}")
    expected_count = _expected_sample_count(matrix, setting, kind)
    expected_per_map = expected_count // len(
        _expected_maps(_spec(matrix, experiment), setting)
    )
    count = _common_count(payload)
    if count != expected_per_map:
        raise PipelineError(
            f"{path}: Common_Count={count}, expected {expected_per_map}"
        )
    expected_split = _split_for_generation(setting, kind)
    if payload.get("map_name") != map_name:
        raise PipelineError(f"{path}: map_name does not match {map_name}")
    if payload.get("benchmark_v2_split") != expected_split:
        raise PipelineError(f"{path}: split does not match {expected_split}")
    if not _path_matches(
        payload.get("benchmark_v2_manifest"),
        _repo_path(str(_protocol(matrix)["manifest"])),
    ):
        raise PipelineError(f"{path}: benchmark_v2_manifest does not match protocol")
    _validate_provenance_reference(
        matrix,
        experiment,
        shots,
        kind,
        setting,
        payload.get("inference_provenance"),
        path,
    )
    for metric_name, _ in RESULT_METRICS[kind]:
        _finite(
            _metric_mapping(payload, path).get(metric_name), f"{path}:{metric_name}"
        )
    return dict(payload)


def _validate_localization_summary(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    setting: str,
) -> dict[str, Any]:
    path = localization_summary_path(matrix, experiment, shots, setting)
    if not path.is_file():
        raise PipelineError(f"missing localization summary: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"invalid localization summary {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PipelineError(f"localization summary must be an object: {path}")
    maps = _expected_maps(_spec(matrix, experiment), setting)
    expected = {
        "split": "crossmap_query_test"
        if setting == "crossmap"
        else "seen_discrete_test",
        "kind": "localization",
        "maps": maps,
        "checkpoint": _expected_checkpoint_string(matrix, experiment, shots),
        "seed": _protocol(matrix)["inference_seed"],
        "support_seed": _protocol(matrix)["support_seed"],
        "shots_per_map": shots,
        "sample_count": _expected_sample_count(matrix, setting, "localization"),
    }
    if not _path_matches(
        payload.get("manifest"), _repo_path(str(_protocol(matrix)["manifest"]))
    ):
        raise PipelineError(f"{path}: manifest does not match protocol")
    for key, value in expected.items():
        actual = payload.get(key)
        if key == "checkpoint":
            matches = _path_matches(actual, _repo_path(str(value)))
        else:
            matches = actual == value
        if not matches:
            raise PipelineError(f"{path}: {key}={actual!r}, expected {value!r}")
    per_map = payload.get("per_map")
    if not isinstance(per_map, Mapping) or list(per_map) != maps:
        raise PipelineError(f"{path}: per_map map order does not match protocol")
    macro = _metric_mapping(payload, path)
    for metric_name, _ in RESULT_METRICS["localization"]:
        _finite(macro.get(metric_name), f"{path}:{metric_name}")
    _validate_provenance_reference(
        matrix,
        experiment,
        shots,
        "localization",
        setting,
        payload.get("inference_provenance"),
        path,
    )
    return dict(payload)


def _validate_seen_generation_aggregate(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    kind: str,
) -> dict[str, Any]:
    path = (
        generation_dir(
            matrix,
            experiment,
            shots,
            "discrete" if kind == "discrete" else "continuous",
            "seen",
        )
        / "summary.json"
    )
    if not path.is_file():
        raise PipelineError(f"missing Seen retention aggregate: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"invalid Seen retention aggregate {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PipelineError(f"Seen retention aggregate must be an object: {path}")
    if not _path_matches(
        payload.get("manifest"), _repo_path(str(_protocol(matrix)["manifest"]))
    ):
        raise PipelineError(f"{path}: manifest does not match protocol")
    expected_kind = kind
    if (
        payload.get("split") != _split_for_generation("seen", kind)
        or payload.get("kind") != expected_kind
    ):
        raise PipelineError(f"{path}: split/kind do not match Seen retention protocol")
    if payload.get("maps") != list(SEEN_MAPS):
        raise PipelineError(f"{path}: map coverage does not match Seen-10")
    macro = _metric_mapping(payload, path)
    for metric_name, _ in RESULT_METRICS[kind]:
        _finite(macro.get(metric_name), f"{path}:{metric_name}")
    return dict(payload)


def _validate_family_summary(
    matrix: Mapping[str, Any], family: str, route: str, shots: int, kind: str
) -> dict[str, Any]:
    path = family_aggregation_output(matrix, family, route, shots, kind)
    if not path.is_file():
        raise PipelineError(f"missing map-models family aggregate: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(
            f"invalid map-models family aggregate {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PipelineError(f"map-models family aggregate must be an object: {path}")
    expected_maps = list(CROSSMAP_MAPS)
    if not _path_matches(
        payload.get("manifest"), _repo_path(str(_protocol(matrix)["manifest"]))
    ):
        raise PipelineError(f"{path}: manifest does not match protocol")
    expected_split = (
        "crossmap_query_test"
        if kind in ("discrete", "localization")
        else "crossmap_continuous"
    )
    if payload.get("split") != expected_split or payload.get("kind") != kind:
        raise PipelineError(f"{path}: split/kind do not match family protocol")
    if payload.get("maps") != expected_maps:
        raise PipelineError(f"{path}: maps do not match CrossMap-4")
    macro = _metric_mapping(payload, path)
    for metric_name, _ in RESULT_METRICS[kind]:
        _finite(macro.get(metric_name), f"{path}:{metric_name}")
    context = payload.get("model_context_by_map")
    if not isinstance(context, Mapping) or list(context) != expected_maps:
        raise PipelineError(f"{path}: model_context_by_map does not preserve map order")
    if payload.get("inference_seed") != _protocol(matrix)["inference_seed"]:
        raise PipelineError(f"{path}: inference_seed does not match protocol")
    if payload.get("support_seed") != _protocol(matrix)["support_seed"]:
        raise PipelineError(f"{path}: support_seed does not match protocol")
    if payload.get("shots_per_map") != shots:
        raise PipelineError(f"{path}: shots_per_map does not match protocol")
    return dict(payload)


def _training_complete(matrix: Mapping[str, Any], experiment: str, shots: int) -> bool:
    directory = model_dir(matrix, experiment, shots)
    model = directory / "model.safetensors"
    state_path = directory / "trainer_state.json"
    if not model.exists() and not state_path.exists():
        return False
    minimum = _artifact_min_bytes(matrix)
    if not _is_nonempty_file(model, minimum):
        raise PipelineError(f"incomplete model artifact: {model}")
    if not state_path.is_file():
        raise PipelineError(f"missing trainer state: {state_path}")
    try:
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"invalid trainer state {state_path}: {exc}") from exc
    expected_steps = _integer(_protocol(matrix).get("max_steps"), "max_steps", 1)
    if (
        state.get("global_step") != expected_steps
        or state.get("max_steps") != expected_steps
    ):
        raise PipelineError(
            f"{state_path}: expected global_step=max_steps={expected_steps}, "
            f"got {state.get('global_step')}/{state.get('max_steps')}"
        )
    return True


def _parent_ready(matrix: Mapping[str, Any], experiment: str) -> bool:
    parent = _repo_path(str(_spec(matrix, experiment).get("parent_checkpoint", "")))
    return _is_nonempty_file(parent, _artifact_min_bytes(matrix))


def _validate_stage_artifact(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    stage: Mapping[str, Any],
) -> None:
    stage_type = stage["type"]
    if stage_type == "train":
        if not _training_complete(matrix, experiment, shots):
            raise PipelineError(
                f"training artifact is incomplete: {experiment}/{shots}"
            )
    elif stage_type == "generation_inference":
        _validate_generation_inference(
            matrix, experiment, shots, str(stage["kind"]), str(stage["setting"])
        )
    elif stage_type == "generation_metric":
        _validate_generation_metric(
            matrix,
            experiment,
            shots,
            str(stage["kind"]),
            str(stage["map"]),
            str(stage["setting"]),
        )
    elif stage_type == "generation_aggregate":
        _validate_seen_generation_aggregate(
            matrix, experiment, shots, str(stage["kind"])
        )
    elif stage_type == "localization_inference":
        _validate_localization_summary(matrix, experiment, shots, str(stage["setting"]))
    else:
        raise PipelineError(f"unknown pipeline stage type: {stage_type}")


def _stage_complete(
    matrix: Mapping[str, Any], experiment: str, shots: int, stage: Mapping[str, Any]
) -> bool:
    stage_type = stage["type"]
    try:
        if stage_type == "train":
            return _training_complete(matrix, experiment, shots)
        if stage_type == "generation_inference":
            output = generation_dir(
                matrix, experiment, shots, str(stage["kind"]), str(stage["setting"])
            )
            if not (output / "inference_manifest.json").is_file():
                return False
            _validate_generation_inference(
                matrix, experiment, shots, str(stage["kind"]), str(stage["setting"])
            )
            return True
        if stage_type == "generation_metric":
            path = _metric_path(
                matrix,
                experiment,
                shots,
                str(stage["kind"]),
                str(stage["map"]),
                str(stage["setting"]),
            )
            if not path.is_file():
                return False
            _validate_generation_metric(
                matrix,
                experiment,
                shots,
                str(stage["kind"]),
                str(stage["map"]),
                str(stage["setting"]),
            )
            return True
        if stage_type == "generation_aggregate":
            output = (
                generation_dir(
                    matrix,
                    experiment,
                    shots,
                    "discrete" if stage["kind"] == "discrete" else "continuous",
                    "seen",
                )
                / "summary.json"
            )
            if not output.is_file():
                return False
            _validate_seen_generation_aggregate(
                matrix, experiment, shots, str(stage["kind"])
            )
            return True
        if stage_type == "localization_inference":
            path = localization_summary_path(
                matrix, experiment, shots, str(stage["setting"])
            )
            if not path.is_file():
                return False
            _validate_localization_summary(
                matrix, experiment, shots, str(stage["setting"])
            )
            return True
    except (FileNotFoundError, PipelineError):
        # A status scan and a resumed pipeline must treat a partial or stale
        # artifact as pending.  The post-command validator still raises on
        # the same artifact, so invalid output cannot be accepted as complete.
        return False
    raise PipelineError(f"unknown pipeline stage type: {stage_type}")


def pipeline_stages(
    matrix: Mapping[str, Any], experiment: str, shots: int
) -> list[dict[str, Any]]:
    """Build the exact serial stage plan for one model pipeline."""

    spec = _spec(matrix, experiment)
    route = _route(spec)
    stages: list[dict[str, Any]] = [
        {
            "label": "train",
            "type": "train",
            "command": build_train_command(matrix, experiment, shots),
        }
    ]
    if route in GENERATION_ROUTES:
        for kind in GENERATION_KINDS:
            stages.append(
                {
                    "label": f"generation_{kind}_crossmap_inference",
                    "type": "generation_inference",
                    "kind": kind,
                    "setting": "crossmap",
                    "command": build_generation_inference_command(
                        matrix, experiment, shots, kind, "crossmap"
                    ),
                }
            )
        for kind in GENERATION_KINDS:
            stages.append(
                {
                    "label": f"generation_{kind}_seen_inference",
                    "type": "generation_inference",
                    "kind": kind,
                    "setting": "seen",
                    "command": build_generation_inference_command(
                        matrix, experiment, shots, kind, "seen"
                    ),
                }
            )
    if route in LOCALIZATION_ROUTES:
        for setting in ("crossmap", "seen"):
            stages.append(
                {
                    "label": f"localization_{setting}_inference_metric",
                    "type": "localization_inference",
                    "setting": setting,
                    "command": build_localization_inference_command(
                        matrix, experiment, shots, setting
                    ),
                }
            )
    if route in GENERATION_ROUTES:
        for kind in GENERATION_KINDS:
            stages.append(
                {
                    "label": f"generation_{kind}_crossmap_metric_{_map_name(spec)}",
                    "type": "generation_metric",
                    "kind": kind,
                    "setting": "crossmap",
                    "map": _map_name(spec),
                    "command": build_metric_command(
                        matrix, experiment, shots, kind, _map_name(spec), "crossmap"
                    ),
                }
            )
        for map_name in SEEN_MAPS:
            stages.append(
                {
                    "label": f"generation_discrete_seen_metric_{map_name}",
                    "type": "generation_metric",
                    "kind": "discrete",
                    "setting": "seen",
                    "map": map_name,
                    "command": build_metric_command(
                        matrix, experiment, shots, "discrete", map_name, "seen"
                    ),
                }
            )
        stages.append(
            {
                "label": "generation_discrete_seen_aggregate",
                "type": "generation_aggregate",
                "kind": "discrete",
                "command": build_maps_aggregate_command(
                    matrix, experiment, shots, "discrete"
                ),
            }
        )
        for map_name in SEEN_MAPS:
            stages.append(
                {
                    "label": f"generation_continuous_seen_metric_{map_name}",
                    "type": "generation_metric",
                    "kind": "continuous",
                    "setting": "seen",
                    "map": map_name,
                    "command": build_metric_command(
                        matrix, experiment, shots, "continuous", map_name, "seen"
                    ),
                }
            )
        stages.append(
            {
                "label": "generation_continuous_seen_aggregate",
                "type": "generation_aggregate",
                "kind": "continuous",
                "command": build_maps_aggregate_command(
                    matrix, experiment, shots, "continuous"
                ),
            }
        )
    return stages


def pipeline_complete(matrix: Mapping[str, Any], experiment: str, shots: int) -> bool:
    return all(
        _stage_complete(matrix, experiment, shots, stage)
        for stage in pipeline_stages(matrix, experiment, shots)
    )


def _next_pending_stage(
    matrix: Mapping[str, Any], experiment: str, shots: int
) -> Mapping[str, Any] | None:
    for stage in pipeline_stages(matrix, experiment, shots):
        if not _stage_complete(matrix, experiment, shots, stage):
            return stage
    return None


def _phase_from_next_pending_stage(
    matrix: Mapping[str, Any], experiment: str, shots: int
) -> str:
    stage = _next_pending_stage(matrix, experiment, shots)
    return COMPLETE_PHASE if stage is None else _stage_phase(str(stage["type"]))


def _pipeline_phase(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    *,
    fallback: str | None = None,
    owner_pid: int | None = None,
) -> str:
    """Read a running phase, falling back to the artifact-derived phase.

    A missing or malformed marker is never treated as evidence that training
    is over.  The artifact-derived phase is conservative and lets an adopted
    legacy pipeline remain compatible with the new scheduler.
    """

    state = _read_pipeline_state(matrix, experiment, shots)
    if state is not None:
        try:
            state_pid = int(state.get("pid", 0))
        except (TypeError, ValueError):
            state_pid = 0
        if owner_pid is not None and state_pid != owner_pid:
            state = None
    if state is not None:
        if state.get("status") == "complete":
            return COMPLETE_PHASE
        phase = state.get("phase")
        if phase in (TRAINING_PHASE, "startup"):
            return TRAINING_PHASE
        if phase in (INFERENCE_PHASE, METRIC_PHASE, COMPLETE_PHASE):
            return str(phase)
    if fallback is not None:
        return fallback
    return _phase_from_next_pending_stage(matrix, experiment, shots)


def _stage_log_name(label: str) -> str:
    return label.replace("/", "_").replace(" ", "_") + ".log"


@contextmanager
def _exclusive_lock(path: Path, *, blocking: bool = False) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(handle.fileno(), flags)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise LockBusy(f"lock is already held: {path}") from exc
            raise
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _lock_is_held(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return True
            raise
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False


def _run_command(command: Sequence[str], log_path: Path, cuda_device: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] COMMAND {shlex.join(command)}\n"
        )
        result = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] EXIT {result.returncode}\n")
    if result.returncode != 0:
        raise PipelineError(
            f"command failed with exit {result.returncode}; see {log_path}"
        )


def run_pipeline(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    *,
    cuda_device: str,
    dry_run: bool = False,
    families: Sequence[str] | None = None,
    routes: Sequence[str] | None = None,
) -> None:
    spec = _spec(matrix, experiment)
    stages = pipeline_stages(matrix, experiment, shots)
    if dry_run:
        for stage in stages:
            print(
                f"[{experiment}/{shots}:{stage['label']}] "
                f"CUDA_VISIBLE_DEVICES={cuda_device} {shlex.join(stage['command'])}"
            )
        return
    if not _parent_ready(matrix, experiment):
        parent = _repo_path(str(spec["parent_checkpoint"]))
        raise PipelineError(f"parent checkpoint is pending for {experiment}: {parent}")
    lock_path = pipeline_lock_path(matrix, experiment, shots)
    with _exclusive_lock(lock_path):
        current_stage: str | None = None
        current_phase = _phase_from_next_pending_stage(matrix, experiment, shots)
        try:
            if pipeline_complete(matrix, experiment, shots):
                _write_pipeline_state(
                    matrix,
                    experiment,
                    shots,
                    status="complete",
                    phase=COMPLETE_PHASE,
                )
                print(f"SKIP complete pipeline {experiment}/{shots}", flush=True)
            else:
                for stage in stages:
                    if _stage_complete(matrix, experiment, shots, stage):
                        print(
                            f"SKIP complete {experiment}/{shots}/{stage['label']}",
                            flush=True,
                        )
                        continue
                    current_stage = str(stage["label"])
                    current_phase = _stage_phase(str(stage["type"]))
                    _write_pipeline_state(
                        matrix,
                        experiment,
                        shots,
                        status="running",
                        phase=current_phase,
                        stage=current_stage,
                    )
                    print(f"START {experiment}/{shots}/{current_stage}", flush=True)
                    _run_command(
                        stage["command"],
                        log_dir(matrix, experiment, shots)
                        / _stage_log_name(current_stage),
                        cuda_device,
                    )
                    _validate_stage_artifact(matrix, experiment, shots, stage)
            sync_results(
                matrix,
                shots=shots,
                initialize=True,
                families=families,
                routes=routes,
            )
            run_family_aggregations(
                matrix,
                shots,
                cuda_device=cuda_device,
                families=families,
                routes=routes,
            )
            sync_results(
                matrix,
                shots=shots,
                initialize=True,
                families=families,
                routes=routes,
            )
            _write_pipeline_state(
                matrix,
                experiment,
                shots,
                status="complete",
                phase=COMPLETE_PHASE,
            )
            print(f"DONE {experiment}/{shots} pipeline", flush=True)
        except Exception as exc:
            _write_pipeline_state(
                matrix,
                experiment,
                shots,
                status="failed",
                phase=current_phase,
                stage=current_stage,
                error=str(exc),
            )
            raise


def _gpu_free_memory_mb(cuda_device: str) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={cuda_device}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise PipelineError(f"nvidia-smi failed: {result.stderr.strip()}")
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (IndexError, ValueError) as exc:
        raise PipelineError(f"cannot parse GPU free memory: {result.stdout!r}") from exc


def _process_tree_pids(pid: int) -> set[int]:
    """Return a process and its descendants without adding a psutil dependency."""

    if pid <= 0:
        return set()
    result = {pid}
    pending = [pid]
    while pending:
        parent = pending.pop()
        children_path = Path(f"/proc/{parent}/task/{parent}/children")
        try:
            children = children_path.read_text(encoding="ascii").split()
        except (OSError, UnicodeError):
            continue
        for value in children:
            try:
                child = int(value)
            except ValueError:
                continue
            if child not in result:
                result.add(child)
                pending.append(child)
    return result


def _gpu_process_memory_mb(cuda_device: str, pid: int) -> int:
    """Return visible GPU memory for a pipeline process tree.

    ``run_pipeline`` owns a Python process which launches torchrun and its
    workers.  Summing the process tree avoids confusing an unrelated process
    on the same GPU with the pipeline whose lifetime reservation is being
    refreshed. A process that has already exited simply reports zero.
    """

    pids = _process_tree_pids(pid)
    if not pids:
        return 0
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={cuda_device}",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return 0
    total = 0
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            process_pid = int(fields[0])
            used_memory = int(fields[1])
        except ValueError:
            continue
        if process_pid in pids:
            total += max(used_memory, 0)
    return total


def _minimum_free_memory(matrix: Mapping[str, Any], family: str, route: str) -> int:
    values = _mapping(
        _scheduling(matrix).get("minimum_free_memory_mb"),
        "minimum_free_memory_mb",
    )
    family_values = _mapping(values.get(family), f"minimum_free_memory_mb.{family}")
    return _integer(
        family_values.get(route),
        f"minimum_free_memory_mb.{family}.{route}",
        1,
    )


def _launch_reservation_memory(
    matrix: Mapping[str, Any], family: str, route: str
) -> int:
    """Return the route-specific target budget for a pipeline reservation.

    Older matrix files did not have this optional mapping.  Falling back to
    the existing free-memory gate keeps those files valid.  The scheduler
    keeps this target for the whole pipeline lifetime and deducts only the
    target minus the process tree's observed GPU memory.
    """

    configured = _scheduling(matrix).get("launch_reservation_memory_mb")
    if configured is None:
        return _minimum_free_memory(matrix, family, route)
    values = _mapping(configured, "launch_reservation_memory_mb")
    family_values = _mapping(
        values.get(family), f"launch_reservation_memory_mb.{family}"
    )
    return _integer(
        family_values.get(route),
        f"launch_reservation_memory_mb.{family}.{route}",
        1,
    )


def _reservation_release_min_gpu_memory_mb(matrix: Mapping[str, Any]) -> int:
    return _integer(
        _scheduling(matrix).get("reservation_release_min_gpu_memory_mb", 1024),
        "reservation_release_min_gpu_memory_mb",
        1,
    )


def _reservation_release_stable_samples(matrix: Mapping[str, Any]) -> int:
    return _integer(
        _scheduling(matrix).get("reservation_release_stable_samples", 2),
        "reservation_release_stable_samples",
        1,
    )


def _pipeline_max_attempts(matrix: Mapping[str, Any]) -> int:
    return _integer(
        _scheduling(matrix).get("pipeline_max_attempts", 3),
        "pipeline_max_attempts",
        1,
    )


def _pipeline_retry_backoff_seconds(matrix: Mapping[str, Any]) -> tuple[int, ...]:
    attempts = _pipeline_max_attempts(matrix)
    raw = _scheduling(matrix).get("pipeline_retry_backoff_seconds")
    if raw is None:
        raw = [30 * (2**index) for index in range(attempts - 1)]
    if not isinstance(raw, list):
        raise PipelineError("pipeline_retry_backoff_seconds must be a list")
    if len(raw) != attempts - 1:
        raise PipelineError(
            "pipeline_retry_backoff_seconds must contain exactly "
            f"pipeline_max_attempts - 1 values (got {len(raw)} for {attempts})"
        )
    return tuple(
        _integer(value, "pipeline_retry_backoff_seconds item", 0) for value in raw
    )


def _effective_minimum_free_memory(
    matrix: Mapping[str, Any],
    route_minimums: Mapping[str, int] | None = None,
    family_route_minimums: Mapping[tuple[str, str], int] | None = None,
) -> dict[tuple[str, str], int]:
    """Resolve nested YAML thresholds plus optional CLI overrides."""

    effective = {
        (family, route): _minimum_free_memory(matrix, family, route)
        for family in FAMILIES
        for route in _family_routes(family)
    }
    if route_minimums:
        for route, value in route_minimums.items():
            if route not in ROUTES:
                raise PipelineError(f"unknown route resource override: {route}")
            checked = _integer(value, f"minimum_free_memory_mb.{route}", 1)
            for family in FAMILIES:
                if _family_route_supported(family, route):
                    effective[(family, route)] = checked
    if family_route_minimums:
        for (family, route), value in family_route_minimums.items():
            if family not in FAMILIES:
                raise PipelineError(f"unknown family resource override: {family}")
            if route not in ROUTES:
                raise PipelineError(f"unknown route resource override: {route}")
            if not _family_route_supported(family, route):
                raise PipelineError(f"unsupported family route: {family}/{route}")
            effective[(family, route)] = _integer(
                value, f"minimum_free_memory_mb.{family}.{route}", 1
            )
    return effective


def partition_pending_jobs(
    matrix: Mapping[str, Any], jobs: Sequence[tuple[str, int]]
) -> tuple[list[tuple[str, int]], set[tuple[str, int]]]:
    """Separate resumable pending jobs from pipelines held by another process."""

    pending: list[tuple[str, int]] = []
    external: set[tuple[str, int]] = set()
    for job in jobs:
        experiment, shots = job
        if _lock_is_held(pipeline_lock_path(matrix, experiment, shots)):
            external.add(job)
            continue
        if pipeline_complete(matrix, experiment, shots):
            continue
        pending.append(job)
    return pending, external


def _lock_owner_pid(path: Path) -> int:
    try:
        first_line = path.read_text(encoding="ascii").splitlines()[0]
        key, value = first_line.split("=", 1)
        if key == "pid":
            return int(value)
    except (OSError, IndexError, ValueError):
        pass
    return 0


def _pipeline_owner_pid(matrix: Mapping[str, Any], experiment: str, shots: int) -> int:
    lock_pid = _lock_owner_pid(pipeline_lock_path(matrix, experiment, shots))
    if lock_pid > 0:
        return lock_pid
    state = _read_pipeline_state(matrix, experiment, shots)
    if state is not None:
        try:
            pid = int(state.get("pid", 0))
        except (TypeError, ValueError):
            pid = 0
        if pid > 0:
            return pid
    return 0


def _confirm_free_memory(
    reader: Callable[[str], int],
    sleeper: Callable[[float], None],
    cuda_device: str,
    minimum: int,
    confirmation_seconds: int,
    reserved_memory_mb: int = 0,
) -> int | None:
    """Return the second sample only when free memory is stable enough.

    The interval is intentionally between the two reads.  A zero interval is
    useful for deterministic tests and preserves the previous immediate
    double-read behavior for explicit callers that need it.
    """

    reserved = _integer(reserved_memory_mb, "reserved_memory_mb", 0)
    first = reader(cuda_device)
    if first - reserved < minimum:
        return None
    if confirmation_seconds:
        sleeper(float(confirmation_seconds))
    second = reader(cuda_device)
    return second if second - reserved >= minimum else None


def select_launch_candidate(
    matrix: Mapping[str, Any],
    pending: Sequence[tuple[str, int]],
    cuda_device: str,
    *,
    free_memory_fn: Callable[[str], int] | None = None,
    launch_memory_confirmation_seconds: int | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    reserved_memory_mb: int = 0,
) -> tuple[tuple[str, int], int] | None:
    """Scan ordered pending jobs and confirm free memory before launch.

    A missing parent is a normal pending state.  A job that does not fit is
    skipped for this scan so a later lower-memory route can run.  Once the
    first free-memory sample reaches the route threshold, the scheduler waits
    for the configured confirmation interval and samples again.  This avoids
    treating a transient CPU-load window during another pipeline's stage
    transition as stable capacity.
    """

    reader = free_memory_fn or _gpu_free_memory_mb
    sleeper = sleep_fn or time.sleep
    confirmation_seconds = (
        _integer(
            launch_memory_confirmation_seconds,
            "launch_memory_confirmation_seconds",
            0,
        )
        if launch_memory_confirmation_seconds is not None
        else _integer(
            _scheduling(matrix).get("launch_memory_confirmation_seconds"),
            "launch_memory_confirmation_seconds",
            0,
        )
    )
    for job in pending:
        experiment, shots = job
        if _lock_is_held(pipeline_lock_path(matrix, experiment, shots)):
            continue
        if not _parent_ready(matrix, experiment):
            continue
        spec = _spec(matrix, experiment)
        family = _family(spec)
        route = _route(spec)
        minimum = _minimum_free_memory(matrix, family, route)
        confirmed = _confirm_free_memory(
            reader,
            sleeper,
            cuda_device,
            minimum,
            confirmation_seconds,
            reserved_memory_mb=reserved_memory_mb,
        )
        if confirmed is None:
            continue
        return job, confirmed
    return None


def _family_complete(
    matrix: Mapping[str, Any], family: str, route: str, shots: int
) -> bool:
    specs = _family_specs(matrix, family, route)
    if len(specs) != len(CROSSMAP_MAPS):
        raise PipelineError(
            f"{family}/{route}: expected {len(CROSSMAP_MAPS)} map models, got {len(specs)}"
        )
    return all(pipeline_complete(matrix, str(spec["name"]), shots) for spec in specs)


def run_family_aggregations(
    matrix: Mapping[str, Any],
    shots: int,
    *,
    cuda_device: str,
    dry_run: bool = False,
    families: Sequence[str] | None = None,
    routes: Sequence[str] | None = None,
) -> list[Path]:
    """Run missing CrossMap map-models aggregates after all four models finish."""

    outputs: list[Path] = []
    selected_families = _selected_families(families)
    selected_routes = _selected_routes(routes)
    for family in selected_families:
        for route in selected_routes:
            if not _family_route_supported(family, route):
                continue
            if route in GENERATION_ROUTES and _family_complete(
                matrix, family, route, shots
            ):
                for kind in GENERATION_KINDS:
                    output = family_aggregation_output(
                        matrix, family, route, shots, kind
                    )
                    if output.is_file():
                        _validate_family_summary(matrix, family, route, shots, kind)
                        continue
                    command = build_map_models_command(
                        matrix, family, route, shots, kind
                    )
                    if dry_run:
                        print(
                            f"[{_family_route_name(family, route)}/{shots}:map-models:{kind}] "
                            f"CUDA_VISIBLE_DEVICES={cuda_device} {shlex.join(command)}"
                        )
                        continue
                    lock = (
                        _repo_path(str(_paths(matrix)["log_root"]))
                        / f".family_{_family_route_name(family, route)}_{shots}_{kind}.lock"
                    )
                    with _exclusive_lock(lock):
                        if output.is_file():
                            _validate_family_summary(matrix, family, route, shots, kind)
                        else:
                            log_path = (
                                _repo_path(str(_paths(matrix)["log_root"]))
                                / _family_route_name(family, route)
                                / f"shot_{shots}"
                                / f"map_models_{kind}.log"
                            )
                            _run_command(command, log_path, cuda_device)
                            _validate_family_summary(matrix, family, route, shots, kind)
                    outputs.append(output)
            if route in LOCALIZATION_ROUTES and _family_complete(
                matrix, family, route, shots
            ):
                output = family_aggregation_output(
                    matrix, family, route, shots, "localization"
                )
                if output.is_file():
                    _validate_family_summary(
                        matrix, family, route, shots, "localization"
                    )
                    continue
                command = build_map_models_command(
                    matrix, family, route, shots, "localization"
                )
                if dry_run:
                    print(
                        f"[{_family_route_name(family, route)}/{shots}:map-models:localization] "
                        f"CUDA_VISIBLE_DEVICES={cuda_device} {shlex.join(command)}"
                    )
                    continue
                lock = (
                    _repo_path(str(_paths(matrix)["log_root"]))
                    / f".family_{_family_route_name(family, route)}_{shots}_localization.lock"
                )
                with _exclusive_lock(lock):
                    if output.is_file():
                        _validate_family_summary(
                            matrix, family, route, shots, "localization"
                        )
                    else:
                        log_path = (
                            _repo_path(str(_paths(matrix)["log_root"]))
                            / _family_route_name(family, route)
                            / f"shot_{shots}"
                            / "map_models_localization.log"
                        )
                        _run_command(command, log_path, cuda_device)
                        _validate_family_summary(
                            matrix, family, route, shots, "localization"
                        )
                outputs.append(output)
    return outputs


def _job_order(
    matrix: Mapping[str, Any],
    shots: int | Sequence[int],
    *,
    families: Sequence[str] | None = None,
    routes: Sequence[str] | None = None,
) -> list[tuple[str, int]]:
    selected_shots = [shots] if isinstance(shots, int) else list(shots)
    for shot in selected_shots:
        _shot_index(matrix, shot)
    selected_families = set(_selected_families(families))
    selected_routes = set(_selected_routes(routes))
    return [
        (str(spec["name"]), shot)
        for shot in selected_shots
        for spec in _spec_list(matrix)
        if _family(spec) in selected_families and _route(spec) in selected_routes
    ]


def _filter_jobs(
    matrix: Mapping[str, Any],
    jobs: Sequence[tuple[str, int]],
    *,
    families: Sequence[str] | None = None,
    routes: Sequence[str] | None = None,
) -> list[tuple[str, int]]:
    selected_families = set(_selected_families(families))
    selected_routes = set(_selected_routes(routes))
    filtered: list[tuple[str, int]] = []
    for experiment, shot in jobs:
        _shot_index(matrix, shot)
        spec = _spec(matrix, experiment)
        if _family(spec) in selected_families and _route(spec) in selected_routes:
            filtered.append((experiment, shot))
    return filtered


def schedule(
    matrix: Mapping[str, Any],
    config_path: Path,
    *,
    cuda_device: str,
    shots: Sequence[int] | None = None,
    families: Sequence[str] | None = None,
    routes: Sequence[str] | None = None,
    route_minimums: Mapping[str, int] | None = None,
    family_route_minimums: Mapping[tuple[str, str], int] | None = None,
    launch_settle_seconds: int | None = None,
    launch_memory_confirmation_seconds: int | None = None,
    poll_seconds: int | None = None,
    stop_launching_on_failure: bool | None = None,
    dry_run: bool = False,
    jobs_override: Sequence[tuple[str, int]] | None = None,
    free_memory_fn: Callable[[str], int] | None = None,
    process_gpu_memory_fn: Callable[[str, int], int] | None = None,
    popen_factory: Callable[..., Any] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> None:
    scheduling = _scheduling(matrix)
    max_active_pipelines = _max_active_pipelines(matrix)
    selected_shots = list(shots) if shots is not None else scheduler_shots(matrix)
    for shot in selected_shots:
        _shot_index(matrix, shot)
    if not selected_shots:
        raise PipelineError("schedule needs at least one shot count")
    selected_families = _selected_families(families)
    selected_routes = _selected_routes(routes)
    launch_settle = (
        _integer(launch_settle_seconds, "launch_settle_seconds", 0)
        if launch_settle_seconds is not None
        else _integer(
            scheduling.get("launch_settle_seconds"), "launch_settle_seconds", 0
        )
    )
    poll = (
        _integer(poll_seconds, "poll_seconds", 0)
        if poll_seconds is not None
        else _integer(scheduling.get("poll_seconds"), "poll_seconds", 0)
    )
    memory_confirmation = (
        _integer(
            launch_memory_confirmation_seconds,
            "launch_memory_confirmation_seconds",
            0,
        )
        if launch_memory_confirmation_seconds is not None
        else _integer(
            scheduling.get("launch_memory_confirmation_seconds"),
            "launch_memory_confirmation_seconds",
            0,
        )
    )
    effective_minimums = _effective_minimum_free_memory(
        matrix,
        route_minimums=route_minimums,
        family_route_minimums=family_route_minimums,
    )
    max_attempts = _pipeline_max_attempts(matrix)
    retry_backoff_seconds = _pipeline_retry_backoff_seconds(matrix)
    if dry_run:
        jobs = _filter_jobs(
            matrix,
            jobs_override
            or _job_order(
                matrix,
                selected_shots,
                families=selected_families,
                routes=selected_routes,
            ),
            families=selected_families,
            routes=selected_routes,
        )
        for experiment, shot in jobs:
            for stage in pipeline_stages(matrix, experiment, shot):
                print(
                    f"[{experiment}/{shot}:{stage['label']}] "
                    f"CUDA_VISIBLE_DEVICES={cuda_device} {shlex.join(stage['command'])}"
                )
        for shot in selected_shots:
            run_family_aggregations(
                matrix,
                shot,
                cuda_device=cuda_device,
                dry_run=True,
                families=selected_families,
                routes=selected_routes,
            )
        return

    stop_on_failure = (
        bool(stop_launching_on_failure)
        if stop_launching_on_failure is not None
        else bool(scheduling.get("stop_launching_on_failure", True))
    )
    reader = free_memory_fn or _gpu_free_memory_mb
    process_memory_reader = process_gpu_memory_fn or _gpu_process_memory_mb
    spawn = popen_factory or subprocess.Popen
    sleeper = sleep_fn or time.sleep
    jobs = _filter_jobs(
        matrix,
        jobs_override
        or _job_order(
            matrix,
            selected_shots,
            families=selected_families,
            routes=selected_routes,
        ),
        families=selected_families,
        routes=selected_routes,
    )
    active: dict[tuple[str, int], tuple[Any, TextIO]] = {}
    # Keep a fixed route-specific target for each live pipeline and derive the
    # remaining reservation from its observed process-tree memory.  This keeps
    # future growth protected without double-counting memory already included
    # in nvidia-smi's global free-memory reading.
    reservation_targets: dict[tuple[str, int], int] = {}
    reservations: dict[tuple[str, int], int] = {}
    reservation_samples: dict[tuple[str, int], int] = {}
    reservation_confirmed: set[tuple[str, int]] = set()
    owner_pids: dict[tuple[str, int], int] = {}
    fallback_phases: dict[tuple[str, int], str] = {}
    failures: list[tuple[str, int, int | None, str]] = []
    attempts: dict[tuple[str, int], int] = {}
    retry_not_before: dict[tuple[str, int], float] = {}
    pending: list[tuple[str, int]]
    external: set[tuple[str, int]]
    with _exclusive_lock(scheduler_lock_path(matrix)):
        # Make the result document ready for incremental updates before any
        # child is launched.  The results lock in sync_results prevents a
        # concurrent generation-runner update from being lost.
        for shot in selected_shots:
            sync_results(
                matrix,
                shots=shot,
                initialize=True,
                families=selected_families,
                routes=selected_routes,
            )
        pending, external = partition_pending_jobs(matrix, jobs)
        print(f"PENDING map-specific pipelines: {pending}", flush=True)
        if external:
            print(f"ADOPT held pipelines: {sorted(external)}", flush=True)

        def drop_reservation(job: tuple[str, int]) -> None:
            reservation_targets.pop(job, None)
            reservations.pop(job, None)
            reservation_samples.pop(job, None)
            reservation_confirmed.discard(job)

        def establish_reservation(job: tuple[str, int]) -> None:
            experiment, shot = job
            spec = _spec(matrix, experiment)
            target = _launch_reservation_memory(matrix, _family(spec), _route(spec))
            reservation_targets[job] = target
            reservations[job] = target
            reservation_samples[job] = 0
            reservation_confirmed.discard(job)

        def adopt_external_job(job: tuple[str, int]) -> None:
            experiment, shot = job
            fallback_phase = _phase_from_next_pending_stage(matrix, experiment, shot)
            fallback_phases[job] = fallback_phase
            owner_pids[job] = _pipeline_owner_pid(matrix, experiment, shot)
            drop_reservation(job)
            # Adopted pipelines need the same lifetime reservation as newly
            # launched pipelines, including inference and metric phases.  A
            # missing owner PID leaves the full target in place conservatively.
            establish_reservation(job)

        for job in external:
            adopt_external_job(job)
        last_launch = 0.0

        def refresh_reservations() -> None:
            stable_samples = _reservation_release_stable_samples(matrix)
            activation_memory = _reservation_release_min_gpu_memory_mb(matrix)
            for job, target in list(reservation_targets.items()):
                pid = owner_pids.get(job, 0)
                observed = process_memory_reader(cuda_device, pid) if pid else 0
                if job not in reservation_confirmed:
                    if observed >= activation_memory:
                        reservation_samples[job] = reservation_samples.get(job, 0) + 1
                    else:
                        reservation_samples[job] = 0
                    # Keep the full target until the first visible reading is
                    # stable.  This preserves the old startup guard while the
                    # target remains alive for later stage transitions.
                    if reservation_samples[job] < stable_samples:
                        reservations[job] = target
                        continue
                    reservation_confirmed.add(job)
                remaining = max(target - max(observed, 0), 0)
                previous = reservations.get(job, target)
                reservations[job] = remaining
                if remaining != previous:
                    print(
                        f"UPDATE reservation {job[0]}/{job[1]} "
                        f"gpu_memory_mb={observed} target_mb={target} "
                        f"reserved_mb={remaining}",
                        flush=True,
                    )

        def gpu_invisible_training_jobs() -> set[tuple[str, int]]:
            startup_jobs: set[tuple[str, int]] = set()
            for job in (*active.keys(), *external):
                owner_pid = owner_pids.get(job, 0)
                if owner_pid <= 0:
                    # A legacy lock cannot prove that a training startup has
                    # reached the GPU, but completed training artifacts do
                    # prove that inference/metric work is past W&B startup.
                    if fallback_phases.get(job) in TRAINING_PHASES:
                        startup_jobs.add(job)
                    continue
                phase = _pipeline_phase(
                    matrix,
                    job[0],
                    job[1],
                    fallback=fallback_phases.get(job),
                    owner_pid=owner_pid,
                )
                if (
                    phase in TRAINING_PHASES
                    and job in reservation_targets
                    and job not in reservation_confirmed
                ):
                    startup_jobs.add(job)
            return startup_jobs

        def record_child_failure(
            job: tuple[str, int],
            returncode: int | None,
            message: str,
        ) -> None:
            attempt = attempts.get(job, 0) + 1
            attempts[job] = attempt
            if attempt < max_attempts:
                backoff = retry_backoff_seconds[attempt - 1]
                retry_not_before[job] = time.monotonic() + backoff
                pending.insert(0, job)
                print(
                    f"RETRY {job[0]}/{job[1]} attempt={attempt + 1}/{max_attempts} "
                    f"backoff_seconds={backoff} exit={returncode}: {message}",
                    flush=True,
                )
                return
            final_message = f"{message} (exhausted attempts={attempt}/{max_attempts})"
            failures.append((job[0], job[1], returncode, final_message))
            print(
                f"FAILED {job[0]}/{job[1]} exhausted attempts={attempt}/"
                f"{max_attempts} exit={returncode}: {message}",
                flush=True,
            )

        while pending or active or external:
            for job, (process, handle) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                handle.close()
                del active[job]
                drop_reservation(job)
                owner_pids.pop(job, None)
                fallback_phases.pop(job, None)
                experiment, shot = job
                if returncode == 0:
                    try:
                        if not pipeline_complete(matrix, experiment, shot):
                            raise PipelineError(
                                "child exited successfully but pipeline artifacts are incomplete"
                            )
                        sync_results(
                            matrix,
                            shots=shot,
                            initialize=True,
                            families=selected_families,
                            routes=selected_routes,
                        )
                        run_family_aggregations(
                            matrix,
                            shot,
                            cuda_device=cuda_device,
                            families=selected_families,
                            routes=selected_routes,
                        )
                        sync_results(
                            matrix,
                            shots=shot,
                            initialize=True,
                            families=selected_families,
                            routes=selected_routes,
                        )
                    except PipelineError as exc:
                        record_child_failure(job, returncode, str(exc))
                    else:
                        attempts.pop(job, None)
                        retry_not_before.pop(job, None)
                        print(f"COMPLETE pipeline {experiment}/{shot}", flush=True)
                else:
                    record_child_failure(job, returncode, "child process failed")

            for job in list(external):
                lock = pipeline_lock_path(matrix, *job)
                if _lock_is_held(lock):
                    continue
                external.remove(job)
                drop_reservation(job)
                owner_pids.pop(job, None)
                fallback_phases.pop(job, None)
                try:
                    adopted_complete = pipeline_complete(matrix, *job)
                except PipelineError as exc:
                    # A released lock means ownership is available again.  A
                    # partially written stage is therefore re-entered by the
                    # normal artifact-resume path instead of aborting the
                    # scheduler while inspecting the hand-off.
                    adopted_complete = False
                    print(
                        f"REQUEUE released pipeline {job[0]}/{job[1]} "
                        f"after incomplete artifact: {exc}",
                        flush=True,
                    )
                if adopted_complete:
                    sync_results(
                        matrix,
                        shots=job[1],
                        initialize=True,
                        families=selected_families,
                        routes=selected_routes,
                    )
                    run_family_aggregations(
                        matrix,
                        job[1],
                        cuda_device=cuda_device,
                        families=selected_families,
                        routes=selected_routes,
                    )
                    sync_results(
                        matrix,
                        shots=job[1],
                        initialize=True,
                        families=selected_families,
                        routes=selected_routes,
                    )
                    print(f"COMPLETE adopted pipeline {job[0]}/{job[1]}", flush=True)
                elif job not in pending:
                    # Preserve stage contiguity for a manually handed-off
                    # pipeline.  Its completed training artifact must be
                    # followed by its own inference/metric stages before the
                    # scheduler considers any later map.
                    pending.insert(0, job)
                    print(
                        f"REQUEUE released pipeline {job[0]}/{job[1]} at queue head",
                        flush=True,
                    )

            refresh_reservations()
            halt = bool(failures and stop_on_failure)
            # Fill every free slot one pipeline at a time.  Each candidate is
            # checked against live free memory minus outstanding launch
            # reservations.  This prevents a CPU-only startup window from
            # looking like capacity for several more W&B initializations.
            while pending and not halt:
                if len(active) + len(external) >= max_active_pipelines:
                    break
                now = time.monotonic()
                if now - last_launch < launch_settle:
                    break

                candidate: tuple[tuple[str, int], int, int] | None = None
                reserved_memory_mb = sum(reservations.values())
                startup_jobs = gpu_invisible_training_jobs()
                for job in list(pending):
                    experiment, shot = job
                    if time.monotonic() < retry_not_before.get(job, 0.0):
                        continue
                    if _lock_is_held(pipeline_lock_path(matrix, experiment, shot)):
                        pending.remove(job)
                        external.add(job)
                        # The lock may have appeared after the initial queue
                        # partition. Adopt it fully so its process tree and
                        # lifetime target reservation are tracked normally.
                        adopt_external_job(job)
                        reserved_memory_mb = sum(reservations.values())
                        startup_jobs = gpu_invisible_training_jobs()
                        continue
                    if not _parent_ready(matrix, experiment):
                        continue
                    spec = _spec(matrix, experiment)
                    family = _family(spec)
                    route = _route(spec)
                    candidate_phase = _phase_from_next_pending_stage(
                        matrix, experiment, shot
                    )
                    if candidate_phase in TRAINING_PHASES and startup_jobs:
                        continue
                    minimum = effective_minimums[(family, route)]
                    confirmed = _confirm_free_memory(
                        reader,
                        sleeper,
                        cuda_device,
                        minimum,
                        memory_confirmation,
                        reserved_memory_mb=reserved_memory_mb,
                    )
                    if confirmed is None:
                        continue
                    candidate = (job, confirmed, minimum)
                    break

                if candidate is None:
                    break
                if len(active) + len(external) >= max_active_pipelines:
                    break

                job, free_mb, minimum = candidate
                pending.remove(job)
                experiment, shot = job
                pipeline_log = log_dir(matrix, experiment, shot) / "scheduler.log"
                pipeline_log.parent.mkdir(parents=True, exist_ok=True)
                handle = pipeline_log.open("a", encoding="utf-8", buffering=1)
                # Recheck directly before Popen; another process may have
                # consumed the free memory during candidate bookkeeping.  The
                # virtual reservation is still applied if the child has not
                # yet become visible to nvidia-smi.
                latest_free_mb = reader(cuda_device)
                available_free_mb = latest_free_mb - reserved_memory_mb
                if available_free_mb < minimum:
                    handle.close()
                    pending.append(job)
                    print(
                        f"DEFER {experiment}/{shot}: free memory dropped "
                        f"to {latest_free_mb} MB, available after reservations "
                        f"{available_free_mb} MB (need {minimum} MB)",
                        flush=True,
                    )
                    sleeper(poll)
                    continue
                free_mb = latest_free_mb
                child_command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--config",
                    str(config_path),
                    "run",
                    "--experiment",
                    experiment,
                    "--shots",
                    str(shot),
                    "--cuda-device",
                    str(cuda_device),
                ]
                _append_asset_cli(child_command, matrix)
                if families is not None:
                    child_command.extend(["--families", *selected_families])
                if routes is not None:
                    child_command.extend(["--routes", *selected_routes])
                environment = dict(os.environ)
                environment["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
                environment["PYTHONUNBUFFERED"] = "1"
                try:
                    process = spawn(
                        child_command,
                        cwd=REPO_ROOT,
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except Exception:
                    handle.close()
                    raise
                active[job] = (process, handle)
                owner_pids[job] = int(getattr(process, "pid", 0) or 0)
                fallback_phases[job] = _phase_from_next_pending_stage(
                    matrix, experiment, shot
                )
                establish_reservation(job)
                last_launch = time.monotonic()
                print(
                    f"LAUNCH {experiment}/{shot} pid={getattr(process, 'pid', '?')} "
                    f"free_mb_before_spawn={free_mb} "
                    f"target_reservation_mb={reservation_targets[job]} "
                    f"reserved_mb={reservations[job]} "
                    f"active={len(active) + len(external)}/{max_active_pipelines}",
                    flush=True,
                )
            if halt and not active and not external:
                break
            if pending or active or external:
                sleeper(poll)
    if failures:
        rendered = ", ".join(
            f"{name}/{shot}:exit={code}:{message}"
            for name, shot, code, message in failures
        )
        raise PipelineError(f"map-specific schedule failed: {rendered}")
    print("DONE map-specific schedule", flush=True)


def _heading_level(line: str) -> int:
    body = line.rstrip("\r\n")
    if not body.startswith("#"):
        return 0
    level = len(body) - len(body.lstrip("#"))
    return level if len(body) > level and body[level] == " " else 0


def _section_bounds(
    lines: Sequence[str], heading: str, *, parent_heading: str | None = None
) -> tuple[int, int]:
    """Return a heading's range, optionally scoped to a top-level parent.

    The results document intentionally has repeated ``## 定位`` headings.  A
    parent-qualified lookup prevents a 50-shot update from landing in the
    first (100-shot) table.
    """

    bodies = [line.rstrip("\r\n") for line in lines]
    if parent_heading is None:
        search_start = 0
        search_end = len(lines)
    else:
        search_start, search_end = _section_bounds(lines, parent_heading)
        search_start += 1
    try:
        start = next(
            index
            for index in range(search_start, search_end)
            if bodies[index] == heading
        )
    except StopIteration as exc:
        scope = f" under {parent_heading}" if parent_heading else ""
        raise PipelineError(f"results heading is missing: {heading}{scope}") from exc

    level = _heading_level(bodies[start])
    end = search_end
    for index in range(start + 1, search_end):
        candidate_level = _heading_level(bodies[index])
        if candidate_level and candidate_level <= level:
            end = index
            break
    return start, end


def _result_headings(lines: Sequence[str]) -> list[str]:
    return [line.rstrip("\r\n") for line in lines if _heading_level(line) == 1]


def _ablation_section_bounds(lines: Sequence[str]) -> tuple[int, int]:
    """Locate the required Setext ablation block without treating it as ATX."""

    bodies = [line.rstrip("\r\n") for line in lines]
    title_indices = [
        index
        for index, body in enumerate(bodies)
        if body == ABLATION_RESULTS_HEADING
    ]
    underline_indices = [
        index
        for index, body in enumerate(bodies)
        if body == ABLATION_RESULTS_UNDERLINE
    ]
    marker_indices = [
        index
        for index in title_indices
        if index + 1 < len(bodies)
        and bodies[index + 1] == ABLATION_RESULTS_UNDERLINE
    ]
    if (
        len(title_indices) != 1
        or len(underline_indices) != 1
        or len(marker_indices) != 1
    ):
        raise PipelineError(
            "results file must contain exactly one Setext ablation marker"
        )

    progress_index = bodies.index(EXPECTED_HEADINGS[0])
    main_index = bodies.index(MAIN_RESULTS_HEADING)
    marker_index = marker_indices[0]
    if not progress_index < marker_index < main_index:
        raise PipelineError(
            "Setext ablation marker must be between experiment progress and main results"
        )
    return marker_index, main_index


def _validate_result_subheadings(
    lines: Sequence[str], start: int, end: int, parent: str
) -> None:
    headings = []
    for line in lines[start:end]:
        level = _heading_level(line)
        if level >= 2:
            headings.append(line.rstrip("\r\n"))
    if headings != list(EXPECTED_RESULT_SUBHEADINGS):
        raise PipelineError(
            f"results file subsections changed under {parent}; refusing to rewrite"
        )


def _validate_results_structure(lines: Sequence[str]) -> None:
    if _result_headings(lines) != list(EXPECTED_HEADINGS):
        raise PipelineError(
            "results file top-level headings changed; refusing to rewrite"
        )
    ablation_start, main_start = _ablation_section_bounds(lines)
    _validate_result_subheadings(
        lines, ablation_start + 2, main_start, ABLATION_RESULTS_HEADING
    )
    for parent in RESULT_PARENT_HEADINGS:
        start, end = _section_bounds(lines, parent)
        _validate_result_subheadings(lines, start + 1, end, parent)


def _line_ending(lines: Sequence[str]) -> str:
    for line in lines:
        if line.endswith("\r\n"):
            return "\r\n"
        if line.endswith("\n"):
            return "\n"
    return "\n"


def _row_prefix(setting: str, kind: str, experiment: str, shots: int) -> str:
    return f"| {setting} | {RESULT_TASK_BY_KIND[kind]} | {experiment} | {shots} |"


def _blank_result_row(
    setting: str,
    kind: str,
    experiment: str,
    shots: int,
    *,
    checkpoint_step: str | None = None,
) -> str:
    values = ["" for _ in RESULT_METRICS[kind]]
    if checkpoint_step is not None:
        values.append(checkpoint_step)
    return (
        _row_prefix(setting, kind, experiment, shots) + " " + " | ".join(values) + " |"
    )


def _result_row(
    setting: str,
    kind: str,
    experiment: str,
    shots: int,
    values: Sequence[str],
    *,
    checkpoint_step: str | None = None,
) -> str:
    rendered_values = list(values)
    if checkpoint_step is not None:
        rendered_values.append(checkpoint_step)
    return (
        _row_prefix(setting, kind, experiment, shots)
        + " "
        + " | ".join(rendered_values)
        + " |"
    )


def _progress_prefix(experiment: str, shots: int) -> str:
    return f"| `{experiment}` {shots}-shot |"


def _insert_or_replace_table_row(
    lines: list[str],
    heading: str,
    prefix: str,
    replacement: str,
    *,
    initialize: bool,
    parent_heading: str | None = None,
) -> bool:
    start, end = _section_bounds(lines, heading, parent_heading=parent_heading)
    ending = _line_ending(lines)
    for index in range(start + 1, end):
        if lines[index].rstrip("\r\n").startswith(prefix):
            lines[index] = replacement + ending
            return True
    if not initialize:
        return False
    row_indices = [
        index
        for index in range(start + 1, end)
        if lines[index].rstrip("\r\n").startswith("|")
    ]
    if not row_indices:
        raise PipelineError(f"results table is missing under {heading}")
    insert_at = row_indices[-1] + 1
    lines.insert(insert_at, replacement + ending)
    return True


def _insert_or_replace_result_row(
    lines: list[str],
    heading: str,
    prefix: str,
    replacement: str | Callable[[str], str],
    *,
    shots: int,
    initialize: bool,
) -> bool:
    """Update a result row wherever it currently lives.

    Users may manually move non-100-shot rows into the main result table.  A
    result sync must honor that placement, while newly initialized rows retain
    the historical default of 100-shot in the main table and other shots in
    the supplement table.
    """

    def render(parent_heading: str) -> str:
        return replacement(parent_heading) if callable(replacement) else replacement

    ending = _line_ending(lines)
    found = False
    for parent_heading in MUTABLE_RESULT_PARENT_HEADINGS:
        start, end = _section_bounds(lines, heading, parent_heading=parent_heading)
        matches = [
            index
            for index in range(start + 1, end)
            if lines[index].rstrip("\r\n").startswith(prefix)
        ]
        if matches:
            found = True
            rendered = render(parent_heading)
            for index in matches:
                lines[index] = rendered + ending

    if found:
        return True

    if not initialize:
        return False
    parent_heading = _results_parent_heading(shots)
    _insert_or_replace_table_row(
        lines,
        heading,
        prefix,
        render(parent_heading),
        initialize=True,
        parent_heading=parent_heading,
    )
    return True


def _metric_values(payload: Mapping[str, Any], kind: str, path: Path) -> list[str]:
    metrics = _metric_mapping(payload, path)
    values: list[str] = []
    for name, digits in RESULT_METRICS[kind]:
        value = _finite(metrics.get(name), f"{path}:{name}")
        values.append(f"{value:.{digits}f}")
    return values


def _read_existing_metric(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    kind: str,
    setting: str,
    map_name: str | None = None,
) -> list[str] | None:
    try:
        if kind == "localization":
            payload = _validate_localization_summary(matrix, experiment, shots, setting)
            return _metric_values(
                payload,
                kind,
                localization_summary_path(matrix, experiment, shots, setting),
            )
        if map_name is None:
            payload = _validate_seen_generation_aggregate(
                matrix, experiment, shots, kind
            )
            path = (
                generation_dir(
                    matrix,
                    experiment,
                    shots,
                    "discrete" if kind == "discrete" else "continuous",
                    "seen",
                )
                / "summary.json"
            )
        else:
            payload = _validate_generation_metric(
                matrix, experiment, shots, kind, map_name, setting
            )
            path = _metric_path(matrix, experiment, shots, kind, map_name, setting)
        return _metric_values(payload, kind, path)
    except PipelineError as exc:
        expected_path = (
            localization_summary_path(matrix, experiment, shots, setting)
            if kind == "localization"
            else _metric_path(
                matrix,
                experiment,
                shots,
                kind,
                map_name or _map_name(_spec(matrix, experiment)),
                setting,
            )
        )
        if not expected_path.exists():
            return None
        raise exc


def _inference_status(
    matrix: Mapping[str, Any], experiment: str, shots: int, route: str
) -> tuple[str, str]:
    parts: list[str] = []
    metrics: list[str] = []
    spec = _spec(matrix, experiment)
    if route in GENERATION_ROUTES:
        cross = [
            _stage_complete(
                matrix,
                experiment,
                shots,
                {
                    "type": "generation_inference",
                    "kind": kind,
                    "setting": "crossmap",
                },
            )
            for kind in GENERATION_KINDS
        ]
        seen = [
            _stage_complete(
                matrix,
                experiment,
                shots,
                {
                    "type": "generation_inference",
                    "kind": kind,
                    "setting": "seen",
                },
            )
            for kind in GENERATION_KINDS
        ]
        if any(cross):
            parts.append(
                "CrossMap-4 generation "
                + ("discrete+continuous" if all(cross) else "partial")
            )
        if any(seen):
            parts.append(
                "Seen-10 retention generation "
                + ("discrete+continuous" if all(seen) else "partial")
            )
        target = _map_name(spec)
        for kind in GENERATION_KINDS:
            try:
                _validate_generation_metric(
                    matrix, experiment, shots, kind, target, "crossmap"
                )
            except PipelineError:
                pass
            else:
                metrics.append(f"CrossMap {kind}")
        for kind in GENERATION_KINDS:
            try:
                _validate_seen_generation_aggregate(matrix, experiment, shots, kind)
            except PipelineError:
                pass
            else:
                metrics.append(f"Seen retention {kind}")
    if route in LOCALIZATION_ROUTES:
        cross = _stage_complete(
            matrix,
            experiment,
            shots,
            {"type": "localization_inference", "setting": "crossmap"},
        )
        seen = _stage_complete(
            matrix,
            experiment,
            shots,
            {"type": "localization_inference", "setting": "seen"},
        )
        if cross:
            parts.append("CrossMap-4 localization")
            metrics.append("CrossMap localization")
        if seen:
            parts.append("Seen-10 retention localization")
            metrics.append("Seen retention localization")
    return "<br>".join(parts), "<br>".join(metrics)


def _progress_cells(
    matrix: Mapping[str, Any], experiment: str, shots: int
) -> tuple[str, str, str]:
    training_partial = False
    try:
        training_complete = _training_complete(matrix, experiment, shots)
    except PipelineError:
        training_complete = False
        training_partial = True
    if training_complete:
        train = f"✅ step {_protocol(matrix)['max_steps']}"
    elif training_partial:
        train = "incomplete training artifact"
    elif not _parent_ready(matrix, experiment):
        train = "等待 parent checkpoint"
    else:
        train = "未开始"
    inference, metric = _inference_status(
        matrix, experiment, shots, _route(_spec(matrix, experiment))
    )
    return train, inference or "未开始", metric or "未开始"


def _new_result_rows(
    matrix: Mapping[str, Any],
    shots: int,
    *,
    families: Sequence[str] | None = None,
    routes: Sequence[str] | None = None,
) -> list[tuple[str, str, str, int, str]]:
    rows: list[tuple[str, str, str, int, str]] = []
    selected_families = _selected_families(families)
    selected_routes = _selected_routes(routes)
    family_set = set(selected_families)
    route_set = set(selected_routes)
    for spec in _spec_list(matrix):
        experiment = str(spec["name"])
        if _family(spec) not in family_set or _route(spec) not in route_set:
            continue
        route = _route(spec)
        if route in LOCALIZATION_ROUTES:
            rows.append(
                ("CrossMap-4 few-shot", "localization", experiment, shots, "model")
            )
            rows.append(
                ("Seen-10 retention", "localization", experiment, shots, "seen")
            )
        if route in GENERATION_ROUTES:
            for kind in GENERATION_KINDS:
                rows.append(("CrossMap-4 few-shot", kind, experiment, shots, "model"))
                rows.append(("Seen-10 retention", kind, experiment, shots, "seen"))
    for family in selected_families:
        for route in selected_routes:
            if not _family_route_supported(family, route):
                continue
            family_name = _family_route_name(family, route)
            if route in LOCALIZATION_ROUTES:
                rows.append(
                    (
                        "CrossMap-4 few-shot",
                        "localization",
                        family_name,
                        shots,
                        "family",
                    )
                )
            if route in GENERATION_ROUTES:
                for kind in GENERATION_KINDS:
                    rows.append(
                        ("CrossMap-4 few-shot", kind, family_name, shots, "family")
                    )
    return rows


def _result_payload(
    matrix: Mapping[str, Any],
    setting: str,
    kind: str,
    experiment: str,
    shots: int,
    row_type: str,
) -> list[str] | None:
    if row_type == "family":
        if experiment.endswith("_gen"):
            family, route = experiment[: -len("_gen")], "gen"
        elif experiment.endswith("_loc"):
            family, route = experiment[: -len("_loc")], "loc"
        else:
            family, route = experiment, "joint"
        output_kind = "localization" if kind == "localization" else kind
        output = family_aggregation_output(matrix, family, route, shots, output_kind)
        if not output.is_file():
            return None
        payload = _validate_family_summary(matrix, family, route, shots, output_kind)
        return _metric_values(payload, output_kind, output)
    spec = _spec(matrix, experiment)
    if kind == "localization":
        return _read_existing_metric(
            matrix,
            experiment,
            shots,
            kind,
            "crossmap" if setting.startswith("CrossMap") else "seen",
        )
    if setting.startswith("CrossMap"):
        return _read_existing_metric(
            matrix, experiment, shots, kind, "crossmap", _map_name(spec)
        )
    return _read_existing_metric(matrix, experiment, shots, kind, "seen", None)


def _results_parent_heading(shots: int) -> str:
    return (
        MAIN_RESULTS_HEADING
        if shots == RESULT_MAIN_SHOT
        else SUPPLEMENT_RESULTS_HEADING
    )


def _sync_results_locked(
    matrix: Mapping[str, Any],
    *,
    shots: int,
    initialize: bool,
    families: Sequence[str] | None = None,
    routes: Sequence[str] | None = None,
) -> None:
    results_path = _repo_path(str(_paths(matrix)["results_file"]))
    if not results_path.is_file():
        raise PipelineError(f"results file does not exist: {results_path}")
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        lines = handle.read().splitlines(keepends=True)
    _validate_results_structure(lines)

    # Progress rows are intentionally one row per model and shot.  The formal
    # initialization creates the default 100-shot rows for every registered
    # model; explicit
    # run --shots 50/20/10 adds only that requested row.
    selected_families = _selected_families(families)
    selected_routes = _selected_routes(routes)
    family_set = set(selected_families)
    route_set = set(selected_routes)
    for spec in _spec_list(matrix):
        experiment = str(spec["name"])
        if _family(spec) not in family_set or _route(spec) not in route_set:
            continue
        train, inference, metric = _progress_cells(matrix, experiment, shots)
        progress = f"| `{experiment}` {shots}-shot | {train} | {inference} | {metric} |"
        _insert_or_replace_table_row(
            lines,
            EXPECTED_HEADINGS[0],
            _progress_prefix(experiment, shots),
            progress,
            initialize=initialize,
        )

    for setting, kind, experiment, shot, row_type in _new_result_rows(
        matrix,
        shots,
        families=selected_families,
        routes=selected_routes,
    ):
        values = _result_payload(matrix, setting, kind, experiment, shot, row_type)

        def render_result_row(parent_heading: str) -> str:
            checkpoint_step = (
                str(_protocol(matrix)["max_steps"])
                if parent_heading == MAIN_RESULTS_HEADING and values is not None
                else "-"
                if parent_heading == MAIN_RESULTS_HEADING
                else None
            )
            if values is not None:
                return _result_row(
                    setting,
                    kind,
                    experiment,
                    shot,
                    values,
                    checkpoint_step=checkpoint_step,
                )
            return _blank_result_row(
                setting,
                kind,
                experiment,
                shot,
                checkpoint_step=checkpoint_step,
            )

        _insert_or_replace_result_row(
            lines,
            RESULT_SECTION_BY_KIND[kind],
            _row_prefix(setting, kind, experiment, shot),
            render_result_row,
            shots=shot,
            initialize=initialize,
        )

    _validate_results_structure(lines)
    rendered = "".join(lines)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=results_path.parent,
            prefix=f".{results_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(rendered)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, results_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def sync_results(
    matrix: Mapping[str, Any],
    *,
    shots: int | None = None,
    initialize: bool = False,
    families: Sequence[str] | None = None,
    routes: Sequence[str] | None = None,
) -> None:
    selected_shots = shots if shots is not None else scheduler_shots(matrix)[0]
    _shot_index(matrix, selected_shots)
    results_path = _repo_path(str(_paths(matrix)["results_file"]))
    # The existing Benchmark v2 runners share this lock.  Wait for their
    # atomic update instead of treating a short-lived writer as a failure.
    with _exclusive_lock(
        results_path.with_suffix(results_path.suffix + ".lock"), blocking=True
    ):
        _sync_results_locked(
            matrix,
            shots=selected_shots,
            initialize=initialize,
            families=families,
            routes=routes,
        )


def validate_configuration(matrix: Mapping[str, Any]) -> None:
    if _integer(matrix.get("schema_version"), "schema_version", 1) != 1:
        raise PipelineError("schema_version must be 1")
    protocol = _protocol(matrix)
    _asset_report(matrix)
    if protocol.get("manifest") is None:
        raise PipelineError("protocol.manifest is required")
    manifest = _repo_path(str(protocol["manifest"]))
    if not manifest.is_file():
        raise PipelineError(f"benchmark manifest does not exist: {manifest}")
    if _integer(protocol.get("support_seed"), "support_seed", 0) != 0:
        raise PipelineError("formal map-specific protocol requires support seed 0")
    if _integer(protocol.get("inference_seed"), "inference_seed", 0) != 42:
        raise PipelineError("formal map-specific protocol requires inference seed 42")
    if _integer(protocol.get("max_steps"), "max_steps", 1) != 400:
        raise PipelineError("formal map-specific protocol requires max_steps=400")
    if protocol.get("crossmap_maps") != list(CROSSMAP_MAPS):
        raise PipelineError("CrossMap map order does not match Benchmark v2")
    if protocol.get("seen_maps") != list(SEEN_MAPS):
        raise PipelineError("Seen-10 map order does not match Benchmark v2")
    if scheduler_shots(matrix) != [100, 50, 20, 10]:
        raise PipelineError(
            "formal scheduler default must preserve launch order [100, 50, 20, 10]"
        )
    _paths(matrix)
    _evaluation(matrix)
    scheduling = _scheduling(matrix)
    _max_active_pipelines(matrix)
    _integer(scheduling.get("launch_settle_seconds"), "launch_settle_seconds", 0)
    _integer(
        scheduling.get("launch_memory_confirmation_seconds"),
        "launch_memory_confirmation_seconds",
        0,
    )
    _reservation_release_min_gpu_memory_mb(matrix)
    _reservation_release_stable_samples(matrix)
    _pipeline_max_attempts(matrix)
    _pipeline_retry_backoff_seconds(matrix)
    _integer(scheduling.get("poll_seconds"), "poll_seconds", 0)
    for family in FAMILIES:
        for route in _family_routes(family):
            _minimum_free_memory(matrix, family, route)
            _launch_reservation_memory(matrix, family, route)
    if scheduling.get("failure_policy") not in ("stop", "continue"):
        raise PipelineError("failure_policy must be stop or continue")

    expected_names: list[str] = []
    for family in FAMILIES:
        for map_name in CROSSMAP_MAPS:
            for route in _family_routes(family):
                expected_names.append(
                    f"{family}_{map_name}"
                    if route == "joint"
                    else f"{family}_{route}_{map_name}"
                )
    if list(experiment_names(matrix)) != expected_names:
        raise PipelineError(
            f"experiment order must be map-section order: expected {expected_names!r}"
        )
    expected_count = sum(
        len(CROSSMAP_MAPS) * len(_family_routes(family)) for family in FAMILIES
    )
    if len(_spec_list(matrix)) != expected_count:
        raise PipelineError(
            f"map-specific matrix must contain {expected_count} model experiments"
        )

    ports: set[int] = set()
    for spec in _spec_list(matrix):
        experiment = str(spec["name"])
        route = _route(spec)
        map_name = _map_name(spec)
        family = _family(spec)
        if not _family_route_supported(family, route):
            raise PipelineError(f"{experiment}: unsupported family/route contract")
        expected_name = (
            f"{family}_{map_name}"
            if route == "joint"
            else f"{family}_{route}_{map_name}"
        )
        if experiment != expected_name:
            raise PipelineError(f"{experiment}: route/name mismatch")
        parent = _repo_path(_config_value(spec, "parent_checkpoint"))
        train_config = _repo_path(_config_value(spec, "train_config"))
        if not train_config.is_file():
            raise PipelineError(f"{experiment}: missing train config {train_config}")
        if not parent.exists():
            # Missing exp31/exp32 joint parents are a supported pending state.
            if parent.name != "model.safetensors":
                raise PipelineError(f"{experiment}: malformed parent checkpoint path")
        try:
            with train_config.open("r", encoding="utf-8") as handle:
                train_data = _mapping(
                    yaml.safe_load(handle), f"{experiment} train config"
                )
            _validate_task_asset_config(matrix, train_config)
        except (OSError, yaml.YAMLError) as exc:
            raise PipelineError(
                f"{experiment}: cannot read train config: {exc}"
            ) from exc
        if not _path_matches(train_data.get("benchmark_v2_manifest"), manifest):
            raise PipelineError(f"{experiment}: train manifest mismatch")
        if train_data.get("benchmark_v2_split") != "crossmap_support":
            raise PipelineError(f"{experiment}: train split must be crossmap_support")
        if (
            train_data.get("benchmark_v2_support_seed") != 0
            or train_data.get("benchmark_v2_shots_per_map") != 100
        ):
            raise PipelineError(
                f"{experiment}: train config must retain seed-0 100-shot defaults"
            )
        if (
            train_data.get("train_maps") != [map_name]
            or train_data.get("val_maps") != [map_name]
            or train_data.get("test_maps") != [map_name]
        ):
            raise PipelineError(
                f"{experiment}: train maps must be the singleton target map"
            )
        configured_parent = train_data.get("finetune_init_ckpt_path")
        if not _path_matches(configured_parent, parent):
            raise PipelineError(
                f"{experiment}: finetune_init_ckpt_path does not match parent"
            )
        if route in GENERATION_ROUTES:
            for kind in GENERATION_KINDS:
                config_path = _repo_path(
                    _config_value(spec, _generation_config_key(kind))
                )
                if not config_path.is_file():
                    raise PipelineError(
                        f"{experiment}: missing generation config {config_path}"
                    )
                with config_path.open("r", encoding="utf-8") as handle:
                    data = _mapping(
                        yaml.safe_load(handle), f"{experiment} {kind} config"
                    )
                _validate_task_asset_config(matrix, config_path)
                if data.get("test_maps") != [map_name] or data.get("val_maps") != [
                    map_name
                ]:
                    raise PipelineError(f"{experiment}: {kind} config map mismatch")
                if data.get("ckpt_path") != _relative(
                    model_dir(matrix, experiment, 100) / "model.safetensors"
                ):
                    raise PipelineError(
                        f"{experiment}: {kind} checkpoint path mismatch"
                    )
                if kind == "continuous" and data.get("is_conti_gen") is not True:
                    raise PipelineError(
                        f"{experiment}: continuous config must enable is_conti_gen"
                    )
                if kind == "discrete" and data.get("is_conti_gen") is True:
                    raise PipelineError(
                        f"{experiment}: discrete config must not enable is_conti_gen"
                    )
        if route in LOCALIZATION_ROUTES:
            config_path = _repo_path(_config_value(spec, "localization_config"))
            if not config_path.is_file():
                raise PipelineError(
                    f"{experiment}: missing localization config {config_path}"
                )
            with config_path.open("r", encoding="utf-8") as handle:
                data = _mapping(
                    yaml.safe_load(handle), f"{experiment} localization config"
                )
            _validate_task_asset_config(matrix, config_path)
            if data.get("test_maps") != [map_name] or data.get("val_maps") != [
                map_name
            ]:
                raise PipelineError(f"{experiment}: localization config map mismatch")
            if data.get("ckpt_path") != _relative(
                model_dir(matrix, experiment, 100) / "model.safetensors"
            ):
                raise PipelineError(
                    f"{experiment}: localization checkpoint path mismatch"
                )
            if data.get("benchmark_v2_allow_map_subset_summary") is not True:
                raise PipelineError(
                    f"{experiment}: localization subset summary gate is missing"
                )
        base = _integer(
            spec.get("master_port_base"), f"{experiment}.master_port_base", 1
        )
        if base in ports:
            raise PipelineError(f"duplicate master port base: {base}")
        ports.add(base)


def print_status(
    matrix: Mapping[str, Any],
    cuda_device: str,
    *,
    shots: Sequence[int] | None = None,
    families: Sequence[str] | None = None,
    routes: Sequence[str] | None = None,
) -> None:
    try:
        free = _gpu_free_memory_mb(cuda_device)
    except PipelineError as exc:
        free = f"unavailable ({exc})"
    print(f"GPU {cuda_device} free_memory_mb={free}")
    selected_shots = list(shots) if shots is not None else scheduler_shots(matrix)
    for shot in selected_shots:
        _shot_index(matrix, shot)
    selected_families = set(_selected_families(families))
    selected_routes = set(_selected_routes(routes))
    for shot in selected_shots:
        for spec in _spec_list(matrix):
            experiment = str(spec["name"])
            if (
                _family(spec) not in selected_families
                or _route(spec) not in selected_routes
            ):
                continue
            try:
                complete = pipeline_complete(matrix, experiment, shot)
                parent = _parent_ready(matrix, experiment)
                held = _lock_is_held(pipeline_lock_path(matrix, experiment, shot))
                print(
                    f"{experiment:24s} shot={shot:3d} "
                    f"{'complete' if complete else 'pending':8s} "
                    f"parent={parent} lock_held={held}"
                )
            except PipelineError as exc:
                print(f"{experiment:24s} shot={shot:3d} invalid {exc}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--benchmark_v2_asset_manifest",
        default=argparse.SUPPRESS,
        help="Override the matrix Benchmark v2 minimal asset report.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)

    status = subparsers.add_parser("status")
    status.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)
    status.add_argument("--cuda-device")
    status.add_argument("--shots", nargs="+", type=int)
    status.add_argument("--families", nargs="+")
    status.add_argument("--routes", nargs="+")

    run = subparsers.add_parser("run")
    run.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)
    run.add_argument("--experiment", required=True)
    run.add_argument("--shots", type=int, required=True)
    run.add_argument("--cuda-device")
    run.add_argument("--families", nargs="+")
    run.add_argument("--routes", nargs="+")
    run.add_argument("--dry-run", action="store_true")

    schedule_parser = subparsers.add_parser("schedule")
    schedule_parser.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)
    schedule_parser.add_argument("--shots", nargs="+", type=int)
    schedule_parser.add_argument("--families", nargs="+")
    schedule_parser.add_argument("--routes", nargs="+")
    schedule_parser.add_argument("--cuda-device")
    schedule_parser.add_argument("--minimum-free-memory-mb", type=int)
    schedule_parser.add_argument("--joint-minimum-free-memory-mb", type=int)
    schedule_parser.add_argument("--gen-minimum-free-memory-mb", type=int)
    schedule_parser.add_argument("--loc-minimum-free-memory-mb", type=int)
    for family in FAMILIES:
        for route in ROUTES:
            if _family_route_supported(family, route):
                schedule_parser.add_argument(
                    f"--{family}-{route}-minimum-free-memory-mb",
                    dest=f"{family}_{route}_minimum_free_memory_mb",
                    type=int,
                )
    schedule_parser.add_argument("--launch-settle-seconds", type=int)
    schedule_parser.add_argument("--launch-memory-confirmation-seconds", type=int)
    schedule_parser.add_argument("--poll-seconds", type=int)
    schedule_parser.add_argument("--continue-on-failure", action="store_true")
    schedule_parser.add_argument("--stop-on-failure", action="store_true")
    schedule_parser.add_argument("--dry-run", action="store_true")

    sync = subparsers.add_parser("sync-results")
    sync.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)
    sync.add_argument("--shots", type=int)
    sync.add_argument("--families", nargs="+")
    sync.add_argument("--routes", nargs="+")
    sync.add_argument("--initialize", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        matrix, config_path = load_matrix(args.config)
        if hasattr(args, "benchmark_v2_asset_manifest"):
            matrix = dict(matrix)
            matrix["benchmark_v2_asset_manifest"] = args.benchmark_v2_asset_manifest
        validate_configuration(matrix)
        scheduling = _scheduling(matrix)
        cuda_device = str(
            getattr(args, "cuda_device", None) or scheduling.get("cuda_device", "0")
        )
        if args.command == "validate":
            print(
                f"VALID {config_path}; experiments={len(_spec_list(matrix))}; scheduler_shots={scheduler_shots(matrix)}"
            )
        elif args.command == "status":
            print_status(
                matrix,
                cuda_device,
                shots=args.shots,
                families=args.families,
                routes=args.routes,
            )
        elif args.command == "run":
            run_pipeline(
                matrix,
                args.experiment,
                args.shots,
                cuda_device=cuda_device,
                dry_run=args.dry_run,
                families=args.families,
                routes=args.routes,
            )
        elif args.command == "sync-results":
            sync_results(
                matrix,
                shots=args.shots,
                initialize=args.initialize,
                families=args.families,
                routes=args.routes,
            )
            print(f"UPDATED {_repo_path(str(_paths(matrix)['results_file']))}")
        elif args.command == "schedule":
            if args.continue_on_failure and args.stop_on_failure:
                raise PipelineError("choose only one failure behavior override")
            route_minimums: dict[str, int] = {}
            if args.minimum_free_memory_mb is not None:
                for route in ROUTES:
                    route_minimums[route] = args.minimum_free_memory_mb
            for route, value in (
                ("joint", args.joint_minimum_free_memory_mb),
                ("gen", args.gen_minimum_free_memory_mb),
                ("loc", args.loc_minimum_free_memory_mb),
            ):
                if value is not None:
                    route_minimums[route] = value
            family_route_minimums: dict[tuple[str, str], int] = {}
            for family in FAMILIES:
                for route in _family_routes(family):
                    value = getattr(args, f"{family}_{route}_minimum_free_memory_mb")
                    if value is not None:
                        family_route_minimums[(family, route)] = value
            failure_override = None
            if args.continue_on_failure:
                failure_override = False
            elif args.stop_on_failure:
                failure_override = True
            schedule(
                matrix,
                config_path,
                cuda_device=cuda_device,
                shots=args.shots,
                families=args.families,
                routes=args.routes,
                route_minimums=route_minimums or None,
                family_route_minimums=family_route_minimums or None,
                launch_settle_seconds=args.launch_settle_seconds,
                launch_memory_confirmation_seconds=args.launch_memory_confirmation_seconds,
                poll_seconds=args.poll_seconds,
                stop_launching_on_failure=failure_override,
                dry_run=args.dry_run,
            )
        else:
            raise PipelineError(f"unsupported command: {args.command}")
        return 0
    except (PipelineError, OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "Interrupted; an already-launched child pipeline was not terminated.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
