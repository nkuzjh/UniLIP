#!/usr/bin/env python3
"""Run isolated Seen-10 evaluation pipelines for exp31 and exp32 checkpoints.

Each task owns one lock and one output directory.  Localization is a single
``eval_csgo_loc.py`` stage because that command already performs inference,
metric computation, and the equal-map summary.  A generation task runs
inference, ten per-map metric commands, and the map aggregation in that order.

The runner deliberately emits repository-relative paths for checkpoints and
outputs.  This matters when ``outputs*`` are symlinks whose resolved target is
different on another host.
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
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, TextIO

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "csgo_configs/benchmark_v2_checkpoint_eval.yaml"
EXPERIMENTS = ("exp31", "exp32")
TASKS = ("localization", "discrete", "continuous")
LOCALIZATION_TASK = "localization"
GENERATION_TASKS = ("discrete", "continuous")
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
SPLITS = {
    "localization": "seen_discrete_test",
    "discrete": "seen_discrete_test",
    "continuous": "seen_continuous",
}
EXPECTED_SAMPLES = {"seen_discrete_test": 20000, "seen_continuous": 12800}
EXPECTED_CHECKPOINTS = {
    experiment: f"outputs/csgo_1b/{experiment}/checkpoint-6000/model.safetensors"
    for experiment in EXPERIMENTS
}
EXPECTED_CHECKPOINT_SHA256 = {
    "exp31": "b56dced04cab9ac35d952a592b6c7b16ff59210c55814605812e5b1996f898c6",
    "exp32": "16642d606c0ae531b99289a44bdfc8b64606c4dbb6a850f935d504ae7bded0d8",
}
EXPECTED_GLOBAL_STEP = 6000
DEFAULT_LAUNCH_ORDER = (
    ("exp31", "localization"),
    ("exp32", "localization"),
    ("exp31", "discrete"),
    ("exp31", "continuous"),
    ("exp32", "discrete"),
    ("exp32", "continuous"),
)


class PipelineError(RuntimeError):
    """Raised when a runner contract or pipeline stage fails."""


def _repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _relative(path: Path) -> str:
    """Return a stable repo-relative path without resolving symlink targets."""

    absolute = path if path.is_absolute() else REPO_ROOT / path
    try:
        return absolute.absolute().relative_to(REPO_ROOT.absolute()).as_posix()
    except ValueError:
        return str(absolute.absolute())


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineError(f"{name} must be a mapping")
    return dict(value)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PipelineError(f"{name} must be an integer >= {minimum}")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineError(f"{name} must be a non-empty string")
    return value


def load_config(
    path: str | os.PathLike[str] = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], Path]:
    config_path = _repo_path(path).resolve()
    if not config_path.is_file():
        raise PipelineError(f"config does not exist: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PipelineError(f"could not read config {config_path}: {exc}") from exc
    return _mapping(config, "config"), config_path


# Alias retained for consistency with the other Benchmark v2 runners.
load_matrix = load_config


def _protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(config.get("protocol"), "protocol")


def _paths(config: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(config.get("paths"), "paths")


def _evaluation(config: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(config.get("evaluation"), "evaluation")


def _asset_report(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]] | None:
    value = config.get("benchmark_v2_asset_manifest")
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


def _asset_expected_provenance(config: Mapping[str, Any]) -> dict[str, str] | None:
    report = _asset_report(config)
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


def _validate_task_asset_config(config: Mapping[str, Any], config_path: Path) -> None:
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PipelineError(f"cannot read task config {config_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PipelineError(f"task config must be a mapping: {config_path}")
    task_value = payload.get("benchmark_v2_asset_manifest")
    matrix_value = config.get("benchmark_v2_asset_manifest")
    if task_value in (None, ""):
        return
    if not isinstance(task_value, (str, os.PathLike)):
        raise PipelineError(
            f"{config_path}: benchmark_v2_asset_manifest must be a path or null"
        )
    if matrix_value not in (None, "") and not isinstance(matrix_value, (str, os.PathLike)):
        raise PipelineError(
            "config benchmark_v2_asset_manifest must be a path or null"
        )
    if matrix_value in (None, "") or _repo_path(os.fspath(task_value)).resolve() != _repo_path(
        os.fspath(matrix_value)
    ).resolve():
        raise PipelineError(
            f"{config_path}: benchmark_v2_asset_manifest conflicts with matrix switch"
        )


def _append_asset_cli(command: list[str], config: Mapping[str, Any]) -> None:
    value = config.get("benchmark_v2_asset_manifest")
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


def _metric_gt_dir(config: Mapping[str, Any], map_name: str) -> str:
    report = _asset_report(config)
    if report is None:
        evaluation = _evaluation(config)
        return f"{evaluation['data_dir']}/{map_name}/imgs"
    path, payload = report
    images = payload["images"]
    assert isinstance(images, Mapping)
    root = images["root"]
    assert isinstance(root, str)
    return _relative((path.parent / root / map_name).resolve())


def _validate_asset_provenance(
    config: Mapping[str, Any], payload: Mapping[str, Any], source_path: Path
) -> None:
    expected = _asset_expected_provenance(config)
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
    if not isinstance(actual, str) or _normalised_path(actual, "asset_manifest") != Path(
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


def _scheduling(config: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(config.get("scheduling"), "scheduling")


def _experiments(config: Mapping[str, Any]) -> dict[str, Any]:
    experiments = _mapping(config.get("experiments"), "experiments")
    if tuple(experiments) != EXPERIMENTS:
        raise PipelineError(
            f"experiments must be ordered as {EXPERIMENTS!r}; got {tuple(experiments)!r}"
        )
    return experiments


def _experiment(config: Mapping[str, Any], experiment: str) -> dict[str, Any]:
    experiments = _experiments(config)
    if experiment not in experiments:
        raise PipelineError(
            f"unknown experiment {experiment!r}; expected one of {EXPERIMENTS!r}"
        )
    return _mapping(experiments[experiment], f"experiments.{experiment}")


def _task(task: str) -> None:
    if task not in TASKS:
        raise PipelineError(f"unknown task {task!r}; expected one of {TASKS!r}")


def _checkpoint_path(config: Mapping[str, Any], experiment: str) -> Path:
    return _repo_path(str(_experiment(config, experiment)["checkpoint"]))


def checkpoint_path(config: Mapping[str, Any], experiment: str) -> Path:
    """Public checkpoint path helper used by tests and shell tooling."""

    return _checkpoint_path(config, experiment)


def _task_config(config: Mapping[str, Any], experiment: str, task: str) -> str:
    _task(task)
    key = f"{task}_config"
    return _nonempty_string(_experiment(config, experiment).get(key), key)


def localization_dir(config: Mapping[str, Any], experiment: str) -> Path:
    _task("localization")
    return (
        _repo_path(str(_paths(config)["localization_root"]))
        / experiment
        / "checkpoint_6000"
        / "seen"
    )


def generation_dir(config: Mapping[str, Any], experiment: str, task: str) -> Path:
    if task not in GENERATION_TASKS:
        raise PipelineError(f"generation task expected, got {task!r}")
    return (
        _repo_path(str(_paths(config)["generation_root"]))
        / experiment
        / "checkpoint_6000"
        / "seen"
        / task
    )


def task_output_dir(config: Mapping[str, Any], experiment: str, task: str) -> Path:
    return (
        localization_dir(config, experiment)
        if task == LOCALIZATION_TASK
        else generation_dir(config, experiment, task)
    )


def task_log_dir(config: Mapping[str, Any], experiment: str, task: str) -> Path:
    _task(task)
    return _repo_path(str(_paths(config)["log_root"])) / experiment / task


def _split(config: Mapping[str, Any], task: str) -> str:
    _task(task)
    splits = _mapping(_protocol(config).get("splits"), "protocol.splits")
    value = _nonempty_string(splits.get(task), f"protocol.splits.{task}")
    if value != SPLITS[task]:
        raise PipelineError(
            f"protocol.splits.{task} must be {SPLITS[task]!r}, got {value!r}"
        )
    return value


def _seen_maps(config: Mapping[str, Any]) -> list[str]:
    value = _protocol(config).get("seen_maps")
    if not isinstance(value, list) or any(
        not isinstance(map_name, str) or not map_name for map_name in value
    ):
        raise PipelineError("protocol.seen_maps must be a non-empty list of names")
    if len(set(value)) != len(value):
        raise PipelineError("protocol.seen_maps contains duplicate names")
    return list(value)


def _expected_sample_count(config: Mapping[str, Any], task: str) -> int:
    split = _split(config, task)
    counts = _mapping(
        _protocol(config).get("expected_samples"), "protocol.expected_samples"
    )
    count = _integer(counts.get(split), f"protocol.expected_samples.{split}", 1)
    if count != EXPECTED_SAMPLES[split]:
        raise PipelineError(
            f"protocol.expected_samples.{split} must be {EXPECTED_SAMPLES[split]}, got {count}"
        )
    return count


def _per_map_sample_count(config: Mapping[str, Any], task: str) -> int:
    total = _expected_sample_count(config, task)
    maps = _seen_maps(config)
    if total % len(maps) != 0:
        raise PipelineError(
            f"{task} sample count {total} is not divisible by map count {len(maps)}"
        )
    return total // len(maps)


def _append_cli(command: list[str], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            command.append(f"--{key}")
            command.extend(str(item) for item in value)
        elif isinstance(value, bool):
            command.extend((f"--{key}", "True" if value else "False"))
        else:
            command.extend((f"--{key}", str(value)))


def _command_option(command: Sequence[str], option: str) -> str:
    try:
        index = command.index(option)
    except ValueError as exc:
        raise PipelineError(f"command is missing {option}") from exc
    if index + 1 >= len(command):
        raise PipelineError(f"command has no value for {option}")
    return command[index + 1]


def build_localization_command(config: Mapping[str, Any], experiment: str) -> list[str]:
    checkpoint = _relative(_checkpoint_path(config, experiment))
    command = [
        sys.executable,
        "eval_csgo_loc.py",
        "--csgo_config",
        _task_config(config, experiment, LOCALIZATION_TASK),
        "--output_dir",
        _relative(localization_dir(config, experiment)),
        # This explicit CLI option is intentional: eval_csgo_loc.py applies it
        # after loading YAML, so checkpoint-6000 wins over the YAML default.
        "--ckpt_path",
        checkpoint,
        "--seed",
        str(_protocol(config)["inference_seed"]),
        "--benchmark_v2_split",
        _split(config, LOCALIZATION_TASK),
        "--benchmark_v2_maps",
        *_seen_maps(config),
    ]
    _append_asset_cli(command, config)
    return command


def build_generation_inference_command(
    config: Mapping[str, Any], experiment: str, task: str
) -> list[str]:
    if task not in GENERATION_TASKS:
        raise PipelineError(f"generation task expected, got {task!r}")
    command = [
        sys.executable,
        "eval_csgo.py",
        "--csgo_config",
        _task_config(config, experiment, task),
        "--output_dir",
        _relative(generation_dir(config, experiment, task)),
        # Required even though the referenced YAML has a ckpt_path of its own.
        "--ckpt_path",
        _relative(_checkpoint_path(config, experiment)),
        "--seed",
        str(_protocol(config)["inference_seed"]),
        "--benchmark_v2_split",
        _split(config, task),
        "--benchmark_v2_maps",
        *_seen_maps(config),
    ]
    _append_asset_cli(command, config)
    return command


def build_metric_command(
    config: Mapping[str, Any], experiment: str, task: str, map_name: str
) -> list[str]:
    if task not in GENERATION_TASKS:
        raise PipelineError(f"generation task expected, got {task!r}")
    maps = _seen_maps(config)
    if map_name not in maps:
        raise PipelineError(f"unknown Seen-10 map {map_name!r}")
    evaluation = _evaluation(config)
    script = (
        "benchmark_csgo_v1.py" if task == "discrete" else "benchmark_csgo_v1_conti.py"
    )
    output = generation_dir(config, experiment, task)
    command = [
        sys.executable,
        script,
        "--gt",
        _metric_gt_dir(config, map_name),
        "--pred",
        f"{_relative(output)}/gen_imgs/{map_name}",
        "--batch_size",
        str(evaluation["batch_size"]),
        "--device",
        "cuda",
        "--paired_size",
        str(evaluation["paired_size"]),
    ]
    if _asset_report(config) is None:
        command.extend(("--data_dir", str(evaluation["data_dir"])))
    command.extend(("--map_name", map_name))
    if task == "continuous":
        command.extend(
            (
                "--frame_diff_threshold",
                str(evaluation["frame_diff_threshold"]),
                "--min_track_len",
                str(evaluation["min_track_len"]),
                "--clip_length",
                str(evaluation["clip_length"]),
                "--clip_stride",
                str(evaluation["clip_stride"]),
                "--fvd_size",
                str(evaluation["fvd_size"]),
            )
        )
    command.extend(
        (
            "--benchmark_v2_manifest",
            str(_protocol(config)["manifest"]),
            "--benchmark_v2_split",
            _split(config, task),
            "--external_loc_repo_root",
            str(evaluation["external_loc_repo_root"]),
            "--external_loc_config_path",
            str(evaluation["external_loc_config_path"]),
            "--external_loc_checkpoint_path",
            str(evaluation["external_loc_checkpoint_path"]),
        )
    )
    _append_asset_cli(command, config)
    return command


def build_aggregate_command(
    config: Mapping[str, Any], experiment: str, task: str
) -> list[str]:
    if task not in GENERATION_TASKS:
        raise PipelineError(f"generation task expected, got {task!r}")
    output = generation_dir(config, experiment, task)
    return [
        sys.executable,
        "scripts/aggregate_csgo_benchmark_v2_metrics.py",
        "maps",
        "--manifest",
        str(_protocol(config)["manifest"]),
        "--split",
        _split(config, task),
        "--input_root",
        _relative(output),
        "--kind",
        task,
        "--output",
        f"{_relative(output)}/summary.json",
    ]


def build_task_commands(
    config: Mapping[str, Any], experiment: str, task: str
) -> list[tuple[str, list[str]]]:
    _task(task)
    if task == LOCALIZATION_TASK:
        return [("localization", build_localization_command(config, experiment))]
    return [
        ("inference", build_generation_inference_command(config, experiment, task)),
        *[
            (
                f"metric_{map_name}",
                build_metric_command(config, experiment, task, map_name),
            )
            for map_name in _seen_maps(config)
        ],
        ("aggregate", build_aggregate_command(config, experiment, task)),
    ]


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PipelineError(f"invalid {description} {path}: {exc}") from exc
    return _mapping(payload, description)


def _read_json_list(path: Path, description: str) -> list[Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PipelineError(f"invalid {description} {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise PipelineError(f"{description} must be a list")
    return payload


def _finite_metrics(
    value: Any, name: str, *, expected_metadata: Mapping[str, Any] | None = None
) -> None:
    metrics = _mapping(value, name)
    if not metrics:
        raise PipelineError(f"{name} must not be empty")
    metadata = dict(expected_metadata or {})
    for key, expected in metadata.items():
        if metrics.get(key) != expected:
            raise PipelineError(
                f"{name}.{key} must be {expected!r}, got {metrics.get(key)!r}"
            )
    for metric, metric_value in metrics.items():
        if metric in metadata:
            continue
        if (
            isinstance(metric_value, bool)
            or not isinstance(metric_value, (int, float))
            or not math.isfinite(float(metric_value))
        ):
            raise PipelineError(f"{name}.{metric} must be a finite number")


def _normalised_path(value: Any, name: str) -> Path:
    return _repo_path(_nonempty_string(value, name)).resolve()


def _expected_inference_payload(
    config: Mapping[str, Any], experiment: str, task: str
) -> dict[str, Any]:
    return {
        "config_path": _task_config(config, experiment, task),
        "benchmark_v2_manifest": str(_protocol(config)["manifest"]),
        "benchmark_v2_split": _split(config, task),
        "benchmark_v2_support_seed": None,
        "benchmark_v2_shots_per_map": None,
        "maps": _seen_maps(config),
        "sample_count": _expected_sample_count(config, task),
        "checkpoint": _relative(_checkpoint_path(config, experiment)),
        "ckpt_path": _relative(_checkpoint_path(config, experiment)),
        "seed": _integer(_protocol(config)["inference_seed"], "inference_seed", 0),
    }


def _validate_checkpoint_artifact(config: Mapping[str, Any], experiment: str) -> None:
    """Validate the immutable checkpoint identity used by every task."""

    spec = _experiment(config, experiment)
    checkpoint_value = _nonempty_string(
        spec.get("checkpoint"), f"{experiment}.checkpoint"
    )
    if checkpoint_value != EXPECTED_CHECKPOINTS[experiment]:
        raise PipelineError(
            f"{experiment}.checkpoint must be the repo-relative checkpoint-6000 path"
        )
    checkpoint = _checkpoint_path(config, experiment)
    if not checkpoint.is_file():
        raise PipelineError(f"missing safetensors checkpoint: {checkpoint}")
    minimum_size = _integer(
        _protocol(config).get("minimum_checkpoint_size_bytes"),
        "protocol.minimum_checkpoint_size_bytes",
        1,
    )
    size = checkpoint.stat().st_size
    if size < minimum_size:
        raise PipelineError(
            f"checkpoint is too small: {checkpoint} has {size} bytes, "
            f"expected at least {minimum_size}"
        )
    expected_sha256 = _nonempty_string(
        spec.get("expected_sha256"), f"{experiment}.expected_sha256"
    ).lower()
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise PipelineError(
            f"{experiment}.expected_sha256 must be a 64-character hex digest"
        )
    if expected_sha256 != EXPECTED_CHECKPOINT_SHA256[experiment]:
        raise PipelineError(
            f"{experiment}.expected_sha256 does not match the frozen checkpoint contract"
        )
    digest = hashlib.sha256()
    try:
        with checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise PipelineError(f"could not hash checkpoint {checkpoint}: {exc}") from exc
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise PipelineError(
            f"checkpoint SHA-256 mismatch for {checkpoint}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    expected_global_step = _integer(
        spec.get("expected_global_step"), f"{experiment}.expected_global_step", 0
    )
    if expected_global_step != EXPECTED_GLOBAL_STEP:
        raise PipelineError(
            f"{experiment}.expected_global_step must be {EXPECTED_GLOBAL_STEP}"
        )
    trainer_state = checkpoint.parent / "trainer_state.json"
    if not trainer_state.is_file():
        raise PipelineError(f"missing trainer state beside checkpoint: {trainer_state}")
    state = _read_json(trainer_state, "trainer state")
    if state.get("global_step") != expected_global_step:
        raise PipelineError(
            f"{trainer_state}: expected global_step={expected_global_step}, "
            f"got {state.get('global_step')!r}"
        )


def _validate_inference_manifest(
    config: Mapping[str, Any], experiment: str, task: str
) -> dict[str, Any]:
    output = task_output_dir(config, experiment, task)
    path = output / "inference_manifest.json"
    if not path.is_file():
        raise PipelineError(f"missing inference manifest: {path}")
    payload = _read_json(path, "inference manifest")
    expected = _expected_inference_payload(config, experiment, task)
    for key, value in expected.items():
        actual = payload.get(key)
        if key == "benchmark_v2_manifest" and isinstance(actual, str):
            matches = _normalised_path(actual, key) == _normalised_path(value, key)
        else:
            matches = actual == value
        if not matches:
            raise PipelineError(f"{path}: expected {key}={value!r}, got {actual!r}")
    _validate_asset_provenance(config, payload, path)

    per_map = _per_map_sample_count(config, task)
    if task == LOCALIZATION_TASK:
        results_path = output / "loc_results.json"
        results = _read_json_list(results_path, "localization results")
        if len(results) != _expected_sample_count(config, task):
            raise PipelineError(
                f"{results_path}: found {len(results)} results, "
                f"expected {_expected_sample_count(config, task)}"
            )
        counts = {map_name: 0 for map_name in _seen_maps(config)}
        for index, item in enumerate(results):
            row = _mapping(item, f"{results_path}[{index}]")
            map_name = row.get("map")
            if map_name not in counts:
                raise PipelineError(
                    f"{results_path}[{index}]: unexpected map {map_name!r}"
                )
            counts[map_name] += 1
        if any(count != per_map for count in counts.values()):
            raise PipelineError(
                f"{results_path}: per-map counts {counts!r}, expected {per_map} each"
            )
    else:
        for map_name in _seen_maps(config):
            map_dir = output / "gen_imgs" / map_name
            count = len(list(map_dir.glob("*.jpg"))) if map_dir.is_dir() else 0
            if count != per_map:
                raise PipelineError(
                    f"{map_dir}: found {count} jpg images, expected {per_map}"
                )
    return payload


def _validate_localization_summary(
    config: Mapping[str, Any], experiment: str
) -> dict[str, Any]:
    output = localization_dir(config, experiment)
    path = output / "benchmark_csgo_v2_loc.json"
    if not path.is_file():
        raise PipelineError(f"missing localization summary: {path}")
    payload = _read_json(path, "localization summary")
    manifest = _normalised_path(payload.get("manifest"), "manifest")
    expected_manifest = _normalised_path(_protocol(config)["manifest"], "manifest")
    if manifest != expected_manifest:
        raise PipelineError(f"{path}: manifest provenance mismatch")
    expected = {
        "split": _split(config, LOCALIZATION_TASK),
        "kind": "localization",
        "maps": _seen_maps(config),
        "checkpoint": _relative(_checkpoint_path(config, experiment)),
        "seed": _protocol(config)["inference_seed"],
        "support_seed": None,
        "shots_per_map": None,
        "sample_count": _expected_sample_count(config, LOCALIZATION_TASK),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PipelineError(
                f"{path}: expected {key}={value!r}, got {payload.get(key)!r}"
            )
    per_map = _mapping(payload.get("per_map"), "localization per_map")
    if list(per_map) != _seen_maps(config):
        raise PipelineError(
            f"{path}: localization per_map order does not match protocol"
        )
    for map_name in _seen_maps(config):
        _finite_metrics(
            per_map[map_name],
            f"{path}: per_map.{map_name}",
            expected_metadata={
                "ckpt_path": _relative(_checkpoint_path(config, experiment))
            },
        )
    _finite_metrics(payload.get("metrics_macro_map"), f"{path}: metrics_macro_map")
    provenance = _mapping(payload.get("inference_provenance"), "inference_provenance")
    provenance_payload = _mapping(
        provenance.get("payload"), "inference_provenance.payload"
    )
    expected_inference = _expected_inference_payload(
        config, experiment, LOCALIZATION_TASK
    )
    for key, value in expected_inference.items():
        actual = provenance_payload.get(key)
        if key == "benchmark_v2_manifest" and isinstance(actual, str):
            matches = _normalised_path(actual, key) == _normalised_path(value, key)
        else:
            matches = actual == value
        if not matches:
            raise PipelineError(f"{path}: inference provenance mismatch for {key}")
    _validate_asset_provenance(config, provenance_payload, path)
    provenance_path = provenance.get("path")
    if (
        not isinstance(provenance_path, str)
        or Path(provenance_path).resolve()
        != (output / "inference_manifest.json").resolve()
    ):
        raise PipelineError(
            f"{path}: inference provenance path does not point to manifest"
        )
    _validate_inference_manifest(config, experiment, LOCALIZATION_TASK)
    return payload


def _per_map_metric_path(output: Path, task: str, map_name: str) -> Path:
    prefix = "benchmark_csgo_v2" if task == "discrete" else "benchmark_csgo_v2_conti"
    return output / f"{prefix}_{map_name}.json"


def _expected_metric_provenance(
    config: Mapping[str, Any], experiment: str, task: str
) -> dict[str, Any]:
    return _expected_inference_payload(config, experiment, task)


def _validate_per_map_metric(
    config: Mapping[str, Any], experiment: str, task: str, map_name: str
) -> dict[str, Any]:
    output = generation_dir(config, experiment, task)
    path = _per_map_metric_path(output, task, map_name)
    if not path.is_file():
        raise PipelineError(f"missing per-map metric: {path}")
    payload = _read_json(path, "per-map metric")
    if payload.get("map_name") != map_name:
        raise PipelineError(f"{path}: map_name does not match {map_name!r}")
    if payload.get("benchmark_v2_split") != _split(config, task):
        raise PipelineError(f"{path}: split does not match protocol")
    manifest = _normalised_path(payload.get("benchmark_v2_manifest"), "manifest")
    if manifest != _normalised_path(_protocol(config)["manifest"], "manifest"):
        raise PipelineError(f"{path}: manifest provenance mismatch")
    expected_count = _per_map_sample_count(config, task)
    common_count = payload.get("common_count")
    if common_count != expected_count:
        raise PipelineError(
            f"{path}: common_count={common_count!r}, expected {expected_count}"
        )
    provenance = _mapping(payload.get("inference_provenance"), "inference_provenance")
    provenance_payload = _mapping(
        provenance.get("payload"), "inference_provenance.payload"
    )
    expected = _expected_metric_provenance(config, experiment, task)
    for key, value in expected.items():
        actual = provenance_payload.get(key)
        if key == "benchmark_v2_manifest" and isinstance(actual, str):
            matches = _normalised_path(actual, key) == _normalised_path(value, key)
        else:
            matches = actual == value
        if not matches:
            raise PipelineError(f"{path}: inference provenance mismatch for {key}")
    _validate_asset_provenance(config, provenance_payload, path)
    provenance_path = provenance.get("path")
    if (
        not isinstance(provenance_path, str)
        or Path(provenance_path).resolve()
        != (output / "inference_manifest.json").resolve()
    ):
        raise PipelineError(
            f"{path}: inference provenance path does not point to manifest"
        )
    metrics = payload.get("metrics_ordered")
    if metrics is not None:
        for metric, value in _mapping(metrics, f"{path}: metrics_ordered").items():
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise PipelineError(f"{path}: metric {metric} is not finite")
    return payload


def _validate_generation_summary(
    config: Mapping[str, Any], experiment: str, task: str
) -> dict[str, Any]:
    output = generation_dir(config, experiment, task)
    path = output / "summary.json"
    if not path.is_file():
        raise PipelineError(f"missing generation summary: {path}")
    payload = _read_json(path, "generation summary")
    expected = {
        "split": _split(config, task),
        "kind": task,
        "maps": _seen_maps(config),
        "checkpoint": _relative(_checkpoint_path(config, experiment)),
        "ckpt_path": _relative(_checkpoint_path(config, experiment)),
        "inference_seed": _protocol(config)["inference_seed"],
        "support_seed": None,
        "shots_per_map": None,
        "sample_count": _expected_sample_count(config, task),
    }
    manifest = _normalised_path(payload.get("manifest"), "manifest")
    if manifest != _normalised_path(_protocol(config)["manifest"], "manifest"):
        raise PipelineError(f"{path}: manifest provenance mismatch")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PipelineError(
                f"{path}: expected {key}={value!r}, got {payload.get(key)!r}"
            )
    per_map = _mapping(payload.get("per_map"), "generation per_map")
    if list(per_map) != _seen_maps(config):
        raise PipelineError(f"{path}: generation per_map order does not match protocol")
    for map_name in _seen_maps(config):
        _finite_metrics(per_map[map_name], f"{path}: per_map.{map_name}")
    _finite_metrics(payload.get("metrics_macro_map"), f"{path}: metrics_macro_map")
    provenance = _mapping(payload.get("inference_provenance"), "inference_provenance")
    provenance_payload = _mapping(
        provenance.get("payload"), "inference_provenance.payload"
    )
    expected_inference = _expected_inference_payload(config, experiment, task)
    for key, value in expected_inference.items():
        actual = provenance_payload.get(key)
        if key == "benchmark_v2_manifest" and isinstance(actual, str):
            matches = _normalised_path(actual, key) == _normalised_path(value, key)
        else:
            matches = actual == value
        if not matches:
            raise PipelineError(f"{path}: inference provenance mismatch for {key}")
    _validate_asset_provenance(config, provenance_payload, path)
    provenance_path = provenance.get("path")
    if (
        not isinstance(provenance_path, str)
        or Path(provenance_path).resolve()
        != (output / "inference_manifest.json").resolve()
    ):
        raise PipelineError(
            f"{path}: inference provenance path does not point to manifest"
        )
    _validate_inference_manifest(config, experiment, task)
    for map_name in _seen_maps(config):
        _validate_per_map_metric(config, experiment, task, map_name)
    return payload


def _localization_complete(config: Mapping[str, Any], experiment: str) -> bool:
    summary = localization_dir(config, experiment) / "benchmark_csgo_v2_loc.json"
    if not summary.exists():
        return False
    _validate_localization_summary(config, experiment)
    return True


def _generation_complete(config: Mapping[str, Any], experiment: str, task: str) -> bool:
    summary = generation_dir(config, experiment, task) / "summary.json"
    if not summary.exists():
        return False
    _validate_generation_summary(config, experiment, task)
    return True


def task_complete(config: Mapping[str, Any], experiment: str, task: str) -> bool:
    _task(task)
    return (
        _localization_complete(config, experiment)
        if task == LOCALIZATION_TASK
        else _generation_complete(config, experiment, task)
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    return _read_json(path, "Benchmark v2 manifest")


def _validate_eval_config(
    config: Mapping[str, Any], experiment: str, task: str, manifest_path: Path
) -> list[Path]:
    config_value = _task_config(config, experiment, task)
    path = _repo_path(config_value)
    if not path.is_file():
        raise PipelineError(f"missing {task} eval config: {path}")
    data = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))
    _validate_task_asset_config(config, path)
    configured_manifest = data.get("benchmark_v2_manifest")
    if (
        not isinstance(configured_manifest, str)
        or _repo_path(configured_manifest).resolve() != manifest_path
    ):
        raise PipelineError(f"{path}: benchmark_v2_manifest does not match matrix")
    if data.get("benchmark_v2_split") != _split(config, task):
        raise PipelineError(f"{path}: benchmark_v2_split does not match {task}")
    expected_maps = _seen_maps(config)
    for key in ("train_maps", "val_maps", "test_maps"):
        if data.get(key) != expected_maps:
            raise PipelineError(f"{path}: {key} does not match Seen-10 protocol")
    if task == "continuous" and data.get("is_conti_gen") is not True:
        raise PipelineError(f"{path}: continuous config must set is_conti_gen: true")
    if task != "continuous" and data.get("is_conti_gen") is True:
        raise PipelineError(
            f"{path}: non-continuous config cannot set is_conti_gen: true"
        )
    # Do not bind this check to the YAML checkpoint: the generated command's
    # explicit --ckpt_path is the tested override to checkpoint-6000.
    _nonempty_string(data.get("ckpt_path"), f"{path}: ckpt_path")
    return [path]


def validate_configuration(
    config: Mapping[str, Any], *, check_runtime_files: bool = True
) -> None:
    if _integer(config.get("schema_version"), "schema_version", 1) != 1:
        raise PipelineError("schema_version must be 1")
    protocol = _protocol(config)
    _asset_report(config)
    if _nonempty_string(protocol.get("manifest"), "protocol.manifest") != str(
        protocol["manifest"]
    ):
        raise PipelineError("invalid protocol.manifest")
    if (
        _integer(protocol.get("checkpoint_step"), "checkpoint_step", 1)
        != EXPECTED_GLOBAL_STEP
    ):
        raise PipelineError(f"checkpoint_step must be {EXPECTED_GLOBAL_STEP}")
    if _integer(protocol.get("inference_seed"), "inference_seed", 0) != 42:
        raise PipelineError("inference_seed must be 42")
    if _seen_maps(config) != list(SEEN_MAPS):
        raise PipelineError("Seen-10 map order does not match Benchmark v2")
    if _mapping(protocol.get("splits"), "protocol.splits") != SPLITS:
        raise PipelineError("protocol.splits does not match the task contract")
    if (
        _mapping(protocol.get("expected_samples"), "protocol.expected_samples")
        != EXPECTED_SAMPLES
    ):
        raise PipelineError(
            "protocol.expected_samples does not match the task contract"
        )
    manifest_path = _repo_path(str(protocol["manifest"])).resolve()
    if check_runtime_files and not manifest_path.is_file():
        raise PipelineError(f"missing Benchmark v2 manifest: {manifest_path}")

    paths = _paths(config)
    for key in ("localization_root", "generation_root", "log_root"):
        _nonempty_string(paths.get(key), f"paths.{key}")
    evaluation = _evaluation(config)
    for key in (
        "external_loc_repo_root",
        "external_loc_config_path",
        "external_loc_checkpoint_path",
        "aesthetic_checkpoint",
    ):
        _nonempty_string(evaluation.get(key), f"evaluation.{key}")
    if _asset_report(config) is None:
        _nonempty_string(evaluation.get("data_dir"), "evaluation.data_dir")
    for key in (
        "batch_size",
        "paired_size",
        "frame_diff_threshold",
        "min_track_len",
        "clip_length",
        "clip_stride",
        "fvd_size",
    ):
        _integer(evaluation.get(key), f"evaluation.{key}", 1)

    experiments = _experiments(config)
    required_files: list[Path] = [manifest_path]
    for experiment in EXPERIMENTS:
        spec = experiments[experiment]
        checkpoint_value = _nonempty_string(
            spec.get("checkpoint"), f"{experiment}.checkpoint"
        )
        if checkpoint_value != EXPECTED_CHECKPOINTS[experiment]:
            raise PipelineError(
                f"{experiment}.checkpoint must be {EXPECTED_CHECKPOINTS[experiment]!r}"
            )
        checkpoint = _checkpoint_path(config, experiment)
        required_files.append(checkpoint)
        for task in TASKS:
            required_files.extend(
                _validate_eval_config(config, experiment, task, manifest_path)
            )

        for task in TASKS:
            for option, expected in {
                "--ckpt_path": _relative(checkpoint),
                "--seed": "42",
                "--benchmark_v2_split": _split(config, task),
            }.items():
                command = (
                    build_localization_command(config, experiment)
                    if task == LOCALIZATION_TASK
                    else build_generation_inference_command(config, experiment, task)
                )
                if _command_option(command, option) != expected:
                    raise PipelineError(
                        f"{experiment}/{task}: command {option} does not override protocol as required"
                    )
            command = (
                build_localization_command(config, experiment)
                if task == LOCALIZATION_TASK
                else build_generation_inference_command(config, experiment, task)
            )
            map_index = command.index("--benchmark_v2_maps")
            maps = _seen_maps(config)
            if command[map_index + 1 : map_index + 1 + len(maps)] != maps:
                raise PipelineError(
                    f"{experiment}/{task}: map command does not match Seen-10"
                )
        expected_sha256 = _nonempty_string(
            spec.get("expected_sha256"), f"{experiment}.expected_sha256"
        )
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256.lower()
        ):
            raise PipelineError(
                f"{experiment}.expected_sha256 must be a 64-character hex digest"
            )
        if (
            _integer(
                spec.get("expected_global_step"),
                f"{experiment}.expected_global_step",
                0,
            )
            != EXPECTED_GLOBAL_STEP
        ):
            raise PipelineError(
                f"{experiment}.expected_global_step must be {EXPECTED_GLOBAL_STEP}"
            )

    scheduling = _scheduling(config)
    cuda_device = _nonempty_string(
        scheduling.get("cuda_device"), "scheduling.cuda_device"
    )
    del cuda_device
    max_parallel = _integer(
        scheduling.get("max_parallel_pipelines"), "max_parallel_pipelines", 1
    )
    if max_parallel > 6:
        raise PipelineError("max_parallel_pipelines must be <= 6")
    _integer(scheduling.get("minimum_free_memory_mb"), "minimum_free_memory_mb", 1)
    launch_settle = _integer(
        scheduling.get("launch_settle_seconds"), "launch_settle_seconds", 120
    )
    if launch_settle < 120:
        raise PipelineError("launch_settle_seconds must be at least 120 seconds")
    _integer(scheduling.get("poll_seconds"), "poll_seconds", 1)
    if not isinstance(scheduling.get("stop_launching_on_failure"), bool):
        raise PipelineError("stop_launching_on_failure must be boolean")
    raw_order = scheduling.get("launch_order")
    if not isinstance(raw_order, list):
        raise PipelineError("scheduling.launch_order must be a list")
    try:
        launch_order = tuple((str(item[0]), str(item[1])) for item in raw_order)
    except (IndexError, TypeError) as exc:
        raise PipelineError(
            "scheduling.launch_order entries must be [experiment, task]"
        ) from exc
    if launch_order != DEFAULT_LAUNCH_ORDER:
        raise PipelineError(
            f"launch_order must start localization-first and contain {DEFAULT_LAUNCH_ORDER!r}"
        )

    if check_runtime_files:
        external_root = _repo_path(str(evaluation["external_loc_repo_root"]))
        required_files.extend(
            (
                external_root / str(evaluation["external_loc_config_path"]),
                external_root / str(evaluation["external_loc_checkpoint_path"]),
                _repo_path(str(evaluation["aesthetic_checkpoint"])),
            )
        )
        if _asset_report(config) is None:
            required_files.append(_repo_path(str(evaluation["data_dir"])))
        for path in required_files:
            if not path.is_file() and not path.is_dir():
                raise PipelineError(f"required runtime path does not exist: {path}")
        for experiment in EXPERIMENTS:
            _validate_checkpoint_artifact(config, experiment)
        manifest = _load_manifest(manifest_path)
        manifest_protocol = _mapping(manifest.get("protocol"), "manifest.protocol")
        if manifest_protocol.get("seen_maps") != list(SEEN_MAPS):
            raise PipelineError("manifest Seen-10 maps do not match runner protocol")
        counts = _mapping(manifest.get("counts"), "manifest.counts")
        seen_counts = _mapping(counts.get("seen"), "manifest.counts.seen")
        for map_name in SEEN_MAPS:
            map_counts = _mapping(
                seen_counts.get(map_name), f"manifest.counts.seen.{map_name}"
            )
            if map_counts.get("discrete_test") != 2000:
                raise PipelineError(
                    f"manifest {map_name}: discrete_test count is not 2000"
                )
            if map_counts.get("continuous_frames") != 1280:
                raise PipelineError(
                    f"manifest {map_name}: continuous_frames count is not 1280"
                )


def _stage_path(
    config: Mapping[str, Any], experiment: str, task: str, stage: str
) -> Path:
    return task_log_dir(config, experiment, task) / f"{stage}.log"


def _run_command(command: Sequence[str], log_path: Path, cuda_device: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = cuda_device
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{time.strftime('%F %T')}] COMMAND {shlex.join(command)}\n")
        result = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"[{time.strftime('%F %T')}] EXIT {result.returncode}\n")
    if result.returncode != 0:
        raise PipelineError(
            f"command failed with exit {result.returncode}; see {log_path}"
        )


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineError(f"another process holds lock: {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _task_lock_path(config: Mapping[str, Any], experiment: str, task: str) -> Path:
    return task_log_dir(config, experiment, task) / ".task.lock"


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


def _run_localization(
    config: Mapping[str, Any], experiment: str, *, cuda_device: str, dry_run: bool
) -> None:
    command = build_localization_command(config, experiment)
    if dry_run:
        print(f"[{experiment}/localization] {shlex.join(command)}")
        return
    output = localization_dir(config, experiment)
    summary = output / "benchmark_csgo_v2_loc.json"
    if summary.exists() and _localization_complete(config, experiment):
        print(f"SKIP complete {experiment}/localization", flush=True)
        return
    _run_command(
        command,
        _stage_path(config, experiment, "localization", "localization"),
        cuda_device,
    )
    _validate_localization_summary(config, experiment)
    print(f"DONE {experiment}/localization", flush=True)


def _run_generation(
    config: Mapping[str, Any],
    experiment: str,
    task: str,
    *,
    cuda_device: str,
    dry_run: bool,
) -> None:
    commands = build_task_commands(config, experiment, task)
    if dry_run:
        for stage, command in commands:
            print(f"[{experiment}/{task}:{stage}] {shlex.join(command)}")
        return
    if _generation_complete(config, experiment, task):
        print(f"SKIP complete {experiment}/{task}", flush=True)
        return
    output = generation_dir(config, experiment, task)
    inference_manifest = output / "inference_manifest.json"
    if not inference_manifest.is_file():
        print(f"START {experiment}/{task} inference", flush=True)
        _run_command(
            commands[0][1],
            _stage_path(config, experiment, task, "inference"),
            cuda_device,
        )
    _validate_inference_manifest(config, experiment, task)

    for stage, command in commands[1:-1]:
        map_name = stage.removeprefix("metric_")
        metric_path = _per_map_metric_path(output, task, map_name)
        if metric_path.is_file():
            _validate_per_map_metric(config, experiment, task, map_name)
            print(f"SKIP complete {experiment}/{task}/{map_name}", flush=True)
            continue
        print(f"START {experiment}/{task} metric {map_name}", flush=True)
        _run_command(command, _stage_path(config, experiment, task, stage), cuda_device)
        _validate_per_map_metric(config, experiment, task, map_name)

    summary = output / "summary.json"
    if summary.is_file():
        _validate_generation_summary(config, experiment, task)
        print(f"SKIP complete {experiment}/{task}/aggregate", flush=True)
    else:
        print(f"START {experiment}/{task} aggregate", flush=True)
        _run_command(
            commands[-1][1],
            _stage_path(config, experiment, task, "aggregate"),
            cuda_device,
        )
        _validate_generation_summary(config, experiment, task)
    print(f"DONE {experiment}/{task}", flush=True)


def run_task(
    config: Mapping[str, Any],
    experiment: str,
    task: str,
    *,
    cuda_device: str,
    dry_run: bool = False,
) -> None:
    _experiment(config, experiment)
    _task(task)
    if dry_run:
        if task == LOCALIZATION_TASK:
            _run_localization(config, experiment, cuda_device=cuda_device, dry_run=True)
        else:
            _run_generation(
                config, experiment, task, cuda_device=cuda_device, dry_run=True
            )
        return
    with _exclusive_lock(_task_lock_path(config, experiment, task)):
        if task == LOCALIZATION_TASK:
            _run_localization(
                config, experiment, cuda_device=cuda_device, dry_run=False
            )
        else:
            _run_generation(
                config, experiment, task, cuda_device=cuda_device, dry_run=False
            )


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
        raise PipelineError(f"cannot parse free GPU memory: {result.stdout!r}") from exc


def _default_jobs(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    scheduling = _scheduling(config)
    raw_order = scheduling.get("launch_order")
    if not isinstance(raw_order, list):
        return list(DEFAULT_LAUNCH_ORDER)
    return [(str(item[0]), str(item[1])) for item in raw_order]


def _adopt_held_locks(
    config: Mapping[str, Any],
    pending: list[tuple[str, str]],
    external: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    adopted: list[tuple[str, str]] = []
    for job in list(pending):
        if not _lock_is_held(_task_lock_path(config, *job)):
            continue
        pending.remove(job)
        external.add(job)
        adopted.append(job)
    return adopted


def _job_complete(config: Mapping[str, Any], job: tuple[str, str]) -> bool:
    return task_complete(config, job[0], job[1])


def schedule(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    cuda_device: str,
    max_parallel: int,
    minimum_free_memory_mb: int,
    launch_settle_seconds: int,
    poll_seconds: int,
) -> None:
    if max_parallel < 1 or max_parallel > 6:
        raise PipelineError("max_parallel must be in [1, 6]")
    if minimum_free_memory_mb < 1:
        raise PipelineError("minimum_free_memory_mb must be positive")
    if launch_settle_seconds < 120:
        raise PipelineError("launch_settle_seconds must be at least 120 seconds")
    if poll_seconds < 1:
        raise PipelineError("poll_seconds must be positive")

    root = _repo_path(str(_paths(config)["log_root"]))
    root.mkdir(parents=True, exist_ok=True)
    jobs = _default_jobs(config)
    priority_jobs = {("exp31", LOCALIZATION_TASK), ("exp32", LOCALIZATION_TASK)}
    active: dict[tuple[str, str], tuple[subprocess.Popen[Any], TextIO]] = {}
    external: set[tuple[str, str]] = set()
    failures: list[tuple[str, str, int | str]] = []
    last_launch = 0.0

    with _exclusive_lock(root / ".scheduler.lock"):
        pending = [job for job in jobs if not _job_complete(config, job)]
        adopted = _adopt_held_locks(config, pending, external)
        if adopted:
            print(f"ADOPT already-running tasks: {adopted}", flush=True)
        print(f"PENDING tasks: {pending}", flush=True)

        while pending or active or external:
            for job, (process, handle) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                handle.close()
                del active[job]
                if returncode == 0 and _job_complete(config, job):
                    print(f"COMPLETE {job[0]}/{job[1]}", flush=True)
                else:
                    failures.append((job[0], job[1], returncode))
                    print(f"FAILED {job[0]}/{job[1]} exit={returncode}", flush=True)

            for job in list(external):
                lock_path = _task_lock_path(config, *job)
                if _lock_is_held(lock_path):
                    continue
                external.remove(job)
                if _job_complete(config, job):
                    print(f"COMPLETE adopted {job[0]}/{job[1]}", flush=True)
                else:
                    failures.append((job[0], job[1], "lost-lock"))
                    print(
                        f"FAILED adopted {job[0]}/{job[1]}: lock released before completion",
                        flush=True,
                    )

            halt_launch = bool(
                failures and _scheduling(config).get("stop_launching_on_failure", True)
            )
            started = {job for job in priority_jobs if job not in pending}
            priority_gate = priority_jobs.issubset(started)
            now = time.monotonic()
            can_launch = (
                bool(pending)
                and not halt_launch
                and len(active) + len(external) < max_parallel
                and now - last_launch >= launch_settle_seconds
                and (priority_gate or pending[0] in priority_jobs)
            )
            if can_launch:
                job = pending[0]
                if not priority_gate and job not in priority_jobs:
                    raise PipelineError(
                        "scheduler launch order violated localization priority"
                    )
                free_mb = _gpu_free_memory_mb(cuda_device)
                if free_mb >= minimum_free_memory_mb:
                    pending.pop(0)
                    experiment, task = job
                    pipeline_log = (
                        task_log_dir(config, experiment, task) / "pipeline.log"
                    )
                    pipeline_log.parent.mkdir(parents=True, exist_ok=True)
                    handle = pipeline_log.open("a", encoding="utf-8", buffering=1)
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--config",
                        str(config_path),
                        "run",
                        "--experiment",
                        experiment,
                        "--task",
                        task,
                        "--cuda-device",
                        cuda_device,
                    ]
                    _append_asset_cli(command, config)
                    environment = dict(os.environ)
                    environment["CUDA_VISIBLE_DEVICES"] = cuda_device
                    environment["PYTHONUNBUFFERED"] = "1"
                    process = subprocess.Popen(
                        command,
                        cwd=REPO_ROOT,
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    active[job] = (process, handle)
                    last_launch = now
                    print(
                        f"LAUNCH {experiment}/{task} pid={process.pid} "
                        f"free_mb_before={free_mb} active={len(active) + len(external)}/{max_parallel}",
                        flush=True,
                    )
                else:
                    print(
                        f"WAIT GPU free memory {free_mb} MB < {minimum_free_memory_mb} MB; "
                        f"active={list(active)} adopted={list(external)}",
                        flush=True,
                    )

            if pending or active or external:
                time.sleep(poll_seconds)

    if failures:
        rendered = ", ".join(
            f"{experiment}/{task}:exit={code}" for experiment, task, code in failures
        )
        raise PipelineError(f"schedule failed: {rendered}")
    print("DONE all checkpoint evaluation tasks", flush=True)


def _task_status(
    config: Mapping[str, Any], experiment: str, task: str
) -> tuple[str, str]:
    try:
        if task == LOCALIZATION_TASK:
            complete = _localization_complete(config, experiment)
            detail = "summary+inference valid" if complete else "pending"
        else:
            complete = _generation_complete(config, experiment, task)
            detail = "inference+10 metrics+aggregate valid" if complete else "pending"
        return ("complete" if complete else "pending", detail)
    except PipelineError as exc:
        return "invalid", str(exc)


def print_status(config: Mapping[str, Any], *, cuda_device: str) -> None:
    try:
        free_memory: int | str = _gpu_free_memory_mb(cuda_device)
    except PipelineError as exc:
        free_memory = f"unavailable ({exc})"
    print(f"GPU {cuda_device} free_memory_mb={free_memory}")
    for experiment in EXPERIMENTS:
        for task in TASKS:
            state, detail = _task_status(config, experiment, task)
            locked = _lock_is_held(_task_lock_path(config, experiment, task))
            print(
                f"{experiment:5s} {task:12s} {state:8s} "
                f"lock={'held' if locked else 'free'} {detail}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--benchmark_v2_asset_manifest",
        default=argparse.SUPPRESS,
        help="Override the matrix Benchmark v2 minimal asset report.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate config and runtime prerequisites"
    )
    validate.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)
    validate.add_argument(
        "--no-runtime-files",
        action="store_true",
        help="only validate structure and command contracts",
    )

    status = subparsers.add_parser(
        "status", help="show six-task artifact and GPU status"
    )
    status.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)
    status.add_argument("--cuda-device")

    run = subparsers.add_parser("run", help="run one serial evaluation task")
    run.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)
    run.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    run.add_argument("--task", choices=TASKS, required=True)
    run.add_argument("--cuda-device")
    run.add_argument("--dry-run", action="store_true")

    schedule_parser = subparsers.add_parser(
        "schedule", help="launch independent tasks under a GPU memory gate"
    )
    schedule_parser.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)
    schedule_parser.add_argument("--cuda-device")
    schedule_parser.add_argument("--max-parallel", type=int)
    schedule_parser.add_argument("--minimum-free-memory-mb", type=int)
    schedule_parser.add_argument("--launch-settle-seconds", type=int)
    schedule_parser.add_argument("--poll-seconds", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config, config_path = load_config(args.config)
        if hasattr(args, "benchmark_v2_asset_manifest"):
            config = dict(config)
            config["benchmark_v2_asset_manifest"] = args.benchmark_v2_asset_manifest
        check_runtime_files = args.command != "validate" or not args.no_runtime_files
        validate_configuration(config, check_runtime_files=check_runtime_files)
        scheduling = _scheduling(config)
        cuda_device = str(
            getattr(args, "cuda_device", None) or scheduling.get("cuda_device", "0")
        )
        if args.command == "validate":
            print(
                "configuration, checkpoint CLI overrides, Seen-10 maps, and "
                "provenance contracts are valid"
            )
            return 0
        if args.command == "status":
            print_status(config, cuda_device=cuda_device)
            return 0
        if args.command == "run":
            run_task(
                config,
                args.experiment,
                args.task,
                cuda_device=cuda_device,
                dry_run=args.dry_run,
            )
            return 0

        schedule(
            config,
            config_path,
            cuda_device=cuda_device,
            max_parallel=(
                args.max_parallel
                if args.max_parallel is not None
                else scheduling["max_parallel_pipelines"]
            ),
            minimum_free_memory_mb=(
                args.minimum_free_memory_mb
                if args.minimum_free_memory_mb is not None
                else scheduling["minimum_free_memory_mb"]
            ),
            launch_settle_seconds=(
                args.launch_settle_seconds
                if args.launch_settle_seconds is not None
                else scheduling["launch_settle_seconds"]
            ),
            poll_seconds=(
                args.poll_seconds
                if args.poll_seconds is not None
                else scheduling["poll_seconds"]
            ),
        )
        return 0
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "Interrupted; already launched child tasks are not terminated.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
