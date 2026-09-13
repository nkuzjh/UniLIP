#!/usr/bin/env python3
"""Run and schedule Benchmark v2 localization few-shot adaptation pipelines."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
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
DEFAULT_CONFIG = REPO_ROOT / "csgo_configs/benchmark_v2_loc_few_shot.yaml"
EXPECTED_EXPERIMENTS = ("exp33_loc", "exp34_loc")


class PipelineError(RuntimeError):
    """Raised when a pipeline contract or stage fails."""


def _repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineError(f"{name} must be a mapping")
    return dict(value)


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PipelineError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise PipelineError(f"{name} must be >= {minimum}")
    return value


def load_matrix(
    path: str | os.PathLike[str] = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], Path]:
    config_path = _repo_path(path).resolve()
    if not config_path.is_file():
        raise PipelineError(f"matrix config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle)
    return _require_mapping(matrix, "matrix config"), config_path


def _protocol(matrix: Mapping[str, Any]) -> dict[str, Any]:
    return _require_mapping(matrix.get("protocol"), "protocol")


def _paths(matrix: Mapping[str, Any]) -> dict[str, Any]:
    return _require_mapping(matrix.get("paths"), "paths")


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
    return _require_mapping(matrix.get("scheduling"), "scheduling")


def _experiments(matrix: Mapping[str, Any]) -> dict[str, Any]:
    experiments = _require_mapping(matrix.get("experiments"), "experiments")
    if tuple(experiments) != EXPECTED_EXPERIMENTS:
        raise PipelineError(
            f"experiments must be ordered as {EXPECTED_EXPERIMENTS!r}; got {tuple(experiments)!r}"
        )
    return experiments


def _shots(matrix: Mapping[str, Any]) -> list[int]:
    raw = _protocol(matrix).get("shots_per_map")
    if not isinstance(raw, list) or not raw:
        raise PipelineError("protocol.shots_per_map must be a non-empty list")
    shots = [_require_int(value, "shots_per_map item", minimum=1) for value in raw]
    if len(set(shots)) != len(shots) or any(value > 100 for value in shots):
        raise PipelineError(
            "protocol.shots_per_map must contain unique values in [1, 100]"
        )
    return shots


def _experiment(matrix: Mapping[str, Any], name: str) -> dict[str, Any]:
    experiments = _experiments(matrix)
    if name not in experiments:
        raise PipelineError(
            f"unknown experiment {name!r}; expected one of {tuple(experiments)!r}"
        )
    return _require_mapping(experiments[name], f"experiments.{name}")


def model_dir(matrix: Mapping[str, Any], experiment: str, shots: int) -> Path:
    root = _repo_path(str(_paths(matrix)["model_root"]))
    seed = _require_int(_protocol(matrix)["support_seed"], "protocol.support_seed")
    return root / experiment / f"shot_{shots}" / f"seed_{seed}"


def checkpoint_path(matrix: Mapping[str, Any], experiment: str, shots: int) -> Path:
    return model_dir(matrix, experiment, shots) / "model.safetensors"


def localization_dir(
    matrix: Mapping[str, Any], experiment: str, shots: int, split: str
) -> Path:
    root = _repo_path(str(_paths(matrix)["localization_root"]))
    seed = _require_int(_protocol(matrix)["support_seed"], "protocol.support_seed")
    output = root / experiment / f"shot_{shots}" / f"seed_{seed}"
    if split == "seen_discrete_test":
        output /= "seen_retention"
    elif split != "crossmap_query_test":
        raise PipelineError(f"unsupported localization split: {split}")
    return output


def summary_path(
    matrix: Mapping[str, Any], experiment: str, shots: int, split: str
) -> Path:
    return (
        localization_dir(matrix, experiment, shots, split)
        / "benchmark_csgo_v2_loc.json"
    )


def log_dir(matrix: Mapping[str, Any], experiment: str, shots: int) -> Path:
    root = _repo_path(str(_paths(matrix)["log_root"]))
    seed = _require_int(_protocol(matrix)["support_seed"], "protocol.support_seed")
    return root / experiment / f"shot_{shots}" / f"seed_{seed}"


def _relative(path: Path) -> str:
    # Keep repository paths logical: outputs* may be symlinks to host-specific storage.
    absolute = path.expanduser()
    if not absolute.is_absolute():
        absolute = REPO_ROOT / absolute
    absolute = absolute.absolute()
    try:
        return absolute.relative_to(REPO_ROOT.absolute()).as_posix()
    except ValueError:
        return str(absolute)


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


def build_train_command(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    *,
    python_executable: str | None = None,
) -> list[str]:
    spec = _experiment(matrix, experiment)
    protocol = _protocol(matrix)
    if shots not in _shots(matrix):
        raise PipelineError(f"shots {shots} is not declared by protocol.shots_per_map")
    ports = _require_mapping(
        spec.get("master_ports"), f"experiments.{experiment}.master_ports"
    )
    port = ports.get(shots, ports.get(str(shots)))
    port = _require_int(port, f"master port for {experiment}/{shots}", minimum=1)
    training = _require_mapping(matrix.get("training"), "training")
    common_args = _require_mapping(training.get("common_args"), "training.common_args")
    batch_sizes = _require_mapping(
        training.get("train_batch_size_by_shot"),
        "training.train_batch_size_by_shot",
    )
    batch_size = _require_int(
        batch_sizes.get(shots, batch_sizes.get(str(shots))),
        f"training batch size for {shots}-shot",
        minimum=1,
    )
    train_args = _require_mapping(
        spec.get("train_args"), f"experiments.{experiment}.train_args"
    )

    command = [
        python_executable or sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        f"--master_port={port}",
        "train_csgo.py",
        "--csgo_config",
        str(spec["train_config"]),
    ]
    _append_cli_mapping(command, common_args)
    command.extend(
        (
            "--per_device_train_batch_size",
            str(batch_size),
            "--output_dir",
            _relative(model_dir(matrix, experiment, shots)),
            "--max_steps",
            str(_require_int(protocol["max_steps"], "protocol.max_steps", minimum=1)),
        )
    )
    _append_cli_mapping(command, train_args)
    command.extend(
        (
            "--benchmark_v2_support_seed",
            str(_require_int(protocol["support_seed"], "protocol.support_seed")),
            "--benchmark_v2_shots_per_map",
            str(shots),
        )
    )
    _append_asset_cli(command, matrix)
    return command


def build_eval_command(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    split: str,
    *,
    python_executable: str | None = None,
) -> list[str]:
    spec = _experiment(matrix, experiment)
    protocol = _protocol(matrix)
    map_key = "crossmap_maps" if split == "crossmap_query_test" else "seen_maps"
    maps = protocol.get(map_key)
    if not isinstance(maps, list) or not maps:
        raise PipelineError(f"protocol.{map_key} must be a non-empty list")
    command = [
        python_executable or sys.executable,
        "eval_csgo_loc.py",
        "--csgo_config",
        str(spec["eval_config"]),
        "--output_dir",
        _relative(localization_dir(matrix, experiment, shots, split)),
        "--ckpt_path",
        _relative(checkpoint_path(matrix, experiment, shots)),
        "--seed",
        str(_require_int(protocol["inference_seed"], "protocol.inference_seed")),
        "--benchmark_v2_split",
        split,
        "--benchmark_v2_support_seed",
        str(_require_int(protocol["support_seed"], "protocol.support_seed")),
        "--benchmark_v2_shots_per_map",
        str(shots),
        "--benchmark_v2_maps",
        *(str(map_name) for map_name in maps),
    ]
    _append_asset_cli(command, matrix)
    return command


def validate_configuration(matrix: Mapping[str, Any]) -> None:
    if _require_int(matrix.get("schema_version"), "schema_version") != 1:
        raise PipelineError("schema_version must be 1")
    protocol = _protocol(matrix)
    _asset_report(matrix)
    manifest_value = protocol.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise PipelineError("protocol.manifest must be a non-empty path")
    manifest_path = _repo_path(manifest_value)
    if not manifest_path.is_file():
        raise PipelineError(f"protocol manifest does not exist: {manifest_path}")
    if _require_int(protocol.get("support_seed"), "protocol.support_seed") != 0:
        raise PipelineError("the formal few-shot protocol requires support seed 0")
    if (
        _require_int(
            protocol.get("prerequisite_shots_per_map"),
            "protocol.prerequisite_shots_per_map",
        )
        != 100
    ):
        raise PipelineError("the prerequisite shot count must be 100")
    shots = _shots(matrix)
    if shots != [50, 20, 10]:
        raise PipelineError("the formal few-shot order must be [50, 20, 10]")

    expected_samples = _require_mapping(
        protocol.get("expected_samples"), "protocol.expected_samples"
    )
    if expected_samples != {"crossmap_query_test": 8000, "seen_discrete_test": 20000}:
        raise PipelineError("protocol.expected_samples does not match Benchmark v2")

    training = _require_mapping(matrix.get("training"), "training")
    batch_sizes = _require_mapping(
        training.get("train_batch_size_by_shot"),
        "training.train_batch_size_by_shot",
    )
    expected_batch_sizes = {50: 128, 20: 80, 10: 40}
    normalized_batch_sizes = {
        shot: _require_int(
            batch_sizes.get(shot, batch_sizes.get(str(shot))),
            f"training batch size for {shot}-shot",
            minimum=1,
        )
        for shot in shots
    }
    if normalized_batch_sizes != expected_batch_sizes:
        raise PipelineError(
            "training.train_batch_size_by_shot must be "
            f"{expected_batch_sizes!r}; got {normalized_batch_sizes!r}"
        )
    common_args = _require_mapping(training.get("common_args"), "training.common_args")
    if "per_device_train_batch_size" in common_args:
        raise PipelineError(
            "training.common_args must not override the per-shot train batch size"
        )

    scheduling = _scheduling(matrix)
    _require_int(
        scheduling.get("max_parallel_pipelines"), "max_parallel_pipelines", minimum=1
    )
    _require_int(
        scheduling.get("minimum_free_memory_mb"), "minimum_free_memory_mb", minimum=1
    )
    _require_int(
        scheduling.get("launch_settle_seconds"), "launch_settle_seconds", minimum=0
    )
    _require_int(scheduling.get("poll_seconds"), "poll_seconds", minimum=1)
    _require_int(
        scheduling.get("prerequisite_timeout_seconds"),
        "prerequisite_timeout_seconds",
        minimum=1,
    )
    _require_int(
        scheduling.get("prerequisite_summary_settle_seconds"),
        "prerequisite_summary_settle_seconds",
        minimum=0,
    )

    for name, raw_spec in _experiments(matrix).items():
        spec = _require_mapping(raw_spec, f"experiments.{name}")
        train_path = _repo_path(str(spec.get("train_config")))
        eval_path = _repo_path(str(spec.get("eval_config")))
        parent_path = _repo_path(str(spec.get("parent_checkpoint")))
        for path, label in (
            (train_path, "train config"),
            (eval_path, "eval config"),
            (parent_path, "parent checkpoint"),
        ):
            if not path.is_file():
                raise PipelineError(f"{name} {label} does not exist: {path}")
        with train_path.open("r", encoding="utf-8") as handle:
            train_config = _require_mapping(
                yaml.safe_load(handle), f"{name} train config"
            )
        _validate_task_asset_config(matrix, train_path)
        with eval_path.open("r", encoding="utf-8") as handle:
            eval_config = _require_mapping(
                yaml.safe_load(handle), f"{name} eval config"
            )
        _validate_task_asset_config(matrix, eval_path)
        if train_config.get("benchmark_v2_split") != "crossmap_support":
            raise PipelineError(f"{name} train config must select crossmap_support")
        configured_manifest = _repo_path(
            str(train_config.get("benchmark_v2_manifest"))
        ).resolve()
        if configured_manifest != manifest_path.resolve():
            raise PipelineError(
                f"{name} train config must use protocol manifest {manifest_path}"
            )
        eval_manifest = _repo_path(
            str(eval_config.get("benchmark_v2_manifest"))
        ).resolve()
        if eval_manifest != manifest_path.resolve():
            raise PipelineError(
                f"{name} eval config must use protocol manifest {manifest_path}"
            )
        if train_config.get("benchmark_v2_support_seed") != 0:
            raise PipelineError(f"{name} train config must default to support seed 0")
        if train_config.get("benchmark_v2_shots_per_map") != 100:
            raise PipelineError(
                f"{name} train config must retain the 100-shot parent default"
            )
        configured_parent = _repo_path(
            str(train_config.get("finetune_init_ckpt_path"))
        ).resolve()
        if configured_parent != parent_path.resolve():
            raise PipelineError(
                f"{name} must initialize from {parent_path}, got {configured_parent}"
            )
        ports = _require_mapping(spec.get("master_ports"), f"{name}.master_ports")
        for shot in shots:
            value = ports.get(shot, ports.get(str(shot)))
            _require_int(value, f"{name} master port for {shot}", minimum=1)
            build_train_command(matrix, name, shot)
            build_eval_command(matrix, name, shot, "crossmap_query_test")
            build_eval_command(matrix, name, shot, "seen_discrete_test")

    _validate_real_support_subsets(matrix)


def _validate_real_support_subsets(matrix: Mapping[str, Any]) -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from csgo_datasets.benchmark_v2 import load_benchmark_v2_selection

    protocol = _protocol(matrix)
    first_spec = _experiment(matrix, EXPECTED_EXPERIMENTS[0])
    with _repo_path(str(first_spec["train_config"])).open(
        "r", encoding="utf-8"
    ) as handle:
        runtime = _require_mapping(yaml.safe_load(handle), "training config")
    runtime["benchmark_v2_manifest"] = str(
        _repo_path(str(runtime["benchmark_v2_manifest"])).resolve()
    )
    if _asset_report(matrix) is not None:
        runtime["benchmark_v2_asset_manifest"] = str(_asset_report(matrix)[0])
    else:
        runtime["data_dir"] = str(_repo_path(str(runtime["data_dir"])).resolve())
    runtime["benchmark_v2_support_seed"] = protocol["support_seed"]
    runtime["benchmark_v2_shots_per_map"] = 100
    full = load_benchmark_v2_selection(runtime)
    full_by_map = {
        map_name: [row["file_frame"] for row in full.rows if row["map"] == map_name]
        for map_name in full.map_names
    }
    for shots in _shots(matrix):
        runtime["benchmark_v2_shots_per_map"] = shots
        selection = load_benchmark_v2_selection(runtime)
        if len(selection.rows) != shots * len(full.map_names):
            raise PipelineError(
                f"{shots}-shot support count mismatch: got {len(selection.rows)}"
            )
        for map_name in full.map_names:
            selected = [
                row["file_frame"] for row in selection.rows if row["map"] == map_name
            ]
            if selected != full_by_map[map_name][:shots]:
                raise PipelineError(f"{map_name} {shots}-shot support is not nested")


def _validate_summary(
    matrix: Mapping[str, Any], experiment: str, shots: int, split: str
) -> None:
    path = summary_path(matrix, experiment, shots, split)
    if not path.is_file():
        raise PipelineError(f"missing localization summary: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"invalid localization summary {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PipelineError(f"localization summary is not an object: {path}")

    protocol = _protocol(matrix)
    expected_manifest = _repo_path(str(protocol["manifest"])).resolve()
    summary_manifest = payload.get("manifest")
    if (
        not isinstance(summary_manifest, str)
        or _repo_path(summary_manifest).resolve() != expected_manifest
    ):
        raise PipelineError(
            f"{path}: expected manifest {expected_manifest}, got {summary_manifest!r}"
        )
    maps = protocol["crossmap_maps" if split == "crossmap_query_test" else "seen_maps"]
    expected_count = protocol["expected_samples"][split]
    expected_checkpoint = _relative(checkpoint_path(matrix, experiment, shots))
    expected = {
        "split": split,
        "kind": "localization",
        "maps": maps,
        "checkpoint": expected_checkpoint,
        "seed": protocol["inference_seed"],
        "support_seed": protocol["support_seed"],
        "shots_per_map": shots,
        "sample_count": expected_count,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PipelineError(
                f"{path}: expected {key}={value!r}, got {payload.get(key)!r}"
            )
    per_map = payload.get("per_map")
    macro = payload.get("metrics_macro_map")
    if not isinstance(per_map, Mapping) or list(per_map) != list(maps):
        raise PipelineError(f"{path}: per_map keys do not match protocol maps")
    if not isinstance(macro, Mapping) or not macro:
        raise PipelineError(f"{path}: metrics_macro_map is missing or empty")
    provenance = payload.get("inference_provenance")
    if not isinstance(provenance, Mapping) or not isinstance(
        provenance.get("payload"), Mapping
    ):
        raise PipelineError(f"{path}: inference provenance is missing")
    provenance_payload = provenance["payload"]
    provenance_manifest = provenance_payload.get("benchmark_v2_manifest")
    if (
        not isinstance(provenance_manifest, str)
        or _repo_path(provenance_manifest).resolve() != expected_manifest
    ):
        raise PipelineError(f"{path}: provenance benchmark manifest mismatch")
    provenance_expected = {
        "benchmark_v2_split": split,
        "benchmark_v2_support_seed": protocol["support_seed"],
        "benchmark_v2_shots_per_map": shots,
        "maps": maps,
        "sample_count": expected_count,
        "checkpoint": expected_checkpoint,
        "ckpt_path": expected_checkpoint,
        "seed": protocol["inference_seed"],
    }
    for key, value in provenance_expected.items():
        if provenance_payload.get(key) != value:
            raise PipelineError(f"{path}: provenance mismatch for {key}")
    _validate_asset_provenance(matrix, provenance_payload, path)
    provenance_path = provenance.get("path")
    if not isinstance(provenance_path, str) or not Path(provenance_path).is_file():
        raise PipelineError(f"{path}: referenced inference manifest does not exist")


def _summary_complete(
    matrix: Mapping[str, Any], experiment: str, shots: int, split: str
) -> bool:
    path = summary_path(matrix, experiment, shots, split)
    if not path.exists():
        return False
    _validate_summary(matrix, experiment, shots, split)
    return True


def _training_complete(matrix: Mapping[str, Any], experiment: str, shots: int) -> bool:
    directory = model_dir(matrix, experiment, shots)
    model_path = directory / "model.safetensors"
    state_path = directory / "trainer_state.json"
    if not model_path.exists() and not state_path.exists():
        return False
    if not model_path.is_file() or model_path.stat().st_size < 1024 * 1024:
        raise PipelineError(f"incomplete final model artifact: {model_path}")
    if not state_path.is_file():
        raise PipelineError(f"missing trainer state beside final model: {state_path}")
    try:
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"invalid trainer state {state_path}: {exc}") from exc
    expected_steps = _protocol(matrix)["max_steps"]
    if (
        state.get("global_step") != expected_steps
        or state.get("max_steps") != expected_steps
    ):
        raise PipelineError(
            f"{state_path}: expected global_step=max_steps={expected_steps}, "
            f"got {state.get('global_step')}/{state.get('max_steps')}"
        )
    return True


def _pipeline_complete(matrix: Mapping[str, Any], experiment: str, shots: int) -> bool:
    if not _training_complete(matrix, experiment, shots):
        return False
    return all(
        _summary_complete(matrix, experiment, shots, split)
        for split in ("crossmap_query_test", "seen_discrete_test")
    )


def prerequisite_gaps(matrix: Mapping[str, Any]) -> list[str]:
    shots = _protocol(matrix)["prerequisite_shots_per_map"]
    settle_seconds = _require_int(
        _scheduling(matrix)["prerequisite_summary_settle_seconds"],
        "prerequisite_summary_settle_seconds",
        minimum=0,
    )
    gaps: list[str] = []
    for experiment in EXPECTED_EXPERIMENTS:
        if not _training_complete(matrix, experiment, shots):
            gaps.append(f"{experiment}/{shots}:training")
        for split in ("crossmap_query_test", "seen_discrete_test"):
            path = summary_path(matrix, experiment, shots, split)
            if path.exists() and time.time() - path.stat().st_mtime < settle_seconds:
                complete = False
            else:
                complete = _summary_complete(matrix, experiment, shots, split)
            if not complete:
                gaps.append(f"{experiment}/{shots}:{split}")
    return gaps


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineError(f"another process holds pipeline lock: {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run_command(command: Sequence[str], path: Path, *, cuda_device: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = cuda_device
    environment["PYTHONUNBUFFERED"] = "1"
    with path.open("a", encoding="utf-8", buffering=1) as log:
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
        raise PipelineError(f"command failed with exit {result.returncode}; see {path}")


def run_pipeline(
    matrix: Mapping[str, Any],
    experiment: str,
    shots: int,
    *,
    cuda_device: str,
    dry_run: bool = False,
) -> None:
    commands = [
        ("train", build_train_command(matrix, experiment, shots)),
        (
            "crossmap_query_test",
            build_eval_command(matrix, experiment, shots, "crossmap_query_test"),
        ),
        (
            "seen_discrete_test",
            build_eval_command(matrix, experiment, shots, "seen_discrete_test"),
        ),
    ]
    if dry_run:
        for stage, command in commands:
            print(f"[{experiment}/{shots}:{stage}] {shlex.join(command)}")
        return

    gaps = prerequisite_gaps(matrix)
    if gaps:
        raise PipelineError("100-shot prerequisite is incomplete: " + ", ".join(gaps))
    directory = log_dir(matrix, experiment, shots)
    lock_path = directory / ".pipeline.lock"
    with _exclusive_lock(lock_path):
        if _pipeline_complete(matrix, experiment, shots):
            print(f"SKIP complete pipeline {experiment}/{shots}", flush=True)
            return
        if not _training_complete(matrix, experiment, shots):
            print(f"START {experiment}/{shots} training", flush=True)
            _run_command(
                commands[0][1], directory / "train.log", cuda_device=cuda_device
            )
            if not _training_complete(matrix, experiment, shots):
                raise PipelineError(
                    f"training did not produce a complete model: {experiment}/{shots}"
                )
        else:
            print(f"SKIP complete training {experiment}/{shots}", flush=True)

        for stage, command in commands[1:]:
            if _summary_complete(matrix, experiment, shots, stage):
                print(
                    f"SKIP complete inference+metrics {experiment}/{shots}/{stage}",
                    flush=True,
                )
                continue
            print(f"START {experiment}/{shots} inference+metrics {stage}", flush=True)
            _run_command(command, directory / f"{stage}.log", cuda_device=cuda_device)
            _validate_summary(matrix, experiment, shots, stage)
        print(f"DONE {experiment}/{shots} pipeline", flush=True)


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


def schedule(
    matrix: Mapping[str, Any],
    config_path: Path,
    *,
    cuda_device: str,
    max_parallel: int,
    minimum_free_memory_mb: int,
    launch_settle_seconds: int,
    poll_seconds: int,
    prerequisite_timeout_seconds: int,
) -> None:
    root = _repo_path(str(_paths(matrix)["log_root"]))
    root.mkdir(parents=True, exist_ok=True)
    jobs = [(name, shots) for shots in _shots(matrix) for name in EXPECTED_EXPERIMENTS]
    active: dict[tuple[str, int], tuple[subprocess.Popen[Any], TextIO]] = {}
    failures: list[tuple[str, int, int]] = []
    last_launch = 0.0
    stop_on_failure = bool(_scheduling(matrix).get("stop_launching_on_failure", True))

    with _exclusive_lock(root / ".scheduler.lock"):
        prerequisite_started = time.monotonic()
        last_gap_report: tuple[str, ...] | None = None
        while True:
            gaps = tuple(prerequisite_gaps(matrix))
            if not gaps:
                print("100-shot prerequisite gate passed", flush=True)
                break
            if gaps != last_gap_report:
                print("WAIT 100-shot prerequisite: " + ", ".join(gaps), flush=True)
                last_gap_report = gaps
            if time.monotonic() - prerequisite_started >= prerequisite_timeout_seconds:
                raise PipelineError(
                    "100-shot prerequisite timed out after "
                    f"{prerequisite_timeout_seconds} seconds: {', '.join(gaps)}"
                )
            time.sleep(poll_seconds)

        pending = [job for job in jobs if not _pipeline_complete(matrix, *job)]
        print(f"PENDING pipelines: {pending}", flush=True)
        while pending or active:
            for job, (process, handle) in list(active.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                handle.close()
                del active[job]
                if returncode == 0 and _pipeline_complete(matrix, *job):
                    print(f"COMPLETE pipeline {job[0]}/{job[1]}", flush=True)
                else:
                    failures.append((job[0], job[1], returncode))
                    print(
                        f"FAILED pipeline {job[0]}/{job[1]} exit={returncode}",
                        flush=True,
                    )

            halt_launch = bool(failures and stop_on_failure)
            if halt_launch and not active:
                break

            now = time.monotonic()
            can_launch = (
                pending
                and not halt_launch
                and len(active) < max_parallel
                and now - last_launch >= launch_settle_seconds
            )
            if can_launch:
                free_mb = _gpu_free_memory_mb(cuda_device)
                if free_mb >= minimum_free_memory_mb:
                    experiment, shots = pending.pop(0)
                    pipeline_log = log_dir(matrix, experiment, shots) / "pipeline.log"
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
                        "--shots",
                        str(shots),
                        "--cuda-device",
                        cuda_device,
                    ]
                    _append_asset_cli(command, matrix)
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
                    active[(experiment, shots)] = (process, handle)
                    last_launch = now
                    print(
                        f"LAUNCH {experiment}/{shots} pid={process.pid} "
                        f"free_mb_before={free_mb} active={len(active)}/{max_parallel}",
                        flush=True,
                    )
                else:
                    print(
                        f"WAIT GPU free memory {free_mb} MB < {minimum_free_memory_mb} MB; "
                        f"active={list(active)}",
                        flush=True,
                    )
            if pending or active:
                time.sleep(poll_seconds)

    if failures:
        rendered = ", ".join(
            f"{name}/{shots}:exit={code}" for name, shots, code in failures
        )
        raise PipelineError(f"few-shot schedule failed: {rendered}")
    print("DONE all configured few-shot pipelines", flush=True)


def print_status(matrix: Mapping[str, Any], *, cuda_device: str) -> None:
    try:
        free_mb: int | str = _gpu_free_memory_mb(cuda_device)
    except PipelineError as exc:
        free_mb = f"unavailable ({exc})"
    print(f"GPU {cuda_device} free_memory_mb={free_mb}")
    prerequisite_shots = _protocol(matrix)["prerequisite_shots_per_map"]
    for experiment in EXPECTED_EXPERIMENTS:
        for shots in (prerequisite_shots, *_shots(matrix)):
            try:
                train = _training_complete(matrix, experiment, shots)
                cross = _summary_complete(
                    matrix, experiment, shots, "crossmap_query_test"
                )
                retention = _summary_complete(
                    matrix, experiment, shots, "seen_discrete_test"
                )
                state = "complete" if train and cross and retention else "pending"
                detail = f"train={train} crossmap={cross} retention={retention}"
            except PipelineError as exc:
                state = "invalid"
                detail = str(exc)
            print(f"{experiment:9s} shot={shots:3d} {state:8s} {detail}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--benchmark_v2_asset_manifest",
        default=argparse.SUPPRESS,
        help="Override the matrix Benchmark v2 minimal asset report.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="validate config and support subsets"
    )
    validate_parser.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)
    validate_parser.add_argument("--require-prerequisites", action="store_true")

    status_parser = subparsers.add_parser("status", help="show artifact and GPU status")
    status_parser.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)
    status_parser.add_argument("--cuda-device")

    run_parser = subparsers.add_parser("run", help="run one train/eval/metric pipeline")
    run_parser.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)
    run_parser.add_argument("--experiment", required=True)
    run_parser.add_argument("--shots", required=True, type=int)
    run_parser.add_argument("--cuda-device")
    run_parser.add_argument("--dry-run", action="store_true")

    schedule_parser = subparsers.add_parser(
        "schedule", help="resource-gated multi-pipeline scheduler"
    )
    schedule_parser.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)
    schedule_parser.add_argument("--cuda-device")
    schedule_parser.add_argument("--max-parallel", type=int)
    schedule_parser.add_argument("--minimum-free-memory-mb", type=int)
    schedule_parser.add_argument("--launch-settle-seconds", type=int)
    schedule_parser.add_argument("--poll-seconds", type=int)
    schedule_parser.add_argument("--prerequisite-timeout-seconds", type=int)
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
            gaps = prerequisite_gaps(matrix)
            if gaps and args.require_prerequisites:
                raise PipelineError(
                    "100-shot prerequisite is incomplete: " + ", ".join(gaps)
                )
            print("configuration and nested support subsets are valid")
            print("100-shot prerequisite: " + ("complete" if not gaps else "pending"))
            return 0
        if args.command == "status":
            print_status(matrix, cuda_device=cuda_device)
            return 0
        if args.command == "run":
            run_pipeline(
                matrix,
                args.experiment,
                args.shots,
                cuda_device=cuda_device,
                dry_run=args.dry_run,
            )
            return 0
        schedule(
            matrix,
            config_path,
            cuda_device=cuda_device,
            max_parallel=args.max_parallel or scheduling["max_parallel_pipelines"],
            minimum_free_memory_mb=(
                args.minimum_free_memory_mb or scheduling["minimum_free_memory_mb"]
            ),
            launch_settle_seconds=(
                args.launch_settle_seconds
                if args.launch_settle_seconds is not None
                else scheduling["launch_settle_seconds"]
            ),
            poll_seconds=args.poll_seconds or scheduling["poll_seconds"],
            prerequisite_timeout_seconds=(
                args.prerequisite_timeout_seconds
                or scheduling["prerequisite_timeout_seconds"]
            ),
        )
        return 0
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "Interrupted; already launched child pipelines are not terminated.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
