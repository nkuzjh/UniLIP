#!/usr/bin/env python3
"""Run Benchmark v2 generation zero-shot and few-shot pipelines."""

from __future__ import annotations

import argparse
import errno
import fcntl
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
from typing import Any, Iterator, Mapping, Sequence, TextIO

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "csgo_configs/benchmark_v2_gen.yaml"
ZERO_SHOT_EXPERIMENTS = ("exp31_gen", "exp32_gen")
FEW_SHOT_EXPERIMENTS = ("exp33_gen", "exp34_gen")
SEEN_EXPERIMENTS = ("exp31_gen", "exp32_gen")
KINDS = ("discrete", "continuous")
SETTINGS = ("crossmap", "seen")
CROSSMAP_MAPS = ("cs_office", "de_golden", "de_palacio", "de_vertigo")
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

SETTING_SPLITS = {
    "crossmap": {
        "discrete": "crossmap_query_test",
        "continuous": "crossmap_continuous",
    },
    "seen": {
        "discrete": "seen_discrete_test",
        "continuous": "seen_continuous",
    },
}


class PipelineError(RuntimeError):
    """Raised when a pipeline contract or stage fails."""


def _repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _relative(path: Path) -> str:
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


def load_matrix(path: str | os.PathLike[str] = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    config_path = _repo_path(path).resolve()
    if not config_path.is_file():
        raise PipelineError(f"matrix config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle)
    return _mapping(matrix, "matrix config"), config_path


def _protocol(matrix: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(matrix.get("protocol"), "protocol")


def _paths(matrix: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(matrix.get("paths"), "paths")


def _evaluation(matrix: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(matrix.get("evaluation"), "evaluation")


def _asset_manifest_path(matrix: Mapping[str, Any]) -> Path | None:
    """Return the explicitly selected minimal-bundle report, if any.

    The runner deliberately does not infer a backend from directories on disk:
    an absent value keeps the historical ``data_dir`` layout, while a non-empty
    value is an explicit contract for the flat bundle.
    """

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
        raise PipelineError(f"benchmark_v2_asset_manifest is not a verified report: {path}")
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
        raise PipelineError(f"{path}: images.root escapes the asset report directory") from exc
    if not root.is_dir():
        raise PipelineError(f"minimal image root does not exist: {root}")
    return path


def _asset_report(matrix: Mapping[str, Any]) -> tuple[Path, dict[str, Any]] | None:
    path = _asset_manifest_path(matrix)
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PipelineError(f"invalid benchmark_v2_asset_manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PipelineError(f"benchmark_v2_asset_manifest must be a JSON object: {path}")
    return path, dict(payload)


def _asset_expected_provenance(matrix: Mapping[str, Any]) -> dict[str, str] | None:
    report = _asset_report(matrix)
    if report is None:
        return None
    path, payload = report
    selected = payload.get("selected_images")
    if not isinstance(selected, Mapping):
        raise PipelineError(f"{path}: selected_images section is missing")
    selected_hash = selected.get("sha256")
    if not isinstance(selected_hash, str) or not selected_hash:
        raise PipelineError(f"{path}: selected_images.sha256 is missing")
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "benchmark_v2_asset_manifest": str(path),
        "benchmark_v2_asset_backend": "minimal",
        "benchmark_v2_asset_manifest_sha256": digest,
        "benchmark_v2_selected_images_sha256": selected_hash,
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


def _minimal_image_dir(matrix: Mapping[str, Any], map_name: str) -> Path | None:
    report = _asset_report(matrix)
    if report is None:
        return None
    path, payload = report
    images = payload.get("images")
    assert isinstance(images, Mapping)
    root_value = images.get("root")
    assert isinstance(root_value, str)
    return (path.parent / root_value / map_name).resolve()


def _metric_gt_dir(matrix: Mapping[str, Any], map_name: str) -> str:
    image_dir = _minimal_image_dir(matrix, map_name)
    if image_dir is not None:
        return _relative(image_dir)
    return f"{_evaluation(matrix)['data_dir']}/{map_name}/imgs"


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
            # Historical exp31--36 artifacts intentionally have no asset fields.
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
    actual_manifest = _asset_provenance_value(
        payload,
        ("benchmark_v2_asset_manifest", "asset_manifest_path", "asset_manifest"),
        ("manifest", "path"),
    )
    if not isinstance(actual_manifest, str) or _repo_path(actual_manifest).resolve() != Path(
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


def _seen_section(matrix: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping(matrix.get("seen"), "seen")


def _seen_spec(matrix: Mapping[str, Any], experiment: str) -> dict[str, Any]:
    specs = _mapping(_seen_section(matrix).get("experiments"), "seen.experiments")
    if tuple(specs) != SEEN_EXPERIMENTS or experiment not in specs:
        raise PipelineError(f"unknown Seen-10 experiment: {experiment}")
    return _mapping(specs[experiment], f"seen.experiments.{experiment}")


def _validate_setting(setting: str) -> None:
    if setting not in SETTINGS:
        raise PipelineError(f"unsupported benchmark setting: {setting}")


def _setting_maps(matrix: Mapping[str, Any], setting: str) -> list[str]:
    _validate_setting(setting)
    if setting == "crossmap":
        value = _protocol(matrix).get("crossmap_maps")
        name = "protocol.crossmap_maps"
    else:
        value = _seen_section(matrix).get("maps")
        name = "seen.maps"
    if not isinstance(value, list) or not value or any(
        not isinstance(map_name, str) or not map_name for map_name in value
    ):
        raise PipelineError(f"{name} must be a non-empty list of map names")
    if len(set(value)) != len(value):
        raise PipelineError(f"{name} contains duplicate maps")
    return list(value)


def _setting_split(setting: str, kind: str) -> str:
    _validate_setting(setting)
    if kind not in KINDS:
        raise PipelineError(f"unsupported generation kind: {kind}")
    return SETTING_SPLITS[setting][kind]


def _setting_expected_samples(matrix: Mapping[str, Any], setting: str, kind: str) -> int:
    split = _setting_split(setting, kind)
    source = _protocol(matrix).get("expected_samples") if setting == "crossmap" else _seen_section(matrix).get("expected_samples")
    values = _mapping(source, f"{setting}.expected_samples")
    return _integer(values.get(split), f"{setting}.expected_samples.{split}", 1)


def _setting_spec(
    matrix: Mapping[str, Any], experiment: str, shots: int | None, setting: str
) -> dict[str, Any]:
    _validate_setting(setting)
    if setting == "seen":
        if shots is not None:
            raise PipelineError("Seen-10 pipelines do not accept shots")
        return _seen_spec(matrix, experiment)
    return _zero_spec(matrix, experiment) if shots is None else _few_spec(matrix, experiment)


def _shots(matrix: Mapping[str, Any]) -> list[int]:
    values = _protocol(matrix).get("shots_per_map")
    if not isinstance(values, list):
        raise PipelineError("protocol.shots_per_map must be a list")
    shots = [_integer(value, "shot", 1) for value in values]
    if shots != [100, 50, 20, 10]:
        raise PipelineError("protocol.shots_per_map must be [100, 50, 20, 10]")
    return shots


def _zero_spec(matrix: Mapping[str, Any], experiment: str) -> dict[str, Any]:
    specs = _mapping(matrix.get("zero_shot"), "zero_shot")
    if tuple(specs) != ZERO_SHOT_EXPERIMENTS or experiment not in specs:
        raise PipelineError(f"unknown zero-shot experiment: {experiment}")
    return _mapping(specs[experiment], f"zero_shot.{experiment}")


def _few_spec(matrix: Mapping[str, Any], experiment: str) -> dict[str, Any]:
    specs = _mapping(matrix.get("few_shot"), "few_shot")
    if tuple(specs) != FEW_SHOT_EXPERIMENTS or experiment not in specs:
        raise PipelineError(f"unknown few-shot experiment: {experiment}")
    return _mapping(specs[experiment], f"few_shot.{experiment}")


def _command_option(command: Sequence[str], option: str) -> str:
    try:
        index = command.index(option)
    except ValueError as exc:
        raise PipelineError(f"generated command is missing {option}") from exc
    if index + 1 >= len(command):
        raise PipelineError(f"generated command has no value for {option}")
    return command[index + 1]


def _validate_seen_config_and_commands(
    matrix: Mapping[str, Any], experiment: str, kind: str
) -> None:
    _seen_spec(matrix, experiment)
    config_value = _config_for_kind(matrix, experiment, None, kind, "seen")
    config_path = _repo_path(config_value)
    if not config_path.is_file():
        raise PipelineError(f"Seen {kind} config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = _mapping(yaml.safe_load(handle), f"{config_path}")
    _validate_task_asset_config(matrix, config_path)

    expected_split = _setting_split("seen", kind)
    expected_maps = _setting_maps(matrix, "seen")
    manifest_path = _repo_path(str(_protocol(matrix)["manifest"])).resolve()
    configured_manifest = config.get("benchmark_v2_manifest")
    if not isinstance(configured_manifest, str) or not configured_manifest:
        raise PipelineError(f"{config_path}: missing benchmark_v2_manifest")
    if _repo_path(configured_manifest).resolve() != manifest_path:
        raise PipelineError(f"{config_path}: benchmark_v2_manifest mismatch")
    if config.get("benchmark_v2_split") != expected_split:
        raise PipelineError(
            f"{config_path}: expected benchmark_v2_split={expected_split!r}"
        )
    if config.get("benchmark_v2_support_seed") is not None:
        raise PipelineError(f"{config_path}: Seen config must not select a support seed")
    if config.get("benchmark_v2_shots_per_map") is not None:
        raise PipelineError(f"{config_path}: Seen config must not select shots")
    for map_key in ("train_maps", "val_maps", "test_maps"):
        if config.get(map_key) != expected_maps:
            raise PipelineError(f"{config_path}: {map_key} does not match Seen-10 maps")
    configured_checkpoint = config.get("ckpt_path")
    if not isinstance(configured_checkpoint, str) or not configured_checkpoint:
        raise PipelineError(f"{config_path}: missing ckpt_path")
    expected_checkpoint = checkpoint_path(matrix, experiment, None, "seen").resolve()
    if _repo_path(configured_checkpoint).resolve() != expected_checkpoint:
        raise PipelineError(f"{config_path}: ckpt_path does not match Seen checkpoint")
    if kind == "continuous" and config.get("is_conti_gen") is not True:
        raise PipelineError(f"{config_path}: continuous config must set is_conti_gen")
    if kind == "discrete" and config.get("is_conti_gen") is True:
        raise PipelineError(f"{config_path}: discrete config cannot set is_conti_gen")

    inference = build_inference_command(matrix, experiment, None, kind, "seen")
    expected_inference = {
        "--csgo_config": config_value,
        "--output_dir": _relative(generation_dir(matrix, experiment, None, kind, "seen")),
        "--ckpt_path": _relative(checkpoint_path(matrix, experiment, None, "seen")),
        "--seed": str(_protocol(matrix)["inference_seed"]),
        "--benchmark_v2_split": expected_split,
    }
    asset_manifest = matrix.get("benchmark_v2_asset_manifest")
    if asset_manifest not in (None, ""):
        expected_inference["--benchmark_v2_asset_manifest"] = str(asset_manifest)
    for option, value in expected_inference.items():
        if _command_option(inference, option) != value:
            raise PipelineError(f"Seen inference command mismatch for {option}")
    map_index = inference.index("--benchmark_v2_maps")
    if inference[map_index + 1 : map_index + 1 + len(expected_maps)] != expected_maps:
        raise PipelineError("Seen inference command map list mismatch")
    if "--benchmark_v2_support_seed" in inference or "--benchmark_v2_shots_per_map" in inference:
        raise PipelineError("Seen inference command must not include support-shot overrides")

    evaluation = _evaluation(matrix)
    for map_name in expected_maps:
        metric = build_metric_command(matrix, experiment, None, kind, map_name, "seen")
        expected_metric = {
            "--gt": _metric_gt_dir(matrix, map_name),
            "--pred": f"{_relative(generation_dir(matrix, experiment, None, kind, 'seen'))}/gen_imgs/{map_name}",
            "--map_name": map_name,
            "--benchmark_v2_manifest": str(_protocol(matrix)["manifest"]),
            "--benchmark_v2_split": expected_split,
            "--external_loc_repo_root": str(evaluation["external_loc_repo_root"]),
            "--external_loc_config_path": str(evaluation["external_loc_config_path"]),
            "--external_loc_checkpoint_path": str(evaluation["external_loc_checkpoint_path"]),
        }
        if matrix.get("benchmark_v2_asset_manifest") not in (None, ""):
            expected_metric["--benchmark_v2_asset_manifest"] = str(
                matrix["benchmark_v2_asset_manifest"]
            )
        for option, value in expected_metric.items():
            if _command_option(metric, option) != value:
                raise PipelineError(f"Seen metric command mismatch for {map_name}/{option}")
        aggregate = build_aggregate_command(matrix, experiment, None, kind, "seen")
        if aggregate[2] != "maps":
            raise PipelineError("Seen aggregation must use the maps aggregate")
        expected_aggregate = {
            "--manifest": str(_protocol(matrix)["manifest"]),
            "--split": expected_split,
            "--input_root": _relative(generation_dir(matrix, experiment, None, kind, "seen")),
            "--kind": kind,
            "--output": f"{_relative(generation_dir(matrix, experiment, None, kind, 'seen'))}/summary.json",
        }
        for option, value in expected_aggregate.items():
            if _command_option(aggregate, option) != value:
                raise PipelineError(f"Seen aggregate command mismatch for {option}")
        break


def _append_cli(command: list[str], values: Mapping[str, Any]) -> None:
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


def checkpoint_path(
    matrix: Mapping[str, Any], experiment: str, shots: int | None, setting: str = "crossmap"
) -> Path:
    _validate_setting(setting)
    if setting == "seen":
        return _repo_path(str(_setting_spec(matrix, experiment, shots, setting)["checkpoint"]))
    if shots is None:
        return _repo_path(str(_zero_spec(matrix, experiment)["checkpoint"]))
    seed = _integer(_protocol(matrix)["support_seed"], "support_seed")
    root = _repo_path(str(_paths(matrix)["model_root"]))
    return root / experiment / f"shot_{shots}" / f"seed_{seed}" / "model.safetensors"


def generation_dir(
    matrix: Mapping[str, Any], experiment: str, shots: int | None, kind: str,
    setting: str = "crossmap",
) -> Path:
    _validate_setting(setting)
    if kind not in KINDS:
        raise PipelineError(f"unsupported generation kind: {kind}")
    root = _repo_path(str(_paths(matrix)["generation_root"])) / experiment
    if setting == "seen":
        _setting_spec(matrix, experiment, shots, setting)
        return root / "seen" / kind
    if shots is None:
        return root / "zero_shot" / "crossmap" / kind
    seed = _integer(_protocol(matrix)["support_seed"], "support_seed")
    return root / f"shot_{shots}" / f"seed_{seed}" / kind


def log_dir(
    matrix: Mapping[str, Any], experiment: str, shots: int | None, setting: str = "crossmap"
) -> Path:
    _validate_setting(setting)
    root = _repo_path(str(_paths(matrix)["log_root"])) / experiment
    if setting == "seen":
        _setting_spec(matrix, experiment, shots, setting)
        return root / "seen"
    if shots is None:
        return root / "zero_shot" / "crossmap"
    seed = _integer(_protocol(matrix)["support_seed"], "support_seed")
    return root / f"shot_{shots}" / f"seed_{seed}"


def _config_for_kind(
    matrix: Mapping[str, Any], experiment: str, shots: int | None, kind: str,
    setting: str = "crossmap",
) -> str:
    spec = _setting_spec(matrix, experiment, shots, setting)
    return str(spec[f"{kind}_config"])


def build_train_command(
    matrix: Mapping[str, Any], experiment: str, shots: int
) -> list[str]:
    if shots not in _shots(matrix):
        raise PipelineError(f"unsupported shot count: {shots}")
    spec = _few_spec(matrix, experiment)
    ports = _mapping(spec.get("master_ports"), f"few_shot.{experiment}.master_ports")
    port = _integer(ports.get(shots, ports.get(str(shots))), "master_port", 1)
    training = _mapping(matrix.get("training"), "training")
    batches = _mapping(training.get("train_batch_size_by_shot"), "train batches")
    batch = _integer(batches.get(shots, batches.get(str(shots))), "train batch", 1)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        f"--master_port={port}",
        "train_csgo.py",
        "--csgo_config",
        str(spec["train_config"]),
    ]
    _append_cli(command, _mapping(training.get("common_args"), "training.common_args"))
    command.extend(
        (
            "--per_device_train_batch_size",
            str(batch),
            "--output_dir",
            _relative(checkpoint_path(matrix, experiment, shots).parent),
            "--max_steps",
            str(_integer(_protocol(matrix)["max_steps"], "max_steps", 1)),
        )
    )
    _append_cli(command, _mapping(spec.get("train_args"), f"{experiment}.train_args"))
    command.extend(
        (
            "--benchmark_v2_support_seed",
            str(_protocol(matrix)["support_seed"]),
            "--benchmark_v2_shots_per_map",
            str(shots),
        )
    )
    _append_asset_cli(command, matrix)
    return command


def build_inference_command(
    matrix: Mapping[str, Any], experiment: str, shots: int | None, kind: str,
    setting: str = "crossmap",
) -> list[str]:
    protocol = _protocol(matrix)
    split = _setting_split(setting, kind)
    maps = _setting_maps(matrix, setting)
    command = [
        sys.executable,
        "eval_csgo.py",
        "--csgo_config",
        _config_for_kind(matrix, experiment, shots, kind, setting),
        "--output_dir",
        _relative(generation_dir(matrix, experiment, shots, kind, setting)),
        "--ckpt_path",
        _relative(checkpoint_path(matrix, experiment, shots, setting)),
        "--seed",
        str(protocol["inference_seed"]),
        "--benchmark_v2_split",
        split,
        "--benchmark_v2_maps",
        *[str(value) for value in maps],
    ]
    if shots is not None:
        command.extend(
            (
                "--benchmark_v2_support_seed",
                str(protocol["support_seed"]),
                "--benchmark_v2_shots_per_map",
                str(shots),
            )
        )
    _append_asset_cli(command, matrix)
    return command


def build_metric_command(
    matrix: Mapping[str, Any], experiment: str, shots: int | None, kind: str, map_name: str,
    setting: str = "crossmap",
) -> list[str]:
    protocol = _protocol(matrix)
    evaluation = _evaluation(matrix)
    split = _setting_split(setting, kind)
    script = "benchmark_csgo_v1.py" if kind == "discrete" else "benchmark_csgo_v1_conti.py"
    output = generation_dir(matrix, experiment, shots, kind, setting)
    command = [
        sys.executable,
        script,
        "--gt",
        _metric_gt_dir(matrix, map_name),
        "--pred",
        f"{_relative(output)}/gen_imgs/{map_name}",
        "--batch_size",
        str(evaluation["batch_size"]),
        "--device",
        "cuda",
        "--paired_size",
        str(evaluation["paired_size"]),
    ]
    if _asset_manifest_path(matrix) is None:
        command.extend(("--data_dir", str(evaluation["data_dir"])))
    command.extend(("--map_name", map_name))
    if kind == "continuous":
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
            str(protocol["manifest"]),
            "--benchmark_v2_split",
            split,
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


def build_aggregate_command(
    matrix: Mapping[str, Any], experiment: str, shots: int | None, kind: str,
    setting: str = "crossmap",
) -> list[str]:
    split = _setting_split(setting, kind)
    output = generation_dir(matrix, experiment, shots, kind, setting)
    return [
        sys.executable,
        "scripts/aggregate_csgo_benchmark_v2_metrics.py",
        "maps",
        "--manifest",
        str(_protocol(matrix)["manifest"]),
        "--split",
        split,
        "--input_root",
        _relative(output),
        "--kind",
        kind,
        "--output",
        f"{_relative(output)}/summary.json",
    ]


def validate_configuration(matrix: Mapping[str, Any]) -> None:
    if _integer(matrix.get("schema_version"), "schema_version", 1) != 1:
        raise PipelineError("schema_version must be 1")
    protocol = _protocol(matrix)
    _asset_manifest_path(matrix)
    if _shots(matrix) != [100, 50, 20, 10]:
        raise PipelineError("invalid shot protocol")
    if protocol.get("crossmap_maps") != list(CROSSMAP_MAPS):
        raise PipelineError("crossmap map order does not match Benchmark v2")
    if protocol.get("expected_samples") != {
        "crossmap_query_test": 8000,
        "crossmap_continuous": 5120,
    }:
        raise PipelineError("expected sample counts do not match Benchmark v2")
    if protocol.get("support_seed") != 0 or protocol.get("inference_seed") != 42:
        raise PipelineError("formal protocol requires support seed 0 and inference seed 42")
    if _setting_maps(matrix, "seen") != list(SEEN_MAPS):
        raise PipelineError("Seen-10 map order does not match Benchmark v2")
    seen_expected_samples = {
        "seen_discrete_test": 20000,
        "seen_continuous": 12800,
    }
    if _mapping(_seen_section(matrix).get("expected_samples"), "seen.expected_samples") != seen_expected_samples:
        raise PipelineError("Seen-10 expected sample counts do not match Benchmark v2")
    required_files = [_repo_path(str(protocol["manifest"]))]
    evaluation = _evaluation(matrix)
    required_files.extend(
        (
            _repo_path(str(evaluation["aesthetic_checkpoint"])),
            _repo_path(str(evaluation["external_loc_repo_root"])) / str(evaluation["external_loc_config_path"]),
            _repo_path(str(evaluation["external_loc_repo_root"])) / str(evaluation["external_loc_checkpoint_path"]),
        )
    )
    zero_specs = _mapping(matrix.get("zero_shot"), "zero_shot")
    if tuple(zero_specs) != ZERO_SHOT_EXPERIMENTS:
        raise PipelineError(f"zero_shot must be ordered as {ZERO_SHOT_EXPERIMENTS}")
    for experiment in ZERO_SHOT_EXPERIMENTS:
        spec = _zero_spec(matrix, experiment)
        required_files.extend(
            (_repo_path(str(spec["checkpoint"])), _repo_path(str(spec["discrete_config"])), _repo_path(str(spec["continuous_config"])))
        )
        _validate_task_asset_config(matrix, _repo_path(str(spec["discrete_config"])))
        _validate_task_asset_config(matrix, _repo_path(str(spec["continuous_config"])))
        for kind in KINDS:
            build_inference_command(matrix, experiment, None, kind)
    seen_specs = _mapping(_seen_section(matrix).get("experiments"), "seen.experiments")
    if tuple(seen_specs) != SEEN_EXPERIMENTS:
        raise PipelineError(f"seen.experiments must be ordered as {SEEN_EXPERIMENTS}")
    for experiment in SEEN_EXPERIMENTS:
        spec = _seen_spec(matrix, experiment)
        required_files.extend(
            (
                _repo_path(str(spec["checkpoint"])),
                _repo_path(str(spec["discrete_config"])),
                _repo_path(str(spec["continuous_config"])),
            )
        )
        _validate_task_asset_config(matrix, _repo_path(str(spec["discrete_config"])))
        _validate_task_asset_config(matrix, _repo_path(str(spec["continuous_config"])))
        for kind in KINDS:
            _validate_seen_config_and_commands(matrix, experiment, kind)
    few_specs = _mapping(matrix.get("few_shot"), "few_shot")
    if tuple(few_specs) != FEW_SHOT_EXPERIMENTS:
        raise PipelineError(f"few_shot must be ordered as {FEW_SHOT_EXPERIMENTS}")
    for experiment in FEW_SHOT_EXPERIMENTS:
        spec = _few_spec(matrix, experiment)
        required_files.extend(
            (
                _repo_path(str(spec["parent_checkpoint"])),
                _repo_path(str(spec["train_config"])),
                _repo_path(str(spec["discrete_config"])),
                _repo_path(str(spec["continuous_config"])),
            )
        )
        with _repo_path(str(spec["train_config"])).open("r", encoding="utf-8") as handle:
            train_config = _mapping(yaml.safe_load(handle), f"{experiment} train config")
        _validate_task_asset_config(matrix, _repo_path(str(spec["train_config"])))
        _validate_task_asset_config(matrix, _repo_path(str(spec["discrete_config"])))
        _validate_task_asset_config(matrix, _repo_path(str(spec["continuous_config"])))
        if train_config.get("benchmark_v2_split") != "crossmap_support":
            raise PipelineError(f"{experiment} must train on crossmap_support")
        if train_config.get("benchmark_v2_support_seed") != 0 or train_config.get("benchmark_v2_shots_per_map") != 100:
            raise PipelineError(f"{experiment} must retain the seed-0 100-shot defaults")
        configured_parent = _repo_path(str(train_config.get("finetune_init_ckpt_path"))).resolve()
        if configured_parent != _repo_path(str(spec["parent_checkpoint"])).resolve():
            raise PipelineError(f"{experiment} parent checkpoint mismatch")
        for shots in _shots(matrix):
            build_train_command(matrix, experiment, shots)
            for kind in KINDS:
                build_inference_command(matrix, experiment, shots, kind)
    batches = _mapping(_mapping(matrix.get("training"), "training").get("train_batch_size_by_shot"), "train batches")
    normalized = {shot: batches.get(shot, batches.get(str(shot))) for shot in _shots(matrix)}
    if normalized != {100: 128, 50: 128, 20: 80, 10: 40}:
        raise PipelineError(f"invalid per-shot batch sizes: {normalized}")
    for path in required_files:
        if not path.is_file():
            raise PipelineError(f"required file does not exist: {path}")
    _validate_real_support_subsets(matrix)


def _validate_real_support_subsets(matrix: Mapping[str, Any]) -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from csgo_datasets.benchmark_v2 import load_benchmark_v2_selection

    spec = _few_spec(matrix, FEW_SHOT_EXPERIMENTS[0])
    with _repo_path(str(spec["train_config"])).open("r", encoding="utf-8") as handle:
        runtime = _mapping(yaml.safe_load(handle), "few-shot train config")
    runtime["benchmark_v2_manifest"] = str(_repo_path(str(runtime["benchmark_v2_manifest"])).resolve())
    if _asset_manifest_path(matrix) is not None:
        runtime["benchmark_v2_asset_manifest"] = str(
            _asset_manifest_path(matrix)
        )
    else:
        runtime["data_dir"] = str(_repo_path(str(runtime["data_dir"])).resolve())
    runtime["benchmark_v2_support_seed"] = _protocol(matrix)["support_seed"]
    runtime["benchmark_v2_shots_per_map"] = 100
    full = load_benchmark_v2_selection(runtime)
    full_rows = {
        map_name: [row["file_frame"] for row in full.rows if row["map"] == map_name]
        for map_name in full.map_names
    }
    for shots in _shots(matrix):
        runtime["benchmark_v2_shots_per_map"] = shots
        selected = load_benchmark_v2_selection(runtime)
        if len(selected.rows) != shots * 4:
            raise PipelineError(f"{shots}-shot support count is {len(selected.rows)}, expected {shots * 4}")
        for map_name in full.map_names:
            rows = [row["file_frame"] for row in selected.rows if row["map"] == map_name]
            if rows != full_rows[map_name][:shots]:
                raise PipelineError(f"{map_name} {shots}-shot support is not nested")


def _training_complete(matrix: Mapping[str, Any], experiment: str, shots: int) -> bool:
    model = checkpoint_path(matrix, experiment, shots)
    state_path = model.parent / "trainer_state.json"
    if not model.exists() and not state_path.exists():
        return False
    if not model.is_file() or model.stat().st_size < 1024 * 1024:
        raise PipelineError(f"incomplete model: {model}")
    if not state_path.is_file():
        raise PipelineError(f"missing trainer state: {state_path}")
    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    steps = _protocol(matrix)["max_steps"]
    if state.get("global_step") != steps or state.get("max_steps") != steps:
        raise PipelineError(f"{state_path}: expected global_step=max_steps={steps}")
    return True


def _expected_for_kind(
    matrix: Mapping[str, Any], kind: str, setting: str = "crossmap"
) -> tuple[str, int, int]:
    split = _setting_split(setting, kind)
    total = _setting_expected_samples(matrix, setting, kind)
    maps = _setting_maps(matrix, setting)
    if total % len(maps) != 0:
        raise PipelineError(
            f"{setting} {kind} sample count {total} is not divisible by map count {len(maps)}"
        )
    return split, total, total // len(maps)


def _expected_inference_payload(
    matrix: Mapping[str, Any], experiment: str, shots: int | None, kind: str, setting: str
) -> dict[str, Any]:
    split, total, _ = _expected_for_kind(matrix, kind, setting)
    return {
        "config_path": _config_for_kind(matrix, experiment, shots, kind, setting),
        "benchmark_v2_manifest": str(_protocol(matrix)["manifest"]),
        "benchmark_v2_split": split,
        "benchmark_v2_support_seed": None if shots is None else _protocol(matrix)["support_seed"],
        "benchmark_v2_shots_per_map": shots,
        "maps": _setting_maps(matrix, setting),
        "sample_count": total,
        "checkpoint": _relative(checkpoint_path(matrix, experiment, shots, setting)),
        "ckpt_path": _relative(checkpoint_path(matrix, experiment, shots, setting)),
        "seed": _protocol(matrix)["inference_seed"],
    }


def _validate_inference(
    matrix: Mapping[str, Any], experiment: str, shots: int | None, kind: str,
    setting: str = "crossmap",
) -> None:
    output = generation_dir(matrix, experiment, shots, kind, setting)
    manifest_path = output / "inference_manifest.json"
    if not manifest_path.is_file():
        raise PipelineError(f"missing inference manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise PipelineError(f"{manifest_path}: inference manifest must be an object")
    expected = _expected_inference_payload(matrix, experiment, shots, kind, setting)
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PipelineError(f"{manifest_path}: expected {key}={value!r}, got {payload.get(key)!r}")
    _validate_asset_provenance(matrix, payload, manifest_path)
    _, _, per_map = _expected_for_kind(matrix, kind, setting)
    for map_name in _setting_maps(matrix, setting):
        count = len(list((output / "gen_imgs" / map_name).glob("*.jpg")))
        if count != per_map:
            raise PipelineError(f"{output}: {map_name} has {count} images, expected {per_map}")


def _inference_complete(
    matrix: Mapping[str, Any], experiment: str, shots: int | None, kind: str,
    setting: str = "crossmap",
) -> bool:
    path = generation_dir(matrix, experiment, shots, kind, setting) / "inference_manifest.json"
    if not path.exists():
        return False
    _validate_inference(matrix, experiment, shots, kind, setting)
    return True


def _per_map_metric_path(output: Path, kind: str, map_name: str) -> Path:
    stem = "benchmark_csgo_v2" if kind == "discrete" else "benchmark_csgo_v2_conti"
    return output / f"{stem}_{map_name}.json"


def _per_map_metric_complete(output: Path, kind: str, map_name: str, expected_count: int) -> bool:
    path = _per_map_metric_path(output, kind, map_name)
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise PipelineError(f"invalid metric JSON: {path}")
    count = payload.get("common_count", payload.get("Common_Count"))
    if count is None and isinstance(payload.get("metrics"), Mapping):
        count = payload["metrics"].get("Common_Count")
    if count != expected_count:
        raise PipelineError(f"{path}: Common_Count={count}, expected {expected_count}")
    return True


def _validate_summary(
    matrix: Mapping[str, Any], experiment: str, shots: int | None, kind: str,
    setting: str = "crossmap",
) -> dict[str, Any]:
    path = generation_dir(matrix, experiment, shots, kind, setting) / "summary.json"
    if not path.is_file():
        raise PipelineError(f"missing summary: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    split, total, _ = _expected_for_kind(matrix, kind, setting)
    maps = _setting_maps(matrix, setting)
    expected = {
        "manifest": str(_repo_path(str(_protocol(matrix)["manifest"])).resolve()),
        "split": split,
        "kind": kind,
        "maps": maps,
        "checkpoint": _relative(checkpoint_path(matrix, experiment, shots, setting)),
        "ckpt_path": _relative(checkpoint_path(matrix, experiment, shots, setting)),
        "inference_seed": _protocol(matrix)["inference_seed"],
        "support_seed": None if shots is None else _protocol(matrix)["support_seed"],
        "shots_per_map": shots,
        "sample_count": total,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise PipelineError(f"{path}: expected {key}={value!r}, got {payload.get(key)!r}")
    if list(_mapping(payload.get("per_map"), "per_map")) != maps:
        raise PipelineError(f"{path}: per_map order mismatch")
    per_map = _mapping(payload.get("per_map"), "per_map")
    for map_name in maps:
        metrics = _mapping(per_map[map_name], f"per_map.{map_name}")
        for metric, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise PipelineError(f"{path}: per_map.{map_name}.{metric} must be finite")
    macro = _mapping(payload.get("metrics_macro_map"), "metrics_macro_map")
    if not macro or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in macro.values()):
        raise PipelineError(f"{path}: macro metrics must be finite numbers")
    expected_inference = _expected_inference_payload(matrix, experiment, shots, kind, setting)
    provenance = _mapping(payload.get("inference_provenance"), "inference_provenance")
    provenance_payload = _mapping(provenance.get("payload"), "inference_provenance.payload")
    for key, value in expected_inference.items():
        actual = provenance_payload.get(key)
        matches = actual == value
        if key == "benchmark_v2_manifest" and isinstance(actual, str):
            matches = _repo_path(actual).resolve() == _repo_path(str(value)).resolve()
        if not matches:
            raise PipelineError(
                f"{path}: inference provenance payload {key}={actual!r}, expected {value!r}"
            )
    _validate_asset_provenance(matrix, provenance_payload, path)
    if not isinstance(provenance.get("path"), str) or not provenance["path"]:
        raise PipelineError(f"{path}: inference provenance path is missing")
    _validate_inference(matrix, experiment, shots, kind, setting)
    return dict(payload)


def _summary_complete(
    matrix: Mapping[str, Any], experiment: str, shots: int | None, kind: str,
    setting: str = "crossmap",
) -> bool:
    path = generation_dir(matrix, experiment, shots, kind, setting) / "summary.json"
    if not path.exists():
        return False
    _validate_summary(matrix, experiment, shots, kind, setting)
    return True


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _pipeline_lock_path(
    matrix: Mapping[str, Any], experiment: str, shots: int
) -> Path:
    return log_dir(matrix, experiment, shots) / ".pipeline.lock"


def _pipeline_lock_is_held(path: Path) -> bool:
    """Return whether another process currently owns a pipeline lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return True
            raise
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False


def _adopt_held_pipeline_locks(
    matrix: Mapping[str, Any],
    pending: list[tuple[str, int]],
    external: set[tuple[str, int]],
) -> list[tuple[str, int]]:
    adopted: list[tuple[str, int]] = []
    for job in list(pending):
        if not _pipeline_lock_is_held(_pipeline_lock_path(matrix, *job)):
            continue
        pending.remove(job)
        external.add(job)
        adopted.append(job)
    return adopted


def _run_command(command: Sequence[str], log_path: Path, cuda_device: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = cuda_device
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{time.strftime('%F %T')}] COMMAND {shlex.join(command)}\n")
        result = subprocess.run(
            list(command), cwd=REPO_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False
        )
        log.write(f"[{time.strftime('%F %T')}] EXIT {result.returncode}\n")
    if result.returncode != 0:
        raise PipelineError(f"command failed with exit {result.returncode}; see {log_path}")


def _metric_value(payload: Mapping[str, Any], name: str, digits: int) -> str:
    value = _mapping(payload.get("metrics_macro_map"), "metrics_macro_map").get(name)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PipelineError(f"summary metric {name} is missing or non-finite")
    return f"{value:.{digits}f}"


def _setting_status(
    matrix: Mapping[str, Any], experiment: str, shots: int | None, setting: str
) -> tuple[str, str]:
    discrete_infer = _inference_complete(matrix, experiment, shots, "discrete", setting)
    continuous_infer = _inference_complete(matrix, experiment, shots, "continuous", setting)
    discrete_metric = _summary_complete(matrix, experiment, shots, "discrete", setting)
    continuous_metric = _summary_complete(matrix, experiment, shots, "continuous", setting)
    if setting == "crossmap":
        if discrete_infer and continuous_infer:
            inference = "✅ CrossMap-4：离散 8,000 + 连续 5,120"
        elif discrete_infer:
            inference = "✅ CrossMap-4 离散 8,000；连续待执行"
        elif continuous_infer:
            inference = "CrossMap-4 离散待执行；✅ 连续 5,120"
        else:
            inference = ""
        if discrete_metric and continuous_metric:
            metric = "✅ CrossMap-4 离散/连续"
        elif discrete_metric:
            metric = "✅ CrossMap-4 离散；连续待执行"
        elif continuous_metric:
            metric = "CrossMap-4 离散待执行；✅ 连续"
        else:
            metric = ""
        return inference, metric

    if discrete_infer and continuous_infer:
        inference = "✅ Seen-10：离散 20,000 + 连续 12,800"
    elif discrete_infer:
        inference = "✅ Seen-10 离散 20,000；连续待执行"
    elif continuous_infer:
        inference = "Seen-10 离散待执行；✅ 连续 12,800"
    else:
        inference = ""
    if discrete_metric and continuous_metric:
        metric = "✅ Seen-10 离散/连续"
    elif discrete_metric:
        metric = "✅ Seen-10 离散；连续待执行"
    elif continuous_metric:
        metric = "Seen-10 离散待执行；✅ 连续"
    else:
        metric = ""
    return inference, metric


def _status_cells(
    matrix: Mapping[str, Any], experiment: str, shots: int | None
) -> tuple[str, str, str]:
    if shots is None:
        train = "✅ Seen-10，step 19500" if checkpoint_path(matrix, experiment, None).is_file() else "未开始"
    else:
        train = "✅ step 400" if _training_complete(matrix, experiment, shots) else "未开始"
    crossmap_inference, crossmap_metric = _setting_status(
        matrix, experiment, shots, "crossmap"
    )
    if shots is None:
        seen_inference, seen_metric = _setting_status(
            matrix, experiment, None, "seen"
        )
        inference = "<br>".join(value for value in (seen_inference, crossmap_inference) if value) or "未开始"
        metric = "<br>".join(value for value in (seen_metric, crossmap_metric) if value) or "未开始"
    else:
        inference = crossmap_inference or "未开始"
        metric = crossmap_metric or "未开始"
    return train, inference, metric


def _replace_row(lines: list[str], prefix: str, replacement: str) -> bool:
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = replacement
            return True
    return False


def _insert_rows_before_blank(lines: list[str], anchor_prefix: str, rows: list[str]) -> None:
    for index, line in enumerate(lines):
        if line.startswith(anchor_prefix):
            lines[index + 1:index + 1] = rows
            return
    raise PipelineError(f"results table anchor is missing: {anchor_prefix}")


def sync_results(matrix: Mapping[str, Any]) -> None:
    path = _repo_path(str(_paths(matrix)["results_file"]))
    if not path.is_file():
        raise PipelineError(f"results file does not exist: {path}")
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _exclusive_lock(lock_path):
        text = path.read_text(encoding="utf-8")
        headings = [line for line in text.splitlines() if line.startswith("#")]
        expected_headings = [
            "# csgo benchmark v2 实验进度",
            "# csgo benchmark v2 主表",
            "## 定位",
            "## 离散生成",
            "## 连续生成",
        ]
        if headings != expected_headings:
            raise PipelineError("results file headings changed; refusing to rewrite")
        lines = text.splitlines()
        for experiment in ZERO_SHOT_EXPERIMENTS:
            train, inference, metric = _status_cells(matrix, experiment, None)
            if not _replace_row(lines, f"| `{experiment}` |", f"| `{experiment}` | {train} | {inference} | {metric} |"):
                raise PipelineError(f"missing progress row for {experiment}")
        for experiment in FEW_SHOT_EXPERIMENTS:
            old_prefix = f"| `{experiment}` |"
            old_index = next((index for index, line in enumerate(lines) if line.startswith(old_prefix)), None)
            if old_index is not None:
                del lines[old_index]
            for shots in _shots(matrix):
                train, inference, metric = _status_cells(matrix, experiment, shots)
                row = f"| `{experiment}` {shots}-shot | {train} | {inference} | {metric} |"
                prefix = f"| `{experiment}` {shots}-shot |"
                if not _replace_row(lines, prefix, row):
                    anchor = "| `exp33_loc` 10-shot |" if experiment == "exp33_gen" else "| `exp34_loc` 10-shot |"
                    anchor_index = next((index for index, line in enumerate(lines) if line.startswith(anchor)), None)
                    if anchor_index is None:
                        raise PipelineError(f"missing progress anchor for {experiment}")
                    while anchor_index + 1 < len(lines) and lines[anchor_index + 1].startswith(f"| `{experiment}` "):
                        anchor_index += 1
                    lines.insert(anchor_index + 1, row)

        for kind, heading in (("discrete", "## 离散生成"), ("continuous", "## 连续生成")):
            section_start = lines.index(heading)
            section_end = next((i for i in range(section_start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
            for experiment in ZERO_SHOT_EXPERIMENTS:
                summary = _validate_summary(matrix, experiment, None, kind) if _summary_complete(matrix, experiment, None, kind) else None
                shot = "-"
                if kind == "discrete":
                    values = [
                        _metric_value(summary, "PSNR", 3), _metric_value(summary, "SSIM", 4),
                        _metric_value(summary, "LPIPS", 4), _metric_value(summary, "Boundary_F1", 4),
                        _metric_value(summary, "FID", 3),
                    ] if summary else [""] * 5
                else:
                    values = [
                        _metric_value(summary, "PSNR", 3), _metric_value(summary, "SSIM", 4),
                        _metric_value(summary, "LPIPS", 4), _metric_value(summary, "Temporal_Warping_Error", 3),
                        _metric_value(summary, "Temporal_Difference_Error", 3), _metric_value(summary, "FVD", 3),
                    ] if summary else [""] * 6
                row = f"| CrossMap-4 zero-shot | {'Discrete' if kind == 'discrete' else 'Continuous'} generation | {experiment} | {shot} | " + " | ".join(values) + " |"
                prefix = f"| CrossMap-4 zero-shot | {'Discrete' if kind == 'discrete' else 'Continuous'} generation | {experiment} |"
                found = False
                for index in range(section_start, section_end):
                    if lines[index].startswith(prefix):
                        lines[index] = row
                        found = True
                        break
                if not found:
                    raise PipelineError(f"missing main-table row for {experiment}/{kind}")

            for experiment in SEEN_EXPERIMENTS:
                summary = (
                    _validate_summary(matrix, experiment, None, kind, "seen")
                    if _summary_complete(matrix, experiment, None, kind, "seen")
                    else None
                )
                if kind == "discrete":
                    values = [
                        _metric_value(summary, "PSNR", 3), _metric_value(summary, "SSIM", 4),
                        _metric_value(summary, "LPIPS", 4), _metric_value(summary, "Boundary_F1", 4),
                        _metric_value(summary, "FID", 3),
                    ] if summary else [""] * 5
                else:
                    values = [
                        _metric_value(summary, "PSNR", 3), _metric_value(summary, "SSIM", 4),
                        _metric_value(summary, "LPIPS", 4), _metric_value(summary, "Temporal_Warping_Error", 3),
                        _metric_value(summary, "Temporal_Difference_Error", 3), _metric_value(summary, "FVD", 3),
                    ] if summary else [""] * 6
                row = f"| Seen-10 | {'Discrete' if kind == 'discrete' else 'Continuous'} generation | {experiment} | - | " + " | ".join(values) + " |"
                prefix = f"| Seen-10 | {'Discrete' if kind == 'discrete' else 'Continuous'} generation | {experiment} |"
                found = False
                for index in range(section_start, section_end):
                    if lines[index].startswith(prefix):
                        lines[index] = row
                        found = True
                        break
                if not found:
                    raise PipelineError(f"missing Seen-10 main-table row for {experiment}/{kind}")

            existing_prefixes = tuple(f"| CrossMap-4 few-shot | {'Discrete' if kind == 'discrete' else 'Continuous'} generation | {experiment} |" for experiment in FEW_SHOT_EXPERIMENTS)
            lines = [line for line in lines if not line.startswith(existing_prefixes)]
            section_start = lines.index(heading)
            section_end = next((i for i in range(section_start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
            insert_at = section_end
            while insert_at > section_start and lines[insert_at - 1] == "":
                insert_at -= 1
            new_rows: list[str] = []
            for experiment in FEW_SHOT_EXPERIMENTS:
                for shots in _shots(matrix):
                    summary = _validate_summary(matrix, experiment, shots, kind) if _summary_complete(matrix, experiment, shots, kind) else None
                    if kind == "discrete":
                        values = [
                            _metric_value(summary, "PSNR", 3), _metric_value(summary, "SSIM", 4),
                            _metric_value(summary, "LPIPS", 4), _metric_value(summary, "Boundary_F1", 4),
                            _metric_value(summary, "FID", 3),
                        ] if summary else [""] * 5
                    else:
                        values = [
                            _metric_value(summary, "PSNR", 3), _metric_value(summary, "SSIM", 4),
                            _metric_value(summary, "LPIPS", 4), _metric_value(summary, "Temporal_Warping_Error", 3),
                            _metric_value(summary, "Temporal_Difference_Error", 3), _metric_value(summary, "FVD", 3),
                        ] if summary else [""] * 6
                    new_rows.append(
                        f"| CrossMap-4 few-shot | {'Discrete' if kind == 'discrete' else 'Continuous'} generation | {experiment} | {shots} | "
                        + " | ".join(values) + " |"
                    )
            lines[insert_at:insert_at] = new_rows

        rendered = "\n".join(lines).rstrip() + "\n"
        new_headings = [line for line in rendered.splitlines() if line.startswith("#")]
        if new_headings != expected_headings:
            raise PipelineError("results rewrite would add or remove headings")
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(rendered)
            temp_path = Path(handle.name)
        os.replace(temp_path, path)


def _run_generation_stages(
    matrix: Mapping[str, Any], experiment: str, shots: int | None, cuda_device: str, dry_run: bool,
    setting: str = "crossmap", kinds: Sequence[str] = KINDS,
) -> None:
    for kind in kinds:
        output = generation_dir(matrix, experiment, shots, kind, setting)
        label = "seen" if setting == "seen" else (shots or "zero")
        if dry_run:
            print(f"[{experiment}/{label}/{kind}:inference] {shlex.join(build_inference_command(matrix, experiment, shots, kind, setting))}")
            for map_name in _setting_maps(matrix, setting):
                print(f"[{experiment}/{label}/{kind}:metric:{map_name}] {shlex.join(build_metric_command(matrix, experiment, shots, kind, map_name, setting))}")
            print(f"[{experiment}/{label}/{kind}:aggregate] {shlex.join(build_aggregate_command(matrix, experiment, shots, kind, setting))}")
            continue
        if not _inference_complete(matrix, experiment, shots, kind, setting):
            print(f"START {experiment}/{label} {kind} inference", flush=True)
            _run_command(build_inference_command(matrix, experiment, shots, kind, setting), log_dir(matrix, experiment, shots, setting) / f"{kind}_inference.log", cuda_device)
            _validate_inference(matrix, experiment, shots, kind, setting)
            sync_results(matrix)
        else:
            print(f"SKIP complete {experiment}/{label} {kind} inference", flush=True)
        if not _summary_complete(matrix, experiment, shots, kind, setting):
            _, _, expected_per_map = _expected_for_kind(matrix, kind, setting)
            for map_name in _setting_maps(matrix, setting):
                if _per_map_metric_complete(output, kind, map_name, expected_per_map):
                    print(f"SKIP complete {experiment}/{label} {kind} metric {map_name}", flush=True)
                    continue
                print(f"START {experiment}/{label} {kind} metric {map_name}", flush=True)
                _run_command(build_metric_command(matrix, experiment, shots, kind, map_name, setting), log_dir(matrix, experiment, shots, setting) / f"{kind}_metric_{map_name}.log", cuda_device)
                if not _per_map_metric_complete(output, kind, map_name, expected_per_map):
                    raise PipelineError(f"metric did not produce valid output for {experiment}/{kind}/{map_name}")
            _run_command(build_aggregate_command(matrix, experiment, shots, kind, setting), log_dir(matrix, experiment, shots, setting) / f"{kind}_aggregate.log", cuda_device)
            _validate_summary(matrix, experiment, shots, kind, setting)
            sync_results(matrix)
        else:
            print(f"SKIP complete {experiment}/{label} {kind} metrics", flush=True)


def run_seen(
    matrix: Mapping[str, Any], experiment: str, kind: str, cuda_device: str, dry_run: bool = False
) -> None:
    _seen_spec(matrix, experiment)
    if kind not in KINDS:
        raise PipelineError(f"unsupported generation kind: {kind}")
    _validate_seen_config_and_commands(matrix, experiment, kind)
    checkpoint = checkpoint_path(matrix, experiment, None, "seen")
    if not checkpoint.is_file():
        raise PipelineError(f"missing Seen-10 checkpoint: {checkpoint}")
    directory = log_dir(matrix, experiment, None, "seen")
    if dry_run:
        _run_generation_stages(
            matrix, experiment, None, cuda_device, True, "seen", (kind,)
        )
        return
    with _exclusive_lock(directory / f".{kind}.pipeline.lock"):
        _run_generation_stages(
            matrix, experiment, None, cuda_device, False, "seen", (kind,)
        )
        sync_results(matrix)
        print(f"DONE {experiment} Seen-10 {kind} pipeline", flush=True)


def run_zero_shot(
    matrix: Mapping[str, Any], experiment: str, cuda_device: str, dry_run: bool = False
) -> None:
    _zero_spec(matrix, experiment)
    if not checkpoint_path(matrix, experiment, None).is_file():
        raise PipelineError(f"missing zero-shot checkpoint: {checkpoint_path(matrix, experiment, None)}")
    directory = log_dir(matrix, experiment, None)
    with _exclusive_lock(directory / ".pipeline.lock"):
        _run_generation_stages(matrix, experiment, None, cuda_device, dry_run)
        if not dry_run:
            sync_results(matrix)
            print(f"DONE {experiment} CrossMap-4 zero-shot pipeline", flush=True)


def run_few_shot(
    matrix: Mapping[str, Any], experiment: str, shots: int, cuda_device: str, dry_run: bool = False
) -> None:
    _few_spec(matrix, experiment)
    if shots not in _shots(matrix):
        raise PipelineError(f"unsupported shot count: {shots}")
    directory = log_dir(matrix, experiment, shots)
    if dry_run:
        print(f"[{experiment}/{shots}:train] {shlex.join(build_train_command(matrix, experiment, shots))}")
        _run_generation_stages(matrix, experiment, shots, cuda_device, True)
        return
    with _exclusive_lock(directory / ".pipeline.lock"):
        if not _training_complete(matrix, experiment, shots):
            print(f"START {experiment}/{shots} training", flush=True)
            _run_command(build_train_command(matrix, experiment, shots), directory / "train.log", cuda_device)
            if not _training_complete(matrix, experiment, shots):
                raise PipelineError(f"training did not produce a complete model: {experiment}/{shots}")
            sync_results(matrix)
        else:
            print(f"SKIP complete {experiment}/{shots} training", flush=True)
        _run_generation_stages(matrix, experiment, shots, cuda_device, False)
        sync_results(matrix)
        print(f"DONE {experiment}/{shots} pipeline", flush=True)


def _pipeline_complete(matrix: Mapping[str, Any], experiment: str, shots: int) -> bool:
    return _training_complete(matrix, experiment, shots) and all(
        _summary_complete(matrix, experiment, shots, kind) for kind in KINDS
    )


def _gpu_free_memory_mb(cuda_device: str) -> int:
    result = subprocess.run(
        ["nvidia-smi", f"--id={cuda_device}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        raise PipelineError(f"nvidia-smi failed: {result.stderr.strip()}")
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (IndexError, ValueError) as exc:
        raise PipelineError(f"cannot parse GPU free memory: {result.stdout!r}") from exc


def schedule_few_shot(
    matrix: Mapping[str, Any], config_path: Path, cuda_device: str
) -> None:
    scheduling = _scheduling(matrix)
    max_parallel = _integer(scheduling["max_parallel_few_shot_pipelines"], "max_parallel", 1)
    minimum_free = _integer(scheduling["minimum_free_memory_mb"], "minimum_free_memory_mb", 1)
    settle = _integer(scheduling["launch_settle_seconds"], "launch_settle_seconds")
    poll = _integer(scheduling["poll_seconds"], "poll_seconds", 1)
    jobs = [(experiment, shots) for shots in _shots(matrix) for experiment in FEW_SHOT_EXPERIMENTS]
    pending: list[tuple[str, int]] = []
    active: dict[tuple[str, int], tuple[subprocess.Popen[Any], TextIO]] = {}
    external: set[tuple[str, int]] = set()
    failures: list[tuple[str, int, int]] = []
    last_launch = 0.0
    root = _repo_path(str(_paths(matrix)["log_root"]))
    with _exclusive_lock(root / ".scheduler.lock"):
        for job in jobs:
            if _pipeline_lock_is_held(_pipeline_lock_path(matrix, *job)):
                pending.append(job)
            elif not _pipeline_complete(matrix, *job):
                pending.append(job)
        initially_external = _adopt_held_pipeline_locks(matrix, pending, external)
        print(f"PENDING few-shot pipelines: {pending}", flush=True)
        for job in initially_external:
            print(
                f"ADOPT external {job[0]}/{job[1]} pipeline lock; "
                f"scheduler_active={len(active)}/{max_parallel} "
                f"external={len(external)}",
                flush=True,
            )
        while pending or active or external:
            for job in list(external):
                lock_path = _pipeline_lock_path(matrix, *job)
                if _pipeline_lock_is_held(lock_path):
                    continue
                external.remove(job)
                try:
                    complete = _pipeline_complete(matrix, *job)
                except PipelineError as exc:
                    complete = False
                    print(
                        f"REQUEUE external {job[0]}/{job[1]} lock released; "
                        f"validation={exc}",
                        flush=True,
                    )
                if complete:
                    print(f"COMPLETE external {job[0]}/{job[1]}", flush=True)
                elif job not in pending:
                    pending.append(job)
                    print(
                        f"REQUEUE external {job[0]}/{job[1]} lock released "
                        "before pipeline completion",
                        flush=True,
                    )

            for job in _adopt_held_pipeline_locks(matrix, pending, external):
                print(
                    f"ADOPT external {job[0]}/{job[1]} pipeline lock; "
                    f"scheduler_active={len(active)}/{max_parallel} "
                    f"external={len(external)}",
                    flush=True,
                )

            for job, (process, handle) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                del active[job]
                if code == 0 and _pipeline_complete(matrix, *job):
                    print(f"COMPLETE {job[0]}/{job[1]}", flush=True)
                else:
                    failures.append((job[0], job[1], code))
                    print(f"FAILED {job[0]}/{job[1]} exit={code}", flush=True)
            halt = bool(failures and scheduling.get("stop_launching_on_failure", True))
            now = time.monotonic()
            if pending and not halt and len(active) < max_parallel and now - last_launch >= settle:
                next_job = pending[0]
                if _pipeline_lock_is_held(_pipeline_lock_path(matrix, *next_job)):
                    pending.pop(0)
                    external.add(next_job)
                    print(
                        f"ADOPT external {next_job[0]}/{next_job[1]} pipeline lock; "
                        f"scheduler_active={len(active)}/{max_parallel} "
                        f"external={len(external)}",
                        flush=True,
                    )
                else:
                    free_mb = _gpu_free_memory_mb(cuda_device)
                    if free_mb >= minimum_free:
                        experiment, shots = pending.pop(0)
                        pipeline_log = log_dir(matrix, experiment, shots) / "pipeline.log"
                        pipeline_log.parent.mkdir(parents=True, exist_ok=True)
                        handle = pipeline_log.open("a", encoding="utf-8", buffering=1)
                        command = [
                            sys.executable, str(Path(__file__).resolve()), "--config", str(config_path),
                            "run-few-shot", "--experiment", experiment, "--shots", str(shots),
                            "--cuda-device", cuda_device,
                        ]
                        _append_asset_cli(command, matrix)
                        environment = dict(os.environ)
                        environment["CUDA_VISIBLE_DEVICES"] = cuda_device
                        environment["PYTHONUNBUFFERED"] = "1"
                        process = subprocess.Popen(
                            command, cwd=REPO_ROOT, env=environment, stdout=handle,
                            stderr=subprocess.STDOUT, start_new_session=True,
                        )
                        active[(experiment, shots)] = (process, handle)
                        last_launch = now
                        print(
                            f"LAUNCH {experiment}/{shots} pid={process.pid} "
                            f"free_mb_before={free_mb} "
                            f"scheduler_active={len(active)}/{max_parallel} "
                            f"external={len(external)}",
                            flush=True,
                        )
                    else:
                        print(
                            f"WAIT GPU free memory {free_mb} MB < {minimum_free} MB; "
                            f"scheduler_active={list(active)} "
                            f"external={sorted(external)}",
                            flush=True,
                        )
            if halt and not active and not external:
                break
            if pending or active or external:
                time.sleep(poll)
    if failures:
        raise PipelineError("few-shot failures: " + ", ".join(f"{name}/{shots}:exit={code}" for name, shots, code in failures))
    print("DONE all generation few-shot pipelines", flush=True)


def print_status(matrix: Mapping[str, Any], cuda_device: str) -> None:
    print(f"GPU {cuda_device} free_memory_mb={_gpu_free_memory_mb(cuda_device)}")
    for experiment in ZERO_SHOT_EXPERIMENTS:
        states = []
        for kind in KINDS:
            states.append(f"{kind}=infer:{_inference_complete(matrix, experiment, None, kind)},metric:{_summary_complete(matrix, experiment, None, kind)}")
        print(f"{experiment:9s} zero-shot " + " ".join(states))
    for shots in _shots(matrix):
        for experiment in FEW_SHOT_EXPERIMENTS:
            try:
                train = _training_complete(matrix, experiment, shots)
                states = [f"{kind}=infer:{_inference_complete(matrix, experiment, shots, kind)},metric:{_summary_complete(matrix, experiment, shots, kind)}" for kind in KINDS]
                complete = train and all(_summary_complete(matrix, experiment, shots, kind) for kind in KINDS)
                print(f"{experiment:9s} shot={shots:3d} {'complete' if complete else 'pending ':8s} train={train} " + " ".join(states))
            except PipelineError as exc:
                print(f"{experiment:9s} shot={shots:3d} invalid  {exc}")


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
    sync = subparsers.add_parser("sync-results")
    sync.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)
    status = subparsers.add_parser("status")
    status.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)
    status.add_argument("--cuda-device", default=None)
    zero = subparsers.add_parser("run-zero-shot")
    zero.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)
    zero.add_argument("--experiment", choices=ZERO_SHOT_EXPERIMENTS, required=True)
    zero.add_argument("--cuda-device", default=None)
    zero.add_argument("--dry-run", action="store_true")
    seen = subparsers.add_parser("run-seen")
    seen.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)
    seen.add_argument("--experiment", choices=SEEN_EXPERIMENTS, required=True)
    seen.add_argument("--kind", choices=KINDS, required=True)
    seen.add_argument("--cuda-device", default=None)
    seen.add_argument("--dry-run", action="store_true")
    few = subparsers.add_parser("run-few-shot")
    few.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)
    few.add_argument("--experiment", choices=FEW_SHOT_EXPERIMENTS, required=True)
    few.add_argument("--shots", choices=(100, 50, 20, 10), required=True, type=int)
    few.add_argument("--cuda-device", default=None)
    few.add_argument("--dry-run", action="store_true")
    schedule = subparsers.add_parser("schedule-few-shot")
    schedule.add_argument("--benchmark_v2_asset_manifest", default=argparse.SUPPRESS)
    schedule.add_argument("--cuda-device", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        matrix, config_path = load_matrix(args.config)
        if hasattr(args, "benchmark_v2_asset_manifest"):
            matrix = dict(matrix)
            matrix["benchmark_v2_asset_manifest"] = args.benchmark_v2_asset_manifest
        validate_configuration(matrix)
        cuda_device = getattr(args, "cuda_device", None) or str(_scheduling(matrix)["cuda_device"])
        if args.command == "validate":
            print(f"VALID {config_path}")
        elif args.command == "sync-results":
            sync_results(matrix)
            print(f"UPDATED {_repo_path(str(_paths(matrix)['results_file']))}")
        elif args.command == "status":
            print_status(matrix, cuda_device)
        elif args.command == "run-zero-shot":
            run_zero_shot(matrix, args.experiment, cuda_device, args.dry_run)
        elif args.command == "run-seen":
            run_seen(matrix, args.experiment, args.kind, cuda_device, args.dry_run)
        elif args.command == "run-few-shot":
            run_few_shot(matrix, args.experiment, args.shots, cuda_device, args.dry_run)
        elif args.command == "schedule-few-shot":
            schedule_few_shot(matrix, config_path, cuda_device)
        else:
            raise PipelineError(f"unsupported command: {args.command}")
    except (PipelineError, OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
