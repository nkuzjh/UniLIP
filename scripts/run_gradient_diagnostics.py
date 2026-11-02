#!/usr/bin/env python3
"""Plan or serially run the fixed Seen-3 gradient diagnostic queue.

Planning is read-only. --execute is required to launch any probe or write the
status manifest. Every probe is a new process; the controller never trains.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "csgo_configs/gradient_diagnostics_seen3.yaml"
STAGES = ("screen", "confirm", "temporal")
MIN_FREE_GIB = 34.0
MICRO2_FREE_GIB = 55.0
RESERVE_GIB = 6.0
CODE_FILES = (
    "scripts/run_gradient_diagnostics.py",
    "scripts/probe_multitask_gradients.py",
    "unilip/train/gradient_diagnostics.py",
    "unilip/train/gradient_probe_loader.py",
    "unilip/model/language_model/unified_unilip.py",
    "train_csgo.py",
    "csgo_datasets/unified_task_dataset.py",
    "csgo_datasets/benchmark_v2.py",
)


@dataclass(frozen=True)
class WorkItem:
    stage: str
    experiment: str
    checkpoint_step: int
    source_offset: int
    noise_seeds: tuple[int, ...]
    source_seed: int
    samples_per_map: int
    coordinate: str
    bf16: bool
    gradient_checkpointing: bool
    precision_reference: bool
    deterministic_probe: bool
    preferred_microbatch: int
    check_total_backward: bool
    numerical_check: bool
    measurement_mode: str

    @property
    def name(self) -> str:
        return f"{self.stage}.{self.experiment}.step{self.checkpoint_step}.offset{self.source_offset}"

    @property
    def probe_stage(self) -> str:
        return "screen" if self.stage == "screen" else "effective"

    @property
    def source_count(self) -> int:
        return 96 if self.stage == "screen" else 128

    def signature(self) -> str:
        body = {key: getattr(self, key) for key in self.__dataclass_fields__}
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def path_in_repo(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def code_sha256() -> str:
    digest = hashlib.sha256()
    for relative in CODE_FILES:
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def run_signature(item: WorkItem, *, code_hash: str, config_hash: str) -> str:
    body = {"item": item.signature(), "code_sha256": code_hash,
            "queue_config_sha256": config_hash}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected gradient diagnostics config schema_version=1")
    if int(config["samples_per_map"]) != 128:
        raise ValueError("Fixed Seen-3 source pool requires 128 samples per map")
    if config.get("coordinate") != "shared":
        raise ValueError("This queue requires the shared parameter coordinate")
    preferred = config.get("preferred_microbatch")
    if not isinstance(preferred, dict) or any(value not in (1, 2, 4, 8) for value in preferred.values()):
        raise ValueError("preferred_microbatch must map experiments to 1, 2, 4, or 8")
    for stage in STAGES:
        mode = config[stage].get("measurement_mode", "full")
        if mode not in ("full", "component_sum") or (stage == "screen" and mode != "full"):
            raise ValueError(f"Invalid {stage} measurement_mode: {mode!r}")
        if mode == "component_sum" and (not config["precision_reference"]
                                        or not config["deterministic_probe"]):
            raise ValueError("component_sum queue requires FP32 deterministic probe")
    return config


def build_items(config: dict[str, Any], stages: tuple[str, ...] = STAGES) -> list[WorkItem]:
    def preferred_microbatch(experiment: str) -> int:
        try:
            return int(config["preferred_microbatch"][experiment])
        except KeyError as exc:
            raise ValueError(f"Missing preferred_microbatch for {experiment}") from exc

    common = dict(
        source_seed=int(config["source_seed"]),
        samples_per_map=int(config["samples_per_map"]),
        coordinate=str(config["coordinate"]),
        bf16=bool(config["bf16"]),
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        precision_reference=bool(config["precision_reference"]),
        deterministic_probe=bool(config["deterministic_probe"]),
    )
    items = []
    if "screen" in stages:
        section = config["screen"]
        for experiment in section["experiments"]:
            items.append(WorkItem("screen", experiment, int(section["checkpoint_step"]), 0,
                                  tuple(section["noise_seeds"]),
                                  preferred_microbatch=preferred_microbatch(experiment),
                                  check_total_backward=bool(section.get("check_total_backward", False)),
                                  numerical_check=bool(section.get("numerical_check", False)),
                                  measurement_mode=str(section.get("measurement_mode", "full")),
                                  **common))
    if "confirm" in stages:
        section = config["confirm"]
        for experiment in section["experiments"]:
            for offset in section["source_offsets"]:
                items.append(WorkItem("confirm", experiment, int(section["checkpoint_step"]),
                                      int(offset), tuple(section["noise_seeds"]),
                                      preferred_microbatch=preferred_microbatch(experiment),
                                      check_total_backward=bool(section.get("check_total_backward", False)),
                                      numerical_check=bool(section.get("numerical_check", False)),
                                      measurement_mode=str(section.get("measurement_mode", "full")),
                                      **common))
    if "temporal" in stages:
        section = config["temporal"]
        for checkpoint in section["checkpoints"]:
            items.append(WorkItem("temporal", checkpoint["experiment"], int(checkpoint["step"]),
                                  int(section["source_offset"]), tuple(section["noise_seeds"]),
                                  preferred_microbatch=preferred_microbatch(checkpoint["experiment"]),
                                  check_total_backward=bool(section.get("check_total_backward", False)),
                                  numerical_check=bool(section.get("numerical_check", False)),
                                  measurement_mode=str(section.get("measurement_mode", "full")),
                                  **common))
    if len({item.name for item in items}) != len(items):
        raise ValueError("Duplicate gradient diagnostic work items")
    if any(item.source_offset not in (0, 128, 256) for item in items):
        raise ValueError("Source offsets must be the three nonoverlapping 128-source windows")
    return items


def checkpoint_path(item: WorkItem) -> Path:
    return ROOT / "outputs/csgo_1b" / item.experiment / f"checkpoint-{item.checkpoint_step}"


def report_path(output_root: Path, item: WorkItem) -> Path:
    return output_root / item.stage / f"{item.experiment}.step{item.checkpoint_step}.offset{item.source_offset}.json"


def gpu_free_gib(device_index: int = 0) -> tuple[float, float]:
    process = subprocess.run(
        ["nvidia-smi", f"--id={device_index}", "--query-gpu=memory.free,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    line = process.stdout.strip().splitlines()[0]
    free_mib, total_mib = (float(part.strip()) for part in line.split(",", 1))
    return free_mib / 1024.0, total_mib / 1024.0


def choose_microbatch(free_gib: float, total_gib: float,
                      preferred_microbatch: int) -> tuple[int, float]:
    if free_gib < MIN_FREE_GIB:
        raise ValueError(f"GPU free {free_gib:.1f} GiB is below {MIN_FREE_GIB:.0f} GiB")
    if preferred_microbatch not in (1, 2, 4, 8):
        raise ValueError("preferred_microbatch must be 1, 2, 4, or 8")
    microbatch = (2 if free_gib >= MICRO2_FREE_GIB else 1) if preferred_microbatch == 1 else preferred_microbatch
    fraction = (free_gib - RESERVE_GIB) / total_gib
    if fraction <= 0 or fraction > 1:
        raise ValueError("Invalid GPU allocator fraction from nvidia-smi memory")
    return microbatch, fraction


def command_for(item: WorkItem, *, python: str, output: Path, device: str,
                microbatch: int, gpu_memory_fraction: float) -> list[str]:
    command = [
        python, "scripts/probe_multitask_gradients.py",
        "--checkpoint-dir", str(checkpoint_path(item)),
        "--experiment", item.experiment,
        "--checkpoint-step", str(item.checkpoint_step),
        "--stage", item.probe_stage,
        "--source-seed", str(item.source_seed),
        "--samples-per-map", str(item.samples_per_map),
        "--source-offset", str(item.source_offset),
        "--noise-seeds", *(str(seed) for seed in item.noise_seeds),
        "--microbatch-size", str(microbatch),
        "--accumulations", str(item.source_count // microbatch),
        "--coordinate", item.coordinate,
        "--device", device,
        "--gpu-memory-fraction", f"{gpu_memory_fraction:.6f}",
        "--output", str(output),
    ]
    command.append("--bf16" if item.bf16 else "--no-bf16")
    command.append("--gradient-checkpointing" if item.gradient_checkpointing else "--no-gradient-checkpointing")
    if item.precision_reference:
        command.append("--fp32-reference")
    if item.deterministic_probe:
        command.append("--deterministic-probe")
    if item.check_total_backward:
        command.append("--check-total-backward")
    if item.numerical_check:
        command.append("--numerical-check")
    command.extend(("--measurement-mode", item.measurement_mode))
    return command


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _shared_language_layer_errors(result: dict[str, Any]) -> list[str]:
    """Validate the complete shared-language partition for confirm/temporal reports."""
    stats = result.get("shared_language_layers")
    coordinate = result.get("coordinate")
    if not isinstance(stats, Mapping) or not isinstance(coordinate, Mapping):
        return ["shared layer payload"]
    errors = []
    if stats.get("status") != "ok" or stats.get("unassigned") is not None:
        errors.append("shared layer status/unassigned")
    layers = stats.get("layers")
    if not isinstance(layers, Mapping) or set(layers) != {str(i) for i in range(24)}:
        return [*errors, "shared layer count"]
    count = coordinate.get("parameter_count")
    numel = coordinate.get("numel")
    if (type(count) is not int or count <= 0 or type(numel) is not int or numel <= 0
            or stats.get("shared_parameter_count") != count
            or stats.get("shared_numel") != numel):
        errors.append("shared layer coordinate")
    layer_counts = [layer.get("parameter_count") if isinstance(layer, Mapping) else None
                    for layer in layers.values()]
    layer_numels = [layer.get("numel") if isinstance(layer, Mapping) else None
                    for layer in layers.values()]
    if (any(type(value) is not int or value <= 0 for value in (*layer_counts, *layer_numels))
            or sum(layer_counts) != count or sum(layer_numels) != numel):
        errors.append("shared layer coverage")
    checks = stats.get("checks")
    required = ("named_layer_dot_matches_global", "partition_dot_matches_global",
                "parameter_count_matches_global", "numel_matches_global")
    if (not isinstance(checks, Mapping)
            or any(checks.get(key) is not True for key in required)
            or checks.get("covered_parameter_count") != count
            or checks.get("covered_numel") != numel):
        errors.append("shared layer partition checks")
    dots = [layer.get("weighted_gen_loc_dot") if isinstance(layer, Mapping) else None
            for layer in layers.values()]
    global_dot = stats.get("global_shared_weighted_gen_loc_dot")
    if (not all(_finite_number(value) for value in dots)
            or not _finite_number(global_dot)):
        errors.append("shared layer dot")
    else:
        dot_sum = math.fsum(dots)
        scale = max(abs(global_dot), math.fsum(abs(value) for value in dots), 1e-12)
        tolerance = 1e-18 + 1e-10 * scale
        reported_sums = (checks.get("named_layer_dot_sum"),
                         checks.get("partition_dot_sum"),
                         checks.get("partition_dot_difference")) if isinstance(checks, Mapping) else ()
        if (abs(dot_sum - global_dot) > tolerance
                or len(reported_sums) != 3
                or any(not _finite_number(value) for value in reported_sums)
                or abs(reported_sums[0] - dot_sum) > tolerance
                or abs(reported_sums[1] - dot_sum) > tolerance
                or abs(reported_sums[2]) > tolerance):
            errors.append("shared layer dot")
    return errors


def _direct_reconstruction_errors(result: Mapping[str, Any]) -> list[str]:
    checks = result.get("first_micro_direct_checks")
    if not isinstance(checks, Mapping) or checks.get("scope") != "first_microbatch_direct_vjp":
        return ["first micro direct checks"]
    errors = []
    for name in ("main_vs_components", "total_vs_components"):
        check = checks.get(name)
        if not isinstance(check, Mapping) or check.get("passed") is not True:
            errors.append(f"first micro {name}")
            continue
        values = (check.get("difference_norm"), check.get("reference_norm"),
                  check.get("rtol"), check.get("atol"))
        if (not all(_finite_number(value) and value >= 0 for value in values)
                or values[0] > values[3] + values[2] * values[1]
                or values[2] > 2e-3 or values[3] > 1e-5):
            errors.append(f"first micro {name}")
    return errors


def report_checks(path: Path, item: WorkItem) -> tuple[bool, list[str]]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError) as exc:
        return False, [f"report unreadable: {exc}"]
    errors = []
    metadata = report.get("metadata", {})
    if report.get("measurement_mode", "full") != item.measurement_mode:
        errors.append("measurement mode")
    if metadata.get("measurement_mode", "full") != item.measurement_mode:
        errors.append("metadata measurement mode")
    if report.get("stage") != item.probe_stage:
        errors.append("stage")
    if Path(report.get("checkpoint_dir", "/missing")).resolve() != checkpoint_path(item).resolve():
        errors.append("checkpoint_dir")
    if metadata.get("experiment") != item.experiment or metadata.get("checkpoint_step") != item.checkpoint_step:
        errors.append("checkpoint identity")
    if report.get("source_seed") != item.source_seed or report.get("source_offset") != item.source_offset:
        errors.append("source window")
    if report.get("noise_seeds") != list(item.noise_seeds) or report.get("coordinate") != item.coordinate:
        errors.append("noise seeds or coordinate")
    if report.get("tasks") != ["gen", "loc", "perception", "aux_loc"]:
        errors.append("tasks")
    if len(report.get("source_indices", [])) != item.source_count:
        errors.append("source count")
    source_rows = report.get("source_rows")
    if (not isinstance(source_rows, list) or len(source_rows) != item.source_count
            or any(not isinstance(row, dict) or not row.get("map") or not row.get("file_frame")
                   for row in source_rows)):
        errors.append("source row identities")
    microbatch = report.get("microbatch_size")
    if (microbatch not in (1, 2, 4, 8)
            or item.source_count % microbatch
            or report.get("accumulations") != item.source_count // microbatch):
        errors.append("microbatch or accumulation count")
    if metadata.get("gradient_checkpointing") is not item.gradient_checkpointing:
        errors.append("gradient checkpointing")
    if metadata.get("dtype") != ("float32" if item.precision_reference else "bfloat16"):
        errors.append("dtype")
    if metadata.get("precision_reference") is not item.precision_reference:
        errors.append("precision reference")
    if metadata.get("deterministic_probe") is not item.deterministic_probe:
        errors.append("deterministic probe")
    if item.precision_reference:
        if set(metadata.get("parameter_dtype_numel", {})) != {"torch.float32"}:
            errors.append("parameter dtype histogram")
        adjustments = metadata.get("precision_reference_adjustments") or {}
        if (adjustments.get("cuda_matmul_allow_tf32") is not False
                or adjustments.get("cudnn_allow_tf32") is not False
                or adjustments.get("flash_sdp_enabled") is not False
                or adjustments.get("mem_efficient_sdp_enabled") is not False
                or adjustments.get("math_sdp_enabled") is not True):
            errors.append("FP32 attention backend")
    if item.deterministic_probe:
        if (metadata.get("deterministic_algorithms") is not True
                or metadata.get("flash_attention_deterministic_env") != "1"):
            errors.append("deterministic kernels")
    config_path = checkpoint_path(item).parent / "resolved_config.yaml"
    if config_path.is_file():
        expected_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if metadata.get("resolved_config_sha256") != expected_hash:
            errors.append("saved config hash")
    else:
        errors.append("saved config missing")
    results = report.get("results", {})
    for seed in item.noise_seeds:
        result = results.get(str(seed), {})
        if result.get("measurement_mode", "full") != item.measurement_mode:
            errors.append(f"seed {seed} measurement mode")
        if item.measurement_mode == "full":
            if not isinstance(result.get("linearity"), Mapping) or result["linearity"].get("passed") is not True:
                errors.append(f"seed {seed} linearity")
        else:
            if result.get("linearity", "missing") is not None:
                errors.append(f"seed {seed} linearity scope")
            errors.extend(f"seed {seed} {error}" for error in _direct_reconstruction_errors(result))
            scope = result.get("measurement_scope")
            if (not isinstance(scope, Mapping)
                    or scope.get("components") != "direct_vjp_every_microbatch"
                    or scope.get("main_total") != "component_sum_of_mean_weighted_vjps"
                    or scope.get("other") != "zero_by_verified_weights"
                    or scope.get("direct_reconstruction") != "first_microbatch_only"):
                errors.append(f"seed {seed} measurement scope")
            if result.get("total_gradient_coverage", "missing") is not None:
                errors.append(f"seed {seed} total coverage scope")
            residual = result.get("components", {}).get("linearity_residual", {})
            combined = result.get("combined", {})
            if (not isinstance(residual, Mapping) or residual.get("weighted_norm", "missing") is not None
                    or residual.get("signed_contribution_to_total", "missing") is not None
                    or not isinstance(combined, Mapping) or any(combined.get(key, "missing") is not None
                        for key in ("main_joint_vjp_norm", "main_joint_vs_component_sum_difference_norm",
                                    "actual_aux_increment_norm", "actual_aux_increment_vs_component_cosine",
                                    "actual_aux_increment_vs_component_difference_norm"))):
                errors.append(f"seed {seed} direct residual scope")
            groups = result.get("groups")
            if not isinstance(groups, Mapping) or any(
                not isinstance(group, Mapping)
                or group.get("linearity_difference_norm", "missing") is not None
                or group.get("linearity_relative_difference", "missing") is not None
                for group in groups.values()):
                errors.append(f"seed {seed} group residual scope")
        scalar = result.get("scalar_linearity", {})
        if (scalar.get("weights_constant") is not True
                or not _finite_number(scalar.get("max_abs_difference"))
                or scalar["max_abs_difference"] > 1e-5):
            errors.append(f"seed {seed} scalar linearity")
        if not _finite_number(result.get("total_gradient_norm")):
            errors.append(f"seed {seed} total norm")
        if result.get("other_objectives_active") is not False:
            errors.append(f"seed {seed} other objectives")
        aux = result.get("aux_freeze", {})
        if aux.get("status") == "checked" and aux.get("passed") is not True:
            errors.append(f"seed {seed} aux freeze")
        if item.check_total_backward and result.get("total_backward_check", {}).get("passed") is not True:
            errors.append(f"seed {seed} total backward")
        if item.numerical_check and result.get("repeated_total_vjp", {}).get("passed") is not True:
            errors.append(f"seed {seed} numerical check")
        if item.stage in ("confirm", "temporal"):
            errors.extend(f"seed {seed} {error}" for error in
                          _shared_language_layer_errors(result))
    return not errors, errors


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _archive_partial_files(output: Path, *, tag: str) -> list[str]:
    candidates = [output, output.with_suffix(output.suffix + ".jsonl"),
                  *output.parent.glob(output.stem + ".attempt*.log"),
                  *output.parent.glob(output.stem + ".attempt*.jsonl"),
                  *output.parent.glob(output.stem + ".attempt*.json")]
    archived = []
    for existing in candidates:
        if existing.is_file():
            destination = existing.with_name(existing.name + f".backup.{tag}")
            existing.replace(destination)
            archived.append(str(destination))
    return archived


def _run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return process.wait()


def _is_cuda_oom(log_path: Path) -> bool:
    with log_path.open("rb") as stream:
        stream.seek(max(0, stream.seek(0, 2) - 65536))
        tail = stream.read().decode("utf-8", errors="replace").lower()
    return "cuda out of memory" in tail or "torch.outofmemoryerror" in tail


def _wait_for_memory(device_index: int, max_wait_seconds: int) -> tuple[float, float]:
    start = time.monotonic()
    while True:
        free, total = gpu_free_gib(device_index)
        if free >= MIN_FREE_GIB:
            return free, total
        waited = time.monotonic() - start
        if waited >= max_wait_seconds:
            raise RuntimeError(f"GPU free memory stayed below {MIN_FREE_GIB:.0f} GiB for {waited:.0f}s")
        print(f"Waiting for GPU memory: {free:.1f} GiB free, need {MIN_FREE_GIB:.0f} GiB; elapsed {waited:.0f}s", flush=True)
        time.sleep(min(30, max_wait_seconds - waited))


def run_queue(config: dict[str, Any], items: list[WorkItem], *, python: str,
              max_wait_seconds: int, device_index: int = 0,
              config_hash: str, force: bool = False) -> dict[str, Any]:
    output_root = path_in_repo(config["output_root"])
    manifest_path = output_root / "manifest.json"
    code_hash = code_sha256()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {
        "schema_version": 1, "created_at": _now(), "items": {},
    }
    manifest["code_sha256"] = code_hash
    manifest["queue_config_sha256"] = config_hash
    for index, item in enumerate(items, 1):
        output = report_path(output_root, item)
        signature = run_signature(item, code_hash=code_hash, config_hash=config_hash)
        valid, errors = report_checks(output, item)
        previous = manifest["items"].get(item.name, {})
        if valid and previous.get("signature") == signature and not force:
            manifest["items"][item.name] = {
                "status": "complete", "signature": signature, "output": str(output),
                "checked_at": _now(), "checks": "passed", "reused": True,
                "code_sha256": code_hash, "queue_config_sha256": config_hash,
            }
            _write_manifest(manifest_path, manifest)
            print(f"[{index}/{len(items)}] skip validated {item.name}", flush=True)
            continue
        if output.is_file() and not force:
            raise RuntimeError(
                f"Existing result {output} lacks matching parameter/code digest or failed checks: {errors}; "
                "inspect it, then use --force to archive and rerun"
            )
        if previous:
            manifest.setdefault("history", []).append({
                "item": item.name, "archived_at": _now(), "record": previous,
            })
        backup_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f".{time.time_ns()}"
        archived = _archive_partial_files(output, tag=backup_tag)
        if archived:
            print(f"Archived {len(archived)} stale files for {item.name}", flush=True)
        if previous or archived:
            _write_manifest(manifest_path, manifest)
        checkpoint = checkpoint_path(item)
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(checkpoint / "model.safetensors")
        free, total = _wait_for_memory(device_index, max_wait_seconds)
        microbatch, fraction = choose_microbatch(free, total, item.preferred_microbatch)
        attempt = 0
        while True:
            attempt += 1
            log_path = output.with_suffix(f".attempt{attempt}.log")
            command = command_for(item, python=python, output=output,
                                  device=config["device"], microbatch=microbatch,
                                  gpu_memory_fraction=fraction)
            manifest["items"][item.name] = {
                "status": "running", "signature": signature, "output": str(output),
                "log": str(log_path), "command": command, "microbatch": microbatch,
                "gpu_free_gib": free, "gpu_total_gib": total,
                "gpu_memory_fraction": fraction, "attempt": attempt, "started_at": _now(),
                "code_sha256": code_hash, "queue_config_sha256": config_hash,
            }
            _write_manifest(manifest_path, manifest)
            print(f"[{index}/{len(items)}] run {item.name}: micro={microbatch}, free={free:.1f} GiB, cap={fraction:.3f}", flush=True)
            exit_code = _run_command(command, log_path)
            valid, errors = report_checks(output, item)
            if exit_code == 0 and valid:
                manifest["items"][item.name].update(status="complete", finished_at=_now(), checks="passed")
                _write_manifest(manifest_path, manifest)
                break
            is_oom = exit_code != 0 and _is_cuda_oom(log_path)
            manifest["items"][item.name].update(status="failed", finished_at=_now(),
                                                 exit_code=exit_code, checks=errors,
                                                 error="cuda_oom" if is_oom else "probe_failed")
            _write_manifest(manifest_path, manifest)
            if is_oom and microbatch > 1:
                next_microbatch = microbatch // 2
                print(f"OOM at microbatch {microbatch}; retrying {item.name} at microbatch {next_microbatch}", flush=True)
                jsonl = output.with_suffix(output.suffix + ".jsonl")
                if jsonl.is_file():
                    jsonl.replace(output.with_suffix(f".attempt{attempt}.jsonl"))
                if output.is_file():
                    output.replace(output.with_suffix(f".attempt{attempt}.json"))
                free, total = _wait_for_memory(device_index, max_wait_seconds)
                microbatch = next_microbatch
                fraction = (free - RESERVE_GIB) / total
                continue
            raise RuntimeError(f"Probe failed for {item.name}: exit={exit_code}, checks={errors}, log={log_path}")
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=STAGES, action="append", help="Only include selected stages")
    parser.add_argument("--execute", action="store_true", help="Run serial probes and write status")
    parser.add_argument("--force", action="store_true", help="Archive existing results and rerun")
    parser.add_argument("--python", help="Probe interpreter override")
    parser.add_argument("--max-wait-seconds", type=int, default=1800)
    parser.add_argument("--gpu-index", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    items = build_items(config, tuple(args.stage) if args.stage else STAGES)
    output_root = path_in_repo(config["output_root"])
    for index, item in enumerate(items, 1):
        print(f"[{index}/{len(items)}] {item.name} sources={item.source_count} seeds={list(item.noise_seeds)} output={report_path(output_root, item)}")
    if not args.execute:
        print("Plan only. Add --execute to launch serial probes.")
        return items
    if args.max_wait_seconds < 0:
        raise ValueError("--max-wait-seconds must be nonnegative")
    python = args.python or config["python"]
    if not Path(python).is_file():
        raise FileNotFoundError(python)
    return run_queue(config, items, python=python, max_wait_seconds=args.max_wait_seconds,
                     device_index=args.gpu_index,
                     config_hash=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                     force=args.force)


if __name__ == "__main__":
    main()
