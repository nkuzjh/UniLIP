#!/usr/bin/env python3
"""Plan or execute the four Seen-3 temporal supplements at fixed microbatch 2.

The original controller supplies the WorkItem, command, and report checks. This
driver adds paired-window identity, offload, and per-microbatch freeze checks.
Planning is read-only; --execute alone starts serial probes or writes a manifest.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_gradient_diagnostics as original


ROOT = original.ROOT
DEFAULT_CONFIG = ROOT / "csgo_configs/gradient_diagnostics_seen3_temporal_extended.yaml"
EXPECTED_CODE_SHA256 = "0897dab849a6ab19408d51e11800412642d0b6fa64afaa6a42566c5c8b11bc48"
EXPECTED_WRAPPER_SHA256 = "bb91e5abc7a37dcb575cc15a64b8cdee6aaba31f784ecb47da1315d71d88ef14"
EXPERIMENTS = ("exp32_3maps_B1_S3_J8", "exp32_3maps_B2_S3_J8")
OFFSETS = (128, 256)
MICROBATCH = 2
ACCUMULATIONS = 64
ALLOWED_METADATA_DIFFERENCES = ("checkpoint", "checkpoint_step", "local_step",
                                "finetune_init_source", "saved_tensor_offload")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _queue_lock(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".queue.lock"
    with lock_path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Supplement queue is already running: {lock_path}") from exc
        yield


def load_plan(config_path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], list[original.WorkItem]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected supplement schema_version=1")
    if (tuple(config.get("experiments", ())) != EXPERIMENTS
            or tuple(config.get("source_offsets", ())) != OFFSETS
            or config.get("checkpoint_step") != 1200
            or config.get("microbatch") != MICROBATCH):
        raise ValueError("Supplement plan must be B1/B2 step1200 offsets128/256 at microbatch2")
    if config.get("output_root") != "outputs_diagnostics/gradient_seen3_temporal_extended":
        raise ValueError("Supplement output_root must be isolated")
    if config.get("anchor_root") != "outputs_diagnostics/gradient_seen3_fp32":
        raise ValueError("Supplement must use the original Seen-3 reports")
    if config.get("wrapper") != "scripts/probe_multitask_gradients_offload.py":
        raise ValueError("Supplement requires the reviewed CPU offload wrapper")
    if config.get("offload_acceptance") != "outputs_diagnostics/grad_20261102_temporal_extended/offload_acceptance.json":
        raise ValueError("Supplement requires the reviewed offload acceptance artifact")
    if not (math.isfinite(float(config["min_free_gib"])) and float(config["min_free_gib"]) > 6):
        raise ValueError("min_free_gib must exceed 6 GiB")
    if not (math.isfinite(float(config["reserve_gib"])) and float(config["reserve_gib"]) >= 6):
        raise ValueError("reserve_gib must be at least 6 GiB")
    base = original.load_config(original.path_in_repo(config["base_config"]))
    if (base["source_seed"] != 20261102 or base["samples_per_map"] != 128
            or base["coordinate"] != "shared" or base["precision_reference"] is not True
            or base["deterministic_probe"] is not True or base["device"] != config["device"]):
        raise ValueError("Base source or FP32 deterministic settings changed")
    confirm = {item.experiment: item for item in original.build_items(base, ("confirm",))
               if item.experiment in EXPERIMENTS and item.source_offset == 0}
    if set(confirm) != set(EXPERIMENTS):
        raise ValueError("Base config lacks B1/B2 confirm work items")
    items = []
    for experiment in EXPERIMENTS:
        for offset in OFFSETS:
            item = replace(confirm[experiment], stage="temporal", checkpoint_step=1200,
                           source_offset=offset, preferred_microbatch=MICROBATCH)
            if (item.noise_seeds != (17, 29) or item.measurement_mode != "component_sum"
                    or not item.check_total_backward or not item.numerical_check):
                raise ValueError(f"Base confirm measurement semantics changed for {experiment}")
            items.append(item)
    return config, items


def anchor_item(item: original.WorkItem, *, temporal_zero: bool = False) -> original.WorkItem:
    if temporal_zero:
        return replace(item, source_offset=0)
    return replace(item, stage="confirm", checkpoint_step=1800, source_offset=item.source_offset)


def _read_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _strict_aux(report: dict[str, Any], item: original.WorkItem) -> list[str]:
    errors = []
    for seed in item.noise_seeds:
        result = report.get("results", {}).get(str(seed), {})
        aux = result.get("aux_freeze", {})
        details = aux.get("details")
        if (aux.get("status") != "checked" or aux.get("passed") is not True
                or aux.get("checks") != ACCUMULATIONS or not isinstance(details, list)
                or len(details) != ACCUMULATIONS):
            errors.append(f"seed {seed} aux freeze count/status")
            continue
        if any(not isinstance(detail, dict) or detail.get("supported") is not True
               or detail.get("passed") is not True
               or type(detail.get("checked_parameters")) is not int
               or detail["checked_parameters"] <= 0
               or detail.get("routed_parameters") != 0 for detail in details):
            errors.append(f"seed {seed} aux freeze routed parameter")
        # Each source is measured once for generation and once for localization.
        if result.get("microbatches") != ACCUMULATIONS or result.get("samples") != 256:
            errors.append(f"seed {seed} sample/microbatch count")
    return errors


def _source_identity(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report.get("metadata", {})
    return {key: report.get(key) for key in ("source_selection", "source_indices", "source_rows") } | {
        key: metadata.get(key) for key in ("dataset_source", "dataset_source_rows_by_map")}


def _paired_checks(report: dict[str, Any], anchor: dict[str, Any], item: original.WorkItem,
                   *, wrapper_hash: str | None) -> list[str]:
    errors = []
    if report.get("microbatch_size") != MICROBATCH or report.get("accumulations") != ACCUMULATIONS:
        errors.append("fixed microbatch 2 x 64")
    if _source_identity(report) != _source_identity(anchor):
        errors.append("source identity differs from step1800 confirm")
    metadata, anchor_metadata = report.get("metadata", {}), anchor.get("metadata", {})
    unexpected = {key for key in metadata.keys() | anchor_metadata.keys()
                  if key not in ALLOWED_METADATA_DIFFERENCES
                  and metadata.get(key) != anchor_metadata.get(key)}
    if unexpected:
        errors.append(f"unexpected anchor metadata differences: {sorted(unexpected)}")
    for key in ("dtype", "precision_reference", "deterministic_probe", "gradient_checkpointing",
                "loss_weights", "probe_batch_float_dtype", "parameter_dtype_numel",
                "dataset_indices_by_map"):
        if metadata.get(key) != anchor_metadata.get(key):
            errors.append(f"anchor metadata {key}")
    if report.get("coordinate_partition") != anchor.get("coordinate_partition"):
        errors.append("shared coordinate partition")
    if report.get("tasks") != anchor.get("tasks"):
        errors.append("task order")
    for seed in item.noise_seeds:
        actual = report.get("results", {}).get(str(seed), {})
        expected = anchor.get("results", {}).get(str(seed), {})
        if actual.get("weights_seen") != expected.get("weights_seen"):
            errors.append(f"seed {seed} weights")
    offload = metadata.get("saved_tensor_offload")
    if not isinstance(offload, dict) or offload.get("enabled") is not True:
        errors.append("CPU saved tensor offload metadata")
    elif (offload.get("entrypoint") != "scripts/probe_multitask_gradients_offload.py"
          or offload.get("original_probe_sha256") != _sha256(ROOT / "scripts/probe_multitask_gradients.py")
          or (wrapper_hash is not None and offload.get("entrypoint_sha256") != wrapper_hash)):
        errors.append("CPU saved tensor offload identity")
    return errors


def report_checks(path: Path, item: original.WorkItem, anchor: dict[str, Any],
                  *, wrapper_hash: str | None) -> tuple[bool, list[str]]:
    valid, errors = original.report_checks(path, item)
    if not valid:
        return False, errors
    report = _read_report(path)
    errors += _strict_aux(report, item)
    errors += _paired_checks(report, anchor, item, wrapper_hash=wrapper_hash)
    return not errors, errors


def validate_anchors(config: dict[str, Any], items: list[original.WorkItem]) -> dict[str, dict[str, Any]]:
    anchor_root = original.path_in_repo(config["anchor_root"])
    old_manifest_path = anchor_root / "manifest.json"
    old_manifest = _read_report(old_manifest_path)
    base_config_path = original.path_in_repo(config["base_config"])
    base_config_hash = _sha256(base_config_path)
    if (old_manifest.get("code_sha256") != EXPECTED_CODE_SHA256
            or old_manifest.get("queue_config_sha256") != base_config_hash):
        raise RuntimeError("Original report manifest code/config hash differs")
    original_items = {item.name: item for item in original.build_items(
        original.load_config(base_config_path), ("confirm", "temporal"))}

    def checked_manifest(old_item: original.WorkItem, path: Path) -> None:
        recorded = old_manifest.get("items", {}).get(old_item.name, {})
        expected_item = original_items.get(old_item.name)
        expected_signature = (original.run_signature(expected_item, code_hash=EXPECTED_CODE_SHA256,
                              config_hash=base_config_hash) if expected_item else None)
        if (recorded.get("status") != "complete" or recorded.get("signature") != expected_signature
                or Path(recorded.get("output", "/missing")).resolve() != path.resolve()):
            raise RuntimeError(f"Original report manifest lacks a matching complete signature: {path}")

    anchors = {}
    for item in items:
        old_item = anchor_item(item)
        path = original.report_path(anchor_root, old_item)
        checked_manifest(old_item, path)
        valid, errors = original.report_checks(path, old_item)
        if not valid:
            raise RuntimeError(f"Invalid step1800 anchor {path}: {errors}")
        anchor = _read_report(path)
        if anchor.get("microbatch_size") != MICROBATCH or anchor.get("accumulations") != ACCUMULATIONS:
            raise RuntimeError(f"Step1800 anchor is not microbatch 2 x 64: {path}")
        if _strict_aux(anchor, old_item):
            raise RuntimeError(f"Step1800 anchor aux freeze failed: {path}")
        source = anchor["metadata"]["dataset_source"]
        for name, hash_key in (("benchmark_v2_manifest", "benchmark_v2_manifest_sha256"),
                               ("benchmark_v2_asset_manifest", "benchmark_v2_asset_manifest_sha256")):
            if _sha256(Path(source[name])) != source[hash_key]:
                raise RuntimeError(f"Anchor source manifest has changed: {source[name]}")
        zero_item = anchor_item(item, temporal_zero=True)
        zero_path = original.report_path(anchor_root, zero_item)
        checked_manifest(zero_item, zero_path)
        valid, errors = original.report_checks(zero_path, zero_item)
        if not valid:
            raise RuntimeError(f"Invalid step1200 offset0 anchor {zero_path}: {errors}")
        zero = _read_report(zero_path)
        if zero.get("microbatch_size") != MICROBATCH or zero.get("accumulations") != ACCUMULATIONS:
            raise RuntimeError(f"Step1200 offset0 anchor is not microbatch 2 x 64: {zero_path}")
        confirm_zero_item = anchor_item(zero_item)
        confirm_zero_path = original.report_path(anchor_root, confirm_zero_item)
        checked_manifest(confirm_zero_item, confirm_zero_path)
        valid, errors = original.report_checks(confirm_zero_path, confirm_zero_item)
        if not valid:
            raise RuntimeError(f"Invalid step1800 offset0 anchor {confirm_zero_path}: {errors}")
        if _source_identity(zero) != _source_identity(_read_report(confirm_zero_path)):
            raise RuntimeError(f"Step1200/1800 offset0 source identities differ: {zero_path}")
        for key in ("dataset_source", "dataset_source_rows_by_map", "loss_weights", "dtype"):
            if zero["metadata"].get(key) != anchor["metadata"].get(key):
                raise RuntimeError(f"Step1200 offset0 anchor differs in {key}: {zero_path}")
        anchors[item.name] = anchor
    return anchors


def input_hashes(config_path: Path, config: dict[str, Any], items: list[original.WorkItem]) -> dict[str, str]:
    paths = [config_path, original.path_in_repo(config["base_config"]),
             ROOT / "scripts/run_gradient_temporal_supplement.py",
             original.path_in_repo(config["wrapper"]),
             original.path_in_repo(config["anchor_root"]) / "manifest.json",
             original.path_in_repo(config["offload_acceptance"])]
    anchor_root = original.path_in_repo(config["anchor_root"])
    paths += [original.report_path(anchor_root, anchor_item(item)) for item in items]
    paths += [original.report_path(anchor_root, anchor_item(item, temporal_zero=True)) for item in items]
    paths += [original.report_path(anchor_root, anchor_item(anchor_item(item, temporal_zero=True)))
              for item in items]
    paths += [original.checkpoint_path(item) / "model.safetensors" for item in items]
    paths += [original.checkpoint_path(item).parent / "resolved_config.yaml" for item in items]
    for item in items:
        anchor = _read_report(original.report_path(anchor_root, anchor_item(item)))
        source = anchor["metadata"]["dataset_source"]
        paths.extend((Path(source["benchmark_v2_manifest"]),
                      Path(source["benchmark_v2_asset_manifest"])))
    return {str(path.resolve()): _sha256(path) for path in dict.fromkeys(paths)}


def _assert_frozen(expected: dict[str, str]) -> None:
    for name, digest in expected.items():
        if _sha256(Path(name)) != digest:
            raise RuntimeError(f"Input changed during supplement queue: {name}")
    if original.code_sha256() != EXPECTED_CODE_SHA256:
        raise RuntimeError("Original probe/controller code hash changed")


def _signature(item: original.WorkItem, hashes: dict[str, str], python: str) -> str:
    body = {"item": item.signature(), "input_hashes": hashes, "python": str(Path(python).resolve()),
            "original_code_sha256": EXPECTED_CODE_SHA256}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def validate_offload_acceptance(config: dict[str, Any]) -> dict[str, Any]:
    path = original.path_in_repo(config["offload_acceptance"])
    acceptance = _read_report(path)
    if (acceptance.get("accepted") is not True
            or acceptance.get("wrapper_sha256") != EXPECTED_WRAPPER_SHA256
            or acceptance.get("original_code_sha256") != EXPECTED_CODE_SHA256):
        raise RuntimeError(f"Offload GPU equivalence acceptance is missing or mismatched: {path}")
    return acceptance


def _wait_for_memory(device_index: int, min_free_gib: float, max_wait_seconds: int) -> tuple[float, float]:
    started = time.monotonic()
    while True:
        free, total = original.gpu_free_gib(device_index)
        if free >= min_free_gib:
            return free, total
        elapsed = time.monotonic() - started
        if elapsed >= max_wait_seconds:
            raise RuntimeError(f"GPU memory below {min_free_gib:.1f} GiB for {elapsed:.0f}s")
        print(f"Waiting for GPU memory: {free:.1f} GiB free; need {min_free_gib:.1f} GiB", flush=True)
        time.sleep(min(30, max_wait_seconds - elapsed))


def _archive_attempt(output: Path, attempt: int) -> list[str]:
    archived = []
    for path, suffix in ((output, ".json"), (output.with_suffix(output.suffix + ".jsonl"), ".jsonl")):
        if path.exists():
            destination = output.with_suffix(f".attempt{attempt}{suffix}")
            if destination.exists():
                raise RuntimeError(f"Refusing to replace previous attempt: {destination}")
            path.replace(destination)
            archived.append(str(destination))
    return archived


def run_queue(config_path: Path, config: dict[str, Any], items: list[original.WorkItem],
              *, python: str, max_wait_seconds: int, device_index: int) -> dict[str, Any]:
    if original.code_sha256() != EXPECTED_CODE_SHA256:
        raise RuntimeError("Original probe/controller code hash changed")
    if not Path(python).is_file():
        raise FileNotFoundError(python)
    if not original.path_in_repo(config["wrapper"]).is_file():
        raise FileNotFoundError(config["wrapper"])
    acceptance = validate_offload_acceptance(config)
    anchors = validate_anchors(config, items)
    hashes = input_hashes(config_path, config, items)
    wrapper_hash = hashes[str(original.path_in_repo(config["wrapper"]).resolve())]
    if wrapper_hash != EXPECTED_WRAPPER_SHA256:
        raise RuntimeError("CPU offload wrapper differs from the reviewed hash")
    output_root = original.path_in_repo(config["output_root"])
    manifest_path = output_root / "manifest.json"
    manifest = _read_report(manifest_path) if manifest_path.exists() else {
        "schema_version": 1, "created_at": _now(), "items": {}}
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("items"), dict):
        raise RuntimeError("Invalid supplement manifest")
    if manifest.get("input_hashes", hashes) != hashes:
        raise RuntimeError("Supplement manifest input signatures differ; refusing to overwrite")
    manifest.update(input_hashes=hashes, original_code_sha256=EXPECTED_CODE_SHA256,
                    wrapper_sha256=wrapper_hash,
                    offload_acceptance=acceptance,
                    allowed_anchor_metadata_differences=list(ALLOWED_METADATA_DIFFERENCES))
    for index, item in enumerate(items, 1):
        _assert_frozen(hashes)
        output = original.report_path(output_root, item)
        signature = _signature(item, hashes, python)
        previous = manifest["items"].get(item.name)
        if previous and previous.get("signature") != signature:
            raise RuntimeError(f"Existing manifest signature differs for {item.name}")
        if output.exists():
            if not previous or previous.get("status") != "complete":
                raise RuntimeError(f"Unsigned or incomplete result exists: {output}")
            valid, errors = report_checks(output, item, anchors[item.name], wrapper_hash=wrapper_hash)
            if not valid:
                raise RuntimeError(f"Existing report failed checks: {output}: {errors}")
            previous.update(reused=True, checked_at=_now())  # Preserve original command and duration.
            _write_json(manifest_path, manifest)
            print(f"[{index}/{len(items)}] reuse {item.name}", flush=True)
            continue
        if previous and previous.get("status") == "complete":
            raise RuntimeError(f"Completed result is missing: {output}")
        if previous and previous.get("status") == "running":
            raise RuntimeError(f"Previous attempt may still run: {item.name}; inspect manifest and process")
        if previous and previous.get("status") == "failed":
            raise RuntimeError(f"Previous failed attempt needs manual review: {item.name}")
        if output.with_suffix(output.suffix + ".jsonl").exists():
            raise RuntimeError(f"Unsigned partial result exists: {output}.jsonl")
        checkpoint = original.checkpoint_path(item)
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(checkpoint / "model.safetensors")
        attempts = []
        for attempt in (1, 2):
            free, total = _wait_for_memory(device_index, float(config["min_free_gib"]), max_wait_seconds)
            fraction = (free - float(config["reserve_gib"])) / total
            if not 0 < fraction <= 1:
                raise RuntimeError(f"Invalid allocator fraction {fraction}")
            command = original.command_for(item, python=python, output=output,
                                           device=config["device"], microbatch=MICROBATCH,
                                           gpu_memory_fraction=fraction)
            command[1] = config["wrapper"]
            log_path = output.with_suffix(f".attempt{attempt}.log")
            if log_path.exists():
                raise RuntimeError(f"Attempt log already exists: {log_path}")
            entry = {"attempt": attempt, "command": command, "log": str(log_path),
                     "microbatch": MICROBATCH, "accumulations": ACCUMULATIONS,
                     "gpu_free_gib": free, "gpu_total_gib": total,
                     "gpu_memory_fraction": fraction, "started_at": _now()}
            attempts.append(entry)
            manifest["items"][item.name] = {"status": "running", "signature": signature,
                "output": str(output), "attempts": attempts, "started_at": attempts[0]["started_at"]}
            _write_json(manifest_path, manifest)
            print(f"[{index}/{len(items)}] {item.name} attempt {attempt} micro=2x64 cap={fraction:.3f}", flush=True)
            exit_code = original._run_command(command, log_path)
            valid, errors = report_checks(output, item, anchors[item.name], wrapper_hash=wrapper_hash)
            entry.update(finished_at=_now(), exit_code=exit_code, checks=errors or "passed")
            if exit_code == 0 and valid:
                completed_report = _read_report(output)
                completed_metadata = completed_report["metadata"]
                anchor_metadata = anchors[item.name]["metadata"]
                manifest["items"][item.name].update(status="complete", finished_at=entry["finished_at"],
                                                    exit_code=0, checks="passed",
                                                    saved_tensor_offload=completed_metadata["saved_tensor_offload"],
                                                    anchor_metadata_differences=sorted(
                                                        key for key in completed_metadata.keys() | anchor_metadata.keys()
                                                        if completed_metadata.get(key) != anchor_metadata.get(key)),
                                                    duration_seconds=sum(
                                                        (datetime.fromisoformat(a["finished_at"]) -
                                                         datetime.fromisoformat(a["started_at"])).total_seconds()
                                                        for a in attempts))
                _write_json(manifest_path, manifest)
                break
            oom = exit_code != 0 and original._is_cuda_oom(log_path)
            entry.update(error="cuda_oom" if oom else "probe_failed",
                         archived=_archive_attempt(output, attempt))
            manifest["items"][item.name].update(status="failed", finished_at=entry["finished_at"],
                                                exit_code=exit_code, checks=errors)
            _write_json(manifest_path, manifest)
            if oom and attempt == 1:
                _assert_frozen(hashes)
                print(f"OOM for {item.name}; retrying once at unchanged microbatch 2", flush=True)
                continue
            raise RuntimeError(f"Probe failed for {item.name}: exit={exit_code}, checks={errors}, log={log_path}")
    if any(manifest["items"].get(item.name, {}).get("status") != "complete" for item in items):
        raise RuntimeError("Supplement queue incomplete")
    return manifest


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--python", help="Probe interpreter override")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--max-wait-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    config, items = load_plan(args.config)
    for index, item in enumerate(items, 1):
        print(f"[{index}/{len(items)}] {item.name} -> "
              f"{original.report_path(original.path_in_repo(config['output_root']), item)}")
    if not args.execute:
        print("Plan only. Add --execute to launch serial probes.")
        return items
    if args.max_wait_seconds < 0:
        raise ValueError("--max-wait-seconds must be nonnegative")
    with _queue_lock(original.path_in_repo(config["output_root"])):
        return run_queue(args.config, config, items, python=args.python or config["python"],
                         max_wait_seconds=args.max_wait_seconds, device_index=args.gpu_index)


if __name__ == "__main__":
    main()
