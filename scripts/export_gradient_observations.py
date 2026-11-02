#!/usr/bin/env python3
"""Export the fixed Seen-3 gradient queue as portable, CPU-only evidence.

Only planned JSON reports are read. No checkpoint, GPU, controller, or training
module is needed. An incomplete --require-complete run never touches exports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

from render_gradient_evidence import PAIRS, TASKS, auxiliary_direction_row, path_get
from run_gradient_diagnostics import (DEFAULT_CONFIG, STAGES,
                                      _direct_reconstruction_errors,
                                      _shared_language_layer_errors, build_items,
                                      load_config, report_path)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs_diagnostics/gradient_seen3_fp32"
DEFAULT_OUTPUT = ROOT / "outputs_diagnostics/gradient_seen3_fp32_export"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def finite(value: object) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def check_report(report: dict, item: object) -> list[str]:
    """Validate identities, counts, coordinate schema, and measurement checks."""
    errors: list[str] = []
    meta = report.get("metadata")
    if not isinstance(meta, dict):
        return ["metadata missing"]
    if report.get("schema_version") != 1:
        errors.append("schema_version")
    if (meta.get("experiment"), meta.get("checkpoint_step")) != (item.experiment, item.checkpoint_step):
        errors.append("checkpoint identity")
    checkpoint = Path(report.get("checkpoint_dir") or "")
    if (checkpoint.name != f"checkpoint-{item.checkpoint_step}"
            or checkpoint.parent.name != item.experiment):
        errors.append("checkpoint path identity")
    if report.get("stage") != item.probe_stage:
        errors.append("probe stage")
    for field, expected in (("source_seed", item.source_seed), ("source_offset", item.source_offset),
                            ("coordinate", item.coordinate), ("noise_seeds", list(item.noise_seeds)),
                            ("measurement_mode", item.measurement_mode)):
        if report.get(field, "full" if field == "measurement_mode" else None) != expected:
            errors.append(field)
    if meta.get("measurement_mode", "full") != item.measurement_mode:
        errors.append("metadata measurement mode")
    if report.get("tasks") != list(TASKS):
        errors.append("tasks")
    if (meta.get("dtype") != "float32" or meta.get("precision_reference") is not True
            or meta.get("deterministic_probe") is not True
            or meta.get("gradient_checkpointing") is not item.gradient_checkpointing):
        errors.append("probe precision/determinism")
    if set(meta.get("parameter_dtype_numel") or {}) != {"torch.float32"}:
        errors.append("parameter dtype")
    rows = report.get("source_rows")
    if (not isinstance(rows, list) or len(rows) != item.source_count
            or not all(isinstance(row, dict) and all(isinstance(row.get(k), str) and row[k]
                                                       for k in ("map", "file_frame", "row_id"))
                       for row in rows)):
        errors.append("source row identities")
    if not isinstance(report.get("source_indices"), list) or len(report["source_indices"]) != item.source_count:
        errors.append("source indices count")
    micro = report.get("microbatch_size")
    if (type(micro) is not int or micro not in (1, 2, 4, 8)
            or item.source_count % micro != 0
            or report.get("accumulations") != item.source_count // micro):
        errors.append("microbatch/accumulations")
    results = report.get("results")
    if not isinstance(results, dict) or set(results) != {str(seed) for seed in item.noise_seeds}:
        return [*errors, "seed set"]
    coordinate_hashes = set()
    for seed in item.noise_seeds:
        result = results[str(seed)]
        prefix = f"seed {seed}: "
        if not isinstance(result, dict):
            errors.append(prefix + "result schema")
            continue
        coord = result.get("coordinate")
        if (not isinstance(coord, dict) or not isinstance(coord.get("names"), list)
                or not all(isinstance(name, str) for name in coord["names"])
                or len(set(coord["names"])) != len(coord["names"])
                or coord.get("parameter_count") != len(coord["names"])
                or type(coord.get("numel")) is not int or coord["numel"] <= 0):
            errors.append(prefix + "coordinate schema")
        else:
            coordinate_hashes.add(digest(coord))
        if (result.get("samples") != 2 * item.source_count
                or result.get("microbatches") != item.source_count // micro):
            errors.append(prefix + "task rows/microbatches")
        if result.get("measurement_mode", "full") != item.measurement_mode:
            errors.append(prefix + "measurement mode")
        if not all(isinstance(result.get(key), dict) for key in
                   ("mean_losses", "weights_seen", "components", "pairwise_weighted_cosine", "combined")):
            errors.append(prefix + "measurement payload")
        else:
            for task in TASKS:
                weight = path_get(result, f"weights_seen.{task}")
                if (not finite(path_get(result, f"mean_losses.{task}"))
                        or not finite(path_get(result, f"components.{task}.raw_norm"))
                        or not finite(path_get(result, f"components.{task}.weighted_norm"))
                        or not isinstance(weight, list) or len(weight) != 1 or not finite(weight[0])):
                    errors.append(prefix + task + " component/loss/weight")
            for task in ("gen", "loc"):
                for part in ("total", "aux_increment"):
                    if not finite(path_get(result, f"combined.primary_directions.{task}.raw_descent_first_order_{part}")):
                        errors.append(prefix + task + " first-order " + part)
        if not finite(result.get("total_gradient_norm")):
            errors.append(prefix + "total gradient norm")
        if result.get("other_objectives_active") is not False:
            errors.append(prefix + "other objectives")
        scalar = result.get("scalar_linearity") or {}
        if scalar.get("weights_constant") is not True or not finite(scalar.get("max_abs_difference")) or scalar["max_abs_difference"] > 1e-5:
            errors.append(prefix + "scalar linearity")
        if item.measurement_mode == "full":
            if (result.get("linearity") or {}).get("passed") is not True:
                errors.append(prefix + "full linearity")
        else:
            if result.get("linearity", "missing") is not None:
                errors.append(prefix + "derived linearity scope")
            errors.extend(prefix + error for error in _direct_reconstruction_errors(result))
            scope = result.get("measurement_scope") or {}
            if (scope.get("components") != "direct_vjp_every_microbatch"
                    or scope.get("main_total") != "component_sum_of_mean_weighted_vjps"
                    or scope.get("direct_reconstruction") != "first_microbatch_only"):
                errors.append(prefix + "measurement scope")
        for name in ("repeated_total_vjp", "total_backward_check"):
            if (result.get(name) or {}).get("passed") is not True:
                errors.append(prefix + name)
        aux = result.get("aux_freeze")
        aux_weight = path_get(result, "weights_seen.aux_loc")
        if not isinstance(aux, dict) or not isinstance(aux_weight, list) or len(aux_weight) != 1 or not finite(aux_weight[0]):
            errors.append(prefix + "aux freeze schema")
        elif aux_weight[0] > 0:
            details = aux.get("details")
            if (aux.get("status") != "checked" or aux.get("passed") is not True
                    or aux.get("checks") != result.get("microbatches")
                    or not isinstance(details, list) or len(details) != aux["checks"]
                    or not all(isinstance(detail, dict) and detail.get("supported") is True
                               and detail.get("passed") is True
                               and type(detail.get("checked_parameters")) is int
                               and detail["checked_parameters"] > 0
                               and type(detail.get("routed_parameters")) is int
                               and detail["routed_parameters"] == 0
                               for detail in details)):
                errors.append(prefix + "active aux freeze")
        elif (aux.get("status") != "skipped" or aux.get("checks") != 0
              or aux.get("passed") is not None or aux.get("details") != []):
            errors.append(prefix + "inactive aux freeze")
        if item.stage in ("confirm", "temporal"):
            errors.extend(prefix + error for error in _shared_language_layer_errors(result))
    if len(coordinate_hashes) != 1:
        errors.append("coordinate mismatch between seeds")
    return errors


def observation(report: dict, item: object, seed: int) -> dict:
    result = report["results"][str(seed)]
    row = {"stage": item.stage, "experiment": item.experiment,
           "checkpoint_step": item.checkpoint_step, "source_offset": item.source_offset,
           "noise_seed": seed, "source_count": item.source_count,
           "task_rows": result.get("samples"), "microbatch": report.get("microbatch_size"),
           "accumulations": report.get("accumulations"),
           "measurement_mode": item.measurement_mode,
           "elapsed_seconds": result.get("elapsed_seconds"),
           "peak_cuda_allocated_gib": (result["peak_cuda_memory_bytes"] / 1024 ** 3
                                        if finite(result.get("peak_cuda_memory_bytes")) else None)}
    for task in TASKS:
        for field, path in (("mean_loss", f"mean_losses.{task}"),
                            ("raw_norm", f"components.{task}.raw_norm"),
                            ("weighted_norm", f"components.{task}.weighted_norm")):
            row[f"{task}_{field}"] = path_get(result, path)
        weights = path_get(result, f"weights_seen.{task}")
        row[f"{task}_weight"] = weights[0] if isinstance(weights, list) and len(weights) == 1 else None
    for left, right in PAIRS:
        row[f"weighted_cos_{left}_{right}"] = path_get(result, f"pairwise_weighted_cosine.{left}.{right}")
    gen_norm = row["gen_weighted_norm"]
    loc_norm = row["loc_weighted_norm"]
    gen_loc_cos = row["weighted_cos_gen_loc"]
    if all(finite(x) and x >= 0 for x in (gen_norm, loc_norm)):
        gen_loc_dot = (0.0 if gen_norm == 0 or loc_norm == 0 else
                       gen_loc_cos * gen_norm * loc_norm if finite(gen_loc_cos) else None)
    else:
        gen_loc_dot = None
    row["gen_other_main_offset_pct"] = (
        (0.0 if gen_loc_dot == 0 else -100 * gen_loc_dot / (gen_norm * gen_norm))
        if gen_loc_dot is not None and gen_norm > 0 else None)
    row["loc_other_main_offset_pct"] = (
        (0.0 if gen_loc_dot == 0 else -100 * gen_loc_dot / (loc_norm * loc_norm))
        if gen_loc_dot is not None and loc_norm > 0 else None)
    ratio = path_get(result, "combined.aux_over_baseline")
    row["aux_over_main_pct"] = 100 * ratio if finite(ratio) else None
    for task in ("gen", "loc"):
        stem = f"combined.primary_directions.{task}.raw_descent_first_order_"
        total, aux = (path_get(result, stem + part) for part in ("total", "aux_increment"))
        main = total - aux if finite(total) and finite(aux) else None
        row.update({f"{task}_raw_total_first_order": total,
                    f"{task}_raw_aux_increment_first_order": aux,
                    f"{task}_raw_main_first_order": main,
                    f"{task}_aux_effect_vs_main_descent_pct": (
                        100 * aux / abs(main) if finite(main) and main < 0 else None)})
    for task, short in (("perception", "p"), ("aux_loc", "u")):
        direction = auxiliary_direction_row({"result": result, "seed": str(seed),
                                             "source_offset": item.source_offset},
                                            task, item.measurement_mode)
        row[f"{short}_direction_status"] = direction["status"]
        row[f"{short}_cos_to_total"] = direction["direct_cosine_to_total"]
        for source, target in (("other_aux_offset_vs_base_pct", "cross_aux_offset_pct"),
                               ("base_dot", "base_dot"), ("other_aux_dot", "other_dot"),
                               ("derived_total_dot", "derived_total_dot")):
            row[f"{short}_{target}"] = direction[source]
    layers = path_get(result, "shared_language_layers.layers")
    for task in ("gen", "loc"):
        field = f"weighted_{task}_total_dot"
        row[f"layer_{task}_total_dot_negative_count"] = (
            sum(finite(layer.get(field)) and layer[field] < 0 for layer in layers.values())
            if isinstance(layers, dict) else None)
    aux = result.get("aux_freeze") or {}
    row.update({"aux_freeze_status": aux.get("status"), "aux_freeze_checks": aux.get("checks"),
                "aux_freeze_passed": aux.get("passed"),
                "full_linearity_passed": (result.get("linearity") or {}).get("passed"),
                "first_micro_main_passed": path_get(result, "first_micro_direct_checks.main_vs_components.passed"),
                "first_micro_total_passed": path_get(result, "first_micro_direct_checks.total_vs_components.passed"),
                "repeated_vjp_passed": path_get(result, "repeated_total_vjp.passed"),
                "ordinary_backward_passed": path_get(result, "total_backward_check.passed")})
    return row


def write_atomic(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    config = load_config(DEFAULT_CONFIG)
    items = build_items(config)
    plan_names = {report_path(args.input_root, item).relative_to(args.input_root).as_posix() for item in items}
    queue_manifest = args.input_root / "manifest.json"
    queue_items = {}
    if queue_manifest.is_file():
        loaded = json.loads(queue_manifest.read_text(encoding="utf-8"))
        queue_items = loaded.get("items") or {}
    missing, invalid, records, rows = [], [], [], []
    for item in items:
        path = report_path(args.input_root, item)
        relative = path.relative_to(args.input_root).as_posix()
        if not path.is_file():
            missing.append(relative)
            continue
        raw = path.read_bytes()
        record = {"path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                  "experiment": item.experiment, "checkpoint_step": item.checkpoint_step,
                  "stage": item.stage, "source_offset": item.source_offset,
                  "code_sha256": (queue_items.get(item.name) or {}).get("code_sha256")}
        try:
            report = json.loads(raw)
            if not isinstance(report, dict):
                raise ValueError("report must be an object")
            errors = check_report(report, item)
            meta = report.get("metadata") or {}
            record["metadata_experiment"] = meta.get("experiment")
            record["metadata_checkpoint_step"] = meta.get("checkpoint_step")
            source_rows = report.get("source_rows")
            if isinstance(source_rows, list) and all(isinstance(r, dict) for r in source_rows):
                record["source_ids_sha256"] = digest([{k: r.get(k) for k in ("map", "file_frame", "row_id")}
                                                        for r in source_rows])
            coords = [result.get("coordinate") for result in (report.get("results") or {}).values()
                      if isinstance(result, dict)]
            record["coordinate_sha256"] = digest(coords[0]) if coords and isinstance(coords[0], dict) else None
            record["coordinate_schema"] = ({"parameter_count": coords[0].get("parameter_count"),
                                             "numel": coords[0].get("numel")}
                                            if coords and isinstance(coords[0], dict) else None)
            record["microbatch"] = report.get("microbatch_size")
            record["accumulations"] = report.get("accumulations")
            record["actual_seed_rows"] = len(report.get("results") or {})
            if not errors:
                rows.extend(observation(report, item, seed) for seed in item.noise_seeds)
        except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
            errors = [f"unreadable or unexportable: {exc}"]
        record["valid"] = not errors
        record["errors"] = errors
        records.append(record)
        if errors:
            invalid.append({"path": relative, "errors": errors})
    # Paired comparisons require the same ordered stable sources within each
    # source-count/window. The fixed queue also uses one shared parameter set.
    source_hashes: dict[tuple, str] = {}
    coordinate_hashes: dict[str, str] = {}
    for record in records:
        if not record["valid"]:
            continue
        checks = ((source_hashes, (96 if record["stage"] == "screen" else 128,
                                   record["source_offset"]),
                   record.get("source_ids_sha256"), "source window mismatch"),
                  (coordinate_hashes, "fixed_shared_coordinate", record.get("coordinate_sha256"),
                   "queue coordinate mismatch"))
        for seen, key, value, reason in checks:
            if key in seen and seen[key] != value:
                record["errors"].append(reason)
            else:
                seen[key] = value
        if record["errors"]:
            record["valid"] = False
            invalid.append({"path": record["path"], "errors": record["errors"]})
    if invalid:
        invalid_keys = {(entry["path"].split("/", 1)[0],
                         Path(entry["path"]).stem) for entry in invalid}
        rows = [row for row in rows if
                (row["stage"], f"{row['experiment']}.step{row['checkpoint_step']}.offset{row['source_offset']}")
                not in invalid_keys]
    expected_rows = sum(len(item.noise_seeds) for item in items)
    complete = not missing and not invalid and len(rows) == expected_rows
    manifest = {"schema_version": 1,
                "plan_config_path": DEFAULT_CONFIG.relative_to(ROOT).as_posix(),
                "plan_config_sha256": hashlib.sha256(DEFAULT_CONFIG.read_bytes()).hexdigest(),
                "input_root": str(args.input_root),
                "expected_reports": len(items), "actual_reports": len(records),
                "expected_seed_rows": expected_rows,
                "actual_seed_rows": sum(record.get("actual_seed_rows", 0) for record in records),
                "exported_valid_seed_rows": len(rows),
                "missing": missing, "invalid": invalid, "complete": complete,
                "reports": records,
                "unplanned_json_excluded": sorted(
                    p.relative_to(args.input_root).as_posix()
                    for stage in STAGES for p in (args.input_root / stage).glob("*.json")
                    if p.relative_to(args.input_root).as_posix() not in plan_names),
                "field_notes": {
                    "source_ids_sha256": "SHA256 of ordered map/file_frame/row_id objects; dataset_index excluded.",
                    "coordinate_sha256": "SHA256 of result.coordinate, including names, parameter_count, numel; both seed coordinates are checked equal.",
                    "empty_csv_cells": "Unmeasured, inactive, or mathematically undefined values remain empty; direction_status and aux_freeze_status distinguish cases.",
                    "aux_effect_vs_main_descent_pct": "100 * auxiliary increment / abs(main); main = total - auxiliary increment; defined only if main < 0. Positive resists main descent.",
                    "gen_other_main_offset_pct": "For G=weighted gen and L=weighted loc, dot=cos(G,L)*||G||||L||; value=-100*dot/||G||².",
                    "loc_other_main_offset_pct": "Using the same dot, value=-100*dot/||L||². A zero vector makes dot=0 even though cosine is undefined; only that vector's own zero denominator is empty. Missing inputs remain empty. Positive means the other main task resists this task's self-gradient descent; negative helps. Same-checkpoint local algebra, not an independent single-task training comparison.",
                    "cross_aux_offset_pct": "-100 * other auxiliary dot / base dot, defined only if base dot > 0. Positive resists this auxiliary's original descent.",
                    "interpretation": "Both percentages are local algebra, not retraining counterfactuals.",
                    "peak_cuda_allocated_gib": "Allocated peak during the measure phase only, not reserved memory or full process peak.",
                    "elapsed_seconds": "Probe measurement duration; excludes complete model loading and prechecks."}}
    if args.require_complete and not complete:
        print(f"incomplete: {len(records)}/{len(items)} reports, {len(rows)}/{expected_rows} valid seed rows; "
              f"missing={len(missing)}, invalid={len(invalid)}; exports untouched")
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        import io
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        csv_payload = buffer.getvalue()
    else:
        csv_payload = ""
    write_atomic(args.output_dir / "observations.csv", csv_payload)
    write_atomic(args.output_dir / "evidence_manifest.json", json.dumps(manifest, indent=2,
                                                                           ensure_ascii=False,
                                                                           allow_nan=False) + "\n")
    print(f"exported {len(rows)} seed rows from {len(records)}/{len(items)} reports; complete={complete}; "
          f"missing={len(missing)}, invalid={len(invalid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
