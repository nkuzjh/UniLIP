#!/usr/bin/env python3
"""Render compact, stage-separated evidence tables from gradient probe reports.

This is a CPU-only JSON reader. It does not load checkpoints or recompute gradients.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

STAGES = ("screen", "confirm", "temporal")
TASKS = ("gen", "loc", "perception", "aux_loc")
PAIRS = (("gen", "loc"), ("perception", "gen"), ("perception", "loc"),
         ("aux_loc", "gen"), ("aux_loc", "loc"), ("perception", "aux_loc"))
DERIVED_MODE_PATHS = (
    "linearity.relative_difference",
    "components.linearity_residual.weighted_norm",
    "combined.actual_aux_increment_norm",
    "combined.actual_aux_increment_vs_component_cosine",
    "combined.actual_aux_increment_vs_component_difference_norm",
)


def path_get(obj, path):
    for part in path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


def exists(obj, path):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def numeric(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def stats(values):
    xs = sorted(v for v in values if numeric(v))
    if not xs:
        return None
    n = len(xs)
    median = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    return {"median": median, "min": xs[0], "max": xs[-1], "count": n}


def summarize_path(rows, path):
    vals = [path_get(r["result"], path) for r in rows]
    stat = stats(vals)
    if stat is not None:
        return {"status": "available", **stat}
    present = any(exists(r["result"], path) for r in rows)
    return {"status": "inactive" if present else "pending", "median": None,
            "min": None, "max": None, "count": 0}


def derived_stat(values, undefined_count=0, pending_count=0):
    stat = stats(values)
    if stat:
        return {"status": "available", **stat, "undefined_count": undefined_count,
                "pending_count": pending_count}
    status = "pending" if pending_count else "undefined" if undefined_count else "pending"
    return {"status": status, "median": None, "min": None, "max": None, "count": 0,
            "undefined_count": undefined_count, "pending_count": pending_count}


def weighted_dot(result, left, right):
    """Dot of two weighted components, with zero components handled exactly."""
    a = path_get(result, f"components.{left}.weighted_norm")
    b = path_get(result, f"components.{right}.weighted_norm")
    if not numeric(a) or not numeric(b) or a < 0 or b < 0:
        return None
    if a == 0 or b == 0:
        return 0.0
    cosine = path_get(result, f"pairwise_weighted_cosine.{left}.{right}")
    return a * b * cosine if numeric(cosine) else None


def auxiliary_direction_row(row, task, mode):
    """Local algebra for P or U; no gradients or counterfactuals are recomputed."""
    result = row["result"]
    other = "aux_loc" if task == "perception" else "perception"
    item = {"seed": row["seed"], "source_offset": row["source_offset"],
            "status": "pending", "base_dot": None, "other_aux_dot": None,
            "derived_total_dot": None, "direct_cosine_to_total": None,
            "direct_total_dot": None, "direct_total_status": "pending",
            "direct_minus_derived_total_dot": None,
            "other_aux_offset_vs_base_pct": None, "base_positive_to_total_negative": None}
    norm = path_get(result, f"components.{task}.weighted_norm")
    if not numeric(norm) or norm < 0:
        return item
    if norm == 0:
        item["status"] = "inactive"
        item["direct_total_status"] = "inactive"
        return item
    base_parts = [norm * norm, weighted_dot(result, task, "gen"),
                  weighted_dot(result, task, "loc")]
    other_dot = weighted_dot(result, task, other)
    if all(numeric(x) for x in base_parts):
        item["base_dot"] = sum(base_parts)
    if numeric(other_dot):
        item["other_aux_dot"] = other_dot
    if numeric(item["base_dot"]) and numeric(other_dot):
        item["derived_total_dot"] = item["base_dot"] + other_dot
        item["base_positive_to_total_negative"] = (
            item["base_dot"] > 0 and item["derived_total_dot"] < 0)
        if item["base_dot"] > 0:
            item["other_aux_offset_vs_base_pct"] = -100 * other_dot / item["base_dot"]
            item["status"] = "available"
        else:
            item["status"] = "undefined_base_nonpositive"
    cosine = path_get(result, f"components.{task}.cosine_to_total")
    total_norm = result.get("total_gradient_norm")
    if numeric(cosine):
        item["direct_cosine_to_total"] = cosine
    if numeric(cosine) and numeric(total_norm) and total_norm >= 0:
        item["direct_total_dot"] = cosine * norm * total_norm
        item["direct_total_status"] = "available"
        if numeric(item["derived_total_dot"]):
            item["direct_minus_derived_total_dot"] = (
                item["direct_total_dot"] - item["derived_total_dot"])
    item["direct_total_scope"] = (
        "full_direct_total_with_linearity_residual" if mode == "full"
        else "component_sum_derived_total")
    return item


def summarize_auxiliary_direction(rows, task, mode):
    per_seed = [auxiliary_direction_row(row, task, mode) for row in rows]
    pending = sum(x["status"] == "pending" for x in per_seed)
    inactive = sum(x["status"] == "inactive" for x in per_seed)
    undefined = sum(x["status"] == "undefined_base_nonpositive" for x in per_seed)
    def field_stat(field):
        undefined_count = undefined if field == "other_aux_offset_vs_base_pct" else 0
        pending_count = sum(
            x[field] is None and x["status"] != "inactive" and not (
                field == "other_aux_offset_vs_base_pct"
                and x["status"] == "undefined_base_nonpositive")
            for x in per_seed)
        value = derived_stat([x[field] for x in per_seed], undefined_count, pending_count)
        value["inactive_count"] = inactive
        if value["count"] == 0 and inactive == len(per_seed):
            value["status"] = "inactive"
        return value
    return {
        "per_seed": per_seed,
        "direct_cosine_to_total": field_stat("direct_cosine_to_total"),
        "base_dot": field_stat("base_dot"),
        "other_aux_dot": field_stat("other_aux_dot"),
        "derived_total_dot": field_stat("derived_total_dot"),
        "direct_total_dot": field_stat("direct_total_dot"),
        "direct_minus_derived_total_dot": field_stat("direct_minus_derived_total_dot"),
        "other_aux_offset_vs_base_pct": field_stat("other_aux_offset_vs_base_pct"),
        "derived_total_dot_negative_count": sum(
            numeric(x["derived_total_dot"]) and x["derived_total_dot"] < 0 for x in per_seed),
        "base_positive_to_total_negative_count": sum(
            x["base_positive_to_total_negative"] is True for x in per_seed),
        "active_count": sum(numeric(path_get(row["result"], f"components.{task}.weighted_norm"))
                            and path_get(row["result"], f"components.{task}.weighted_norm") > 0
                            for row in rows),
        "inactive_count": inactive, "pending_count": pending,
        "base_nonpositive_count": undefined,
        "measurement_scope": ("full_direct_total_with_linearity_residual" if mode == "full"
                              else "component_sum_derived_total"),
    }


def compact(value, digits=5):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def report_is_complete(data):
    results = data.get("results")
    if not isinstance(results, dict) or not results:
        return False, "missing/empty results"
    expected = data.get("noise_seeds")
    if isinstance(expected, list) and expected:
        missing = [str(seed) for seed in expected if str(seed) not in results]
        if missing:
            return False, "missing seeds " + ",".join(missing)
    required = ("components", "pairwise_weighted_cosine", "combined", "samples", "microbatches")
    for seed, result in results.items():
        if not isinstance(result, dict) or any(key not in result for key in required):
            return False, f"incomplete result seed={seed}"
    return True, None


def measurement_mode(result, report=None):
    report = report or {}
    metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
    return result.get("measurement_mode", report.get("measurement_mode",
           metadata.get("measurement_mode", "full")))


def measurement_scope(result, mode):
    scope = result.get("measurement_scope")
    if isinstance(scope, dict):
        return scope
    if mode == "component_sum":
        return {"components": "direct_vjp_every_microbatch",
                "main_total": "component_sum_of_mean_weighted_vjps",
                "other": "zero_by_verified_weights",
                "direct_reconstruction": "first_microbatch_only"}
    return {"components": "direct_vjp_every_microbatch",
            "main_total": "direct_vjp_every_microbatch",
            "other": "direct_vjp_every_microbatch",
            "direct_reconstruction": "not applicable"}


def layer_entries(result):
    """Read the actual schema: shared_language_layers.{layers,checks,...}."""
    container = result.get("shared_language_layers")
    layers = container.get("layers") if isinstance(container, dict) else None
    if not isinstance(layers, dict):
        return []
    entries = []
    for name, value in layers.items():
        if isinstance(value, dict):
            row = dict(value)
            row["layer"] = name
            entries.append(row)
    return entries


def layer_value(layer, *names):
    for name in names:
        if name in layer:
            return layer[name]
    return None


def layer_index(layer):
    value = layer_value(layer, "layer", "layer_index", "index", "name")
    match = re.search(r"(\d+)$", str(value)) if value is not None else None
    return int(match.group(1)) if match else None


def collect(input_root):
    root = Path(input_root)
    rows, skipped, pending_stages, pending_reports, warnings = [], [], [], [], []
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            for item_id, item in manifest.get("items", {}).items():
                output = Path(item.get("output", "")) if item.get("output") else None
                if item.get("status") != "complete" or output is None or not output.exists():
                    pending_reports.append({"item": item_id, "status": item.get("status", "unknown"),
                                            "output": str(output) if output else None})
        except Exception as exc:
            skipped.append({"file": str(manifest_path), "reason": f"invalid manifest: {exc}"})
    for stage in STAGES:
        folder = root / stage
        found = False
        if folder.exists():
            for path in sorted(folder.glob("*.json")):
                if "manifest" in path.name.lower():
                    continue
                try:
                    data = json.loads(path.read_text())
                except Exception as exc:
                    skipped.append({"file": str(path), "reason": f"invalid JSON: {exc}"})
                    continue
                if not isinstance(data, dict) or not isinstance(data.get("metadata"), dict):
                    skipped.append({"file": str(path), "reason": "unrecognized report schema"})
                    continue
                complete, reason = report_is_complete(data)
                if not complete:
                    skipped.append({"file": str(path), "reason": reason})
                    continue
                found = True
                metadata = data["metadata"]
                exp = metadata.get("experiment", "unknown")
                step = metadata.get("checkpoint_step", metadata.get("local_step"))
                for seed, result in data["results"].items():
                    mode = measurement_mode(result, data)
                    if mode == "component_sum":
                        direct = result.get("first_micro_direct_checks")
                        if not isinstance(direct, dict):
                            warnings.append(f"{path}: seed={seed} missing first_micro_direct_checks (pending)")
                        else:
                            for name in ("main_vs_components", "total_vs_components"):
                                check = direct.get(name)
                                if not isinstance(check, dict):
                                    warnings.append(f"{path}: seed={seed} missing first_micro_direct_checks.{name} (pending)")
                                elif check.get("passed") is False:
                                    warnings.append(f"{path}: seed={seed} failed first_micro_direct_checks.{name}")
                    for check_name in ("linearity", "repeated_total_vjp", "total_backward_check", "aux_freeze"):
                        check = result.get(check_name)
                        if isinstance(check, dict) and check.get("passed") is False:
                            warnings.append(f"{path}: seed={seed} failed {check_name}")
                    rows.append({"stage": stage, "experiment": exp, "step": step,
                                 "file": str(path), "seed": str(seed), "result": result,
                                 "measurement_mode": mode,
                                 "measurement_scope": measurement_scope(result, mode),
                                 "source_offset": data.get("source_offset"),
                                 "source_rows": data.get("source_rows") or [],
                                 "metadata": metadata})
        if not found:
            pending_stages.append(stage)
    return rows, skipped, pending_stages, pending_reports, warnings


def group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["stage"], row["experiment"], row["step"], row["measurement_mode"])].append(row)
    out = []
    for (stage, exp, step, mode), rs in sorted(groups.items(), key=lambda x: (STAGES.index(x[0][0]), x[0][1], str(x[0][2]), x[0][3])):
        metrics = {}
        for task in ("gen", "loc", "perception", "aux_loc"):
            metrics[f"{task}_weighted_norm"] = summarize_path(rs, f"components.{task}.weighted_norm")
        ratios = []
        for r in rs:
            gen = path_get(r["result"], "components.gen.weighted_norm")
            loc = path_get(r["result"], "components.loc.weighted_norm")
            if numeric(gen) and numeric(loc) and gen != 0:
                ratios.append(loc / gen)
        metrics["loc_over_gen_ratio"] = ({"status": "available", **stats(ratios)} if stats(ratios)
                                          else {"status": "inactive" if all(exists(r["result"], "components.loc.weighted_norm") and exists(r["result"], "components.gen.weighted_norm") for r in rs) else "pending",
                                                "median": None, "min": None, "max": None, "count": 0})
        for a, b in PAIRS:
            metrics[f"cos_{a}_{b}"] = summarize_path(rs, f"pairwise_weighted_cosine.{a}.{b}")
        for path in DERIVED_MODE_PATHS:
            key = path.replace(".", "_")
            if mode == "component_sum":
                metrics[key] = {"status": "not_measured_derived_mode", "median": None,
                                "min": None, "max": None, "count": 0}
            else:
                metrics[key] = summarize_path(rs, path)
        # These ratios are defined for auxiliary terms only.
        for task in ("perception", "aux_loc"):
            p = f"combined.{task}_over_baseline"
            values = [100 * path_get(r["result"], p) if numeric(path_get(r["result"], p)) else None for r in rs]
            raw_stat = stats(values)
            metrics[f"{task}_aux_over_main_pct"] = ({"status": "available", **raw_stat} if raw_stat else
                {"status": "inactive" if any(exists(r["result"], p) for r in rs) else "pending", "median": None, "min": None, "max": None, "count": 0})
        values = [100 * path_get(r["result"], "combined.aux_over_baseline") if numeric(path_get(r["result"], "combined.aux_over_baseline")) else None for r in rs]
        raw_stat = stats(values)
        metrics["aux_perception_aux_loc_over_baseline_pct"] = ({"status": "available", **raw_stat} if raw_stat else
            {"status": "inactive" if any(exists(r["result"], "combined.aux_over_baseline") for r in rs) else "pending", "median": None, "min": None, "max": None, "count": 0})

        directions = {}
        aux_effect_rows = {"gen": [], "loc": []}
        main_direction_rows = {"gen": [], "loc": []}
        aux_increment_rows = {"gen": [], "loc": []}
        for task in ("gen", "loc"):
            for direction, field in (("main_total", "raw_descent_first_order_total"),
                                     ("aux_increment", "raw_descent_first_order_aux_increment")):
                vals = [path_get(r["result"], f"combined.primary_directions.{task}.{field}") for r in rs]
                directions[f"{task}.{direction}"] = {
                    "positive": sum(1 for v in vals if numeric(v) and v > 0),
                    "negative": sum(1 for v in vals if numeric(v) and v < 0),
                    "defined": sum(1 for v in vals if numeric(v)),
                }
            for r in rs:
                total_path = f"combined.primary_directions.{task}.raw_descent_first_order_total"
                aux_path = f"combined.primary_directions.{task}.raw_descent_first_order_aux_increment"
                total = path_get(r["result"], total_path)
                aux = path_get(r["result"], aux_path)
                if numeric(total) and numeric(aux):
                    main = total - aux
                    main_direction_rows[task].append(main)
                    aux_increment_rows[task].append(aux)
                    # Direction scores are local first-order descent approximations.
                    if main < 0:
                        aux_effect_rows[task].append(100 * aux / abs(main))

        direction_effect = {}
        for task in ("gen", "loc"):
            defined_main = len(main_direction_rows[task])
            defined_effect = len(aux_effect_rows[task])
            per_seed = []
            for r in rs:
                total = path_get(r["result"], f"combined.primary_directions.{task}.raw_descent_first_order_total")
                aux = path_get(r["result"], f"combined.primary_directions.{task}.raw_descent_first_order_aux_increment")
                if numeric(total) and numeric(aux):
                    main = total - aux
                    per_seed.append({"seed": r["seed"], "source_offset": r["source_offset"], "main_direction": main,
                                     "aux_increment": aux,
                                     "aux_effect_vs_main_descent_pct": 100 * aux / abs(main) if main < 0 else None,
                                     "status": "available" if main < 0 else "undefined_main_direction_nonnegative"})
                else:
                    per_seed.append({"seed": r["seed"], "source_offset": r["source_offset"], "main_direction": None,
                                     "aux_increment": None, "aux_effect_vs_main_descent_pct": None,
                                     "status": "pending_missing_direction"})
            direction_effect[task] = {
                "main_direction": derived_stat(main_direction_rows[task],
                                                pending_count=len(rs) - defined_main),
                "aux_increment": derived_stat(aux_increment_rows[task],
                                               pending_count=len(rs) - defined_main),
                "aux_effect_vs_main_descent_pct": derived_stat(
                    aux_effect_rows[task], undefined_count=defined_main - defined_effect,
                    pending_count=len(rs) - defined_main),
                "per_seed": per_seed,
                "interpretation": "local first-order approximation; positive opposes main descent, negative assists it",
            }

        p_plus_u_ratios, p_plus_u_undefined, p_plus_u_pending = [], 0, 0
        for r in rs:
            p = path_get(r["result"], "components.perception.weighted_norm")
            u = path_get(r["result"], "components.aux_loc.weighted_norm")
            cos_pu = path_get(r["result"], "pairwise_weighted_cosine.perception.aux_loc")
            if not numeric(p) or not numeric(u):
                p_plus_u_pending += 1
            elif u <= 0:
                p_plus_u_undefined += 1
            elif p == 0:
                p_plus_u_ratios.append(1.0)
            elif not numeric(cos_pu):
                p_plus_u_pending += 1
            else:
                squared = p * p + u * u + 2 * p * u * cos_pu
                p_plus_u_ratios.append(math.sqrt(max(0.0, squared)) / u)
        p_plus_u_ratio = derived_stat(p_plus_u_ratios, p_plus_u_undefined, p_plus_u_pending)

        check_names = ("linearity", "repeated_total_vjp", "total_backward_check", "aux_freeze")
        checks = {}
        for name in check_names:
            values = []
            for r in rs:
                if name == "linearity" and mode == "component_sum":
                    values.append("not_measured_derived_mode")
                    continue
                check = r["result"].get(name)
                if isinstance(check, dict):
                    status = check.get("passed")
                    if status is None and name == "aux_freeze":
                        status = "skipped" if check.get("status") == "skipped" else None
                    if status is not None:
                        values.append(status)
            checks[name] = {"passed": sum(v is True for v in values),
                            "failed": sum(v is False for v in values),
                            "skipped": sum(v == "skipped" for v in values),
                            "not_measured_derived_mode": sum(v == "not_measured_derived_mode" for v in values),
                            "pending": len(rs) - len(values)}
        first_micro_checks = []
        for r in rs:
            direct = r["result"].get("first_micro_direct_checks")
            first_micro_checks.append({"seed": r["seed"], "source_offset": r["source_offset"],
                "scope": direct.get("scope") if isinstance(direct, dict) else None,
                **{name: (direct.get(name) if isinstance(direct, dict) else None)
                   for name in ("main_vs_components", "total_vs_components")}})

        windows = len({r["file"] for r in rs})
        per_file = {}
        for r in rs:
            per_file.setdefault(r["file"], r)
        source_rows_by_window = [r["source_rows"] for r in per_file.values()]
        source_ids = set()
        for source_rows in source_rows_by_window:
            for source in source_rows:
                identity = tuple(source.get(k) for k in ("map", "file_frame", "row_id"))
                source_ids.add(identity if any(v is not None for v in identity)
                               else json.dumps(source, sort_keys=True, ensure_ascii=False))
        source_counts = [len(r["source_rows"]) for r in per_file.values()]
        task_rows = [r["result"].get("samples") for r in rs]
        mem_values = [r["result"].get("peak_cuda_memory_bytes") for r in rs]
        elapsed_values = [r["result"].get("elapsed_seconds") for r in rs]
        layers_by_index = defaultdict(list)
        layers_present = any("shared_language_layers" in r["result"] for r in rs)
        for r in rs:
            for layer in layer_entries(r["result"]):
                idx = layer_index(layer)
                if idx is not None:
                    layers_by_index[idx].append(layer)
        layers = []
        if layers_present:
            # Keep the requested full 24-layer table even if the field is absent in some rows.
            for idx in range(24):
                ls = layers_by_index.get(idx, [])
                layer_summary = {"layer": idx}
                for metric, aliases in {
                    "weighted_gen_loc_cosine": ("weighted_gen_loc_cosine",),
                    "weighted_gen_total_dot": ("weighted_gen_total_dot",),
                    "weighted_loc_total_dot": ("weighted_loc_total_dot",),
                    "weighted_gen_aux_dot": ("weighted_gen_aux_dot",),
                    "weighted_loc_aux_dot": ("weighted_loc_aux_dot",),
                    "negative_alignment_mass": ("negative_alignment_mass", "negative_mass"),
                    "positive_alignment_mass": ("positive_alignment_mass", "positive_mass"),
                }.items():
                    values = [layer_value(l, *aliases) for l in ls]
                    raw_stat = stats(values)
                    has_field = any(any(alias in layer for alias in aliases) for layer in ls)
                    layer_summary[metric] = ({"status": "available", **raw_stat} if raw_stat else
                        {"status": "inactive" if has_field else "pending", "median": None, "min": None, "max": None, "count": 0})
                for field in ("weighted_gen_total_dot", "weighted_loc_total_dot",
                              "weighted_gen_aux_dot", "weighted_loc_aux_dot"):
                    layer_summary[field + "_negative_count"] = sum(
                        1 for l in ls if numeric(l.get(field)) and l[field] < 0)
                layer_summary["observations"] = len(ls)
                layers.append(layer_summary)

        global_layer_mass = {}
        for field in ("positive_alignment_mass", "negative_alignment_mass"):
            values = [path_get(r["result"], f"shared_language_layers.{field}") for r in rs]
            stat = stats(values)
            present = any(exists(r["result"], f"shared_language_layers.{field}") for r in rs)
            global_layer_mass[field] = ({"status": "available", **stat} if stat else
                                        {"status": "inactive" if present else "pending",
                                         "median": None, "min": None, "max": None, "count": 0})

        out.append({"stage": stage, "experiment": exp, "step": step,
                    "measurement_mode": mode,
                    "measurement_scope": rs[0]["measurement_scope"], "rows": len(rs),
                    "windows": windows, "source_count_per_window": stats(source_counts),
                    "window_source_rows_total": sum(source_counts),
                    "window_source_coverage_unique": len(source_ids),
                    "task_rows_per_result": stats(task_rows),
                    "task_rows_cumulative": sum(v for v in task_rows if isinstance(v, int)),
                    "peak_memory_bytes_max": max((v for v in mem_values if numeric(v)), default=None),
                    "elapsed_seconds_total": sum(v for v in elapsed_values if numeric(v)),
                    "elapsed_seconds_per_result": stats(elapsed_values), "metrics": metrics,
                    "direction_counts": directions, "direction_effect": direction_effect,
                    "auxiliary_objective_direction": {
                        task: summarize_auxiliary_direction(rs, task, mode)
                        for task in ("perception", "aux_loc")},
                    "perception_plus_auxloc_over_auxloc_norm": p_plus_u_ratio,
                    "checks": checks,
                    "first_micro_direct_checks": first_micro_checks,
                    "shared_language_layers_present": layers_present,
                    "shared_language_global_alignment_mass": global_layer_mass,
                    "shared_language_layers": layers})
    return out


def fmt_stat(value, digits=4):
    if value is None:
        return "pending"
    if value.get("status") == "inactive":
        return "None (inactive)"
    if value.get("status") == "pending":
        return "pending"
    if value.get("status") == "undefined":
        return "undefined"
    if value.get("status") == "not_measured_derived_mode":
        return "not measured (derived mode)"
    return f"{compact(value['median'], digits)} [{compact(value['min'], digits)}, {compact(value['max'], digits)}] (n={value['count']})"


def fmt_check(check):
    if not isinstance(check, dict):
        return "pending"
    return "passed" if check.get("passed") is True else "FAILED" if check.get("passed") is False else "pending"


def write_outputs(root, outdir, groups, skipped, pending, pending_reports, warnings):
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {"input_root": str(root), "groups": groups, "skipped_incomplete_reports": skipped,
               "pending_stages": pending, "pending_reports": pending_reports, "warnings": warnings,
               "notes": ["Aggregates are across report seed rows within each stage/experiment/step.",
                         "Cosines are read from pre-aggregated report values; no microbatch cosine averaging or vector recomputation.",
                         "screen, confirm, temporal are never merged.",
                         "main_direction = raw_descent_first_order_total - raw_descent_first_order_aux_increment; aux effect (%) = 100 * aux_increment / abs(main_direction), defined only when main_direction < 0.",
                         "perception_plus_auxloc_over_auxloc_norm = ||P+U|| / ||U|| and is defined only when ||U|| > 0; this norm ratio alone does not establish that the P objective is unharmed.",
                         "For each active auxiliary objective A in {P,U}, base_dot=A·(G+L+A), other_aux_dot=A·B, derived_total_dot=base_dot+other_aux_dot, and other_aux_offset_vs_base_pct=-100*other_aux_dot/base_dot only when base_dot>0. Positive percent offsets A descent; negative percent assists it.",
                         "Auxiliary direction removal is same-checkpoint, same-batch local algebra, not a retrained counterfactual. Full direct_total_dot uses component cosine_to_total*component norm*actual total norm; direct_minus_derived_total_dot retains the small linearity residual. In component_sum mode both totals are algebraic and are not an independent acceptance check.",
                         "Direction effects are local first-order approximations; full mode retains the independent reconstruction residual.",
                         "Null values mean undefined/inactive except direct metrics intentionally not measured in component_sum mode; absent required fields are pending."]}
    (outdir / "gradient_evidence.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    md = ["# Gradient diagnostics evidence", "",
          "Compact report-level summary. Rows are grouped by stage, experiment, checkpoint step, and measurement mode; stages are kept separate. Median [min, max] is across seed rows. Cosines are copied from already aggregated report results, never reconstructed from microbatch means. Direction effects are local first-order scalar approximations; full mode includes a tiny independent reconstruction residual. A P+U norm ratio does not establish that P is unharmed: a small P contribution can be overwhelmed by U even when the combined norm is close to U. Missing required fields are pending; null is undefined/inactive except direct metrics intentionally not measured in component_sum mode.", ""]
    for stage in STAGES:
        md += [f"## {stage}", ""]
        gs = [g for g in groups if g["stage"] == stage]
        if not gs:
            md += ["Pending: no complete reports found.", ""]
            continue
        md += ["| Experiment | Step | Mode | Main/total measurement scope | Windows | Source rows/window | Unique source coverage | Task rows/seed | Cumulative task rows | Peak memory (GiB) | Elapsed total (s) |",
               "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for g in gs:
            mem = g["peak_memory_bytes_max"] / (1024 ** 3) if numeric(g["peak_memory_bytes_max"]) else None
            md.append(f"| {g['experiment']} | {compact(g['step'])} | {g['measurement_mode']} | `{g['measurement_scope'].get('main_total')}`; components=`{g['measurement_scope'].get('components')}` | {g['windows']} | {fmt_stat(g['source_count_per_window'])} | {g['window_source_coverage_unique']} | {fmt_stat(g['task_rows_per_result'])} | {g['task_rows_cumulative']} | {compact(mem)} | {compact(g['elapsed_seconds_total'])} |")
        md += ["", "| Experiment | Step | gen norm | weighted loc norm | P norm | U norm | loc/gen | gen-loc cos | P-gen cos | P-loc cos | U-gen cos | U-loc cos | P-U cos | P aux/main % | U aux/main % | all aux/main % |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        metric_keys = ["gen_weighted_norm", "loc_weighted_norm", "perception_weighted_norm", "aux_loc_weighted_norm", "loc_over_gen_ratio", "cos_gen_loc", "cos_perception_gen", "cos_perception_loc", "cos_aux_loc_gen", "cos_aux_loc_loc", "cos_perception_aux_loc"]
        for g in gs:
            m = g["metrics"]
            vals = [fmt_stat(m.get(k)) for k in metric_keys]
            vals += [fmt_stat(m.get("perception_aux_over_main_pct")), fmt_stat(m.get("aux_loc_aux_over_main_pct")), fmt_stat(m.get("aux_perception_aux_loc_over_baseline_pct"))]
            md.append("| " + " | ".join([g["experiment"], compact(g["step"]), *vals]) + " |")
        md += ["", "### Checks and direction counts", ""]
        for g in gs:
            md += [f"**{g['experiment']} step {compact(g['step'])}** ({g['windows']} windows, {g['rows']} seed rows)", ""]
            md.append("Checks: " + "; ".join(f"{k} pass={v['passed']}, fail={v['failed']}, skipped={v['skipped']}, not_measured={v['not_measured_derived_mode']}, pending={v['pending']}" for k, v in g["checks"].items()))
            md.append("Direction counts (>0 / <0 / defined): " + "; ".join(f"{k} {v['positive']}/{v['negative']}/{v['defined']}" for k, v in g["direction_counts"].items()))
            md += ["", "#### Local first-order direction effects", "",
                   "| Task | Main direction | Aux increment | Aux effect vs main descent (%) | Undefined rows | Pending rows |",
                   "|---|---:|---:|---:|---:|---:|"]
            for task in ("gen", "loc"):
                effect = g["direction_effect"][task]
                ratio = effect["aux_effect_vs_main_descent_pct"]
                md.append(f"| {task} | {fmt_stat(effect['main_direction'])} | {fmt_stat(effect['aux_increment'])} | {fmt_stat(ratio)} | {ratio.get('undefined_count', 0)} | {ratio.get('pending_count', 0)} |")
                per_seed = ", ".join(f"offset {item['source_offset']} seed {item['seed']}={compact(item['aux_effect_vs_main_descent_pct'])}%" if item['status'] == "available" else f"offset {item['source_offset']} seed {item['seed']}=undefined" if item['status'] == "undefined_main_direction_nonnegative" else f"offset {item['source_offset']} seed {item['seed']}=pending" for item in effect["per_seed"])
                md.append(f"| {task} per-seed |  |  | {per_seed} |  |  |")
            pu = g["perception_plus_auxloc_over_auxloc_norm"]
            md += ["", f"P+U norm / U norm (defined only for U > 0): {fmt_stat(pu)}; undefined rows={pu.get('undefined_count', 0)}, pending rows={pu.get('pending_count', 0)}."]
            md += ["", "#### Auxiliary objective directions", "",
                   "Weighted G=gen, L=loc, P=perception, U=aux_loc. For A=P or U, base=A·(G+L+A), other=A·(other auxiliary), and derived total=base+other. When base>0, offset %=-100×other/base: positive offsets A descent, negative assists. This is same-checkpoint, same-batch local algebra, not a retrained counterfactual. Full direct total comes from A cosine-to-total × A norm × actual total norm, so direct−derived retains the linearity residual. In component_sum mode the direct field uses a derived total and is not an independent check.", "",
                   "| Objective | cos(A,total) | base dot | other dot | derived total dot | offset % | total dot <0 | base>0 → total<0 | inactive / pending / base≤0 |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for task, label in (("perception", "P"), ("aux_loc", "U")):
                direction = g["auxiliary_objective_direction"][task]
                md.append(f"| {label} | {fmt_stat(direction['direct_cosine_to_total'])} | {fmt_stat(direction['base_dot'])} | {fmt_stat(direction['other_aux_dot'])} | {fmt_stat(direction['derived_total_dot'])} | {fmt_stat(direction['other_aux_offset_vs_base_pct'])} | {direction['derived_total_dot_negative_count']} | {direction['base_positive_to_total_negative_count']} | {direction['inactive_count']} / {direction['pending_count']} / {direction['base_nonpositive_count']} |")
            md += ["", "| Objective | Offset | Seed | base dot | other dot | derived total dot | direct cos | direct total dot | direct−derived | offset % | status |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
            for task, label in (("perception", "P"), ("aux_loc", "U")):
                for item in g["auxiliary_objective_direction"][task]["per_seed"]:
                    md.append("| " + " | ".join([
                        label, compact(item["source_offset"]), item["seed"],
                        compact(item["base_dot"]), compact(item["other_aux_dot"]),
                        compact(item["derived_total_dot"]), compact(item["direct_cosine_to_total"]),
                        compact(item["direct_total_dot"]), compact(item["direct_minus_derived_total_dot"]),
                        compact(item["other_aux_offset_vs_base_pct"]), item["status"]]) + " |")
            if g["measurement_mode"] == "full":
                md.append("Full-window direct checks: linearity relative difference=" +
                    fmt_stat(g["metrics"].get("linearity_relative_difference")) +
                    "; linearity residual norm=" + fmt_stat(g["metrics"].get("components_linearity_residual_weighted_norm")) +
                    "; actual auxiliary increment norm=" + fmt_stat(g["metrics"].get("combined_actual_aux_increment_norm")) +
                    "; component cosine=" + fmt_stat(g["metrics"].get("combined_actual_aux_increment_vs_component_cosine")) +
                    "; difference norm=" + fmt_stat(g["metrics"].get("combined_actual_aux_increment_vs_component_difference_norm")))
            if g["measurement_mode"] == "component_sum":
                md.append("Direct checks (first microbatch only): " + "; ".join(
                    f"offset {c['source_offset']} seed {c['seed']} main={fmt_check(c['main_vs_components'])}, total={fmt_check(c['total_vs_components'])}"
                    for c in g["first_micro_direct_checks"]))
                md.append("Full-window direct-linearity and actual auxiliary-increment checks: not measured (derived mode).")
            md.append("")
        for g in gs:
            if g["shared_language_layers_present"]:
                md += [f"### Shared-language layer evidence: {g['experiment']} step {compact(g['step'])}", "",
                       f"Global alignment mass: positive={fmt_stat(g['shared_language_global_alignment_mass']['positive_alignment_mass'])}; negative={fmt_stat(g['shared_language_global_alignment_mass']['negative_alignment_mass'])}.", "",
                       "| Layer | Weighted gen-loc cos | Negative mass | Positive mass | gen totaldot <0 | loc totaldot <0 | gen auxdot <0 | loc auxdot <0 | n |",
                       "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
                for l in g["shared_language_layers"]:
                    md.append(f"| {l['layer']} | {fmt_stat(l['weighted_gen_loc_cosine'])} | {fmt_stat(l['negative_alignment_mass'])} | {fmt_stat(l['positive_alignment_mass'])} | {l['weighted_gen_total_dot_negative_count']} | {l['weighted_loc_total_dot_negative_count']} | {l['weighted_gen_aux_dot_negative_count']} | {l['weighted_loc_aux_dot_negative_count']} | {l['observations']} |")
                md.append("")
            else:
                layer_status = ("Not measured in this screen report; layer diagnostics are reported in confirm/temporal."
                                if g["stage"] == "screen" else
                                "Pending: shared_language_layers not present.")
                md += [f"### Shared-language layer evidence: {g['experiment']} step {compact(g['step'])}", "", layer_status, ""]
    md += ["## Missing / incomplete", ""]
    md += [f"- Pending stage: {s}" for s in pending] or ["- No entirely missing stages."]
    md += [f"- Pending report: `{x['item']}` (status={x['status']}; output={x['output'] or 'not specified'})" for x in pending_reports] or ["- No manifest entries are pending."]
    md += [f"- Skipped incomplete report: `{x['file']}` — {x['reason']}" for x in skipped] or ["- No incomplete reports skipped."]
    md += [f"- Warning: {x}" for x in warnings] or ["- No check warnings."]
    (outdir / "gradient_evidence.md").write_text("\n".join(md) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", type=Path, default=Path("outputs_diagnostics/gradient_seen3_fp32"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs_diagnostics/gradient_seen3_fp32_evidence"))
    args = ap.parse_args()
    rows, skipped, pending, pending_reports, warnings = collect(args.input_root)
    groups = group_rows(rows)
    write_outputs(args.input_root, args.output_dir, groups, skipped, pending, pending_reports, warnings)
    print(f"complete_seed_rows={len(rows)} groups={len(groups)} skipped={len(skipped)} warnings={len(warnings)} pending_stages={','.join(pending) or 'none'} pending_reports={len(pending_reports)}")
    print(f"json={args.output_dir / 'gradient_evidence.json'}")
    print(f"markdown={args.output_dir / 'gradient_evidence.md'}")


if __name__ == "__main__":
    main()
