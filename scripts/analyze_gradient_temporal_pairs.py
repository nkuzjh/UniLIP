#!/usr/bin/env python3
"""Pair the fixed B1/B2 Seen-3 gradient reports at steps 1200 and 1800.

Reads JSON on CPU only. Offset 0 comes from the original temporal queue;
offsets 128 and 256 at step 1200 come only from --supplement-root/temporal.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import statistics

from export_gradient_observations import check_report, finite, observation, write_atomic
from render_gradient_evidence import auxiliary_direction_row, path_get
from run_gradient_diagnostics import DEFAULT_CONFIG, build_items, load_config, report_path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs_diagnostics/gradient_seen3_fp32"
DEFAULT_OUTPUT = ROOT / "outputs_diagnostics/gradient_seen3_temporal_pairs"
EXPERIMENTS = ("exp32_3maps_B1_S3_J8", "exp32_3maps_B2_S3_J8")
OFFSETS = (0, 128, 256)
SEEDS = (17, 29)
TASKS = ("gen", "loc")
CORE_SIGN_FIELDS = ("gen_loc_main_cosine", "perception_aux_loc_cosine",
                    "gen_aux_effect_vs_main_descent_pct",
                    "loc_aux_effect_vs_main_descent_pct",
                    "gen_other_main_cancellation_pct",
                    "loc_other_main_cancellation_pct",
                    "p_cosine_to_total", "u_cosine_to_total")


def delta_unit(field: str) -> str:
    if field.endswith("_pct"):
        return "percentage_points"
    if "cosine" in field:
        return "absolute"
    if field.endswith("_ascent") or field.endswith("_ascent_count"):
        return "count"
    return "native_units"


def window_delta_signs(pairs: list[dict], field: str) -> dict:
    """Classify two paired seed deltas separately for each source window."""
    windows = {}
    for offset in OFFSETS:
        by_seed = {p["noise_seed"]: p["delta"].get(field)
                   for p in pairs if p["source_offset"] == offset}
        if len(by_seed) != 2:
            category = "pending"
        elif any(not finite(by_seed.get(seed)) for seed in SEEDS):
            category = "undefined"
        else:
            a, b = (by_seed[seed] for seed in SEEDS)
            if a > 0 and b > 0:
                category = "both_positive"
            elif a < 0 and b < 0:
                category = "both_negative"
            elif a * b < 0:
                category = "mixed"
            else:
                category = "zero_or_tie"
        windows[str(offset)] = {"classification": category,
                                "delta_by_seed": {str(seed): by_seed.get(seed) for seed in SEEDS}}
    categories = [window["classification"] for window in windows.values()]
    positive = categories.count("both_positive")
    negative = categories.count("both_negative")
    return {"windows": windows,
            "same_direction_window_count": positive + negative,
            "both_positive_window_count": positive,
            "both_negative_window_count": negative,
            "mixed_window_count": categories.count("mixed"),
            "zero_or_tie_window_count": categories.count("zero_or_tie"),
            "pending_window_count": categories.count("pending"),
            "undefined_window_count": categories.count("undefined"),
            "shared_direction_window_count": max(positive, negative),
            "all_three_windows_share_direction": positive == 3 or negative == 3}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def offload_metadata(report: dict, seed: int) -> dict:
    """Preserve reported diagnostic offload settings without assuming a key name."""
    found = {}

    def visit(value: object, prefix: str) -> None:
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            path = f"{prefix}.{key}"
            if any(term in key.lower() for term in ("offload", "save_on_cpu", "saved_tensor")):
                found[path] = child
            elif isinstance(child, dict):
                visit(child, path)

    visit(report.get("metadata"), "metadata")
    visit((report.get("results") or {}).get(str(seed)), f"results.{seed}")
    for key, value in report.items():
        if key not in ("metadata", "results") and any(
                term in key.lower() for term in ("offload", "save_on_cpu", "saved_tensor")):
            found[key] = value
    return found


def pair_evidence(early: dict, late: dict, seed: int,
                  early_sha256: str, late_sha256: str) -> dict:
    a, b = early["results"][str(seed)], late["results"][str(seed)]
    source_a, source_b = identity(early), identity(late)
    rng_a = a.get("rng_state_after_measure_sha256")
    rng_b = b.get("rng_state_after_measure_sha256")
    rng_available = isinstance(rng_a, dict) and isinstance(rng_b, dict)
    rng_equal = rng_a == rng_b if rng_available else None
    rng_different = (sorted(key for key in set(rng_a) | set(rng_b)
                            if rng_a.get(key) != rng_b.get(key)) if rng_available else None)
    checks = {"source_seed_equal": early["source_seed"] == late["source_seed"],
              "source_offset_equal": early["source_offset"] == late["source_offset"],
              "ordered_source_ids_equal": source_a == source_b,
              "noise_seeds_equal": early["noise_seeds"] == late["noise_seeds"],
              "microbatch_size_equal": early["microbatch_size"] == late["microbatch_size"],
              "accumulations_equal": early["accumulations"] == late["accumulations"],
              "report_coordinate_equal": early["coordinate"] == late["coordinate"],
              "result_coordinate_equal": a["coordinate"] == b["coordinate"],
              "metadata_loss_weights_equal": (early["metadata"]["loss_weights"] ==
                                              late["metadata"]["loss_weights"]),
              "measured_weights_equal": a["weights_seen"] == b["weights_seen"],
              "report_measurement_mode_equal": (early["measurement_mode"] ==
                                                late["measurement_mode"]),
              "result_measurement_mode_equal": (a["measurement_mode"] ==
                                                b["measurement_mode"]),
              "tasks_equal": early["tasks"] == late["tasks"]}
    if not all(checks.values()):
        raise ValueError("pair evidence checks failed: " +
                         ", ".join(key for key, value in checks.items() if not value))
    return {
        "pair_checks": checks,
        "individual_reports_passed_check_report": True,
        "report_sha256": {"step1200": early_sha256, "step1800": late_sha256},
        "checkpoint_dir": {"step1200": early["checkpoint_dir"],
                           "step1800": late["checkpoint_dir"]},
        "source_seed": early["source_seed"], "source_offset": early["source_offset"],
        "source_count": len(source_a), "ordered_source_ids_sha256": digest(source_a),
        "ordered_source_ids_equal": checks["ordered_source_ids_equal"],
        "source_indices_equal": early.get("source_indices") == late.get("source_indices"),
        "noise_seeds": early["noise_seeds"], "microbatch_size": early["microbatch_size"],
        "accumulations": early["accumulations"], "coordinate": early["coordinate"],
        "coordinate_sha256": digest(a["coordinate"]),
        "coordinate_equal": checks["result_coordinate_equal"],
        "metadata_loss_weights": early["metadata"]["loss_weights"],
        "measured_weights": a["weights_seen"],
        "measured_weights_equal": checks["measured_weights_equal"],
        "measurement_mode": early["measurement_mode"],
        "rng_state_after_measure_equal": rng_equal,
        "rng_state_after_measure_status": ("equal" if rng_equal is True else
                                            "different" if rng_equal is False else "unavailable"),
        "rng_state_after_measure_different_components": rng_different,
        "rng_state_after_measure_sha256": {"step1200": rng_a, "step1800": rng_b},
        "diagnostic_offload_metadata": {"step1200": offload_metadata(early, seed),
                                        "step1800": offload_metadata(late, seed)},
    }


def identity(report: dict) -> list[tuple[str, str, str]]:
    return [(row["map"], row["file_frame"], row["row_id"])
            for row in report["source_rows"]]


def pair_errors(early: dict, late: dict) -> list[str]:
    """Check every condition that must match before a checkpoint delta is valid."""
    errors = []
    fields = ("source_seed", "source_offset", "noise_seeds", "microbatch_size",
              "accumulations", "coordinate", "measurement_mode", "tasks")
    for field in fields:
        if early.get(field) != late.get(field):
            errors.append(field)
    if identity(early) != identity(late):
        errors.append("ordered map/file_frame/row_id source identities")
    if early["metadata"].get("loss_weights") != late["metadata"].get("loss_weights"):
        errors.append("metadata loss weights")
    for seed in SEEDS:
        a, b = early["results"][str(seed)], late["results"][str(seed)]
        if a.get("coordinate") != b.get("coordinate"):
            errors.append(f"seed {seed} coordinate names/shape")
        if a.get("weights_seen") != b.get("weights_seen"):
            errors.append(f"seed {seed} measured weights")
        if a.get("measurement_mode") != b.get("measurement_mode"):
            errors.append(f"seed {seed} measurement mode")
    return errors


def layer_metrics(result: dict, task: str) -> dict[str, float | int | None]:
    layers = path_get(result, "shared_language_layers.layers")
    if not isinstance(layers, dict) or len(layers) != 24 or set(layers) != {str(i) for i in range(24)}:
        raise ValueError("expected exactly 24 shared language layers")
    total_ascent = main_ascent = 0
    cancellations = []
    for layer in layers.values():
        total_dot = layer[f"weighted_{task}_total_dot"]
        aux_dot = layer[f"weighted_{task}_aux_dot"]
        own = layer["components"][task]["weighted_norm"]
        cross_dot = layer["weighted_gen_loc_dot"]
        if not all(finite(x) for x in (total_dot, aux_dot, own, cross_dot)) or own < 0:
            raise ValueError("nonfinite layer primary measurement")
        total_ascent += total_dot < 0
        main_ascent += total_dot - aux_dot < 0
        if own > 0:
            cancellations.append(-100 * cross_dot / (own * own))
    return {f"{task}_layer_total_ascent_count": total_ascent,
            f"{task}_layer_main_only_ascent_count": main_ascent,
            f"{task}_layer_max_cross_main_cancellation_pct": max(cancellations, default=None)}


def metrics(report: dict, seed: int, item: object) -> dict:
    result = report["results"][str(seed)]
    base = observation(report, item, seed)
    out = {"gen_loc_main_cosine": base["weighted_cos_gen_loc"],
           "perception_aux_loc_cosine": base["weighted_cos_perception_aux_loc"],
           "aux_over_main_pct": base["aux_over_main_pct"]}
    for task in ("gen", "loc", "perception", "aux_loc"):
        for suffix in ("raw_norm", "weighted_norm"):
            out[f"{task}_{suffix}"] = base[f"{task}_{suffix}"]
    for task in TASKS:
        for source, target in (("raw_total_first_order", "total_descent_first_order"),
                               ("raw_main_first_order", "main_only_descent_first_order"),
                               ("aux_effect_vs_main_descent_pct", "aux_effect_vs_main_descent_pct"),
                               ("other_main_offset_pct", "other_main_cancellation_pct")):
            out[f"{task}_{target}"] = base[f"{task}_{source}"]
        out[f"{task}_global_total_ascent"] = int(base[f"{task}_raw_total_first_order"] > 0)
        out[f"{task}_global_main_only_ascent"] = int(base[f"{task}_raw_main_first_order"] > 0)
        out.update(layer_metrics(result, task))
    for task, short in (("perception", "p"), ("aux_loc", "u")):
        direction = auxiliary_direction_row({"result": result, "seed": str(seed),
                                             "source_offset": report["source_offset"]},
                                            task, item.measurement_mode)
        out[f"{short}_cosine_to_total"] = direction["direct_cosine_to_total"]
        out[f"{short}_base_dot"] = direction["base_dot"]
        out[f"{short}_other_aux_dot"] = direction["other_aux_dot"]
        out[f"{short}_total_dot"] = direction["derived_total_dot"]
        out[f"{short}_cross_aux_cancellation_pct"] = direction["other_aux_offset_vs_base_pct"]
    required = ["gen_loc_main_cosine", "perception_aux_loc_cosine", "aux_over_main_pct"]
    for task in ("gen", "loc", "perception", "aux_loc"):
        required += [f"{task}_raw_norm", f"{task}_weighted_norm"]
    for task in TASKS:
        required += [f"{task}_total_descent_first_order",
                     f"{task}_main_only_descent_first_order"]
        if out[f"{task}_weighted_norm"] > 0:
            required.append(f"{task}_other_main_cancellation_pct")
        if out[f"{task}_main_only_descent_first_order"] < 0:
            required.append(f"{task}_aux_effect_vs_main_descent_pct")
    for task, short in (("perception", "p"), ("aux_loc", "u")):
        if out[f"{task}_weighted_norm"] > 0:
            required += [f"{short}_cosine_to_total", f"{short}_base_dot",
                         f"{short}_other_aux_dot", f"{short}_total_dot"]
            if finite(out[f"{short}_base_dot"]) and out[f"{short}_base_dot"] > 0:
                required.append(f"{short}_cross_aux_cancellation_pct")
    missing = [key for key in required if not finite(out.get(key))]
    if missing:
        raise ValueError("missing/nonfinite required metrics: " + ", ".join(missing))
    return out


def summarize(pairs: list[dict], experiment: str) -> dict:
    selected = [p for p in pairs if p["experiment"] == experiment]
    fields = sorted(set(CORE_SIGN_FIELDS) | {key for pair in selected for key in pair["delta"]})
    output = {}
    for field in fields:
        values = [p["delta"].get(field) for p in selected if finite(p["delta"].get(field))]
        if not values:
            continue
        # Directional labels apply only where larger alignment or less
        # cancellation/ascent has an unambiguous local interpretation.
        higher_better = field == "gen_loc_main_cosine" or field.endswith("_cosine_to_total")
        lower_better = (field.endswith("_cancellation_pct") or
                        field.endswith("_aux_effect_vs_main_descent_pct") or
                        field.endswith("_ascent_count") or
                        field.endswith("_ascent"))
        stat = {"n": len(values), "unit": delta_unit(field),
                "median": statistics.median(values),
                "min": min(values), "max": max(values),
                "increased": sum(v > 0 for v in values),
                "decreased": sum(v < 0 for v in values),
                "unchanged": sum(v == 0 for v in values)}
        if higher_better or lower_better:
            stat["improved"] = stat["increased" if higher_better else "decreased"]
            stat["worsened"] = stat["decreased" if higher_better else "increased"]
        output[field] = stat
    return {"paired_observations": len(selected), "expected_observations": 6,
            "complete": len(selected) == 6,
            "improvement_scope": "Only local direction: more positive cosine to total/main alignment, less cancellation or auxiliary resistance, and fewer ascent flags/layers. This does not imply evaluation improvement.",
            "delta_1800_minus_1200": output,
            "window_delta_signs": {field: {"unit": delta_unit(field),
                                           **window_delta_signs(selected, field)}
                                   for field in fields}}


def markdown(data: dict) -> str:
    def fmt(value: object) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.5g}"
        return str(value)

    lines = ["# B1/B2 temporal gradient pairs", "",
             f"Status: **{'complete' if data['complete'] else 'incomplete'}**; "
             f"{len(data['pairs'])}/12 valid window × seed pairs.", "",
             "Each experiment has three source windows × two noise seeds. The six observations share training checkpoints and are not independent training repeats. No significance inference is made.", "",
             "Δ = step 1800 minus step 1200. Positive auxiliary effect or cancellation means local resistance to descent. Auxiliary effect is defined only when main-only descent is negative.", "",
             "Percentage metric Δ uses percentage_points; cosine Δ uses absolute cosine units (a signed difference, not |Δ|). No window-averaged cosine is presented as a 384-source gradient cosine.", "",
             "Improved/worsened labels mean only more same-direction alignment, less local cancellation/resistance, or fewer ascent cases. They do not indicate evaluation improvement. Other metrics show increase/decrease without a quality direction.", ""]
    if data["missing"] or data["invalid"]:
        lines += ["## Missing or invalid inputs", ""]
        for entry in data["missing"]:
            lines.append(f"- Missing: `{entry}`")
        for entry in data["invalid"]:
            lines.append(f"- Invalid: `{entry['path']}`: {', '.join(entry['errors'])}")
        lines.append("")
    for experiment in EXPERIMENTS:
        summary = data["summary"][experiment]
        lines += [f"## {experiment}", "",
                  f"Valid pairs: {summary['paired_observations']}/6.", "",
                  "| Metric | Δ unit | Median Δ | Range Δ | Improved / worsened |",
                  "|---|---|---:|---:|---:|"]
        for key, stat in summary["delta_1800_minus_1200"].items():
            direction = (f"{stat['improved']} / {stat['worsened']}"
                         if "improved" in stat else
                         f"increase {stat['increased']} / decrease {stat['decreased']}; direction not assigned")
            lines.append(f"| {key} (n={stat['n']}) | {stat['unit']} | {stat['median']:.5g} | "
                         f"{stat['min']:.5g} to {stat['max']:.5g} | {direction} |")
        lines.append("")
        lines += ["### Window sign consistency", "",
                  "A window is same direction when both seed deltas have the same nonzero sign. Shared direction counts the largest group of windows with the same such sign; pending and undefined windows are excluded.", "",
                  "| Metric | Offset | Δ seed 17 | Δ seed 29 | Classification | Same direction windows | Shared direction windows |",
                  "|---|---:|---:|---:|---|---:|---:|"]
        for key in CORE_SIGN_FIELDS:
            signs = summary["window_delta_signs"][key]
            for offset in OFFSETS:
                window = signs["windows"][str(offset)]
                lines.append(f"| {key} ({signs['unit']}) | {offset} | "
                             f"{fmt(window['delta_by_seed']['17'])} | "
                             f"{fmt(window['delta_by_seed']['29'])} | "
                             f"{window['classification']} | "
                             f"{signs['same_direction_window_count']} | "
                             f"{signs['shared_direction_window_count']} |")
        lines.append("")
    lines += ["## Individual pairs", "",
              "All metric values below are local measurements or algebra from the paired reports.", ""]
    for pair in data["pairs"]:
        evidence = pair["evidence"]
        lines += [f"### {pair['experiment']} · offset {pair['source_offset']} · seed {pair['noise_seed']}", "",
                  "Individual report validation: passed; pair identity and settings checks: " +
                  ", ".join(f"{key}={value}" for key, value in evidence["pair_checks"].items()) + ".", "",
                  f"Report SHA256 (1200 / 1800): `{evidence['report_sha256']['step1200']}` / `{evidence['report_sha256']['step1800']}`.", "",
                  f"Matched source IDs: {evidence['ordered_source_ids_equal']} ({evidence['source_count']} rows; SHA256 `{evidence['ordered_source_ids_sha256']}`); source indices equal: {evidence['source_indices_equal']}.", "",
                  f"Coordinate: {evidence['coordinate']} (equal: {evidence['coordinate_equal']}; SHA256 `{evidence['coordinate_sha256']}`). Microbatch: {evidence['microbatch_size']}; accumulations: {evidence['accumulations']}; mode: {evidence['measurement_mode']}; source seed: {evidence['source_seed']}; noise seeds: {evidence['noise_seeds']}.", "",
                  f"Metadata weights: `{json.dumps(evidence['metadata_loss_weights'], sort_keys=True)}`. Measured weights equal: {evidence['measured_weights_equal']}; values: `{json.dumps(evidence['measured_weights'], sort_keys=True)}`.", "",
                  f"RNG state after measurement: **{evidence['rng_state_after_measure_status']}**; differing components: {evidence['rng_state_after_measure_different_components']}.", "",
                  f"Diagnostic offload metadata (1200 / 1800): `{json.dumps(evidence['diagnostic_offload_metadata']['step1200'], sort_keys=True)}` / `{json.dumps(evidence['diagnostic_offload_metadata']['step1800'], sort_keys=True)}`. Empty means not reported by that report.", "",
                  "| Metric | Step 1200 | Step 1800 | Δ |",
                  "|---|---:|---:|---:|"]
        for key in sorted(pair["delta"]):
            lines.append(f"| {key} | {fmt(pair['step1200'][key])} | "
                         f"{fmt(pair['step1800'][key])} | {fmt(pair['delta'][key])} "
                         f"({delta_unit(key)}) |")
        lines.append("")
    return "\n".join(lines) + "\n"


def csv_text(pairs: list[dict]) -> str:
    fieldnames = ["experiment", "source_offset", "noise_seed", "step1200_report", "step1800_report"]
    evidence_names = ("pair_checks", "individual_reports_passed_check_report",
                      "report_sha256", "checkpoint_dir", "source_seed", "source_count",
                      "ordered_source_ids_sha256", "ordered_source_ids_equal",
                      "source_indices_equal", "noise_seeds", "microbatch_size",
                      "accumulations", "coordinate", "coordinate_sha256", "coordinate_equal",
                      "metadata_loss_weights", "measured_weights", "measured_weights_equal",
                      "measurement_mode", "rng_state_after_measure_equal",
                      "rng_state_after_measure_status",
                      "rng_state_after_measure_different_components",
                      "rng_state_after_measure_sha256", "diagnostic_offload_metadata")
    fieldnames += [f"evidence_{name}" for name in evidence_names]
    metric_names = sorted({key for pair in pairs for key in pair["delta"]})
    fieldnames += [f"{step}_{name}" for name in metric_names for step in ("step1200", "step1800", "delta")]
    fieldnames += [f"delta_unit_{name}" for name in metric_names]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for pair in pairs:
        row = {key: pair[key] for key in fieldnames[:5]}
        for name in evidence_names:
            value = pair["evidence"][name]
            row[f"evidence_{name}"] = (json.dumps(value, sort_keys=True, ensure_ascii=False)
                                         if isinstance(value, (dict, list)) else value)
        for name in metric_names:
            for step in ("step1200", "step1800", "delta"):
                row[f"{step}_{name}"] = pair[step].get(name)
            row[f"delta_unit_{name}"] = delta_unit(name)
        writer.writerow(row)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--supplement-root", type=Path, required=True,
                        help="Only step-1200 offsets 128/256 are read from ROOT/temporal")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    config = load_config(DEFAULT_CONFIG)
    planned = {(i.experiment, i.checkpoint_step, i.source_offset): i
               for i in build_items(config) if i.experiment in EXPERIMENTS}
    reports, paths, hashes, missing, invalid = {}, {}, {}, [], []
    for experiment in EXPERIMENTS:
        for offset in OFFSETS:
            for step in (1200, 1800):
                key = (experiment, step, offset)
                template = planned[(experiment, 1200, 0)] if step == 1200 else planned[key]
                item = replace(template, source_offset=offset)
                root = args.supplement_root if step == 1200 and offset != 0 else args.input_root
                path = report_path(root, item)
                paths[key] = path
                if not path.is_file():
                    missing.append(str(path))
                    continue
                try:
                    raw = path.read_bytes()
                    report = json.loads(raw)
                    if not isinstance(report, dict):
                        raise ValueError("report is not a JSON object")
                    errors = check_report(report, item)
                    if errors:
                        invalid.append({"path": str(path), "errors": errors})
                    else:
                        reports[key] = report
                        hashes[key] = hashlib.sha256(raw).hexdigest()
                except (OSError, ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
                    invalid.append({"path": str(path), "errors": [str(exc)]})
    pairs = []
    for experiment in EXPERIMENTS:
        for offset in OFFSETS:
            early_key, late_key = (experiment, 1200, offset), (experiment, 1800, offset)
            if early_key not in reports or late_key not in reports:
                continue
            errors = pair_errors(reports[early_key], reports[late_key])
            if errors:
                invalid.append({"path": f"{paths[early_key]} <-> {paths[late_key]}",
                                "errors": errors})
                continue
            item = replace(planned[early_key] if early_key in planned else planned[(experiment, 1200, 0)],
                           source_offset=offset)
            for seed in SEEDS:
                try:
                    early = metrics(reports[early_key], seed, item)
                    late = metrics(reports[late_key], seed, replace(item, checkpoint_step=1800))
                    delta = {key: (late[key] - early[key] if finite(early[key]) and finite(late[key]) else None)
                             for key in early}
                    pairs.append({"experiment": experiment, "source_offset": offset,
                                  "noise_seed": seed, "step1200_report": str(paths[early_key]),
                                  "step1800_report": str(paths[late_key]),
                                  "evidence": pair_evidence(reports[early_key], reports[late_key],
                                                            seed, hashes[early_key], hashes[late_key]),
                                  "step1200": early, "step1800": late, "delta": delta})
                except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
                    invalid.append({"path": f"{paths[early_key]} <-> {paths[late_key]} seed={seed}",
                                    "errors": [str(exc)]})
    complete = not missing and not invalid and len(pairs) == 12
    data = {"schema_version": 1, "complete": complete, "expected_pairs": 12,
            "input_root": str(args.input_root), "supplement_root": str(args.supplement_root),
            "supplement_scope": "step1200 offsets 128 and 256 only; temporal subdirectory",
            "missing": missing, "invalid": invalid, "pairs": pairs,
            "summary": {exp: summarize(pairs, exp) for exp in EXPERIMENTS},
            "delta_units": {field: delta_unit(field)
                            for field in sorted(set(CORE_SIGN_FIELDS) |
                                                {key for pair in pairs for key in pair["delta"]})},
            "interpretation": "Six paired window×seed observations per experiment are not independent training repeats; three windows only; no significance inference. Improved/worsened describes only more same-direction local alignment, less local cancellation/resistance, or fewer ascent cases; it does not imply evaluation improvement."}
    if args.require_complete and not complete:
        print(f"incomplete: {len(pairs)}/12 pairs; missing={len(missing)} invalid={len(invalid)}; outputs untouched")
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(args.output_dir / "temporal_pairs.json", json.dumps(data, indent=2, ensure_ascii=False,
                                                                        allow_nan=False) + "\n")
    write_atomic(args.output_dir / "temporal_pairs.csv", csv_text(pairs))
    write_atomic(args.output_dir / "temporal_pairs.md", markdown(data))
    print(f"wrote {len(pairs)}/12 pairs; complete={complete}; missing={len(missing)} invalid={len(invalid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
