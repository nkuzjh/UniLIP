#!/usr/bin/env python3
"""Accept and archive the four Seen-3 temporal supplements using CPU evidence.

No probe, Torch module, GPU, or model weights are loaded. Frozen inputs,
including checkpoint model files, are checked by streaming SHA256 reads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile

import export_gradient_observations as export
import run_gradient_diagnostics as original
import run_gradient_temporal_supplement as supplement

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/gradient_diagnostics_20261102/temporal_extended"
PORTABLE = ROOT / "docs/gradient_diagnostics_20261102"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def old_evidence() -> tuple[list[dict], dict, list[dict], int, int]:
    portable_manifest_path = PORTABLE / "evidence_manifest.json"
    portable_csv_path = PORTABLE / "observations.csv"
    exported_manifest_path = ROOT / "outputs_diagnostics/gradient_seen3_fp32_export/evidence_manifest.json"
    exported_csv_path = ROOT / "outputs_diagnostics/gradient_seen3_fp32_export/observations.csv"
    require(sha256(portable_manifest_path) == sha256(exported_manifest_path),
            "Original portable/export evidence manifests differ")
    require(sha256(portable_csv_path) == sha256(exported_csv_path),
            "Original portable/export observation CSVs differ")
    manifest = read_json(portable_manifest_path)
    require(manifest.get("complete") is True and manifest.get("expected_reports") == 27
            and manifest.get("exported_valid_seed_rows") == 54
            and not manifest.get("missing") and not manifest.get("invalid")
            and len(manifest.get("reports", [])) == 27,
            "Original 27-report evidence manifest is incomplete")
    with portable_csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        original_rows = list(reader)
        original_fields = reader.fieldnames
    require(len(original_rows) == 54 and len({(r["stage"], r["experiment"],
            r["checkpoint_step"], r["source_offset"], r["noise_seed"])
            for r in original_rows}) == 54, "Original CSV is not 54 unique observations")
    require(original_fields is not None, "Original CSV has no header")
    original_root = ROOT / "outputs_diagnostics/gradient_seen3_fp32"
    config = original.load_config(original.DEFAULT_CONFIG)
    items = original.build_items(config)
    require(len(items) == 27, "Base plan no longer has 27 reports")
    listed = {record["path"]: record for record in manifest["reports"]}
    queue = read_json(original_root / "manifest.json")
    freeze = layers = 0
    for item in items:
        path = original.report_path(original_root, item)
        relative = path.relative_to(original_root).as_posix()
        record = listed.get(relative)
        require(isinstance(record, dict) and record.get("valid") is True and not record.get("errors"),
                f"Original evidence missing valid record: {relative}")
        require(sha256(path) == record["sha256"], f"Original report SHA256 changed: {relative}")
        queue_record = queue.get("items", {}).get(item.name, {})
        # The original controller records successful exit via status/checks;
        # its manifest has no separate exit_code field.
        require(queue_record.get("status") == "complete" and queue_record.get("checks") == "passed",
                f"Original queue item incomplete: {item.name}")
        valid, errors = original.report_checks(path, item)
        require(valid, f"Original report_checks failed {relative}: {errors}")
        report = read_json(path)
        require(not export.check_report(report, item), f"Original export checks failed: {relative}")
        for seed in item.noise_seeds:
            result = report["results"][str(seed)]
            freeze += result["aux_freeze"]["checks"]
            layer_rows = export.path_get(result, "shared_language_layers.layers")
            layers += len(layer_rows) if isinstance(layer_rows, dict) else 0
    return original_rows, {"fields": original_fields, "manifest": manifest,
                           "manifest_sha256": sha256(portable_manifest_path),
                           "csv_sha256": sha256(portable_csv_path)}, items, freeze, layers


def supplement_evidence() -> tuple[list[dict], list[dict], dict, dict, int, int]:
    config, items = supplement.load_plan()
    require(len(items) == 4, "Supplement plan no longer has four reports")
    run_root = original.path_in_repo(config["output_root"])
    manifest_path = run_root / "manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("schema_version") == 1 and len(manifest.get("items", {})) == 4,
            "Supplement manifest must contain exactly four items")
    acceptance_path = original.path_in_repo(config["offload_acceptance"])
    acceptance = supplement.validate_offload_acceptance(config)
    require(manifest.get("offload_acceptance") == acceptance,
            "Supplement manifest offload acceptance differs from accepted artifact")
    require(manifest.get("original_code_sha256") == supplement.EXPECTED_CODE_SHA256
            and manifest.get("wrapper_sha256") == supplement.EXPECTED_WRAPPER_SHA256,
            "Supplement frozen code/wrapper hash differs")
    frozen = manifest.get("input_hashes")
    require(isinstance(frozen, dict) and frozen, "Supplement manifest input hashes missing")
    expected_paths = supplement.input_hashes(supplement.DEFAULT_CONFIG, config, items)
    require(expected_paths == frozen, "Supplement frozen input paths or hashes changed")
    wrapper_path = str(original.path_in_repo(config["wrapper"]).resolve())
    require(frozen.get(wrapper_path) == supplement.EXPECTED_WRAPPER_SHA256,
            "Reviewed offload wrapper hash changed")
    require(original.code_sha256() == supplement.EXPECTED_CODE_SHA256,
            "Original code hash changed")
    anchors = supplement.validate_anchors(config, items)
    rows, records = [], []
    freeze = layers = 0
    wrapper_hash = supplement.EXPECTED_WRAPPER_SHA256
    for item in items:
        path = original.report_path(run_root, item)
        entry = manifest["items"].get(item.name, {})
        attempts = entry.get("attempts") or []
        require(entry.get("status") == "complete" and entry.get("exit_code") == 0
                and entry.get("checks") == "passed" and attempts
                and attempts[-1].get("exit_code") == 0
                and attempts[-1].get("checks") == "passed"
                and Path(entry.get("output", "/missing")).resolve() == path.resolve(),
                f"Supplement manifest item not complete/exit0: {item.name}")
        signature = supplement._signature(item, frozen, attempts[-1]["command"][0])
        require(entry.get("signature") == signature,
                f"Supplement item signature changed: {item.name}")
        valid, errors = supplement.report_checks(path, item, anchors[item.name],
                                                  wrapper_hash=wrapper_hash)
        require(valid, f"Supplement report_checks failed {path}: {errors}")
        report = read_json(path)
        require(not export.check_report(report, item),
                f"Supplement export check failed: {path}")
        require(entry.get("saved_tensor_offload") ==
                report["metadata"].get("saved_tensor_offload"),
                f"Supplement offload metadata differs from manifest: {path}")
        record = {"item": item.name, "path": str(path), "sha256": sha256(path),
                  "status": "complete", "exit_code": 0,
                  "report_checks": "passed", "noise_seeds": list(item.noise_seeds),
                  "microbatch": report["microbatch_size"],
                  "accumulations": report["accumulations"],
                  "offload": report["metadata"]["saved_tensor_offload"]}
        records.append(record)
        for seed in item.noise_seeds:
            result = report["results"][str(seed)]
            freeze += result["aux_freeze"]["checks"]
            layers += len(result["shared_language_layers"]["layers"])
            row = export.observation(report, item, seed)
            row["stage"] = "temporal_extended"
            rows.append(row)
    require(len(rows) == 8 and freeze == 512 and layers == 192,
            f"Supplement counts differ: rows={len(rows)}, freeze={freeze}, layers={layers}")
    return rows, records, manifest, acceptance, freeze, layers


def direction_counts(rows: list[dict]) -> dict:
    result = {}
    for group, selected in (("all", rows),
                            ("original", [r for r in rows if r["stage"] != "temporal_extended"]),
                            ("temporal_extended", [r for r in rows if r["stage"] == "temporal_extended"])):
        directions = {}
        for task in ("gen", "loc"):
            for scope in ("raw_total_first_order", "raw_main_first_order"):
                key = f"{task}_{scope}"
                values = [float(r[key]) for r in selected if r[key] not in (None, "")]
                directions[key] = {"descent_negative": sum(v < 0 for v in values),
                                   "ascent_positive": sum(v > 0 for v in values),
                                   "zero": sum(v == 0 for v in values), "available": len(values)}
        for short in ("p", "u"):
            for scope in ("base_dot", "other_dot", "derived_total_dot", "cos_to_total"):
                key = f"{short}_{scope}"
                values = [float(r[key]) for r in selected if r[key] not in (None, "")]
                directions[key] = {"negative": sum(v < 0 for v in values),
                                   "positive": sum(v > 0 for v in values),
                                   "zero": sum(v == 0 for v in values),
                                   "available": len(values),
                                   "undefined_or_inactive": len(selected) - len(values)}
        result[group] = directions
    return result


def csv_payload(rows: list[dict], fields: list[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        require(not args.output_dir.exists(), f"Output already exists: {args.output_dir}")
        old_rows, old, old_items, old_freeze, old_layers = old_evidence()
        new_rows, reports, run_manifest, acceptance, new_freeze, new_layers = supplement_evidence()
        require(old_freeze + new_freeze == 2496 and old_layers + new_layers == 1200,
                f"Combined checks differ: freeze={old_freeze + new_freeze}, "
                f"layers={old_layers + new_layers}")
        fields = old["fields"]
        require(all(set(row) == set(fields) for row in new_rows),
                "Supplement observation fields differ from original CSV schema")
        combined = [*old_rows, *new_rows]
        require(len(combined) == 62 and len({(r["stage"], r["experiment"],
                str(r["checkpoint_step"]), str(r["source_offset"]), str(r["noise_seed"]))
                for r in combined}) == 62, "Combined observations are not 62 unique rows")
        run_manifest_path = ROOT / "outputs_diagnostics/gradient_seen3_temporal_extended/manifest.json"
        acceptance_path = ROOT / "outputs_diagnostics/grad_20261102_temporal_extended/offload_acceptance.json"
        require(read_json(run_manifest_path) == run_manifest and
                read_json(acceptance_path) == acceptance,
                "Supplement manifest or offload acceptance changed during validation")
        supplemental_csv = csv_payload(new_rows, fields)
        all_csv = csv_payload(combined, fields)
        supplemental_manifest = {"schema_version": 1, "complete": True,
            "scope": "Four B1/B2 step1200 offsets128/256 reports, two seeds each; CPU JSON export",
            "reports": reports, "report_count": 4, "observation_count": 8,
            "freeze_checks": new_freeze, "language_layer_observations": new_layers,
            "frozen_input_hashes": run_manifest["input_hashes"],
            "original_evidence_manifest_sha256": old["manifest_sha256"],
            "original_observations_csv_sha256": old["csv_sha256"],
            "supplement_run_manifest_sha256": sha256(run_manifest_path),
            "offload_acceptance_sha256": sha256(acceptance_path),
            "offload_acceptance": acceptance}
        final = {"passed": True, "cpu_only": True,
                 "counts": {"original_reports": len(old_items), "supplement_reports": len(reports),
                            "original_observations": len(old_rows), "supplement_observations": len(new_rows),
                            "global_observations": len(combined),
                            "original_freeze_checks": old_freeze, "supplement_freeze_checks": new_freeze,
                            "global_freeze_checks": old_freeze + new_freeze,
                            "original_language_layer_observations": old_layers,
                            "supplement_language_layer_observations": new_layers,
                            "global_language_layer_observations": old_layers + new_layers},
                 "direction_counts": direction_counts(combined),
                 "direction_note": "Primary first-order values are negative for descent and positive for ascent. Auxiliary P/U own-direction dot and cosine signs are reported without assuming positive alignment.",
                 "original_54_rows_preserved": True,
                 "supplement_stage": "temporal_extended",
                 "no_training_or_gradient_recomputation": True}
        payloads = {"supplemental_observations.csv": supplemental_csv,
                    "all_observations.csv": all_csv,
                    "supplemental_evidence_manifest.json": json.dumps(supplemental_manifest, indent=2, allow_nan=False) + "\n",
                    "final_acceptance.json": json.dumps(final, indent=2, allow_nan=False) + "\n",
                    "offload_acceptance.json": acceptance_path.read_text(encoding="utf-8"),
                    "supplement_run_manifest.json": run_manifest_path.read_text(encoding="utf-8")}
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staged = Path(tempfile.mkdtemp(prefix=".temporal_extended.", dir=args.output_dir.parent))
        try:
            for name, payload in payloads.items():
                (staged / name).write_text(payload, encoding="utf-8")
            require(not args.output_dir.exists(), f"Output appeared during validation: {args.output_dir}")
            staged.rename(args.output_dir)
        finally:
            if staged.exists():
                shutil.rmtree(staged)
        print(f"accepted 4 supplement reports, 8 new rows, 62 total rows -> {args.output_dir}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"not accepted; outputs untouched: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
