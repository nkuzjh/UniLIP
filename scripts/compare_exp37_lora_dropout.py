#!/usr/bin/env python3
"""Read-only, single-seed comparison of exp37 LoRA dropout on and off."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "outputs_eval/benchmark_v2/exp37_s2_both/checkpoint-6000/test/seed_42/infer_bs8/discrete"
CANDIDATE = BASELINE.parent / "lora_dropout_off/discrete"
MAPS = ("cs_agency", "cs_italy", "de_ancient", "de_anubis", "de_dust2",
        "de_inferno", "de_mirage", "de_nuke", "de_overpass", "de_train")
METRICS = ("PSNR", "SSIM", "Boundary_F1", "LPIPS", "Pixel_MAE_255", "FID", "CLIP")
HIGHER = frozenset(("PSNR", "SSIM", "Boundary_F1", "CLIP"))
CHECKPOINT_ID = "outputs/csgo_1b/exp37_s2_both/checkpoint-6000/model.safetensors"
EXECUTION_EQUAL = ("schema_version", "task", "batch_size", "implementation", "rng_policy",
                   "sample_order_sha256", "sample_count", "batch_count", "last_batch_size",
                   "max_batch_size", "inference_steps", "rng_device", "rng_dtype")
ASSET_FIELDS = ("benchmark_v2_asset_backend", "benchmark_v2_asset_manifest_sha256",
                "benchmark_v2_selected_images_sha256")
SUMMARY_FIELDS = ("Coverage_GT", "Coverage_Pred", "Common_Count") + METRICS
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ComparisonError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"cannot read {path}: {exc}") from exc
    require(isinstance(data, dict), f"{path}: expected JSON object")
    return data


def normalized_path(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label}: missing path")
    # These anchors identify the same project asset across mounted servers.
    source = Path(value).expanduser()
    candidates = (source.resolve(), source) if source.exists() else (source,)
    for candidate in candidates:
        parts = candidate.parts
        for anchor in (("outputs_eval", "benchmark_v2"), ("outputs", "csgo_1b"),
                       ("data", "csgo_benchmark_v2"),
                       ("data", "preprocessed_data"), ("openpi-assets", "checkpoints")):
            for i in range(len(parts) - len(anchor) + 1):
                if parts[i:i + len(anchor)] == anchor:
                    return "/".join(parts[i:])
    return str(source.resolve()) if source.exists() else value


def checkpoint_id(value: Any, label: str) -> str:
    identity = normalized_path(value, label)
    require(identity == CHECKPOINT_ID, f"{label}: expected {CHECKPOINT_ID}, got {value!r}")
    return identity


def same_number(actual: Any, expected: Any, label: str) -> None:
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual), f"{label}: missing or non-finite number")
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9),
            f"{label}: {actual!r} != {expected!r}")


def semantic_config(path: Path, role: str) -> dict[str, Any]:
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ComparisonError(f"cannot read config {path}: {exc}") from exc
    require(isinstance(config, dict), f"{path}: config must be a mapping")
    require(config.get("disable_lora_dropout", False) is (role == "off"),
            f"{role}: config disable_lora_dropout does not match run mode")
    require(config.get("test_gen_batch_size") == 8, f"{path}: test_gen_batch_size must be 8")
    require(config.get("guidance_scale") == 4.5, f"{path}: guidance_scale must be 4.5")
    checkpoint_id(config.get("ckpt_path"), f"{path}: ckpt_path")
    config = dict(config)
    config.pop("disable_lora_dropout", None)
    for key, value in config.items():
        if isinstance(value, str) and (key.endswith(("_path", "_dir", "_root"))
                                       or key in ("benchmark_v2_manifest", "benchmark_v2_asset_manifest")):
            config[key] = normalized_path(value, f"{path}: {key}")
    return config


def semantic_metric_config(config: dict[str, Any], label: str) -> dict[str, Any]:
    normalized = dict(config)
    for key, value in normalized.items():
        if isinstance(value, str) and key.endswith(("_path", "_dir", "_root")):
            normalized[key] = normalized_path(value, f"{label}: {key}")
    return normalized


def validate_provenance(value: Any, manifest: dict[str, Any], root: Path, label: str) -> None:
    require(isinstance(value, dict), f"{label}: inference_provenance missing")
    path, payload = value.get("path"), value.get("payload")
    require(isinstance(path, str) and path.endswith("/inference_manifest.json"),
            f"{label}: invalid inference_provenance.path")
    require(normalized_path(path, f"{label}: provenance path") ==
            normalized_path(str(root / "inference_manifest.json"), f"{label}: local manifest"),
            f"{label}: inference_provenance.path points to another run")
    require(isinstance(payload, dict), f"{label}: inference_provenance.payload missing")
    # Existing summaries add resolved asset fields and use an absolute remote
    # benchmark manifest path; every overlapping field still has to agree.
    extra = {"asset_backend", "asset_manifest_path", "asset_manifest_sha256",
             "selected_images_sha256"}
    require(set(payload) - set(manifest) <= extra, f"{label}: unexpected provenance fields")
    require(set(manifest) <= set(payload), f"{label}: provenance omits manifest fields")
    for key in manifest:
        left, right = payload[key], manifest[key]
        if key in ("benchmark_v2_manifest", "config_path", "benchmark_v2_asset_manifest", "ckpt_path", "checkpoint"):
            require(normalized_path(left, f"{label}.{key}") == normalized_path(right, f"{label}.{key}"),
                    f"{label}: provenance {key} differs from manifest")
        elif key == "benchmark_v2_asset":
            require(isinstance(left, dict) and isinstance(right, dict), f"{label}: asset object missing")
            for asset_key, expected in right.items():
                actual = left.get(asset_key)
                if asset_key == "manifest":
                    require(normalized_path(actual, f"{label}.asset.manifest") ==
                            normalized_path(expected, f"{label}.asset.manifest"),
                            f"{label}: asset manifest differs")
                else:
                    require(actual == expected, f"{label}: asset {asset_key} differs")
        else:
            require(left == right, f"{label}: provenance {key} differs from manifest")
    for key, manifest_key in (("asset_backend", "benchmark_v2_asset_backend"),
                              ("asset_manifest_sha256", "benchmark_v2_asset_manifest_sha256"),
                              ("selected_images_sha256", "benchmark_v2_selected_images_sha256")):
        if key in payload:
            require(payload[key] == manifest[manifest_key], f"{label}: {key} differs")
    if "asset_manifest_path" in payload:
        require(normalized_path(payload["asset_manifest_path"], label) ==
                normalized_path(manifest["benchmark_v2_asset_manifest"], label),
                f"{label}: asset_manifest_path differs")


def validate_dropout(execution: dict[str, Any], role: str) -> list[dict[str, Any]]:
    expected_policy = ("preserve_loaded_module_modes_v1" if role == "on"
                       else "disable_lora_dropout_v1")
    require(execution.get("model_mode_policy") == expected_policy,
            f"{role}: model_mode_policy must be {expected_policy}")
    active = execution.get("active_dropout_modules")
    require(isinstance(active, list), f"{role}: active_dropout_modules missing")
    modules = active if role == "on" else execution.get("disabled_lora_dropout_modules")
    require(isinstance(modules, list) and len(modules) == 438,
            f"{role}: expected 438 LoRA dropout modules")
    names = set()
    for item in modules:
        require(isinstance(item, dict) and set(item) == {"name", "p"},
                f"{role}: malformed dropout module")
        name, p = item["name"], item["p"]
        require(isinstance(name, str) and ".lora_dropout." in name and name not in names,
                f"{role}: invalid or duplicate LoRA module {name!r}")
        same_number(p, 0.05, f"{role}: {name}.p")
        names.add(name)
    if role == "on":
        require(execution.get("stochastic_layers_batch_dependent") is True,
                "on: expected batch-dependent stochastic layers")
    else:
        require(active == [], "off: active_dropout_modules must be empty")
        require(execution.get("stochastic_layers_batch_dependent") is False,
                "off: stochastic_layers_batch_dependent must be false")
    return modules


def load_run(root: Path, role: str) -> dict[str, Any]:
    require(root.is_dir(), f"{role} result unfinished: missing directory {root}")
    manifest = read_json(root / "inference_manifest.json")
    summary = read_json(root / "summary.json")
    for key in ("checkpoint", "ckpt_path"):
        checkpoint_id(manifest.get(key), f"{role} manifest.{key}")
        checkpoint_id(summary.get(key), f"{role} summary.{key}")
    require(manifest.get("benchmark_v2_split") == summary.get("split") == "seen_discrete_test",
            f"{role}: split must be seen_discrete_test")
    require(summary.get("kind") == "discrete", f"{role}: kind must be discrete")
    require(manifest.get("maps") == summary.get("maps") == list(MAPS),
            f"{role}: expected all ten protocol maps in order")
    require(manifest.get("sample_count") == summary.get("sample_count") == 20000,
            f"{role}: sample_count must be 20000")
    require(manifest.get("seed") == summary.get("inference_seed") == 42,
            f"{role}: inference seed must be 42")
    for key in ASSET_FIELDS:
        require(isinstance(manifest.get(key), str) and manifest[key] == summary.get(key),
                f"{role}: {key} missing or differs")
    for key in ASSET_FIELDS[1:]:
        require(bool(SHA256.fullmatch(manifest[key])), f"{role}: invalid {key}")
    require(manifest.get("benchmark_v2_asset_backend") == "minimal",
            f"{role}: unexpected asset backend")
    asset = manifest.get("benchmark_v2_asset")
    require(isinstance(asset, dict), f"{role}: asset object missing")
    for key, asset_key in (("benchmark_v2_asset_backend", "backend"),
                           ("benchmark_v2_asset_manifest_sha256", "sha256"),
                           ("benchmark_v2_selected_images_sha256", "selected_images_sha256")):
        require(asset.get(asset_key) == manifest[key], f"{role}: asset {asset_key} differs")
    summary_asset = summary.get("benchmark_v2_asset")
    if summary_asset is not None:
        require(isinstance(summary_asset, dict), f"{role}: summary asset object malformed")
        for asset_key in ("backend", "sha256", "selected_images_sha256"):
            require(summary_asset.get(asset_key) == asset.get(asset_key),
                    f"{role}: summary asset {asset_key} differs")
    validate_provenance(summary.get("inference_provenance"), manifest, root, f"{role} summary")
    execution = manifest.get("inference_execution")
    require(isinstance(execution, dict), f"{role}: inference_execution missing")
    require(execution.get("sample_count") == 20000 and execution.get("batch_size") == 8
            and execution.get("inference_steps") == 20, f"{role}: execution count/batch/steps mismatch")
    require(execution.get("batch_count") == 2500 and execution.get("last_batch_size") == 8
            and execution.get("max_batch_size") == 8, f"{role}: execution batch accounting mismatch")
    require(isinstance(execution.get("sample_order_sha256"), str)
            and bool(SHA256.fullmatch(execution["sample_order_sha256"])),
            f"{role}: invalid sample_order_sha256")
    modules = validate_dropout(execution, role)
    config = semantic_config(root / "evaluation_config.yaml", role)
    expected_files = {f"benchmark_csgo_v2_{name}.json" for name in MAPS}
    actual_files = {p.name for p in root.glob("benchmark_csgo_v2_*.json")}
    require(actual_files == expected_files,
            f"{role}: per-map JSON coverage mismatch; missing={sorted(expected_files - actual_files)}, extra={sorted(actual_files - expected_files)}")
    per_map = summary.get("per_map")
    macro = summary.get("metrics_macro_map")
    require(isinstance(per_map, dict) and set(per_map) == set(MAPS), f"{role}: summary per_map coverage mismatch")
    require(isinstance(macro, dict), f"{role}: summary metrics_macro_map missing")
    map_config = None
    map_profile = None
    map_order = None
    values: dict[str, dict[str, float]] = {}
    for name in MAPS:
        result = read_json(root / f"benchmark_csgo_v2_{name}.json")
        label = f"{role} {name}"
        require(result.get("map_name") == name and result.get("benchmark_v2_split") == "seen_discrete_test",
                f"{label}: map or split mismatch")
        validate_provenance(result.get("inference_provenance"), manifest, root, label)
        for key, result_key in (("benchmark_v2_asset_backend", "benchmark_v2_asset_backend"),
                                ("benchmark_v2_asset_manifest_sha256", "benchmark_v2_asset_manifest_sha256"),
                                ("benchmark_v2_selected_images_sha256", "benchmark_v2_selected_images_sha256")):
            require(result.get(result_key) == manifest[key], f"{label}: {result_key} mismatch")
        for key in ("gt_count", "pred_count", "common_count"):
            require(result.get(key) == 2000, f"{label}: {key} must be 2000")
        require(result.get("missing_pred_files_count") == 0 and
                result.get("unmatched_pred_files_count") == 0 and
                result.get("missing_pred_files") == [] and
                result.get("unmatched_pred_files") == [], f"{label}: incomplete prediction coverage")
        selection = result.get("benchmark_v2_selection_counts")
        require(isinstance(selection, dict) and selection.get("rows") == 2000 and
                selection.get("expected_stems") == 2000, f"{label}: selection coverage mismatch")
        profile = result.get("metric_profile")
        require(profile == "benchmark_v2_core", f"{label}: metric_profile mismatch")
        metric_order = result.get("metrics_order")
        require(isinstance(metric_order, list) and all(k in metric_order for k in METRICS),
                f"{label}: metrics_order missing target metric")
        require(map_profile in (None, profile) and map_order in (None, metric_order),
                f"{label}: metric protocol differs between maps")
        map_profile, map_order = profile, metric_order
        config_result = result.get("config")
        require(isinstance(config_result, dict) and config_result.get("paired_size") == 448
                and config_result.get("batch_size") == 1 and config_result.get("device") == "cuda",
                f"{label}: metric paired_size/batch/device mismatch")
        config_result = semantic_metric_config(config_result, label)
        require(map_config in (None, config_result), f"{label}: metric config differs between maps")
        map_config = config_result
        raw = result.get("metrics_ordered")
        require(isinstance(raw, dict), f"{label}: metrics_ordered missing")
        values[name] = {}
        for metric in SUMMARY_FIELDS:
            value = raw.get(metric)
            require(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value), f"{label}: {metric} missing/non-finite")
            if metric in ("Coverage_GT", "Coverage_Pred"):
                same_number(value, 1.0, f"{label}.{metric}")
            if metric == "Common_Count":
                same_number(value, 2000, f"{label}.{metric}")
            same_number(per_map[name].get(metric) if isinstance(per_map[name], dict) else None,
                        value, f"{label}: summary.{metric}")
            values[name][metric] = float(value)
    for metric in SUMMARY_FIELDS:
        same_number(macro.get(metric), sum(values[name][metric] for name in MAPS) / 10,
                    f"{role}: macro.{metric}")
    return {"root": root, "manifest": manifest, "execution": execution, "modules": modules,
            "config": config, "metric_config": map_config, "metric_profile": map_profile,
            "metric_order": map_order, "values": values}


def compare(baseline: Path, candidate: Path) -> dict[str, Any]:
    on = load_run(baseline, "on")
    off = load_run(candidate, "off")
    for key in EXECUTION_EQUAL:
        require(on["execution"].get(key) == off["execution"].get(key),
                f"inference_execution.{key} differs")
    require(on["modules"] == off["modules"], "disabled LoRA list differs from on active list")
    for key in ASSET_FIELDS:
        require(on["manifest"][key] == off["manifest"][key], f"asset provenance {key} differs")
    for key in ("benchmark_v2_manifest", "benchmark_v2_asset_manifest"):
        require(normalized_path(on["manifest"].get(key), f"on {key}") ==
                normalized_path(off["manifest"].get(key), f"off {key}"),
                f"{key} differs")
    require(on["config"] == off["config"], "evaluation_config semantics differ beyond disable_lora_dropout")
    require(on["metric_config"] == off["metric_config"] and
            on["metric_profile"] == off["metric_profile"] and
            on["metric_order"] == off["metric_order"], "metric evaluation protocol differs")
    rows = {}
    for name in ("macro",) + MAPS:
        rows[name] = {}
        for metric in METRICS:
            a = (sum(on["values"][m][metric] for m in MAPS) / 10 if name == "macro"
                 else on["values"][name][metric])
            b = (sum(off["values"][m][metric] for m in MAPS) / 10 if name == "macro"
                 else off["values"][name][metric])
            delta = b - a
            improvement = delta if metric in HIGHER else -delta
            rows[name][metric] = {"on": a, "off": b, "off_minus_on": delta,
                                  "direction": "improved" if improvement > 1e-12 else
                                  "worsened" if improvement < -1e-12 else "unchanged"}
    return {"status": "complete", "scope": "single inference seed; descriptive comparison only",
            "baseline": str(baseline), "candidate": str(candidate), "split": "seen_discrete_test",
            "sample_count": 20000, "seed": 42, "checkpoint": CHECKPOINT_ID,
            "dropout_modules": 438, "metrics": list(METRICS), "maps": list(MAPS), "results": rows,
            "limitations": ["One inference seed; differences are descriptive, not statistical significance.",
                            "The original run has no checkpoint weight hash or complete GPU/software environment versions; matching checkpoint path and recorded settings cannot exclude those differences."]}


def markdown(report: dict[str, Any]) -> str:
    lines = ["# exp37_s2_both LoRA dropout comparison", "",
             "Single inference seed (42), 20,000 paired test samples, ten-map equal-weight macro. Differences are descriptive; no statistical significance claim.", "",
             f"- Baseline (dropout on): `{report['baseline']}`",
             f"- Candidate (dropout off): `{report['candidate']}`",
             f"- Checkpoint: `{report['checkpoint']}`; LoRA dropout modules: {report['dropout_modules']}", "",
             "`Δ = off − on`. Higher is better for PSNR, SSIM, Boundary_F1, CLIP; lower is better for LPIPS, Pixel_MAE_255, FID.", ""]
    for metric in METRICS:
        lines += [f"## {metric}", "", "| Map | On | Off | Δ off − on | Direction |",
                  "| --- | ---: | ---: | ---: | --- |"]
        for name in ("macro",) + MAPS:
            row = report["results"][name][metric]
            lines.append(f"| {'**Macro**' if name == 'macro' else name} | {row['on']:.6f} | {row['off']:.6f} | {row['off_minus_on']:+.6f} | {row['direction']} |")
        lines.append("")
    lines += ["## Provenance limits", ""] + [f"- {item}" for item in report["limitations"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--output-dir", type=Path, help="write comparison.json and comparison.md here")
    args = parser.parse_args(argv)
    baseline, candidate = args.baseline.expanduser().resolve(), args.candidate.expanduser().resolve()
    try:
        report = compare(baseline, candidate)
        output = markdown(report)
        if args.output_dir:
            destination = args.output_dir.expanduser().resolve()
            require(destination != baseline and baseline not in destination.parents and
                    destination != candidate and candidate not in destination.parents,
                    "output-dir cannot be inside either result directory")
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            (destination / "comparison.md").write_text(output, encoding="utf-8")
        print(output, end="")
    except ComparisonError as exc:
        print(f"comparison unavailable: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
