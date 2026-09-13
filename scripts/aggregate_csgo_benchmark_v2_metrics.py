#!/usr/bin/env python3
"""Aggregate per-map CSGO Benchmark v2 metric JSON files.

This utility deliberately has no project, torch, or machine-learning
dependencies.  It operates on the JSON contract emitted by the Benchmark v2
evaluators and keeps two kinds of averaging separate:

* ``maps`` computes an equal-map macro over one result per protocol map.
* ``seeds`` computes uncertainty over support-set selections.  The seed axis
  is therefore labelled ``support_selection`` and must not be interpreted as
  independent model-training seeds.
* ``map-models`` computes an equal-map macro when each map was evaluated with
  a different checkpoint, while retaining that map-specific provenance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence


MAP_SPLITS = (
    "seen_discrete_test",
    "seen_continuous",
    "crossmap_query_test",
    "crossmap_continuous",
)
MAP_MODEL_SPLITS = ("crossmap_query_test", "crossmap_continuous")
KIND_VALUES = ("discrete", "continuous")
LOCALIZATION_KIND = "localization"
SEED_KIND_VALUES = KIND_VALUES + (LOCALIZATION_KIND,)
ASSET_PROVENANCE_FIELDS = (
    "asset_backend",
    "asset_manifest_path",
    "asset_manifest_sha256",
    "selected_images_sha256",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Two-sided t critical values for a 95% interval, with df = n - 1.  The
# n=5 value is kept at the exact value required by the Benchmark v2 protocol.
T_CRITICAL_95 = {
    2: 12.706204736432095,
    3: 4.302652729911275,
    4: 3.182446305284263,
    5: 2.7764451051977987,
    6: 2.570581835636305,
    7: 2.446911851144,
    8: 2.3646242510102993,
    9: 2.306004135204166,
    10: 2.2621571628540993,
}


class AggregationError(ValueError):
    """Raised when an input does not satisfy the Benchmark v2 contract."""


def _resolve_file(value: str | os.PathLike[str], description: str) -> Path:
    try:
        path = Path(value).expanduser().resolve()
    except (TypeError, ValueError, OSError) as exc:
        raise AggregationError(f"invalid {description}: {value!r}") from exc
    if not path.is_file():
        raise AggregationError(f"{description} does not exist: {path}")
    return path


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AggregationError(f"could not read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AggregationError(f"{description} must be a JSON object: {path}")
    return value


def _load_manifest(manifest: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    path = _resolve_file(manifest, "manifest")
    return path, _read_json_object(path, "manifest")


def _protocol_maps(manifest: Mapping[str, Any], split: str) -> list[str]:
    if split not in MAP_SPLITS:
        raise AggregationError(
            f"unsupported split {split!r}; expected one of {', '.join(MAP_SPLITS)}"
        )
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise AggregationError("manifest.protocol must be a JSON object")
    key = "seen_maps" if split.startswith("seen_") else "crossmap_maps"
    maps = protocol.get(key)
    if not isinstance(maps, list) or any(
        not isinstance(map_name, str) or not map_name for map_name in maps
    ):
        raise AggregationError(f"manifest.protocol.{key} must be a list of map names")
    if len(set(maps)) != len(maps):
        raise AggregationError(f"manifest.protocol.{key} contains duplicate maps")
    return list(maps)


def _select_protocol_maps(
    manifest: Mapping[str, Any], split: str, maps: Sequence[str] | None
) -> list[str]:
    protocol_maps = _protocol_maps(manifest, split)
    if maps is None:
        return protocol_maps
    if isinstance(maps, (str, bytes)) or not isinstance(maps, Sequence):
        raise AggregationError("maps must be a sequence of map names")

    selected_maps = list(maps)
    if not selected_maps or any(
        not isinstance(map_name, str) or not map_name for map_name in selected_maps
    ):
        raise AggregationError("maps must be a non-empty sequence of map names")
    if len(set(selected_maps)) != len(selected_maps):
        raise AggregationError(
            f"maps must be unique; duplicate map name(s): {selected_maps!r}"
        )

    protocol_map_set = set(protocol_maps)
    unknown_maps = [
        map_name for map_name in selected_maps if map_name not in protocol_map_set
    ]
    if unknown_maps:
        raise AggregationError(
            f"maps contain unknown protocol map(s): {unknown_maps!r}"
        )

    expected_order = [
        map_name for map_name in protocol_maps if map_name in set(selected_maps)
    ]
    if selected_maps != expected_order:
        raise AggregationError(
            "maps must preserve manifest protocol order; "
            f"expected {expected_order!r}, got {selected_maps!r}"
        )
    return selected_maps


def _result_prefix(kind: str) -> str:
    if kind not in KIND_VALUES:
        raise AggregationError(
            f"unsupported kind {kind!r}; expected one of {', '.join(KIND_VALUES)}"
        )
    return "benchmark_csgo_v2_" if kind == "discrete" else "benchmark_csgo_v2_conti_"


def _result_filename(kind: str, map_name: str) -> str:
    return f"{_result_prefix(kind)}{map_name}.json"


def _discover_result_files(input_root: Path, kind: str) -> dict[str, Path]:
    """Discover files for one result kind and retain extra-map detection.

    Files for the other kind are intentionally ignored.  This permits one
    directory to contain both discrete and continuous evaluation outputs.
    """

    if not input_root.is_dir():
        raise AggregationError(f"input_root is not a directory: {input_root}")
    prefix = _result_prefix(kind)
    other_prefix = _result_prefix("continuous" if kind == "discrete" else "discrete")
    found: dict[str, Path] = {}
    try:
        entries = list(input_root.iterdir())
    except OSError as exc:
        raise AggregationError(f"could not list input_root {input_root}: {exc}") from exc
    for path in entries:
        if not path.is_file() or path.suffix != ".json":
            continue
        if kind == "discrete" and path.name.startswith(other_prefix):
            continue
        if not path.name.startswith(prefix):
            continue
        map_name = path.name[len(prefix) : -len(path.suffix)]
        if not map_name:
            raise AggregationError(f"invalid per-map result filename: {path.name}")
        if map_name in found:
            raise AggregationError(
                f"multiple per-map result files for map {map_name!r}: "
                f"{found[map_name]} and {path}"
            )
        found[map_name] = path.resolve()
    return found


def _validate_manifest_provenance(
    payload: Mapping[str, Any],
    *,
    path: Path,
    manifest_path: Path,
    map_name: str,
    split: str,
    kind: str,
    expected_maps: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if payload.get("map_name") != map_name:
        raise AggregationError(
            f"{path}: payload map_name {payload.get('map_name')!r} does not match "
            f"filename/protocol map {map_name!r}"
        )
    if payload.get("benchmark_v2_split") != split:
        raise AggregationError(
            f"{path}: payload benchmark_v2_split {payload.get('benchmark_v2_split')!r} "
            f"does not match requested split {split!r}"
        )
    raw_manifest = payload.get("benchmark_v2_manifest")
    if not isinstance(raw_manifest, str) or not raw_manifest:
        raise AggregationError(
            f"{path}: payload benchmark_v2_manifest must be a non-empty path"
        )
    try:
        payload_manifest = Path(raw_manifest).expanduser().resolve()
    except (TypeError, ValueError, OSError) as exc:
        raise AggregationError(
            f"{path}: invalid payload benchmark_v2_manifest {raw_manifest!r}"
        ) from exc
    if payload_manifest != manifest_path:
        raise AggregationError(
            f"{path}: manifest provenance mismatch; resolves to {payload_manifest}, "
            f"expected {manifest_path}"
        )
    for optional_kind_key in ("kind", "benchmark_v2_kind"):
        if optional_kind_key in payload and payload[optional_kind_key] != kind:
            raise AggregationError(
                f"{path}: {optional_kind_key} {payload[optional_kind_key]!r} "
                f"does not match requested kind {kind!r}"
            )
    metrics = payload.get("metrics_ordered")
    if not isinstance(metrics, dict):
        raise AggregationError(f"{path}: metrics_ordered must be a JSON object")
    checked_metrics = dict(metrics)
    # Validate every numeric value, including metrics that are not common to
    # all maps.  Otherwise a NaN/inf in a map-specific metric could be hidden
    # by the common-key intersection below.
    for metric, value in checked_metrics.items():
        _finite_number(value, path=path, metric=str(metric))
    provenance, provenance_context = _validate_inference_provenance(
        payload.get("inference_provenance"),
        path=path,
        manifest_path=manifest_path,
        split=split,
        expected_maps=expected_maps,
    )
    asset_context = _normalise_asset_provenance(payload, path=path)
    for field in ASSET_PROVENANCE_FIELDS:
        if asset_context[field] != provenance_context[field]:
            raise AggregationError(
                f"{path}: metric {field} does not match inference provenance"
            )
    return checked_metrics, provenance, provenance_context


def _finite_number(value: Any, *, path: Path, metric: str) -> float | None:
    """Return a finite number; bool and other non-numbers are non-metrics."""

    if isinstance(value, bool):
        return None
    if not isinstance(value, Real):
        return None
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise AggregationError(
            f"{path}: metric {metric!r} is not representable as a finite number"
        ) from exc
    if not math.isfinite(numeric):
        raise AggregationError(f"{path}: metric {metric!r} is NaN or infinite")
    return numeric


def _validate_map_names(value: Any, *, path: Path, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(map_name, str) or not map_name for map_name in value
    ):
        raise AggregationError(f"{path}: {field} must be a list of map names")
    if len(set(value)) != len(value):
        raise AggregationError(f"{path}: {field} contain duplicates")
    return list(value)


def validate_localization_result_coverage(
    results: Sequence[Mapping[str, Any]],
    expected_maps: Sequence[str],
    expected_entries: Sequence[Mapping[str, Any]],
) -> None:
    """Require localization results to match the expected dataset rows exactly."""

    if isinstance(expected_maps, (str, bytes)) or not isinstance(expected_maps, Sequence):
        raise AggregationError("expected localization maps must be a sequence")
    expected = list(expected_maps)
    if not expected or any(not isinstance(map_name, str) or not map_name for map_name in expected):
        raise AggregationError("expected localization maps must be non-empty names")
    if len(set(expected)) != len(expected):
        raise AggregationError("expected localization maps contain duplicates")
    if (
        isinstance(results, (str, bytes))
        or not isinstance(results, Sequence)
        or not results
    ):
        raise AggregationError("localization results must be non-empty")
    if (
        isinstance(expected_entries, (str, bytes))
        or not isinstance(expected_entries, Sequence)
        or not expected_entries
    ):
        raise AggregationError("expected localization entries must be non-empty")

    expected_keys: list[tuple[str, str]] = []
    expected_counts = {map_name: 0 for map_name in expected}
    for index, entry in enumerate(expected_entries):
        if not isinstance(entry, Mapping):
            raise AggregationError(
                f"expected localization entry {index} must be an object"
            )
        map_name = entry.get("map", entry.get("map_name"))
        file_frame = entry.get("file_frame", entry.get("id", entry.get("ids")))
        if not isinstance(map_name, str) or not map_name:
            raise AggregationError(
                f"expected localization entry {index} has no valid map"
            )
        if not isinstance(file_frame, str) or not file_frame:
            raise AggregationError(
                f"expected localization entry {index} has no valid file_frame"
            )
        if map_name not in expected_counts:
            raise AggregationError(
                f"expected localization entry {index} uses unexpected map {map_name!r}"
            )
        expected_keys.append((map_name, file_frame))
        expected_counts[map_name] += 1
    if len(expected_keys) != len(set(expected_keys)):
        raise AggregationError("expected localization entries contain duplicate (map,id)")
    missing_expected_maps = [
        map_name for map_name, count in expected_counts.items() if count == 0
    ]
    if missing_expected_maps:
        raise AggregationError(
            "expected localization entries do not cover every expected map: "
            f"{missing_expected_maps!r}"
        )
    if len(results) != len(expected_keys):
        raise AggregationError(
            "localization result count mismatch; "
            f"expected={len(expected_keys)}, got={len(results)}"
        )

    counts = {map_name: 0 for map_name in expected}
    actual_key_set: set[tuple[str, str]] = set()
    actual_counts = {map_name: 0 for map_name in expected}
    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise AggregationError(f"localization result {index} must be an object")
        map_name = result.get("map")
        file_frame = result.get("file_frame", result.get("id", result.get("ids")))
        if not isinstance(map_name, str) or not map_name:
            raise AggregationError(f"localization result {index} has no valid map")
        if not isinstance(file_frame, str) or not file_frame:
            raise AggregationError(
                f"localization result {index} has no valid file_frame"
            )
        key = (map_name, file_frame)
        if key in actual_key_set:
            raise AggregationError(
                "localization results contain duplicate (map,id): "
                f"{key!r}"
            )
        actual_key_set.add(key)
        if map_name not in counts:
            continue
        counts[map_name] += 1
        actual_counts[map_name] += 1

    missing = [map_name for map_name in expected if counts[map_name] == 0]
    expected_key_set = set(expected_keys)
    missing_keys = sorted(expected_key_set - actual_key_set)
    extra_keys = sorted(actual_key_set - expected_key_set)
    if (
        counts != expected_counts
        or actual_counts != expected_counts
        or missing
        or missing_keys
        or extra_keys
    ):
        raise AggregationError(
            "localization result map coverage mismatch; "
            f"expected_counts={expected_counts!r}, actual_counts={counts!r}, "
            f"missing_maps={missing!r}, missing_rows={missing_keys!r}, "
            f"extra_rows={extra_keys!r}, expected_maps={expected!r}"
        )


def _normalise_localization_per_map(
    per_map: Any, *, maps: Sequence[str], path: Path
) -> dict[str, dict[str, Any]]:
    if not isinstance(per_map, Mapping):
        raise AggregationError(f"{path}: per_map must be a JSON object")
    if set(per_map) != set(maps):
        raise AggregationError(
            f"{path}: per_map map coverage does not match maps; "
            f"expected {list(maps)!r}, got {sorted(per_map)!r}"
        )

    normalised: dict[str, dict[str, Any]] = {}
    for map_name in maps:
        metrics = per_map[map_name]
        if not isinstance(metrics, Mapping):
            raise AggregationError(
                f"{path}: per_map[{map_name!r}] must be a JSON object"
            )
        map_metrics: dict[str, Any] = {}
        for metric, value in metrics.items():
            if not isinstance(metric, str):
                raise AggregationError(
                    f"{path}: per_map metric names must be strings"
                )
            if metric == "ckpt_path":
                if value is not None and not isinstance(value, str):
                    raise AggregationError(
                        f"{path}: per_map[{map_name!r}].ckpt_path must be a string"
                    )
                map_metrics[metric] = value
                continue
            numeric = _finite_number(
                value,
                path=path,
                metric=f"per_map[{map_name}].{metric}",
            )
            if numeric is None:
                raise AggregationError(
                    f"{path}: per_map[{map_name!r}][{metric!r}] must be a finite number"
                )
            map_metrics[metric] = numeric
        normalised[map_name] = map_metrics
    return normalised


def _normalise_localization_macro(
    metrics_macro_map: Any, *, path: Path
) -> dict[str, float]:
    if not isinstance(metrics_macro_map, Mapping):
        raise AggregationError(f"{path}: metrics_macro_map must be a JSON object")
    normalised: dict[str, float] = {}
    for metric, value in metrics_macro_map.items():
        if not isinstance(metric, str):
            raise AggregationError(f"{path}: metric names must be strings")
        numeric = _finite_number(value, path=path, metric=metric)
        if numeric is None:
            raise AggregationError(
                f"{path}: metrics_macro_map[{metric!r}] must be numeric and non-bool"
            )
        normalised[metric] = numeric
    return normalised


def _normalise_optional_int(value: Any, *, path: Path, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise AggregationError(f"{path}: {field} must be an integer or null")
    return int(value)


def _nonempty_path_value(value: Any, *, path: Path, field: str) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise AggregationError(f"{path}: {field} must be a non-empty path")
    try:
        value_string = os.fspath(value)
    except TypeError as exc:
        raise AggregationError(f"{path}: {field} must be a non-empty path") from exc
    if not value_string or (isinstance(value_string, str) and not value_string.strip()):
        raise AggregationError(f"{path}: {field} must be a non-empty path")
    return value_string


def _asset_provenance_value(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    nested_values = [payload.get("asset_provenance"), payload.get("benchmark_v2_asset")]
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    for nested in nested_values:
        if isinstance(nested, Mapping):
            nested_keys = list(keys)
            if "asset_manifest_path" in keys:
                nested_keys.extend(("manifest", "path"))
            if "asset_manifest_sha256" in keys:
                nested_keys.extend(("manifest_sha256", "sha256"))
            for key in nested_keys:
                value = nested.get(key)
                if value not in (None, ""):
                    return value
    return None


def _assert_asset_aliases_consistent(
    payload: Mapping[str, Any], *, keys: Sequence[str], field: str, path: Path
) -> None:
    values: list[Any] = [payload.get(key) for key in keys if payload.get(key) not in (None, "")]
    for nested in (payload.get("asset_provenance"), payload.get("benchmark_v2_asset")):
        if isinstance(nested, Mapping):
            nested_keys = list(keys)
            if "asset_manifest_path" in keys:
                nested_keys.extend(("manifest", "path"))
            if "asset_manifest_sha256" in keys:
                nested_keys.extend(("manifest_sha256", "sha256"))
            values.extend(
                nested.get(key)
                for key in nested_keys
                if nested.get(key) not in (None, "")
            )
    if len({str(value) for value in values}) > 1:
        raise AggregationError(f"{path}: conflicting asset provenance aliases for {field}")


def _normalise_asset_provenance(
    payload: Mapping[str, Any], *, path: Path
) -> dict[str, str | None]:
    """Normalize the optional source/minimal asset backend contract.

    Artifacts created before the asset backend existed have no asset fields;
    those are explicitly interpreted as the historical ``source`` backend.
    Once ``minimal`` is declared, all identity fields are mandatory so an
    aggregate cannot silently mix different bundles.
    """

    alias_groups = {
        "asset_backend": ("asset_backend", "benchmark_v2_asset_backend", "backend"),
        "asset_manifest_path": (
            "asset_manifest_path",
            "benchmark_v2_asset_manifest",
            "asset_manifest",
        ),
        "asset_manifest_sha256": (
            "asset_manifest_sha256",
            "benchmark_v2_asset_manifest_sha256",
            "asset_manifest_hash",
        ),
        "selected_images_sha256": (
            "selected_images_sha256",
            "benchmark_v2_selected_images_sha256",
            "selected_image_sha256",
        ),
    }
    for field, keys in alias_groups.items():
        _assert_asset_aliases_consistent(payload, keys=keys, field=field, path=path)

    raw_backend = _asset_provenance_value(
        payload,
        alias_groups["asset_backend"],
    )
    raw_manifest = _asset_provenance_value(
        payload,
        alias_groups["asset_manifest_path"],
    )
    if raw_backend in (None, ""):
        raw_backend = "minimal" if raw_manifest not in (None, "") else "source"
    if not isinstance(raw_backend, str) or raw_backend not in {"source", "minimal"}:
        raise AggregationError(
            f"{path}: asset_backend must be 'source' or 'minimal', got {raw_backend!r}"
        )

    manifest_path: str | None = None
    if raw_manifest not in (None, ""):
        manifest_value = _nonempty_path_value(
            raw_manifest, path=path, field="asset_manifest_path"
        )
        try:
            manifest_path = str(Path(manifest_value).expanduser().resolve())
        except (TypeError, ValueError, OSError) as exc:
            raise AggregationError(
                f"{path}: invalid asset_manifest_path {manifest_value!r}"
            ) from exc

    raw_manifest_hash = _asset_provenance_value(
        payload,
        alias_groups["asset_manifest_sha256"],
    )
    raw_selected_hash = _asset_provenance_value(
        payload,
        alias_groups["selected_images_sha256"],
    )

    # The inference writer in older minimal runs records the verified report
    # path/hash but predates the selected-image checksum field.  Recover that
    # checksum from the same report so those artifacts remain aggregatable,
    # while still requiring a verified identity for the minimal backend.
    if raw_selected_hash in (None, "") and manifest_path:
        try:
            report = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            selected = report.get("selected_images") if isinstance(report, Mapping) else None
            if isinstance(selected, Mapping):
                raw_selected_hash = selected.get("sha256")
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass

    def normalise_hash(value: Any, field: str) -> str | None:
        if value in (None, ""):
            return None
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise AggregationError(
                f"{path}: {field} must be a lowercase SHA-256 digest"
            )
        return value

    manifest_hash = normalise_hash(raw_manifest_hash, "asset_manifest_sha256")
    selected_hash = normalise_hash(raw_selected_hash, "selected_images_sha256")
    if raw_backend == "source":
        if manifest_path is not None or manifest_hash is not None or selected_hash is not None:
            raise AggregationError(
                f"{path}: source asset backend must not carry minimal asset identity"
            )
    else:
        required_asset_values = {
            "asset_manifest_path": manifest_path,
            "asset_manifest_sha256": manifest_hash,
            "selected_images_sha256": selected_hash,
        }
        missing = [
            field for field, value in required_asset_values.items() if value is None
        ]
        if missing:
            raise AggregationError(
                f"{path}: minimal asset provenance is missing {', '.join(missing)}"
            )

    return {
        "asset_backend": raw_backend,
        "asset_manifest_path": manifest_path,
        "asset_manifest_sha256": manifest_hash,
        "selected_images_sha256": selected_hash,
    }


def _asset_provenance_output(context: Mapping[str, Any]) -> dict[str, Any]:
    """Emit both core and runner spellings in newly written summaries."""

    result = {field: context[field] for field in ASSET_PROVENANCE_FIELDS}
    result.update(
        {
            "benchmark_v2_asset_manifest": context["asset_manifest_path"],
            "benchmark_v2_asset_backend": context["asset_backend"],
            "benchmark_v2_asset_manifest_sha256": context["asset_manifest_sha256"],
            "benchmark_v2_selected_images_sha256": context["selected_images_sha256"],
        }
    )
    if context["asset_backend"] == "minimal":
        result["benchmark_v2_asset"] = {
            "manifest": context["asset_manifest_path"],
            "backend": context["asset_backend"],
            "sha256": context["asset_manifest_sha256"],
            "selected_images_sha256": context["selected_images_sha256"],
        }
    return result


def _required_int(value: Any, *, path: Path, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AggregationError(f"{path}: {field} must be an integer")
    return int(value)


def _validate_inference_payload(
    payload: Mapping[str, Any],
    *,
    path: Path,
    manifest_path: Path,
    split: str,
    expected_maps: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and normalize one evaluator inference-manifest payload."""

    if not isinstance(payload, Mapping):
        raise AggregationError(f"{path}: inference manifest payload must be an object")

    raw_manifest = payload.get("benchmark_v2_manifest")
    manifest_value = _nonempty_path_value(
        raw_manifest, path=path, field="benchmark_v2_manifest"
    )
    try:
        payload_manifest = Path(manifest_value).expanduser().resolve()
    except (TypeError, ValueError, OSError) as exc:
        raise AggregationError(
            f"{path}: invalid inference benchmark_v2_manifest {manifest_value!r}"
        ) from exc
    if payload_manifest != manifest_path:
        raise AggregationError(
            f"{path}: inference provenance manifest mismatch; "
            f"resolves to {payload_manifest}, expected {manifest_path}"
        )

    if payload.get("benchmark_v2_split") != split:
        raise AggregationError(
            f"{path}: inference provenance split {payload.get('benchmark_v2_split')!r} "
            f"does not match requested split {split!r}"
        )
    payload_maps = _validate_map_names(payload.get("maps"), path=path, field="inference provenance maps")
    expected_map_names = list(expected_maps)
    if payload_maps != expected_map_names:
        raise AggregationError(
            f"{path}: inference provenance maps do not match protocol order; "
            f"expected {expected_map_names!r}, got {payload_maps!r}"
        )

    checkpoint = _nonempty_path_value(payload.get("checkpoint"), path=path, field="checkpoint")
    ckpt_path = _nonempty_path_value(payload.get("ckpt_path"), path=path, field="ckpt_path")
    sample_count = _required_int(payload.get("sample_count"), path=path, field="sample_count")
    if sample_count <= 0:
        raise AggregationError(f"{path}: sample_count must be a positive integer")
    inference_seed = _required_int(payload.get("seed"), path=path, field="inference seed")
    if "benchmark_v2_support_seed" not in payload:
        raise AggregationError(
            f"{path}: inference provenance is missing benchmark_v2_support_seed"
        )
    if "benchmark_v2_shots_per_map" not in payload:
        raise AggregationError(
            f"{path}: inference provenance is missing benchmark_v2_shots_per_map"
        )
    support_seed = _normalise_optional_int(
        payload["benchmark_v2_support_seed"],
        path=path,
        field="benchmark_v2_support_seed",
    )
    shots_per_map = _normalise_optional_int(
        payload["benchmark_v2_shots_per_map"],
        path=path,
        field="benchmark_v2_shots_per_map",
    )
    if shots_per_map is not None and shots_per_map <= 0:
        raise AggregationError(
            f"{path}: benchmark_v2_shots_per_map must be positive or null"
        )
    asset_context = _normalise_asset_provenance(payload, path=path)

    normalized_payload = dict(payload)
    normalized_payload["benchmark_v2_manifest"] = str(manifest_path)
    normalized_payload["maps"] = payload_maps
    normalized_payload["checkpoint"] = checkpoint
    normalized_payload["ckpt_path"] = ckpt_path
    normalized_payload["sample_count"] = sample_count
    normalized_payload["seed"] = inference_seed
    normalized_payload["benchmark_v2_support_seed"] = support_seed
    normalized_payload["benchmark_v2_shots_per_map"] = shots_per_map
    normalized_payload.update(asset_context)
    return normalized_payload, {
        "checkpoint": checkpoint,
        "ckpt_path": ckpt_path,
        "sample_count": sample_count,
        "inference_seed": inference_seed,
        "support_seed": support_seed,
        "shots_per_map": shots_per_map,
        **_asset_provenance_output(asset_context),
    }


def _validate_inference_provenance(
    provenance: Any,
    *,
    path: Path,
    manifest_path: Path,
    split: str,
    expected_maps: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate an embedded provenance object and its referenced JSON file."""

    if not isinstance(provenance, Mapping):
        raise AggregationError(
            f"{path}: inference_provenance must contain path and payload"
        )
    raw_path = provenance.get("path", provenance.get("manifest_path"))
    provenance_path = _resolve_file(raw_path, "inference manifest")
    embedded_payload = provenance.get("payload")
    normalized_embedded, embedded_context = _validate_inference_payload(
        embedded_payload,
        path=path,
        manifest_path=manifest_path,
        split=split,
        expected_maps=expected_maps,
    )
    referenced_payload = _read_json_object(provenance_path, "inference manifest")
    normalized_referenced, referenced_context = _validate_inference_payload(
        referenced_payload,
        path=provenance_path,
        manifest_path=manifest_path,
        split=split,
        expected_maps=expected_maps,
    )
    if normalized_embedded != normalized_referenced:
        raise AggregationError(
            f"{path}: inference provenance payload does not match referenced "
            f"inference manifest {provenance_path}"
        )
    if embedded_context != referenced_context:
        raise AggregationError(
            f"{path}: inference provenance context does not match referenced "
            f"inference manifest {provenance_path}"
        )

    normalized_provenance = dict(provenance)
    normalized_provenance["path"] = str(provenance_path)
    normalized_provenance["payload"] = normalized_embedded
    return normalized_provenance, embedded_context


def build_localization_summary(
    *,
    manifest: str | os.PathLike[str],
    split: str,
    maps: Sequence[str],
    per_map: Mapping[str, Mapping[str, Any]],
    metrics_macro_map: Mapping[str, Any],
    inference_provenance: Mapping[str, Any],
    checkpoint: str | os.PathLike[str] | None,
    seed: int,
    support_seed: int | None,
    shots_per_map: int | None,
    sample_count: int,
    allow_protocol_map_subset: bool = False,
) -> dict[str, Any]:
    """Build the pure-JSON localization summary contract used by ``seeds``.

    By default, ``maps`` must be the complete protocol map list.  The
    optional subset mode is for map-specific evaluation artifacts and still
    requires a nonempty subset in the manifest's original order.
    """

    manifest_path = _resolve_file(manifest, "manifest")
    if split not in MAP_SPLITS:
        raise AggregationError(
            f"{manifest_path}: localization split is invalid: {split!r}"
        )
    _, manifest_data = _load_manifest(manifest_path)
    expected_protocol_maps = _protocol_maps(manifest_data, split)
    map_names = list(maps)
    if any(not isinstance(map_name, str) or not map_name for map_name in map_names):
        raise AggregationError("localization summary maps must contain names")
    if not map_names or len(set(map_names)) != len(map_names):
        raise AggregationError("localization summary maps must be unique and non-empty")
    if not isinstance(allow_protocol_map_subset, bool):
        raise AggregationError("allow_protocol_map_subset must be a boolean")
    if allow_protocol_map_subset:
        expected_map_set = set(expected_protocol_maps)
        unknown_maps = [
            map_name for map_name in map_names if map_name not in expected_map_set
        ]
        if unknown_maps:
            raise AggregationError(
                "localization summary maps contain unknown protocol map(s): "
                f"{unknown_maps!r}"
            )
        expected_order = [
            map_name for map_name in expected_protocol_maps if map_name in set(map_names)
        ]
        if map_names != expected_order:
            raise AggregationError(
                "localization summary map subset must preserve manifest protocol order; "
                f"expected {expected_order!r}, got {map_names!r}"
            )
    elif map_names != expected_protocol_maps:
        raise AggregationError(
            "localization summary maps do not match manifest protocol; "
            f"expected {expected_protocol_maps!r}, got {map_names!r}"
        )

    summary_path = Path("<localization-summary>")
    normalised_per_map = _normalise_localization_per_map(
        per_map, maps=map_names, path=summary_path
    )
    normalised_macro = _normalise_localization_macro(
        metrics_macro_map, path=summary_path
    )
    common_metrics = set(normalised_per_map[map_names[0]]) - {"ckpt_path"}
    for map_name in map_names[1:]:
        common_metrics.intersection_update(
            set(normalised_per_map[map_name]) - {"ckpt_path"}
        )
    if set(normalised_macro) != common_metrics:
        raise AggregationError(
            "localization summary metric set mismatch; "
            f"expected {sorted(common_metrics)!r}, "
            f"got {sorted(normalised_macro)!r}"
        )

    if not isinstance(inference_provenance, Mapping):
        raise AggregationError(
            "localization summary inference_provenance must be an object"
        )
    provenance = dict(inference_provenance)
    raw_provenance_path = provenance.get("path", provenance.get("manifest_path"))
    if not isinstance(raw_provenance_path, (str, os.PathLike)) or not os.fspath(
        raw_provenance_path
    ):
        raise AggregationError(
            "localization summary inference_provenance.path must be a path"
        )
    provenance["path"] = str(Path(raw_provenance_path).expanduser().resolve())
    if not Path(provenance["path"]).is_file():
        raise AggregationError(
            "localization summary inference_provenance.path does not exist: "
            f"{provenance['path']}"
        )

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AggregationError("localization summary seed must be an integer")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise AggregationError(
            "localization summary sample_count must be a positive integer"
        )
    normalised_support_seed = _normalise_optional_int(
        support_seed, path=summary_path, field="support_seed"
    )
    normalised_shots = _normalise_optional_int(
        shots_per_map, path=summary_path, field="shots_per_map"
    )
    if not isinstance(checkpoint, (str, os.PathLike)) or not os.fspath(checkpoint):
        raise AggregationError(
            "localization summary checkpoint must be a non-empty path"
        )
    checkpoint_value = os.fspath(checkpoint)
    if isinstance(checkpoint_value, str) and not checkpoint_value.strip():
        raise AggregationError(
            "localization summary checkpoint must be a non-empty path"
        )

    provenance_payload = provenance.get("payload")
    if not isinstance(provenance_payload, Mapping):
        raise AggregationError(
            "localization summary inference_provenance.payload must be an object"
        )
    payload_manifest = provenance_payload.get("benchmark_v2_manifest")
    if not isinstance(payload_manifest, (str, os.PathLike)) or not os.fspath(
        payload_manifest
    ):
        raise AggregationError(
            "localization summary provenance payload has no manifest"
        )
    if Path(payload_manifest).expanduser().resolve() != manifest_path:
        raise AggregationError(
            "localization summary provenance manifest does not match summary"
        )
    if provenance_payload.get("benchmark_v2_split") != split:
        raise AggregationError(
            "localization summary provenance split does not match summary"
        )
    if provenance_payload.get("maps") != map_names:
        raise AggregationError(
            "localization summary provenance maps do not match summary"
        )
    if provenance_payload.get("benchmark_v2_support_seed") != normalised_support_seed:
        raise AggregationError(
            "localization summary provenance support_seed does not match summary"
        )
    if provenance_payload.get("benchmark_v2_shots_per_map") != normalised_shots:
        raise AggregationError(
            "localization summary provenance shots_per_map does not match summary"
        )
    if provenance_payload.get("checkpoint") != checkpoint_value:
        raise AggregationError(
            "localization summary provenance checkpoint does not match summary"
        )
    if "ckpt_path" in provenance_payload and provenance_payload["ckpt_path"] != checkpoint_value:
        raise AggregationError(
            "localization summary provenance ckpt_path does not match summary"
        )
    if provenance_payload.get("seed") != int(seed):
        raise AggregationError(
            "localization summary provenance seed does not match summary"
        )
    if provenance_payload.get("sample_count") != int(sample_count):
        raise AggregationError(
            "localization summary provenance sample_count does not match summary"
        )

    asset_context = _normalise_asset_provenance(
        provenance_payload, path=summary_path
    )
    referenced_payload = _read_json_object(Path(provenance["path"]), "inference manifest")
    referenced_assets = _normalise_asset_provenance(
        referenced_payload, path=Path(provenance["path"])
    )
    if asset_context != referenced_assets:
        raise AggregationError(
            "localization summary asset provenance does not match referenced "
            f"inference manifest {provenance['path']}"
        )

    return {
        "manifest": str(manifest_path),
        "split": split,
        "kind": LOCALIZATION_KIND,
        "maps": map_names,
        "per_map": normalised_per_map,
        "metrics_macro_map": normalised_macro,
        "inference_provenance": provenance,
        "checkpoint": checkpoint_value,
        "seed": int(seed),
        "support_seed": normalised_support_seed,
        "shots_per_map": normalised_shots,
        "sample_count": int(sample_count),
        **_asset_provenance_output(asset_context),
    }


def _equal_map_macro(
    per_map: Mapping[str, Mapping[str, Any]],
    source_files: Mapping[str, Path],
) -> dict[str, float]:
    map_names = list(per_map)
    if not map_names:
        raise AggregationError("cannot aggregate an empty map set")
    common_metrics = set(per_map[map_names[0]])
    for map_name in map_names[1:]:
        common_metrics.intersection_update(per_map[map_name])

    # Preserve the first result's metric ordering in the JSON artifact.
    macro: dict[str, float] = {}
    for metric in per_map[map_names[0]]:
        if metric not in common_metrics:
            continue
        values = [
            _finite_number(
                per_map[map_name][metric],
                path=source_files[map_name],
                metric=metric,
            )
            for map_name in map_names
        ]
        # A bool or another nonnumeric value is not a numeric metric.  It is
        # excluded rather than accidentally coerced to 0/1.
        if all(value is not None for value in values):
            macro[metric] = sum(value for value in values if value is not None) / len(
                values
            )
    return macro


def _atomic_write_json(payload: Mapping[str, Any], output: str | os.PathLike[str]) -> Path:
    """Write JSON through a same-directory temporary file and replace."""

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, ensure_ascii=False, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return output_path


def _maps_output_path(input_root: Path, split: str, kind: str) -> Path:
    return input_root / f"benchmark_v2_{kind}_{split}_map_macro.json"


def aggregate_maps(
    *,
    manifest: str | os.PathLike[str],
    split: str,
    input_root: str | os.PathLike[str],
    kind: str,
    output: str | os.PathLike[str] | None = None,
    maps: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Aggregate one exact per-map result for every selected protocol map."""

    manifest_path, manifest_data = _load_manifest(manifest)
    expected_maps = _select_protocol_maps(manifest_data, split, maps)
    input_path = Path(input_root).expanduser().resolve()
    result_files = _discover_result_files(input_path, kind)
    expected_set = set(expected_maps)
    found_set = set(result_files)
    missing = [map_name for map_name in expected_maps if map_name not in found_set]
    extra = sorted(found_set - expected_set)
    if missing:
        raise AggregationError(
            f"missing per-map {kind} result(s) for {split}: {', '.join(missing)}"
        )
    if extra:
        raise AggregationError(
            f"extra per-map {kind} result(s) outside manifest protocol for {split}: "
            f"{', '.join(extra)}"
        )

    per_map: dict[str, dict[str, Any]] = {}
    source_files: dict[str, str] = {}
    source_paths: dict[str, Path] = {}
    common_inference_provenance: dict[str, Any] | None = None
    common_inference_context: dict[str, Any] | None = None
    for map_name in expected_maps:
        path = result_files[map_name]
        payload = _read_json_object(path, "per-map result")
        metrics, inference_provenance, inference_context = _validate_manifest_provenance(
            payload,
            path=path,
            manifest_path=manifest_path,
            map_name=map_name,
            split=split,
            kind=kind,
            expected_maps=expected_maps,
        )
        if common_inference_provenance is None:
            common_inference_provenance = inference_provenance
            common_inference_context = inference_context
        elif inference_provenance != common_inference_provenance:
            raise AggregationError(
                f"{path}: inference provenance path/payload does not match "
                "the other protocol maps"
            )
        per_map[map_name] = metrics
        source_files[map_name] = str(path)
        source_paths[map_name] = path

    if common_inference_provenance is None or common_inference_context is None:
        raise AggregationError("no inference provenance found in per-map results")
    metrics_macro_map = _equal_map_macro(per_map, source_paths)
    result: dict[str, Any] = {
        "manifest": str(manifest_path),
        "split": split,
        "kind": kind,
        "maps": expected_maps,
        "per_map": per_map,
        "metrics_macro_map": metrics_macro_map,
        "inference_provenance": common_inference_provenance,
        "checkpoint": common_inference_context["checkpoint"],
        "ckpt_path": common_inference_context["ckpt_path"],
        "inference_seed": common_inference_context["inference_seed"],
        "support_seed": common_inference_context["support_seed"],
        "shots_per_map": common_inference_context["shots_per_map"],
        "sample_count": common_inference_context["sample_count"],
        "source_files": source_files,
        **_asset_provenance_output(common_inference_context),
    }
    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else _maps_output_path(input_path, split, kind)
    )
    _atomic_write_json(result, output_path)
    return result


def _validate_maps_aggregate(
    payload: Mapping[str, Any],
    *,
    path: Path,
) -> tuple[str, str, str, list[str], dict[str, float], dict[str, Any]]:
    raw_manifest = payload.get("manifest")
    if not isinstance(raw_manifest, str) or not raw_manifest:
        raise AggregationError(f"{path}: aggregate manifest must be a non-empty path")
    manifest_path = _resolve_file(raw_manifest, "aggregate manifest")
    manifest = str(manifest_path)
    split = payload.get("split")
    kind = payload.get("kind")
    if not isinstance(split, str) or split not in MAP_SPLITS:
        raise AggregationError(f"{path}: aggregate split is invalid: {split!r}")
    if kind not in KIND_VALUES:
        raise AggregationError(f"{path}: aggregate kind is invalid: {kind!r}")
    maps = _validate_map_names(payload.get("maps"), path=path, field="aggregate maps")
    raw_metrics = payload.get("metrics_macro_map")
    if not isinstance(raw_metrics, dict):
        raise AggregationError(f"{path}: metrics_macro_map must be a JSON object")
    metrics: dict[str, float] = {}
    for metric, value in raw_metrics.items():
        if not isinstance(metric, str):
            raise AggregationError(f"{path}: metric names must be strings")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise AggregationError(
                f"{path}: metrics_macro_map[{metric!r}] must be numeric and non-bool"
            )
        numeric = _finite_number(value, path=path, metric=metric)
        if numeric is None:
            raise AggregationError(
                f"{path}: metrics_macro_map[{metric!r}] must be numeric and non-bool"
            )
        metrics[metric] = numeric
    required_context_fields = (
        "inference_provenance",
        "checkpoint",
        "ckpt_path",
        "inference_seed",
        "support_seed",
        "shots_per_map",
        "sample_count",
    )
    missing_context = [field for field in required_context_fields if field not in payload]
    if missing_context:
        raise AggregationError(
            f"{path}: maps aggregate missing inference context field(s): "
            f"{', '.join(missing_context)}"
        )
    inference_provenance, provenance_context = _validate_inference_provenance(
        payload["inference_provenance"],
        path=path,
        manifest_path=manifest_path,
        split=split,
        expected_maps=maps,
    )
    asset_context = _normalise_asset_provenance(payload, path=path)
    for field in ASSET_PROVENANCE_FIELDS:
        if asset_context[field] != provenance_context[field]:
            raise AggregationError(
                f"{path}: maps aggregate {field} does not match inference provenance"
            )
    checkpoint = _nonempty_path_value(payload["checkpoint"], path=path, field="checkpoint")
    ckpt_path = _nonempty_path_value(payload["ckpt_path"], path=path, field="ckpt_path")
    inference_seed = _required_int(payload["inference_seed"], path=path, field="inference_seed")
    support_seed = _normalise_optional_int(
        payload["support_seed"], path=path, field="support_seed"
    )
    shots_per_map = _normalise_optional_int(
        payload["shots_per_map"], path=path, field="shots_per_map"
    )
    if shots_per_map is not None and shots_per_map <= 0:
        raise AggregationError(f"{path}: shots_per_map must be positive or null")
    sample_count = _required_int(payload["sample_count"], path=path, field="sample_count")
    if sample_count <= 0:
        raise AggregationError(f"{path}: sample_count must be a positive integer")

    summary_context = {
        "checkpoint": checkpoint,
        "ckpt_path": ckpt_path,
        "inference_seed": inference_seed,
        "support_seed": support_seed,
        "shots_per_map": shots_per_map,
        "sample_count": sample_count,
    }
    expected_context = {
        "checkpoint": provenance_context["checkpoint"],
        "ckpt_path": provenance_context["ckpt_path"],
        "inference_seed": provenance_context["inference_seed"],
        "support_seed": provenance_context["support_seed"],
        "shots_per_map": provenance_context["shots_per_map"],
        "sample_count": provenance_context["sample_count"],
    }
    for field, expected in expected_context.items():
        if summary_context[field] != expected:
            raise AggregationError(
                f"{path}: inference provenance {field} does not match maps aggregate"
            )
    summary_context["inference_provenance"] = inference_provenance
    summary_context.update(asset_context)
    return manifest, split, kind, maps, metrics, summary_context


def _validate_localization_summary(
    payload: Mapping[str, Any],
    *,
    path: Path,
) -> tuple[str, str, str, list[str], dict[str, float], dict[str, str | None]]:
    required_fields = (
        "manifest",
        "split",
        "kind",
        "maps",
        "per_map",
        "metrics_macro_map",
        "inference_provenance",
        "checkpoint",
        "seed",
        "support_seed",
        "shots_per_map",
        "sample_count",
    )
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise AggregationError(
            f"{path}: localization summary missing required field(s): {', '.join(missing)}"
        )
    if payload.get("kind") != LOCALIZATION_KIND:
        raise AggregationError(
            f"{path}: localization summary kind must be {LOCALIZATION_KIND!r}"
        )

    summary = build_localization_summary(
        manifest=payload["manifest"],
        split=payload["split"],
        maps=payload["maps"],
        per_map=payload["per_map"],
        metrics_macro_map=payload["metrics_macro_map"],
        inference_provenance=payload["inference_provenance"],
        checkpoint=payload["checkpoint"],
        seed=payload["seed"],
        support_seed=payload["support_seed"],
        shots_per_map=payload["shots_per_map"],
        sample_count=payload["sample_count"],
    )
    payload_assets = _normalise_asset_provenance(payload, path=path)
    for field in ASSET_PROVENANCE_FIELDS:
        if payload_assets[field] != summary[field]:
            raise AggregationError(
                f"{path}: localization summary {field} does not match "
                "inference provenance"
            )
    return (
        summary["manifest"],
        summary["split"],
        summary["kind"],
        summary["maps"],
        summary["metrics_macro_map"],
        payload_assets,
    )


def _validate_localization_summary_for_map(
    payload: Mapping[str, Any],
    *,
    path: Path,
    manifest_path: Path,
    target_map: str,
    split: str,
) -> dict[str, Any]:
    """Strictly validate a localization summary containing one target map.

    The regular localization summary contract represents a complete protocol
    evaluation and therefore requires all protocol maps.  ``map-models``
    intentionally consumes one summary per target map, so it validates the
    same fields and provenance contract against ``[target_map]`` here.
    """

    required_fields = (
        "manifest",
        "split",
        "kind",
        "maps",
        "per_map",
        "metrics_macro_map",
        "inference_provenance",
        "checkpoint",
        "seed",
        "support_seed",
        "shots_per_map",
        "sample_count",
    )
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise AggregationError(
            f"{path}: localization summary missing required field(s): {', '.join(missing)}"
        )
    if payload.get("kind") != LOCALIZATION_KIND:
        raise AggregationError(
            f"{path}: localization summary kind must be {LOCALIZATION_KIND!r}"
        )

    raw_manifest = payload.get("manifest")
    summary_manifest = _resolve_file(raw_manifest, "localization summary manifest")
    if summary_manifest != manifest_path:
        raise AggregationError(
            f"{path}: localization summary manifest mismatch; resolves to "
            f"{summary_manifest}, expected {manifest_path}"
        )
    if payload.get("split") != split:
        raise AggregationError(
            f"{path}: localization summary split {payload.get('split')!r} does not "
            f"match requested split {split!r}"
        )

    maps = _validate_map_names(payload.get("maps"), path=path, field="summary maps")
    expected_maps = [target_map]
    if maps != expected_maps:
        raise AggregationError(
            f"{path}: localization summary maps do not match target map; "
            f"expected {expected_maps!r}, got {maps!r}"
        )

    normalised_per_map = _normalise_localization_per_map(
        payload["per_map"], maps=maps, path=path
    )
    normalised_macro = _normalise_localization_macro(
        payload["metrics_macro_map"], path=path
    )
    common_metrics = set(normalised_per_map[target_map]) - {"ckpt_path"}
    if set(normalised_macro) != common_metrics:
        raise AggregationError(
            f"{path}: localization summary metric set mismatch; "
            f"expected {sorted(common_metrics)!r}, got {sorted(normalised_macro)!r}"
        )

    provenance, provenance_context = _validate_inference_provenance(
        payload["inference_provenance"],
        path=path,
        manifest_path=manifest_path,
        split=split,
        expected_maps=expected_maps,
    )
    asset_context = _normalise_asset_provenance(payload, path=path)
    for field in ASSET_PROVENANCE_FIELDS:
        if asset_context[field] != provenance_context[field]:
            raise AggregationError(
                f"{path}: localization summary {field} does not match "
                "inference provenance"
            )
    checkpoint = _nonempty_path_value(
        payload["checkpoint"], path=path, field="checkpoint"
    )
    inference_seed = _required_int(payload["seed"], path=path, field="seed")
    support_seed = _normalise_optional_int(
        payload["support_seed"], path=path, field="support_seed"
    )
    shots_per_map = _normalise_optional_int(
        payload["shots_per_map"], path=path, field="shots_per_map"
    )
    if shots_per_map is not None and shots_per_map <= 0:
        raise AggregationError(f"{path}: shots_per_map must be positive or null")
    sample_count = _required_int(
        payload["sample_count"], path=path, field="sample_count"
    )
    if sample_count <= 0:
        raise AggregationError(f"{path}: sample_count must be a positive integer")

    summary_context = {
        "checkpoint": checkpoint,
        "ckpt_path": provenance_context["ckpt_path"],
        "inference_seed": inference_seed,
        "support_seed": support_seed,
        "shots_per_map": shots_per_map,
        "sample_count": sample_count,
        **asset_context,
    }
    expected_context = {
        "checkpoint": provenance_context["checkpoint"],
        "ckpt_path": provenance_context["ckpt_path"],
        "inference_seed": provenance_context["inference_seed"],
        "support_seed": provenance_context["support_seed"],
        "shots_per_map": provenance_context["shots_per_map"],
        "sample_count": provenance_context["sample_count"],
    }
    for field, expected in expected_context.items():
        if summary_context[field] != expected:
            raise AggregationError(
                f"{path}: localization summary {field} does not match "
                "inference provenance"
            )

    return {
        "manifest": str(manifest_path),
        "split": split,
        "kind": LOCALIZATION_KIND,
        "maps": maps,
        "per_map": normalised_per_map,
        "metrics_macro_map": normalised_macro,
        "inference_provenance": provenance,
        **summary_context,
    }


def _validate_seed_payload(
    payload: Mapping[str, Any], *, path: Path
) -> dict[str, Any]:
    if payload.get("kind") == LOCALIZATION_KIND:
        manifest, split, kind, maps, metrics, asset_context = _validate_localization_summary(
            payload, path=path
        )
        provenance_payload = payload["inference_provenance"]["payload"]
        return {
            "manifest": manifest,
            "split": split,
            "kind": kind,
            "maps": maps,
            "metrics": metrics,
            "inference_seed": payload["seed"],
            "support_seed": payload["support_seed"],
            "shots_per_map": payload["shots_per_map"],
            "checkpoint": payload["checkpoint"],
            "sample_count": payload["sample_count"],
            "inference_provenance": payload["inference_provenance"],
            "provenance_ckpt_path": provenance_payload.get("ckpt_path"),
            **asset_context,
        }
    manifest, split, kind, maps, metrics, context = _validate_maps_aggregate(
        payload, path=path
    )
    return {
        "manifest": manifest,
        "split": split,
        "kind": kind,
        "maps": maps,
        "metrics": metrics,
        **context,
    }


def _seed_result_path(pattern: str, seed: int) -> Path:
    if "{seed}" not in pattern:
        raise AggregationError(
            "--seed_root_pattern must contain the literal '{seed}' placeholder"
        )
    # Replace only the documented placeholder so unrelated braces in a path
    # are treated literally.
    rendered = pattern.replace("{seed}", str(seed))
    return Path(rendered).expanduser().resolve()


def _sample_std(values: Sequence[float], mean: float) -> float:
    denominator = len(values) - 1
    if denominator <= 0:
        raise AggregationError("sample standard deviation requires at least two seeds")
    return math.sqrt(sum((value - mean) ** 2 for value in values) / denominator)


def _seeds_output_path(first_input: Path, split: str, kind: str) -> Path:
    return first_input.parent / f"benchmark_v2_{kind}_{split}_support_selection.json"


def aggregate_seeds(
    *,
    seed_root_pattern: str,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    output: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Aggregate maps-level JSON files over support-set selection seeds."""

    if not isinstance(seed_root_pattern, str) or "{seed}" not in seed_root_pattern:
        raise AggregationError(
            "seed_root_pattern must be a string containing the literal '{seed}'"
        )
    seed_values = list(seeds)
    if not seed_values or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seed_values):
        raise AggregationError("seeds must be a non-empty sequence of integers")
    if len(set(seed_values)) != len(seed_values):
        raise AggregationError("seeds must not contain duplicates")
    if len(seed_values) not in T_CRITICAL_95:
        raise AggregationError(
            f"95% t critical lookup supports {min(T_CRITICAL_95)}..{max(T_CRITICAL_95)} "
            f"seeds; got {len(seed_values)}"
        )

    paths = [_seed_result_path(seed_root_pattern, seed) for seed in seed_values]
    if len(set(paths)) != len(paths):
        raise AggregationError("seed_root_pattern resolves multiple seeds to one file")
    aggregates: list[dict[str, Any]] = []
    for seed, path in zip(seed_values, paths):
        if not path.is_file():
            raise AggregationError(f"missing seed aggregate for seed {seed}: {path}")
        payload = _read_json_object(path, "seed aggregate")
        record = _validate_seed_payload(payload, path=path)
        if record["support_seed"] != seed:
            raise AggregationError(
                f"{path}: {record['kind']} summary support_seed "
                f"{record['support_seed']!r} does not match rendered seed {seed}"
            )
        shots_per_map = record["shots_per_map"]
        if (
            isinstance(shots_per_map, bool)
            or not isinstance(shots_per_map, int)
            or shots_per_map <= 0
        ):
            raise AggregationError(
                f"{path}: {record['kind']} summary shots_per_map must be a positive integer"
            )
        aggregates.append(record)

    reference = aggregates[0]
    reference_manifest = reference["manifest"]
    reference_split = reference["split"]
    reference_kind = reference["kind"]
    reference_maps = reference["maps"]
    reference_metrics = reference["metrics"]
    manifest_path, manifest_data = _load_manifest(reference_manifest)
    expected_maps = _protocol_maps(manifest_data, reference_split)
    if reference_maps != expected_maps:
        raise AggregationError(
            f"{paths[0]}: protocol map coverage does not match manifest "
            f"{manifest_path}; expected {expected_maps!r}, got {reference_maps!r}"
        )
    for seed, path, current in zip(seed_values[1:], paths[1:], aggregates[1:]):
        manifest = current["manifest"]
        split = current["split"]
        kind = current["kind"]
        maps = current["maps"]
        metrics = current["metrics"]
        if manifest != reference_manifest:
            raise AggregationError(
                f"{path}: manifest provenance mismatch for seed {seed}; "
                f"expected {reference_manifest}, got {manifest}"
            )
        if split != reference_split:
            raise AggregationError(
                f"{path}: split provenance mismatch for seed {seed}; "
                f"expected {reference_split!r}, got {split!r}"
            )
        if kind != reference_kind:
            raise AggregationError(
                f"{path}: kind provenance mismatch for seed {seed}; "
                f"expected {reference_kind!r}, got {kind!r}"
            )
        if maps != reference_maps:
            raise AggregationError(
                f"{path}: protocol map coverage mismatch for seed {seed}; "
                f"expected {reference_maps!r}, got {maps!r}"
            )
        if set(metrics) != set(reference_metrics):
            raise AggregationError(
                f"{path}: numeric macro metric set mismatch for seed {seed}; "
                f"expected {sorted(reference_metrics)!r}, got {sorted(metrics)!r}"
            )
        if current["inference_seed"] != reference["inference_seed"]:
            raise AggregationError(
                f"{path}: inference seed mismatch; expected "
                f"{reference['inference_seed']!r}, got {current['inference_seed']!r}"
            )
        if current["shots_per_map"] != reference["shots_per_map"]:
            raise AggregationError(
                f"{path}: shots_per_map mismatch for seed {seed}; "
                f"expected {reference['shots_per_map']!r}, "
                f"got {current['shots_per_map']!r}"
            )
        for field in ASSET_PROVENANCE_FIELDS:
            if current[field] != reference[field]:
                raise AggregationError(
                    f"{path}: {field} mismatch for seed {seed}; expected "
                    f"{reference[field]!r}, got {current[field]!r}"
                )

    t_critical = T_CRITICAL_95[len(seed_values)]
    metrics_support_selection: dict[str, dict[str, float | int]] = {}
    for metric in reference_metrics:
        values = [aggregate["metrics"][metric] for aggregate in aggregates]
        mean = sum(values) / len(values)
        sample_std = _sample_std(values, mean)
        standard_error = sample_std / math.sqrt(len(values))
        margin = t_critical * standard_error
        metrics_support_selection[metric] = {
            "mean": mean,
            "sample_std": sample_std,
            "standard_error": standard_error,
            "t_critical": t_critical,
            "ci95_low": mean - margin,
            "ci95_high": mean + margin,
            "n": len(values),
        }

    result: dict[str, Any] = {
        "manifest": reference_manifest,
        "split": reference_split,
        "kind": reference_kind,
        "maps": reference_maps,
        "seeds": seed_values,
        "inference_seed": reference["inference_seed"],
        "shots_per_map": reference["shots_per_map"],
        "uncertainty": "support_selection",
        "per_seed": {
            str(seed): aggregate["metrics"]
            for seed, aggregate in zip(seed_values, aggregates)
        },
        "metrics_support_selection": metrics_support_selection,
        **_asset_provenance_output(reference),
        "source_files": {
            str(seed): str(path) for seed, path in zip(seed_values, paths)
        },
    }
    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else _seeds_output_path(paths[0], reference_split, reference_kind)
    )
    _atomic_write_json(result, output_path)
    return result


def _map_model_input_paths(
    input_pattern: str | os.PathLike[str], maps: Sequence[str]
) -> dict[str, Path]:
    try:
        pattern = os.fspath(input_pattern)
    except TypeError as exc:
        raise AggregationError("--input_pattern must be a path containing '{map}'") from exc
    if not isinstance(pattern, str) or "{map}" not in pattern:
        raise AggregationError(
            "--input_pattern must contain the literal '{map}' placeholder"
        )

    paths: dict[str, Path] = {}
    for map_name in maps:
        rendered = pattern.replace("{map}", map_name)
        paths[map_name] = _resolve_file(
            rendered, f"map-models input for map {map_name!r}"
        )
    if len(set(paths.values())) != len(paths):
        raise AggregationError(
            "--input_pattern resolves multiple protocol maps to the same file"
        )
    return paths


def aggregate_map_models(
    *,
    manifest: str | os.PathLike[str],
    split: str,
    kind: str,
    input_pattern: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> dict[str, Any]:
    """Aggregate CrossMap results produced by one checkpoint per map.

    Unlike :func:`aggregate_maps`, this mode permits checkpoint and inference
    provenance to differ across maps.  The inference seed, support seed, shot
    count, manifest, and split remain protocol-wide invariants.  Checkpoint
    paths are preserved per map but are not required to be unique, since
    symlinked or shared storage is valid.
    """

    if output is None:
        raise AggregationError("--output is required for map-models")
    if split not in MAP_MODEL_SPLITS:
        raise AggregationError(
            f"unsupported map-models split {split!r}; expected one of "
            f"{', '.join(MAP_MODEL_SPLITS)}"
        )
    if kind not in SEED_KIND_VALUES:
        raise AggregationError(
            f"unsupported map-models kind {kind!r}; expected one of "
            f"{', '.join(SEED_KIND_VALUES)}"
        )
    expected_kind_split = {
        "discrete": "crossmap_query_test",
        "continuous": "crossmap_continuous",
        LOCALIZATION_KIND: "crossmap_query_test",
    }[kind]
    if split != expected_kind_split:
        raise AggregationError(
            f"map-models kind {kind!r} requires split "
            f"{expected_kind_split!r}; got {split!r}"
        )

    manifest_path, manifest_data = _load_manifest(manifest)
    expected_maps = _protocol_maps(manifest_data, split)
    if not expected_maps:
        raise AggregationError(
            f"manifest protocol has no CrossMap maps for split {split!r}"
        )
    input_paths = _map_model_input_paths(input_pattern, expected_maps)

    per_map: dict[str, dict[str, Any]] = {}
    model_context_by_map: dict[str, dict[str, Any]] = {}
    source_files: dict[str, str] = {}
    source_paths: dict[str, Path] = {}
    contexts: dict[str, dict[str, Any]] = {}

    for map_name in expected_maps:
        path = input_paths[map_name]
        payload = _read_json_object(path, "map-models input")
        if kind == LOCALIZATION_KIND:
            record = _validate_localization_summary_for_map(
                payload,
                path=path,
                manifest_path=manifest_path,
                target_map=map_name,
                split=split,
            )
            metrics = record["per_map"][map_name]
            inference_provenance = record["inference_provenance"]
            context = {
                "checkpoint": record["checkpoint"],
                "ckpt_path": record["ckpt_path"],
                "inference_seed": record["inference_seed"],
                "support_seed": record["support_seed"],
                "shots_per_map": record["shots_per_map"],
                "sample_count": record["sample_count"],
                **{
                    field: record[field]
                    for field in ASSET_PROVENANCE_FIELDS
                },
            }
        else:
            metrics, inference_provenance, context = _validate_manifest_provenance(
                payload,
                path=path,
                manifest_path=manifest_path,
                map_name=map_name,
                split=split,
                kind=kind,
                expected_maps=[map_name],
            )

        per_map[map_name] = metrics
        source_files[map_name] = str(path)
        source_paths[map_name] = path
        contexts[map_name] = context
        model_context_by_map[map_name] = {
            "checkpoint": context["checkpoint"],
            "ckpt_path": context["ckpt_path"],
            "inference_seed": context["inference_seed"],
            "support_seed": context["support_seed"],
            "shots_per_map": context["shots_per_map"],
            "sample_count": context["sample_count"],
            "inference_provenance": inference_provenance,
            **_asset_provenance_output(context),
        }

    reference_context = contexts[expected_maps[0]]
    for map_name in expected_maps[1:]:
        current_context = contexts[map_name]
        for field in (
            "inference_seed",
            "support_seed",
            "shots_per_map",
            *ASSET_PROVENANCE_FIELDS,
        ):
            if current_context[field] != reference_context[field]:
                raise AggregationError(
                    f"{input_paths[map_name]}: {field} mismatch across map-specific "
                    f"models; expected {reference_context[field]!r}, got "
                    f"{current_context[field]!r}"
                )

    metrics_macro_map = _equal_map_macro(per_map, source_paths)
    result: dict[str, Any] = {
        "aggregation": "map_specific_models",
        "manifest": str(manifest_path),
        "split": split,
        "kind": kind,
        "maps": expected_maps,
        "per_map": per_map,
        "metrics_macro_map": metrics_macro_map,
        "inference_seed": reference_context["inference_seed"],
        "support_seed": reference_context["support_seed"],
        "shots_per_map": reference_context["shots_per_map"],
        "model_context_by_map": model_context_by_map,
        "sample_count": sum(
            context["sample_count"] for context in contexts.values()
        ),
        **_asset_provenance_output(reference_context),
        "source_files": source_files,
    }
    output_path = Path(output).expanduser().resolve()
    _atomic_write_json(result, output_path)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate pure-JSON CSGO Benchmark v2 evaluation metrics."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    maps_parser = subparsers.add_parser(
        "maps", help="compute an equal-map macro from per-map results"
    )
    maps_parser.add_argument("--manifest", required=True)
    maps_parser.add_argument("--split", choices=MAP_SPLITS, required=True)
    maps_parser.add_argument("--input_root", required=True)
    maps_parser.add_argument("--kind", choices=KIND_VALUES, required=True)
    maps_parser.add_argument("--maps", nargs="+")
    maps_parser.add_argument("--output")

    seeds_parser = subparsers.add_parser(
        "seeds", help="compute support-selection mean and 95%% Student-t intervals"
    )
    seeds_parser.add_argument("--seed_root_pattern", required=True)
    seeds_parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    seeds_parser.add_argument("--output")

    map_models_parser = subparsers.add_parser(
        "map-models",
        help="compute an equal-map macro from map-specific model results",
    )
    map_models_parser.add_argument("--manifest", required=True)
    map_models_parser.add_argument("--split", choices=MAP_MODEL_SPLITS, required=True)
    map_models_parser.add_argument(
        "--kind", choices=SEED_KIND_VALUES, required=True
    )
    map_models_parser.add_argument("--input_pattern", required=True)
    map_models_parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "maps":
            result = aggregate_maps(
                manifest=args.manifest,
                split=args.split,
                input_root=args.input_root,
                kind=args.kind,
                output=args.output,
                maps=args.maps,
            )
            output_path = Path(args.output).expanduser().resolve() if args.output else _maps_output_path(Path(args.input_root).expanduser().resolve(), args.split, args.kind)
        elif args.command == "seeds":
            result = aggregate_seeds(
                seed_root_pattern=args.seed_root_pattern,
                seeds=args.seeds,
                output=args.output,
            )
            first_path = _seed_result_path(args.seed_root_pattern, args.seeds[0])
            output_path = Path(args.output).expanduser().resolve() if args.output else _seeds_output_path(first_path, result["split"], result["kind"])
        else:
            result = aggregate_map_models(
                manifest=args.manifest,
                split=args.split,
                kind=args.kind,
                input_pattern=args.input_pattern,
                output=args.output,
            )
            output_path = Path(args.output).expanduser().resolve()
    except (AggregationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
