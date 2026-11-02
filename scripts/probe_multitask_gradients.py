#!/usr/bin/env python3
"""Measure checkpoint multitask gradients on a fixed Seen-3 training source pool.

Examples:
  python scripts/probe_multitask_gradients.py --checkpoint-dir outputs/.../checkpoint-1800 \
      --experiment exp32_3maps_stage2_both_j8 --stage screen --output /tmp/probe.json
  python scripts/probe_multitask_gradients.py --checkpoint-dir outputs/.../checkpoint-1800 \
      --stage effective --microbatch-size 4 --accumulations 32 --output /tmp/effective.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unilip.train.gradient_diagnostics import (
    TASKS, ParameterCoordinate, check_total_backward, measure_gradients,
    select_coordinate,
)


MAPS = ("de_ancient", "de_dust2", "de_nuke")
ROUTE_GROUPS = {
    "shared_language": r"(^|\.)language_model\.",
    "gen_private": r"(^|\.)(llm_connector|dit|projector|latent_queries)(\.|$)",
    "gen_connector": r"(^|\.)llm_connector\.",
    "gen_dit": r"(^|\.)dit\.",
    "gen_projector_queries": r"(^|\.)(projector|latent_queries)(\.|$)",
    "loc_private": r"(^|\.)(action_dit|action_out_proj|action_in_proj|action_dit_connector|action_dit_norm|action_dit_projector|time_mlp_in|time_mlp_out)(\.|$)",
    "loc_action_dit": r"(^|\.)action_dit\.",
    "loc_output": r"(^|\.)action_out_proj\.",
    "loc_connector": r"(^|\.)action_dit_connector\.",
    "loc_norm_projector_io": r"(^|\.)(action_dit_norm|action_dit_projector|action_in_proj|time_mlp_in|time_mlp_out)\.",
}
PARTITION_GROUPS = {
    "shared_language": ROUTE_GROUPS["shared_language"],
    "gen_private": ROUTE_GROUPS["gen_private"],
    "loc_private": ROUTE_GROUPS["loc_private"],
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--experiment", help="Assert the resolved experiment name")
    parser.add_argument("--checkpoint-step", type=int, help="Assert checkpoint step")
    parser.add_argument("--stage", choices=("screen", "effective", "pool", "custom"), default="screen")
    parser.add_argument("--source-seed", type=int, default=20261102)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[17, 29])
    parser.add_argument("--samples-per-map", type=int, default=128,
                        help="Fixed source pool size per map; default 128")
    parser.add_argument("--source-samples", type=int,
                        help="For custom stage, sample count drawn from the fixed source pool")
    parser.add_argument("--source-offset", type=int, default=0,
                        help="Offset into interleaved source pool for effective/custom windows")
    parser.add_argument("--source-indices", type=int, nargs="+",
                        help="Explicit dataset indices; override staged source sampling for exact replay")
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--accumulations", type=int,
                        help="Limit to this many microbatches; effective stage defaults to 32")
    parser.add_argument("--max-source-samples", type=int,
                        help="Limit source samples for staged smoke checks")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--coordinate", choices=("shared", "route", "custom"), default="shared")
    parser.add_argument("--param-pattern", action="append", default=[],
                        help="Regex for fixed trainable parameters; required for custom coordinate")
    parser.add_argument("--accumulator-device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-fraction", type=float,
                        help="CUDA allocator fraction of total device memory for this probe process")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp32-reference", action="store_true",
                        help="Probe-only FP32 model and eager attention after strict BF16 checkpoint loading")
    parser.add_argument("--deterministic-probe", "--deterministic-algorithms",
                        dest="deterministic_probe", action="store_true",
                        help="Require deterministic kernels and math SDPA in this probe process")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-aux-freeze", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-total-backward", action="store_true",
                        help="On the first microbatch compare autograd.grad with total.backward")
    parser.add_argument("--measurement-mode", choices=("full", "component_sum"), default="full",
                        help="Direct VJPs throughout, or checked component sum for FP32 deterministic windows")
    parser.add_argument("--numerical-check", action="store_true",
                        help="Repeat total VJP on the first graph to check recomputation stability")
    parser.add_argument("--save-gradient-vectors", action="store_true",
                        help="Save averaged FP32 vectors per seed as .pt alongside the JSON report")
    parser.add_argument("--linearity-rtol", type=float, default=2e-3)
    parser.add_argument("--linearity-atol", type=float, default=1e-5)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def source_indices(source_rows_by_map, *, source_seed: int, samples_per_map: int,
                   stage: str, source_samples: int | None, max_source_samples: int | None,
                   source_offset: int = 0) -> list[int]:
    """Build one stable 384-source pool, then stage-specific prefixes of it."""
    if samples_per_map < 1:
        raise ValueError("samples_per_map must be positive")
    rng = random.Random(source_seed)
    per_map = {}
    for map_name in MAPS:
        rows = source_rows_by_map[map_name]
        if rows and isinstance(rows[0], dict):
            # Dataset construction may shuffle its internal indices. Sample
            # stable source identities so different recipes see the same frames.
            identities = [(str(row["file_frame"]), str(row.get("row_id", ""))) for row in rows]
            if len(identities) != len(set(identities)):
                raise ValueError(f"Duplicate source identities on {map_name}")
            available = [int(row["dataset_index"]) for row in sorted(
                rows, key=lambda row: (str(row["file_frame"]), str(row.get("row_id", ""))))]
        else:
            available = sorted(set(int(i) for i in rows))
        if len(available) < samples_per_map:
            raise ValueError(f"{map_name}: need {samples_per_map} sources, found {len(available)}")
        per_map[map_name] = rng.sample(available, samples_per_map)
    if stage == "screen":
        if source_offset:
            raise ValueError("screen stage has a fixed source window; source_offset must be 0")
        selected = [i for map_name in MAPS for i in per_map[map_name][:min(32, samples_per_map)]]
    else:
        interleaved = [per_map[map_name][j] for j in range(samples_per_map) for map_name in MAPS]
        wanted = {"effective": 128, "pool": samples_per_map * len(MAPS)}.get(stage, source_samples)
        if wanted is None or wanted < 1 or source_offset < 0 or source_offset + wanted > len(interleaved):
            raise ValueError("custom stage needs valid --source-samples")
        selected = interleaved[source_offset:source_offset + wanted]
    if max_source_samples is not None:
        if max_source_samples < 1:
            raise ValueError("max_source_samples must be positive")
        selected = selected[:max_source_samples]
    return selected


def route_coordinate(model: torch.nn.Module) -> ParameterCoordinate:
    """Two fixed tensors per structural route, plus complete output heads."""
    chosen = []
    seen = set()
    for label, pattern in ROUTE_GROUPS.items():
        regex = re.compile(pattern)
        matches = [(name, param) for name, param in model.named_parameters()
                   if param.requires_grad and regex.search(name)]
        if not matches:
            continue
        selected = matches if label == "loc_output" else matches[:2]
        for name, param in selected:
            if name not in seen:
                chosen.append((name, param))
                seen.add(name)
    if not chosen:
        raise ValueError("No trainable route parameters found")
    return ParameterCoordinate(tuple(n for n, _ in chosen), tuple(p for _, p in chosen))


def coordinate_partition(coordinate: ParameterCoordinate) -> dict:
    """Report a disjoint structural assignment and expose unclassified tensors."""
    regexes = {label: re.compile(pattern) for label, pattern in PARTITION_GROUPS.items()}
    groups = {label: [] for label in (*PARTITION_GROUPS, "unclassified")}
    for name, param in zip(coordinate.names, coordinate.params):
        matches = [label for label, regex in regexes.items() if regex.search(name)]
        if len(matches) > 1:
            raise ValueError(f"Overlapping parameter partitions for {name}: {matches}")
        groups[matches[0] if matches else "unclassified"].append((name, param.numel()))
    return {label: {"names": [name for name, _ in items],
                    "numel": sum(numel for _, numel in items)}
            for label, items in groups.items()}


def _move(value, device, *, float_dtype=None):
    if torch.is_tensor(value):
        return value.to(device=device,
                        dtype=float_dtype if float_dtype is not None and value.is_floating_point() else value.dtype,
                        non_blocking=False)
    if isinstance(value, dict):
        return {k: _move(v, device, float_dtype=float_dtype) for k, v in value.items()}
    if isinstance(value, list):
        return [_move(v, device, float_dtype=float_dtype) for v in value]
    if isinstance(value, tuple):
        return tuple(_move(v, device, float_dtype=float_dtype) for v in value)
    return value


def collated_batches(dataset, collator, indices, microbatch_size, device, *, float_dtype=None):
    for offset in range(0, len(indices), microbatch_size):
        chunk = indices[offset:offset + microbatch_size]
        if len(chunk) != microbatch_size:
            raise ValueError("Source sample count must be divisible by microbatch size")
        batch = collator([dataset[index] for index in chunk])
        if not isinstance(batch, dict):
            raise TypeError("Training collator must return a dict")
        yield _move(batch, device, float_dtype=float_dtype)


def rng_state_hashes(device: str) -> dict:
    """Record small fingerprints for full/component-sum random-trajectory comparison."""
    states = {
        "python": hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest(),
        "cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    }
    if str(device).startswith("cuda"):
        states["cuda"] = [hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
                          for state in torch.cuda.get_rng_state_all()]
    return states


def seed_noise(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv=None):
    args = parse_args(argv)
    if args.measurement_mode == "component_sum":
        if args.stage == "screen":
            raise ValueError("screen stage requires full gradient measurement")
        if not args.fp32_reference or not args.deterministic_probe or tuple(args.tasks) != TASKS:
            raise ValueError("component_sum requires FP32 deterministic probe and all four tasks in order")
    if args.deterministic_probe:
        os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    if args.coordinate == "custom" and not args.param_pattern:
        raise ValueError("--coordinate custom needs --param-pattern")
    if args.gpu_memory_fraction is not None:
        if not 0 < args.gpu_memory_fraction <= 1:
            raise ValueError("--gpu-memory-fraction must be in (0, 1]")
        if not str(args.device).startswith("cuda"):
            raise ValueError("--gpu-memory-fraction requires a CUDA --device")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction,
                                                   device=torch.device(args.device))
    from unilip.train.gradient_probe_loader import load_gradient_probe_context

    context = load_gradient_probe_context(
        args.checkpoint_dir, device=args.device, bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        precision_reference=args.fp32_reference,
        deterministic_probe=args.deterministic_probe)
    metadata = context.metadata
    metadata["measurement_mode"] = args.measurement_mode
    metadata["probe_batch_float_dtype"] = "float32" if args.fp32_reference else "collator_default"
    metadata["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    metadata["flash_attention_deterministic_env"] = os.environ.get("FLASH_ATTENTION_DETERMINISTIC")
    metadata["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if args.experiment and args.experiment != metadata["experiment"]:
        raise ValueError(f"Experiment mismatch: {args.experiment} != {metadata['experiment']}")
    if args.checkpoint_step is not None and args.checkpoint_step != metadata["checkpoint_step"]:
        raise ValueError(f"Checkpoint step mismatch: {args.checkpoint_step} != {metadata['checkpoint_step']}")
    microbatch_size = args.microbatch_size or int(metadata["train_microbatch"])
    if microbatch_size < 1:
        raise ValueError("microbatch size must be positive")
    if args.source_indices is not None:
        indices = list(args.source_indices)
        if len(indices) != len(set(indices)):
            raise ValueError("--source-indices contains duplicate dataset indices")
        if any(index < 0 or index >= len(context.dataset) for index in indices):
            raise ValueError("--source-indices contains an out-of-range dataset index")
        if args.accumulations is not None and len(indices) != microbatch_size * args.accumulations:
            raise ValueError("Explicit source indices must match --microbatch-size * --accumulations")
        source_selection = "explicit_indices"
    else:
        indices = source_indices(metadata.get("dataset_source_rows_by_map",
                                          metadata["dataset_indices_by_map"]),
                                 source_seed=args.source_seed, samples_per_map=args.samples_per_map,
                                 stage=args.stage, source_samples=args.source_samples,
                                 source_offset=args.source_offset,
                                 max_source_samples=args.max_source_samples)
        source_selection = "staged_stable_identity"
    if args.accumulations is not None and args.source_indices is None:
        if args.accumulations < 1:
            raise ValueError("accumulations must be positive")
        indices = indices[:microbatch_size * args.accumulations]
    if len(indices) % microbatch_size:
        raise ValueError(f"{len(indices)} source samples cannot form full microbatches of {microbatch_size}")
    if args.coordinate == "shared":
        coordinate = select_coordinate(context.model, [r"(^|\.)language_model\."])
    elif args.coordinate == "route":
        coordinate = route_coordinate(context.model)
    else:
        coordinate = select_coordinate(context.model, args.param_pattern)
    partition = coordinate_partition(coordinate)
    selected_source_rows = None
    if "dataset_source_rows_by_map" in metadata:
        by_index = {int(row["dataset_index"]): row
                    for rows in metadata["dataset_source_rows_by_map"].values() for row in rows}
        selected_source_rows = [by_index[index] for index in indices]
    results = {}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    jsonl = output.with_suffix(output.suffix + ".jsonl")
    jsonl.write_text("", encoding="utf-8")
    for seed in args.noise_seeds:
        seed_noise(seed)
        backward_check = None
        if args.check_total_backward:
            first_batch = next(collated_batches(context.dataset, context.collator,
                                                indices[:microbatch_size], microbatch_size, args.device,
                                                float_dtype=torch.float32 if args.fp32_reference else None))
            backward_check = check_total_backward(context.model, first_batch, coordinate,
                                                  rtol=args.linearity_rtol,
                                                  atol=args.linearity_atol)
            seed_noise(seed)
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        expected = len(indices) // microbatch_size
        stride = max(1, round(expected / 5))
        def progress(done, total, elapsed):
            if done == 1 or done % stride == 0 or done == total:
                peak = (torch.cuda.max_memory_allocated(torch.device(args.device)) / 2**30
                        if torch.cuda.is_available() and str(args.device).startswith("cuda") else None)
                print(f"noise_seed={seed} microbatch={done}/{total} elapsed={elapsed:.1f}s "
                      f"peak_cuda_gib={peak}", flush=True)
        vectors = {} if args.save_gradient_vectors else None
        try:
            result = measure_gradients(
                context.model,
                collated_batches(context.dataset, context.collator, indices, microbatch_size,
                                 args.device, float_dtype=torch.float32 if args.fp32_reference else None),
                coordinate, tasks=args.tasks, accumulator_device=args.accumulator_device,
                group_patterns=ROUTE_GROUPS, check_aux_freeze=args.check_aux_freeze,
                linearity_rtol=args.linearity_rtol, linearity_atol=args.linearity_atol,
                progress_callback=progress, expected_microbatches=expected, vector_sink=vectors,
                numerical_check_first_batch=args.numerical_check,
                measurement_mode=args.measurement_mode,
                fp32_reference=args.fp32_reference,
                deterministic_probe=args.deterministic_probe)
        except Exception as exc:
            with jsonl.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"noise_seed": seed, "status": "failed",
                                         "error": repr(exc)}, allow_nan=False) + "\n")
                stream.flush()
            raise
        result["rng_state_after_measure_sha256"] = rng_state_hashes(args.device)
        if backward_check is not None:
            result["total_backward_check"] = backward_check
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            result["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(torch.device(args.device))
        if vectors is not None:
            vector_path = output.with_name(f"{output.stem}.seed{seed}.vectors.pt")
            torch.save(vectors, vector_path)
            result["gradient_vectors_path"] = str(vector_path)
        results[str(seed)] = result
        with jsonl.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"noise_seed": seed, "status": "complete",
                                     "result": result}, allow_nan=False) + "\n")
            stream.flush()
        if result["linearity"] is not None:
            check_summary = f"linearity={result['linearity']['passed']}"
        else:
            checks = result["first_micro_direct_checks"]
            check_summary = ("first_micro_direct=" + str(
                checks["main_vs_components"]["passed"] and
                checks["total_vs_components"]["passed"]))
        print(f"noise_seed={seed} total_norm={result['total_gradient_norm']:.6g} "
              f"{check_summary}", flush=True)
    payload = {
        "schema_version": 1, "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "metadata": metadata, "stage": args.stage,
        "measurement_mode": args.measurement_mode,
        "source_seed": args.source_seed, "source_offset": args.source_offset,
        "source_selection": source_selection,
        "source_indices": indices,
        "source_rows": selected_source_rows,
        "microbatch_size": microbatch_size, "accumulations": len(indices) // microbatch_size,
        "noise_seeds": args.noise_seeds, "coordinate": args.coordinate,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "coordinate_partition": partition,
        "tasks": args.tasks, "results": results,
        "note": "Cross-checkpoint paired source IDs do not guarantee paired internal loc noise because optional branches consume RNG differently.",
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
