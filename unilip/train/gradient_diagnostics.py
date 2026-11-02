"""Offline, graph-connected multitask gradient measurements.

The probe never calls backward, optimizer.step, or changes model parameters.
All measurements use the exact scalar losses produced by the training forward.
"""
from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch


TASKS = ("gen", "loc", "perception", "aux_loc")
OTHER_TASKS = ("repa", "loc_repa", "gen_aux")
_SHARED_LANGUAGE = re.compile(r"(?:^|\.)language_model\.")
_LANGUAGE_LAYER = re.compile(r"(?:^|\.)language_model\.(?:[^.]+\.)*layers\.(\d+)(?:\.|$)")
_PRIVATE_PARENT = re.compile(
    r"(?:^|\.)(?:llm_connector|dit|projector|latent_queries|action_dit|"
    r"action_out_proj|action_in_proj|action_dit_connector|action_dit_norm|"
    r"action_dit_projector|time_mlp_in|time_mlp_out)(?:\.|$)")


@dataclass(frozen=True)
class ParameterCoordinate:
    names: tuple[str, ...]
    params: tuple[torch.nn.Parameter, ...]

    @property
    def numel(self) -> int:
        return sum(p.numel() for p in self.params)


def select_coordinate(model: torch.nn.Module, patterns: Sequence[str]) -> ParameterCoordinate:
    """Select a stable named coordinate, preserving model.named_parameters order."""
    if not patterns:
        raise ValueError("At least one --param-pattern is required to define a shared coordinate")
    regexes = [re.compile(pattern) for pattern in patterns]
    selected = [(name, p) for name, p in model.named_parameters()
                if p.requires_grad and any(regex.search(name) for regex in regexes)]
    if not selected:
        raise ValueError(f"No trainable parameters match {list(patterns)!r}")
    return ParameterCoordinate(tuple(name for name, _ in selected),
                               tuple(p for _, p in selected))


def _zero_buffers(coordinate: ParameterCoordinate, device: str) -> list[torch.Tensor]:
    return [torch.zeros(p.shape, dtype=torch.float32, device=device) for p in coordinate.params]


def _grad(loss: torch.Tensor, params: Sequence[torch.nn.Parameter], *, retain_graph: bool):
    if not loss.requires_grad:
        return (None,) * len(params)
    return torch.autograd.grad(loss, params, allow_unused=True, retain_graph=retain_graph)


def clone_batch(value):
    """Clone tensor leaves before replaying a forward that mutates its inputs."""
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, Mapping):
        return {key: clone_batch(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_batch(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_batch(item) for item in value)
    return value


def _dot(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float:
    return sum(float(torch.sum(a.double() * b.double()).item()) for a, b in zip(left, right))


def _norm(left: Sequence[torch.Tensor]) -> float:
    return math.sqrt(max(_dot(left, left), 0.0))


def _cos(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float | None:
    den = _norm(left) * _norm(right)
    return _dot(left, right) / den if den else None


def shared_language_layer_stats(
    coordinate: ParameterCoordinate,
    weighted: Mapping[str, Sequence[torch.Tensor]],
    raw: Mapping[str, Sequence[torch.Tensor]],
) -> dict:
    """Summarize existing averaged VJPs over the shared language coordinate.

    Each selected shared tensor belongs to exactly one numbered layer or to the
    explicit unassigned bucket. Private branches with a nested language_model
    name are excluded. No gradients or full-vector copies are made here.
    """
    buckets: dict[str, list[int]] = defaultdict(list)
    shared = []
    for index, name in enumerate(coordinate.names):
        match = _SHARED_LANGUAGE.search(name)
        if match is None or _PRIVATE_PARENT.search(name[:match.start()]):
            continue
        shared.append(index)
        layer = _LANGUAGE_LAYER.search(name)
        buckets[layer.group(1) if layer else "unassigned"].append(index)
    if not shared:
        return {"status": "no_shared_parameters", "shared_parameter_count": 0,
                "shared_numel": 0, "layers": {}, "unassigned": None}
    if not {"gen", "loc"}.issubset(weighted) or not {"gen", "loc"}.issubset(raw):
        return {"status": "requires_gen_and_loc", "shared_parameter_count": len(shared),
                "shared_numel": sum(coordinate.params[i].numel() for i in shared),
                "layers": {}, "unassigned": None}

    def vectors(buffers, indices):
        return [buffers[i] for i in indices]

    shared_gen = vectors(weighted["gen"], shared)
    shared_loc = vectors(weighted["loc"], shared)
    global_dot = _dot(shared_gen, shared_loc)
    global_denominator = _norm(shared_gen) * _norm(shared_loc)

    def summarize(indices):
        gen = vectors(weighted["gen"], indices)
        loc = vectors(weighted["loc"], indices)
        total = vectors(weighted["total"], indices)
        raw_gen = vectors(raw["gen"], indices)
        raw_loc = vectors(raw["loc"], indices)
        active = {name: vectors(weighted[name], indices) for name in raw}
        active_norms = {name: _norm(values) for name, values in active.items()}
        pairwise = {name: {} for name in active}
        active_names = tuple(active)
        for position, left in enumerate(active_names):
            for right in active_names[position:]:
                denominator = active_norms[left] * active_norms[right]
                cosine = _dot(active[left], active[right]) / denominator if denominator else None
                pairwise[left][right] = cosine
                pairwise[right][left] = cosine
        has_aux = "perception" in active and "aux_loc" in active
        dot = _dot(gen, loc)
        return {
            "parameter_count": len(indices),
            "numel": sum(coordinate.params[i].numel() for i in indices),
            "components": {
                name: {"raw_norm": _norm(vectors(raw[name], indices)),
                       "weighted_norm": active_norms[name]}
                for name in raw
            },
            "total_norm": _norm(total),
            "weighted_gen_loc_dot": dot,
            "weighted_gen_total_dot": _dot(gen, total),
            "weighted_loc_total_dot": _dot(loc, total),
            "raw_loc_total_dot": _dot(raw_loc, total),
            "weighted_gen_aux_dot": (
                _dot(gen, active["perception"]) + _dot(gen, active["aux_loc"])
                if has_aux else None),
            "weighted_loc_aux_dot": (
                _dot(loc, active["perception"]) + _dot(loc, active["aux_loc"])
                if has_aux else None),
            "weighted_gen_loc_cosine": _cos(gen, loc),
            "raw_gen_loc_cosine": _cos(raw_gen, raw_loc),
            "weighted_gen_total_cosine": _cos(gen, total),
            "weighted_loc_total_cosine": _cos(loc, total),
            "raw_loc_total_cosine": _cos(raw_loc, total),
            "pairwise_weighted_cosine": pairwise,
            "negative_dot_contribution": max(-dot, 0.0),
            "positive_alignment_mass": max(dot, 0.0) / global_denominator
                if global_denominator else None,
            "negative_alignment_mass": max(-dot, 0.0) / global_denominator
                if global_denominator else None,
        }

    layers = {key: summarize(buckets[key]) for key in sorted(
        (key for key in buckets if key != "unassigned"), key=int)}
    unassigned = summarize(buckets["unassigned"]) if buckets.get("unassigned") else None
    named_dot = sum(item["weighted_gen_loc_dot"] for item in layers.values())
    assigned_dot = named_dot + (unassigned["weighted_gen_loc_dot"] if unassigned else 0.0)
    assigned_count = sum(item["parameter_count"] for item in layers.values())
    assigned_numel = sum(item["numel"] for item in layers.values())
    if unassigned:
        assigned_count += unassigned["parameter_count"]
        assigned_numel += unassigned["numel"]
    shared_numel = sum(coordinate.params[i].numel() for i in shared)
    dot_tolerance = 1e-8 + 1e-8 * abs(global_dot)
    return {
        "status": "ok" if not unassigned else "has_unassigned_shared_parameters",
        "shared_parameter_count": len(shared), "shared_numel": shared_numel,
        "global_shared_weighted_gen_loc_dot": global_dot,
        "global_shared_weighted_gen_norm": _norm(shared_gen),
        "global_shared_weighted_loc_norm": _norm(shared_loc),
        "global_alignment_denominator": global_denominator,
        "layers": layers, "unassigned": unassigned,
        "layers_by_negative_dot_contribution": sorted(
            (key for key, item in layers.items() if item["weighted_gen_loc_dot"] < 0),
            key=lambda key: layers[key]["negative_dot_contribution"], reverse=True),
        "positive_alignment_mass": (
            sum(item["positive_alignment_mass"] for item in
                (*layers.values(), *((unassigned,) if unassigned else ())))
            if global_denominator else None),
        "negative_alignment_mass": (
            sum(item["negative_alignment_mass"] for item in
                (*layers.values(), *((unassigned,) if unassigned else ())))
            if global_denominator else None),
        "checks": {
            "named_layer_dot_sum": named_dot,
            "named_layer_dot_matches_global": abs(named_dot - global_dot) <= dot_tolerance,
            "partition_dot_sum": assigned_dot,
            "partition_dot_difference": assigned_dot - global_dot,
            "partition_dot_matches_global": abs(assigned_dot - global_dot) <= dot_tolerance,
            "covered_parameter_count": assigned_count,
            "covered_numel": assigned_numel,
            "parameter_count_matches_global": assigned_count == len(shared),
            "numel_matches_global": assigned_numel == shared_numel,
        },
    }


def _add(buffers: Sequence[torch.Tensor], grads: Sequence[torch.Tensor | None],
         factor: float, counts: dict[str, int]) -> None:
    for buf, grad in zip(buffers, grads):
        if grad is None:
            counts["none"] += 1
            continue
        counts["present"] += 1
        # For the default CPU accumulator, transfer once before statistics. A
        # CUDA scalar bool here would synchronize the device for every tensor.
        value = (grad.detach().to(device="cpu", dtype=torch.float32)
                 if buf.device.type == "cpu" else grad)
        finite = torch.isfinite(value)
        if not bool(finite.all()):
            counts["nonfinite"] += int((~finite).sum().item())
            raise FloatingPointError("Nonfinite gradient encountered in gradient probe")
        if bool(torch.count_nonzero(value) == 0):
            counts["zero"] += 1
        if buf.device.type != "cpu":
            value = grad.detach().to(device=buf.device, dtype=torch.float32)
        buf.add_(value, alpha=factor)


def _add_raw_weighted(raw_buffers: Sequence[torch.Tensor],
                      weighted_buffers: Sequence[torch.Tensor],
                      grads: Sequence[torch.Tensor | None], weight: float,
                      counts: dict[str, int]) -> None:
    if raw_buffers and raw_buffers[0].device.type != "cpu":
        # Preserve the original CUDA-accumulator behavior and operation order.
        _add(raw_buffers, grads, 1.0, defaultdict(int))
        _add(weighted_buffers, grads, weight, counts)
        return
    for raw, weighted, grad in zip(raw_buffers, weighted_buffers, grads):
        if grad is None:
            counts["none"] += 1
            continue
        counts["present"] += 1
        value = grad.detach().to(device="cpu", dtype=torch.float32)
        finite = torch.isfinite(value)
        if not bool(finite.all()):
            counts["nonfinite"] += int((~finite).sum().item())
            raise FloatingPointError("Nonfinite gradient encountered in gradient probe")
        if bool(torch.count_nonzero(value) == 0):
            counts["zero"] += 1
        raw.add_(value)
        weighted.add_(value, alpha=weight)


def _check_aux_freeze(model: torch.nn.Module, aux_loss: torch.Tensor) -> dict:
    """Verify the aux loss does not differentiate into its frozen loc head."""
    if not hasattr(model, "_aux_loc_head_modules_for_freeze"):
        return {"supported": False}
    modules = model._aux_loc_head_modules_for_freeze()
    seen: set[int] = set()
    params = []
    for module in modules:
        for param in module.parameters():
            if param.requires_grad and id(param) not in seen:
                seen.add(id(param))
                params.append(param)
    if not params:
        return {"supported": True, "checked_parameters": 0, "routed_parameters": 0}
    grads = _grad(aux_loss, params, retain_graph=True)
    routed = sum(grad is not None and bool(torch.count_nonzero(grad)) for grad in grads)
    return {"supported": True, "checked_parameters": len(params),
            "routed_parameters": routed, "passed": routed == 0}


def measure_gradients(
    model: torch.nn.Module,
    batches: Iterable[Mapping[str, object]],
    coordinate: ParameterCoordinate,
    *,
    tasks: Sequence[str] = TASKS,
    accumulator_device: str = "cpu",
    group_patterns: Mapping[str, str] | None = None,
    check_aux_freeze: bool = True,
    linearity_rtol: float = 2e-3,
    linearity_atol: float = 1e-5,
    progress_callback=None,
    expected_microbatches: int | None = None,
    vector_sink: dict | None = None,
    numerical_check_first_batch: bool = False,
    measurement_mode: str = "full",
    fp32_reference: bool = False,
    deterministic_probe: bool = False,
) -> dict:
    """Average per-microbatch VJPs, matching Trainer gradient accumulation scale.

    `batches` must already contain the intended fixed source samples and have the
    same microbatch size as training when comparing with a training step. An aux
    active-mean normalization remains local to each microbatch, exactly as trained.
    """
    tasks = tuple(tasks)
    if measurement_mode not in ("full", "component_sum"):
        raise ValueError(f"Unknown gradient measurement mode: {measurement_mode!r}")
    component_sum_mode = measurement_mode == "component_sum"
    if component_sum_mode and (not fp32_reference or not deterministic_probe or tasks != TASKS):
        raise ValueError("component_sum requires FP32 deterministic probe and all four tasks in order")
    unknown = set(tasks) - set(TASKS)
    if unknown or not tasks:
        raise ValueError(f"Unsupported or empty task selection: {tasks!r}")
    # The residual groups every remaining term so total-gradient linearity is
    # checked even when optional REPA and gen auxiliary objectives are active.
    names = (*tasks, "other", "main", "total")
    sums = {name: _zero_buffers(coordinate, accumulator_device) for name in names}
    raw_sums = {name: _zero_buffers(coordinate, accumulator_device) for name in tasks}
    counts = {name: defaultdict(int) for name in names}
    scalar_sums = defaultdict(float)
    scalar_difference_sum = 0.0
    scalar_difference_max_abs = 0.0
    weights_seen = defaultdict(set)
    freeze_checks = []
    repeated_total_check = None
    first_micro_direct_checks = None
    count = 0
    samples = 0
    started = time.monotonic()
    initial_grad_flags = tuple(p.requires_grad for p in model.parameters())
    initial_modes = {name: module.training for name, module in model.named_modules()}
    for batch in batches:
        if not isinstance(batch, Mapping):
            raise TypeError("Each probe batch must be a mapping for model(**batch)")
        output = model(**batch, return_loss_tensors=True)
        losses = output.loss_tensors
        weight_map = losses["weights"]
        if not isinstance(weight_map, Mapping):
            raise TypeError("loss_tensors['weights'] must be a mapping")
        for name in (*TASKS, *OTHER_TASKS, "total"):
            scalar = losses[name]
            if not torch.is_tensor(scalar) or scalar.ndim != 0:
                raise TypeError(f"loss_tensors[{name!r}] must be a scalar tensor")
            if not bool(torch.isfinite(scalar)):
                raise FloatingPointError(f"Nonfinite {name} loss")
            scalar_sums[name] += float(scalar.detach().float().item())
        for name in (*TASKS, *OTHER_TASKS):
            weight = float(weight_map[name])
            if not math.isfinite(weight):
                raise FloatingPointError(f"Nonfinite {name} weight")
            weights_seen[name].add(weight)
            if component_sum_mode and name in OTHER_TASKS and weight != 0:
                raise ValueError(f"component_sum requires zero {name} weight on every microbatch")
        scalar_difference = float((losses["total"] - sum(
            losses[name] * float(weight_map[name]) for name in (*TASKS, *OTHER_TASKS)
        )).detach().float().item())
        scalar_difference_sum += scalar_difference
        scalar_difference_max_abs = max(scalar_difference_max_abs, abs(scalar_difference))
        if check_aux_freeze and float(weight_map["aux_loc"]) != 0:
            freeze_checks.append(_check_aux_freeze(model, losses["aux_loc"]))
        for name in tasks:
            # First microbatch still needs the graph for direct reconstruction.
            # Later component-sum microbatches release saved tensors on their
            # final raw VJP before the next forward allocates a new graph.
            retain_graph = not (component_sum_mode and count > 0 and name == tasks[-1])
            grads = _grad(losses[name], coordinate.params, retain_graph=retain_graph)
            _add_raw_weighted(raw_sums[name], sums[name], grads,
                              float(weight_map[name]), counts[name])
        del grads
        if component_sum_mode:
            if count == 0:
                # These direct VJPs are independent checks on the first graph.
                main = losses["gen"] * float(weight_map["gen"]) + losses["loc"] * float(weight_map["loc"])
                direct_main = _grad(main, coordinate.params, retain_graph=True)
                direct_total = _grad(losses["total"], coordinate.params, retain_graph=True)
                def cpu_vector(grads):
                    return [torch.zeros_like(p, dtype=torch.float32, device="cpu") if g is None
                            else g.detach().float().cpu().clone()
                            for p, g in zip(coordinate.params, grads)]
                main_vector = cpu_vector(direct_main)
                total_vector = cpu_vector(direct_total)
                gen_loc = [a.detach().cpu() + b.detach().cpu()
                           for a, b in zip(sums["gen"], sums["loc"])]
                all_components = [a + b + c + d for a, b, c, d in zip(
                    sums["gen"], sums["loc"], sums["perception"], sums["aux_loc"])]
                all_components = [value.detach().cpu() for value in all_components]
                def reconstruction(actual, estimated):
                    reference_norm = _norm(actual)
                    difference_norm = _norm([a - b for a, b in zip(actual, estimated)])
                    return {"passed": difference_norm <= linearity_atol + linearity_rtol * reference_norm,
                            "difference_norm": difference_norm, "reference_norm": reference_norm,
                            "relative_difference": difference_norm / reference_norm if reference_norm else None,
                            "rtol": linearity_rtol, "atol": linearity_atol}
                first_micro_direct_checks = {
                    "scope": "first_microbatch_direct_vjp",
                    "main_vs_components": reconstruction(main_vector, gen_loc),
                    "total_vs_components": reconstruction(total_vector, all_components),
                }
                repeated = cpu_vector(_grad(losses["total"], coordinate.params, retain_graph=False))
                repeated_total_check = reconstruction(total_vector, repeated)
                checks = {name: first_micro_direct_checks[name]
                          for name in ("main_vs_components", "total_vs_components")}
                checks["repeated_total_vjp"] = repeated_total_check
                failed = [f"{name} difference_norm={check['difference_norm']:.6g} "
                          f"reference_norm={check['reference_norm']:.6g}"
                          for name, check in checks.items() if not check["passed"]]
                if failed:
                    raise FloatingPointError(
                        "First microbatch direct gradient check failed: " + "; ".join(failed))
                del main, direct_main, direct_total, main_vector, total_vector
                del gen_loc, all_components, repeated
        else:
            omitted = (*OTHER_TASKS, *(name for name in TASKS if name not in tasks))
            other = sum((losses[name] * float(weight_map[name]) for name in omitted),
                        torch.zeros_like(losses["total"]))
            _add(sums["other"], _grad(other, coordinate.params, retain_graph=True),
                 1.0, counts["other"])
            main = losses["gen"] * float(weight_map["gen"]) + losses["loc"] * float(weight_map["loc"])
            _add(sums["main"], _grad(main, coordinate.params, retain_graph=True),
                 1.0, counts["main"])
            total_grads = _grad(losses["total"], coordinate.params,
                                retain_graph=numerical_check_first_batch and count == 0)
            _add(sums["total"], total_grads, 1.0, counts["total"])
            if numerical_check_first_batch and count == 0:
                repeated = _grad(losses["total"], coordinate.params, retain_graph=False)
                first_vec = [torch.zeros_like(p, dtype=torch.float32, device="cpu") if g is None
                             else g.detach().float().cpu() for p, g in zip(coordinate.params, total_grads)]
                second_vec = [torch.zeros_like(p, dtype=torch.float32, device="cpu") if g is None
                              else g.detach().float().cpu() for p, g in zip(coordinate.params, repeated)]
                reference_norm = _norm(first_vec)
                difference_norm = _norm([a - b for a, b in zip(first_vec, second_vec)])
                repeated_total_check = {
                    "difference_norm": difference_norm,
                    "reference_norm": reference_norm,
                    "relative_difference": difference_norm / reference_norm if reference_norm else None,
                    "passed": difference_norm <= linearity_atol + linearity_rtol * reference_norm,
                }
        count += 1
        task_id = batch.get("task_id")
        if torch.is_tensor(task_id):
            samples += int(task_id.shape[0])
        # The scalar validation loop ends on losses["total"]. Drop that alias,
        # along with all other graph references, before advancing the iterator.
        del output, losses, scalar, weight_map, batch
        if not component_sum_mode:
            del other, main, total_grads
            if repeated_total_check is not None and count == 1:
                del repeated, first_vec, second_vec
        if tuple(p.requires_grad for p in model.parameters()) != initial_grad_flags:
            raise RuntimeError("Forward changed model requires_grad flags")
        if any(module.training != initial_modes[name] for name, module in model.named_modules()):
            raise RuntimeError("Forward changed model train/eval modes")
        if progress_callback is not None:
            progress_callback(count, expected_microbatches, time.monotonic() - started)
    if count == 0:
        raise ValueError("No batches supplied to gradient probe")
    for collection in (sums, raw_sums):
        for buffers in collection.values():
            for buf in buffers:
                buf.div_(count)
    if component_sum_mode:
        for name, members in (("main", ("gen", "loc")),
                              ("total", TASKS)):
            for dest, component_vectors in zip(sums[name], zip(*(sums[part] for part in members))):
                for component in component_vectors:
                    dest.add_(component)
    component_sum = _zero_buffers(coordinate, accumulator_device)
    for name in (*tasks, "other"):
        for dest, src in zip(component_sum, sums[name]):
            dest.add_(src)
    delta = [a - b for a, b in zip(sums["total"], component_sum)]
    total_norm = _norm(sums["total"])
    delta_norm = _norm(delta)
    linearity_passed = delta_norm <= linearity_atol + linearity_rtol * total_norm
    components = {}
    for name in tasks:
        components[name] = {
            "raw_norm": _norm(raw_sums[name]),
            "weighted_norm": _norm(sums[name]),
            "raw_cosine_to_total": _cos(raw_sums[name], sums["total"]),
            "cosine_to_total": _cos(sums[name], sums["total"]),
            "signed_contribution_to_total": (
                _dot(sums[name], sums["total"]) / (total_norm ** 2)
                if total_norm else None),
            "gradient_coverage": dict(counts[name]),
        }
    components["other"] = {
        "weighted_norm": _norm(sums["other"]),
        "cosine_to_total": _cos(sums["other"], sums["total"]),
        "signed_contribution_to_total": (
            _dot(sums["other"], sums["total"]) / (total_norm ** 2)
            if total_norm else None),
        "gradient_coverage": None if component_sum_mode else dict(counts["other"]),
    }
    components["linearity_residual"] = {
        "weighted_norm": None if component_sum_mode else delta_norm,
        "signed_contribution_to_total":
            None if component_sum_mode else (_dot(delta, sums["total"]) / (total_norm ** 2)
                                             if total_norm else None),
    }
    pairwise = {left: {right: _cos(sums[left], sums[right]) for right in tasks}
                for left in tasks}
    pairwise_raw = {left: {right: _cos(raw_sums[left], raw_sums[right]) for right in tasks}
                    for left in tasks}
    def combine(names):
        buffers = _zero_buffers(coordinate, accumulator_device)
        for name in names:
            if name in sums:
                for dest, src in zip(buffers, sums[name]):
                    dest.add_(src)
        return buffers

    baseline = combine(("gen", "loc"))
    auxiliary = combine(("perception", "aux_loc"))
    actual_aux_increment = [total - main for total, main in zip(sums["total"], sums["main"])]
    auxiliary_difference = [actual - estimated for actual, estimated
                            in zip(actual_aux_increment, auxiliary)]
    baseline_norm = _norm(baseline)
    auxiliary_norm = _norm(auxiliary)
    primary_directions = {
        name: {
            "raw_descent_first_order_total": -_dot(raw_sums[name], sums["total"]),
            "raw_descent_first_order_aux_increment": -_dot(raw_sums[name], auxiliary),
            "weighted_descent_first_order_total": -_dot(sums[name], sums["total"]),
            "weighted_descent_first_order_aux_increment": -_dot(sums[name], auxiliary),
        }
        for name in ("gen", "loc") if name in sums and name in raw_sums
    }
    weights_constant = all(len(weights_seen[name]) == 1 for name in (*TASKS, *OTHER_TASKS))
    group_results = {}
    for label, pattern in (group_patterns or {}).items():
        regex = re.compile(pattern)
        indices = [i for i, name in enumerate(coordinate.names) if regex.search(name)]
        if not indices:
            continue
        group_total = [sums["total"][i] for i in indices]
        group_delta = [delta[i] for i in indices]
        group_total_norm = _norm(group_total)
        group_delta_norm = _norm(group_delta)
        group_weighted = {name: [sums[name][i] for i in indices] for name in tasks}
        group_raw = {name: [raw_sums[name][i] for i in indices] for name in tasks}
        group_results[label] = {
            "parameter_count": len(indices),
            "numel": sum(coordinate.params[i].numel() for i in indices),
            "total_norm": group_total_norm,
            "linearity_difference_norm": None if component_sum_mode else group_delta_norm,
            "linearity_relative_difference":
                None if component_sum_mode else (group_delta_norm / group_total_norm
                                                 if group_total_norm else None),
            "components": {
                name: {
                    "weighted_norm": _norm([sums[name][i] for i in indices]),
                    "raw_norm": _norm(group_raw[name]) if name in group_raw else None,
                    "cosine_to_total": _cos([sums[name][i] for i in indices], group_total),
                } for name in (*tasks, "other")},
            "pairwise_weighted_cosine": {
                left: {right: _cos(group_weighted[left], group_weighted[right])
                       for right in tasks} for left in tasks},
            "pairwise_raw_cosine": {
                left: {right: _cos(group_raw[left], group_raw[right])
                       for right in tasks} for left in tasks},
        }
    result = {
        "measurement_mode": measurement_mode,
        "measurement_scope": {
            "components": "direct_vjp_every_microbatch",
            "main_total": "component_sum_of_mean_weighted_vjps" if component_sum_mode
                          else "direct_vjp_every_microbatch",
            "other": "zero_by_verified_weights" if component_sum_mode
                     else "direct_vjp_every_microbatch",
            "direct_reconstruction": "first_microbatch_only" if component_sum_mode else None,
        },
        "first_micro_direct_checks": first_micro_direct_checks,
        "coordinate": {"names": list(coordinate.names), "numel": coordinate.numel,
                       "parameter_count": len(coordinate.params)},
        "microbatches": count, "samples": samples,
        "mean_losses": {name: value / count for name, value in scalar_sums.items()},
        "weights_seen": {name: sorted(values) for name, values in weights_seen.items()},
        "components": components, "pairwise_weighted_cosine": pairwise,
        "pairwise_raw_cosine": pairwise_raw,
        "combined": {
            "baseline_gen_loc_norm": baseline_norm,
            "aux_perception_aux_loc_norm": auxiliary_norm,
            "aux_over_baseline": auxiliary_norm / baseline_norm if baseline_norm else None,
            "perception_over_baseline": _norm(sums["perception"]) / baseline_norm
                if baseline_norm and "perception" in sums else None,
            "aux_loc_over_baseline": _norm(sums["aux_loc"]) / baseline_norm
                if baseline_norm and "aux_loc" in sums else None,
            "aux_signed_contribution_to_total":
                _dot(auxiliary, sums["total"]) / (total_norm ** 2) if total_norm else None,
            "baseline_signed_contribution_to_total":
                _dot(baseline, sums["total"]) / (total_norm ** 2) if total_norm else None,
            "primary_directions": primary_directions,
            "main_joint_vjp_norm": None if component_sum_mode else _norm(sums["main"]),
            "main_joint_vs_component_sum_difference_norm":
                None if component_sum_mode else _norm([joint - separate for joint, separate in zip(sums["main"], baseline)]),
            "actual_aux_increment_norm": None if component_sum_mode else _norm(actual_aux_increment),
            "actual_aux_increment_vs_component_cosine":
                None if component_sum_mode else _cos(actual_aux_increment, auxiliary),
            "actual_aux_increment_vs_component_difference_norm":
                None if component_sum_mode else _norm(auxiliary_difference),
        },
        "groups": group_results,
        "shared_language_layers": shared_language_layer_stats(coordinate, sums, raw_sums),
        "total_gradient_norm": total_norm,
        "total_gradient_coverage": None if component_sum_mode else dict(counts["total"]),
        "linearity": None if component_sum_mode else {
            "passed": linearity_passed, "difference_norm": delta_norm,
            "relative_difference": delta_norm / total_norm if total_norm else None,
            "rtol": linearity_rtol, "atol": linearity_atol},
        "repeated_total_vjp": repeated_total_check,
        "scalar_linearity": {
            "weights_constant": weights_constant,
            "mean_difference": scalar_difference_sum / count,
            "max_abs_difference": scalar_difference_max_abs,
        },
        "other_objectives_active": any(
            abs(scalar_sums[name]) > 0 and any(weight != 0 for weight in weights_seen[name])
            for name in OTHER_TASKS),
        "elapsed_seconds": time.monotonic() - started,
        "aux_freeze": {"checks": len(freeze_checks),
                       "status": "checked" if freeze_checks else "skipped",
                       "passed": all(check.get("passed", True) for check in freeze_checks)
                           if freeze_checks else None,
                       "details": freeze_checks},
    }
    if vector_sink is not None:
        vector_sink.update({
            "parameter_names": list(coordinate.names),
            "weighted": {name: [buf.detach().cpu().clone() for buf in buffers]
                         for name, buffers in sums.items()},
            "raw": {name: [buf.detach().cpu().clone() for buf in buffers]
                    for name, buffers in raw_sums.items()},
        })
    return result


def check_total_backward(model: torch.nn.Module, batch: Mapping[str, object],
                         coordinate: ParameterCoordinate, *, rtol: float = 2e-3,
                         atol: float = 1e-5) -> dict:
    """Compare autograd.grad with a normal total.backward on identical RNG draws.

    This opt-in check is intended for one microbatch because `.backward()` creates
    gradients for every trainable model parameter. It clears those gradients on
    exit and never performs an optimizer update.
    """
    if any(param.grad is not None for param in model.parameters()):
        raise RuntimeError("Backward check requires initially empty .grad fields")
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    pristine_batch = clone_batch(batch)
    try:
        first = model(**clone_batch(pristine_batch), return_loss_tensors=True)
        expected = _grad(first.loss_tensors["total"], coordinate.params, retain_graph=False)
        expected = [torch.zeros_like(p, dtype=torch.float32, device="cpu") if g is None
                    else g.detach().float().cpu().clone()
                    for p, g in zip(coordinate.params, expected)]
        del first
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        second = model(**clone_batch(pristine_batch), return_loss_tensors=True)
        second.loss_tensors["total"].backward()
        actual = [torch.zeros_like(p, dtype=torch.float32, device="cpu") if p.grad is None
                  else p.grad.detach().float().cpu().clone() for p in coordinate.params]
        difference = _norm([a - e for a, e in zip(actual, expected)])
        reference = _norm(expected)
        return {"passed": difference <= atol + rtol * reference,
                "difference_norm": difference, "reference_norm": reference,
                "relative_difference": difference / reference if reference else None,
                "rtol": rtol, "atol": atol}
    finally:
        model.zero_grad(set_to_none=True)
