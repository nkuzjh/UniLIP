"""Scoped DeepSpeed BF16 stage-0 optimizer resume from DP=1 to DP>1.

DeepSpeed 0.16.9 saves BF16 master weights and AdamW moments per DP rank
even when ZeRO is disabled.  Its ordinary loader cannot index a DP=1 state
list from a larger data-parallel job.  This module adapts only that layout;
all matching-world-size loads still use DeepSpeed's original method.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import os
from types import MethodType
from typing import Any

import torch


_BASE = "base_optimizer_state"
_MASTER = "single_partition_of_fp32_groups"
_PADDING = "group_paddings"
_COUNT = "partition_count"
_MAPPING = "param_slice_mappings"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"BF16 cross-world resume: {message}")


def _fragment(address: Any, context: str) -> tuple[int, int]:
    try:
        start, length = address.start, address.numel
    except AttributeError as exc:
        raise ValueError(f"BF16 cross-world resume: invalid mapping for {context}") from exc
    _require(isinstance(start, int) and isinstance(length, int) and start >= 0 and length > 0,
             f"invalid fragment offset or length for {context}")
    return start, length


def _partition_tensor(tensor: torch.Tensor, useful_length: int, partition_length: int, rank: int) -> torch.Tensor:
    """Copy just this rank's slice, inserting zeroes for new alignment padding."""
    start = rank * partition_length
    result = tensor.new_zeros(partition_length)
    overlap = min(max(useful_length - start, 0), partition_length)
    if overlap:
        result[:overlap].copy_(tensor.narrow(0, start, overlap))
    return result


def repartition_single_rank_bf16_state(
    saved_state: dict[str, Any], target_optimizer: Any, rank: int, new_dp: int
) -> dict[str, Any]:
    """Build one CPU BF16 optimizer shard from a DP=1 AdamW checkpoint.

    The target optimizer has already been constructed by DeepSpeed.  Its
    BF16 parameter groups, names, shard lengths, and mappings define the new
    layout.  No tensor from ``saved_state`` is modified.
    """
    _require(isinstance(rank, int) and isinstance(new_dp, int) and new_dp > 1 and 0 <= rank < new_dp,
             "target rank must be within a DP world size greater than one")
    _require(isinstance(saved_state, dict), "optimizer checkpoint is not a dictionary")
    _require(all(key in saved_state for key in (_BASE, _MASTER, _PADDING, _COUNT, _MAPPING, "ds_version")),
             "optimizer checkpoint is missing required BF16 fields")
    groups = target_optimizer.bf16_groups
    masters = target_optimizer.fp32_groups_flat_partition
    names = target_optimizer.param_names
    current_mappings = target_optimizer._param_slice_mappings
    current_paddings = target_optimizer.group_paddings
    current_counts = target_optimizer.partition_count
    saved_masters = saved_state[_MASTER]
    saved_paddings = saved_state[_PADDING]
    saved_counts = saved_state[_COUNT]
    saved_mappings = saved_state[_MAPPING]
    count = len(groups)
    _require(count > 0 and all(len(x) == count for x in (masters, current_mappings, current_paddings,
                                                          current_counts, saved_masters, saved_paddings,
                                                          saved_counts, saved_mappings)),
             "parameter-group counts differ")
    _require(all(value == 1 for value in saved_counts), "source optimizer was not saved with DP=1")
    _require(all(value == new_dp for value in current_counts), "target optimizer DP count differs")
    base = saved_state[_BASE]
    _require(isinstance(base, dict) and isinstance(base.get("state"), dict)
             and isinstance(base.get("param_groups"), list) and len(base["param_groups"]) == count,
             "invalid base AdamW state")
    _require(isinstance(target_optimizer.optimizer, torch.optim.AdamW), "target optimizer is not torch AdamW")
    _require(len(target_optimizer.optimizer.param_groups) == count, "target AdamW group count differs")
    _require(all(len(group["params"]) == 1 and group["params"][0] is masters[i]
                 for i, group in enumerate(target_optimizer.optimizer.param_groups)),
             "target AdamW parameters are not the BF16 master partitions")

    new_masters: list[torch.Tensor] = []
    new_states: dict[Any, dict[str, Any]] = {}
    seen_keys: set[Any] = set()
    for i in range(count):
        parameters = groups[i]
        _require(parameters and all(p.requires_grad for p in parameters), f"group {i} has missing trainable parameters")
        expected_names = [names[p] for p in parameters]
        _require(len(expected_names) == len(set(expected_names)), f"duplicate parameter name in group {i}")
        old_map = saved_mappings[i]
        _require(list(old_map) == expected_names, f"parameter name/order mismatch in group {i}")
        useful_length = sum(p.numel() for p in parameters)
        offset = 0
        for name, parameter in zip(expected_names, parameters):
            start, length = _fragment(old_map[name], f"source group {i} parameter {name}")
            _require(start == offset and length == parameter.numel(),
                     f"parameter offset/length mismatch in group {i} for {name}")
            offset += length
        _require(offset == useful_length, f"source mapping does not cover group {i}")

        old_master = saved_masters[i]
        partition_length = masters[i].numel()
        _require(isinstance(old_master, torch.Tensor) and old_master.ndim == 1
                 and old_master.dtype == torch.float32 and old_master.device.type == "cpu",
                 f"source master tensor has unsupported type in group {i}")
        old_padding = saved_paddings[i]
        _require(isinstance(old_padding, int) and old_padding >= 0
                 and old_master.numel() - old_padding == useful_length,
                 f"source master length/padding mismatch in group {i}")
        expected_padding = partition_length * new_dp - useful_length
        local_padding = expected_padding if rank == new_dp - 1 else 0
        _require(isinstance(partition_length, int) and partition_length > 0
                 and expected_padding >= 0 and expected_padding < 2 * new_dp
                 and current_paddings[i] == local_padding,
                 f"target alignment/padding mismatch in group {i}")
        _require(partition_length * new_dp == target_optimizer.bf16_groups_flat[i].numel(),
                 f"target BF16 flat length mismatch in group {i}")

        # The target mapping only contains parameters that overlap this rank.
        expected_fragments: list[tuple[str, int, int]] = []
        group_offset = 0
        lower, upper = rank * partition_length, (rank + 1) * partition_length
        for name, parameter in zip(expected_names, parameters):
            lo, hi = max(group_offset, lower), min(group_offset + parameter.numel(), upper)
            if lo < hi:
                expected_fragments.append((name, lo - lower, hi - lo))
            group_offset += parameter.numel()
        target_map = current_mappings[i]
        _require(list(target_map) == [item[0] for item in expected_fragments],
                 f"target parameter mapping order mismatch in group {i}")
        for name, expected_start, expected_length in expected_fragments:
            _require(_fragment(target_map[name], f"target group {i} parameter {name}")
                     == (expected_start, expected_length),
                     f"target fragment mismatch in group {i} for {name}")

        saved_group = base["param_groups"][i]
        _require(isinstance(saved_group, dict) and isinstance(saved_group.get("params"), list)
                 and len(saved_group["params"]) == 1 and not saved_group.get("amsgrad", False),
                 f"unsupported AdamW parameter group {i}")
        key = saved_group["params"][0]
        _require(key not in seen_keys and key in base["state"], f"missing or duplicate AdamW state in group {i}")
        seen_keys.add(key)
        old_state = base["state"][key]
        _require(isinstance(old_state, dict) and set(old_state) == {"step", "exp_avg", "exp_avg_sq"},
                 f"unsupported AdamW state fields in group {i}")
        step = old_state["step"]
        _require((isinstance(step, torch.Tensor) and step.ndim == 0 and step.device.type == "cpu")
                 or (isinstance(step, (int, float)) and not isinstance(step, bool)),
                 f"unsupported AdamW step in group {i}")
        moments = {}
        for field in ("exp_avg", "exp_avg_sq"):
            value = old_state[field]
            _require(isinstance(value, torch.Tensor) and value.ndim == 1
                     and value.dtype == torch.float32 and value.device.type == "cpu"
                     and value.numel() == old_master.numel(),
                     f"unsupported {field} tensor in group {i}")
            moments[field] = _partition_tensor(value, useful_length, partition_length, rank)
        new_masters.append(_partition_tensor(old_master, useful_length, partition_length, rank))
        new_states[key] = {"step": step.clone() if isinstance(step, torch.Tensor) else step, **moments}

    _require(set(base["state"]) == seen_keys, "checkpoint contains unexpected AdamW state entries")
    result = dict(saved_state)
    result[_MASTER] = new_masters
    result[_PADDING] = list(current_paddings)
    result[_COUNT] = list(current_counts)
    result[_MAPPING] = current_mappings
    result[_BASE] = {"state": new_states, "param_groups": deepcopy(base["param_groups"])}
    return result


def _cross_world_zero_checkpoint(engine: Any, original: Any, load_dir: str, tag: str,
                                 load_optimizer_states: bool = True) -> bool:
    old_dp = engine.loaded_checkpoint_dp_world_size
    new_dp = engine.seq_dp_world_size
    if old_dp == new_dp:
        return original(load_dir, tag, load_optimizer_states=load_optimizer_states)

    from deepspeed.runtime.bf16_optimizer import BF16_Optimizer

    _require(old_dp == 1 and isinstance(new_dp, int) and new_dp > 1,
             f"only source DP=1 to target DP>1 is supported (got {old_dp} to {new_dp})")
    _require(load_optimizer_states, "full optimizer-state loading is required")
    _require(isinstance(engine.optimizer, BF16_Optimizer), "engine optimizer is not BF16_Optimizer")
    _require(engine.bfloat16_enabled() and int(engine.zero_optimization_stage()) == 0,
             "only BF16 ZeRO stage 0 is supported")
    _require(not engine.has_moe_layers and not engine.optimizer.has_moe_layers, "MoE is unsupported")
    _require(engine.loaded_checkpoint_mp_world_size == 1
             and (engine.mpu is None or engine.mpu.get_model_parallel_world_size() == 1),
             "model parallel world size must be one in source and target")
    _require(not engine.load_universal_checkpoint(), "universal checkpoints are unsupported")
    _require(not engine.zero_elastic_checkpoint(), "elastic checkpoints are unsupported")
    rank = engine.optimizer.dp_rank
    _require(isinstance(rank, int) and 0 <= rank < new_dp,
             "optimizer DP rank is outside target world size")
    names = engine._get_all_zero_checkpoint_names(load_dir, tag, bf16_mode=True)
    _require(names is not None and len(names) == 1 and os.path.isfile(names[0]),
             "expected exactly one source BF16 optimizer shard")
    payload = engine.checkpoint_engine.load(names[0], map_location=torch.device("cpu"))
    _require(isinstance(payload, dict) and isinstance(payload.get("optimizer_state_dict"), dict),
             "source shard lacks optimizer_state_dict")
    source_config = payload.get("ds_config")
    _require(isinstance(source_config, dict)
             and source_config.get("zero_optimization", {}).get("stage") == 0
             and source_config.get("bf16", {}).get("enabled") is True,
             "source checkpoint is not BF16 ZeRO stage 0")
    converted = repartition_single_rank_bf16_state(payload["optimizer_state_dict"], engine.optimizer, rank, new_dp)
    shards: list[Any] = [None] * new_dp
    shards[rank] = converted
    engine.optimizer.load_state_dict(state_dict_list=shards, checkpoint_folder=None,
                                     load_optimizer_states=True, load_from_fp32_weights=True,
                                     load_serial=None, param_shapes=engine._get_zero_param_shapes())
    from deepspeed.utils import logger
    logger.info(f"BF16 cross-world optimizer resume succeeded: DP {old_dp} -> {new_dp}, rank {rank}, tag {tag}")
    return True


@contextmanager
def bf16_cross_world_resume(enabled: bool = True):
    """Temporarily adapt Transformers' DeepSpeed checkpoint load call."""
    if not enabled:
        yield
        return

    import transformers.trainer as trainer_module

    original_entry = trainer_module.deepspeed_load_checkpoint

    def compatible_entry(engine: Any, *args: Any, **kwargs: Any) -> Any:
        previous_instance_method = engine.__dict__.get("_load_zero_checkpoint")
        original_zero = engine._load_zero_checkpoint

        def compatible_zero(self: Any, load_dir: str, tag: str,
                            load_optimizer_states: bool = True) -> bool:
            return _cross_world_zero_checkpoint(self, original_zero, load_dir, tag, load_optimizer_states)

        engine._load_zero_checkpoint = MethodType(compatible_zero, engine)
        try:
            return original_entry(engine, *args, **kwargs)
        finally:
            if previous_instance_method is None:
                del engine._load_zero_checkpoint
            else:
                engine._load_zero_checkpoint = previous_instance_method

    trainer_module.deepspeed_load_checkpoint = compatible_entry
    try:
        yield
    finally:
        trainer_module.deepspeed_load_checkpoint = original_entry
