from collections import OrderedDict
from copy import deepcopy
import sys
import unittest
from types import SimpleNamespace
from types import ModuleType

import torch

from unilip.train.bf16_resume import (
    _cross_world_zero_checkpoint,
    bf16_cross_world_resume,
    repartition_single_rank_bf16_state,
)


def _layout(full_optimizer, rank, dp=2):
    groups = []
    names = {}
    mappings = []
    flat_groups = []
    shards = []
    paddings = []
    for i, group in enumerate(full_optimizer.param_groups):
        full = group["params"][0].detach()
        # Include both odd and even individual parameter sizes.
        sizes = ([3, 2] if i == 0 else [2, 4])
        assert sum(sizes) == full.numel()
        params = [torch.nn.Parameter(torch.zeros(n, dtype=torch.bfloat16)) for n in sizes]
        groups.append(params)
        for j, param in enumerate(params):
            names[param] = f"group{i}.param{j}"
        padded = ((full.numel() + 2 * dp - 1) // (2 * dp)) * (2 * dp)
        part = padded // dp
        shards.append(torch.nn.Parameter(torch.zeros(part)))
        flat_groups.append(torch.zeros(padded, dtype=torch.bfloat16))
        paddings.append(padded - full.numel() if rank == dp - 1 else 0)
        mapping = OrderedDict()
        offset = 0
        for param in params:
            lo, hi = max(offset, rank * part), min(offset + param.numel(), (rank + 1) * part)
            if lo < hi:
                mapping[names[param]] = SimpleNamespace(start=lo - rank * part, numel=hi - lo)
            offset += param.numel()
        mappings.append(mapping)
    base_groups = []
    for shard, source_group in zip(shards, full_optimizer.param_groups):
        base_groups.append({k: deepcopy(v) for k, v in source_group.items() if k != "params"} | {"params": [shard]})
    adamw = torch.optim.AdamW(base_groups)
    return SimpleNamespace(bf16_groups=groups, bf16_groups_flat=flat_groups,
                           fp32_groups_flat_partition=shards, group_paddings=paddings,
                           partition_count=[dp] * len(groups), _param_slice_mappings=mappings,
                           param_names=names, optimizer=adamw)


def _saved(full_optimizer):
    state = full_optimizer.state_dict()
    full_groups = [g["params"][0] for g in full_optimizer.param_groups]
    mappings = []
    for i, p in enumerate(full_groups):
        sizes = [3, 2] if i == 0 else [2, 4]
        offset = 0
        mapping = OrderedDict()
        for j, n in enumerate(sizes):
            mapping[f"group{i}.param{j}"] = SimpleNamespace(start=offset, numel=n)
            offset += n
        mappings.append(mapping)
    return {"ds_version": "0.16.9", "base_optimizer_state": state,
            "single_partition_of_fp32_groups": [p.detach().clone() for p in full_groups],
            "group_paddings": [0, 0], "partition_count": [1, 1],
            "param_slice_mappings": mappings, "clip_grad": 1.0}


def _full_optimizer():
    params = [torch.nn.Parameter(torch.arange(5, dtype=torch.float32) / 10),
              torch.nn.Parameter(torch.arange(6, dtype=torch.float32) / 7)]
    optimizer = torch.optim.AdamW([
        {"params": [params[0]], "lr": 0.005, "weight_decay": 0.1},
        {"params": [params[1]], "lr": 0.009, "weight_decay": 0.0},
    ])
    for step in range(3):
        for i, param in enumerate(params):
            param.grad = torch.arange(param.numel(), dtype=torch.float32).add_(1 + i + step) / 13
        optimizer.step()
        optimizer.zero_grad()
    return optimizer


def _check_repartition_recovers_master_moments_and_one_adamw_step():
    full = _full_optimizer()
    saved = _saved(full)
    targets = [_layout(full, rank) for rank in (0, 1)]
    converted = [repartition_single_rank_bf16_state(saved, targets[rank], rank, 2) for rank in (0, 1)]

    for i, source_param in enumerate(g["params"][0] for g in full.param_groups):
        key = saved["base_optimizer_state"]["param_groups"][i]["params"][0]
        useful = source_param.numel()
        for field, source in (("master", saved["single_partition_of_fp32_groups"][i]),
                              ("exp_avg", saved["base_optimizer_state"]["state"][key]["exp_avg"]),
                              ("exp_avg_sq", saved["base_optimizer_state"]["state"][key]["exp_avg_sq"])):
            shards = [part["single_partition_of_fp32_groups"][i] if field == "master"
                      else part["base_optimizer_state"]["state"][key][field] for part in converted]
            assert torch.equal(torch.cat(shards)[:useful], source)
            assert torch.count_nonzero(torch.cat(shards)[useful:]) == 0
        for rank in (0, 1):
            assert converted[rank]["base_optimizer_state"]["state"][key]["step"].item() == 3
            assert converted[rank]["group_paddings"][i] == targets[rank].group_paddings[i]

    gradients = []
    for i, group in enumerate(full.param_groups):
        full_param = group["params"][0]
        grad = torch.arange(full_param.numel(), dtype=torch.float32).add_(4 + i) / 13
        gradients.append(grad)
        full_param.grad = grad.clone()
    for rank, target in enumerate(targets):
        target.optimizer.load_state_dict(converted[rank]["base_optimizer_state"])
        for i, group in enumerate(target.optimizer.param_groups):
            parameter = group["params"][0]
            parameter.data.copy_(converted[rank]["single_partition_of_fp32_groups"][i])
            start = rank * parameter.numel()
            parameter.grad = torch.zeros_like(parameter)
            overlap = min(max(gradients[i].numel() - start, 0), parameter.numel())
            if overlap:
                parameter.grad[:overlap].copy_(gradients[i][start:start + overlap])
        target.optimizer.step()
    full.step()
    for i, group in enumerate(full.param_groups):
        actual = torch.cat([target.optimizer.param_groups[i]["params"][0].detach() for target in targets])
        assert torch.allclose(actual[:group["params"][0].numel()], group["params"][0].detach(), atol=1e-7)


def _check_rejects_bad_source_and_mapping_and_unknown_state():
    full = _full_optimizer()
    saved = _saved(full)
    target = _layout(full, rank=0)
    cases = []
    wrong_dp = deepcopy(saved)
    wrong_dp["partition_count"] = [2, 2]
    cases.append((wrong_dp, "source optimizer was not saved with DP=1"))
    wrong_order = deepcopy(saved)
    wrong_order["param_slice_mappings"][0] = OrderedDict(reversed(list(wrong_order["param_slice_mappings"][0].items())))
    cases.append((wrong_order, "parameter name/order mismatch"))
    unknown = deepcopy(saved)
    unknown["base_optimizer_state"]["state"][0]["unknown"] = torch.zeros(5)
    cases.append((unknown, "unsupported AdamW state fields"))
    for case, message in cases:
        with unittest.TestCase().assertRaisesRegex(ValueError, message):
            repartition_single_rank_bf16_state(case, target, 0, 2)


def _check_source_alignment_padding_is_removed_before_repartition():
    full = _full_optimizer()
    saved = _saved(full)
    # DP=1 BF16 alignment is two elements: a five-element group has one pad.
    saved["group_paddings"][0] = 1
    saved["single_partition_of_fp32_groups"][0] = torch.cat(
        (saved["single_partition_of_fp32_groups"][0], torch.zeros(1)))
    key = saved["base_optimizer_state"]["param_groups"][0]["params"][0]
    for field in ("exp_avg", "exp_avg_sq"):
        saved["base_optimizer_state"]["state"][key][field] = torch.cat(
            (saved["base_optimizer_state"]["state"][key][field], torch.zeros(1)))
    outputs = [repartition_single_rank_bf16_state(saved, _layout(full, rank), rank, 2) for rank in (0, 1)]
    for field in ("master", "exp_avg", "exp_avg_sq"):
        parts = [item["single_partition_of_fp32_groups"][0] if field == "master"
                 else item["base_optimizer_state"]["state"][key][field] for item in outputs]
        source = (saved["single_partition_of_fp32_groups"][0] if field == "master"
                  else saved["base_optimizer_state"]["state"][key][field])
        assert torch.equal(torch.cat(parts)[:5], source[:5])
        assert torch.count_nonzero(torch.cat(parts)[5:]) == 0


def _fake_trainer(monkeypatch):
    package = ModuleType("transformers")
    package.__path__ = []
    trainer_module = ModuleType("transformers.trainer")
    trainer_module.deepspeed_load_checkpoint = lambda *_args, **_kwargs: None
    package.trainer = trainer_module
    monkeypatch.setitem(sys.modules, "transformers", package)
    monkeypatch.setitem(sys.modules, "transformers.trainer", trainer_module)
    return trainer_module


def _check_scope_restores_global_and_engine_methods_on_error(monkeypatch):
    trainer_module = _fake_trainer(monkeypatch)

    class Engine:
        def _load_zero_checkpoint(self, *_args, **_kwargs):
            return True

    engine = Engine()
    original_method = engine._load_zero_checkpoint.__func__

    def entry(selected, *_args, **_kwargs):
        assert "_load_zero_checkpoint" in selected.__dict__
        raise RuntimeError("synthetic loader failure")

    monkeypatch.setattr(trainer_module, "deepspeed_load_checkpoint", entry)
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "synthetic loader failure"):
        with bf16_cross_world_resume():
            trainer_module.deepspeed_load_checkpoint(engine, "unused")
    assert trainer_module.deepspeed_load_checkpoint is entry
    assert "_load_zero_checkpoint" not in engine.__dict__
    assert engine._load_zero_checkpoint.__func__ is original_method


def _check_disabled_scope_does_not_patch(monkeypatch):
    trainer_module = _fake_trainer(monkeypatch)
    trainer_module.deepspeed_load_checkpoint = lambda *_args: None
    original = trainer_module.deepspeed_load_checkpoint
    with bf16_cross_world_resume(enabled=False):
        assert trainer_module.deepspeed_load_checkpoint is original


def _check_scope_restores_global_and_engine_methods_on_success(monkeypatch):
    trainer_module = _fake_trainer(monkeypatch)

    class Engine:
        loaded_checkpoint_dp_world_size = 1
        seq_dp_world_size = 1

        def _load_zero_checkpoint(self, load_dir, tag, load_optimizer_states=True):
            return load_dir, tag, load_optimizer_states

    def entry(engine, checkpoint_path, load_module_strict=True):
        return engine._load_zero_checkpoint(checkpoint_path, "tag", load_optimizer_states=load_module_strict)

    trainer_module.deepspeed_load_checkpoint = entry
    engine = Engine()
    with bf16_cross_world_resume():
        assert trainer_module.deepspeed_load_checkpoint(engine, "path", load_module_strict=False) == ("path", "tag", False)
        assert "_load_zero_checkpoint" not in engine.__dict__
    assert trainer_module.deepspeed_load_checkpoint is entry
    assert "_load_zero_checkpoint" not in engine.__dict__


def _check_rejects_non_stage_zero_cross_world(monkeypatch):
    module = ModuleType("deepspeed.runtime.bf16_optimizer")

    class FakeBF16Optimizer:
        pass

    module.BF16_Optimizer = FakeBF16Optimizer
    monkeypatch.setitem(sys.modules, "deepspeed.runtime.bf16_optimizer", module)
    engine = SimpleNamespace(loaded_checkpoint_dp_world_size=1, seq_dp_world_size=2,
                             optimizer=FakeBF16Optimizer(), bfloat16_enabled=lambda: True,
                             zero_optimization_stage=lambda: 1)
    with unittest.TestCase().assertRaisesRegex(ValueError, "only BF16 ZeRO stage 0"):
        _cross_world_zero_checkpoint(engine, lambda *_args, **_kwargs: True, "path", "tag")


class _MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        old = getattr(obj, name)
        setattr(obj, name, value)
        self._undo.append(lambda: setattr(obj, name, old))

    def setitem(self, mapping, key, value):
        existed = key in mapping
        old = mapping.get(key)
        mapping[key] = value
        if existed:
            self._undo.append(lambda: mapping.__setitem__(key, old))
        else:
            self._undo.append(lambda: mapping.pop(key, None))

    def undo(self):
        for action in reversed(self._undo):
            action()


class BF16ResumeTests(unittest.TestCase):
    def _with_patch(self, function):
        patch = _MonkeyPatch()
        try:
            function(patch)
        finally:
            patch.undo()

    def test_repartition_recovers_master_moments_and_one_adamw_step(self):
        _check_repartition_recovers_master_moments_and_one_adamw_step()

    def test_rejects_bad_source_and_mapping_and_unknown_state(self):
        _check_rejects_bad_source_and_mapping_and_unknown_state()

    def test_source_alignment_padding_is_removed_before_repartition(self):
        _check_source_alignment_padding_is_removed_before_repartition()

    def test_scope_restores_global_and_engine_methods_on_error(self):
        self._with_patch(_check_scope_restores_global_and_engine_methods_on_error)

    def test_scope_restores_global_and_engine_methods_on_success(self):
        self._with_patch(_check_scope_restores_global_and_engine_methods_on_success)

    def test_disabled_scope_does_not_patch(self):
        self._with_patch(_check_disabled_scope_does_not_patch)

    def test_rejects_non_stage_zero_cross_world(self):
        self._with_patch(_check_rejects_non_stage_zero_cross_world)
