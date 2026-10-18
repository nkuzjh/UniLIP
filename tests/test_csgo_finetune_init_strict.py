"""CPU-only checks of model-only stage-two initialization coverage."""
from __future__ import annotations

import ast
import logging
from pathlib import Path
import unittest
from types import SimpleNamespace

import torch


SOURCE = Path(__file__).resolve().parents[1] / "train_csgo.py"
TREE = ast.parse(SOURCE.read_text())
FUNCTION = next(node for node in TREE.body if isinstance(node, ast.FunctionDef)
                and node.name == "load_finetune_init_checkpoint")
NAMESPACE = {"logging": logging, "smart_matching_state_dict_keys": lambda state, model: state}
exec(compile(ast.Module(body=[FUNCTION], type_ignores=[]), str(SOURCE), "exec"), NAMESPACE)
load_finetune_init_checkpoint = NAMESPACE["load_finetune_init_checkpoint"]


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.trainable = torch.nn.Parameter(torch.tensor([0.0]))
        self.frozen = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)


class StrictFinetuneInitTests(unittest.TestCase):
    def setUp(self):
        self.source = {}
        NAMESPACE["load_checkpoint_state_dict"] = lambda path: self.source

    def test_default_still_allows_missing_frozen_weight(self):
        model = SmallModel()
        self.source = {"trainable": torch.tensor([3.0])}
        load_finetune_init_checkpoint(model, "dummy.safetensors")
        self.assertEqual(model.trainable.item(), 3.0)
        self.assertEqual(model.frozen.item(), 0.0)
        with self.assertRaisesRegex(RuntimeError, "missing_non_alias"):
            load_finetune_init_checkpoint(SmallModel(), "dummy.safetensors", strict=True)

    def test_strict_requires_shape_and_all_trainable_weights(self):
        model = SmallModel()
        self.source = {"trainable": torch.tensor([1.0, 2.0]),
                       "frozen": torch.tensor([1.0])}
        with self.assertRaisesRegex(RuntimeError, "shape mismatches"):
            load_finetune_init_checkpoint(model, "dummy.safetensors", strict=True)
        self.source = {"frozen": torch.tensor([1.0])}
        with self.assertRaisesRegex(RuntimeError, "all parameters"):
            load_finetune_init_checkpoint(model, "dummy.safetensors", strict=True)
        self.source = {"trainable": torch.tensor([1.0]),
                       "frozen": torch.tensor([2.0]),
                       "unexpected": torch.tensor([3.0])}
        with self.assertRaisesRegex(RuntimeError, "unexpected=1"):
            load_finetune_init_checkpoint(model, "dummy.safetensors", strict=True)

    def test_strict_accepts_only_identical_tied_views(self):
        model = torch.nn.Module()
        model.config = SimpleNamespace()
        model.canonical = torch.nn.Parameter(torch.tensor([0.0]))
        model.alias = torch.nn.Parameter(model.canonical.data)
        self.source = {"alias": torch.tensor([7.0])}
        load_finetune_init_checkpoint(model, "dummy.safetensors", strict=True)
        self.assertEqual(model.canonical.item(), 7.0)
        self.assertEqual(model.alias.item(), 7.0)

        different_view = torch.nn.Module()
        different_view.canonical = torch.nn.Parameter(torch.zeros(2, 2))
        different_view.alias = torch.nn.Parameter(different_view.canonical.data.T)
        self.source = {"alias": torch.ones(2, 2)}
        with self.assertRaisesRegex(RuntimeError, "all parameters"):
            load_finetune_init_checkpoint(different_view, "dummy.safetensors", strict=True)


if __name__ == "__main__":
    unittest.main()
