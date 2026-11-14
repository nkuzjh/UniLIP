"""CPU checks for preserving global batches and restoring single-rank RNG."""

import ast
import hashlib
import logging
import math
import os
import random
import tempfile
import types
import unittest
from pathlib import Path
from typing import Iterator, List, Optional
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset


SOURCE = Path(__file__).resolve().parents[1] / "unilip/train/nonmix_trainer.py"


def load_components():
    tree = ast.parse(SOURCE.read_text())
    names = {"DistributedTaskTypeBatchSampler", "DistributedGlobalBatchSampler"}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NonMixTrainer")
    method = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "_load_rng_state")
    nodes.append(ast.ClassDef(name="RngLoader", bases=[], keywords=[], body=[method], decorator_list=[]))
    scope = dict(torch=torch, np=np, random=random, hashlib=hashlib, os=os, math=math,
                 BatchSampler=BatchSampler, Dataset=Dataset, Iterator=Iterator,
                 List=List, Optional=Optional, dist=torch.distributed,
                 logger=logging.getLogger(__name__),
                 ParallelMode=types.SimpleNamespace(DISTRIBUTED="distributed"))
    for device in ("xla", "npu", "hpu", "mlu", "musa"):
        scope[f"is_torch_{device}_available"] = lambda: False
    scope["set_rng_state_for_device"] = lambda *args: None
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), "exec"), scope)
    return scope


class Rows(Dataset):
    def __init__(self, count=50000):
        self.list_data_dict = {"type": ["gen"] * count}


class GlobalBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = load_components()

    def test_original_global_batch_sequence_survives_repartition_and_resume_skip(self):
        data = Rows()
        old = self.scope["DistributedTaskTypeBatchSampler"](
            data, batch_size=128, num_replicas=1, rank=0, drop_last=False)
        old_batches = list(old)
        self.assertEqual(len(old_batches), 390)
        for world, micro, gas in ((1, 128, 1), (2, 64, 1), (2, 32, 2), (4, 16, 2)):
            with self.subTest(world=world, micro=micro, gas=gas):
                samplers = [self.scope["DistributedGlobalBatchSampler"](
                    data, batch_size=micro, global_batch_size=128, num_replicas=world,
                    rank=rank, gradient_accumulation_steps=gas) for rank in range(world)]
                streams = [list(s) for s in samplers]
                self.assertTrue(all(len(s) == 390 * gas for s in samplers))
                recovered = [[item for offset in range(gas) for rank in range(world)
                              for item in streams[rank][step * gas + offset]]
                             for step in range(390)]
                self.assertEqual(recovered, old_batches)
                # Checkpoint-4000 resumes at epoch 10, update 100 within that epoch.
                self.assertEqual(recovered[4000 % 390:], old_batches[100:])
                self.assertEqual(len(streams[0]) // gas * 35, 13650)

    def test_inconsistent_effective_batch_fails(self):
        with self.assertRaisesRegex(ValueError, "must equal"):
            self.scope["DistributedGlobalBatchSampler"](
                Rows(300), batch_size=64, global_batch_size=128,
                num_replicas=2, rank=0, gradient_accumulation_steps=2)

    def test_global_batches_remain_task_homogeneous(self):
        data = Rows(600)
        data.list_data_dict["type"] = ["gen"] * 300 + ["loc"] * 300
        sampler = self.scope["DistributedGlobalBatchSampler"](
            data, batch_size=64, global_batch_size=128, num_replicas=2,
            rank=0, gradient_accumulation_steps=1)
        self.assertEqual(len(sampler), 4)
        for batch in sampler:
            self.assertEqual(len({data.list_data_dict["type"][i] for i in batch}), 1)


class ResumeRngTests(unittest.TestCase):
    def setUp(self):
        self.scope = load_components()
        self.py_state = random.getstate()
        self.np_state = np.random.get_state()
        self.cpu_state = torch.get_rng_state()
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.saved = {"python": self.py_state, "numpy": self.np_state,
                      "cpu": self.cpu_state, "cuda": [torch.tensor([1, 2, 3], dtype=torch.uint8)]}
        torch.save(self.saved, self.path / "rng_state.pth")

    def tearDown(self):
        random.setstate(self.py_state)
        np.random.set_state(self.np_state)
        torch.set_rng_state(self.cpu_state)
        self.temp.cleanup()

    def loader(self, rank):
        loader = self.scope["RngLoader"]()
        loader.args = types.SimpleNamespace(world_size=2, process_index=rank,
                                           parallel_mode="distributed",
                                           resume_allow_bf16_world_size_change=True)
        return loader

    def test_saved_cuda_list_or_tensor_restores_current_device(self):
        for cuda_state in (self.saved["cuda"], self.saved["cuda"][0]):
            self.saved["cuda"] = cuda_state
            torch.save(self.saved, self.path / "rng_state.pth")
            with patch.object(torch.cuda, "is_available", return_value=True), \
                 patch.object(torch.cuda, "current_device", return_value=1), \
                 patch.object(torch.cuda, "set_rng_state") as restore:
                self.loader(0)._load_rng_state(str(self.path))
                self.assertEqual(restore.call_args.kwargs["device"], 1)
                self.assertTrue(torch.equal(restore.call_args.args[0], torch.tensor([1, 2, 3], dtype=torch.uint8)))

    def test_added_rank_has_repeatable_independent_streams(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            self.loader(1)._load_rng_state(str(self.path))
            first = (random.random(), float(np.random.rand()), torch.rand(4))
            self.loader(1)._load_rng_state(str(self.path))
            second = (random.random(), float(np.random.rand()), torch.rand(4))
            self.assertEqual(first[:2], second[:2])
            self.assertTrue(torch.equal(first[2], second[2]))
            self.loader(0)._load_rng_state(str(self.path))
            self.assertFalse(torch.equal(first[2], torch.rand(4)))


if __name__ == "__main__":
    unittest.main()
