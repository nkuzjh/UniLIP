"""CPU checks for the opt-in few-shot batch sampler epoch forwarding."""

import ast
import types
import unittest
from pathlib import Path

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]


def _load_sampler_and_dataloader_method():
    """Execute only the real sampler, loader and loader factory; avoid model imports."""
    tree = ast.parse((ROOT / "unilip/train/nonmix_trainer.py").read_text())
    class_names = {"DistributedTaskTypeBatchSampler", "SamplerEpochDataLoader"}
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in class_names
    ]
    trainer = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NonMixTrainer"
    )
    get_loader = next(
        node for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_train_dataloader"
    )
    nodes.append(ast.ClassDef(
        name="LoaderFactory", bases=[], keywords=[], body=[get_loader], decorator_list=[]
    ))
    namespace = {
        "DataLoader": DataLoader,
        "BatchSampler": BatchSampler,
        "Dataset": Dataset,
        "torch": torch,
        "math": __import__("math"),
        "dist": torch.distributed,
        "Iterator": __import__("typing").Iterator,
        "Optional": __import__("typing").Optional,
        "List": __import__("typing").List,
    }
    code = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(code, str(ROOT / "unilip/train/nonmix_trainer.py"), "exec"), namespace)
    return namespace["LoaderFactory"], namespace["SamplerEpochDataLoader"]


class SupportDataset(Dataset):
    def __init__(self, shots, maps=4):
        self.list_data_dict = {
            "type": [map_name for map_name in range(maps) for _ in range(shots)]
        }

    def __len__(self):
        return len(self.list_data_dict["type"])

    def __getitem__(self, index):
        return index


class FewshotSamplerEpochTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.LoaderFactory, cls.SamplerEpochDataLoader = _load_sampler_and_dataloader_method()

    def make_loaders(self, shots, forward, maps=4, replicas=4):
        dataset = SupportDataset(shots, maps=maps)
        loaders = []
        for rank in range(replicas):
            factory = self.LoaderFactory()
            factory.train_dataset = dataset
            factory._train_batch_size = 4
            factory.data_collator = None
            factory.args = types.SimpleNamespace(
                world_size=replicas,
                process_index=rank,
                dataloader_drop_last=False,
                dataloader_num_workers=0,
                dataloader_persistent_workers=False,
                dataloader_prefetch_factor=None,
                dataloader_pin_memory=False,
                **({"csgo_forward_sampler_epoch": True} if forward else {}),
            )
            loaders.append(factory.get_train_dataloader())
        return loaders

    def test_yaml_flag_validation_and_training_argument_assignment(self):
        tree = ast.parse((ROOT / "train_csgo.py").read_text())
        training_arguments = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TrainingArguments"
        )
        flag_field = next(
            node for node in training_arguments.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "csgo_forward_sampler_epoch"
        )
        self.assertIs(flag_field.value.value, False)

        train = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train"
        )
        start = next(
            index for index, node in enumerate(train.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "csgo_forward_sampler_epoch"
                for target in node.targets
            )
        )
        statements = train.body[start:start + 3]
        code = compile(
            ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])),
            str(ROOT / "train_csgo.py"),
            "exec",
        )
        for config, expected in (({}, False), ({"csgo_forward_sampler_epoch": True}, True)):
            args = types.SimpleNamespace()
            exec(code, {"csgo_config": config, "training_args": args})
            self.assertIs(args.csgo_forward_sampler_epoch, expected)
        with self.assertRaisesRegex(ValueError, "csgo_forward_sampler_epoch must be a boolean"):
            exec(code, {
                "csgo_config": {"csgo_forward_sampler_epoch": 1},
                "training_args": types.SimpleNamespace(),
            })

    @staticmethod
    def observed(loaders):
        return {int(index) for loader in loaders for batch in loader for index in batch}

    def test_default_still_uses_plain_loader_and_fixed_epoch(self):
        loaders = self.make_loaders(10, forward=False)
        self.assertTrue(all(type(loader) is DataLoader for loader in loaders))
        self.assertTrue(all(not hasattr(loader, "set_epoch") for loader in loaders))
        seen = self.observed(loaders)
        self.assertEqual(len(seen), 32)
        self.assertEqual(self.observed(loaders), seen)
        self.assertTrue(all(loader.batch_sampler.epoch == 0 for loader in loaders))

    def test_opt_in_forwards_epoch_and_uses_expected_microbatch_counts(self):
        for shots, per_map_batches in ((100, 25), (50, 12), (20, 5), (10, 2)):
            with self.subTest(shots=shots):
                loaders = self.make_loaders(shots, forward=True)
                self.assertTrue(all(type(loader) is self.SamplerEpochDataLoader for loader in loaders))
                self.assertEqual(sum(len(loader) for loader in loaders), per_map_batches * 4)
                for loader in loaders:
                    loader.set_epoch(7)
                    self.assertEqual(loader.batch_sampler.epoch, 7)
                self.assertTrue(all(len(batch) == 4 for loader in loaders for batch in loader))

    def test_dropped_support_rotates_at_50_and_10_shots(self):
        for shots in (50, 10):
            with self.subTest(shots=shots):
                loaders = self.make_loaders(shots, forward=True)
                all_support = set(range(4 * shots))
                initial = self.observed(loaders)
                self.assertEqual(len(all_support - initial), 8)
                seen = set(initial)
                for epoch in range(1, 400):
                    for loader in loaders:
                        loader.set_epoch(epoch)
                    seen.update(self.observed(loaders))
                self.assertEqual(seen, all_support)

    def test_single_map_single_gpu_batch_counts_and_support_coverage(self):
        for shots, expected_batches in ((100, 25), (50, 12), (20, 5), (10, 2)):
            with self.subTest(shots=shots):
                loaders = self.make_loaders(shots, forward=True, maps=1, replicas=1)
                self.assertEqual(len(loaders), 1)
                loader = loaders[0]
                self.assertEqual(len(loader), expected_batches)
                seen = set()
                for epoch in range(400):
                    loader.set_epoch(epoch)
                    self.assertEqual(loader.batch_sampler.epoch, epoch)
                    seen.update(self.observed(loaders))
                self.assertEqual(seen, set(range(shots)))


if __name__ == "__main__":
    unittest.main()
