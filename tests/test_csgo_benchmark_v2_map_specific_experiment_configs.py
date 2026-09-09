"""Contract tests for map-specific Benchmark v2 experiment configs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
import unittest

import yaml

from csgo_datasets.benchmark_v2 import load_benchmark_v2_selection


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "csgo_configs"
TEST_CONFIG_ROOT = CONFIG_ROOT / "test"
MANIFEST_PATH = REPO_ROOT / "data/csgo_benchmark_v2/benchmark_manifest.json"

MAPS = ("cs_office", "de_golden", "de_palacio", "de_vertigo")
MAP_FIELDS = ("train_maps", "val_maps", "test_maps")
KINDS = ("joint", "gen", "loc")
SHOTS = (100, 50, 20, 10)
FILE_FRAME_RE = re.compile(r"^file_num\d+_frame_\d+$")
MAP_SUBSET_SUMMARY_FLAG = "benchmark_v2_allow_map_subset_summary"
LOC_TEST_SOURCE_KEYS = frozenset({"joint_loc", "loc_loc"})

FAMILIES = {
    "exp35": {
        "train_sources": {
            "joint": "exp33.yaml",
            "gen": "exp33_gen.yaml",
            "loc": "exp33_loc.yaml",
        },
        "test_sources": {
            "joint_gen": "exp33_gen.yaml",
            "joint_gen_conti": "exp33_gen_conti.yaml",
            "joint_loc": "exp33_loc.yaml",
            "gen_gen": "exp33_gen_gen.yaml",
            "gen_gen_conti": "exp33_gen_gen_conti.yaml",
            "loc_loc": "exp33_loc_loc.yaml",
        },
        "parents": {"joint": "exp31", "gen": "exp31_gen", "loc": "exp31_loc"},
    },
    "exp36": {
        "train_sources": {
            "joint": "exp34.yaml",
            "gen": "exp34_gen.yaml",
            "loc": "exp34_loc.yaml",
        },
        "test_sources": {
            "joint_gen": "exp34_gen.yaml",
            "joint_gen_conti": "exp34_gen_conti.yaml",
            "joint_loc": "exp34_loc.yaml",
            "gen_gen": "exp34_gen_gen.yaml",
            "gen_gen_conti": "exp34_gen_gen_conti.yaml",
            "loc_loc": "exp34_loc_loc.yaml",
        },
        "parents": {"joint": "exp32", "gen": "exp32_gen", "loc": "exp32_loc"},
    },
}


def training_name(family: str, kind: str, map_name: str) -> str:
    if kind == "joint":
        return f"{family}_{map_name}.yaml"
    return f"{family}_{kind}_{map_name}.yaml"


def training_specs():
    for family, family_spec in FAMILIES.items():
        for map_name in MAPS:
            for kind in KINDS:
                yield family, family_spec, map_name, kind, training_name(
                    family, kind, map_name
                )


def test_specs():
    for family, family_spec in FAMILIES.items():
        for map_name in MAPS:
            joint = f"{family}_{map_name}"
            gen = f"{family}_gen_{map_name}"
            loc = f"{family}_loc_{map_name}"
            yield from (
                (
                    family,
                    family_spec,
                    map_name,
                    "joint_gen",
                    f"{joint}_gen.yaml",
                    False,
                    joint,
                ),
                (
                    family,
                    family_spec,
                    map_name,
                    "joint_gen_conti",
                    f"{joint}_gen_conti.yaml",
                    True,
                    joint,
                ),
                (
                    family,
                    family_spec,
                    map_name,
                    "joint_loc",
                    f"{joint}_loc.yaml",
                    False,
                    joint,
                ),
                (
                    family,
                    family_spec,
                    map_name,
                    "gen_gen",
                    f"{gen}_gen.yaml",
                    False,
                    gen,
                ),
                (
                    family,
                    family_spec,
                    map_name,
                    "gen_gen_conti",
                    f"{gen}_gen_conti.yaml",
                    True,
                    gen,
                ),
                (
                    family,
                    family_spec,
                    map_name,
                    "loc_loc",
                    f"{loc}_loc.yaml",
                    False,
                    loc,
                ),
            )


class CSGOBenchmarkV2MapSpecificExperimentConfigTest(unittest.TestCase):
    def _load_yaml(self, path: Path) -> dict:
        self.assertTrue(path.is_file(), f"missing YAML: {path}")
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.assertIsInstance(config, Mapping, f"YAML is not a mapping: {path}")
        return dict(config)

    def _load_training(self, name: str) -> dict:
        return self._load_yaml(CONFIG_ROOT / name)

    def _load_test(self, name: str) -> dict:
        return self._load_yaml(TEST_CONFIG_ROOT / name)

    def _runtime_config(self, config: dict, shots: int) -> dict:
        runtime = dict(config)
        runtime["benchmark_v2_manifest"] = str(MANIFEST_PATH)
        runtime["data_dir"] = str(REPO_ROOT / "data/preprocessed_data")
        runtime["benchmark_v2_shots_per_map"] = shots
        return runtime

    def test_all_24_training_and_48_inference_configs_exist(self):
        train_names = [spec[-1] for spec in training_specs()]
        test_names = [spec[4] for spec in test_specs()]
        self.assertEqual(len(train_names), 24)
        self.assertEqual(len(set(train_names)), 24)
        self.assertEqual(len(test_names), 48)
        self.assertEqual(len(set(test_names)), 48)

        for name in train_names:
            with self.subTest(path=f"csgo_configs/{name}"):
                self._load_training(name)
        for name in test_names:
            with self.subTest(path=f"csgo_configs/test/{name}"):
                self._load_test(name)

    def test_training_configs_equal_sources_except_singleton_maps(self):
        for family, family_spec, map_name, kind, name in training_specs():
            with self.subTest(config=name):
                actual = self._load_training(name)
                source = self._load_training(family_spec["train_sources"][kind])
                expected = dict(source)
                for key in MAP_FIELDS:
                    expected[key] = [map_name]
                self.assertEqual(actual, expected)

                for key in MAP_FIELDS:
                    self.assertEqual(actual[key], [map_name])
                self.assertEqual(actual["benchmark_v2_split"], "crossmap_support")
                self.assertEqual(actual["benchmark_v2_support_seed"], 0)
                self.assertEqual(actual["benchmark_v2_shots_per_map"], 100)
                self.assertEqual(
                    actual["finetune_init_ckpt_path"],
                    f"outputs/csgo_1b/{family_spec['parents'][kind]}/model.safetensors",
                )

    def test_inference_configs_equal_sources_except_checkpoint_and_maps(self):
        expected_splits = {
            False: "crossmap_query_test",
            True: "crossmap_continuous",
        }
        for (
            family,
            family_spec,
            map_name,
            source_key,
            name,
            continuous,
            experiment,
        ) in test_specs():
            with self.subTest(config=name):
                actual = self._load_test(name)
                source = self._load_test(family_spec["test_sources"][source_key])
                expected = dict(source)
                for key in MAP_FIELDS:
                    expected[key] = [map_name]
                expected["ckpt_path"] = (
                    f"outputs/csgo_1b/{experiment}/shot_100/seed_0/model.safetensors"
                )
                if source_key in LOC_TEST_SOURCE_KEYS:
                    expected[MAP_SUBSET_SUMMARY_FLAG] = True
                self.assertEqual(actual, expected)

                for key in MAP_FIELDS:
                    self.assertEqual(actual[key], [map_name])
                self.assertEqual(actual["benchmark_v2_split"], expected_splits[continuous])
                self.assertEqual(actual["benchmark_v2_support_seed"], 0)
                self.assertEqual(actual["benchmark_v2_shots_per_map"], 100)
                self.assertEqual(
                    actual["ckpt_path"],
                    f"outputs/csgo_1b/{experiment}/shot_100/seed_0/model.safetensors",
                )
                if continuous:
                    self.assertTrue(actual.get("is_conti_gen"))
                    self.assertEqual(
                        actual["benchmark_v2_split"], "crossmap_continuous"
                    )
                else:
                    self.assertNotIn("is_conti_gen", actual)
                    self.assertEqual(
                        actual["benchmark_v2_split"], "crossmap_query_test"
                    )

    def test_map_subset_summary_flag_is_enabled_only_for_localization(self):
        flagged_configs = []
        for (
            family,
            family_spec,
            map_name,
            source_key,
            name,
            continuous,
            experiment,
        ) in test_specs():
            with self.subTest(config=name):
                config = self._load_test(name)
                if source_key in LOC_TEST_SOURCE_KEYS:
                    self.assertIs(config.get(MAP_SUBSET_SUMMARY_FLAG), True)
                    flagged_configs.append(name)
                else:
                    self.assertNotIn(MAP_SUBSET_SUMMARY_FLAG, config)

        self.assertEqual(len(flagged_configs), 16)
        for family_spec in FAMILIES.values():
            for source_name in set(family_spec["test_sources"].values()):
                with self.subTest(old_config=source_name):
                    self.assertNotIn(
                        MAP_SUBSET_SUMMARY_FLAG,
                        self._load_test(source_name),
                    )

    def test_loader_selects_requested_map_and_each_shot_count(self):
        for family, family_spec, map_name, kind, name in training_specs():
            with self.subTest(config=name):
                config = self._load_training(name)
                selection_100 = load_benchmark_v2_selection(
                    self._runtime_config(config, 100),
                    map_names=config["train_maps"],
                )
                self.assertEqual(selection_100.split, "crossmap_support")
                self.assertEqual(selection_100.map_names, [map_name])
                self.assertEqual(selection_100.support_seed, 0)
                self.assertEqual(selection_100.shots_per_map, 100)
                self.assertEqual(len(selection_100.rows), 100)
                self.assertEqual(
                    {row["map"] for row in selection_100.rows},
                    {map_name},
                )
                self.assertTrue(
                    all(
                        FILE_FRAME_RE.fullmatch(row["file_frame"])
                        for row in selection_100.rows
                    )
                )
                ids_100 = [row["file_frame"] for row in selection_100.rows]
                self.assertEqual(len(ids_100), len(set(ids_100)))

                for shots in SHOTS[1:]:
                    with self.subTest(shots=shots):
                        selection = load_benchmark_v2_selection(
                            self._runtime_config(config, shots),
                            map_names=config["train_maps"],
                        )
                        self.assertEqual(selection.split, "crossmap_support")
                        self.assertEqual(selection.map_names, [map_name])
                        self.assertEqual(selection.support_seed, 0)
                        self.assertEqual(selection.shots_per_map, shots)
                        self.assertEqual(len(selection.rows), shots)
                        self.assertEqual(
                            {row["map"] for row in selection.rows},
                            {map_name},
                        )
                        self.assertEqual(
                            [row["file_frame"] for row in selection.rows],
                            ids_100[:shots],
                        )
                        self.assertTrue(
                            all(
                                FILE_FRAME_RE.fullmatch(row["file_frame"])
                                for row in selection.rows
                            )
                        )
                        split_file = selection.split_files[map_name]
                        self.assertEqual(split_file.parent.name, map_name)
                        self.assertEqual(split_file.parent.parent.name, "crossmap")


if __name__ == "__main__":
    unittest.main()
