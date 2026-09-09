"""Contract tests for the Benchmark v2 experiment configuration matrix."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import unittest

import yaml

from csgo_datasets.benchmark_v2 import load_benchmark_v2_selection


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "csgo_configs"
TEST_CONFIG_ROOT = CONFIG_ROOT / "test"
MANIFEST_PATH = REPO_ROOT / "data/csgo_benchmark_v2/benchmark_manifest.json"

SEEN_MAPS = [
    "cs_agency",
    "cs_italy",
    "de_ancient",
    "de_anubis",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
    "de_overpass",
    "de_train",
]
CROSSMAP_MAPS = ["cs_office", "de_golden", "de_palacio", "de_vertigo"]
DEFAULT_CROSSMAP_SUPPORT_SEED = 0

TRAINING_CONFIGS = [
    "exp31.yaml",
    "exp31_1.yaml",
    "exp31_loc.yaml",
    "exp31_gen.yaml",
    "exp32.yaml",
    "exp32_loc.yaml",
    "exp32_gen.yaml",
    "exp33.yaml",
    "exp33_loc.yaml",
    "exp33_gen.yaml",
    "exp34.yaml",
    "exp34_loc.yaml",
    "exp34_gen.yaml",
]

TEST_CONFIGS = [
    f"{family}{suffix}.yaml"
    for family in ("exp31", "exp32", "exp33", "exp34")
    for suffix in (
        "_loc",
        "_gen",
        "_gen_conti",
        "_loc_loc",
        "_gen_gen",
        "_gen_gen_conti",
    )
]
TEST_CONFIGS.extend(
    [
        "exp31_1_gen.yaml",
        "exp31_1_gen_conti.yaml",
        "exp31_1_loc.yaml",
    ]
)

DIRECT_PARENTS = {
    "exp31.yaml": "exp28_1.yaml",
    "exp31_loc.yaml": "exp14_3_loc.yaml",
    "exp31_gen.yaml": "exp14_3_gen.yaml",
    "exp32.yaml": "exp30_2.yaml",
    "exp32_loc.yaml": "exp14_2_loc.yaml",
    "exp32_gen.yaml": "exp14_2_gen.yaml",
}

SEEN_TRAINING_CONFIGS = {"exp31.yaml", "exp31_loc.yaml", "exp31_gen.yaml"}
SEEN_TRAINING_CONFIGS.add("exp31_1.yaml")
CROSSMAP_TRAINING_CONFIGS = {
    "exp33.yaml",
    "exp33_loc.yaml",
    "exp33_gen.yaml",
    "exp34.yaml",
    "exp34_loc.yaml",
    "exp34_gen.yaml",
}

CROSSMAP_PARENTS = {
    "exp33.yaml": "exp31.yaml",
    "exp33_loc.yaml": "exp31_loc.yaml",
    "exp33_gen.yaml": "exp31_gen.yaml",
    "exp34.yaml": "exp32.yaml",
    "exp34_loc.yaml": "exp32_loc.yaml",
    "exp34_gen.yaml": "exp32_gen.yaml",
}

# These are the only fields allowed to differ between a Seen-10 config and
# its named three-map parent.  The map fields are checked separately below.
SEEN_PARENT_EXCEPTIONS = {
    "benchmark_v2_manifest",
    "benchmark_v2_split",
    "train_maps",
    "val_maps",
    "test_maps",
    "data_dir",
}

# CrossMap adaptation changes the protocol, initialization checkpoint, and
# the mature static loss values.  All other method/model fields must remain
# inherited from the Seen-10 counterpart.
CROSSMAP_EXCEPTIONS = {
    "benchmark_v2_manifest",
    "benchmark_v2_split",
    "benchmark_v2_support_seed",
    "benchmark_v2_shots_per_map",
    "finetune_init_ckpt_path",
    "train_maps",
    "val_maps",
    "test_maps",
    "data_dir",
    "alpha_loc_loss",
    "alpha_loc_schedule_steps",
    "alpha_loc_schedule_values",
    "alpha_loc_aux_loss",
    "alpha_loc_aux_schedule_steps",
    "alpha_loc_aux_schedule_values",
    "alpha_loc_perception_loss",
    "alpha_loc_perception_schedule_steps",
    "alpha_loc_perception_schedule_values",
}

NO_JOINT_SCHEDULE_KEYS = (
    "alpha_loc_schedule_steps",
    "alpha_loc_schedule_values",
    "alpha_loc_aux_schedule_steps",
    "alpha_loc_aux_schedule_values",
    "alpha_loc_perception_schedule_steps",
    "alpha_loc_perception_schedule_values",
)

# Test configs intentionally add inference-only fields.  For architecture or
# trainability fields present in a training config, however, equality is
# required.  Some fields are optional in the older single-task baselines.
CRITICAL_TRAINING_KEYS = (
    "is_multi_task",
    "is_multi_task_balanced",
    "task_mix_ratio",
    "is_lora",
    "llm_train_mode",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "enable_language_model_lora",
    "enable_gen_head_lora",
    "enable_loc_head_lora",
    "freeze_inactive_head",
    "freeze_gen_head",
    "freeze_loc_head",
    "fix_vit",
    "fix_llm",
    "fix_connect",
    "fix_dit",
    "is_action_dit_dense_timestep",
    "is_action_dit_projector",
    "action_dit_projector_lr",
    "action_dit_lr",
    "is_aciton_dit_vae_small_init",
    "use_pi05_action_dit",
    "pi05_pytorch_weight_path",
    "use_external_loc_model",
    "action_dim",
    "img_size",
    "use_short_instruction",
)


def _test_spec(family: str, suffix: str, training_name: str, split: str, ckpt: str):
    return {
        "name": f"{family}{suffix}.yaml",
        "training": training_name,
        "split": split,
        "continuous": suffix.endswith("_conti"),
        "ckpt": ckpt,
    }


TEST_MATRIX = {}
for _family in ("exp31", "exp32", "exp33", "exp34"):
    _is_crossmap = _family in ("exp33", "exp34")
    _maps_root = "outputs/csgo_1b/"
    _split_discrete = "crossmap_query_test" if _is_crossmap else "seen_discrete_test"
    _split_continuous = "crossmap_continuous" if _is_crossmap else "seen_continuous"
    _shot_suffix = (
        f"/shot_100/seed_{DEFAULT_CROSSMAP_SUPPORT_SEED}" if _is_crossmap else ""
    )
    _joint_ckpt = f"{_maps_root}{_family}{_shot_suffix}/model.safetensors"
    _loc_ckpt = f"{_maps_root}{_family}_loc{_shot_suffix}/model.safetensors"
    _gen_ckpt = f"{_maps_root}{_family}_gen{_shot_suffix}/model.safetensors"
    TEST_MATRIX[_family] = [
        _test_spec(_family, "_loc", f"{_family}.yaml", _split_discrete, _joint_ckpt),
        _test_spec(_family, "_gen", f"{_family}.yaml", _split_discrete, _joint_ckpt),
        _test_spec(
            _family,
            "_gen_conti",
            f"{_family}.yaml",
            _split_continuous,
            _joint_ckpt,
        ),
        _test_spec(
            _family,
            "_loc_loc",
            f"{_family}_loc.yaml",
            _split_discrete,
            _loc_ckpt,
        ),
        _test_spec(
            _family,
            "_gen_gen",
            f"{_family}_gen.yaml",
            _split_discrete,
            _gen_ckpt,
        ),
        _test_spec(
            _family,
            "_gen_gen_conti",
            f"{_family}_gen.yaml",
            _split_continuous,
            _gen_ckpt,
        ),
    ]

TEST_MATRIX["exp31_1"] = [
    _test_spec(
        "exp31_1",
        "_gen",
        "exp31_1.yaml",
        "seen_discrete_test",
        "outputs/csgo_1b/exp31_1/model.safetensors",
    ),
    _test_spec(
        "exp31_1",
        "_gen_conti",
        "exp31_1.yaml",
        "seen_continuous",
        "outputs/csgo_1b/exp31_1/model.safetensors",
    ),
    _test_spec(
        "exp31_1",
        "_loc",
        "exp31_1.yaml",
        "seen_discrete_test",
        "outputs/csgo_1b/exp31_1/model.safetensors",
    ),
]


class CSGOBenchmarkV2ExperimentConfigTest(unittest.TestCase):
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

    def _runtime_config(self, config: dict) -> dict:
        runtime_config = dict(config)
        runtime_config["benchmark_v2_manifest"] = str(MANIFEST_PATH)
        runtime_config["data_dir"] = str(REPO_ROOT / "data/preprocessed_data")
        return runtime_config

    def test_all_new_yaml_files_exist_and_parse_as_mappings(self):
        self.assertEqual(len(TRAINING_CONFIGS), 13)
        self.assertEqual(len(TEST_CONFIGS), 27)
        for name in TRAINING_CONFIGS:
            with self.subTest(path=name):
                self._load_training(name)
        for name in TEST_CONFIGS:
            with self.subTest(path=f"test/{name}"):
                self._load_test(name)

    def test_seen_training_configs_match_direct_parents(self):
        for child_name, parent_name in DIRECT_PARENTS.items():
            with self.subTest(child=child_name, parent=parent_name):
                child = self._load_training(child_name)
                parent = self._load_training(parent_name)
                for key in sorted(set(child) | set(parent)):
                    if key in SEEN_PARENT_EXCEPTIONS:
                        continue
                    self.assertIn(key, child, f"missing inherited key {key}")
                    self.assertIn(key, parent, f"missing parent key {key}")
                    self.assertEqual(
                        child[key],
                        parent[key],
                        f"unexpected {child_name} change for key {key}",
                    )
                for map_key in ("train_maps", "val_maps", "test_maps"):
                    self.assertEqual(child[map_key], SEEN_MAPS, map_key)
                self.assertEqual(child["benchmark_v2_split"], "seen_train")
                self.assertEqual(
                    child["benchmark_v2_manifest"],
                    "data/csgo_benchmark_v2/benchmark_manifest.json",
                )

    def test_exp31_1_is_exp31_strict_loss_only_control(self):
        child = self._load_training("exp31_1.yaml")
        parent = self._load_training("exp31.yaml")

        changed_loss_keys = {
            "is_loc_aux_loss",
            "is_aux_loc_combined_em_unc_loss",
            "is_loc_perception_loss",
            "alpha_loc_aux_schedule_steps",
            "alpha_loc_aux_schedule_values",
            "alpha_loc_perception_schedule_steps",
            "alpha_loc_perception_schedule_values",
        }
        for key in sorted(set(parent) - changed_loss_keys):
            with self.subTest(inherited_key=key):
                self.assertIn(key, child)
                self.assertEqual(child[key], parent[key])

        for key in ("alpha_loc_loss", "alpha_loc_schedule_steps", "alpha_loc_schedule_values"):
            with self.subTest(main_loc_schedule_key=key):
                self.assertEqual(child[key], parent[key])

        for map_key in ("train_maps", "val_maps", "test_maps"):
            self.assertEqual(child[map_key], SEEN_MAPS)
        self.assertEqual(child["benchmark_v2_manifest"], "data/csgo_benchmark_v2/benchmark_manifest.json")
        self.assertEqual(child["benchmark_v2_split"], "seen_train")
        self.assertEqual(child["task_mix_ratio"], 0.5)

        expected_trainability = {
            "is_lora": False,
            "llm_train_mode": "frozen",
            "fix_vit": True,
            "fix_llm": True,
            "fix_connect": False,
            "fix_dit": False,
            "freeze_gen_head": False,
            "freeze_loc_head": False,
        }
        for key, expected in expected_trainability.items():
            with self.subTest(trainability_key=key):
                self.assertIn(key, child)
                self.assertEqual(child[key], expected)

        for key, expected in {
            "learning_rate": 1.0e-4,
            "mm_projector_lr": 1.0e-4,
            "action_dit_connector_lr": 5.0e-4,
            "action_dit_norm_lr": 5.0e-4,
            "action_io_mlp_lr": 1.0e-4,
            "weight_decay": 0.0,
            "warmup_ratio": 0.003,
            "lr_scheduler_type": "cosine_with_min_lr",
        }.items():
            with self.subTest(optimizer_key=key):
                self.assertEqual(child[key], expected)
        self.assertEqual(child["lr_scheduler_kwargs"], {"min_lr": 1.0e-5})

        for key in (
            "is_loc_aux_loss",
            "is_aux_loc_em_loss",
            "is_aux_loc_uncertainty_loss",
            "is_aux_loc_combined_em_unc_loss",
            "is_loc_perception_loss",
            "is_loc_repa_loss",
            "is_noisy_loc_loss",
        ):
            with self.subTest(disabled_loss_key=key):
                self.assertFalse(child.get(key, False))
        self.assertEqual(child["alpha_loc_aux_loss"], 0.0)
        self.assertEqual(child["alpha_loc_perception_loss"], 0.0)
        for key in (
            "alpha_loc_aux_schedule_steps",
            "alpha_loc_aux_schedule_values",
            "alpha_loc_perception_schedule_steps",
            "alpha_loc_perception_schedule_values",
        ):
            self.assertNotIn(key, child)

    def test_crossmap_training_configs_match_seen_methods(self):
        for child_name, parent_name in CROSSMAP_PARENTS.items():
            with self.subTest(child=child_name, parent=parent_name):
                child = self._load_training(child_name)
                parent = self._load_training(parent_name)
                for key in sorted(set(child) | set(parent)):
                    if key in CROSSMAP_EXCEPTIONS:
                        continue
                    self.assertIn(key, child, f"missing inherited key {key}")
                    self.assertIn(key, parent, f"missing parent key {key}")
                    self.assertEqual(
                        child[key],
                        parent[key],
                        f"unexpected {child_name} change for key {key}",
                    )

                for map_key in ("train_maps", "val_maps", "test_maps"):
                    self.assertEqual(child[map_key], CROSSMAP_MAPS, map_key)
                self.assertEqual(child["benchmark_v2_split"], "crossmap_support")
                self.assertEqual(
                    child["benchmark_v2_support_seed"],
                    DEFAULT_CROSSMAP_SUPPORT_SEED,
                )
                self.assertEqual(child["benchmark_v2_shots_per_map"], 100)
                self.assertEqual(
                    child["benchmark_v2_manifest"],
                    "data/csgo_benchmark_v2/benchmark_manifest.json",
                )
                self.assertEqual(
                    child["finetune_init_ckpt_path"],
                    f"outputs/csgo_1b/{parent_name.removesuffix('.yaml')}/model.safetensors",
                )
                for key in NO_JOINT_SCHEDULE_KEYS:
                    self.assertNotIn(key, child)

                if child_name in {"exp33.yaml", "exp34.yaml"}:
                    self.assertEqual(child["alpha_loc_loss"], 20)

                if child_name == "exp33.yaml":
                    self.assertTrue(child["is_loc_aux_loss"])
                    self.assertTrue(child["is_aux_loc_combined_em_unc_loss"])
                    self.assertEqual(child["alpha_loc_aux_loss"], 2)
                    self.assertTrue(child["is_loc_perception_loss"])
                    self.assertEqual(child["alpha_loc_perception_loss"], 0.1)
                elif child_name == "exp34.yaml":
                    self.assertFalse(child["is_loc_aux_loss"])
                    self.assertFalse(child["is_aux_loc_combined_em_unc_loss"])
                    self.assertEqual(child["alpha_loc_aux_loss"], 0.0)
                    self.assertFalse(child["is_loc_perception_loss"])
                    self.assertEqual(child["alpha_loc_perception_loss"], 0.0)
                else:
                    for key in (
                        "alpha_loc_loss",
                        "alpha_loc_aux_loss",
                        "is_loc_aux_loss",
                        "is_aux_loc_combined_em_unc_loss",
                        "alpha_loc_perception_loss",
                        "is_loc_perception_loss",
                    ):
                        if key in parent:
                            self.assertEqual(child.get(key), parent[key], key)

    def test_each_test_config_has_exact_matrix_protocol_and_checkpoint(self):
        self.assertEqual(
            set(TEST_CONFIGS),
            {
                spec["name"]
                for family_specs in TEST_MATRIX.values()
                for spec in family_specs
            },
        )
        for family, specs in TEST_MATRIX.items():
            expected_spec_count = 3 if family == "exp31_1" else 6
            self.assertEqual(len(specs), expected_spec_count)
            expected_maps = CROSSMAP_MAPS if family in ("exp33", "exp34") else SEEN_MAPS
            for spec in specs:
                with self.subTest(config=spec["name"]):
                    config = self._load_test(spec["name"])
                    self.assertEqual(config["benchmark_v2_split"], spec["split"])
                    self.assertEqual(config["ckpt_path"], spec["ckpt"])
                    self.assertEqual(config["train_maps"], expected_maps)
                    self.assertEqual(config["val_maps"], expected_maps)
                    self.assertEqual(config["test_maps"], expected_maps)
                    self.assertEqual(
                        config["benchmark_v2_manifest"],
                        "data/csgo_benchmark_v2/benchmark_manifest.json",
                    )
                    if family in ("exp33", "exp34"):
                        self.assertEqual(
                            config["benchmark_v2_support_seed"],
                            DEFAULT_CROSSMAP_SUPPORT_SEED,
                        )
                        self.assertEqual(config["benchmark_v2_shots_per_map"], 100)
                        self.assertIn(
                            f"/seed_{DEFAULT_CROSSMAP_SUPPORT_SEED}/",
                            config["ckpt_path"],
                        )
                    if spec["continuous"]:
                        self.assertTrue(config.get("is_conti_gen"))
                    else:
                        self.assertNotIn("is_conti_gen", config)

    def test_test_configs_match_their_training_architecture_and_trainability(self):
        for family, specs in TEST_MATRIX.items():
            for spec in specs:
                with self.subTest(config=spec["name"]):
                    test_config = self._load_test(spec["name"])
                    train_config = self._load_training(spec["training"])
                    for key in CRITICAL_TRAINING_KEYS:
                        if key not in train_config:
                            continue
                        self.assertIn(
                            key,
                            test_config,
                            f"{spec['name']} missing training key {key}",
                        )
                        self.assertEqual(
                            test_config[key],
                            train_config[key],
                            f"{spec['name']} architecture/trainability mismatch: {key}",
                        )

    def test_selection_loader_loads_every_new_config_with_expected_counts(self):
        expected_training_counts = {
            "seen_train": 50000,
            "crossmap_support": 400,
        }
        expected_test_counts = {
            "seen_discrete_test": 20000,
            "seen_continuous": 12800,
            "crossmap_query_test": 8000,
            "crossmap_continuous": 5120,
        }

        for name in TRAINING_CONFIGS:
            with self.subTest(config=name):
                config = self._load_training(name)
                selection = load_benchmark_v2_selection(self._runtime_config(config))
                self.assertEqual(
                    len(selection.rows),
                    expected_training_counts[config["benchmark_v2_split"]],
                )

        for name in TEST_CONFIGS:
            with self.subTest(config=f"test/{name}"):
                config = self._load_test(name)
                selection = load_benchmark_v2_selection(self._runtime_config(config))
                self.assertEqual(
                    len(selection.rows),
                    expected_test_counts[config["benchmark_v2_split"]],
                )


if __name__ == "__main__":
    unittest.main()
