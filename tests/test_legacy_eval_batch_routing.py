"""Static guard for older runners whose output directories are batch-one artifacts."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def batch_options_in_function(filename: str, name: str) -> list[tuple[str, str]]:
    tree = ast.parse((ROOT / "scripts" / filename).read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    options = []
    for node in ast.walk(function):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        for flag, value in zip(node.elts, node.elts[1:]):
            if isinstance(flag, ast.Constant) and flag.value in ("--test-loc-batch-size", "--test-gen-batch-size"):
                self_value = value.value if isinstance(value, ast.Constant) else None
                options.append((flag.value, self_value))
    return options


class LegacyEvalBatchRoutingTests(unittest.TestCase):
    def test_old_eval_commands_pin_batch_one(self):
        expected = (
            ("run_exp32_3maps.py", "evaluation_plan", ["--test-loc-batch-size", "--test-gen-batch-size"]),
            ("run_csgo_benchmark_v2_checkpoint_eval.py", "build_localization_command", ["--test-loc-batch-size"]),
            ("run_csgo_benchmark_v2_checkpoint_eval.py", "build_generation_inference_command", ["--test-gen-batch-size"]),
            ("run_csgo_benchmark_v2_gen.py", "build_inference_command", ["--test-gen-batch-size"]),
            ("run_csgo_benchmark_v2_loc_few_shot.py", "build_eval_command", ["--test-loc-batch-size"]),
            ("run_csgo_benchmark_v2_map_specific.py", "build_generation_inference_command", ["--test-gen-batch-size"]),
            ("run_csgo_benchmark_v2_map_specific.py", "build_localization_inference_command", ["--test-loc-batch-size"]),
        )
        for filename, function, flags in expected:
            with self.subTest(filename=filename, function=function):
                self.assertCountEqual(batch_options_in_function(filename, function),
                                      [(flag, "1") for flag in flags])
        self.assertEqual(batch_options_in_function("run_csgo_benchmark_v2_map_specific.py", "build_train_command"), [])
        self.assertEqual(batch_options_in_function("run_csgo_benchmark_v2_gen.py", "build_train_command"), [])
        self.assertEqual(batch_options_in_function("run_csgo_benchmark_v2_loc_few_shot.py", "build_train_command"), [])


if __name__ == "__main__":
    unittest.main()
