from __future__ import annotations

import ast
import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _extract_function(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    function = next(node for node in functions if node.name == name)
    namespace = {
        "Mapping": Mapping,
        "Path": Path,
        "hashlib": hashlib,
        "os": os,
    }

    def _benchmark_v2_value(value, key, default=None):
        if isinstance(value, Mapping):
            return value.get(key, default)
        return getattr(value, key, default)

    namespace["_benchmark_v2_value"] = _benchmark_v2_value
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class BenchmarkV2AssetPlumbingTests(unittest.TestCase):
    def test_training_declares_and_applies_asset_manifest_cli_override(self):
        source = (REPO_ROOT / "train_csgo.py").read_text(encoding="utf-8")
        tree = ast.parse(source, filename="train_csgo.py")
        data_arguments = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DataArguments"
        )
        fields = {
            node.target.id
            for node in data_arguments.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertIn("benchmark_v2_asset_manifest", fields)
        self.assertIn("benchmark_v2_asset_manifest", source)
        self.assertIn("benchmark_v2_image_path(", source)

    def test_generation_and_localization_use_core_image_resolver(self):
        for filename in ("eval_csgo.py", "eval_csgo_loc.py"):
            source = (REPO_ROOT / filename).read_text(encoding="utf-8")
            self.assertIn("from csgo_datasets.benchmark_v2 import (", source)
            self.assertIn("benchmark_v2_image_path,", source)
            self.assertIn("fps_path = benchmark_v2_image_path(", source)
            self.assertIn("--benchmark_v2_asset_manifest", source)

    def test_minimal_provenance_contains_manifest_sha_and_backend(self):
        for filename in ("eval_csgo.py", "eval_csgo_loc.py"):
            writer = _extract_function(REPO_ROOT / filename, "_benchmark_v2_asset_provenance")
            with self.subTest(filename=filename):
                with self.subTest(mode="source"):
                    self.assertEqual(
                        writer({"benchmark_v2_asset_manifest": None}, SimpleNamespace()),
                        {},
                    )
                with self.subTest(mode="minimal"):
                    import tempfile

                    with tempfile.TemporaryDirectory() as temp_dir:
                        report = Path(temp_dir) / "minimal_dataset_report.json"
                        report.write_text('{"schema_version": 1}\n', encoding="utf-8")
                        digest = hashlib.sha256(report.read_bytes()).hexdigest()
                        result = writer(
                            {"benchmark_v2_asset_manifest": str(report)},
                            SimpleNamespace(
                                asset_backend="flat_minimal",
                                asset_manifest_sha256=digest,
                                asset_root="images",
                                selected_images_sha256="a" * 64,
                            ),
                        )
                        self.assertEqual(
                            result["benchmark_v2_asset_manifest"], str(report)
                        )
                        self.assertEqual(
                            result["benchmark_v2_asset_manifest_sha256"], digest
                        )
                        self.assertEqual(
                            result["benchmark_v2_asset_backend"], "flat_minimal"
                        )
                        self.assertEqual(result["benchmark_v2_asset_root"], "images")
                        self.assertEqual(
                            result["benchmark_v2_selected_images_sha256"],
                            "a" * 64,
                        )


if __name__ == "__main__":
    unittest.main()
