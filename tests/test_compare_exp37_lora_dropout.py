"""Contract tests for the exp37 dropout comparison, with small synthetic metrics."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.compare_exp37_lora_dropout import (
    CHECKPOINT_ID, MAPS, METRICS, ComparisonError, checkpoint_id, compare, main,
    markdown,
)


class ComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.on = root / "on" / "discrete"
        self.off = root / "off" / "discrete"
        self.modules = [{"name": f"model.layers.{i}.q_proj.lora_dropout.default", "p": 0.05}
                        for i in range(438)]
        self._write_run(self.on, "on")
        self._write_run(self.off, "off")

    @staticmethod
    def _write(path: Path, data: dict) -> None:
        path.write_text(json.dumps(data), encoding="utf-8")

    @staticmethod
    def _change(path: Path, update) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        update(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    def _write_run(self, root: Path, role: str) -> None:
        root.mkdir(parents=True, exist_ok=True)
        execution = {
            "schema_version": 1, "task": "gen", "batch_size": 8,
            "implementation": "unilip_inference_batch_v1",
            "rng_policy": "legacy_ordered_initial_noise_draws_v1",
            "sample_order_sha256": "a" * 64, "sample_count": 20000,
            "batch_count": 2500, "last_batch_size": 8, "max_batch_size": 8,
            "inference_steps": 20, "rng_device": "cuda:0", "rng_dtype": "torch.bfloat16",
            "model_mode_policy": ("preserve_loaded_module_modes_v1" if role == "on"
                                  else "disable_lora_dropout_v1"),
            "active_dropout_modules": self.modules if role == "on" else [],
            "stochastic_layers_batch_dependent": role == "on",
        }
        if role == "off":
            execution["disabled_lora_dropout_modules"] = self.modules
        asset_digest, selected_digest = "b" * 64, "c" * 64
        manifest = {
            "config_path": str(root / "evaluation_config.yaml"),
            "benchmark_v2_manifest": "data/csgo_benchmark_v2/benchmark_manifest.json",
            "benchmark_v2_split": "seen_discrete_test", "maps": list(MAPS),
            "sample_count": 20000, "checkpoint": CHECKPOINT_ID, "ckpt_path": CHECKPOINT_ID,
            "seed": 42, "benchmark_v2_asset_manifest": "data/csgo_benchmark_v2/minimal_dataset_report.json",
            "benchmark_v2_asset_backend": "minimal",
            "benchmark_v2_asset_manifest_sha256": asset_digest,
            "benchmark_v2_selected_images_sha256": selected_digest,
            "benchmark_v2_asset": {"manifest": "data/csgo_benchmark_v2/minimal_dataset_report.json",
                                   "backend": "minimal", "sha256": asset_digest,
                                   "selected_images_sha256": selected_digest},
            "inference_execution": execution,
        }
        self._write(root / "inference_manifest.json", manifest)
        (root / "evaluation_config.yaml").write_text(yaml.safe_dump({
            "ckpt_path": CHECKPOINT_ID, "test_gen_batch_size": 8,
            "guidance_scale": 4.5, "seed": 42, "lora_dropout": 0.05,
            "disable_lora_dropout": role == "off",
        }), encoding="utf-8")
        provenance = {"path": str(root / "inference_manifest.json"), "payload": manifest}
        values = {metric: float(i + 1) + (0.25 if role == "off" else 0.0)
                  for i, metric in enumerate(METRICS)}
        values.update({"Coverage_GT": 1.0, "Coverage_Pred": 1.0, "Common_Count": 2000})
        metric_order = list(values)
        config = {"paired_size": 448, "batch_size": 1, "device": "cuda"}
        for name in MAPS:
            result = {
                "map_name": name, "benchmark_v2_split": "seen_discrete_test",
                "gt_count": 2000, "pred_count": 2000, "common_count": 2000,
                "missing_pred_files_count": 0, "unmatched_pred_files_count": 0,
                "missing_pred_files": [], "unmatched_pred_files": [],
                "benchmark_v2_selection_counts": {"rows": 2000, "expected_stems": 2000},
                "metric_profile": "benchmark_v2_core", "metrics_order": metric_order,
                "metrics_ordered": values, "config": config,
                "inference_provenance": provenance,
                "benchmark_v2_asset_backend": "minimal",
                "benchmark_v2_asset_manifest_sha256": asset_digest,
                "benchmark_v2_selected_images_sha256": selected_digest,
            }
            self._write(root / f"benchmark_csgo_v2_{name}.json", result)
        summary = {
            "split": "seen_discrete_test", "kind": "discrete", "maps": list(MAPS),
            "per_map": {name: values for name in MAPS}, "metrics_macro_map": values,
            "inference_provenance": provenance, "checkpoint": CHECKPOINT_ID,
            "ckpt_path": CHECKPOINT_ID, "inference_seed": 42, "sample_count": 20000,
            "benchmark_v2_asset_backend": "minimal",
            "benchmark_v2_asset_manifest_sha256": asset_digest,
            "benchmark_v2_selected_images_sha256": selected_digest,
        }
        self._write(root / "summary.json", summary)

    def test_valid_comparison_and_explicit_output(self) -> None:
        report = compare(self.on, self.off)
        self.assertAlmostEqual(report["results"]["macro"]["PSNR"]["off_minus_on"], 0.25)
        self.assertEqual(report["results"]["macro"]["PSNR"]["direction"], "improved")
        self.assertEqual(report["results"]["macro"]["FID"]["direction"], "worsened")
        self.assertIn("single", report["scope"])
        self.assertIn("Macro", markdown(report))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--baseline", str(self.on), "--candidate", str(self.off),
                                   "--output-dir", str(self.off.parent / "comparison")]), 0)
        self.assertTrue((self.off.parent / "comparison/comparison.json").is_file())

    def test_mismatch_seed_batch_modes_assets(self) -> None:
        manifest = self.off / "inference_manifest.json"
        cases = [
            (lambda d: d.update(seed=43), "seed"),
            (lambda d: d["inference_execution"].update(batch_size=4), "inference_execution"),
            (lambda d: d["inference_execution"].update(model_mode_policy="wrong"), "inference_execution"),
            (lambda d: d.update(benchmark_v2_selected_images_sha256="d" * 64), "selected_images_sha256"),
        ]
        original = manifest.read_text(encoding="utf-8")
        for edit, error in cases:
            with self.subTest(error=error):
                manifest.write_text(original, encoding="utf-8")
                self._change(manifest, edit)
                with self.assertRaisesRegex(ComparisonError, error):
                    compare(self.on, self.off)
        manifest.write_text(original, encoding="utf-8")

    def test_metric_integrity_and_coverage(self) -> None:
        result = self.off / f"benchmark_csgo_v2_{MAPS[0]}.json"
        original = result.read_text(encoding="utf-8")
        cases = [
            (lambda d: d["metrics_ordered"].update(PSNR=float("nan")), "non-finite"),
            (lambda d: d["metrics_ordered"].pop("PSNR"), "missing"),
            (lambda d: d.update(common_count=1999), "common_count"),
            (lambda d: d["config"].update(paired_size=256), "paired_size"),
            (lambda d: d.update(metric_profile="other"), "metric_profile"),
        ]
        for edit, error in cases:
            with self.subTest(error=error):
                result.write_text(original, encoding="utf-8")
                self._change(result, edit)
                with self.assertRaisesRegex(ComparisonError, error):
                    compare(self.on, self.off)
        result.write_text(original, encoding="utf-8")
        result.unlink()
        with self.assertRaisesRegex(ComparisonError, "coverage"):
            compare(self.on, self.off)

    def test_summary_and_config_drift(self) -> None:
        summary = self.off / "summary.json"
        self._change(summary, lambda d: d["per_map"][MAPS[0]].update(CLIP=999.0))
        with self.assertRaisesRegex(ComparisonError, "summary.CLIP"):
            compare(self.on, self.off)
        self._write_run(self.off, "off")
        config = self.off / "evaluation_config.yaml"
        data = yaml.safe_load(config.read_text(encoding="utf-8"))
        data["guidance_scale"] = 5.0
        config.write_text(yaml.safe_dump(data), encoding="utf-8")
        with self.assertRaisesRegex(ComparisonError, "guidance_scale"):
            compare(self.on, self.off)

    def test_checkpoint_identity_and_provenance_path(self) -> None:
        self.assertEqual(checkpoint_id("/remote/mount/" + CHECKPOINT_ID, "checkpoint"),
                         CHECKPOINT_ID)
        with self.assertRaisesRegex(ComparisonError, "expected"):
            checkpoint_id("/remote/other/model.safetensors", "checkpoint")
        target = Path(self.temp.name) / CHECKPOINT_ID
        target.parent.mkdir(parents=True)
        target.touch()
        alias = Path(self.temp.name) / "checkpoint_alias"
        alias.symlink_to(target)
        self.assertEqual(checkpoint_id(str(alias), "checkpoint"), CHECKPOINT_ID)
        summary = self.off / "summary.json"
        self._change(summary, lambda d: d["inference_provenance"].update(
            path=str(self.on / "inference_manifest.json")))
        with self.assertRaisesRegex(ComparisonError, "another run"):
            compare(self.on, self.off)


if __name__ == "__main__":
    unittest.main()
