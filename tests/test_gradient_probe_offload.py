"""CPU checks for the separate saved-tensor offload probe entrypoint."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from scripts.probe_multitask_gradients_offload import (
    main as offload_main, offload_metadata, offload_model_forward,
    selective_saved_tensors_offload,
)
from scripts import run_gradient_diagnostics as controller
from tests.test_run_gradient_diagnostics import valid_report
from unilip.train import gradient_probe_loader
from unilip.train.gradient_diagnostics import (
    check_total_backward, measure_gradients, select_coordinate,
)


class CheckpointedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = torch.nn.Linear(3, 4)
        self.gen_head = torch.nn.Linear(4, 1)
        self.action_head = torch.nn.Linear(4, 1)

    def _features(self, x):
        return F.dropout(self.language_model(x).tanh(), p=0.25, training=True)

    def forward(self, x, *, return_loss_tensors=False):
        hidden = checkpoint(self._features, x, use_reentrant=False)
        gen = self.gen_head(hidden).square().mean()
        loc = self.action_head(hidden).square().mean()
        perception = (hidden - 1).square().mean()
        aux_loc = (self.gen_head(hidden) - 1).square().mean()
        zero = torch.zeros_like(gen)
        weights = {"gen": 1., "loc": 2., "perception": .5,
                   "aux_loc": 3., "repa": 0., "loc_repa": 0., "gen_aux": 0.}
        losses = {"gen": gen, "loc": loc, "perception": perception,
                  "aux_loc": aux_loc, "repa": zero, "loc_repa": zero,
                  "gen_aux": zero, "weights": weights}
        losses["total"] = sum(losses[name] * weight for name, weight in weights.items())
        return SimpleNamespace(loss_tensors=losses)


class GradientProbeOffloadTest(unittest.TestCase):
    def setUp(self):
        # These fixtures must not initialize CUDA even on a GPU host.
        cuda_available = patch.object(torch.cuda, "is_available", return_value=False)
        cuda_available.start()
        self.addCleanup(cuda_available.stop)
        torch.manual_seed(7)
        self.model = CheckpointedModel()
        self.coordinate = select_coordinate(self.model, [r"language_model\."])
        self.batches = [{"x": torch.tensor([[1., 2., 3.]])},
                        {"x": torch.tensor([[2., 3., 1.]])}]

    def assert_vectors_equal(self, left, right):
        self.assertEqual(left["parameter_names"], right["parameter_names"])
        for category in ("raw", "weighted"):
            for name in left[category]:
                for actual, expected in zip(left[category][name], right[category][name]):
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_noncontiguous_checkpoint_input_keeps_recompute_layout(self):
        source = torch.randn(1, 4, 4, requires_grad=True)
        noncontiguous = source.transpose(1, 2)
        self.assertFalse(noncontiguous.is_contiguous())

        def layout_sensitive(value):
            shaped = value if value.is_contiguous() else value.flatten()
            return shaped.square().sum()

        baseline = checkpoint(layout_sensitive, noncontiguous, use_reentrant=False)
        expected = torch.autograd.grad(baseline, source)[0]
        with self.assertRaisesRegex(torch.utils.checkpoint.CheckpointError,
                                    "different metadata"):
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                broken = checkpoint(layout_sensitive, noncontiguous,
                                    use_reentrant=False)
            torch.autograd.grad(broken, source)
        with selective_saved_tensors_offload(self.model, device_type="cpu"):
            actual_loss = checkpoint(layout_sensitive, noncontiguous,
                                     use_reentrant=False)
        actual = torch.autograd.grad(actual_loss, source)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_selective_hooks_copy_only_eligible_nonparameter_tensors(self):
        data = torch.randn(3, requires_grad=True)
        original_empty = torch.empty
        packed_shapes = []

        def recorded_empty(*args, **kwargs):
            packed_shapes.append(tuple(args[0]))
            return original_empty(*args, **kwargs)

        with patch.object(torch, "empty", side_effect=recorded_empty):
            with selective_saved_tensors_offload(self.model, device_type="cpu"):
                loss = data.square().sum()
                parameter_loss = self.model.language_model.weight.square().sum()
        torch.autograd.grad(loss + parameter_loss,
                            (data, self.model.language_model.weight))
        self.assertIn((3,), packed_shapes)
        self.assertNotIn(tuple(self.model.language_model.weight.shape), packed_shapes)

    def test_direct_and_component_sum_vjps_match_with_checkpoint_rng(self):
        for mode in ("full", "component_sum"):
            with self.subTest(mode=mode):
                kwargs = {"check_aux_freeze": False,
                          "numerical_check_first_batch": True,
                          "measurement_mode": mode}
                if mode == "component_sum":
                    kwargs.update(fp32_reference=True, deterministic_probe=True)
                torch.manual_seed(131)
                reference_vectors = {}
                reference = measure_gradients(self.model, self.batches, self.coordinate,
                                              vector_sink=reference_vectors, **kwargs)
                reference_rng = torch.get_rng_state().clone()
                torch.manual_seed(131)
                offloaded_vectors = {}
                with offload_model_forward(self.model):
                    offloaded = measure_gradients(self.model, self.batches, self.coordinate,
                                                 vector_sink=offloaded_vectors, **kwargs)
                self.assertTrue(torch.equal(torch.get_rng_state(), reference_rng))
                self.assert_vectors_equal(offloaded_vectors, reference_vectors)
                self.assertTrue(reference["repeated_total_vjp"]["passed"])
                self.assertTrue(offloaded["repeated_total_vjp"]["passed"])
                if mode == "component_sum":
                    self.assertTrue(offloaded["first_micro_direct_checks"]
                                    ["total_vs_components"]["passed"])
                else:
                    self.assertTrue(offloaded["linearity"]["passed"])
                self.assertAlmostEqual(offloaded["total_gradient_norm"],
                                       reference["total_gradient_norm"], places=6)

    def test_normal_backward_replay_uses_offloaded_forwards_and_matches(self):
        torch.manual_seed(83)
        reference = check_total_backward(self.model, self.batches[0], self.coordinate)
        reference_rng = torch.get_rng_state().clone()
        torch.manual_seed(83)
        with offload_model_forward(self.model):
            offloaded = check_total_backward(self.model, self.batches[0], self.coordinate)
        self.assertTrue(reference["passed"])
        self.assertTrue(offloaded["passed"])
        self.assertAlmostEqual(offloaded["difference_norm"],
                               reference["difference_norm"], places=7)
        self.assertTrue(torch.equal(torch.get_rng_state(), reference_rng))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_forward_exception_restores_outer_hooks_and_original_forward(self):
        class FailAfterSave(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(3))

            def forward(self, x, *, fail=False):
                value = (x * self.weight).square()
                if fail:
                    raise RuntimeError("expected forward failure")
                return value.sum()

        model = FailAfterSave()
        packed = []
        def pack_tensor(tensor):
            packed.append(tensor)
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack_tensor, lambda tensor: tensor):
            with self.assertRaisesRegex(RuntimeError, "expected forward failure"):
                with offload_model_forward(model):
                    model(torch.ones(3), fail=True)
            self.assertNotIn("forward", model.__dict__)
            self.assertEqual(len(packed), 0)
            model(torch.ones(3)).backward()
        self.assertGreater(len(packed), 0)
        self.assertEqual(model.weight.grad.tolist(), [2., 2., 2.])

    def test_entrypoint_preserves_report_schema_and_records_offload(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            rows = [{"dataset_index": i, "map": map_name,
                     "file_frame": f"frame_{i}"}
                    for i, map_name in enumerate(("de_ancient", "de_dust2"))]
            context = SimpleNamespace(
                model=self.model,
                dataset=self.batches,
                collator=lambda samples: {"x": torch.cat([s["x"] for s in samples])},
                metadata={"experiment": "cpu_fixture", "checkpoint_step": 1200,
                          "train_microbatch": 1,
                          "dataset_indices_by_map": {"de_ancient": [0],
                                                     "de_dust2": [1], "de_nuke": []},
                          "dataset_source_rows_by_map": {"de_ancient": [rows[0]],
                                                         "de_dust2": [rows[1]],
                                                         "de_nuke": []}},
            )
            with patch.object(gradient_probe_loader, "load_gradient_probe_context",
                              return_value=context):
                result = offload_main([
                    "--checkpoint-dir", directory, "--experiment", "cpu_fixture",
                    "--checkpoint-step", "1200", "--device", "cpu",
                    "--stage", "custom", "--source-samples", "2",
                    "--source-indices", "0", "1", "--microbatch-size", "1",
                    "--accumulations", "2", "--noise-seeds", "17",
                    "--no-check-aux-freeze", "--numerical-check",
                    "--check-total-backward", "--save-gradient-vectors",
                    "--output", str(output),
                ])
            report = json.loads(output.read_text())
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["microbatch_size"], 1)
            self.assertEqual(report["accumulations"], 2)
            self.assertEqual(report["source_indices"], [0, 1])
            self.assertTrue(report["results"]["17"]["total_backward_check"]["passed"])
            self.assertTrue(report["results"]["17"]["repeated_total_vjp"]["passed"])
            self.assertEqual(report["metadata"]["saved_tensor_offload"], offload_metadata())
            self.assertEqual(result["metadata"]["saved_tensor_offload"], offload_metadata())
            self.assertTrue(Path(report["results"]["17"]["gradient_vectors_path"]).is_file())
            self.assertNotIn("forward", self.model.__dict__)

    def test_controller_accepts_additive_offload_metadata(self):
        config = controller.load_config(controller.DEFAULT_CONFIG)
        item = controller.build_items(config, ("confirm",))[0]
        report = valid_report(item)
        report["metadata"]["saved_tensor_offload"] = offload_metadata()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report))
            self.assertEqual(controller.report_checks(path, item), (True, []))


if __name__ == "__main__":
    unittest.main()
