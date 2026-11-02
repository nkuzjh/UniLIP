from types import SimpleNamespace
import gc
import math
import random
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from unilip.train import gradient_diagnostics as diagnostic_module
from unilip.train.gradient_diagnostics import (
    ParameterCoordinate, check_total_backward, measure_gradients, select_coordinate,
    shared_language_layer_stats,
)
from scripts.probe_multitask_gradients import (
    main as probe_main, parse_args, rng_state_hashes, source_indices,
)


class ToyMultitaskModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = torch.nn.Linear(2, 1, bias=False)
        self.gen_head = torch.nn.Linear(1, 1, bias=False)
        self.action_head = torch.nn.Linear(1, 1, bias=False)

    def _aux_loc_head_modules_for_freeze(self):
        return [self.action_head]

    def forward(self, x, *, return_loss_tensors=False, task_id=None):
        hidden = self.language_model(x)
        gen = self.gen_head(hidden).square().mean()
        loc = self.action_head(hidden).square().mean()
        perception = (hidden - 1).square().mean()
        aux_loc = (self.gen_head(hidden) - 1).square().mean()
        zero = torch.zeros((), dtype=gen.dtype, device=gen.device)
        weights = {"gen": 1., "loc": 2., "perception": .5,
                   "aux_loc": 3., "repa": 0., "loc_repa": 0., "gen_aux": 0.}
        losses = {"gen": gen, "loc": loc, "perception": perception,
                  "aux_loc": aux_loc, "repa": zero, "loc_repa": zero,
                  "gen_aux": zero, "weights": weights}
        losses["total"] = sum(losses[name] * weight for name, weight in weights.items())
        return SimpleNamespace(loss=losses["total"], loss_tensors=losses)


class MutatingToyMultitaskModel(ToyMultitaskModel):
    def forward(self, x, *, labels, return_loss_tensors=False, task_id=None):
        scale = labels[0].float().clone()
        labels[0] = -100
        return super().forward(x * scale, return_loss_tensors=return_loss_tensors,
                               task_id=task_id)


class CheckpointedToyMultitaskModel(ToyMultitaskModel):
    def forward(self, x, *, return_loss_tensors=False, task_id=None):
        hidden = checkpoint(lambda value: torch.nn.functional.dropout(
            value, p=0.25, training=True), self.language_model(x), use_reentrant=False)
        gen = self.gen_head(hidden).square().mean()
        loc = self.action_head(hidden).square().mean()
        perception = (hidden - 1).square().mean()
        aux_loc = (self.gen_head(hidden) - 1).square().mean()
        zero = torch.zeros((), dtype=gen.dtype, device=gen.device)
        weights = {"gen": 1., "loc": 2., "perception": .5,
                   "aux_loc": 3., "repa": 0., "loc_repa": 0., "gen_aux": 0.}
        losses = {"gen": gen, "loc": loc, "perception": perception,
                  "aux_loc": aux_loc, "repa": zero, "loc_repa": zero,
                  "gen_aux": zero, "weights": weights}
        losses["total"] = sum(losses[name] * weight for name, weight in weights.items())
        return SimpleNamespace(loss=losses["total"], loss_tensors=losses)


class OtherWeightOnSecondMicrobatch(ToyMultitaskModel):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, *, return_loss_tensors=False, task_id=None):
        result = super().forward(x, return_loss_tensors=return_loss_tensors,
                                 task_id=task_id)
        self.calls += 1
        if self.calls == 2:
            result.loss_tensors["weights"]["repa"] = 1.
        return result


class GraphLifecycleToyModel(ToyMultitaskModel):
    def __init__(self):
        super().__init__()
        self.saved_refs = []
        self.forward_count = 0

    def forward(self, x, *, return_loss_tensors=False, task_id=None):
        if self.forward_count:
            gc.collect()
            if any(reference() is not None for reference in self.saved_refs):
                raise AssertionError("Previous microbatch still owns saved graph tensors")
            self.saved_refs.clear()
        self.forward_count += 1
        return super().forward(x, return_loss_tensors=return_loss_tensors,
                               task_id=task_id)


class BadTotalFirstMicroModel(ToyMultitaskModel):
    def __init__(self):
        super().__init__()
        self.forward_count = 0

    def forward(self, x, *, return_loss_tensors=False, task_id=None):
        self.forward_count += 1
        result = super().forward(x, return_loss_tensors=return_loss_tensors,
                                 task_id=task_id)
        result.loss_tensors["total"] = result.loss_tensors["total"] + self.language_model(x).sum()
        return result


class GradientDiagnosticsTests(unittest.TestCase):
    def test_accumulated_component_gradients_reconstruct_total(self):
        model = ToyMultitaskModel()
        coordinate = select_coordinate(model, [r"language_model\."])
        batches = [
            {"x": torch.tensor([[1., 2.], [2., 1.]]), "task_id": torch.tensor([0, 1])},
            {"x": torch.tensor([[3., 1.], [1., 3.]]), "task_id": torch.tensor([0, 1])},
        ]
        result = measure_gradients(model, batches, coordinate,
                                   numerical_check_first_batch=True,
                                   group_patterns={"shared": r"language_model\."})
        self.assertTrue(result["linearity"]["passed"])
        self.assertLess(result["linearity"]["difference_norm"], 1e-5)
        self.assertTrue(result["aux_freeze"]["passed"])
        self.assertTrue(result["repeated_total_vjp"]["passed"])
        self.assertLess(result["combined"]["main_joint_vs_component_sum_difference_norm"], 1e-5)
        self.assertLess(result["combined"]["actual_aux_increment_vs_component_difference_norm"], 1e-5)
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["microbatches"], 2)
        self.assertAlmostEqual(result["components"]["aux_loc"]["weighted_norm"],
                               3 * result["components"]["aux_loc"]["raw_norm"], places=5)
        self.assertEqual(result["groups"]["shared"]["numel"], 2)
        self.assertEqual(result["shared_language_layers"]["status"],
                         "has_unassigned_shared_parameters")
        self.assertEqual(result["shared_language_layers"]["unassigned"]["parameter_count"], 1)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_shared_language_layer_partition_and_negative_dot_ranking(self):
        names = (
            "model.language_model.layers.1.weight",
            "model.language_model.layers.2.weight",
            "model.language_model.layers.3.weight",
            "model.language_model.norm.weight",
            "model.llm_connector.language_model.layers.4.weight",
        )
        gen = [[10., 0.], [1.], [2.], [1.], [100.]]
        loc = [[-1., 1.], [-1.], [3.], [1.], [-100.]]
        total = [[3., 4.], [5.], [6.], [7.], [8.]]
        coordinate = ParameterCoordinate(names, tuple(
            torch.nn.Parameter(torch.zeros(len(values))) for values in gen))
        weighted = {"gen": [torch.tensor(v) for v in gen],
                    "loc": [torch.tensor(v) for v in loc],
                    "perception": [torch.tensor(v) for v in
                                   ([1., 0.], [0.], [0.], [0.], [5.])],
                    "aux_loc": [torch.tensor(v) for v in
                                ([0., 1.], [0.], [0.], [0.], [5.])],
                    "total": [torch.tensor(v) for v in total]}
        raw = {name: weighted[name] for name in
               ("gen", "loc", "perception", "aux_loc")}
        stats = shared_language_layer_stats(coordinate, weighted, raw)
        self.assertEqual(stats["status"], "has_unassigned_shared_parameters")
        self.assertEqual(stats["shared_parameter_count"], 4)
        self.assertEqual(stats["shared_numel"], 5)
        self.assertEqual(stats["layers_by_negative_dot_contribution"], ["1", "2"])
        self.assertGreater(stats["layers"]["2"]["weighted_gen_loc_cosine"] * -1,
                           stats["layers"]["1"]["weighted_gen_loc_cosine"] * -1)
        self.assertEqual(stats["layers"]["1"]["weighted_gen_total_dot"], 30.)
        self.assertEqual(stats["layers"]["1"]["raw_loc_total_dot"], 1.)
        self.assertEqual(stats["layers"]["1"]["weighted_gen_aux_dot"], 10.)
        self.assertEqual(stats["layers"]["1"]["weighted_loc_aux_dot"], 0.)
        self.assertAlmostEqual(
            stats["layers"]["1"]["pairwise_weighted_cosine"]["perception"]["aux_loc"], 0.)
        self.assertIsNone(
            stats["layers"]["2"]["pairwise_weighted_cosine"]["gen"]["perception"])
        self.assertIsNone(
            stats["layers"]["2"]["pairwise_weighted_cosine"]["perception"]["perception"])
        self.assertEqual(stats["global_shared_weighted_gen_loc_dot"], -4.)
        self.assertEqual(stats["checks"]["named_layer_dot_sum"], -5.)
        self.assertFalse(stats["checks"]["named_layer_dot_matches_global"])
        self.assertTrue(stats["checks"]["partition_dot_matches_global"])
        self.assertTrue(stats["checks"]["parameter_count_matches_global"])
        self.assertTrue(stats["checks"]["numel_matches_global"])
        denominator = stats["global_alignment_denominator"]
        self.assertAlmostEqual(stats["negative_alignment_mass"], 11 / denominator)
        self.assertAlmostEqual(stats["positive_alignment_mass"], 7 / denominator)

    def test_shared_language_zero_norm_and_private_only_coordinate(self):
        coordinate = ParameterCoordinate(
            ("model.language_model.layers.0.weight",),
            (torch.nn.Parameter(torch.zeros(2)),))
        zero = [torch.zeros(2)]
        one = [torch.ones(2)]
        stats = shared_language_layer_stats(coordinate,
                                            {"gen": zero, "loc": one, "total": one},
                                            {"gen": zero, "loc": one})
        self.assertIsNone(stats["negative_alignment_mass"])
        self.assertIsNone(stats["positive_alignment_mass"])
        self.assertIsNone(stats["layers"]["0"]["weighted_gen_loc_cosine"])
        self.assertIsNone(stats["layers"]["0"]["weighted_gen_aux_dot"])
        self.assertTrue(stats["checks"]["partition_dot_matches_global"])
        private = ParameterCoordinate(
            ("model.llm_connector.language_model.layers.0.weight",),
            (torch.nn.Parameter(torch.zeros(2)),))
        self.assertEqual(shared_language_layer_stats(
            private, {"gen": zero, "loc": one, "total": one},
            {"gen": zero, "loc": one})["status"], "no_shared_parameters")

    def test_component_sum_matches_full_vectors_directions_and_rng_with_checkpointing(self):
        torch.manual_seed(11)
        model = CheckpointedToyMultitaskModel()
        coordinate = select_coordinate(model, [r"language_model\."])
        batches = [{"x": torch.tensor([[float(i), float(i + 1)]]),
                    "task_id": torch.tensor([i % 4])} for i in (1, 2, 3)]
        kwargs = dict(group_patterns={"shared": r"language_model\."},
                      numerical_check_first_batch=True)
        torch.manual_seed(91)
        direct_vectors = {}
        direct = measure_gradients(model, batches, coordinate, vector_sink=direct_vectors,
                                   **kwargs)
        direct_rng = torch.get_rng_state().clone()
        torch.manual_seed(91)
        summed_vectors = {}
        summed = measure_gradients(model, batches, coordinate, vector_sink=summed_vectors,
                                   measurement_mode="component_sum", fp32_reference=True,
                                   deterministic_probe=True, **kwargs)
        self.assertTrue(torch.equal(direct_rng, torch.get_rng_state()))
        for category in ("raw", "weighted"):
            for name in direct_vectors[category]:
                for left, right in zip(direct_vectors[category][name],
                                       summed_vectors[category][name]):
                    self.assertTrue(torch.allclose(left, right, atol=1e-5, rtol=1e-5),
                                    (category, name))
        self.assertAlmostEqual(direct["total_gradient_norm"],
                               summed["total_gradient_norm"], places=5)
        for name in ("gen", "loc", "perception", "aux_loc"):
            self.assertAlmostEqual(direct["components"][name]["cosine_to_total"],
                                   summed["components"][name]["cosine_to_total"], places=5)
        for name in ("gen", "loc"):
            for key, value in direct["combined"]["primary_directions"][name].items():
                self.assertTrue(math.isclose(value,
                    summed["combined"]["primary_directions"][name][key],
                    rel_tol=1e-5, abs_tol=1e-5), (name, key))
        self.assertTrue(summed["first_micro_direct_checks"]["main_vs_components"]["passed"])
        self.assertTrue(summed["first_micro_direct_checks"]["total_vs_components"]["passed"])
        self.assertTrue(summed["repeated_total_vjp"]["passed"])
        self.assertIsNone(summed["linearity"])
        self.assertIsNone(summed["components"]["linearity_residual"]["weighted_norm"])
        self.assertIsNone(summed["combined"]["actual_aux_increment_norm"])
        self.assertIsNone(summed["groups"]["shared"]["linearity_difference_norm"])
        self.assertTrue(summed["shared_language_layers"]["checks"]["partition_dot_matches_global"])

    def test_component_sum_releases_each_microbatch_graph_before_next_forward(self):
        class SavedBox:
            def __init__(self, tensor):
                self.tensor = tensor

        model = GraphLifecycleToyModel()
        coordinate = select_coordinate(model, [r"language_model\."])
        def pack(tensor):
            box = SavedBox(tensor)
            model.saved_refs.append(weakref.ref(box))
            return box
        def unpack(box):
            return box.tensor
        batches = [{"x": torch.tensor([[float(i), float(i + 1)]])} for i in (1, 2, 3)]
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            result = measure_gradients(model, batches, coordinate,
                                       measurement_mode="component_sum", fp32_reference=True,
                                       deterministic_probe=True, check_aux_freeze=False,
                                       numerical_check_first_batch=True)
        self.assertEqual(model.forward_count, 3)
        gc.collect()
        self.assertFalse(any(reference() is not None for reference in model.saved_refs))
        self.assertTrue(result["first_micro_direct_checks"]["total_vs_components"]["passed"])
        self.assertTrue(result["repeated_total_vjp"]["passed"])

    def test_component_sum_stops_after_failed_first_micro_direct_check(self):
        model = BadTotalFirstMicroModel()
        coordinate = select_coordinate(model, [r"language_model\."])
        batches = [{"x": torch.ones(1, 2)} for _ in range(3)]
        with self.assertRaisesRegex(FloatingPointError, "total_vs_components"):
            measure_gradients(model, batches, coordinate, measurement_mode="component_sum",
                              fp32_reference=True, deterministic_probe=True,
                              check_aux_freeze=False)
        self.assertEqual(model.forward_count, 1)

    def test_component_sum_stops_after_failed_first_micro_repeat(self):
        model = GraphLifecycleToyModel()
        coordinate = select_coordinate(model, [r"language_model\."])
        original_grad = diagnostic_module._grad
        calls = 0
        def altered_grad(loss, params, *, retain_graph):
            nonlocal calls
            calls += 1
            gradients = original_grad(loss, params, retain_graph=retain_graph)
            return tuple(gradient * 2 if gradient is not None else None
                         for gradient in gradients) if calls == 7 else gradients
        with patch.object(diagnostic_module, "_grad", side_effect=altered_grad):
            with self.assertRaisesRegex(FloatingPointError, "repeated_total_vjp"):
                measure_gradients(model, [{"x": torch.ones(1, 2)} for _ in range(3)],
                                  coordinate, measurement_mode="component_sum",
                                  fp32_reference=True, deterministic_probe=True,
                                  check_aux_freeze=False)
        self.assertEqual(model.forward_count, 1)
        self.assertEqual(calls, 7)

    def test_component_sum_rejects_incomplete_tasks_and_other_weight(self):
        model = ToyMultitaskModel()
        coordinate = select_coordinate(model, [r"language_model\."])
        batch = {"x": torch.ones(1, 2)}
        with self.assertRaisesRegex(ValueError, "all four tasks"):
            measure_gradients(model, [batch], coordinate, tasks=("gen",),
                              measurement_mode="component_sum", fp32_reference=True,
                              deterministic_probe=True)
        with self.assertRaisesRegex(ValueError, "FP32 deterministic"):
            measure_gradients(model, [batch], coordinate,
                              measurement_mode="component_sum")
        model = OtherWeightOnSecondMicrobatch()
        coordinate = select_coordinate(model, [r"language_model\."])
        with self.assertRaisesRegex(ValueError, "zero repa weight"):
            measure_gradients(model, [batch, batch], coordinate,
                              measurement_mode="component_sum", fp32_reference=True,
                              deterministic_probe=True)

    def test_omitted_tasks_remain_in_linearity_residual(self):
        model = ToyMultitaskModel()
        coordinate = select_coordinate(model, [r"language_model\."])
        result = measure_gradients(model, [{"x": torch.ones(2, 2)}], coordinate,
                                   tasks=("gen",), check_aux_freeze=False)
        self.assertTrue(result["linearity"]["passed"])
        self.assertGreater(result["components"]["other"]["weighted_norm"], 0)

    def test_total_backward_agrees_with_vjp(self):
        model = ToyMultitaskModel()
        coordinate = select_coordinate(model, [r"language_model\."])
        result = check_total_backward(model, {"x": torch.ones(2, 2)}, coordinate)
        self.assertTrue(result["passed"])
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_total_backward_replays_pristine_batch_after_input_mutation(self):
        model = MutatingToyMultitaskModel()
        coordinate = select_coordinate(model, [r"language_model\."])
        labels = torch.tensor([2, 3])
        result = check_total_backward(model, {"x": torch.ones(2, 2), "labels": labels},
                                      coordinate)
        self.assertTrue(result["passed"])
        self.assertEqual(labels.tolist(), [2, 3])

    def test_nonfinite_loss_is_rejected(self):
        model = ToyMultitaskModel()
        coordinate = select_coordinate(model, [r"language_model\."])
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite"):
            measure_gradients(model, [{"x": torch.full((1, 2), float("nan"))}], coordinate)

    def test_cpu_raw_weighted_buffers_match_independent_vjp(self):
        model = ToyMultitaskModel()
        coordinate = select_coordinate(model, [r".*"])
        batch = {"x": torch.tensor([[1., 2.], [2., 1.]])}
        output = model(**batch, return_loss_tensors=True).loss_tensors
        expected = {}
        for name in ("gen", "loc", "perception", "aux_loc"):
            grads = torch.autograd.grad(output[name], coordinate.params,
                                        retain_graph=True, allow_unused=True)
            expected[name] = [torch.zeros_like(p, dtype=torch.float32) if grad is None
                              else grad.detach().float().clone()
                              for p, grad in zip(coordinate.params, grads)]
        total_grads = torch.autograd.grad(output["total"], coordinate.params,
                                          allow_unused=True)
        expected_total_norm = sum(float(grad.double().square().sum())
                                  for grad in total_grads if grad is not None) ** .5
        vectors = {}
        result = measure_gradients(model, [batch], coordinate, vector_sink=vectors)
        self.assertAlmostEqual(result["total_gradient_norm"], expected_total_norm, places=5)
        self.assertTrue(result["shared_language_layers"]["checks"]["partition_dot_matches_global"])
        for name, raw in expected.items():
            for actual, reference in zip(vectors["raw"][name], raw):
                self.assertTrue(torch.equal(actual, reference))
            weight = output["weights"][name]
            for actual, reference in zip(vectors["weighted"][name], raw):
                self.assertTrue(torch.equal(actual, reference * weight))
        self.assertGreater(result["components"]["gen"]["gradient_coverage"]["none"], 0)
        self.assertGreater(result["components"]["loc"]["gradient_coverage"]["none"], 0)

    def test_cpu_zero_gradient_coverage(self):
        model = ToyMultitaskModel()
        coordinate = select_coordinate(model, [r"language_model\."])
        result = measure_gradients(model, [{"x": torch.zeros(1, 2)}], coordinate,
                                   check_aux_freeze=False)
        self.assertGreater(result["components"]["gen"]["gradient_coverage"]["zero"], 0)

    def test_coordinate_requires_explicit_trainable_match(self):
        model = ToyMultitaskModel()
        with self.assertRaisesRegex(ValueError, "No trainable parameters"):
            select_coordinate(model, [r"missing_module"])

    def test_fixed_source_pool_stages(self):
        by_map = {name: list(range(offset, offset + 160)) for name, offset in
                  (("de_ancient", 0), ("de_dust2", 160), ("de_nuke", 320))}
        kwargs = dict(source_seed=7, samples_per_map=128,
                      source_samples=None, max_source_samples=None)
        screen = source_indices(by_map, stage="screen", **kwargs)
        effective = source_indices(by_map, stage="effective", **kwargs)
        effective2 = source_indices(by_map, stage="effective", source_offset=128, **kwargs)
        effective3 = source_indices(by_map, stage="effective", source_offset=256, **kwargs)
        pool = source_indices(by_map, stage="pool", **kwargs)
        self.assertEqual((len(screen), len(effective), len(pool)), (96, 128, 384))
        self.assertEqual(len(set(pool)), 384)
        self.assertEqual(set(effective) | set(effective2) | set(effective3), set(pool))
        self.assertTrue(set(screen).issubset(set(pool)))
        self.assertTrue(set(effective).issubset(set(pool)))

    def test_source_selection_uses_stable_frame_identity(self):
        rows = {name: [
            {"dataset_index": offset + i, "file_frame": f"frame_{i:03d}", "row_id": str(i)}
            for i in range(40)] for name, offset in
            (("de_ancient", 0), ("de_dust2", 40), ("de_nuke", 80))}
        reordered = {name: [
            {**row, "dataset_index": 1000 + j + 40 * map_idx}
            for j, row in enumerate(reversed(items))]
            for map_idx, (name, items) in enumerate(rows.items())}
        kwargs = dict(source_seed=17, samples_per_map=32, stage="screen",
                      source_samples=None, max_source_samples=None)
        selected1 = source_indices(rows, **kwargs)
        selected2 = source_indices(reordered, **kwargs)
        frames1 = {row["dataset_index"]: row["file_frame"] for items in rows.values() for row in items}
        frames2 = {row["dataset_index"]: row["file_frame"] for items in reordered.values() for row in items}
        self.assertEqual([frames1[i] for i in selected1], [frames2[i] for i in selected2])

    def test_rng_state_hashes_cover_python_numpy_and_torch(self):
        original_python = random.getstate()
        original_numpy = np.random.get_state()
        original_torch = torch.get_rng_state()
        try:
            random.seed(113)
            np.random.seed(113)
            torch.manual_seed(113)
            initial = rng_state_hashes("cpu")
            self.assertEqual(initial, rng_state_hashes("cpu"))
            self.assertEqual(set(initial), {"python", "numpy", "cpu"})
            random.random()
            after_python = rng_state_hashes("cpu")
            self.assertNotEqual(initial["python"], after_python["python"])
            self.assertEqual(initial["numpy"], after_python["numpy"])
            self.assertEqual(initial["cpu"], after_python["cpu"])
            np.random.random()
            after_numpy = rng_state_hashes("cpu")
            self.assertNotEqual(after_python["numpy"], after_numpy["numpy"])
            self.assertEqual(after_python["python"], after_numpy["python"])
            self.assertEqual(after_python["cpu"], after_numpy["cpu"])
        finally:
            random.setstate(original_python)
            np.random.set_state(original_numpy)
            torch.set_rng_state(original_torch)

    def test_component_sum_cli_rejects_screen_and_missing_reference_flags(self):
        args = ["--checkpoint-dir", "/tmp/checkpoint-1", "--output", "/tmp/probe.json",
                "--measurement-mode", "component_sum"]
        with self.assertRaisesRegex(ValueError, "screen stage"):
            probe_main(args)
        with self.assertRaisesRegex(ValueError, "FP32 deterministic"):
            probe_main([*args, "--stage", "effective"])

    def test_explicit_source_indices_parse(self):
        args = parse_args(["--checkpoint-dir", "/tmp/checkpoint-1", "--output", "/tmp/probe.json",
                           "--source-indices", "7", "12", "19"])
        self.assertEqual(args.source_indices, [7, 12, 19])
