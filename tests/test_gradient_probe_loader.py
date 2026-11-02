import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml
import torch
from scripts.probe_multitask_gradients import _move, parse_args

from unilip.train.gradient_probe_loader import (
    _prepare_deterministic_attention,
    _training_argv,
    _validate_checkpoint,
    load_gradient_probe_context,
)


class GradientProbeLoaderTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "exp32_3maps_P4_S1_J8"
        self.checkpoint = self.output / "checkpoint-1800"
        self.checkpoint.mkdir(parents=True)
        (self.checkpoint / "model.safetensors").touch()
        (self.checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1800}))
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        self.manifest = self.root / "benchmark_manifest.json"
        self.asset_manifest = self.root / "asset_manifest.json"
        self.manifest.write_text("{}")
        self.asset_manifest.write_text("{}")
        self.config = {
            "exp32_3maps_experiment": self.output.name,
            "benchmark_v2_split": "seen_train",
            "benchmark_v2_manifest": str(self.manifest),
            "benchmark_v2_asset_manifest": str(self.asset_manifest),
            "train_maps": ["de_ancient", "de_dust2", "de_nuke"],
            "data_dir": str(self.data_dir),
        }
        (self.output / "resolved_config.yaml").write_text(yaml.safe_dump(self.config))
        self.plan = {
            "experiment": self.output.name,
            "runtime": {"micro_batch": 4, "gradient_accumulation_steps": 32, "world_size": 1},
            "command": ["python", "train_csgo.py", "--csgo_config", "old.yaml",
                        "--deepspeed", "zero0.json", "--bf16", "True",
                        "--gradient_checkpointing", "True", "--report_to", "wandb"],
        }
        (self.output / "experiment_plan.json").write_text(json.dumps(self.plan))

    def test_validation_rejects_step_or_split_mismatch(self):
        self.assertEqual(_validate_checkpoint(self.checkpoint)[2], 1800)
        (self.checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1799}))
        with self.assertRaisesRegex(ValueError, "global_step"):
            _validate_checkpoint(self.checkpoint)
        (self.checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1800}))
        self.config["benchmark_v2_split"] = "seen_val"
        (self.output / "resolved_config.yaml").write_text(yaml.safe_dump(self.config))
        with self.assertRaisesRegex(ValueError, "seen_train"):
            _validate_checkpoint(self.checkpoint)

    def test_training_argv_removes_deepspeed_and_sets_probe_options(self):
        argv = _training_argv(self.plan, self.output / "resolved_config.yaml",
                              bf16=False, gradient_checkpointing=False)
        self.assertNotIn("--deepspeed", argv)
        self.assertEqual(argv[argv.index("--csgo_config") + 1], str(self.output / "resolved_config.yaml"))
        self.assertEqual(argv[argv.index("--bf16") + 1], "False")
        self.assertEqual(argv[argv.index("--gradient_checkpointing") + 1], "False")
        self.assertEqual(argv[argv.index("--report_to") + 1], "none")

    def test_context_passes_strict_checkpoint_build_and_map_indices(self):
        model = SimpleNamespace(
            config=SimpleNamespace(alpha_loc_loss=8.0, alpha_loc_aux_loss=0.0,
                                   alpha_gen_aux_loss=0.0, alpha_loc_perception_loss=0.05),
            training=True,
            train=lambda: None,
            parameters=lambda: [torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16)),
                                torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))],
        )
        dataset = SimpleNamespace(data_entries=[{"map": "de_ancient", "file_frame": "1"},
                                                {"map": "de_dust2", "file_frame": "2"},
                                                {"map": "de_nuke", "file_frame": "3"}])
        collator = object()
        build_calls = []
        fake_train_module = SimpleNamespace(train=lambda **kwargs: (
            build_calls.append(kwargs) or (model, dataset, collator)
        ))
        with patch.dict(sys.modules, {"train_csgo": fake_train_module}):
            context = load_gradient_probe_context(self.checkpoint)
        self.assertIs(context.model, model)
        self.assertIs(context.collator, collator)
        self.assertEqual(context.metadata["train_effective_batch"], 128)
        self.assertEqual(context.metadata["train_world_size"], 1)
        self.assertEqual(context.metadata["trainable_parameters"], 2)
        self.assertEqual(context.metadata["parameter_dtype_numel"],
                         {"torch.bfloat16": 1, "torch.float32": 1})
        self.assertEqual(context.metadata["parameter_dtype_summary"], "bf16_fp32_mixed")
        self.assertEqual(context.metadata["checkpoint_step"], 1800)
        self.assertEqual(context.metadata["dataset_indices_by_map"]["de_nuke"], [2])
        self.assertEqual(context.metadata["dataset_source_rows_by_map"]["de_nuke"][0]["file_frame"], "3")
        self.assertEqual(len(context.metadata["resolved_config_sha256"]), 64)
        self.assertEqual(len(context.metadata["dataset_source"]["benchmark_v2_manifest_sha256"]), 64)
        request = build_calls[0]["probe_request"]
        self.assertEqual(request["checkpoint_weights"], self.checkpoint / "model.safetensors")
        self.assertEqual(request["checkpoint_step"], 1800)

    def test_off_mode_explicitly_disables_constructor_checkpointing(self):
        class ProbeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
                self.language_model.gradient_checkpointing = True
                self.config = SimpleNamespace()

            def get_model(self):
                return self

        model = ProbeModel()
        dataset = SimpleNamespace(data_entries=[
            {"map": map_name, "file_frame": str(index)}
            for index, map_name in enumerate(self.config["train_maps"])
        ])
        fake_train_module = SimpleNamespace(train=lambda **kwargs: (model, dataset, object()))
        with patch.dict(sys.modules, {"train_csgo": fake_train_module}):
            context = load_gradient_probe_context(self.checkpoint,
                                                  gradient_checkpointing=False)
        self.assertFalse(model.language_model.gradient_checkpointing)
        self.assertFalse(context.metadata["gradient_checkpointing_modules"]["language_model"]["enabled"])
        self.assertGreater(context.metadata["gradient_checkpointing_disable"]["gradient_checkpointing_flags_cleared"], 0)

    def test_fp32_request_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError, "FP32 probe is not supported"):
            load_gradient_probe_context(self.checkpoint, bf16=False)

    def test_fp32_reference_converts_only_probe_model_and_records_adjustments(self):
        class ProbeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_tower = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
                self.vision_tower.config = SimpleNamespace(
                    _attn_implementation="flash_attention_2", use_flash_attn=True,
                )
                self.vision_tower.use_flash_attn = True
                self.config = SimpleNamespace(
                    _attn_implementation="flash_attention_2",
                    alpha_loc_loss=8.0,
                )

            def get_model(self):
                return self

        model = ProbeModel()
        dataset = SimpleNamespace(data_entries=[
            {"map": map_name, "file_frame": str(index)}
            for index, map_name in enumerate(self.config["train_maps"])
        ])
        fake_train_module = SimpleNamespace(train=lambda **kwargs: (model, dataset, object()))
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        old_precision = torch.get_float32_matmul_precision()
        old_sdp = (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
            torch.backends.cuda.cudnn_sdp_enabled(),
        )
        try:
            with patch.dict(sys.modules, {"train_csgo": fake_train_module}):
                context = load_gradient_probe_context(self.checkpoint,
                                                      precision_reference=True)
            self.assertEqual(context.metadata["dtype"], "float32")
            self.assertTrue(context.metadata["precision_reference"])
            self.assertEqual(context.metadata["parameter_dtype_numel"], {"torch.float32": 6})
            self.assertEqual(model.vision_tower.weight.dtype, torch.float32)
            self.assertEqual(model.config._attn_implementation, "eager")
            self.assertEqual(model.vision_tower.config._attn_implementation, "eager")
            self.assertFalse(model.vision_tower.use_flash_attn)
            self.assertFalse(model.vision_tower.config.use_flash_attn)
            self.assertFalse(context.metadata["precision_reference_adjustments"]["cuda_matmul_allow_tf32"])
            self.assertFalse(context.metadata["precision_reference_adjustments"]["flash_sdp_enabled"])
            self.assertTrue(context.metadata["precision_reference_adjustments"]["math_sdp_enabled"])
            self.assertEqual(len(context.metadata["precision_reference_adjustments"]["attention_configs"]), 2)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            torch.set_float32_matmul_precision(old_precision)
            torch.backends.cuda.enable_flash_sdp(old_sdp[0])
            torch.backends.cuda.enable_mem_efficient_sdp(old_sdp[1])
            torch.backends.cuda.enable_math_sdp(old_sdp[2])
            torch.backends.cuda.enable_cudnn_sdp(old_sdp[3])

    def test_fp32_reference_batch_move_keeps_integer_and_boolean_inputs(self):
        batch = {"image": torch.ones(2, dtype=torch.bfloat16),
                 "labels": torch.ones(2, dtype=torch.long),
                 "mask": torch.ones(2, dtype=torch.bool),
                 "nested": [torch.ones(1, dtype=torch.float16)]}
        converted = _move(batch, "cpu", float_dtype=torch.float32)
        self.assertEqual(converted["image"].dtype, torch.float32)
        self.assertEqual(converted["nested"][0].dtype, torch.float32)
        self.assertEqual(converted["labels"].dtype, torch.long)
        self.assertEqual(converted["mask"].dtype, torch.bool)

    def test_fp32_reference_cli_keeps_bf16_training_loader(self):
        args = parse_args(["--checkpoint-dir", "checkpoint", "--output", "report.json",
                           "--fp32-reference", "--deterministic-probe"])
        self.assertTrue(args.fp32_reference)
        self.assertTrue(args.bf16)
        self.assertTrue(args.deterministic_probe)

    def test_deterministic_attention_disables_vision_flash_and_fused_sdpa(self):
        model = torch.nn.Module()
        model.vision_tower = torch.nn.Linear(2, 2)
        model.vision_tower.use_flash_attn = True
        model.vision_tower.config = SimpleNamespace(use_flash_attn=True)
        old_sdp = (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
            torch.backends.cuda.cudnn_sdp_enabled(),
        )
        try:
            changes = _prepare_deterministic_attention(model)
            self.assertFalse(model.vision_tower.use_flash_attn)
            self.assertFalse(model.vision_tower.config.use_flash_attn)
            self.assertFalse(changes["flash_sdp_enabled"])
            self.assertFalse(changes["mem_efficient_sdp_enabled"])
            self.assertTrue(changes["math_sdp_enabled"])
            self.assertFalse(changes["cudnn_sdp_enabled"])
        finally:
            torch.backends.cuda.enable_flash_sdp(old_sdp[0])
            torch.backends.cuda.enable_mem_efficient_sdp(old_sdp[1])
            torch.backends.cuda.enable_math_sdp(old_sdp[2])
            torch.backends.cuda.enable_cudnn_sdp(old_sdp[3])


if __name__ == "__main__":
    unittest.main()
