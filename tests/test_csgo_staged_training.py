import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import Mock

import torch
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel, PretrainedConfig, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerControl, TrainerState

from unilip.train.csgo_staged_training import (
    StopAndSaveAtStepCallback,
    SaveAtExactStepsCallback,
    enable_microbatch_mean_loss,
    save_final_model_if_enabled,
    validate_checkpoint_saving,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x, **kwargs):
        return self.linear(x)


class TinyTrainModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, input_ids, labels=None):
        output = self.linear(input_ids.float())
        loss = torch.nn.functional.mse_loss(output, labels.float())
        return {"loss": loss, "logits": output}


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, length=20):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor([float(index % 3), 1.0]),
            "labels": torch.tensor([float(index % 2)]),
        }


class CSGOStagedTrainingTests(unittest.TestCase):
    def test_stop_saves_at_absolute_optimizer_step(self):
        callback = StopAndSaveAtStepCallback(2400)
        state = TrainerState(global_step=2399, max_steps=10000)
        control = TrainerControl()

        self.assertIs(callback.on_step_end(None, state, control), control)
        self.assertFalse(control.should_save)
        self.assertFalse(control.should_training_stop)

        state.global_step = 2400
        callback.on_step_end(None, state, control)
        self.assertTrue(control.should_save)
        self.assertTrue(control.should_training_stop)
        self.assertEqual(state.max_steps, 10000)

    def test_invalid_stop_rejected(self):
        for value in (0, -1, True, 2.5, "2400"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                StopAndSaveAtStepCallback(value)

    def test_exact_checkpoint_plan_validation(self):
        with tempfile.TemporaryDirectory() as output_dir:
            args = TrainingArguments(
                output_dir=output_dir, max_steps=5, save_strategy="no",
                save_total_limit=2, save_only_model=False, use_cpu=True, report_to="none",
            )
            self.assertIsInstance(
                validate_checkpoint_saving(args, [2, 5], 2, False),
                SaveAtExactStepsCallback,
            )
            self.assertIsNone(validate_checkpoint_saving(args, None, None, True))
            for steps in ([], [2, 2, 5], [5, 2], [0, 5], [True, 5], [2.0, 5], (2, 5), [2, 4]):
                with self.subTest(steps=steps), self.assertRaises(ValueError):
                    validate_checkpoint_saving(args, steps, 2, False)
            with self.assertRaisesRegex(ValueError, "must be in"):
                validate_checkpoint_saving(args, [2, 5], 3, False)
            with self.assertRaisesRegex(ValueError, "requires training_checkpoint_steps"):
                validate_checkpoint_saving(args, None, None, False)
            with self.assertRaisesRegex(ValueError, "must be a boolean"):
                validate_checkpoint_saving(args, [2, 5], None, 0)
            args.save_only_model = True
            with self.assertRaisesRegex(ValueError, "save_only_model"):
                validate_checkpoint_saving(args, [2, 5], None, False)
            args.save_only_model = False
            args.save_total_limit = 1
            with self.assertRaisesRegex(ValueError, "save_total_limit"):
                validate_checkpoint_saving(args, [2, 5], None, False)
            args.save_total_limit = 2
            args.save_strategy = "steps"
            with self.assertRaisesRegex(ValueError, "save_strategy"):
                validate_checkpoint_saving(args, [2, 5], None, False)

    def test_epoch_end_cannot_request_unlisted_checkpoint(self):
        callback = SaveAtExactStepsCallback([2, 5])
        state = TrainerState(global_step=3, max_steps=5)
        control = TrainerControl(should_save=True)
        callback.on_step_end(None, state, control)
        self.assertFalse(control.should_save)
        state.global_step = 5
        callback.on_step_end(None, state, control)
        self.assertTrue(control.should_save)
        callback.on_epoch_end(None, state, control)
        self.assertFalse(control.should_save)

    def test_precise_checkpoints_stop_and_resume_with_same_horizon(self):
        with tempfile.TemporaryDirectory() as output_dir:
            args = TrainingArguments(
                output_dir=output_dir, max_steps=5, per_device_train_batch_size=1,
                save_strategy="no", save_only_model=False, save_total_limit=2,
                use_cpu=True, report_to="none", disable_tqdm=True,
            )
            steps = [2, 5]
            def make_trainer(stop):
                callbacks = [validate_checkpoint_saving(args, steps, stop, False)]
                if stop is not None:
                    callbacks.append(StopAndSaveAtStepCallback(stop))
                return Trainer(
                    model=TinyTrainModel(PretrainedConfig()), args=args,
                    train_dataset=TinyDataset(), callbacks=callbacks,
                )

            paused = make_trainer(2)
            paused.train()
            root = Path(output_dir)
            self.assertEqual(paused.state.global_step, 2)
            self.assertEqual(sorted(p.name for p in root.glob("checkpoint-*")), ["checkpoint-2"])
            paused_state = json.loads((root / "checkpoint-2" / "trainer_state.json").read_text())
            self.assertEqual((paused_state["global_step"], paused_state["max_steps"]), (2, 5))
            for filename in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
                self.assertTrue((root / "checkpoint-2" / filename).is_file(), filename)

            resumed = make_trainer(None)
            resumed.train(resume_from_checkpoint=str(root / "checkpoint-2"))
            resumed.save_state()
            save_final_model_if_enabled(False, resumed.save_model, output_dir=output_dir)
            self.assertEqual(resumed.state.global_step, 5)
            self.assertEqual(sorted(p.name for p in root.glob("checkpoint-*")), ["checkpoint-2", "checkpoint-5"])
            final_state = json.loads((root / "checkpoint-5" / "trainer_state.json").read_text())
            self.assertEqual((final_state["global_step"], final_state["max_steps"]), (5, 5))
            self.assertFalse((root / "model.safetensors").exists())
            self.assertFalse((root / "adapter_model.safetensors").exists())
            self.assertTrue((root / "trainer_state.json").is_file())

    def test_full_run_saves_only_selected_steps_across_epochs(self):
        with tempfile.TemporaryDirectory() as output_dir:
            args = TrainingArguments(
                output_dir=output_dir, max_steps=5, per_device_train_batch_size=1,
                save_strategy="no", save_only_model=False, save_total_limit=2,
                use_cpu=True, report_to="none", disable_tqdm=True,
            )
            trainer = Trainer(
                model=TinyTrainModel(PretrainedConfig()), args=args,
                train_dataset=TinyDataset(length=1),
                callbacks=[validate_checkpoint_saving(args, [2, 5], None, False)],
            )
            trainer.train()
            self.assertEqual(trainer.state.global_step, 5)
            self.assertEqual(
                sorted(p.name for p in Path(output_dir).glob("checkpoint-*")),
                ["checkpoint-2", "checkpoint-5"],
            )

    def test_final_root_export_is_opt_in_for_exact_checkpoint_runs(self):
        save_model = Mock()
        save_final_model_if_enabled(False, save_model, output_dir="unused")
        save_model.assert_not_called()
        save_final_model_if_enabled(True, save_model, output_dir="default")
        save_model.assert_called_once_with(output_dir="default")

    def test_peft_base_is_the_model_inspected_by_trainer(self):
        model = get_peft_model(
            TinyModel(), LoraConfig(r=2, lora_alpha=2, target_modules=["linear"])
        )
        base_model = enable_microbatch_mean_loss(model)
        self.assertIs(base_model, model.get_base_model())
        self.assertIs(base_model.accepts_loss_kwargs, False)
        self.assertIs(model.accepts_loss_kwargs, False)

        with tempfile.TemporaryDirectory() as output_dir:
            args = TrainingArguments(output_dir=output_dir, report_to="none", use_cpu=True)
            trainer = Trainer(model=model, args=args)
            self.assertIs(trainer.model_accepts_loss_kwargs, False)


if __name__ == "__main__":
    unittest.main()
