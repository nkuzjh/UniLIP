import tempfile
import unittest

import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerControl, TrainerState

from unilip.train.csgo_staged_training import (
    StopAndSaveAtStepCallback,
    enable_microbatch_mean_loss,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x, **kwargs):
        return self.linear(x)


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
