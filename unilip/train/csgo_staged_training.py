"""Opt-in controls for staged CSGO training runs."""

from peft import PeftModel
from transformers import TrainerCallback


class StopAndSaveAtStepCallback(TrainerCallback):
    """Save a resumable checkpoint and stop at an absolute optimizer step."""

    def __init__(self, stop_after_step: int):
        if isinstance(stop_after_step, bool) or not isinstance(stop_after_step, int) or stop_after_step <= 0:
            raise ValueError("training_stop_after_step must be a positive integer")
        self.stop_after_step = stop_after_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_step:
            control.should_save = True
            control.should_training_stop = True
        return control


def enable_microbatch_mean_loss(model):
    """Tell Trainer that CSGO's model loss is already a microbatch mean.

    Trainer inspects the unwrapped PEFT base model, not the outer wrapper. Set
    both so the same declaration is visible on either path.
    """
    # Use PEFT's public interface rather than a version-specific Trainer helper.
    unwrapped_model = model.get_base_model() if isinstance(model, PeftModel) else model
    unwrapped_model.accepts_loss_kwargs = False
    model.accepts_loss_kwargs = False
    return unwrapped_model
