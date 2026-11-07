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


class SaveAtExactStepsCallback(TrainerCallback):
    """Restrict Trainer checkpoint saves to selected optimizer steps."""

    def __init__(self, steps):
        if (
            not isinstance(steps, list)
            or not steps
            or any(isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in steps)
            or steps != sorted(set(steps))
        ):
            raise ValueError("training_checkpoint_steps must be a sorted, unique list of positive integers")
        self.steps = frozenset(steps)

    def on_step_end(self, args, state, control, **kwargs):
        # DefaultFlowCallback runs first and may request a periodic save.
        control.should_save = state.global_step in self.steps
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        # An epoch boundary must not add another checkpoint.
        control.should_save = False
        return control


def validate_checkpoint_saving(training_args, steps, stop_after_step, save_final_model_at_output_root):
    """Validate the opt-in, fully resumable checkpoint plan before training."""
    if not isinstance(save_final_model_at_output_root, bool):
        raise ValueError("save_final_model_at_output_root must be a boolean")
    if steps is None:
        if not save_final_model_at_output_root:
            raise ValueError("save_final_model_at_output_root=false requires training_checkpoint_steps")
        return None

    callback = SaveAtExactStepsCallback(steps)
    if training_args.max_steps <= 0 or steps[-1] != training_args.max_steps:
        raise ValueError("training_checkpoint_steps must end at training max_steps")
    if stop_after_step is not None and stop_after_step not in callback.steps:
        raise ValueError("training_stop_after_step must be in training_checkpoint_steps")
    if training_args.save_strategy != "no":
        raise ValueError("training_checkpoint_steps requires save_strategy=no")
    if training_args.save_only_model:
        raise ValueError("training_checkpoint_steps requires save_only_model=False")
    if training_args.save_total_limit is not None and training_args.save_total_limit < len(steps):
        raise ValueError("save_total_limit must retain every training_checkpoint_steps checkpoint")
    return callback


def save_final_model_if_enabled(enabled, save_model, **kwargs):
    """Skip the root model and adapter export for precise checkpoint runs."""
    if enabled:
        save_model(**kwargs)


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
