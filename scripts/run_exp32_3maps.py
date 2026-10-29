#!/usr/bin/env python3
"""Plan or explicitly execute one exp32 Seen-3 experiment stage.

No arguments launch a queue. Commands are printed only unless --execute is set.
The original two rounds retain their 50-epoch schedule. Stage-two arms
use a new 1800-step schedule, initialized from stage-one model weights only.
--resume restores the complete checkpoint in the selected arm's output directory.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "exp32_3maps_"
ROUND1 = ("loc_single", "gen_single", "joint_original", "joint_headlr",
          "joint_sharedlr", "joint_constloc", "joint_sharedlr_constloc")
ROUND1_EXTRA = ("joint_original_constloc", "joint_constloc1",
                "joint_constloc4", "joint_constloc8",
                "joint_constloc12", "joint_constloc16")
ROUND2 = ("aux_control", "perception", "auxloc", "both")
ROUND2_J8 = ("perception_j8", "auxloc_j8", "both_j8")
STAGE2_J8 = ("stage2_control_j8", "stage2_perception_j8",
             "stage2_auxloc_j8", "stage2_both_j8")
# Each new arm is (original Stage 2 recipe, changed fields, removed fields).
STAGE2_NEW_ARMS = {
    "P1_S1_J8": ("stage2_perception_j8", {"alpha_loc_perception_loss": 0.1},
                 ("alpha_loc_perception_schedule_steps", "alpha_loc_perception_schedule_values")),
    "U1_S2_J8": ("stage2_auxloc_j8", {"alpha_loc_aux_loss": 2.0},
                 ("alpha_loc_aux_schedule_steps", "alpha_loc_aux_schedule_values")),
    "B1_S3_J8": ("stage2_both_j8", {"alpha_loc_perception_loss": 0.0,
                                   "alpha_loc_aux_loss": 0.0,
                                   "alpha_loc_perception_schedule_values": [0.0, 0.05],
                                   "alpha_loc_aux_schedule_values": [0.0, 1.0]}, ()),
    "P2_S1_J8": ("stage2_perception_j8", {"alpha_loc_perception_schedule_values": [0.0, 0.2]}, ()),
    "P3_S1_J8": ("stage2_perception_j8", {"alpha_loc_perception_schedule_values": [0.0, 0.05]}, ()),
    "U2_S2_J8": ("stage2_auxloc_j8", {"alpha_loc_aux_schedule_values": [0.0, 1.0]}, ()),
    "U3_S2_J8": ("stage2_auxloc_j8", {"alpha_loc_aux_schedule_values": [0.0, 4.0]}, ()),
    "P4_S1_J8": ("stage2_perception_j8", {"loc_perception_use_attention_weight": False,
                                          "alpha_loc_perception_schedule_values": [0.0, 0.05]}, ()),
    "U4_S2_J8": ("stage2_auxloc_j8", {"aux_loc_combined_candidate_tau": 0.001,
                                    "alpha_loc_aux_schedule_values": [0.0, 1.0]}, ()),
    "U5_S2_J8": ("stage2_auxloc_j8", {"alpha_loc_aux_schedule_values": [0.0, 0.5]}, ()),
    "P5_S1_J8": ("stage2_perception_j8", {"alpha_loc_perception_schedule_values": [0.0, 0.025]}, ()),
}
STAGE2_NEW_EXPERIMENTS = tuple(PREFIX + name for name in STAGE2_NEW_ARMS)
STAGE2_SINGLE = ("stage2_control_g0", "stage2_control_l0")
STAGE2_SINGLE_EXPERIMENTS = tuple(PREFIX + name for name in STAGE2_SINGLE)
LEGACY_EXPERIMENTS = tuple(PREFIX + name for name in ROUND1 + ROUND1_EXTRA + ROUND2 + ROUND2_J8 + STAGE2_J8 + tuple(STAGE2_NEW_ARMS))
EXPERIMENTS = LEGACY_EXPERIMENTS + STAGE2_SINGLE_EXPERIMENTS
JOINT_BASES = tuple(PREFIX + name for name in ROUND1[2:] + ROUND1_EXTRA)
MAPS = ["de_ancient", "de_dust2", "de_nuke"]
DEFAULT_BASE = PREFIX + "joint_sharedlr_constloc"
J8_BASE = PREFIX + "joint_constloc8"
STAGE2_SOURCE = ("outputs/csgo_1b/exp32_3maps_joint_constloc8/"
                 "checkpoint-5850/model.safetensors")
STAGE2_SINGLE_SOURCES = {
    PREFIX + "stage2_control_g0": (PREFIX + "gen_single", 2578, 876,
                                    ("language_model.model.", "model.llm_connector.", "model.dit."),
                                    ("model.action_dit.",)),
    PREFIX + "stage2_control_l0": (PREFIX + "loc_single", 2290, 588,
                                    ("language_model.model.", "model.action_dit."),
                                    ("model.llm_connector.", "model.dit.")),
}
STAGE2_MAX_STEPS = 1800
STAGE2_SAVE_STEPS = 300
STAGE2_NEW_SAVE_STEPS = 600
STAGE2_WARMUP_STEPS = 60
STAGE2_LR_PEAKS = {
    "learning_rate": 2e-5, "mm_projector_lr": 2e-5,
    "action_dit_projector_lr": 1e-4, "action_dit_lr": 2e-5,
    "llm_lora_lr": 2e-5, "llm_connector_lora_lr": 2e-5,
    "gen_dit_lora_lr": 2e-5, "action_dit_lora_lr": 2e-5,
    "action_dit_connector_lr": 1e-4, "action_dit_norm_lr": 1e-4,
    "action_io_mlp_lr": 2e-5,
}
STAGE2_LR_KEYS = tuple(STAGE2_LR_PEAKS)
BASE_KEYS = (
    "alpha_loc_loss", "alpha_loc_schedule_steps", "alpha_loc_schedule_values",
    "llm_lora_lr", "llm_connector_lora_lr", "gen_dit_lora_lr",
    "action_dit_lora_lr", "action_dit_connector_lr", "action_dit_norm_lr",
    "action_io_mlp_lr", "action_dit_projector_lr", "action_dit_lr",
    "learning_rate", "mm_projector_lr", "weight_decay", "warmup_ratio",
    "lr_scheduler_type", "lr_scheduler_kwargs", "lora_r", "lora_alpha",
    "lora_dropout",
)


def read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def config_path(experiment: str) -> Path:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    return ROOT / "csgo_configs" / f"{experiment}.yaml"


def is_stage2(experiment: str) -> bool:
    return experiment in tuple(PREFIX + name for name in STAGE2_J8) + STAGE2_NEW_EXPERIMENTS + STAGE2_SINGLE_EXPERIMENTS


def stage2_save_steps(experiment: str) -> int:
    return STAGE2_NEW_SAVE_STEPS if experiment in STAGE2_NEW_EXPERIMENTS + STAGE2_SINGLE_EXPERIMENTS else STAGE2_SAVE_STEPS


def is_round2(experiment: str) -> bool:
    return experiment.removeprefix(PREFIX) in ROUND2 + ROUND2_J8


def resolve_config(args: argparse.Namespace, *, for_training: bool) -> dict:
    source = read_yaml(config_path(args.experiment))
    out = ROOT / "outputs/csgo_1b" / args.experiment
    snapshot = out / "resolved_config.yaml"
    if is_stage2(args.experiment):
        if args.joint_base:
            raise ValueError("Stage 2 uses fixed source weights; --joint-base is not allowed")
        if source.get("exp32_3maps_experiment") != args.experiment:
            raise ValueError("Stage 2 recipe identity differs from experiment")
    if args.experiment.removeprefix(PREFIX) in ROUND2_J8:
        if args.joint_base and args.joint_base != J8_BASE:
            raise ValueError("J8 experiment requires --joint-base exp32_3maps_joint_constloc8")
        if snapshot.is_file():
            if read_yaml(snapshot).get("exp32_3maps_joint_base") != J8_BASE:
                raise ValueError("Saved J8 training recipe has a different joint base")
        elif source.get("exp32_3maps_joint_base") != J8_BASE:
            raise ValueError("J8 experiment requires the joint_constloc8 base in its source recipe")
    trained = None
    # Evaluation follows the recipe actually trained, even after editing a YAML.
    if not for_training and snapshot.is_file():
        source = read_yaml(snapshot)
        if is_stage2(args.experiment) and source.get("exp32_3maps_experiment") != args.experiment:
            raise ValueError("Saved stage 2 recipe belongs to another experiment")
        trained = copy.deepcopy(source)
    elif is_round2(args.experiment):
        saved_base = None
        if for_training and args.resume and snapshot.is_file():
            saved_base = read_yaml(snapshot).get("exp32_3maps_joint_base")
        base_name = args.joint_base or saved_base or source.get("exp32_3maps_joint_base", DEFAULT_BASE)
        if args.experiment.removeprefix(PREFIX) in ROUND2_J8 and base_name != J8_BASE:
            raise ValueError("J8 experiment requires the joint_constloc8 base")
        if base_name not in JOINT_BASES:
            raise ValueError(f"Invalid joint base: {base_name}")
        base = read_yaml(config_path(base_name))
        for key in BASE_KEYS:
            source.pop(key, None)
            if key in base:
                source[key] = copy.deepcopy(base[key])
        source["exp32_3maps_joint_base"] = base_name
    elif args.joint_base:
        raise ValueError("--joint-base is only valid for round two")
    if not for_training and args.joint_base:
        if source.get("exp32_3maps_joint_base") != args.joint_base:
            raise ValueError("Evaluation joint base differs from the saved training recipe")
    if args.asset_mode == "full":
        source["benchmark_v2_asset_manifest"] = None
    elif args.asset_mode == "minimal":
        source["benchmark_v2_asset_manifest"] = (
            args.asset_manifest or "data/csgo_benchmark_v2/minimal_dataset_report.json"
        )
    elif args.asset_manifest:
        source["benchmark_v2_asset_manifest"] = args.asset_manifest
    if args.data_dir:
        source["data_dir"] = args.data_dir
    if args.pi05_path:
        source["pi05_pytorch_weight_path"] = str(Path(args.pi05_path).expanduser())
    if args.model_path:
        source["model_name_or_path"] = args.model_path
    source.setdefault("model_name_or_path", "UniLIP-1B")
    if trained is not None:
        for key in ("benchmark_v2_asset_manifest", "data_dir", "pi05_pytorch_weight_path",
                    "model_name_or_path"):
            if source.get(key) != trained.get(key, "UniLIP-1B" if key == "model_name_or_path" else None):
                raise ValueError(f"Evaluation {key} differs from the saved training recipe")
    if for_training:
        source["seed"] = args.seed
    if for_training and args.stop_after_step is not None:
        if args.stop_after_step < 0:
            raise ValueError("--stop-after-step must be zero (full horizon) or positive")
        if args.stop_after_step == 0:
            source.pop("training_stop_after_step", None)
        else:
            source["training_stop_after_step"] = args.stop_after_step
    return source


def validate_config(config: dict) -> None:
    for key in ("train_maps", "val_maps", "test_maps"):
        if config.get(key) != MAPS:
            raise ValueError(f"{key} must preserve the fixed Seen-3 map order")
    if config.get("benchmark_v2_split") != "seen_train":
        raise ValueError("Training recipes must use seen_train")
    if config.get("csgo_loss_per_microbatch_mean") is not True:
        raise ValueError("All arms must use microbatch-mean GAS normalization")
    if config.get("is_lora") is not True or config.get("llm_train_mode") != "lora":
        raise ValueError("Expected the exp32 LoRA architecture")
    if is_stage2(config.get("exp32_3maps_experiment", "")):
        validate_stage2_config(config)
    elif any(config.get(key) for key in (
        "resume_ckpt_path", "finetune_init_ckpt_path", "base_init_ckpt_path",
        "gen_init_ckpt_path", "loc_init_ckpt_path",
    )):
        raise ValueError("Each arm must start from the common original initialization")
    for stem in ("alpha_loc", "alpha_loc_aux", "alpha_loc_perception"):
        steps, values = config.get(stem + "_schedule_steps"), config.get(stem + "_schedule_values")
        if (steps is None) != (values is None):
            raise ValueError(f"Incomplete {stem} schedule")
        if steps is not None and (len(steps) != len(values) or not steps
                                  or steps != sorted(set(steps))):
            raise ValueError(f"Invalid {stem} schedule")


def stage2_source_path(experiment: str) -> str:
    if experiment in STAGE2_SINGLE_SOURCES:
        base = STAGE2_SINGLE_SOURCES[experiment][0]
        return f"outputs/csgo_1b/{base}/checkpoint-5850/model.safetensors"
    return STAGE2_SOURCE


def validate_stage2_single_config(config: dict) -> None:
    experiment = config["exp32_3maps_experiment"]
    base_name = STAGE2_SINGLE_SOURCES[experiment][0]
    expected = read_yaml(config_path(base_name))
    expected.pop("training_stop_after_step", None)
    expected.update(STAGE2_LR_PEAKS)
    expected.update(weight_decay=0.0, warmup_ratio=0.0,
                    lr_scheduler_type="cosine_with_min_lr",
                    lr_scheduler_kwargs={"min_lr_rate": 0.1},
                    exp32_3maps_experiment=experiment,
                    finetune_init_ckpt_path=stage2_source_path(experiment),
                    finetune_init_strict=True)
    allowed = {"benchmark_v2_asset_manifest", "data_dir", "pi05_pytorch_weight_path",
               "model_name_or_path", "seed", "training_stop_after_step"}
    for key in set(expected) | set(config):
        if key not in allowed and (key not in expected or key not in config
                                   or config[key] != expected[key]):
            raise ValueError(f"Invalid single-task stage 2 recipe: {key}")
    stop = config.get("training_stop_after_step")
    if "training_stop_after_step" in config and (type(stop) is not int or stop <= 0):
        raise ValueError("Invalid stage 2 stop")
    if stop is not None and stop > STAGE2_MAX_STEPS:
        raise ValueError("Stage 2 stop exceeds its 1800-step horizon")
    if any(config.get(key) for key in ("resume_ckpt_path", "base_init_ckpt_path",
                                       "gen_init_ckpt_path", "loc_init_ckpt_path")):
        raise ValueError("Stage 2 cannot use another initialization or source resume")


def validate_stage2_config(config: dict) -> None:
    experiment = config["exp32_3maps_experiment"]
    if experiment in STAGE2_SINGLE_EXPERIMENTS:
        validate_stage2_single_config(config)
        return
    suffix = experiment.removeprefix(PREFIX)
    if suffix in STAGE2_NEW_ARMS:
        base_name, changes, removed = STAGE2_NEW_ARMS[suffix]
        expected = read_yaml(config_path(PREFIX + base_name))
        expected.update(copy.deepcopy(changes))
        for key in removed:
            expected.pop(key)
        expected["exp32_3maps_experiment"] = experiment
        expected["training_stop_after_step"] = STAGE2_MAX_STEPS
        runtime_overrides = {"benchmark_v2_asset_manifest", "data_dir", "pi05_pytorch_weight_path",
                             "model_name_or_path", "seed", "training_stop_after_step"}
        for key in set(expected) | set(config):
            if key not in runtime_overrides and (key not in expected or key not in config
                                                 or config[key] != expected[key]):
                raise ValueError(f"Invalid stage 2 {suffix} single-factor recipe: {key}")
    if config.get("finetune_init_ckpt_path") != STAGE2_SOURCE or config.get("finetune_init_strict") is not True:
        raise ValueError("Stage 2 must strictly initialize from J8 checkpoint-5850")
    if any(config.get(key) for key in ("resume_ckpt_path", "base_init_ckpt_path",
                                       "gen_init_ckpt_path", "loc_init_ckpt_path")):
        raise ValueError("Stage 2 cannot use another initialization or J8 resume")
    if config.get("alpha_loc_loss") != 8 or any(config.get(key) for key in
        ("alpha_loc_schedule_steps", "alpha_loc_schedule_values")):
        raise ValueError("Stage 2 requires constant alpha_loc=8")
    if config.get("exp32_3maps_joint_base") != J8_BASE:
        raise ValueError("Stage 2 requires J8 identity")
    if config.get("training_stop_after_step", STAGE2_MAX_STEPS) > STAGE2_MAX_STEPS:
        raise ValueError("Stage 2 stop exceeds its 1800-step horizon")
    for key, expected_lr in STAGE2_LR_PEAKS.items():
        if config.get(key) != expected_lr:
            raise ValueError(f"Stage 2 {key} must match its fixed LR peak")
    for key, expected in (("weight_decay", 0), ("warmup_ratio", 0),
                          ("lr_scheduler_type", "cosine_with_min_lr"),
                          ("lr_scheduler_kwargs", {"min_lr_rate": 0.1})):
        if config.get(key) != expected:
            raise ValueError(f"Invalid stage 2 {key}")
    reference = STAGE2_NEW_ARMS[suffix][0] if suffix in STAGE2_NEW_ARMS else suffix
    aux = reference in ("stage2_auxloc_j8", "stage2_both_j8")
    perception = reference in ("stage2_perception_j8", "stage2_both_j8")
    aux_ramp = aux and suffix != "U1_S2_J8"
    perception_ramp = perception and suffix != "P1_S1_J8"
    aux_end = {"B1_S3_J8": 1.0, "U2_S2_J8": 1.0, "U3_S2_J8": 4.0,
               "U4_S2_J8": 1.0, "U5_S2_J8": 0.5}.get(suffix, 2.0)
    perception_end = {"B1_S3_J8": 0.05, "P2_S1_J8": 0.2, "P3_S1_J8": 0.05,
                      "P4_S1_J8": 0.05, "P5_S1_J8": 0.025}.get(suffix, 0.1)
    expected = (("is_loc_aux_loss", aux),
                ("is_aux_loc_combined_em_unc_loss", aux),
                ("is_loc_perception_loss", perception),
                ("alpha_loc_aux_loss", 2.0 if aux and not aux_ramp else 0.0),
                ("alpha_loc_perception_loss", 0.1 if perception and not perception_ramp else 0.0),
                ("alpha_loc_aux_schedule_steps", [0, 120] if aux_ramp else None),
                ("alpha_loc_aux_schedule_values", [0.0, aux_end] if aux_ramp else None),
                ("alpha_loc_perception_schedule_steps", [0, 120] if perception_ramp else None),
                ("alpha_loc_perception_schedule_values", [0.0, perception_end] if perception_ramp else None))
    for key, value in expected:
        if config.get(key) != value:
            raise ValueError(f"Invalid stage 2 {key}")


def fingerprint(config: dict, runtime: dict) -> str:
    immutable = copy.deepcopy(config)
    immutable.pop("training_stop_after_step", None)
    raw = json.dumps({"config": immutable, "runtime": runtime}, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def latest_checkpoint(out: Path) -> Path | None:
    choices = []
    for path in out.glob("checkpoint-*"):
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit():
            choices.append((int(path.name.split("-")[-1]), path))
    return max(choices)[1] if choices else None


def final_checkpoint_step(experiment: str) -> int:
    """Resolve a completed training checkpoint without guessing from its name."""
    out = ROOT / "outputs/csgo_1b" / experiment
    if not (out / "resolved_config.yaml").is_file():
        raise ValueError(f"Final evaluation requires the saved training recipe: {out}")

    def state_at(path: Path) -> tuple[int, int]:
        if not path.is_file():
            raise ValueError(f"Final checkpoint is missing Trainer state: {path}")
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError(f"Invalid Trainer state: {path}")
        step, horizon = state.get("global_step"), state.get("max_steps")
        if (type(step) is not int or type(horizon) is not int
                or step <= 0 or horizon < step):
            raise ValueError(f"Invalid Trainer global_step/max_steps: {path}")
        return step, horizon

    root_state = out / "trainer_state.json"
    if root_state.is_file():
        step, horizon = state_at(root_state)
        if step != horizon:
            raise ValueError(f"Training has not reached max_steps: {root_state}")
    else:
        completed = []
        for checkpoint in out.glob("checkpoint-*"):
            suffix = checkpoint.name.removeprefix("checkpoint-")
            if not checkpoint.is_dir() or not suffix.isdigit():
                continue
            checkpoint_state = checkpoint / "trainer_state.json"
            if checkpoint_state.is_file():
                saved_step, horizon = state_at(checkpoint_state)
                if saved_step != int(suffix):
                    raise ValueError(f"Checkpoint directory and Trainer step differ: {checkpoint}")
                if saved_step == horizon:
                    completed.append(saved_step)
        if len(completed) != 1:
            raise ValueError(f"Expected one completed final checkpoint, found {len(completed)}: {out}")
        step = completed[0]

    checkpoint = out / f"checkpoint-{step}"
    saved_step, saved_horizon = state_at(checkpoint / "trainer_state.json")
    if saved_step != step or saved_horizon != step:
        raise ValueError(f"Final checkpoint step/global_step/max_steps differ: {checkpoint}")
    if not (checkpoint / "model.safetensors").is_file():
        raise ValueError(f"Final checkpoint is missing model weights: {checkpoint}")
    return step


def check_resume(out: Path, identity: str, resume: bool) -> Path | None:
    checkpoint = latest_checkpoint(out)
    metadata = out / "experiment_plan.json"
    if not resume:
        if checkpoint or (out / "model.safetensors").exists():
            raise ValueError(f"Existing run at {out}; use --resume to continue its exact recipe")
        if metadata.exists() and json.loads(metadata.read_text()).get("fingerprint") != identity:
            raise ValueError("An earlier launch used a different recipe in this output directory")
        return None
    if checkpoint is None or not metadata.is_file():
        raise ValueError("--resume requires a managed run and a complete checkpoint")
    previous = json.loads(metadata.read_text(encoding="utf-8"))
    if previous.get("fingerprint") != identity:
        raise ValueError("Resume recipe/layout changed; only the stopping step may change")
    required = ("trainer_state.json", "training_args.bin", "model.safetensors")
    if any(not (checkpoint / name).is_file() for name in required):
        raise ValueError(f"Incomplete checkpoint metadata/model: {checkpoint}")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    step, horizon = state.get("global_step"), state.get("max_steps")
    if (type(step) is not int or step != int(checkpoint.name.split("-")[-1])
            or type(horizon) is not int or horizon < step):
        raise ValueError(f"Checkpoint step/global_step or training horizon is inconsistent: {checkpoint}")
    if not list(checkpoint.glob("rng_state*.pth")):
        raise ValueError(f"Missing checkpoint RNG state: {checkpoint}")
    optimizer = (checkpoint / "optimizer.pt").is_file()
    deepspeed = any(checkpoint.glob("global_step*/*optim_states.pt"))
    if not (optimizer and (checkpoint / "scheduler.pt").is_file()) and not deepspeed:
        raise ValueError(f"Missing optimizer/scheduler/DeepSpeed state: {checkpoint}")
    return checkpoint


def training_plan(args: argparse.Namespace, config: dict) -> tuple[list[list[str]], dict, Path]:
    devices = args.cuda_devices.split(",")
    if not all(d.strip() for d in devices) or len(set(devices)) != len(devices):
        raise ValueError("--cuda-devices must be a nonempty list of distinct devices")
    micro_batch = args.micro_batch if args.micro_batch is not None else (
        4 if args.experiment in STAGE2_SINGLE_EXPERIMENTS else 64)
    if micro_batch <= 0 or 128 % (len(devices) * micro_batch):
        raise ValueError("GPU count × micro batch must divide effective source batch 128")
    accumulation = 128 // (len(devices) * micro_batch)
    runtime = dict(epochs=50, world_size=len(devices), micro_batch=micro_batch,
                   gradient_accumulation_steps=accumulation, seed=args.seed,
                   save_steps=1200, save_total_limit=6, report_to=args.report_to)
    if is_stage2(args.experiment):
        runtime.update(max_steps=STAGE2_MAX_STEPS, warmup_steps=STAGE2_WARMUP_STEPS,
                       save_steps=stage2_save_steps(args.experiment), optimizer="adamw_torch",
                       adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
                       max_grad_norm=1.0)
    out = ROOT / "outputs/csgo_1b" / args.experiment
    identity = fingerprint(config, runtime)
    checkpoint = check_resume(out, identity, args.resume)
    previous = json.loads((out / "experiment_plan.json").read_text()) if checkpoint else None
    if checkpoint and is_stage2(args.experiment):
        source = previous.get("source_checkpoint")
        if not isinstance(source, dict) or source.get("path") != stage2_source_path(args.experiment) or not source.get("sha256"):
            raise ValueError("Stage 2 resume lacks the saved source provenance")
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if state.get("max_steps") != STAGE2_MAX_STEPS:
            raise ValueError("Stage 2 resume checkpoint has a different horizon")
    stop = config.get("training_stop_after_step")
    if checkpoint and stop is not None:
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if int(state["global_step"]) >= stop:
            raise ValueError(f"Checkpoint already reached step {stop}; request a later stop or 0")
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={len(devices)}",
           f"--master_port={args.master_port}", "train_csgo.py",
           "--csgo_config", relative(out / "resolved_config.yaml"),
           "--deepspeed", "deepspeed_scripts/zero0.json",
           "--model_name_or_path", config["model_name_or_path"],
           "--unilip_factor", "10.6", "--mllm_hf_path", config["mllm_hf_path"],
           "--version", "internvl", "--data_type", "mix",
           "--csgo_image_folder", config["data_dir"],
           "--output_dir", relative(out), "--num_train_epochs", "50",
           "--per_device_train_batch_size", str(micro_batch),
           "--per_device_eval_batch_size", "1", "--gradient_accumulation_steps", str(accumulation),
           "--eval_strategy", "no", "--save_strategy", "steps", "--save_steps",
           str(stage2_save_steps(args.experiment) if is_stage2(args.experiment) else 1200),
           "--save_total_limit", "6", "--save_only_model", "False", "--seed", str(args.seed),
           "--learning_rate", str(config["learning_rate"]), "--weight_decay", str(config["weight_decay"]),
           "--warmup_ratio", str(config["warmup_ratio"]), "--lr_scheduler_type", config["lr_scheduler_type"],
           "--model_max_length", "1024", "--logging_steps", "10", "--dataloader_num_workers", "4",
           "--n_query", "256", "--n_und_query", "0", "--report_to", args.report_to,
           "--lora_r", str(config["lora_r"]), "--lora_alpha", str(config["lora_alpha"])]
    for key, value in dict(mm_use_im_start_end=False, mm_use_im_patch_token=False, bf16=True,
                           tf32=True, gradient_checkpointing=True, lazy_preprocess=True,
                           fix_vit=config["fix_vit"], fix_llm=config["fix_llm"],
                           fix_connect=config["fix_connect"], fix_dit=config["fix_dit"]).items():
        cmd.extend(["--" + key, str(value)])
    if is_stage2(args.experiment):
        cmd.extend(["--max_steps", str(STAGE2_MAX_STEPS),
                    "--warmup_steps", str(STAGE2_WARMUP_STEPS),
                    "--optim", "adamw_torch", "--adam_beta1", "0.9",
                    "--adam_beta2", "0.999", "--adam_epsilon", "1e-8",
                    "--max_grad_norm", "1.0"])
    metadata = dict(experiment=args.experiment, fingerprint=identity, runtime=runtime,
                    joint_base=config.get("exp32_3maps_joint_base"), command=cmd,
                    resume_checkpoint=relative(checkpoint) if checkpoint else None)
    if is_stage2(args.experiment):
        metadata["source_checkpoint"] = previous["source_checkpoint"] if checkpoint else None
    return [cmd], metadata, out


def preflight_stage2_source(config: dict) -> dict:
    """Read-only provenance and format checks before any stage 2 launch writes."""
    from safetensors import safe_open

    experiment = config["exp32_3maps_experiment"]
    source_name = stage2_source_path(experiment)
    single = experiment in STAGE2_SINGLE_SOURCES
    label = "single-task" if single else "J8"
    source = repo_path(source_name)
    if not source.is_file():
        raise ValueError(f"Missing {label} source model: {source}")
    source_dir = source.parent
    source_recipe = read_yaml(source_dir.parent / "resolved_config.yaml")
    state_path = source_dir / "trainer_state.json"
    if not state_path.is_file():
        state_path = source_dir.parent / "trainer_state.json"
    if not state_path.is_file():
        raise ValueError(f"{label} source is not complete at checkpoint-5850")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("global_step") != 5850 or state.get("max_steps") != 5850:
        raise ValueError(f"{label} source is not complete at checkpoint-5850")
    if single:
        reference = read_yaml(config_path(STAGE2_SINGLE_SOURCES[experiment][0]))
        allowed = {"training_stop_after_step", "model_name_or_path", "seed", "data_dir",
                   "benchmark_v2_asset_manifest", "pi05_pytorch_weight_path"}
        for key in set(reference) | set(source_recipe):
            if key not in allowed and (key not in reference or key not in source_recipe
                                       or source_recipe[key] != reference[key]):
                raise ValueError(f"{label} source recipe differs from its single-task base: {key}")
    checks = ("train_maps", "val_maps", "test_maps", "benchmark_v2_split",
              "alpha_loc_loss", "action_dim", "is_multi_task", "is_multi_task_balanced",
              "task_mix_ratio", "csgo_loss_per_microbatch_mean", "is_lora",
              "llm_train_mode", "lora_r", "lora_alpha", "lora_dropout",
              "enable_language_model_lora", "enable_gen_head_lora", "enable_loc_head_lora",
              "fix_vit", "fix_llm", "fix_connect", "fix_dit",
              "is_action_dit_projector", "use_pi05_action_dit")
    for key in checks:
        if config.get(key) != source_recipe.get(key):
            raise ValueError(f"{label} source recipe differs in {key}")
    if source_recipe.get("model_name_or_path") != "UniLIP-1B" or source_recipe.get("mllm_hf_path") != "OpenGVLab/InternVL3-1B-hf":
        raise ValueError(f"Unexpected {label} source model identity")
    with safe_open(source, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        lora = sum("lora_" in key for key in keys)
        expected_total, expected_lora = ((STAGE2_SINGLE_SOURCES[experiment][1:3])
                                          if single else (2830, 1128))
        if len(keys) != expected_total or lora != expected_lora:
            raise ValueError(f"Incomplete {label} source tensor header: total={len(keys)}, lora={lora}")
        required_keys = ("language_model.lm_head.weight",
                         "model.action_dit_norm.weight",
                         "model.action_dit_connector.0.bias",
                         "model.action_dit_projector.0.bias",
                         "multi_modal_projector.0.bias")
        required_lora_prefixes = (STAGE2_SINGLE_SOURCES[experiment][3] if single else
                                  ("language_model.model.", "model.llm_connector.",
                                   "model.dit.", "model.action_dit."))
        forbidden_lora_prefixes = STAGE2_SINGLE_SOURCES[experiment][4] if single else ()
        if any(key.startswith(prefix) and "lora_" in key for key in keys
               for prefix in forbidden_lora_prefixes):
            raise ValueError(f"{label} source tensor header has inactive-head LoRA")
        if any(key not in keys for key in required_keys) or any(
            not any(key.startswith(prefix) and "lora_" in key for key in keys)
            for prefix in required_lora_prefixes
        ):
            raise ValueError(f"{label} source tensor header lacks required model/LoRA modules")
        if any(not handle.get_slice(key).get_shape() for key in keys):
            raise ValueError(f"{label} source contains a scalar/empty tensor header")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": source_name, "sha256": digest.hexdigest(),
            "size_bytes": source.stat().st_size, "step": 5850,
            "tensor_count": len(keys), "lora_tensor_count": lora,
            "recipe_path": relative(source_dir.parent / "resolved_config.yaml"),
            "state_path": relative(state_path)}


def ground_truth_dir(config: dict, map_name: str) -> str:
    report_path = config.get("benchmark_v2_asset_manifest")
    if not report_path:
        return str(Path(config["data_dir"]) / map_name / "imgs")
    report_path = repo_path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "verified" or report.get("images", {}).get("status") != "verified":
        raise ValueError(f"Minimal asset report is not verified: {report_path}")
    template = report["images"]["target_template"]
    target = Path(template.format(map=map_name, file_frame="__frame__"))
    if target.is_absolute() or ".." in target.parts:
        raise ValueError("Minimal asset target must remain inside the bundle")
    return relative(report_path.parent / target.parent)


def evaluation_plan(args: argparse.Namespace, config: dict) -> tuple[list[list[str]], Path, dict]:
    step = (final_checkpoint_step(args.experiment) if args.checkpoint_step == "final"
            else args.checkpoint_step)
    if type(step) is not int or step <= 0:
        raise ValueError("infer/metrics require --checkpoint-step with a positive saved global step or final")
    if args.experiment in STAGE2_NEW_EXPERIMENTS + STAGE2_SINGLE_EXPERIMENTS:
        allowed = (1200, 1800) if args.split == "validation" else (1800,)
        if step not in allowed:
            raise ValueError(f"New Stage 2 {args.split} evaluation requires checkpoint step {allowed}")
    if "," in args.cuda_devices:
        raise ValueError("Inference/metrics use one --cuda-devices device")
    checkpoint = Path("outputs/csgo_1b") / args.experiment / f"checkpoint-{step}" / "model.safetensors"
    if args.execute and not repo_path(checkpoint).is_file():
        raise ValueError(f"Checkpoint is absent: {checkpoint}")
    if args.execute and not (ROOT / "outputs/csgo_1b" / args.experiment / "resolved_config.yaml").is_file():
        raise ValueError("Evaluation requires the saved training recipe")
    target = f"{args.experiment}/checkpoint-{step}/{args.split}/seed_{args.seed}"
    eval_config = ROOT / "outputs_eval/benchmark_v2" / target / "evaluation_config.yaml"
    split = "seen_validation" if args.split == "validation" else "seen_discrete_test"
    config = copy.deepcopy(config)
    config["ckpt_path"] = str(checkpoint)
    config["benchmark_v2_split"] = split
    config["seed"] = args.seed
    # Only this Seen-3 evaluation may summarize its ordered protocol map subset.
    config["benchmark_v2_allow_map_subset_summary"] = True
    # These training-only objectives do not add modules in this experiment family.
    for key in ("is_loc_aux_loss", "is_aux_loc_combined_em_unc_loss", "is_loc_perception_loss"):
        config[key] = False
    config.pop("training_stop_after_step", None)
    loc = args.experiment not in (PREFIX + "gen_single", PREFIX + "stage2_control_g0")
    gen = args.experiment not in (PREFIX + "loc_single", PREFIX + "stage2_control_l0")
    if args.task == "localization" and not loc or args.task == "generation" and not gen:
        raise ValueError("Requested task is inactive in this single-task checkpoint")
    loc = loc and args.task in ("all", "localization")
    gen = gen and args.task in ("all", "generation")
    commands = []
    common = ["--csgo_config", relative(eval_config), "--ckpt_path", str(checkpoint),
              "--seed", str(args.seed), "--benchmark_v2_maps", *MAPS]
    if args.command == "infer" and loc:
        commands.append([sys.executable, "eval_csgo_loc.py", *common,
                         "--output_dir", f"outputs_loc/benchmark_v2/{target}",
                         "--benchmark_v2_split", split])
    if gen:
        tasks = [("discrete", split)]
        if args.split == "test":
            tasks.append(("continuous", "seen_continuous"))
        for kind, selected_split in tasks:
            output = f"outputs_eval/benchmark_v2/{target}/{kind}"
            if args.command == "infer":
                commands.append([sys.executable, "eval_csgo.py", *common,
                                 "--output_dir", output, "--benchmark_v2_split", selected_split])
                continue
            for map_name in MAPS:
                cmd = [sys.executable, "benchmark_csgo_v1_conti.py" if kind == "continuous" else "benchmark_csgo_v1.py",
                       "--gt", ground_truth_dir(config, map_name), "--pred", f"{output}/gen_imgs/{map_name}",
                       "--batch_size", "1", "--device", "cuda", "--paired_size", "448",
                       "--data_dir", config["data_dir"], "--map_name", map_name,
                       "--metric_profile", "benchmark_v2_core",
                       "--benchmark_v2_manifest", config["benchmark_v2_manifest"],
                       "--benchmark_v2_split", selected_split]
                if config.get("benchmark_v2_asset_manifest"):
                    cmd.extend(["--benchmark_v2_asset_manifest", config["benchmark_v2_asset_manifest"]])
                if kind == "continuous":
                    cmd.extend(["--frame_diff_threshold", "2", "--min_track_len", "4",
                                "--clip_length", "16", "--clip_stride", "16", "--fvd_size", "224"])
                commands.append(cmd)
            commands.append([sys.executable, "scripts/aggregate_csgo_benchmark_v2_metrics.py", "maps",
                             "--manifest", config["benchmark_v2_manifest"], "--split", selected_split,
                             "--maps", *MAPS, "--input_root", output, "--kind", kind,
                             "--output", f"{output}/summary.json"])
    return commands, eval_config, config


def parse_checkpoint_step(value: str) -> int | str:
    if value == "final":
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint step must be an integer or final") from exc


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("validate", "train", "infer", "metrics"))
    result.add_argument("--experiment", choices=EXPERIMENTS)
    result.add_argument("--execute", action="store_true", help="Execute the displayed commands; default is print only")
    result.add_argument("--joint-base", choices=JOINT_BASES)
    result.add_argument("--stop-after-step", type=int, help="Absolute optimizer step; 0 removes the early stop")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--cuda-devices", default="0")
    result.add_argument("--micro-batch", type=int)
    result.add_argument("--master-port", type=int, default=29632)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--report-to", default="none")
    result.add_argument("--checkpoint-step", type=parse_checkpoint_step,
                        help="Saved step number, or final to verify and select a completed checkpoint")
    result.add_argument("--split", choices=("validation", "test"), default="validation")
    result.add_argument("--task", choices=("all", "localization", "generation"), default="all")
    result.add_argument("--asset-mode", choices=("minimal", "full"))
    result.add_argument("--asset-manifest")
    result.add_argument("--data-dir")
    result.add_argument("--model-path")
    result.add_argument("--pi05-path")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command in ("train", "validate") and args.checkpoint_step == "final":
            raise ValueError("--checkpoint-step final applies only to infer/metrics")
        if args.asset_mode == "full" and args.asset_manifest:
            raise ValueError("--asset-manifest cannot be combined with --asset-mode full")
        if args.command != "train" and (args.resume or args.stop_after_step is not None):
            raise ValueError("--resume and --stop-after-step apply only to train")
        if args.command == "validate":
            names = [args.experiment] if args.experiment else EXPERIMENTS
            for name in names:
                local_args = copy.copy(args)
                local_args.experiment = name
                config = resolve_config(local_args, for_training=True)
                validate_config(config)
            print(f"Validated {len(names)} static recipes; no training, inference or metrics launched.")
            return 0
        if args.experiment is None:
            raise ValueError("--experiment is required")
        config = resolve_config(args, for_training=args.command == "train")
        validate_config(config)
        if args.command == "train":
            commands, metadata, out = training_plan(args, config)
            if args.execute and is_stage2(args.experiment) and not args.resume:
                source = preflight_stage2_source(config)
                previous_plan = out / "experiment_plan.json"
                if previous_plan.is_file():
                    old_source = json.loads(previous_plan.read_text()).get("source_checkpoint")
                    if old_source != source:
                        raise ValueError("Stage 2 source changed since the previous plan")
                metadata["source_checkpoint"] = source
            writes = [(out / "resolved_config.yaml", yaml.safe_dump(config, sort_keys=False)),
                      (out / "experiment_plan.json", json.dumps(metadata, indent=2) + "\n")]
        else:
            commands, path, eval_config = evaluation_plan(args, config)
            if path.is_file() and read_yaml(path) != eval_config:
                raise ValueError(f"Existing evaluation uses different settings: {relative(path)}")
            writes = [(path, yaml.safe_dump(eval_config, sort_keys=False))]
        print("EXECUTE" if args.execute else "DRY RUN — no files written and no subprocesses launched")
        for path, _ in writes:
            print(f"Resolved configuration/artifact: {relative(path)}")
        for command in commands:
            print(f"CUDA_VISIBLE_DEVICES={shlex.quote(args.cuda_devices)} {shlex.join(command)}")
        if not commands:
            print("Localization metrics and summary are already produced by eval_csgo_loc.py during inference.")
        if args.execute:
            for path, content in writes:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.cuda_devices}
            for command in commands:
                subprocess.run(command, cwd=ROOT, env=env, check=True)
        return 0
    except (ValueError, OSError, KeyError, yaml.YAMLError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
