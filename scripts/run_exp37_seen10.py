#!/usr/bin/env python3
"""Plan or explicitly execute the five fixed exp37 Seen10 arms.

The default is read-only planning. --execute is required to write snapshots or
launch a training, inference, or metrics process. No queue is launched.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ("exp37", "exp37_s2_gen", "exp37_s2_loc", "exp37_s2", "exp37_s2_both")
SINGLES = ("exp37_s2_gen", "exp37_s2_loc")
JOINT_STAGE2 = ("exp37_s2", "exp37_s2_both")
MANIFEST = json.loads((ROOT / "data/csgo_benchmark_v2/benchmark_manifest.json").read_text())
MAPS = MANIFEST["protocol"]["seen_maps"]
EXPECTED_COUNTS = {"train": 50000, "validation": 5000, "discrete_test": 20000, "continuous_frames": 12800}
SOURCE = {
    "exp37_s2_gen": "outputs/csgo_1b/exp32_gen/checkpoint-19500/model.safetensors",
    "exp37_s2_loc": "outputs/csgo_1b/exp32_loc/checkpoint-19500/model.safetensors",
}
SOURCE_SHAPES = {
    "exp37_s2_gen": (2578, 876, ("language_model.model.", "model.llm_connector.", "model.dit."), ("model.action_dit.",)),
    "exp37_s2_loc": (2290, 588, ("language_model.model.", "model.action_dit."), ("model.llm_connector.", "model.dit.")),
    "exp37_s2": (2830, 1128, ("language_model.model.", "model.llm_connector.", "model.dit.", "model.action_dit."), ()),
    "exp37_s2_both": (2830, 1128, ("language_model.model.", "model.llm_connector.", "model.dit.", "model.action_dit."), ()),
}
LR_STAGE2 = {k: 2e-5 for k in ("learning_rate", "mm_projector_lr", "action_dit_lr", "llm_lora_lr", "llm_connector_lora_lr", "gen_dit_lora_lr", "action_dit_lora_lr", "action_io_mlp_lr")}
LR_STAGE2.update({k: 1e-4 for k in ("action_dit_projector_lr", "action_dit_connector_lr", "action_dit_norm_lr")})
RECIPE_OVERRIDES = {"model_name_or_path", "pi05_pytorch_weight_path", "data_dir", "benchmark_v2_asset_manifest", "seed", "training_stop_after_step", "finetune_init_ckpt_path"}


def path_in_repo(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return data


def recipe_path(experiment: str) -> Path:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    return ROOT / "csgo_configs" / f"{experiment}.yaml"


def output_dir(experiment: str) -> Path:
    return ROOT / "outputs/csgo_1b" / experiment


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def trainer_state(path: Path) -> dict:
    state = json.loads(path.read_text(encoding="utf-8"))
    step, horizon = state.get("global_step"), state.get("max_steps")
    if type(step) is not int or type(horizon) is not int or step <= 0 or horizon < step:
        raise ValueError(f"Invalid Trainer global_step/max_steps: {path}")
    return state


def final_checkpoint_step(experiment: str) -> int:
    """Only accept a saved full model checkpoint with completed Trainer state."""
    out = output_dir(experiment)
    if not (out / "resolved_config.yaml").is_file() or not (out / "experiment_plan.json").is_file():
        raise ValueError(f"Final evaluation requires a managed training recipe: {out}")
    root_state = out / "trainer_state.json"
    if root_state.is_file():
        state = trainer_state(root_state)
        if state["global_step"] != state["max_steps"]:
            raise ValueError(f"Training has not reached max_steps: {root_state}")
        step = state["global_step"]
    else:
        completed = []
        for checkpoint in out.glob("checkpoint-*"):
            if checkpoint.is_dir() and checkpoint.name[11:].isdigit() and (checkpoint / "trainer_state.json").is_file():
                state = trainer_state(checkpoint / "trainer_state.json")
                if state["global_step"] == state["max_steps"] == int(checkpoint.name[11:]):
                    completed.append(state["global_step"])
        if len(completed) != 1:
            raise ValueError(f"Expected one completed final checkpoint, found {len(completed)}: {out}")
        step = completed[0]
    ckpt = out / f"checkpoint-{step}"
    state = trainer_state(ckpt / "trainer_state.json")
    if state["global_step"] != step or state["max_steps"] != step or not (ckpt / "model.safetensors").is_file():
        raise ValueError(f"Final checkpoint is incomplete: {ckpt}")
    if experiment == "exp37":
        if state.get("num_train_epochs") != 50 or float(state.get("epoch", 0)) < 50:
            raise ValueError("Stage 1 final requires all 50 epochs")
    elif step != 6000:
        raise ValueError("Stage 2 final requires step 6000")
    return step


def resolve_config(args: argparse.Namespace, *, for_training: bool) -> dict:
    out = output_dir(args.experiment)
    saved = out / "resolved_config.yaml"
    if (args.resume or not for_training) and saved.is_file():
        config = read_yaml(saved)
    elif not for_training and args.execute:
        raise ValueError(f"Evaluation requires saved training recipe: {saved}")
    elif not for_training:
        config = read_yaml(recipe_path(args.experiment))
    elif args.resume:
        raise ValueError(f"Resume requires saved training recipe: {saved}")
    else:
        config = read_yaml(recipe_path(args.experiment))
    if config.get("exp37_experiment") != args.experiment:
        raise ValueError("Saved/source recipe belongs to another exp37 arm")
    original = copy.deepcopy(config)
    if args.asset_mode == "full":
        config["benchmark_v2_asset_manifest"] = None
    elif args.asset_mode == "minimal":
        config["benchmark_v2_asset_manifest"] = "data/csgo_benchmark_v2/minimal_dataset_report.json"
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.model_path:
        config["model_name_or_path"] = args.model_path
    else:
        config.setdefault("model_name_or_path", "UniLIP-1B")
    if args.pi05_path:
        config["pi05_pytorch_weight_path"] = str(Path(args.pi05_path).expanduser())
    if args.resume or (not for_training and saved.is_file()):
        for key in ("benchmark_v2_asset_manifest", "data_dir", "model_name_or_path", "pi05_pytorch_weight_path"):
            if config.get(key) != original.get(key, "UniLIP-1B" if key == "model_name_or_path" else None):
                raise ValueError(f"Saved training recipe differs in {key}")
        config = original
    if for_training:
        if args.resume and args.seed != config.get("seed"):
            raise ValueError("Resume seed differs from saved training recipe")
        config["seed"] = args.seed
        if args.stop_after_step is not None:
            if args.stop_after_step < 0:
                raise ValueError("--stop-after-step must be nonnegative")
            if args.stop_after_step == 0:
                config.pop("training_stop_after_step", None)
            else:
                config["training_stop_after_step"] = args.stop_after_step
        if args.experiment in JOINT_STAGE2 and not args.resume:
            try:
                step = final_checkpoint_step("exp37")
                config["finetune_init_ckpt_path"] = f"outputs/csgo_1b/exp37/checkpoint-{step}/model.safetensors"
            except (ValueError, OSError):
                # A dry plan may precede Stage 1. Execute requires preflight.
                pass
    return config


def validate_config(config: dict) -> None:
    experiment = config.get("exp37_experiment")
    expected = read_yaml(recipe_path(experiment))
    for key in set(config) | set(expected):
        if key not in RECIPE_OVERRIDES and config.get(key, object()) != expected.get(key, object()):
            raise ValueError(f"{experiment} recipe differs in fixed field {key}")
    if config.get("benchmark_v2_manifest") != "data/csgo_benchmark_v2/benchmark_manifest.json":
        raise ValueError("Benchmark manifest changed")
    observed_counts = {key: sum(MANIFEST["counts"]["seen"][name][key] for name in MAPS)
                       for key in EXPECTED_COUNTS}
    if len(MAPS) != 10 or observed_counts != EXPECTED_COUNTS:
        raise ValueError(f"Seen10 manifest counts changed: {observed_counts}")
    for key in ("train_maps", "val_maps", "test_maps"):
        if config.get(key) != MAPS:
            raise ValueError(f"{key} must contain all ten ordered Seen maps")
    if config.get("benchmark_v2_split") != "seen_train" or config.get("benchmark_v2_allow_map_subset_summary"):
        raise ValueError("Seen10 requires full-map seen_train without subset summaries")
    if config.get("csgo_loss_per_microbatch_mean") is not True:
        raise ValueError("All arms require microbatch-mean GAS normalization")
    if config.get("is_lora") is not True or config.get("llm_train_mode") != "lora":
        raise ValueError("LoRA architecture changed")
    if experiment == "exp37":
        if any(config.get(k) for k in ("finetune_init_ckpt_path", "resume_ckpt_path", "base_init_ckpt_path", "gen_init_ckpt_path", "loc_init_ckpt_path")):
            raise ValueError("Stage 1 must start from original model and pi05")
        stop = config.get("training_stop_after_step")
        if stop is not None and (type(stop) is not int or stop <= 0):
            raise ValueError("Invalid Stage 1 stop")
    else:
        if config.get("finetune_init_strict") is not True or any(config.get(k) for k in ("resume_ckpt_path", "base_init_ckpt_path", "gen_init_ckpt_path", "loc_init_ckpt_path")):
            raise ValueError("Stage 2 requires strict model-only initialization")
        source = config.get("finetune_init_ckpt_path", "")
        if experiment in SINGLES and source != SOURCE[experiment]:
            raise ValueError("Single-task Stage 2 source changed")
        if experiment in JOINT_STAGE2 and not (source == "outputs/csgo_1b/exp37/checkpoint-FINAL/model.safetensors" or
                source.startswith("outputs/csgo_1b/exp37/checkpoint-") and source.endswith("/model.safetensors") and source.removeprefix("outputs/csgo_1b/exp37/checkpoint-").removesuffix("/model.safetensors").isdigit()):
            raise ValueError("Joint Stage 2 must use exp37 final")
        stop = config.get("training_stop_after_step")
        if stop is not None and (type(stop) is not int or not 0 < stop <= 6000):
            raise ValueError("Stage 2 stop exceeds 6000-step horizon")
        for key, value in LR_STAGE2.items():
            if config.get(key) != value:
                raise ValueError(f"Stage 2 {key} changed")
    for stem in ("alpha_loc", "alpha_loc_aux", "alpha_loc_perception"):
        steps, values = config.get(stem + "_schedule_steps"), config.get(stem + "_schedule_values")
        if (steps is None) != (values is None) or steps is not None and (len(steps) != len(values) or steps != sorted(set(steps))):
            raise ValueError(f"Invalid {stem} schedule")


def fingerprint(config: dict, runtime: dict) -> str:
    fixed = copy.deepcopy(config)
    fixed.pop("training_stop_after_step", None)
    return hashlib.sha256(json.dumps({"config": fixed, "runtime": runtime}, sort_keys=True).encode()).hexdigest()


def latest_checkpoint(out: Path) -> Path | None:
    choices = [(int(p.name[11:]), p) for p in out.glob("checkpoint-*") if p.is_dir() and p.name[11:].isdigit()]
    return max(choices)[1] if choices else None


def check_resume(out: Path, identity: str, resume: bool) -> tuple[Path | None, dict | None]:
    plan_path = out / "experiment_plan.json"
    checkpoint = latest_checkpoint(out)
    if not resume:
        if checkpoint or (out / "model.safetensors").exists():
            raise ValueError(f"Existing run at {out}; use --resume")
        if plan_path.is_file() and json.loads(plan_path.read_text()).get("fingerprint") != identity:
            raise ValueError("Previous plan has another recipe")
        return None, None
    if checkpoint is None or not plan_path.is_file():
        raise ValueError("--resume requires a managed full checkpoint")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("fingerprint") != identity:
        raise ValueError("Resume recipe or runtime changed")
    for filename in ("trainer_state.json", "training_args.bin", "model.safetensors"):
        if not (checkpoint / filename).is_file():
            raise ValueError(f"Incomplete resume checkpoint: missing {filename}")
    state = trainer_state(checkpoint / "trainer_state.json")
    if state["global_step"] != int(checkpoint.name[11:]):
        raise ValueError("Resume checkpoint directory and Trainer step differ")
    world_size = plan.get("runtime", {}).get("world_size")
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("Resume plan has invalid world size")
    rng_names = (["rng_state.pth"] if world_size == 1 else
                 [f"rng_state_{rank}.pth" for rank in range(world_size)])
    if any(not (checkpoint / name).is_file() for name in rng_names):
        raise ValueError(f"Resume checkpoint lacks rank RNG state: {rng_names}")
    if not ((checkpoint / "optimizer.pt").is_file() and (checkpoint / "scheduler.pt").is_file()) and not list(checkpoint.glob("global_step*/*optim_states.pt")):
        raise ValueError("Resume checkpoint lacks optimizer/scheduler state")
    if plan.get("experiment") != out.name:
        raise ValueError("Resume plan experiment identity changed")
    if out.name != "exp37":
        source = plan.get("source_checkpoint")
        if not isinstance(source, dict) or not isinstance(source.get("sha256"), str) or len(source["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in source["sha256"]) or source.get("path") != read_yaml(out / "resolved_config.yaml").get("finetune_init_ckpt_path"):
            raise ValueError("Resume plan lacks pinned source provenance")
    return checkpoint, plan


def training_plan(args: argparse.Namespace, config: dict) -> tuple[list[list[str]], dict, Path]:
    devices = args.cuda_devices.split(",")
    if not all(d.strip() for d in devices) or len(set(devices)) != len(devices):
        raise ValueError("--cuda-devices must list distinct devices")
    micro = args.micro_batch if args.micro_batch is not None else 4
    if micro <= 0 or 128 % (len(devices) * micro):
        raise ValueError("GPU count × micro batch must divide source batch 128")
    gas = 128 // (len(devices) * micro)
    stage2 = args.experiment != "exp37"
    runtime = dict(epochs=50, source_batch=128, world_size=len(devices), micro_batch=micro,
                   gradient_accumulation_steps=gas, seed=args.seed, save_steps=2000 if stage2 else 4000,
                   save_total_limit=6, report_to=args.report_to, max_steps=6000 if stage2 else None,
                   warmup_steps=200 if stage2 else None, warmup_ratio=config["warmup_ratio"],
                   optimizer="adamw_torch", adam_beta1=0.9, adam_beta2=0.999,
                   adam_epsilon=1e-8, weight_decay=0.0, max_grad_norm=1.0)
    out = output_dir(args.experiment)
    identity = fingerprint(config, runtime)
    checkpoint, previous = check_resume(out, identity, args.resume)
    if checkpoint:
        state = trainer_state(checkpoint / "trainer_state.json")
        if stage2 and state["max_steps"] != 6000:
            raise ValueError("Resume checkpoint has wrong Stage 2 horizon")
        stop = config.get("training_stop_after_step")
        if stop is not None and state["global_step"] >= stop:
            raise ValueError("Checkpoint has reached requested stop")
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={len(devices)}",
           f"--master_port={args.master_port}", "train_csgo.py", "--csgo_config", relative(out / "resolved_config.yaml"),
           "--deepspeed", "deepspeed_scripts/zero0.json", "--model_name_or_path", config["model_name_or_path"],
           "--unilip_factor", "10.6", "--mllm_hf_path", config["mllm_hf_path"], "--version", "internvl",
           "--data_type", "mix", "--csgo_image_folder", config["data_dir"], "--output_dir", relative(out),
           "--num_train_epochs", "50", "--per_device_train_batch_size", str(micro),
           "--per_device_eval_batch_size", "1", "--gradient_accumulation_steps", str(gas),
           "--eval_strategy", "no", "--save_strategy", "steps", "--save_steps", str(runtime["save_steps"]),
           "--save_total_limit", "6", "--save_only_model", "False", "--seed", str(args.seed),
           "--learning_rate", str(config["learning_rate"]), "--weight_decay", "0", "--warmup_ratio", str(config["warmup_ratio"]),
           "--lr_scheduler_type", config["lr_scheduler_type"], "--model_max_length", "1024", "--logging_steps", "10",
           "--dataloader_num_workers", "4", "--n_query", "256", "--n_und_query", "0", "--report_to", args.report_to,
           "--lora_r", str(config["lora_r"]), "--lora_alpha", str(config["lora_alpha"]),
           "--optim", "adamw_torch", "--adam_beta1", "0.9", "--adam_beta2", "0.999", "--adam_epsilon", "1e-8", "--max_grad_norm", "1.0"]
    for key, value in dict(mm_use_im_start_end=False, mm_use_im_patch_token=False, bf16=True, tf32=True,
                           gradient_checkpointing=True, lazy_preprocess=True, fix_vit=config["fix_vit"],
                           fix_llm=config["fix_llm"], fix_connect=config["fix_connect"], fix_dit=config["fix_dit"]).items():
        cmd.extend(["--" + key, str(value)])
    if stage2:
        cmd.extend(["--max_steps", "6000", "--warmup_steps", "200"])
    metadata = dict(experiment=args.experiment, fingerprint=identity, runtime=runtime, command=cmd,
                    resume_checkpoint=relative(checkpoint) if checkpoint else None,
                    source_checkpoint=previous.get("source_checkpoint") if previous else None,
                    evaluation_steps={"validation": [4000, 6000] if stage2 else [8000, "final"],
                                      "test": [6000] if stage2 else ["final"]},
                    dataset_counts=dict(EXPECTED_COUNTS))
    return [cmd], metadata, out


class _ArgumentUnpickler(pickle.Unpickler):
    """Read trusted local legacy training_args without changing __main__ globals."""

    def find_class(self, module: str, name: str):
        if module == "__main__" and name == "TrainingArguments":
            return SimpleNamespace
        return super().find_class(module, name)


def legacy_training_args(path: Path) -> dict:
    try:
        import torch
        module = SimpleNamespace(__name__="exp37_legacy_pickle", Unpickler=_ArgumentUnpickler,
                                 Pickler=pickle.Pickler, load=pickle.load, dump=pickle.dump)
        args = torch.load(path, map_location="cpu", weights_only=False, pickle_module=module)
    except Exception as exc:
        raise ValueError(f"Could not decode trusted local legacy training_args: {path}: {exc}") from exc
    keys = ("num_train_epochs", "max_steps", "per_device_train_batch_size", "gradient_accumulation_steps",
            "save_only_model", "learning_rate", "warmup_ratio", "seed", "lora_r", "lora_alpha", "output_dir")
    return {key: getattr(args, key, None) for key in keys}


def tensor_header(path: Path, experiment: str) -> dict:
    from safetensors import safe_open
    total, lora_count, required, forbidden = SOURCE_SHAPES[experiment]
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        count = sum("lora_" in key for key in keys)
        if (len(keys), count) != (total, lora_count):
            raise ValueError(f"Source tensor/LoRA count differs from strict architecture: {len(keys)}/{count}")
        if any(not any(k.startswith(prefix) and "lora_" in k for k in keys) for prefix in required):
            raise ValueError("Source lacks required LoRA modules")
        if any(k.startswith(prefix) and "lora_" in k for k in keys for prefix in forbidden):
            raise ValueError("Source has inactive-head LoRA tensors")
        for key in keys:
            rank = 16 if key.startswith("model.llm_connector.") else 32
            if ".lora_A." in key and handle.get_slice(key).get_shape()[0] != rank:
                raise ValueError(f"Source LoRA A rank differs from {rank}: {key}")
            if ".lora_B." in key and handle.get_slice(key).get_shape()[-1] != rank:
                raise ValueError(f"Source LoRA B rank differs from {rank}: {key}")
        for key in ("language_model.lm_head.weight", "model.action_dit_norm.weight", "model.action_dit_connector.0.bias", "model.action_dit_projector.0.bias", "multi_modal_projector.0.bias"):
            if key not in keys:
                raise ValueError(f"Source model lacks {key}")
        if any(not handle.get_slice(k).get_shape() for k in keys):
            raise ValueError("Source contains scalar/empty tensor header")
    return {"tensor_count": len(keys), "lora_tensor_count": count,
            "lora_ranks": {"llm_connector": 16, "other_active_adapters": 32},
            "shape_validation": "header counts/modules/LoRA rank; model loader enforces every key and shape with finetune_init_strict"}


def preflight_stage2_source(config: dict) -> dict:
    """Verify completion, provenance, header, and hash before writing a new arm."""
    experiment = config["exp37_experiment"]
    if experiment in JOINT_STAGE2:
        step = final_checkpoint_step("exp37")
        source_name = f"outputs/csgo_1b/exp37/checkpoint-{step}/model.safetensors"
        if config.get("finetune_init_ckpt_path") != source_name:
            raise ValueError("Joint Stage 2 recipe does not name completed exp37 final")
        base_recipe = read_yaml(output_dir("exp37") / "resolved_config.yaml")
        validate_config(base_recipe)
        if base_recipe.get("exp37_experiment") != "exp37":
            raise ValueError("Joint Stage 2 source recipe identity changed")
        for key in ("train_maps", "val_maps", "test_maps", "benchmark_v2_split", "benchmark_v2_asset_manifest",
                    "data_dir", "model_name_or_path", "pi05_pytorch_weight_path", "lora_r", "lora_alpha",
                    "enable_gen_head_lora", "enable_loc_head_lora", "is_multi_task_balanced", "alpha_loc_loss"):
            if base_recipe.get(key) != config.get(key):
                raise ValueError(f"Joint source and Stage 2 differ in {key}")
        plan = json.loads((output_dir("exp37") / "experiment_plan.json").read_text())
        if plan.get("fingerprint") != fingerprint(base_recipe, plan.get("runtime", {})):
            raise ValueError("Stage 1 source plan fingerprint changed")
        runtime = plan.get("runtime", {})
        if plan.get("experiment") != "exp37" or runtime.get("epochs") != 50 or runtime.get("source_batch") != 128 or runtime.get("save_steps") != 4000 or runtime.get("warmup_ratio") != 0.003:
            raise ValueError("Stage 1 source runtime differs from approved plan")
        evidence = {"source_recipe": relative(output_dir("exp37") / "resolved_config.yaml"),
                    "source_plan": relative(output_dir("exp37") / "experiment_plan.json"),
                    "provenance": "managed exp37 snapshot and completed Trainer state"}
    else:
        source_name = SOURCE[experiment]
        if config.get("finetune_init_ckpt_path") != source_name:
            raise ValueError("Legacy source recipe changed")
        legacy = "exp32_gen" if experiment.endswith("gen") else "exp32_loc"
        source_yaml = ROOT / "csgo_configs" / f"{legacy}.yaml"
        source_recipe = read_yaml(source_yaml)
        for key in ("train_maps", "val_maps", "test_maps", "benchmark_v2_split", "is_multi_task", "is_multi_task_balanced",
                    "task_mix_ratio", "alpha_loc_loss", "is_lora", "llm_train_mode", "lora_r", "lora_alpha", "lora_dropout",
                    "enable_language_model_lora", "enable_gen_head_lora", "enable_loc_head_lora", "freeze_inactive_head",
                    "fix_vit", "fix_llm", "fix_connect", "fix_dit", "is_action_dit_projector", "use_pi05_action_dit"):
            if source_recipe.get(key) != config.get(key):
                raise ValueError(f"Legacy source YAML differs in {key}")
        evidence = {"source_recipe": relative(source_yaml), "source_recipe_sha256": digest_file(source_yaml),
                    "provenance": "legacy repository YAML is intended recipe; checkpoint config/state/header and logs are observed evidence"}
    source = path_in_repo(source_name)
    if not source.is_file():
        raise ValueError(f"Missing Stage 2 source model: {source}")
    checkpoint = source.parent
    state = trainer_state(checkpoint / "trainer_state.json")
    step = int(checkpoint.name[11:])
    if state["global_step"] != state["max_steps"] or state["global_step"] != step:
        raise ValueError("Stage 2 source is not a completed checkpoint")
    if experiment in SINGLES and (step != 19500 or state.get("num_train_epochs") != 50 or float(state.get("epoch", 0)) < 50):
        raise ValueError("Legacy source did not complete 50 epochs at step 19500")
    header = tensor_header(source, experiment)
    if experiment in SINGLES:
        cfg_path = checkpoint / "config.json"
        cfg = json.loads(cfg_path.read_text())
        for key in ("enable_language_model_lora", "enable_gen_head_lora", "enable_loc_head_lora", "freeze_inactive_head", "is_lora", "llm_train_mode", "alpha_loc_loss", "is_action_dit_projector", "use_pi05_action_dit"):
            if cfg.get(key) != config.get(key):
                raise ValueError(f"Legacy checkpoint model config differs in {key}")
        args_path = checkpoint / "training_args.bin"
        args = legacy_training_args(args_path)
        if (float(args["num_train_epochs"]) != 50 or args["max_steps"] != -1 or
            int(args["per_device_train_batch_size"]) * int(args["gradient_accumulation_steps"]) != 128 or
            args["lora_r"] != 32 or args["lora_alpha"] != 64 or args["save_only_model"] is not False):
            raise ValueError("Legacy training_args contradict completed Seen10 source")
        last_log = state.get("log_history", [])[-1] if state.get("log_history") else {}
        counts = last_log.get("other_info", {})
        if last_log.get("step") != 19500 or counts.get("gen_indices") != (128 if experiment.endswith("gen") else 0) or counts.get("loc_indices") != (0 if experiment.endswith("gen") else 128):
            raise ValueError("Legacy Trainer log does not confirm single-task head")
        evidence.update(source_config=relative(cfg_path), source_training_args=relative(args_path),
                        observed_training_args=args, observed_last_log={"step": last_log["step"], "gen_indices": counts["gen_indices"], "loc_indices": counts["loc_indices"]},
                        adapter_metadata="No checkpoint adapter_config.json; LoRA rank verified by training_args and tensor shapes")
        adapters = []
        for adapter_path in (checkpoint.parent / "lora_adapters").glob("*/adapter_config.json"):
            adapter = json.loads(adapter_path.read_text())
            adapter_name = adapter_path.parent.name
            rank = 16 if adapter_name == "llm_connector" else 32
            if adapter.get("r") != rank or adapter.get("lora_alpha") != 64 or adapter.get("lora_dropout") != 0.05:
                raise ValueError(f"Legacy adapter metadata differs: {adapter_path}")
            adapters.append({"path": relative(adapter_path), "sha256": digest_file(adapter_path),
                             "r": rank, "lora_alpha": 64, "lora_dropout": 0.05})
        if adapters:
            evidence["optional_exported_adapter_configs"] = adapters
    result = {"path": source_name, "sha256": digest_file(source), "size_bytes": source.stat().st_size,
              "step": step, "state_path": relative(checkpoint / "trainer_state.json"), **header, **evidence}
    if experiment in JOINT_STAGE2:
        sibling = "exp37_s2_both" if experiment == "exp37_s2" else "exp37_s2"
        sibling_plan = output_dir(sibling) / "experiment_plan.json"
        if sibling_plan.is_file():
            peer = json.loads(sibling_plan.read_text()).get("source_checkpoint")
            if not isinstance(peer, dict) or (peer.get("path"), peer.get("sha256")) != (result["path"], result["sha256"]):
                raise ValueError("Joint Stage 2 sibling uses a different Stage 1 source hash")
    return result


def ground_truth_dir(config: dict, map_name: str) -> str:
    report_name = config.get("benchmark_v2_asset_manifest")
    if not report_name:
        return str(Path(config["data_dir"]) / map_name / "imgs")
    report_path = path_in_repo(report_name)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "verified" or report.get("images", {}).get("status") != "verified":
        raise ValueError(f"Minimal asset report is not verified: {report_path}")
    target = Path(report["images"]["target_template"].format(map=map_name, file_frame="__frame__"))
    if target.is_absolute() or ".." in target.parts:
        raise ValueError("Minimal asset target escapes bundle")
    return relative(report_path.parent / target.parent)


def evaluation_plan(args: argparse.Namespace, config: dict) -> tuple[list[list[str]], Path, dict]:
    step = final_checkpoint_step(args.experiment) if args.checkpoint_step == "final" else args.checkpoint_step
    if type(step) is not int or step <= 0:
        raise ValueError("infer/metrics require positive --checkpoint-step or final")
    stage2 = args.experiment != "exp37"
    if stage2:
        allowed = (4000, 6000) if args.split == "validation" else (6000,)
    elif args.split == "validation" and step == 8000:
        allowed = (8000,)
    else:
        allowed = (final_checkpoint_step("exp37"),)
    if step not in allowed:
        raise ValueError(f"{args.experiment} {args.split} evaluation requires checkpoint in {allowed}")
    if "," in args.cuda_devices:
        raise ValueError("Evaluation uses one device")
    checkpoint = Path("outputs/csgo_1b") / args.experiment / f"checkpoint-{step}" / "model.safetensors"
    if args.execute:
        out = output_dir(args.experiment)
        plan_path = out / "experiment_plan.json"
        if not plan_path.is_file():
            raise ValueError(f"Evaluation requires managed training plan: {plan_path}")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("experiment") != args.experiment or plan.get("fingerprint") != fingerprint(config, plan.get("runtime", {})):
            raise ValueError("Evaluation plan identity/fingerprint differs from saved recipe")
        if not path_in_repo(checkpoint).is_file():
            raise ValueError(f"Checkpoint is absent: {checkpoint}")
        state = trainer_state(path_in_repo(checkpoint).parent / "trainer_state.json")
        if state["global_step"] != step or (stage2 and state["max_steps"] != 6000):
            raise ValueError("Selected evaluation checkpoint has wrong Trainer step/horizon")
        if args.split == "test" or stage2 and step == 6000:
            if final_checkpoint_step(args.experiment) != step:
                raise ValueError("Test/final evaluation requires completed checkpoint")
    target = f"{args.experiment}/checkpoint-{step}/{args.split}/seed_{args.seed}"
    eval_config = ROOT / "outputs_eval/benchmark_v2" / target / "evaluation_config.yaml"
    split = "seen_validation" if args.split == "validation" else "seen_discrete_test"
    config = copy.deepcopy(config)
    config["ckpt_path"] = str(checkpoint)
    config["benchmark_v2_split"] = split
    config["seed"] = args.seed
    config.pop("training_stop_after_step", None)
    for key in ("is_loc_aux_loss", "is_aux_loc_combined_em_unc_loss", "is_loc_perception_loss"):
        config[key] = False
    loc = args.experiment != "exp37_s2_gen"
    gen = args.experiment != "exp37_s2_loc"
    if args.task == "localization" and not loc or args.task == "generation" and not gen:
        raise ValueError("Requested task is inactive in this checkpoint")
    loc = loc and args.task in ("all", "localization")
    gen = gen and args.task in ("all", "generation")
    common = ["--csgo_config", relative(eval_config), "--ckpt_path", str(checkpoint),
              "--seed", str(args.seed), "--benchmark_v2_maps", *MAPS]
    commands = []
    if args.command == "infer" and loc:
        commands.append([sys.executable, "eval_csgo_loc.py", *common,
                         "--output_dir", f"outputs_loc/benchmark_v2/{target}", "--benchmark_v2_split", split])
    if gen:
        tasks = [("discrete", split)] + ([("continuous", "seen_continuous")] if args.split == "test" else [])
        for kind, selected_split in tasks:
            output = f"outputs_eval/benchmark_v2/{target}/{kind}"
            if args.command == "infer":
                commands.append([sys.executable, "eval_csgo.py", *common, "--output_dir", output,
                                 "--benchmark_v2_split", selected_split])
                continue
            for map_name in MAPS:
                cmd = [sys.executable, "benchmark_csgo_v1_conti.py" if kind == "continuous" else "benchmark_csgo_v1.py",
                       "--gt", ground_truth_dir(config, map_name), "--pred", f"{output}/gen_imgs/{map_name}",
                       "--batch_size", "1", "--device", "cuda", "--paired_size", "448", "--data_dir", config["data_dir"],
                       "--map_name", map_name, "--metric_profile", "benchmark_v2_core", "--benchmark_v2_manifest",
                       config["benchmark_v2_manifest"], "--benchmark_v2_split", selected_split]
                if config.get("benchmark_v2_asset_manifest"):
                    cmd.extend(["--benchmark_v2_asset_manifest", config["benchmark_v2_asset_manifest"]])
                if kind == "continuous":
                    cmd.extend(["--frame_diff_threshold", "2", "--min_track_len", "4", "--clip_length", "16",
                                "--clip_stride", "16", "--fvd_size", "224"])
                commands.append(cmd)
            commands.append([sys.executable, "scripts/aggregate_csgo_benchmark_v2_metrics.py", "maps",
                             "--manifest", config["benchmark_v2_manifest"], "--split", selected_split,
                             "--maps", *MAPS, "--input_root", output, "--kind", kind, "--output", f"{output}/summary.json"])
    return commands, eval_config, config


def parse_checkpoint_step(value: str) -> int | str:
    if value == "final":
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint step must be an integer or final") from exc


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("validate", "train", "infer", "metrics"))
    p.add_argument("--experiment", choices=EXPERIMENTS)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-step", type=int, help="Absolute optimizer step; 0 removes the stop")
    p.add_argument("--cuda-devices", default="0")
    p.add_argument("--micro-batch", type=int)
    p.add_argument("--master-port", type=int, default=29637)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-to", default="none")
    p.add_argument("--checkpoint-step", type=parse_checkpoint_step)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--task", choices=("all", "localization", "generation"), default="all")
    p.add_argument("--asset-mode", choices=("minimal", "full"))
    p.add_argument("--data-dir")
    p.add_argument("--model-path")
    p.add_argument("--pi05-path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command != "train" and (args.resume or args.stop_after_step is not None):
            raise ValueError("--resume and --stop-after-step apply only to train")
        if args.command in ("train", "validate") and args.checkpoint_step is not None:
            raise ValueError("--checkpoint-step applies only to infer/metrics")
        if args.asset_mode == "full" and args.command in ("train", "validate") and not args.resume and not args.data_dir:
            raise ValueError("--asset-mode full requires --data-dir for a fresh recipe")
        if args.command == "validate":
            for name in ([args.experiment] if args.experiment else EXPERIMENTS):
                local = copy.copy(args)
                local.experiment = name
                config = resolve_config(local, for_training=True)
                validate_config(config)
            print("Validated fixed exp37 Seen10 recipes; no files written or jobs launched.")
            return 0
        if args.experiment is None:
            raise ValueError("--experiment is required")
        config = resolve_config(args, for_training=args.command == "train")
        validate_config(config)
        if args.command == "train":
            commands, metadata, out = training_plan(args, config)
            if args.execute and args.experiment != "exp37" and not args.resume:
                if args.experiment in JOINT_STAGE2:
                    step = final_checkpoint_step("exp37")
                    config["finetune_init_ckpt_path"] = f"outputs/csgo_1b/exp37/checkpoint-{step}/model.safetensors"
                    validate_config(config)
                    commands, metadata, out = training_plan(args, config)
                metadata["source_checkpoint"] = preflight_stage2_source(config)
                previous_plan = out / "experiment_plan.json"
                if previous_plan.is_file():
                    prior_source = json.loads(previous_plan.read_text()).get("source_checkpoint")
                    if not isinstance(prior_source, dict) or (prior_source.get("path"), prior_source.get("sha256")) != (metadata["source_checkpoint"]["path"], metadata["source_checkpoint"]["sha256"]):
                        raise ValueError("Stage 2 source changed since earlier plan")
            writes = [(out / "resolved_config.yaml", yaml.safe_dump(config, sort_keys=False)),
                      (out / "experiment_plan.json", json.dumps(metadata, indent=2) + "\n")]
        else:
            commands, path, eval_config = evaluation_plan(args, config)
            if path.is_file() and read_yaml(path) != eval_config:
                raise ValueError(f"Existing evaluation settings differ: {relative(path)}")
            writes = [(path, yaml.safe_dump(eval_config, sort_keys=False))]
        print("EXECUTE" if args.execute else "DRY RUN — no files written and no subprocesses launched")
        for path, _ in writes:
            print(f"Resolved configuration/artifact: {relative(path)}")
        for command in commands:
            print(f"CUDA_VISIBLE_DEVICES={shlex.quote(args.cuda_devices)} {shlex.join(command)}")
        if not commands:
            print("Localization metrics are produced during localization inference.")
        if args.execute:
            for path, content in writes:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.cuda_devices}
            for command in commands:
                subprocess.run(command, cwd=ROOT, env=env, check=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
