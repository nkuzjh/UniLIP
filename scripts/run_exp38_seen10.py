#!/usr/bin/env python3
"""Plan or explicitly execute the two fixed exp38 Seen10 arms.

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
import shlex
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from csgo_datasets.inference_batching import (  # noqa: E402
    DEFAULT_TEST_GEN_BATCH_SIZE, DEFAULT_TEST_LOC_BATCH_SIZE,
    positive_batch_size, validate_execution_metadata,
)
EXPERIMENTS = ("exp38_s2_joint", "exp38_s2_both")
SOURCE = "outputs/csgo_1b/exp37/checkpoint-12000/model.safetensors"
SOURCE_STEP = 12000
SOURCE_HORIZON = 19550
MAX_STEPS = 7550
CHECKPOINT_STEPS = (4000, MAX_STEPS)
MANIFEST = json.loads((ROOT / "data/csgo_benchmark_v2/benchmark_manifest.json").read_text())
MAPS = MANIFEST["protocol"]["seen_maps"]
EXPECTED_COUNTS = {"train": 50000, "validation": 5000, "discrete_test": 20000, "continuous_frames": 12800}
SOURCE_SHAPE = (2830, 1128, ("language_model.model.", "model.llm_connector.", "model.dit.", "model.action_dit."))
LR_STAGE2 = {k: 2e-5 for k in ("learning_rate", "mm_projector_lr", "action_dit_lr", "llm_lora_lr", "llm_connector_lora_lr", "gen_dit_lora_lr", "action_dit_lora_lr", "action_io_mlp_lr")}
LR_STAGE2.update({k: 1e-4 for k in ("action_dit_projector_lr", "action_dit_connector_lr", "action_dit_norm_lr")})
RECIPE_OVERRIDES = {"model_name_or_path", "pi05_pytorch_weight_path", "data_dir", "benchmark_v2_asset_manifest", "seed", "training_stop_after_step"}


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
    """Require the managed local 7550 checkpoint to be complete."""
    out = output_dir(experiment)
    if not (out / "resolved_config.yaml").is_file() or not (out / "experiment_plan.json").is_file():
        raise ValueError(f"Final evaluation requires a managed training recipe: {out}")
    ckpt = out / f"checkpoint-{MAX_STEPS}"
    state = trainer_state(ckpt / "trainer_state.json")
    if state["global_step"] != MAX_STEPS or state["max_steps"] != MAX_STEPS or not (ckpt / "model.safetensors").is_file():
        raise ValueError(f"Final checkpoint is incomplete: {ckpt}")
    return MAX_STEPS


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
    if config.get("exp38_experiment") != args.experiment:
        raise ValueError("Saved/source recipe belongs to another exp38 arm")
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
        if not args.resume and args.stop_after_step is None:
            config["training_stop_after_step"] = CHECKPOINT_STEPS[0]
        if args.stop_after_step is not None:
            if args.stop_after_step not in (0, *CHECKPOINT_STEPS):
                raise ValueError("--stop-after-step must be 0, 4000, or 7550")
            if args.stop_after_step == 0:
                config.pop("training_stop_after_step", None)
            else:
                config["training_stop_after_step"] = args.stop_after_step
    return config


def validate_config(config: dict) -> None:
    experiment = config.get("exp38_experiment")
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
        raise ValueError("Microbatch-mean GAS normalization required")
    if config.get("is_lora") is not True or config.get("llm_train_mode") != "lora":
        raise ValueError("LoRA architecture changed")
    if config.get("finetune_init_ckpt_path") != SOURCE or config.get("finetune_init_strict") is not True:
        raise ValueError("exp38 requires strict model-only initialization from exp37@12000")
    if any(config.get(k) for k in ("resume_ckpt_path", "base_init_ckpt_path", "gen_init_ckpt_path", "loc_init_ckpt_path")):
        raise ValueError("exp38 may not initialize from another checkpoint")
    if config.get("training_checkpoint_steps") != list(CHECKPOINT_STEPS):
        raise ValueError("Only checkpoints 4000 and 7550 may be saved")
    if config.get("save_final_model_at_output_root") is not False:
        raise ValueError("Root model export must be disabled")
    stop = config.get("training_stop_after_step")
    if stop is not None and (type(stop) is not int or stop not in CHECKPOINT_STEPS):
        raise ValueError("Invalid exp38 stop step")
    for key, value in LR_STAGE2.items():
        if config.get(key) != value:
            raise ValueError(f"Stage 2 {key} changed")
    for stem in ("alpha_loc", "alpha_loc_aux", "alpha_loc_perception"):
        steps, values = config.get(stem + "_schedule_steps"), config.get(stem + "_schedule_values")
        if (steps is None) != (values is None) or steps is not None and (len(steps) != len(values) or steps != sorted(set(steps))):
            raise ValueError(f"Invalid {stem} schedule")
    if experiment == "exp38_s2_joint":
        if any(config.get(k) for k in ("is_loc_aux_loss", "is_aux_loc_combined_em_unc_loss", "is_loc_perception_loss")):
            raise ValueError("Joint control must have no auxiliary losses")
    elif (config.get("is_loc_aux_loss") is not True or config.get("is_aux_loc_combined_em_unc_loss") is not True
          or config.get("is_loc_perception_loss") is not True
          or config.get("loc_perception_use_attention_weight") is not False
          or config.get("alpha_loc_aux_schedule_steps") != [0, 503]
          or config.get("alpha_loc_perception_schedule_steps") != [0, 503]):
        raise ValueError("Both arm requires P4+U2 ramp through local step 503")


def fingerprint(config: dict, runtime: dict) -> str:
    fixed = copy.deepcopy(config)
    fixed.pop("training_stop_after_step", None)
    return hashlib.sha256(json.dumps({"config": fixed, "runtime": runtime}, sort_keys=True).encode()).hexdigest()


def latest_checkpoint(out: Path) -> Path | None:
    choices = [(int(p.name[11:]), p) for p in out.glob("checkpoint-*") if p.is_dir() and p.name[11:].isdigit()]
    return max(choices)[1] if choices else None


def check_resume(out: Path, identity: str, resume: bool) -> tuple[Path | None, dict | None]:
    plan_path = out / "experiment_plan.json"
    present_steps = {int(p.name[11:]) for p in out.glob("checkpoint-*")
                     if p.is_dir() and p.name[11:].isdigit()}
    if not present_steps.issubset(CHECKPOINT_STEPS):
        raise ValueError(f"Unexpected exp38 checkpoint steps: {sorted(present_steps)}")
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
    if state["max_steps"] != MAX_STEPS:
        raise ValueError("Resume checkpoint has wrong exp38 horizon")
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
    source = plan.get("source_checkpoint")
    if not isinstance(source, dict) or not isinstance(source.get("sha256"), str) or len(source["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in source["sha256"]) or source.get("path") != SOURCE or source.get("step") != SOURCE_STEP:
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
    runtime = dict(epochs=50, source_batch=128, world_size=len(devices), micro_batch=micro,
                   gradient_accumulation_steps=gas, seed=args.seed, checkpoint_steps=list(CHECKPOINT_STEPS),
                   save_total_limit=2, report_to=args.report_to, max_steps=MAX_STEPS,
                   warmup_steps=252, warmup_ratio=0.0,
                   optimizer="adamw_torch", adam_beta1=0.9, adam_beta2=0.999,
                   adam_epsilon=1e-8, weight_decay=0.0, max_grad_norm=1.0)
    out = output_dir(args.experiment)
    identity = fingerprint(config, runtime)
    checkpoint, previous = check_resume(out, identity, args.resume)
    if checkpoint:
        state = trainer_state(checkpoint / "trainer_state.json")
        if state["max_steps"] != MAX_STEPS:
            raise ValueError("Resume checkpoint has wrong exp38 horizon")
        stop = config.get("training_stop_after_step")
        if stop is not None and state["global_step"] >= stop:
            raise ValueError("Checkpoint has reached requested stop")
        if state["global_step"] == MAX_STEPS:
            raise ValueError("Experiment already reached final checkpoint")
    cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={len(devices)}",
           f"--master_port={args.master_port}", "train_csgo.py", "--csgo_config", relative(out / "resolved_config.yaml"),
           "--deepspeed", "deepspeed_scripts/zero0.json", "--model_name_or_path", config["model_name_or_path"],
           "--unilip_factor", "10.6", "--mllm_hf_path", config["mllm_hf_path"], "--version", "internvl",
           "--data_type", "mix", "--csgo_image_folder", config["data_dir"], "--output_dir", relative(out),
           "--num_train_epochs", "50", "--max_steps", str(MAX_STEPS), "--warmup_steps", "252",
           "--per_device_train_batch_size", str(micro), "--per_device_eval_batch_size", "1",
           "--gradient_accumulation_steps", str(gas), "--eval_strategy", "no", "--save_strategy", "no",
           "--save_total_limit", "2", "--save_only_model", "False", "--seed", str(args.seed),
           "--learning_rate", str(config["learning_rate"]), "--weight_decay", "0", "--warmup_ratio", "0",
           "--lr_scheduler_type", config["lr_scheduler_type"], "--model_max_length", "1024", "--logging_steps", "10",
           "--dataloader_num_workers", "4", "--n_query", "256", "--n_und_query", "0", "--report_to", args.report_to,
           "--lora_r", str(config["lora_r"]), "--lora_alpha", str(config["lora_alpha"]),
           "--optim", "adamw_torch", "--adam_beta1", "0.9", "--adam_beta2", "0.999", "--adam_epsilon", "1e-8", "--max_grad_norm", "1.0"]
    for key, value in dict(mm_use_im_start_end=False, mm_use_im_patch_token=False, bf16=True, tf32=True,
                           gradient_checkpointing=True, lazy_preprocess=True, fix_vit=config["fix_vit"],
                           fix_llm=config["fix_llm"], fix_connect=config["fix_connect"], fix_dit=config["fix_dit"]).items():
        cmd.extend(["--" + key, str(value)])
    metadata = dict(experiment=args.experiment, fingerprint=identity, runtime=runtime, command=cmd,
                    resume_checkpoint=relative(checkpoint) if checkpoint else None,
                    source_checkpoint=previous.get("source_checkpoint") if previous else None,
                    evaluation_steps={"test": list(CHECKPOINT_STEPS)},
                    cumulative_steps={str(step): SOURCE_STEP + step for step in CHECKPOINT_STEPS},
                    dataset_counts=dict(EXPECTED_COUNTS))
    return [cmd], metadata, out


def tensor_header(path: Path) -> dict:
    from safetensors import safe_open
    total, lora_count, required = SOURCE_SHAPE
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        count = sum("lora_" in key for key in keys)
        if (len(keys), count) != (total, lora_count):
            raise ValueError(f"Source tensor/LoRA count differs from strict architecture: {len(keys)}/{count}")
        if any(not any(k.startswith(prefix) and "lora_" in k for k in keys) for prefix in required):
            raise ValueError("Source lacks required LoRA modules")
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
    """Pin the specified exp37@12000 model without requiring Stage 1 completion."""
    if config.get("finetune_init_ckpt_path") != SOURCE:
        raise ValueError("exp38 source must be exp37@12000")
    source = path_in_repo(SOURCE)
    checkpoint = source.parent
    if not source.is_file():
        raise ValueError(f"Missing exp37@12000 model: {source}")
    state = trainer_state(checkpoint / "trainer_state.json")
    if state["global_step"] != SOURCE_STEP or state["max_steps"] != SOURCE_HORIZON:
        raise ValueError("exp37 source must be global_step=12000/max_steps=19550")
    source_out = output_dir("exp37")
    source_recipe = read_yaml(source_out / "resolved_config.yaml")
    source_plan = json.loads((source_out / "experiment_plan.json").read_text(encoding="utf-8"))
    if source_recipe.get("exp37_experiment") != "exp37" or source_plan.get("experiment") != "exp37":
        raise ValueError("Source is not a managed exp37 run")
    expected_source = read_yaml(ROOT / "csgo_configs/exp37.yaml")
    for key in set(source_recipe) | set(expected_source):
        if key not in RECIPE_OVERRIDES and source_recipe.get(key, object()) != expected_source.get(key, object()):
            raise ValueError(f"Source exp37 fixed recipe differs in {key}")
    if source_plan.get("fingerprint") != fingerprint(source_recipe, source_plan.get("runtime", {})):
        raise ValueError("Source exp37 plan fingerprint changed")
    source_runtime = source_plan.get("runtime", {})
    if (source_runtime.get("epochs") != 50 or source_runtime.get("source_batch") != 128
        or source_runtime.get("save_steps") != 4000 or source_runtime.get("warmup_ratio") != 0.003):
        raise ValueError("Source exp37 training runtime differs")
    for key in ("benchmark_v2_manifest", "benchmark_v2_split", "train_maps", "val_maps", "test_maps",
                "is_multi_task", "is_multi_task_balanced", "task_mix_ratio", "alpha_loc_loss",
                "is_lora", "llm_train_mode", "lora_r", "lora_alpha", "lora_dropout",
                "enable_language_model_lora", "enable_gen_head_lora", "enable_loc_head_lora",
                "fix_vit", "fix_llm", "fix_connect", "fix_dit", "is_action_dit_projector", "use_pi05_action_dit"):
        if source_recipe.get(key) != config.get(key):
            raise ValueError(f"Source exp37 recipe differs in {key}")
    # Source and destination may use different minimal/full asset layouts.
    header = tensor_header(source)
    result = {"path": SOURCE, "sha256": digest_file(source), "size_bytes": source.stat().st_size,
              "step": SOURCE_STEP, "source_max_steps": SOURCE_HORIZON,
              "state_path": relative(checkpoint / "trainer_state.json"),
              "source_recipe": relative(source_out / "resolved_config.yaml"),
              "source_plan": relative(source_out / "experiment_plan.json"), **header}
    for sibling in EXPERIMENTS:
        if sibling == config["exp38_experiment"]:
            continue
        sibling_plan = output_dir(sibling) / "experiment_plan.json"
        if sibling_plan.is_file():
            peer = json.loads(sibling_plan.read_text(encoding="utf-8")).get("source_checkpoint")
            if not isinstance(peer, dict) or (peer.get("path"), peer.get("sha256")) != (SOURCE, result["sha256"]):
                raise ValueError(f"exp38 sibling {sibling} uses a different source")
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


def selected_evaluation_tasks(args: argparse.Namespace) -> tuple[bool, tuple[str, ...]]:
    """Return the active localization route and selected generation kinds."""
    if args.task == "gen_continuous" and args.split != "test":
        raise ValueError("Continuous generation requires --split test")
    loc = args.task in ("all", "localization", "loc")
    if args.task in ("localization", "loc"):
        kinds = ()
    elif args.task == "gen_continuous":
        kinds = ("continuous",)
    elif args.task == "gen":
        kinds = ("discrete",)
    else:
        kinds = ("discrete",) if args.split == "validation" else ("discrete", "continuous")
    return loc, kinds


def evaluation_snapshots(args: argparse.Namespace, legacy_path: Path, config: dict) -> dict[Path, dict]:
    """Resolve task-specific inference configs without changing the batch-one layout."""
    target = legacy_path.parent.relative_to(ROOT / "outputs_eval/benchmark_v2")
    loc, kinds = selected_evaluation_tasks(args)
    snapshots: dict[Path, dict] = {}
    if loc:
        path = legacy_path if args.test_loc_batch_size == 1 else ROOT / "outputs_loc/benchmark_v2" / target / f"infer_bs{args.test_loc_batch_size}" / "evaluation_config.yaml"
        value = copy.deepcopy(config)
        if args.test_loc_batch_size > 1:
            value["test_loc_batch_size"] = args.test_loc_batch_size
        snapshots[path] = value
    for kind in kinds:
        path = legacy_path if args.test_gen_batch_size == 1 else ROOT / "outputs_eval/benchmark_v2" / target / f"infer_bs{args.test_gen_batch_size}" / kind / "evaluation_config.yaml"
        value = copy.deepcopy(config)
        if kind == "continuous" and args.test_gen_batch_size > 1:
            value["benchmark_v2_split"] = "seen_continuous"
        if args.test_gen_batch_size > 1:
            value["test_gen_batch_size"] = args.test_gen_batch_size
        snapshots[path] = value
    return snapshots


def validate_batched_localization_summary(config_path: Path, config: dict,
                                          manifest: dict, split: str, seed: int) -> None:
    """Require the completed localization result before treating batched metrics as ready."""
    output = config_path.parent
    summary_path = output / "benchmark_csgo_v2_loc.json"
    if not summary_path.is_file():
        raise ValueError(f"Missing localization summary: {relative(summary_path)}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    aggregates._validate_localization_summary(summary, path=summary_path)
    expected_count = EXPECTED_COUNTS["validation"] if split == "validation" else EXPECTED_COUNTS["discrete_test"]
    expected = {"split": config["benchmark_v2_split"], "maps": MAPS,
                "sample_count": expected_count, "seed": seed, "kind": "localization"}
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Localization summary split/maps/count/seed differs: {relative(summary_path)}")
    if (path_in_repo(summary.get("checkpoint", "")) != path_in_repo(config["ckpt_path"]) or
        path_in_repo(summary.get("manifest", "")) != path_in_repo(config["benchmark_v2_manifest"])):
        raise ValueError(f"Localization summary checkpoint/manifest differs: {relative(summary_path)}")
    provenance = summary.get("inference_provenance", {})
    if (path_in_repo(provenance.get("path", "")) != output / "inference_manifest.json" or
        provenance.get("payload") != manifest):
        raise ValueError(f"Localization summary inference provenance differs: {relative(summary_path)}")


def evaluation_plan(args: argparse.Namespace, config: dict) -> tuple[list[list[str]], Path, dict]:
    step = MAX_STEPS if args.checkpoint_step == "final" else args.checkpoint_step
    if args.split != "test" or step not in CHECKPOINT_STEPS:
        raise ValueError("exp38 evaluation requires test checkpoint 4000 or final=7550")
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
        if state["global_step"] != step or state["max_steps"] != MAX_STEPS:
            raise ValueError("Selected evaluation checkpoint has wrong Trainer step/horizon")
        if step == MAX_STEPS and final_checkpoint_step(args.experiment) != step:
            raise ValueError("Final evaluation requires completed checkpoint")
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
    loc, kinds = selected_evaluation_tasks(args)
    common = ["--ckpt_path", str(checkpoint),
              "--seed", str(args.seed), "--benchmark_v2_maps", *MAPS]
    commands = []
    if args.command == "infer" and loc:
        output = f"outputs_loc/benchmark_v2/{target}" + ("" if args.test_loc_batch_size == 1 else f"/infer_bs{args.test_loc_batch_size}")
        path = eval_config if args.test_loc_batch_size == 1 else ROOT / output / "evaluation_config.yaml"
        commands.append([sys.executable, "eval_csgo_loc.py", "--csgo_config", relative(path), *common,
                         "--output_dir", output, "--benchmark_v2_split", split,
                         "--test-loc-batch-size", str(args.test_loc_batch_size)])
    for kind in kinds:
        selected_split = split if kind == "discrete" else "seen_continuous"
        output = f"outputs_eval/benchmark_v2/{target}" + ("" if args.test_gen_batch_size == 1 else f"/infer_bs{args.test_gen_batch_size}") + f"/{kind}"
        if args.command == "infer":
            path = eval_config if args.test_gen_batch_size == 1 else ROOT / output / "evaluation_config.yaml"
            commands.append([sys.executable, "eval_csgo.py", "--csgo_config", relative(path), *common, "--output_dir", output,
                             "--benchmark_v2_split", selected_split,
                             "--test-gen-batch-size", str(args.test_gen_batch_size)])
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
    p.add_argument("--stop-after-step", type=int, help="Local optimizer step 4000 or 7550; 0 removes the stop (default first run stops at 4000)")
    p.add_argument("--cuda-devices", default="0")
    p.add_argument("--micro-batch", type=int)
    p.add_argument("--master-port", type=int, default=29638)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report-to", default="none")
    p.add_argument("--checkpoint-step", type=parse_checkpoint_step)
    p.add_argument("--split", choices=("validation", "test"), default="test")
    p.add_argument("--task", choices=("all", "localization", "loc", "generation", "gen", "gen_continuous"), default="all")
    p.add_argument("--test-loc-batch-size", type=positive_batch_size, default=DEFAULT_TEST_LOC_BATCH_SIZE)
    p.add_argument("--test-gen-batch-size", type=positive_batch_size, default=DEFAULT_TEST_GEN_BATCH_SIZE)
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
            print("Validated fixed exp38 Seen10 recipes; no files written or jobs launched.")
            return 0
        if args.experiment is None:
            raise ValueError("--experiment is required")
        config = resolve_config(args, for_training=args.command == "train")
        validate_config(config)
        if args.command == "train":
            commands, metadata, out = training_plan(args, config)
            if args.execute and not args.resume:
                metadata["source_checkpoint"] = preflight_stage2_source(config)
                previous_plan = out / "experiment_plan.json"
                if previous_plan.is_file():
                    prior_source = json.loads(previous_plan.read_text()).get("source_checkpoint")
                    if not isinstance(prior_source, dict) or (prior_source.get("path"), prior_source.get("sha256")) != (metadata["source_checkpoint"]["path"], metadata["source_checkpoint"]["sha256"]):
                        raise ValueError("exp38 source changed since earlier plan")
            writes = [(out / "resolved_config.yaml", yaml.safe_dump(config, sort_keys=False)),
                      (out / "experiment_plan.json", json.dumps(metadata, indent=2) + "\n")]
        else:
            commands, path, eval_config = evaluation_plan(args, config)
            snapshots = evaluation_snapshots(args, path, eval_config)
            writes = []
            for snapshot_path, snapshot_cfg in snapshots.items():
                if snapshot_path.is_file():
                    if read_yaml(snapshot_path) != snapshot_cfg:
                        raise ValueError(f"Existing evaluation settings differ: {relative(snapshot_path)}")
                elif args.execute and args.command == "metrics" and snapshot_path != path:
                    raise ValueError(f"Missing inference evaluation snapshot: {relative(snapshot_path)}")
                if args.command == "infer" or snapshot_path == path:
                    writes.append((snapshot_path, yaml.safe_dump(snapshot_cfg, sort_keys=False)))
                if args.execute and args.command == "metrics" and snapshot_path != path:
                    manifest_path = snapshot_path.parent / "inference_manifest.json"
                    if not manifest_path.is_file():
                        raise ValueError(f"Missing inference provenance: {relative(manifest_path)}")
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    task = "loc" if "outputs_loc" in snapshot_path.parts else "gen"
                    batch_size = args.test_loc_batch_size if task == "loc" else args.test_gen_batch_size
                    validate_execution_metadata(manifest, task, batch_size)
                    if path_in_repo(manifest.get("config_path", "")) != snapshot_path:
                        raise ValueError(f"Inference config path differs: {relative(manifest_path)}")
                    if task == "loc":
                        validate_batched_localization_summary(snapshot_path, snapshot_cfg,
                                                              manifest, args.split, args.seed)
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
