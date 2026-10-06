#!/usr/bin/env python3
"""Plan or explicitly execute one exp32 Seen-3 experiment stage.

No arguments launch a queue. Commands are printed only unless --execute is set.
Training stops are prefixes of the unchanged 50-epoch schedule; --resume uses
the same output directory and restores the complete Trainer checkpoint.
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
ROUND2 = ("aux_control", "perception", "auxloc", "both")
EXPERIMENTS = tuple(PREFIX + name for name in ROUND1 + ROUND2)
JOINT_BASES = tuple(PREFIX + name for name in ROUND1[2:])
MAPS = ["de_ancient", "de_dust2", "de_nuke"]
DEFAULT_BASE = PREFIX + "joint_sharedlr_constloc"
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


def is_round2(experiment: str) -> bool:
    return experiment.removeprefix(PREFIX) in ROUND2


def resolve_config(args: argparse.Namespace, *, for_training: bool) -> dict:
    source = read_yaml(config_path(args.experiment))
    out = ROOT / "outputs/csgo_1b" / args.experiment
    snapshot = out / "resolved_config.yaml"
    trained = None
    # Evaluation follows the recipe actually trained, even after editing a YAML.
    if not for_training and snapshot.is_file():
        source = read_yaml(snapshot)
        trained = copy.deepcopy(source)
    elif is_round2(args.experiment):
        saved_base = None
        if for_training and args.resume and snapshot.is_file():
            saved_base = read_yaml(snapshot).get("exp32_3maps_joint_base")
        base_name = args.joint_base or saved_base or source.get("exp32_3maps_joint_base", DEFAULT_BASE)
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
        raise ValueError("All eleven arms must use microbatch-mean GAS normalization")
    if config.get("is_lora") is not True or config.get("llm_train_mode") != "lora":
        raise ValueError("Expected the exp32 LoRA architecture")
    if any(config.get(key) for key in (
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
    if args.micro_batch <= 0 or 128 % (len(devices) * args.micro_batch):
        raise ValueError("GPU count × micro batch must divide effective source batch 128")
    accumulation = 128 // (len(devices) * args.micro_batch)
    runtime = dict(epochs=50, world_size=len(devices), micro_batch=args.micro_batch,
                   gradient_accumulation_steps=accumulation, seed=args.seed,
                   save_steps=1200, save_total_limit=6, report_to=args.report_to)
    out = ROOT / "outputs/csgo_1b" / args.experiment
    identity = fingerprint(config, runtime)
    checkpoint = check_resume(out, identity, args.resume)
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
           "--per_device_train_batch_size", str(args.micro_batch),
           "--per_device_eval_batch_size", "1", "--gradient_accumulation_steps", str(accumulation),
           "--eval_strategy", "no", "--save_strategy", "steps", "--save_steps", "1200",
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
    metadata = dict(experiment=args.experiment, fingerprint=identity, runtime=runtime,
                    joint_base=config.get("exp32_3maps_joint_base"), command=cmd,
                    resume_checkpoint=relative(checkpoint) if checkpoint else None)
    return [cmd], metadata, out


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
    if args.checkpoint_step is None or args.checkpoint_step <= 0:
        raise ValueError("infer/metrics require --checkpoint-step with a positive saved global step")
    if "," in args.cuda_devices:
        raise ValueError("Inference/metrics use one --cuda-devices device")
    checkpoint = Path("outputs/csgo_1b") / args.experiment / f"checkpoint-{args.checkpoint_step}" / "model.safetensors"
    if args.execute and not repo_path(checkpoint).is_file():
        raise ValueError(f"Checkpoint is absent: {checkpoint}")
    if args.execute and not (ROOT / "outputs/csgo_1b" / args.experiment / "resolved_config.yaml").is_file():
        raise ValueError("Evaluation requires the saved training recipe")
    target = f"{args.experiment}/checkpoint-{args.checkpoint_step}/{args.split}/seed_{args.seed}"
    eval_config = ROOT / "outputs_eval/benchmark_v2" / target / "evaluation_config.yaml"
    split = "seen_validation" if args.split == "validation" else "seen_discrete_test"
    config = copy.deepcopy(config)
    config["ckpt_path"] = str(checkpoint)
    config["benchmark_v2_split"] = split
    config["seed"] = args.seed
    # These training-only objectives do not add modules in this experiment family.
    for key in ("is_loc_aux_loss", "is_aux_loc_combined_em_unc_loss", "is_loc_perception_loss"):
        config[key] = False
    config.pop("training_stop_after_step", None)
    loc = args.experiment != PREFIX + "gen_single"
    gen = args.experiment != PREFIX + "loc_single"
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


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("validate", "train", "infer", "metrics"))
    result.add_argument("--experiment", choices=EXPERIMENTS)
    result.add_argument("--execute", action="store_true", help="Execute the displayed commands; default is print only")
    result.add_argument("--joint-base", choices=JOINT_BASES)
    result.add_argument("--stop-after-step", type=int, help="Absolute optimizer step; 0 removes the early stop")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--cuda-devices", default="0")
    result.add_argument("--micro-batch", type=int, default=4)
    result.add_argument("--master-port", type=int, default=29632)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--report-to", default="none")
    result.add_argument("--checkpoint-step", type=int)
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
