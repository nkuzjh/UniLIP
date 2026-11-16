#!/usr/bin/env python3
"""Plan the fixed exp32_gen@12000 Unseen-4 transfer; run only with --execute."""
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
from scripts import run_exp37_seen10 as seen37  # noqa: E402
from scripts import run_exp37_unseen4 as legacy  # noqa: E402
from csgo_datasets.inference_batching import execution_contract, positive_batch_size  # noqa: E402

MATRIX_PATH = ROOT / "csgo_configs/benchmark_v2_exp32_gen_unseen4.yaml"
MATRIX = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
FAMILY = "exp32_fs_gen"
FEW_SHOT_LR = {"exp32_fs_gen": 2e-5,
               "exp32_fs_gen_lr1e5": 1e-5,
               "exp32_fs_gen_lr5e6": 5e-6}
MAPS = ("cs_office", "de_golden", "de_palacio", "de_vertigo")
SHOTS = (100, 50, 20, 10)
TASKS = ("gen", "gen_continuous")
SOURCE = ROOT / "outputs/csgo_1b/exp32_gen/checkpoint-12000/model.safetensors"
SOURCE_RECIPE = ROOT / "csgo_configs/exp32_gen.yaml"
BASE_RECIPE = ROOT / "csgo_configs/exp37_s2_gen.yaml"
LOW_LR = legacy.LR_LOW
HIGH_LR = legacy.LR_HIGH
RESUME_KEYS = ("csgo_global_batch_size", "training_stop_after_step",
               "training_checkpoint_steps")
INIT_KEYS = ("base_init_ckpt_path", "gen_init_ckpt_path", "loc_init_ckpt_path",
             "pretrain_path")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def yaml_text(config: dict) -> str:
    return yaml.safe_dump(config, sort_keys=False)


def json_text(value: dict) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def validate_matrix() -> None:
    expected = {
        "schema_version": 1, "family": FAMILY,
        "few_shot_learning_rates": FEW_SHOT_LR, "source": rel(SOURCE),
        "source_recipe": rel(SOURCE_RECIPE),
        "manifest": "data/csgo_benchmark_v2/benchmark_manifest.json",
        "minimal_asset_manifest": "data/csgo_benchmark_v2/minimal_dataset_report.json",
        "maps": list(MAPS), "shots": list(SHOTS), "support_seed": 0,
        "seed": 42, "max_steps": 400, "inference_batch_size": 32,
        "metric_batch_size": 1, "retention": False,
        "evaluation": {"paired_size": 448, "frame_diff_threshold": 2,
                       "min_track_len": 4, "clip_length": 16,
                       "clip_stride": 16, "fvd_size": 224,
                       "external_loc_repo_root": "csgosquare",
                       "external_loc_config_path": "configs_reg_newdata/exp5_2.yaml",
                       "external_loc_checkpoint_path":
                           "checkpoints_reg_newdata/exp5_2/20251227_091745/current_model.pth"},
    }
    if MATRIX != expected:
        raise ValueError("exp32_gen@12000 Unseen-4 matrix differs from fixed protocol")
    if (legacy.MATRIX["manifest"] != MATRIX["manifest"] or
        legacy.MATRIX["evaluation"] != {"batch_size": 1, **MATRIX["evaluation"]}):
        raise ValueError("Reused generation metric helper no longer matches fixed evaluation")


def source_recipe() -> dict:
    config = seen37.read_yaml(SOURCE_RECIPE)
    template = seen37.read_yaml(BASE_RECIPE)
    keys = ("is_multi_task", "is_multi_task_balanced", "task_mix_ratio", "alpha_loc_loss",
            "is_lora", "llm_train_mode", "lora_r", "lora_alpha", "lora_dropout",
            "enable_language_model_lora", "enable_gen_head_lora", "enable_loc_head_lora",
            "freeze_inactive_head", "fix_vit", "fix_llm", "fix_connect", "fix_dit",
            "is_loc_aux_loss", "is_aux_loc_combined_em_unc_loss", "is_loc_perception_loss",
            "is_gen_aux_loss", "is_repa_loss", "is_loc_repa_loss", "benchmark_v2_manifest",
            "use_pi05_action_dit", "is_action_dit_projector")
    for key in keys:
        if config.get(key) != template.get(key):
            raise ValueError(f"Source exp32_gen recipe architecture differs from gen template: {key}")
    if config.get("benchmark_v2_split") != "seen_train" or config.get("train_maps") != list(legacy.SEEN_MAPS):
        raise ValueError("Source exp32_gen recipe must be Seen-10 training")
    return config


def source_checkpoint(*, strict: bool) -> tuple[Path, dict]:
    evidence = {"path": rel(SOURCE), "step": 12000, "verified": False}
    if not strict:
        return SOURCE, evidence
    source_recipe()
    state_path = SOURCE.parent / "trainer_state.json"
    if not SOURCE.is_file() or not state_path.is_file():
        raise ValueError(f"Missing exact source checkpoint-12000 model/state: {SOURCE.parent}")
    state = seen37.trainer_state(state_path)
    if state["global_step"] != 12000:  # max_steps may exceed 12000.
        raise ValueError(f"Source Trainer global_step is not 12000: {state_path}")
    model_config_path = SOURCE.parent / "config.json"
    if not model_config_path.is_file():
        raise ValueError(f"Missing source model config: {model_config_path}")
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    expected_model = {"model_type": "unified_unilip", "is_lora": True,
                      "llm_train_mode": "lora", "enable_language_model_lora": True,
                      "enable_gen_head_lora": True, "enable_loc_head_lora": False,
                      "freeze_inactive_head": True, "is_loc_aux_loss": False,
                      "is_loc_perception_loss": False, "img_size": 224,
                      "action_dim": 5, "mllm_hf_path": "OpenGVLab/InternVL3-1B-hf"}
    if any(model_config.get(k) != v for k, v in expected_model.items()):
        raise ValueError(f"Source checkpoint model config is not exp32_gen gen-only: {model_config_path}")
    # exp32_gen and exp37_s2_gen have the same gen-only tensor structure. This
    # reads safetensors metadata only; strict loading later checks every shape.
    header = seen37.tensor_header(SOURCE, "exp37_s2_gen")
    saved_recipe_path = SOURCE.parent.parent / "resolved_config.yaml"
    if saved_recipe_path.is_file():
        saved_recipe = seen37.read_yaml(saved_recipe_path)
        canonical = source_recipe()
        for key in ("is_lora", "llm_train_mode", "lora_r", "lora_alpha",
                    "enable_gen_head_lora", "enable_loc_head_lora", "train_maps",
                    "benchmark_v2_split"):
            if saved_recipe.get(key) != canonical.get(key):
                raise ValueError(f"Saved source recipe identity differs: {key}")
    evidence.update(verified=True, sha256=legacy.sha256(SOURCE),
                    trainer_state_sha256=legacy.sha256(state_path),
                    source_recipe_sha256=legacy.sha256(SOURCE_RECIPE),
                    model_config_sha256=legacy.sha256(model_config_path),
                    trainer_state={"global_step": state["global_step"],
                                   "max_steps": state["max_steps"]},
                    header=header)
    if saved_recipe_path.is_file():
        evidence["saved_recipe_sha256"] = legacy.sha256(saved_recipe_path)
    return SOURCE, evidence


def asset_backend(args: argparse.Namespace) -> tuple[str | None, str]:
    if args.asset_mode == "full":
        if not args.data_dir:
            raise ValueError("--asset-mode full requires --data-dir PATH")
        return None, args.data_dir
    if args.data_dir:
        raise ValueError("--data-dir requires --asset-mode full")
    return MATRIX["minimal_asset_manifest"], "data/preprocessed_data"


def selected_tasks(task: str) -> tuple[str, ...]:
    if task == "all":
        return TASKS
    if task not in TASKS:
        raise ValueError(f"Only {TASKS} are supported")
    return (task,)


def model_dir(map_name: str, shots: int, family: str = FAMILY) -> Path:
    return ROOT / "outputs/csgo_1b" / f"{family}_{map_name}" / f"shot_{shots}" / "seed_0"


def selected_model_dir(args: argparse.Namespace, map_name: str, shots: int) -> Path:
    # Keep the original two-argument call for the default family and its callers.
    if args.experiment == FAMILY:
        return model_dir(map_name, shots)
    return model_dir(map_name, shots, args.experiment)


def eval_dir(map_name: str | None, shots: int, task: str, zero_shot: bool,
             batch_size: int, family: str = FAMILY) -> Path:
    return legacy.eval_dir(family, map_name, shots, "query", task, zero_shot, batch_size)


def recipe(args: argparse.Namespace, map_name: str | None, shots: int | None,
           *, zero_shot: bool) -> dict:
    family = args.experiment
    if family not in FEW_SHOT_LR:
        raise ValueError(f"Unknown few-shot family: {family}")
    if zero_shot and family != FAMILY:
        raise ValueError(f"{family} is few-shot only; use {FAMILY} --zero-shot")
    config = copy.deepcopy(seen37.read_yaml(BASE_RECIPE))
    config.pop("exp37_experiment", None)
    for key in tuple(config):
        if (key.startswith("resume_") or key.startswith("training_stop_") or
            key.startswith("training_checkpoint_") or key.startswith("finetune_init_") or
            key in (*RESUME_KEYS, *INIT_KEYS, "ckpt_path", "warmup_steps",
                    "alpha_loc_schedule_steps", "alpha_loc_schedule_values")):
            config.pop(key)
    config["exp32_gen_unseen4_experiment"] = family
    config["benchmark_v2_asset_manifest"], config["data_dir"] = asset_backend(args)
    config["model_name_or_path"] = args.model_path or "UniLIP-1B"
    config["pi05_pytorch_weight_path"] = str(Path(args.pi05_path or
        "~/.cache/openpi/openpi-assets/checkpoints/pi05_base").expanduser())
    config["batch_size"] = 1
    config["guidance_scale"] = 4.5
    config["action_dim"] = 5
    config["mllm_hf_path"] = "OpenGVLab/InternVL3-1B-hf"
    config["img_size"] = 224
    config["save_final_model_at_output_root"] = True
    config["disable_lora_dropout"] = False
    config["seed"] = 42
    config["is_multi_task_balanced"] = False
    config["task_mix_ratio"] = 0.0
    config["enable_loc_head_lora"] = False
    config["freeze_inactive_head"] = True
    config["alpha_loc_loss"] = 1.0
    for key in ("is_loc_aux_loss", "is_aux_loc_combined_em_unc_loss", "is_loc_perception_loss",
                "is_gen_aux_loss", "is_repa_loss", "is_loc_repa_loss", "is_aux_loc_em_loss",
                "is_aux_loc_uncertainty_loss"):
        config[key] = False
    for key in LOW_LR:
        config[key] = FEW_SHOT_LR[family]
    for key in HIGH_LR:
        config[key] = 1e-4
    config["warmup_ratio"] = 0.003
    config["lr_scheduler_type"] = "cosine_with_min_lr"
    config["lr_scheduler_kwargs"] = {"min_lr_rate": 0.1}
    config["weight_decay"] = 0.0
    config["csgo_loss_per_microbatch_mean"] = True
    config["csgo_forward_sampler_epoch"] = True
    if zero_shot:
        config.pop("benchmark_v2_support_seed", None)
        config.pop("benchmark_v2_shots_per_map", None)
    else:
        if map_name not in MAPS or shots not in SHOTS:
            raise ValueError("Unknown map or shot count")
        config["benchmark_v2_split"] = "crossmap_support"
        config["benchmark_v2_support_seed"] = 0
        config["benchmark_v2_shots_per_map"] = shots
        config["train_maps"] = config["val_maps"] = config["test_maps"] = [map_name]
        config["finetune_init_ckpt_path"] = rel(SOURCE)
        config["finetune_init_strict"] = True
    validate_recipe(config, map_name, shots, zero_shot=zero_shot, family=family)
    return config


def validate_recipe(config: dict, map_name: str | None, shots: int | None,
                    *, zero_shot: bool, family: str = FAMILY) -> None:
    if family not in FEW_SHOT_LR or (zero_shot and family != FAMILY):
        raise ValueError("Unsupported family or zero-shot variant")
    if config.get("exp32_gen_unseen4_experiment") != family or config.get("seed") != 42:
        raise ValueError("Recipe identity/seed differs")
    if any(key.startswith(("resume_", "training_stop_", "training_checkpoint_", "finetune_init_"))
           for key in config if key != "finetune_init_ckpt_path" and key != "finetune_init_strict"):
        raise ValueError("Recipe carries source continuation fields")
    if any(key in config for key in (*RESUME_KEYS, *INIT_KEYS, "ckpt_path")):
        raise ValueError("Recipe carries source resume or selective-init fields")
    if (config.get("batch_size"), config.get("guidance_scale"), config.get("action_dim"),
        config.get("mllm_hf_path"), config.get("img_size"),
        config.get("save_final_model_at_output_root"),
        config.get("disable_lora_dropout")) != (
            1, 4.5, 5, "OpenGVLab/InternVL3-1B-hf", 224, True, False):
        raise ValueError("Recipe generation inference/final-save fields differ")
    if config.get("is_lora") is not True or (config.get("lora_r"), config.get("lora_alpha")) != (32, 64):
        raise ValueError("Recipe LoRA structure differs")
    if (config.get("task_mix_ratio"), config.get("enable_gen_head_lora"),
        config.get("enable_loc_head_lora"), config.get("freeze_inactive_head")) != (0.0, True, False, True):
        raise ValueError("Recipe generation-only route differs")
    if any(config.get(key) is not False for key in ("is_loc_aux_loss", "is_aux_loc_combined_em_unc_loss",
                                                 "is_loc_perception_loss", "is_gen_aux_loss",
                                                 "is_repa_loss", "is_loc_repa_loss")):
        raise ValueError("Recipe enables auxiliary objective")
    if any(config.get(key) != FEW_SHOT_LR[family] for key in LOW_LR) or any(config.get(key) != 1e-4 for key in HIGH_LR):
        raise ValueError("Recipe learning rates differ")
    if (config.get("warmup_ratio"), config.get("lr_scheduler_type"),
        config.get("lr_scheduler_kwargs")) != (0.003, "cosine_with_min_lr", {"min_lr_rate": 0.1}):
        raise ValueError("Recipe schedule differs")
    if config.get("csgo_loss_per_microbatch_mean") is not True or config.get("csgo_forward_sampler_epoch") is not True:
        raise ValueError("Recipe microbatch/sampler semantics differ")
    if zero_shot:
        if any(k in config for k in ("benchmark_v2_support_seed", "benchmark_v2_shots_per_map",
                                     "finetune_init_ckpt_path", "finetune_init_strict")):
            raise ValueError("Zero-shot recipe carries support or adaptation init")
    elif (config.get("benchmark_v2_split"), config.get("train_maps"),
          config.get("benchmark_v2_support_seed"), config.get("benchmark_v2_shots_per_map"),
          config.get("finetune_init_ckpt_path"), config.get("finetune_init_strict")) != (
              "crossmap_support", [map_name], 0, shots, rel(SOURCE), True):
        raise ValueError("Few-shot recipe source/map/support differs")


def model_snapshot(args: argparse.Namespace, map_name: str, shots: int,
                   *, strict_source: bool) -> tuple[dict, dict, Path]:
    _, source = source_checkpoint(strict=strict_source)
    config = recipe(args, map_name, shots, zero_shot=False)
    plan = {"schema_version": 1, "family": args.experiment, "map": map_name, "shots": shots,
            "support_seed": 0, "seed": 42, "max_steps": 400,
            "source": source, "asset_mode": args.asset_mode or "minimal",
            "config_sha256": digest(yaml_text(config)),
            "effective_batch": {"micro_batch": shots, "configured_gas": 1,
                                "microbatches_per_update": 1,
                                "base_samples_per_update": shots,
                                "base_sample_exposures_400_steps": shots * 400}}
    return config, plan, selected_model_dir(args, map_name, shots)


def _check_overrides(args: argparse.Namespace, config: dict, plan: dict) -> None:
    if ((args.asset_mode and args.asset_mode != plan.get("asset_mode")) or
        (args.data_dir and args.data_dir != config.get("data_dir")) or
        (args.model_path and args.model_path != config.get("model_name_or_path")) or
        (args.pi05_path and str(Path(args.pi05_path).expanduser()) != config.get("pi05_pytorch_weight_path"))):
        raise ValueError("Requested backend/model differs from saved snapshot")


def read_model_snapshot(args: argparse.Namespace, map_name: str, shots: int,
                        *, strict_source: bool) -> tuple[dict, dict, Path]:
    out = selected_model_dir(args, map_name, shots)
    cp, pp = out / "resolved_config.yaml", out / "experiment_plan.json"
    if not cp.is_file() or not pp.is_file():
        raise ValueError(f"Missing adaptation snapshot: {out}")
    config = seen37.read_yaml(cp)
    plan = json.loads(pp.read_text(encoding="utf-8"))
    validate_recipe(config, map_name, shots, zero_shot=False, family=args.experiment)
    if any(plan.get(k) != v for k, v in {"family": args.experiment, "map": map_name, "shots": shots,
                                           "support_seed": 0, "seed": 42, "max_steps": 400}.items()):
        raise ValueError("Saved adaptation plan identity differs")
    if plan.get("config_sha256") != digest(yaml_text(config)) or plan.get("source", {}).get("path") != rel(SOURCE):
        raise ValueError("Saved adaptation recipe/source hash differs")
    if plan.get("effective_batch") != {"micro_batch": shots, "configured_gas": 1,
                                      "microbatches_per_update": 1, "base_samples_per_update": shots,
                                      "base_sample_exposures_400_steps": shots * 400}:
        raise ValueError("Saved effective batch differs")
    if plan["source"].get("step") != 12000 or plan["source"].get("verified") is not True:
        raise ValueError("Saved source is not verified checkpoint-12000")
    if strict_source:
        _, actual = source_checkpoint(strict=True)
        if plan["source"] != actual:
            raise ValueError("Saved source weight/state/recipe identity differs")
    _check_overrides(args, config, plan)
    return config, plan, out


def zero_snapshot(args: argparse.Namespace, *, strict_source: bool) -> tuple[dict, dict, list[tuple[Path, str]]]:
    base = ROOT / "outputs_eval/benchmark_v2" / FAMILY / "zero_shot" / "seed_42"
    cp, pp = base / "zero_shot_config.yaml", base / "zero_shot_plan.json"
    if cp.is_file() or pp.is_file():
        if not cp.is_file() or not pp.is_file():
            raise ValueError(f"Incomplete zero-shot snapshot: {base}")
        config = seen37.read_yaml(cp)
        plan = json.loads(pp.read_text(encoding="utf-8"))
        validate_recipe(config, None, None, zero_shot=True)
        if (plan.get("family"), plan.get("seed"), plan.get("source", {}).get("path"),
            plan.get("config_sha256")) != (FAMILY, 42, rel(SOURCE), digest(yaml_text(config))):
            raise ValueError("Saved zero-shot identity/config differs")
        if plan["source"].get("step") != 12000 or plan["source"].get("verified") is not True:
            raise ValueError("Saved zero-shot source is not verified")
        if strict_source and plan["source"] != source_checkpoint(strict=True)[1]:
            raise ValueError("Saved zero-shot source weight/state/recipe identity differs")
        _check_overrides(args, config, plan)
        return config, plan, []
    _, source = source_checkpoint(strict=strict_source)
    config = recipe(args, None, None, zero_shot=True)
    plan = {"schema_version": 1, "family": FAMILY, "seed": 42,
            "asset_mode": args.asset_mode or "minimal", "source": source,
            "config_sha256": digest(yaml_text(config))}
    return config, plan, [(cp, yaml_text(config)), (pp, json_text(plan))]


def final_model(out: Path, *, strict: bool) -> Path:
    model = out / "model.safetensors"
    if strict:
        state_path = out / "trainer_state.json"
        missing = [str(path) for path in (model, state_path) if not path.is_file()]
        if missing:
            raise ValueError(
                f"Missing completed 400-step adaptation: {out}; missing files: "
                f"{', '.join(missing)}; expected Trainer global_step/max_steps=400/400. "
                "Check the earlier training log for the first error.")
        state = seen37.trainer_state(state_path)
        if state["global_step"] != 400 or state["max_steps"] != 400:
            raise ValueError(
                f"Adaptation has not completed exactly 400 steps: {state_path}; "
                f"Trainer global_step/max_steps={state['global_step']}/{state['max_steps']} "
                "(expected 400/400). Check the earlier training log for the first error.")
    return model


def adaptation_status(out: Path) -> dict:
    paths = {"resolved_config.yaml": out / "resolved_config.yaml",
             "experiment_plan.json": out / "experiment_plan.json",
             "model.safetensors": out / "model.safetensors",
             "trainer_state.json": out / "trainer_state.json"}
    present = {name: path.is_file() for name, path in paths.items()}
    result = {"snapshot": present["experiment_plan.json"],
              "model": present["model.safetensors"],
              "state": present["trainer_state.json"],
              "global_step": None, "max_steps": None,
              "missing_artifacts": [name for name, exists in present.items() if not exists]}
    if present["trainer_state.json"]:
        try:
            state = seen37.trainer_state(paths["trainer_state.json"])
            result["global_step"], result["max_steps"] = state["global_step"], state["max_steps"]
        except (ValueError, OSError, KeyError, TypeError) as exc:
            result["training_status"] = "invalid_state"
            result["state_error"] = str(exc)
            return result
    if not any(present.values()):
        result["training_status"] = "missing"
    elif (present["resolved_config.yaml"] and present["experiment_plan.json"] and
          not present["model.safetensors"] and not present["trainer_state.json"]):
        result["training_status"] = "snapshot_only"
    elif (not result["missing_artifacts"] and
          (result["global_step"], result["max_steps"]) == (400, 400)):
        result["training_status"] = "complete"
    else:
        result["training_status"] = "incomplete"
    return result


def train_command(config: dict, out: Path, shots: int, args: argparse.Namespace) -> list[str]:
    local = copy.copy(args)
    local.experiment, local.shots = "exp37_fs_gen", shots
    command = legacy.train_command(config, out, local)
    # The legacy builder pins this CLI option to 2e-5; Trainer reads CLI
    # options after the YAML and would otherwise override the new family LR.
    lr_option = command.index("--learning_rate") + 1
    command[lr_option] = f"{config['learning_rate']:.0e}".replace("e-0", "e-")
    if args.master_port == 0:
        port_index = command.index("--master_port=0")
        command[port_index:port_index + 1] = ["--standalone", "--nnodes=1"]
    return command


def validate_predictions(output: Path, maps: tuple[str, ...], task: str) -> None:
    expected = 1280 if task == "gen_continuous" else 2000
    for map_name in maps:
        folder = output / "gen_imgs" / map_name
        if not folder.is_dir():
            raise ValueError(f"Missing prediction directory: {folder}")
        count = sum(1 for suffix in ("*.jpg", "*.jpeg", "*.png") for _ in folder.glob(suffix))
        if count != expected:
            raise ValueError(f"Prediction count differs for {map_name}: {count} != {expected}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("validate", "source-check", "status", "train", "infer", "metrics", "aggregate"))
    p.add_argument("--experiment", choices=tuple(FEW_SHOT_LR), default=FAMILY)
    p.add_argument("--map", choices=MAPS)
    p.add_argument("--shots", type=int, default=100)
    p.add_argument("--zero-shot", action="store_true")
    p.add_argument("--split", choices=("query",), default="query")
    p.add_argument("--task", choices=("all", *TASKS), default="all")
    p.add_argument("--test-gen-batch-size", type=positive_batch_size, default=32)
    p.add_argument("--asset-mode", choices=("minimal", "full"))
    p.add_argument("--data-dir")
    p.add_argument("--model-path")
    p.add_argument("--pi05-path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--support-seed", type=int, default=0)
    p.add_argument("--master-port", type=int, default=29642,
                   help="0: independent single-node torchrun rendezvous; 1-65535: fixed port (default: 29642)")
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--report-to", default="none")
    p.add_argument("--execute", action="store_true")
    return p


def plan(args: argparse.Namespace) -> tuple[list[list[str]], list[tuple[Path, str]], dict]:
    validate_matrix()
    source_recipe()
    if args.experiment not in FEW_SHOT_LR:
        raise ValueError(f"Unknown few-shot family: {args.experiment}")
    if args.zero_shot and args.experiment != FAMILY:
        raise ValueError(f"{args.experiment} is few-shot only; use {FAMILY} --zero-shot")
    if args.seed != 42 or args.support_seed != 0 or args.shots not in SHOTS:
        raise ValueError("Protocol fixes seed 42, support seed 0 and shots 100/50/20/10")
    if (not args.cuda_device or "," in args.cuda_device or
        any(c.isspace() for c in args.cuda_device) or not 0 <= args.master_port <= 65535):
        raise ValueError("Select one CUDA device and a valid master port")
    if args.zero_shot and args.map is not None:
        raise ValueError("Zero-shot has no --map")
    if args.command == "train" and (args.zero_shot or args.map is None or args.task != "all"):
        raise ValueError("train requires --map and no --zero-shot or --task")
    if args.command in ("infer", "metrics") and not args.zero_shot and args.map is None:
        raise ValueError(f"{args.command} requires --map or --zero-shot")
    if args.command == "aggregate" and args.map is not None:
        raise ValueError("aggregate spans all four maps; omit --map")
    commands: list[list[str]] = []
    writes: list[tuple[Path, str]] = []
    details = {"family": args.experiment, "map": args.map, "shots": args.shots,
               "zero_shot": args.zero_shot, "split": "query", "task": args.task,
               "execute": args.execute}
    if args.command == "validate":
        for map_name in MAPS:
            for shots in SHOTS:
                model_snapshot(args, map_name, shots, strict_source=False)
        details["validated_points"] = 16
        details["source_ready"] = SOURCE.is_file() and (SOURCE.parent / "trainer_state.json").is_file()
        return commands, writes, details
    if args.command == "source-check":
        details["source"] = source_checkpoint(strict=True)[1]
        return commands, writes, details
    if args.command == "status":
        try:
            details["source"] = source_checkpoint(strict=True)[1]
            details["source_status"] = "ready"
        except (ValueError, OSError) as exc:
            details["source_error"] = str(exc)
            details["source_status"] = "waiting" if not SOURCE.is_file() else "invalid"
        details["adaptations"] = [
            {"map": m, "shots": n, **adaptation_status(selected_model_dir(args, m, n))}
            for m in MAPS for n in SHOTS]
        return commands, writes, details
    if args.command == "train":
        config, saved, out = model_snapshot(args, args.map, args.shots, strict_source=args.execute)
        if args.execute and out.exists() and any(out.iterdir()):
            raise ValueError(f"Refusing to overwrite or auto-resume adaptation: {out}")
        commands.append(train_command(config, out, args.shots, args))
        writes.extend(((out / "resolved_config.yaml", yaml_text(config)),
                       (out / "experiment_plan.json", json_text(saved))))
        details["source"] = saved["source"]
        details["effective_batch"] = saved["effective_batch"]
        return commands, writes, details
    if args.command == "aggregate":
        for task in selected_tasks(args.task):
            if args.execute:
                if args.zero_shot:
                    config, saved, _ = zero_snapshot(args, strict_source=True)
                    output = eval_dir(None, args.shots, task, True,
                                      args.test_gen_batch_size, args.experiment)
                    inf = legacy.validate_inference(output, config, SOURCE, MAPS,
                        legacy.SPLITS["query"][task], task, None, None, saved["source"], args.test_gen_batch_size)
                    for m in MAPS:
                        legacy.validate_generation_result(output, m, task, legacy.SPLITS["query"][task], inf)
                else:
                    for m in MAPS:
                        config, saved, out = read_model_snapshot(args, m, args.shots, strict_source=True)
                        checkpoint = final_model(out, strict=True)
                        output = eval_dir(m, args.shots, task, False,
                                          args.test_gen_batch_size, args.experiment)
                        inf = legacy.validate_inference(output, config, checkpoint, (m,),
                            legacy.SPLITS["query"][task], task, args.shots, 0, saved["source"], args.test_gen_batch_size)
                        legacy.validate_generation_result(output, m, task, legacy.SPLITS["query"][task], inf)
            commands.append(legacy.aggregate_command(args.experiment, args.shots, task, args.zero_shot,
                                                      batch_size=args.test_gen_batch_size))
        return commands, writes, details
    if args.zero_shot:
        config, saved, new_writes = zero_snapshot(args, strict_source=args.execute)
        if args.execute and args.command == "metrics" and new_writes:
            raise ValueError("Zero-shot metrics require prior inference snapshot")
        if args.command == "infer":
            writes.extend(new_writes)
        checkpoint, maps = SOURCE, MAPS
    else:
        if args.execute or (selected_model_dir(args, args.map, args.shots) / "resolved_config.yaml").is_file():
            config, saved, out = read_model_snapshot(args, args.map, args.shots, strict_source=args.execute)
        else:
            config, saved, out = model_snapshot(args, args.map, args.shots, strict_source=False)
        checkpoint, maps = final_model(out, strict=args.execute), (args.map,)
    details["source"] = saved["source"]
    for task in selected_tasks(args.task):
        split = legacy.SPLITS["query"][task]
        output = eval_dir(args.map, args.shots, task, args.zero_shot,
                          args.test_gen_batch_size, args.experiment)
        if args.command == "infer":
            ecfg = legacy.eval_config(config, checkpoint, split, maps, task, args.test_gen_batch_size)
            cp, pp = output / "evaluation_config.yaml", output / "evaluation_plan.json"
            config_content = yaml_text(ecfg)
            eval_plan = {"schema_version": 1, "checkpoint": rel(checkpoint),
                         "checkpoint_sha256": legacy.sha256(checkpoint) if args.execute else None,
                         "config_sha256": digest(config_content), "split": split,
                         "maps": list(maps), "seed": 42,
                         "support_seed": None if args.zero_shot else 0,
                         "shots": None if args.zero_shot else args.shots,
                         "asset_manifest": config.get("benchmark_v2_asset_manifest"),
                         "source": saved["source"],
                         "expected_samples": len(maps) * (1280 if task == "gen_continuous" else 2000),
                         "continuous_clips_per_map": 20 if task == "gen_continuous" else None,
                         "continuous_frames_per_clip": 64 if task == "gen_continuous" else None,
                         "model_mode_policy": "preserve_loaded_module_modes_v1",
                         "inference_execution": execution_contract(task, args.test_gen_batch_size)}
            if args.execute and output.exists() and any(output.iterdir()):
                raise ValueError(f"Refusing to overwrite existing inference output: {output}")
            writes.extend(((cp, config_content), (pp, json_text(eval_plan))))
            command = [sys.executable, "eval_csgo.py", "--csgo_config", rel(cp),
                       "--output_dir", rel(output), "--ckpt_path", rel(checkpoint),
                       "--seed", "42", "--benchmark_v2_split", split,
                       "--benchmark_v2_maps", *maps,
                       "--test-gen-batch-size", str(args.test_gen_batch_size)]
            if not args.zero_shot:
                command.extend(("--benchmark_v2_support_seed", "0", "--benchmark_v2_shots_per_map", str(args.shots)))
            if config.get("benchmark_v2_asset_manifest"):
                command.extend(("--benchmark_v2_asset_manifest", config["benchmark_v2_asset_manifest"]))
            commands.append(command)
        else:
            if args.execute:
                legacy.validate_inference(output, config, checkpoint, maps, split, task,
                    None if args.zero_shot else args.shots,
                    None if args.zero_shot else 0, saved["source"], args.test_gen_batch_size)
                validate_predictions(output, maps, task)
            for m in maps:
                commands.append(legacy.metric_command(config, output, split, task, m))
    return commands, writes, details


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        commands, writes, details = plan(args)
        print(json.dumps({"mode": "execute" if args.execute else "dry-run", "details": details,
                          "writes": [rel(path) for path, _ in writes],
                          "commands": [shlex.join(command) for command in commands]}, indent=2), flush=True)
        if args.execute and args.command not in ("validate", "source-check", "status"):
            for path, content in writes:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.cuda_device}
            for command in commands:
                subprocess.run(command, cwd=ROOT, env=env, check=True)
                if args.command == "train":
                    final_model(selected_model_dir(args, args.map, args.shots), strict=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError,
            subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
