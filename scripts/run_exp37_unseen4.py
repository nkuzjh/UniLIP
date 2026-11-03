#!/usr/bin/env python3
"""Plan exp37 Unseen-4 adaptation and evaluation; execute only with --execute."""
from __future__ import annotations

import argparse
import copy
from functools import lru_cache
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
from scripts import aggregate_csgo_benchmark_v2_metrics as aggregates  # noqa: E402
from scripts import run_csgo_benchmark_v2_map_specific as map_runner  # noqa: E402

MATRIX_PATH = ROOT / "csgo_configs/benchmark_v2_exp37_unseen4.yaml"
MATRIX = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
FAMILIES = MATRIX["families"]
MAPS = tuple(MATRIX["maps"])
SEEN_MAPS = tuple(MATRIX["seen_maps"])
SHOTS = tuple(MATRIX["shots"])
SEED = MATRIX["seed"]
SUPPORT_SEED = MATRIX["support_seed"]
ROUTES = {"gen": ("gen", "gen_continuous"), "loc": ("loc",),
          "joint": ("loc", "gen", "gen_continuous")}
SPLITS = {"query": {"loc": "crossmap_query_test", "gen": "crossmap_query_test",
                    "gen_continuous": "crossmap_continuous"},
          "retention": {"loc": "seen_discrete_test", "gen": "seen_discrete_test",
                        "gen_continuous": "seen_continuous"}}
LR_LOW = ("learning_rate", "mm_projector_lr", "action_dit_lr", "llm_lora_lr",
          "llm_connector_lora_lr", "gen_dit_lora_lr", "action_dit_lora_lr", "action_io_mlp_lr")
LR_HIGH = ("action_dit_connector_lr", "action_dit_norm_lr", "action_dit_projector_lr")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def absolute(value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


@lru_cache(maxsize=64)
def _sha256_cached(path: Path, size: int, mtime_ns: int, ctime_ns: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def validate_matrix() -> None:
    expected = {
        "schema_version": 1,
        "manifest": "data/csgo_benchmark_v2/benchmark_manifest.json",
        "minimal_asset_manifest": "data/csgo_benchmark_v2/minimal_dataset_report.json",
        "maps": ["cs_office", "de_golden", "de_palacio", "de_vertigo"],
        "seen_maps": ["cs_agency", "cs_italy", "de_ancient", "de_anubis", "de_dust2",
                      "de_inferno", "de_mirage", "de_nuke", "de_overpass", "de_train"],
        "shots": [100, 50, 20, 10], "support_seed": 0, "seed": 42,
        "max_steps": 400,
        "families": {"exp37_fs_gen": {"source": "exp37_s2_gen", "route": "gen"},
                     "exp37_fs_loc": {"source": "exp37_s2_loc", "route": "loc"},
                     "exp37_fs_joint": {"source": "exp37_s2", "route": "joint"},
                     "exp37_fs_both": {"source": "exp37_s2_both", "route": "joint"},
                     "exp37_fs_both_b1": {"source": "exp37_s2_both_b1", "route": "joint"}},
        "training": {"save_strategy": "no", "eval_strategy": "no",
                     "joint_micro_batch": 4, "joint_gradient_accumulation_steps": 32,
                     "single_micro_batch": "shots", "single_gradient_accumulation_steps": 1,
                     "csgo_loss_per_microbatch_mean": True,
                     "csgo_forward_sampler_epoch": True,
                     "alpha_loc_joint": 8.0, "alpha_loc_single": 1.0,
                     "alpha_loc_aux_both": 1.0, "alpha_loc_perception_both": 0.05,
                     "active_lora_lr": 2e-5, "action_connector_norm_projector_lr": 1e-4,
                     "optimizer": "adamw_torch", "adam_beta1": 0.9,
                     "adam_beta2": 0.999, "adam_epsilon": 1e-8,
                     "weight_decay": 0.0, "max_grad_norm": 1.0,
                     "scheduler": "cosine_with_min_lr", "min_lr_rate": 0.1,
                     "warmup_ratio": 0.003},
        "evaluation": {"batch_size": 1, "paired_size": 448,
                       "frame_diff_threshold": 2, "min_track_len": 4,
                       "clip_length": 16, "clip_stride": 16, "fvd_size": 224,
                       "external_loc_repo_root": "csgosquare",
                       "external_loc_config_path": "configs_reg_newdata/exp5_2.yaml",
                       "external_loc_checkpoint_path": "checkpoints_reg_newdata/exp5_2/20251227_091745/current_model.pth"},
    }
    if MATRIX != expected:
        raise ValueError("exp37 Unseen-4 matrix differs from fixed protocol")


def model_dir(family: str, map_name: str, shots: int) -> Path:
    return ROOT / "outputs/csgo_1b" / f"{family}_{map_name}" / f"shot_{shots}" / "seed_0"


def eval_dir(family: str, map_name: str | None, shots: int, split: str,
             task: str, zero_shot: bool) -> Path:
    root = ROOT / ("outputs_loc/benchmark_v2" if task == "loc" else "outputs_eval/benchmark_v2")
    if zero_shot:
        base = root / family / "zero_shot" / "seed_42"
    else:
        base = root / f"{family}_{map_name}" / f"shot_{shots}" / "seed_0"
    if split == "retention":
        base /= "seen_retention"
    if task == "gen":
        base /= "discrete"
    elif task == "gen_continuous":
        base /= "continuous"
    return base


def parent_checkpoint(family: str, *, strict: bool) -> tuple[Path, dict]:
    source = FAMILIES[family]["source"]
    path = ROOT / "outputs/csgo_1b" / source / "checkpoint-6000/model.safetensors"
    if not strict:
        return path, {"path": rel(path), "sha256": None, "step": 6000, "verified": False}
    step = seen37.final_checkpoint_step(source)
    if step != 6000 or not path.is_file():
        raise ValueError(f"Missing completed source checkpoint-6000 model: {path}")
    state = seen37.trainer_state(path.parent / "trainer_state.json")
    if state["global_step"] != state["max_steps"] or state["global_step"] != 6000:
        raise ValueError(f"Incomplete source Trainer state: {path.parent}")
    recipe_path = path.parent.parent / "resolved_config.yaml"
    recipe = seen37.read_yaml(recipe_path)
    seen37.validate_config(recipe)
    if recipe.get("exp37_experiment") != source:
        raise ValueError(f"Source recipe identity differs: {recipe_path}")
    plan_path = path.parent.parent / "experiment_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("experiment") != source or plan.get("fingerprint") != seen37.fingerprint(recipe, plan.get("runtime", {})):
        raise ValueError(f"Source plan fingerprint differs: {plan_path}")
    runtime = plan.get("runtime", {})
    if (runtime.get("max_steps") != 6000 or runtime.get("seed") != 42 or
        runtime.get("source_batch") != 128 or runtime.get("warmup_steps") != 200):
        raise ValueError(f"Source runtime differs from completed exp37 Stage 2 protocol: {plan_path}")
    canonical = seen37.read_yaml(ROOT / "csgo_configs" / f"{source}.yaml")
    for key in ("is_multi_task", "is_multi_task_balanced", "task_mix_ratio", "alpha_loc_loss",
                "is_lora", "llm_train_mode", "lora_r", "lora_alpha", "lora_dropout",
                "enable_language_model_lora", "enable_gen_head_lora", "enable_loc_head_lora",
                "freeze_inactive_head", "fix_vit", "fix_llm", "fix_connect", "fix_dit",
                "is_loc_aux_loss", "is_aux_loc_combined_em_unc_loss", "is_loc_perception_loss",
                "loc_perception_use_attention_weight", "train_maps", "benchmark_v2_split"):
        if recipe.get(key) != canonical.get(key):
            raise ValueError(f"Source recipe differs from canonical {source}.{key}")
    header = seen37.tensor_header(path, source)
    return path, {"path": rel(path), "sha256": sha256(path), "step": 6000,
                  "verified": True, "recipe_path": rel(recipe_path),
                  "recipe_sha256": sha256(recipe_path), "plan_path": rel(plan_path),
                  "plan_sha256": sha256(plan_path), "header": header}


def asset_backend(args: argparse.Namespace) -> tuple[str | None, str]:
    if args.asset_mode == "full":
        if not args.data_dir:
            raise ValueError("--asset-mode full requires --data-dir PATH")
        return None, args.data_dir
    if args.data_dir:
        raise ValueError("--data-dir requires --asset-mode full")
    return MATRIX["minimal_asset_manifest"], "data/preprocessed_data"


def training_recipe(family: str, map_name: str, shots: int, args: argparse.Namespace,
                    source_path: Path) -> dict:
    cfg = seen37.read_yaml(ROOT / "csgo_configs" / f"{FAMILIES[family]['source']}.yaml")
    cfg = copy.deepcopy(cfg)
    cfg.pop("exp37_experiment", None)
    cfg["exp37_unseen4_experiment"] = family
    cfg["benchmark_v2_manifest"] = MATRIX["manifest"]
    cfg["benchmark_v2_split"] = "crossmap_support"
    cfg["benchmark_v2_support_seed"] = SUPPORT_SEED
    cfg["benchmark_v2_shots_per_map"] = shots
    cfg["train_maps"] = cfg["val_maps"] = cfg["test_maps"] = [map_name]
    cfg["finetune_init_ckpt_path"] = rel(source_path)
    cfg["finetune_init_strict"] = True
    cfg["seed"] = SEED
    cfg["benchmark_v2_asset_manifest"], cfg["data_dir"] = asset_backend(args)
    cfg.setdefault("model_name_or_path", "UniLIP-1B")
    if args.model_path:
        cfg["model_name_or_path"] = args.model_path
    if args.pi05_path:
        cfg["pi05_pytorch_weight_path"] = str(Path(args.pi05_path).expanduser())
    cfg["alpha_loc_loss"] = 1.0 if FAMILIES[family]["route"] == "loc" else 8.0
    if FAMILIES[family]["route"] == "gen":
        cfg["is_multi_task_balanced"] = False
        cfg["task_mix_ratio"] = 0.0
        cfg["alpha_loc_loss"] = 1.0  # inactive localization route
    if family in ("exp37_fs_both", "exp37_fs_both_b1"):
        for stem, value in (("alpha_loc_aux", 1.0), ("alpha_loc_perception", 0.05)):
            cfg[stem + "_loss"] = value
            cfg.pop(stem + "_schedule_steps", None)
            cfg.pop(stem + "_schedule_values", None)
        cfg["loc_perception_use_attention_weight"] = family.endswith("_b1")
    for key in LR_LOW:
        cfg[key] = 2e-5
    for key in LR_HIGH:
        cfg[key] = 1e-4
    cfg["weight_decay"] = 0.0
    cfg["lr_scheduler_type"] = "cosine_with_min_lr"
    cfg["lr_scheduler_kwargs"] = {"min_lr_rate": 0.1}
    cfg["warmup_ratio"] = 0.003
    cfg.pop("warmup_steps", None)
    cfg["csgo_loss_per_microbatch_mean"] = True
    cfg["csgo_forward_sampler_epoch"] = True
    return cfg


def validate_recipe(cfg: dict, family: str, map_name: str, shots: int) -> None:
    if cfg.get("exp37_unseen4_experiment") != family or cfg.get("train_maps") != [map_name] or cfg.get("benchmark_v2_shots_per_map") != shots:
        raise ValueError("Saved recipe family/map/shots differs from requested experiment")
    if cfg.get("benchmark_v2_support_seed") != 0 or cfg.get("seed") != 42 or cfg.get("benchmark_v2_split") != "crossmap_support":
        raise ValueError("Saved recipe support/train seed or split differs")
    if cfg.get("finetune_init_strict") is not True or cfg.get("finetune_init_ckpt_path") != rel(parent_checkpoint(family, strict=False)[0]):
        raise ValueError("Saved recipe source differs from fixed checkpoint-6000")
    if cfg.get("warmup_ratio") != 0.003 or "warmup_steps" in cfg or cfg.get("lr_scheduler_kwargs") != {"min_lr_rate": 0.1}:
        raise ValueError("Saved recipe optimizer schedule differs")
    if cfg.get("csgo_loss_per_microbatch_mean") is not True or cfg.get("csgo_forward_sampler_epoch") is not True:
        raise ValueError("Saved recipe microbatch normalization or sampler epoch forwarding differs")
    if any(cfg.get(k) != 2e-5 for k in LR_LOW) or any(cfg.get(k) != 1e-4 for k in LR_HIGH):
        raise ValueError("Saved recipe learning rates differ")
    route = FAMILIES[family]["route"]
    expected_gating = {"gen": (False, 1.0, True, False),
                       "loc": (False, 1.0, False, True),
                       "joint": (True, 8.0, True, True)}[route]
    actual_gating = (cfg.get("is_multi_task_balanced"), cfg.get("alpha_loc_loss"),
                     cfg.get("enable_gen_head_lora"), cfg.get("enable_loc_head_lora"))
    if actual_gating != expected_gating:
        raise ValueError("Saved recipe task gating/alpha_loc differs")
    if cfg.get("task_mix_ratio") != {"gen": 0.0, "loc": 1.0, "joint": 0.5}[route]:
        raise ValueError("Saved recipe task sampling ratio differs")
    if cfg.get("benchmark_v2_manifest") != MATRIX["manifest"] or cfg.get("val_maps") != [map_name] or cfg.get("test_maps") != [map_name]:
        raise ValueError("Saved recipe manifest or map selectors differ")
    if cfg.get("is_lora") is not True or cfg.get("llm_train_mode") != "lora" or cfg.get("lora_r") != 32 or cfg.get("lora_alpha") != 64:
        raise ValueError("Saved recipe LoRA architecture differs")
    for key in ("is_gen_aux_loss", "is_repa_loss", "is_loc_repa_loss", "is_aux_loc_em_loss",
                "is_aux_loc_uncertainty_loss"):
        if cfg.get(key) is not False:
            raise ValueError(f"Saved recipe unexpectedly enables {key}")
    if cfg.get("weight_decay") != 0 or cfg.get("finetune_init_strict") is not True:
        raise ValueError("Saved recipe optimizer or strict init differs")
    if family in ("exp37_fs_both", "exp37_fs_both_b1"):
        if (cfg.get("alpha_loc_aux_loss"), cfg.get("alpha_loc_perception_loss")) != (1.0, 0.05):
            raise ValueError("Combined loss weights differ")
        if any(k in cfg for stem in ("alpha_loc_aux", "alpha_loc_perception") for k in (stem+"_schedule_steps", stem+"_schedule_values")):
            raise ValueError("Combined loss schedule must be fixed")
        if cfg.get("loc_perception_use_attention_weight") != family.endswith("_b1"):
            raise ValueError("Combined attention variant differs")
        if (cfg.get("is_loc_aux_loss"), cfg.get("is_aux_loc_combined_em_unc_loss"),
            cfg.get("is_loc_perception_loss")) != (True, True, True):
            raise ValueError("Combined objective flags differ")
        for key in ("aux_loc_combined_num_samples", "aux_loc_combined_candidate_tau",
                    "aux_loc_combined_unc_metric", "aux_loc_combined_unc_tau",
                    "aux_loc_combined_unc_min_weight", "aux_loc_combined_share_loc_noise",
                    "aux_loc_timestep_weight_type", "loc_perception_teacher_type",
                    "loc_perception_feature_source", "loc_perception_loss_type",
                    "loc_perception_timestep_weight"):
            canonical = seen37.read_yaml(ROOT / "csgo_configs" / f"{FAMILIES[family]['source']}.yaml")
            if cfg.get(key) != canonical.get(key):
                raise ValueError(f"Combined objective parameter differs: {key}")
    else:
        if any(cfg.get(key) is not False for key in ("is_loc_aux_loss", "is_aux_loc_combined_em_unc_loss", "is_loc_perception_loss")):
            raise ValueError("Control recipe enables auxiliary objective")
    for stem in ("alpha_loc", "alpha_loc_aux", "alpha_loc_perception"):
        if stem + "_schedule_steps" in cfg or stem + "_schedule_values" in cfg:
            raise ValueError(f"Adaptation has unexpected {stem} schedule")


def snapshot(family: str, map_name: str, shots: int, args: argparse.Namespace,
             *, strict: bool) -> tuple[dict, dict, Path]:
    out = model_dir(family, map_name, shots)
    source, evidence = parent_checkpoint(family, strict=strict)
    cfg = training_recipe(family, map_name, shots, args, source)
    validate_recipe(cfg, family, map_name, shots)
    plan = {"schema_version": 1, "family": family, "map": map_name, "shots": shots,
            "support_seed": SUPPORT_SEED, "seed": SEED, "max_steps": 400,
            "source": evidence, "asset_mode": args.asset_mode or "minimal",
            "config_sha256": hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()}
    route = FAMILIES[family]["route"]
    microbatches = shots // 4 if route == "joint" else 1
    actual_batch = microbatches * 4 if route == "joint" else shots
    plan["effective_batch"] = {"micro_batch": 4 if route == "joint" else shots,
                               "configured_gas": 32 if route == "joint" else 1,
                               "microbatches_per_update": microbatches,
                               "base_samples_per_update": actual_batch,
                               "base_sample_exposures_400_steps": actual_batch * 400}
    return cfg, plan, out


def read_snapshot(family: str, map_name: str, shots: int, args: argparse.Namespace) -> tuple[dict, dict, Path]:
    out = model_dir(family, map_name, shots)
    cfg_path, plan_path = out / "resolved_config.yaml", out / "experiment_plan.json"
    if not cfg_path.is_file() or not plan_path.is_file():
        raise ValueError(f"Missing adaptation snapshot: {out}")
    cfg = seen37.read_yaml(cfg_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_recipe(cfg, family, map_name, shots)
    if any(plan.get(k) != v for k, v in {"family": family, "map": map_name, "shots": shots,
                                           "seed": SEED, "support_seed": SUPPORT_SEED, "max_steps": 400}.items()):
        raise ValueError("Saved plan identity differs")
    if plan.get("config_sha256") != hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest():
        raise ValueError("Saved recipe hash differs")
    source = plan.get("source", {})
    parent, evidence = parent_checkpoint(family, strict=args.execute)
    if source.get("path") != rel(parent) or source.get("step") != 6000 or not source.get("verified"):
        raise ValueError("Saved source is not verified checkpoint-6000")
    if args.execute and source.get("sha256") != evidence["sha256"]:
        raise ValueError("Saved source model hash differs")
    if args.execute and any(source.get(k) != evidence[k] for k in ("recipe_sha256", "plan_sha256")):
        raise ValueError("Saved source recipe or plan hash differs")
    if ((args.asset_mode and args.asset_mode != plan.get("asset_mode")) or
        (args.data_dir and args.data_dir != cfg.get("data_dir"))):
        raise ValueError("Requested asset backend differs from snapshot")
    if args.model_path and args.model_path != cfg.get("model_name_or_path"):
        raise ValueError("Requested model path differs from snapshot")
    if args.pi05_path and str(Path(args.pi05_path).expanduser()) != cfg.get("pi05_pytorch_weight_path"):
        raise ValueError("Requested Pi05 path differs from snapshot")
    return cfg, plan, out


def zero_shot_snapshot(family: str, args: argparse.Namespace) -> tuple[dict, dict, Path, list[tuple[Path, str]]]:
    base = ROOT / "outputs_eval/benchmark_v2" / family / "zero_shot" / "seed_42"
    cfg_path, plan_path = base / "zero_shot_config.yaml", base / "zero_shot_plan.json"
    existing = cfg_path.is_file() or plan_path.is_file()
    if existing:
        if not cfg_path.is_file() or not plan_path.is_file():
            raise ValueError(f"Incomplete zero-shot snapshot: {base}")
        cfg = seen37.read_yaml(cfg_path)
        saved = json.loads(plan_path.read_text(encoding="utf-8"))
        if saved.get("family") != family or saved.get("seed") != 42 or saved.get("source", {}).get("path") != rel(parent_checkpoint(family, strict=False)[0]):
            raise ValueError("Zero-shot snapshot identity/source differs")
        if saved.get("config_sha256") != hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest():
            raise ValueError("Zero-shot snapshot recipe hash differs")
        if ((args.asset_mode and saved.get("asset_mode") != args.asset_mode) or
            (args.data_dir and cfg.get("data_dir") != args.data_dir) or
            (args.model_path and cfg.get("model_name_or_path") != args.model_path) or
            (args.pi05_path and cfg.get("pi05_pytorch_weight_path") != str(Path(args.pi05_path).expanduser()))):
            raise ValueError("Requested zero-shot backend/model differs from snapshot")
        parent, evidence = parent_checkpoint(family, strict=args.execute)
        if args.execute and (saved.get("source", {}).get("sha256") != evidence["sha256"] or
                             saved.get("source", {}).get("recipe_sha256") != evidence["recipe_sha256"] or
                             saved.get("source", {}).get("plan_sha256") != evidence["plan_sha256"]):
            raise ValueError("Zero-shot source hash differs from snapshot")
        return cfg, saved, parent, []
    if args.execute and args.command != "infer":
        raise ValueError(f"Zero-shot metrics/aggregate require prior inference snapshot: {base}")
    parent, evidence = parent_checkpoint(family, strict=args.execute)
    cfg = (seen37.read_yaml(parent.parent.parent / "resolved_config.yaml") if args.execute else
           seen37.read_yaml(ROOT / "csgo_configs" / f"{FAMILIES[family]['source']}.yaml"))
    cfg = copy.deepcopy(cfg)
    cfg["benchmark_v2_asset_manifest"], cfg["data_dir"] = asset_backend(args)
    cfg.pop("benchmark_v2_support_seed", None)
    cfg.pop("benchmark_v2_shots_per_map", None)
    if args.model_path:
        cfg["model_name_or_path"] = args.model_path
    else:
        cfg.setdefault("model_name_or_path", "UniLIP-1B")
    if args.pi05_path:
        cfg["pi05_pytorch_weight_path"] = str(Path(args.pi05_path).expanduser())
    saved = {"schema_version": 1, "family": family, "seed": 42,
             "asset_mode": args.asset_mode or "minimal", "source": evidence,
             "config_sha256": hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()}
    writes = [(cfg_path, yaml.safe_dump(cfg, sort_keys=False)),
              (plan_path, json.dumps(saved, indent=2) + "\n")]
    return cfg, saved, parent, writes


def final_model(out: Path, *, strict: bool) -> Path:
    model = out / "model.safetensors"
    if strict:
        state_path = out / "trainer_state.json"
        if not model.is_file() or not state_path.is_file():
            raise ValueError(f"Missing completed 400-step adaptation: {out}")
        state = seen37.trainer_state(state_path)
        if state["global_step"] != 400 or state["max_steps"] != 400:
            raise ValueError(f"Adaptation did not finish at step 400: {state_path}")
    return model


def train_command(cfg: dict, out: Path, args: argparse.Namespace) -> list[str]:
    route = FAMILIES[args.experiment]["route"]
    batch, gas = (4, 32) if route == "joint" else (args.shots, 1)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1",
           f"--master_port={args.master_port}", "train_csgo.py", "--csgo_config",
           rel(out / "resolved_config.yaml"), "--deepspeed", "deepspeed_scripts/zero0.json",
           "--model_name_or_path", cfg["model_name_or_path"], "--unilip_factor", "10.6",
           "--mllm_hf_path", cfg["mllm_hf_path"], "--version", "internvl", "--data_type", "mix",
           "--csgo_image_folder", cfg["data_dir"], "--output_dir", rel(out),
           "--num_train_epochs", "10", "--per_device_train_batch_size", str(batch),
           "--per_device_eval_batch_size", str(batch), "--gradient_accumulation_steps", str(gas),
           "--max_steps", "400", "--eval_strategy", "no", "--save_strategy", "no",
           "--seed", "42", "--learning_rate", "2e-5", "--weight_decay", "0",
           "--warmup_ratio", "0.003", "--lr_scheduler_type", "cosine_with_min_lr",
           "--model_max_length", "1024", "--logging_steps", "1", "--dataloader_num_workers", "4",
           "--n_query", "256", "--n_und_query", "0", "--report_to", args.report_to,
           "--lora_r", str(cfg["lora_r"]), "--lora_alpha", str(cfg["lora_alpha"]),
           "--optim", "adamw_torch", "--adam_beta1", "0.9", "--adam_beta2", "0.999",
           "--adam_epsilon", "1e-8", "--max_grad_norm", "1.0",
           "--benchmark_v2_support_seed", "0", "--benchmark_v2_shots_per_map", str(args.shots)]
    for key, value in {"mm_use_im_start_end": False, "mm_use_im_patch_token": False,
                       "bf16": True, "tf32": True, "gradient_checkpointing": True,
                       "lazy_preprocess": True, "fix_vit": cfg["fix_vit"],
                       "fix_llm": cfg["fix_llm"], "fix_connect": cfg["fix_connect"],
                       "fix_dit": cfg["fix_dit"]}.items():
        cmd.extend(("--"+key, str(value)))
    if cfg.get("benchmark_v2_asset_manifest"):
        cmd.extend(("--benchmark_v2_asset_manifest", cfg["benchmark_v2_asset_manifest"]))
    return cmd


def selected_tasks(family: str, task: str) -> tuple[str, ...]:
    allowed = ROUTES[FAMILIES[family]["route"]]
    if task == "all":
        return allowed
    if task not in allowed:
        raise ValueError(f"{family} does not support {task}")
    return (task,)


def eval_config(cfg: dict, checkpoint: Path, split: str, maps: tuple[str, ...]) -> dict:
    result = copy.deepcopy(cfg)
    result["ckpt_path"] = rel(checkpoint)
    result["benchmark_v2_split"] = split
    result["test_maps"] = list(maps)
    result["benchmark_v2_allow_map_subset_summary"] = len(maps) == 1
    result["seed"] = 42
    for key in ("is_loc_aux_loss", "is_aux_loc_combined_em_unc_loss", "is_loc_perception_loss"):
        result[key] = False
    return result


def gt_dir(cfg: dict, map_name: str) -> str:
    return seen37.ground_truth_dir(cfg, map_name)


def metric_command(cfg: dict, output: Path, split: str, task: str, map_name: str) -> list[str]:
    ev = MATRIX["evaluation"]
    continuous = task == "gen_continuous"
    cmd = [sys.executable, "benchmark_csgo_v1_conti.py" if continuous else "benchmark_csgo_v1.py",
           "--gt", gt_dir(cfg, map_name), "--pred", rel(output / "gen_imgs" / map_name),
           "--batch_size", "1", "--device", "cuda", "--paired_size", "448",
           "--map_name", map_name, "--metric_profile", "benchmark_v2_core",
           "--benchmark_v2_manifest", MATRIX["manifest"], "--benchmark_v2_split", split]
    if not cfg.get("benchmark_v2_asset_manifest"):
        cmd.extend(("--data_dir", cfg["data_dir"]))
    if cfg.get("benchmark_v2_asset_manifest"):
        cmd.extend(("--benchmark_v2_asset_manifest", cfg["benchmark_v2_asset_manifest"]))
    for key in ("external_loc_repo_root", "external_loc_config_path", "external_loc_checkpoint_path"):
        cmd.extend(("--"+key, ev[key]))
    if continuous:
        for key in ("frame_diff_threshold", "min_track_len", "clip_length", "clip_stride", "fvd_size"):
            cmd.extend(("--"+key, str(ev[key])))
    return cmd


def aggregate_command(family: str, shots: int, task: str, zero_shot: bool,
                      split: str = "query") -> list[str]:
    if split != "query":
        raise ValueError("aggregate covers the four Unseen query maps")
    kind = {"loc": "localization", "gen": "discrete", "gen_continuous": "continuous"}[task]
    benchmark_split = SPLITS[split][task]
    root = ROOT / ("outputs_loc/benchmark_v2" if task == "loc" else "outputs_eval/benchmark_v2")
    if zero_shot:
        input_root = eval_dir(family, None, shots, split, task, True)
        output = input_root / "summary.json"
        return [sys.executable, "scripts/aggregate_csgo_benchmark_v2_metrics.py", "maps",
                "--manifest", MATRIX["manifest"], "--split", benchmark_split,
                "--maps", *MAPS, "--input_root", rel(input_root), "--kind", kind,
                "--output", rel(output)]
    base = root / family / f"shot_{shots}" / "seed_0" / "map_models"
    if task == "loc":
        pattern = root / (family+"_{map}") / f"shot_{shots}" / "seed_0" / "benchmark_csgo_v2_loc.json"
    else:
        subdir = "discrete" if task == "gen" else "continuous"
        stem = "benchmark_csgo_v2_{map}.json" if task == "gen" else "benchmark_csgo_v2_conti_{map}.json"
        pattern = root / (family+"_{map}") / f"shot_{shots}" / "seed_0" / subdir / stem
    output = base / f"benchmark_v2_{kind}_crossmap.json"
    return [sys.executable, "scripts/aggregate_csgo_benchmark_v2_metrics.py", "map-models",
            "--manifest", MATRIX["manifest"], "--split", benchmark_split,
            "--kind", kind, "--input_pattern", rel(pattern), "--output", rel(output)]


def validate_loc_summary(path: Path, checkpoint: Path, maps: tuple[str, ...],
                         split: str, shots: int | None, support_seed: int | None) -> None:
    if not path.is_file():
        raise ValueError(f"Localization metrics are written during inference; missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("sample_count") != len(maps) * 2000 or
        payload.get("maps") != list(maps) or payload.get("seed") != 42 or
        payload.get("split") != split or payload.get("kind") != "localization" or
        payload.get("support_seed") != support_seed or
        payload.get("shots_per_map") != shots or
        absolute(payload.get("checkpoint", "")) != checkpoint or
        absolute(payload.get("manifest", "")) != absolute(MATRIX["manifest"])):
        raise ValueError(f"Localization summary protocol/provenance differs: {path}")
    if len(maps) == 1:
        aggregates._validate_localization_summary_for_map(
            payload, path=path, manifest_path=absolute(MATRIX["manifest"]),
            target_map=maps[0], split=split)
    else:
        aggregates._validate_localization_summary(payload, path=path)


def validate_inference(output: Path, cfg: dict, checkpoint: Path,
                       maps: tuple[str, ...], split: str, task: str,
                       shots: int | None, support_seed: int | None,
                       source: dict) -> dict:
    config_path = output / "evaluation_config.yaml"
    expected_cfg = eval_config(cfg, checkpoint, split, maps)
    if not config_path.is_file() or seen37.read_yaml(config_path) != expected_cfg:
        raise ValueError(f"Evaluation config differs from saved experiment: {config_path}")
    evaluation_plan_path = output / "evaluation_plan.json"
    if not evaluation_plan_path.is_file():
        raise ValueError(f"Missing evaluation weight provenance: {evaluation_plan_path}")
    evaluation_plan = json.loads(evaluation_plan_path.read_text(encoding="utf-8"))
    expected_plan = {"checkpoint": rel(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "config_sha256": sha256(config_path), "split": split, "maps": list(maps),
                     "seed": 42, "support_seed": support_seed, "shots": shots,
                     "asset_manifest": cfg.get("benchmark_v2_asset_manifest")}
    if any(evaluation_plan.get(k) != v for k, v in expected_plan.items()) or evaluation_plan.get("source") != source:
        raise ValueError(f"Evaluation weight/config provenance differs: {evaluation_plan_path}")
    manifest_path = output / "inference_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing inference provenance: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    count = len(maps) * (1280 if task == "gen_continuous" else 2000)
    expected = {"benchmark_v2_split": split, "maps": list(maps), "sample_count": count,
                "seed": 42, "benchmark_v2_support_seed": support_seed,
                "benchmark_v2_shots_per_map": shots}
    if any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Inference split/maps/count/seed/shot differs: {manifest_path}")
    for key, path in (("config_path", config_path), ("benchmark_v2_manifest", absolute(MATRIX["manifest"])),
                      ("checkpoint", checkpoint), ("ckpt_path", checkpoint)):
        if absolute(payload.get(key, "")) != path:
            raise ValueError(f"Inference {key} differs: {manifest_path}")
    map_runner._validate_asset_provenance(
        {"benchmark_v2_asset_manifest": cfg.get("benchmark_v2_asset_manifest")},
        payload, manifest_path)
    return payload


def validate_generation_result(output: Path, map_name: str, task: str,
                               split: str, inference: dict) -> None:
    filename = (f"benchmark_csgo_v2_{map_name}.json" if task == "gen" else
                f"benchmark_csgo_v2_conti_{map_name}.json")
    path = output / filename
    if not path.is_file():
        raise ValueError(f"Missing generation metric: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = payload.get("inference_provenance", {})
    if (payload.get("map_name") != map_name or payload.get("benchmark_v2_split") != split or
        absolute(payload.get("benchmark_v2_manifest", "")) != absolute(MATRIX["manifest"]) or
        absolute(provenance.get("path", "")) != output / "inference_manifest.json" or
        provenance.get("payload") != inference or
        map_runner._common_count(payload) != (1280 if task == "gen_continuous" else 2000)):
        raise ValueError(f"Generation metric does not match inference: {path}")


def plan(args: argparse.Namespace) -> tuple[list[list[str]], list[tuple[Path, str]], dict]:
    validate_matrix()
    family, map_name, shots = args.experiment, args.map, args.shots
    if not args.cuda_device or not args.cuda_device.strip() or "," in args.cuda_device or any(c.isspace() for c in args.cuda_device):
        raise ValueError("--cuda-device must select exactly one device")
    if not 1 <= args.master_port <= 65535:
        raise ValueError("--master-port must be between 1 and 65535")
    if family not in FAMILIES and not (args.command == "validate" and family is None):
        raise ValueError("--experiment must name one exp37_fs family")
    if shots not in SHOTS:
        raise ValueError(f"--shots must be one of {SHOTS}")
    if args.seed != 42 or args.support_seed != 0:
        raise ValueError("Protocol fixes --seed 42 and --support-seed 0")
    if args.zero_shot:
        if args.command == "train" or map_name is not None:
            raise ValueError("Zero-shot runs have no training or --map")
    elif args.command not in ("aggregate", "validate") and map_name not in MAPS:
        raise ValueError(f"--map must be one of {MAPS}")
    if args.command == "aggregate" and map_name is not None:
        raise ValueError("aggregate uses all four maps; omit --map")
    if args.command == "aggregate" and args.split != "query":
        raise ValueError("aggregate applies to query; retention is already a single Seen-10 run")
    if args.command == "train" and args.task != "all":
        raise ValueError("--task applies only to evaluation")
    if args.command == "train" and args.zero_shot:
        raise ValueError("Zero-shot has no adaptation training")
    if args.command != "validate" and args.command != "aggregate":
        selected_tasks(family, args.task)
    commands: list[list[str]] = []
    writes: list[tuple[Path, str]] = []
    details = {"family": family, "map": map_name, "shots": shots, "zero_shot": args.zero_shot,
               "split": args.split, "task": args.task, "execute": args.execute}
    if args.command == "validate":
        for fam in ([family] if args.experiment else FAMILIES):
            for m in MAPS:
                for n in SHOTS:
                    local = copy.copy(args)
                    local.experiment, local.map, local.shots = fam, m, n
                    snapshot(fam, m, n, local, strict=False)
        details["validated_points"] = 80 if not args.experiment else 16
        return commands, writes, details
    if args.command == "train":
        cfg, source_plan, out = snapshot(family, map_name, shots, args, strict=args.execute)
        if args.execute and out.exists() and any(out.iterdir()):
            raise ValueError(f"Refusing to overwrite an existing adaptation directory: {out}")
        commands.append(train_command(cfg, out, args))
        writes.extend(((out / "resolved_config.yaml", yaml.safe_dump(cfg, sort_keys=False)),
                       (out / "experiment_plan.json", json.dumps(source_plan, indent=2)+"\n")))
        details["source"] = source_plan["source"]
        return commands, writes, details
    if args.command == "aggregate":
        if args.execute:
            if args.zero_shot:
                cfg, zero_plan, checkpoint, _ = zero_shot_snapshot(family, args)
                if "loc" in selected_tasks(family, args.task):
                    validate_inference(eval_dir(family, None, shots, "query", "loc", True),
                                       cfg, checkpoint, MAPS, SPLITS["query"]["loc"],
                                       "loc", None, None, zero_plan["source"])
                    validate_loc_summary(eval_dir(family, None, shots, "query", "loc", True) /
                                         "benchmark_csgo_v2_loc.json", checkpoint, MAPS,
                                         SPLITS["query"]["loc"], None, None)
                for task in selected_tasks(family, args.task):
                    if task == "loc":
                        continue
                    output = eval_dir(family, None, shots, "query", task, True)
                    split = SPLITS["query"][task]
                    inference = validate_inference(output, cfg, checkpoint, MAPS, split, task,
                                                   None, None, zero_plan["source"])
                    for m in MAPS:
                        validate_generation_result(output, m, task, split, inference)
            else:
                for m in MAPS:
                    cfg, source_plan, out = read_snapshot(family, m, shots, args)
                    checkpoint = final_model(out, strict=True)
                    if "loc" in selected_tasks(family, args.task):
                        validate_inference(eval_dir(family, m, shots, "query", "loc", False),
                                           cfg, checkpoint, (m,), SPLITS["query"]["loc"],
                                           "loc", shots, 0, source_plan["source"])
                        validate_loc_summary(eval_dir(family, m, shots, "query", "loc", False) /
                                             "benchmark_csgo_v2_loc.json", checkpoint, (m,),
                                             SPLITS["query"]["loc"], shots, 0)
                    for task in selected_tasks(family, args.task):
                        if task == "loc":
                            continue
                        output = eval_dir(family, m, shots, "query", task, False)
                        split = SPLITS["query"][task]
                        inference = validate_inference(output, cfg, checkpoint, (m,), split, task,
                                                       shots, 0, source_plan["source"])
                        validate_generation_result(output, m, task, split, inference)
        for task in selected_tasks(family, args.task):
            if args.zero_shot and task == "loc":
                details.setdefault("localization_summaries", []).append(
                    rel(eval_dir(family, None, shots, "query", "loc", True) / "benchmark_csgo_v2_loc.json"))
            else:
                commands.append(aggregate_command(family, shots, task, args.zero_shot))
        return commands, writes, details
    if args.zero_shot:
        cfg, zero_plan, checkpoint, new_writes = zero_shot_snapshot(family, args)
        if args.command == "infer":
            writes.extend(new_writes)
        maps = MAPS if args.split == "query" else SEEN_MAPS
        details["source"] = zero_plan["source"]
    else:
        if args.execute or (model_dir(family, map_name, shots) / "resolved_config.yaml").is_file():
            cfg, source_plan, out = read_snapshot(family, map_name, shots, args)
            checkpoint = final_model(out, strict=args.execute)
        else:
            cfg, source_plan, out = snapshot(family, map_name, shots, args, strict=False)
            checkpoint = final_model(out, strict=False)
        maps = (map_name,) if args.split == "query" else SEEN_MAPS
        details["source"] = source_plan["source"]
    for task in selected_tasks(family, args.task):
        split = SPLITS[args.split][task]
        output = eval_dir(family, map_name, shots, args.split, task, args.zero_shot)
        if args.command == "infer":
            ecfg = eval_config(cfg, checkpoint, split, maps)
            path = output / "evaluation_config.yaml"
            if args.execute and path.is_file() and seen37.read_yaml(path) != ecfg:
                raise ValueError(f"Existing evaluation config differs: {path}")
            config_content = yaml.safe_dump(ecfg, sort_keys=False)
            evaluation_plan = {"schema_version": 1, "checkpoint": rel(checkpoint),
                               "checkpoint_sha256": sha256(checkpoint) if args.execute else None,
                               "config_sha256": hashlib.sha256(config_content.encode()).hexdigest(),
                               "split": split, "maps": list(maps), "seed": 42,
                               "support_seed": None if args.zero_shot else 0,
                               "shots": None if args.zero_shot else shots,
                               "asset_manifest": cfg.get("benchmark_v2_asset_manifest"),
                               "source": details["source"]}
            ep = output / "evaluation_plan.json"
            if args.execute and ep.is_file() and json.loads(ep.read_text(encoding="utf-8")) != evaluation_plan:
                raise ValueError(f"Existing evaluation weight provenance differs: {ep}")
            writes.extend(((path, config_content), (ep, json.dumps(evaluation_plan, indent=2) + "\n")))
            cmd = [sys.executable, "eval_csgo_loc.py" if task == "loc" else "eval_csgo.py",
                   "--csgo_config", rel(path), "--output_dir", rel(output),
                   "--ckpt_path", rel(checkpoint), "--seed", "42", "--benchmark_v2_split", split,
                   "--benchmark_v2_maps", *maps]
            if not args.zero_shot:
                cmd.extend(("--benchmark_v2_support_seed", "0", "--benchmark_v2_shots_per_map", str(shots)))
            if cfg.get("benchmark_v2_asset_manifest"):
                cmd.extend(("--benchmark_v2_asset_manifest", cfg["benchmark_v2_asset_manifest"]))
            commands.append(cmd)
        elif task == "loc":
            if args.execute:
                validate_inference(output, cfg, checkpoint, maps, split, task,
                                   None if args.zero_shot else shots,
                                   None if args.zero_shot else 0, details["source"])
                validate_loc_summary(output / "benchmark_csgo_v2_loc.json", checkpoint, maps,
                                     split, None if args.zero_shot else shots,
                                     None if args.zero_shot else 0)
            details.setdefault("localization_summaries", []).append(rel(output / "benchmark_csgo_v2_loc.json"))
        else:
            if args.execute:
                validate_inference(output, cfg, checkpoint, maps, split, task,
                                   None if args.zero_shot else shots,
                                   None if args.zero_shot else 0,
                                   details["source"])
            for m in maps:
                commands.append(metric_command(cfg, output, split, task, m))
            if args.zero_shot and args.split == "query":
                commands.append(aggregate_command(family, shots, task, True))
            elif args.split == "retention":
                kind = "discrete" if task == "gen" else "continuous"
                commands.append([sys.executable, "scripts/aggregate_csgo_benchmark_v2_metrics.py", "maps",
                                 "--manifest", MATRIX["manifest"], "--split", split,
                                 "--input_root", rel(output), "--kind", kind,
                                 "--output", rel(output / "summary.json")])
    return commands, writes, details


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("validate", "train", "infer", "metrics", "aggregate"))
    p.add_argument("--experiment", choices=tuple(FAMILIES))
    p.add_argument("--map", choices=MAPS)
    p.add_argument("--shots", type=int, default=100)
    p.add_argument("--zero-shot", action="store_true")
    p.add_argument("--split", choices=("query", "retention"), default="query")
    p.add_argument("--task", choices=("all", "loc", "gen", "gen_continuous"), default="all")
    p.add_argument("--asset-mode", choices=("minimal", "full"))
    p.add_argument("--data-dir")
    p.add_argument("--model-path")
    p.add_argument("--pi05-path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--support-seed", type=int, default=0)
    p.add_argument("--master-port", type=int, default=29638)
    p.add_argument("--cuda-device", "--cuda-devices", dest="cuda_device", default="0")
    p.add_argument("--report-to", default="none")
    p.add_argument("--execute", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command != "validate" and not args.experiment:
            raise ValueError("--experiment is required")
        commands, writes, details = plan(args)
        print(json.dumps({"mode": "execute" if args.execute else "dry-run", "details": details,
                          "writes": [rel(path) for path, _ in writes],
                          "commands": [shlex.join(c) for c in commands]}, indent=2))
        if args.execute:
            for path, content in writes:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.cuda_device}
            for command in commands:
                subprocess.run(command, cwd=ROOT, env=env, check=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
