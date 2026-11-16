#!/usr/bin/env python3
"""Plan exp32_loc@19500 Unseen-4 localization; run only with --execute."""
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
from csgo_datasets.inference_batching import positive_batch_size, execution_contract  # noqa: E402

MATRIX_PATH = ROOT / "csgo_configs/benchmark_v2_exp32_loc_unseen4.yaml"
MATRIX = yaml.safe_load(MATRIX_PATH.read_text(encoding="utf-8"))
FAMILY = "exp32_fs_loc_19500"
SOURCE = ROOT / "outputs/csgo_1b/exp32_loc/checkpoint-19500/model.safetensors"
SOURCE_RECIPE = ROOT / "csgo_configs/exp32_loc.yaml"
BASE_RECIPE = ROOT / "csgo_configs/exp37_s2_loc.yaml"
MAPS = ("cs_office", "de_golden", "de_palacio", "de_vertigo")
SHOTS = (100, 50, 20, 10)
SPLIT = "crossmap_query_test"
LOW_LR = legacy.LR_LOW
HIGH_LR = legacy.LR_HIGH


def rel(path: Path) -> str:
    return legacy.rel(path)


def digest(path: Path) -> str:
    return legacy.sha256(path)


def yaml_text(config: dict) -> str:
    return yaml.safe_dump(config, sort_keys=False)


def json_text(value: dict) -> str:
    return json.dumps(value, indent=2) + "\n"


def hash_config(config: dict) -> str:
    return hashlib.sha256(yaml.safe_dump(config, sort_keys=True).encode()).hexdigest()


def validate_matrix() -> None:
    expected = {"schema_version": 1, "family": FAMILY,
                "source": "outputs/csgo_1b/exp32_loc/checkpoint-19500/model.safetensors",
                "source_recipe": "csgo_configs/exp32_loc.yaml",
                "base_recipe": "csgo_configs/exp37_s2_loc.yaml",
                "manifest": legacy.MATRIX["manifest"],
                "minimal_asset_manifest": legacy.MATRIX["minimal_asset_manifest"],
                "maps": list(MAPS), "shots": list(SHOTS), "support_seed": 0,
                "seed": 42, "max_steps": 400, "test_loc_batch_size": 16,
                "training": {"low_lr": 2e-5, "high_lr": 1e-4,
                             "optimizer": "adamw_torch", "adam_beta1": 0.9,
                             "adam_beta2": 0.999, "adam_epsilon": 1e-8,
                             "weight_decay": 0.0, "max_grad_norm": 1.0,
                             "scheduler": "cosine_with_min_lr", "min_lr_rate": 0.1,
                             "warmup_ratio": 0.003, "save_strategy": "no",
                             "eval_strategy": "no"}}
    if MATRIX != expected:
        raise ValueError("exp32_loc@19500 matrix differs from fixed protocol")


def source_checkpoint(*, strict: bool) -> tuple[Path, dict]:
    evidence = {"path": rel(SOURCE), "step": 19500, "verified": False}
    if not strict:
        return SOURCE, evidence
    recipe = seen37.read_yaml(SOURCE_RECIPE)
    template = seen37.read_yaml(BASE_RECIPE)
    identity = ("is_multi_task", "is_multi_task_balanced", "task_mix_ratio",
                "alpha_loc_loss", "is_lora", "llm_train_mode", "lora_r",
                "lora_alpha", "lora_dropout", "enable_language_model_lora",
                "enable_gen_head_lora", "enable_loc_head_lora", "freeze_inactive_head",
                "fix_vit", "fix_llm", "fix_connect", "fix_dit",
                "is_loc_aux_loss", "is_aux_loc_combined_em_unc_loss",
                "is_loc_perception_loss", "is_gen_aux_loss", "is_repa_loss",
                "is_loc_repa_loss", "benchmark_v2_manifest")
    if any(recipe.get(k) != template.get(k) for k in identity):
        raise ValueError("Source exp32_loc architecture differs from loc template")
    if recipe.get("benchmark_v2_split") != "seen_train" or recipe.get("train_maps") != list(legacy.SEEN_MAPS):
        raise ValueError("Source recipe is not Seen-10 localization")
    config_path = SOURCE.parent / "config.json"
    state_path = SOURCE.parent / "trainer_state.json"
    for path in (SOURCE, config_path, state_path):
        if not path.is_file():
            raise ValueError(f"Missing exact checkpoint-19500 source file: {path}")
    state = seen37.trainer_state(state_path)
    if state.get("global_step") != 19500:
        raise ValueError(f"Source global_step is not 19500: {state_path}")
    model = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {"model_type": "unified_unilip", "is_lora": True,
                "llm_train_mode": "lora", "enable_language_model_lora": True,
                "enable_gen_head_lora": False, "enable_loc_head_lora": True,
                "freeze_inactive_head": True, "is_loc_aux_loss": False,
                "is_loc_perception_loss": False, "img_size": 224,
                "action_dim": 5, "mllm_hf_path": "OpenGVLab/InternVL3-1B-hf"}
    if any(model.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Source checkpoint config is not loc-only: {config_path}")
    header = seen37.tensor_header(SOURCE, "exp37_s2_loc")
    evidence.update(verified=True, sha256=digest(SOURCE),
                    trainer_state_sha256=digest(state_path),
                    model_config_sha256=digest(config_path),
                    source_recipe_sha256=digest(SOURCE_RECIPE),
                    base_recipe_sha256=digest(BASE_RECIPE),
                    header=header,
                    trainer_state={"global_step": state["global_step"],
                                   "max_steps": state.get("max_steps")})
    return SOURCE, evidence


def asset_backend(args: argparse.Namespace) -> tuple[str | None, str]:
    if args.asset_mode == "full":
        if not args.data_dir:
            raise ValueError("--asset-mode full requires --data-dir")
        return None, args.data_dir
    if args.data_dir:
        raise ValueError("--data-dir requires --asset-mode full")
    return MATRIX["minimal_asset_manifest"], "data/preprocessed_data"


def recipe(args: argparse.Namespace, map_name: str | None, shots: int | None,
           *, zero_shot: bool) -> dict:
    config = copy.deepcopy(seen37.read_yaml(BASE_RECIPE))
    for key in tuple(config):
        if (key.startswith(("resume_", "training_stop_", "training_checkpoint_", "finetune_init_"))
            or key in ("ckpt_path", "warmup_steps", "alpha_loc_schedule_steps",
                       "alpha_loc_schedule_values", "global_batch_size", "csgo_global_batch_size",
                       "gradient_accumulation_steps", "base_init_ckpt_path", "gen_init_ckpt_path",
                       "loc_init_ckpt_path", "pretrain_path")):
            config.pop(key)
    config.pop("exp37_experiment", None)
    config["exp32_loc_unseen4_experiment"] = FAMILY
    config["benchmark_v2_asset_manifest"], config["data_dir"] = asset_backend(args)
    config["model_name_or_path"] = args.model_path or "UniLIP-1B"
    if args.pi05_path:
        config["pi05_pytorch_weight_path"] = str(Path(args.pi05_path).expanduser())
    config["seed"] = 42
    config["is_multi_task_balanced"] = False
    config["task_mix_ratio"] = 1.0
    config["alpha_loc_loss"] = 1.0
    config["enable_gen_head_lora"] = False
    config["enable_loc_head_lora"] = True
    config["freeze_inactive_head"] = True
    for key in ("is_loc_aux_loss", "is_aux_loc_combined_em_unc_loss",
                "is_loc_perception_loss", "is_gen_aux_loss", "is_repa_loss",
                "is_loc_repa_loss", "is_aux_loc_em_loss", "is_aux_loc_uncertainty_loss"):
        config[key] = False
    for key in LOW_LR:
        config[key] = 2e-5
    for key in HIGH_LR:
        config[key] = 1e-4
    config.update(save_final_model_at_output_root=True, disable_lora_dropout=False,
                  warmup_ratio=0.003, lr_scheduler_type="cosine_with_min_lr",
                  lr_scheduler_kwargs={"min_lr_rate": 0.1}, weight_decay=0.0,
                  csgo_loss_per_microbatch_mean=True, csgo_forward_sampler_epoch=True)
    if zero_shot:
        config.pop("benchmark_v2_support_seed", None)
        config.pop("benchmark_v2_shots_per_map", None)
    else:
        config.update(benchmark_v2_split="crossmap_support",
                      benchmark_v2_support_seed=0, benchmark_v2_shots_per_map=shots,
                      train_maps=[map_name], val_maps=[map_name], test_maps=[map_name],
                      finetune_init_ckpt_path=rel(SOURCE), finetune_init_strict=True)
    validate_recipe(config, map_name, shots, zero_shot=zero_shot)
    return config


def validate_recipe(config: dict, map_name: str | None, shots: int | None,
                    *, zero_shot: bool) -> None:
    template = seen37.read_yaml(BASE_RECIPE)
    fixed = {"exp32_loc_unseen4_experiment": FAMILY, "seed": 42,
             "is_multi_task_balanced": False, "task_mix_ratio": 1.0,
             "alpha_loc_loss": 1.0, "enable_gen_head_lora": False,
             "enable_loc_head_lora": True, "freeze_inactive_head": True,
             "is_lora": True, "llm_train_mode": "lora", "lora_r": 32,
             "lora_alpha": 64, "lora_dropout": template["lora_dropout"],
             "warmup_ratio": 0.003, "lr_scheduler_type": "cosine_with_min_lr",
             "lr_scheduler_kwargs": {"min_lr_rate": 0.1}, "weight_decay": 0.0,
             "csgo_loss_per_microbatch_mean": True, "csgo_forward_sampler_epoch": True,
             "save_final_model_at_output_root": True, "disable_lora_dropout": False,
             "benchmark_v2_manifest": MATRIX["manifest"]}
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError("Saved localization recipe protocol differs")
    for key in ("is_loc_aux_loss", "is_aux_loc_combined_em_unc_loss",
                "is_loc_perception_loss", "is_gen_aux_loss", "is_repa_loss",
                "is_loc_repa_loss", "is_aux_loc_em_loss", "is_aux_loc_uncertainty_loss"):
        if config.get(key) is not False:
            raise ValueError(f"Saved recipe enables {key}")
    if any(config.get(k) != 2e-5 for k in LOW_LR) or any(config.get(k) != 1e-4 for k in HIGH_LR):
        raise ValueError("Saved localization recipe learning rates differ")
    for key in ("fix_vit", "fix_llm", "fix_connect", "fix_dit", "use_pi05_action_dit"):
        if config.get(key) != template.get(key):
            raise ValueError(f"Saved localization recipe changes {key}")
    if any(k in config for k in ("warmup_steps", "gradient_accumulation_steps", "global_batch_size",
                                  "csgo_global_batch_size", "base_init_ckpt_path", "gen_init_ckpt_path",
                                  "loc_init_ckpt_path", "pretrain_path", "alpha_loc_schedule_steps",
                                  "alpha_loc_schedule_values", "ckpt_path")):
        raise ValueError("Saved recipe contains inherited training/resume fields")
    if zero_shot:
        if any(k in config for k in ("benchmark_v2_support_seed", "benchmark_v2_shots_per_map",
                                      "finetune_init_ckpt_path", "finetune_init_strict")):
            raise ValueError("Zero-shot recipe has adaptation fields")
    elif (map_name not in MAPS or shots not in SHOTS or
          any(config.get(k) != [map_name] for k in ("train_maps", "val_maps", "test_maps")) or
          config.get("benchmark_v2_split") != "crossmap_support" or
          config.get("benchmark_v2_support_seed") != 0 or
          config.get("benchmark_v2_shots_per_map") != shots or
          config.get("finetune_init_ckpt_path") != rel(SOURCE) or
          config.get("finetune_init_strict") is not True):
        raise ValueError("Saved adaptation recipe map/shot/source differs")


def model_dir(map_name: str, shots: int) -> Path:
    return ROOT / "outputs/csgo_1b" / f"{FAMILY}_{map_name}" / f"shot_{shots}" / "seed_0"


def eval_dir(map_name: str | None, shots: int, zero_shot: bool, batch: int) -> Path:
    return legacy.eval_dir(FAMILY, map_name, shots, "query", "loc", zero_shot, batch)


def zero_dir() -> Path:
    return ROOT / "outputs_loc/benchmark_v2" / FAMILY / "zero_shot" / "seed_42"


def plan_identity(config: dict, source: dict, args: argparse.Namespace,
                  map_name: str | None, shots: int | None) -> dict:
    result = {"schema_version": 1, "family": FAMILY, "map": map_name,
              "shots": shots, "seed": 42, "support_seed": None if map_name is None else 0,
              "max_steps": None if map_name is None else 400,
              "asset_mode": args.asset_mode or "minimal", "source": source,
              "config_sha256": hash_config(config)}
    if map_name is not None:
        result["effective_batch"] = {"micro_batch": shots, "configured_gas": 1,
                                     "samples_per_update": shots}
    return result


def _check_overrides(args: argparse.Namespace, config: dict, saved: dict) -> None:
    if args.asset_mode and args.asset_mode != saved.get("asset_mode"):
        raise ValueError("Requested asset mode differs from snapshot")
    if args.data_dir and args.data_dir != config.get("data_dir"):
        raise ValueError("Requested data directory differs from snapshot")
    if args.model_path and args.model_path != config.get("model_name_or_path"):
        raise ValueError("Requested model path differs from snapshot")
    if args.pi05_path and str(Path(args.pi05_path).expanduser()) != config.get("pi05_pytorch_weight_path"):
        raise ValueError("Requested Pi05 path differs from snapshot")


def read_snapshot(args: argparse.Namespace, map_name: str | None,
                  shots: int | None, *, strict: bool) -> tuple[dict, dict, Path]:
    base = zero_dir() if map_name is None else model_dir(map_name, shots)
    config_path = base / ("zero_shot_config.yaml" if map_name is None else "resolved_config.yaml")
    plan_path = base / ("zero_shot_plan.json" if map_name is None else "experiment_plan.json")
    if not config_path.is_file() or not plan_path.is_file():
        raise ValueError(f"Missing experiment snapshot: {base}")
    config = seen37.read_yaml(config_path)
    saved = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_recipe(config, map_name, shots, zero_shot=map_name is None)
    expected = {"family": FAMILY, "map": map_name, "shots": shots, "seed": 42,
                "support_seed": None if map_name is None else 0,
                "max_steps": None if map_name is None else 400}
    if any(saved.get(k) != v for k, v in expected.items()) or saved.get("config_sha256") != hash_config(config):
        raise ValueError("Saved plan identity or recipe hash differs")
    mode = saved.get("asset_mode")
    if mode not in ("minimal", "full"):
        raise ValueError("Saved asset mode differs from fixed protocol")
    if mode == "minimal" and (config.get("benchmark_v2_asset_manifest") != MATRIX["minimal_asset_manifest"]
                              or config.get("data_dir") != "data/preprocessed_data"):
        raise ValueError("Saved minimal asset backend differs")
    if mode == "full" and (config.get("benchmark_v2_asset_manifest") is not None
                           or not config.get("data_dir")):
        raise ValueError("Saved full asset backend differs")
    source = saved.get("source", {})
    if source.get("path") != rel(SOURCE) or source.get("step") != 19500 or source.get("verified") is not True:
        raise ValueError("Saved source is not verified exp32_loc checkpoint-19500")
    for key in ("sha256", "trainer_state_sha256", "model_config_sha256",
                "source_recipe_sha256", "base_recipe_sha256"):
        if not isinstance(source.get(key), str) or len(source[key]) != 64:
            raise ValueError(f"Saved source lacks {key}")
    if (source["source_recipe_sha256"] != digest(SOURCE_RECIPE)
            or source["base_recipe_sha256"] != digest(BASE_RECIPE)
            or not isinstance(source.get("header"), dict)
            or source.get("trainer_state", {}).get("global_step") != 19500):
        raise ValueError("Saved source recipe/header/state identity differs")
    if map_name is not None and saved.get("effective_batch") != {"micro_batch": shots, "configured_gas": 1,
                                                                "samples_per_update": shots}:
        raise ValueError("Saved effective batch differs")
    _check_overrides(args, config, saved)
    if strict:
        _, actual = source_checkpoint(strict=True)
        if source != actual:
            raise ValueError("Saved source checkpoint/recipe hash or state differs")
    return config, saved, base


def final_model(base: Path, *, strict: bool) -> Path:
    path = base / "model.safetensors"
    if strict:
        state_path = base / "trainer_state.json"
        if not path.is_file() or not state_path.is_file():
            raise ValueError(f"Missing completed 400-step adaptation: {base}")
        state = seen37.trainer_state(state_path)
        if state.get("global_step") != 400 or state.get("max_steps") != 400:
            raise ValueError(f"Adaptation did not complete 400 steps: {state_path}")
    return path


def train_command(config: dict, base: Path, shots: int, args: argparse.Namespace) -> list[str]:
    inherited = copy.copy(args)
    inherited.experiment = "exp37_fs_loc"
    inherited.shots = shots
    command = legacy.train_command(config, base, inherited)
    if args.master_port == 0:
        command[command.index("--master_port=0")] = "--standalone"
        command.insert(command.index("--standalone") + 1, "--nnodes=1")
    return command


def infer_command(config: dict, checkpoint: Path, output: Path,
                  maps: tuple[str, ...], batch: int, shots: int | None) -> list[str]:
    command = [sys.executable, "eval_csgo_loc.py", "--csgo_config", rel(output / "evaluation_config.yaml"),
               "--output_dir", rel(output), "--ckpt_path", rel(checkpoint), "--seed", "42",
               "--benchmark_v2_split", SPLIT, "--benchmark_v2_maps", *maps,
               "--test-loc-batch-size", str(batch)]
    if shots is not None:
        command += ["--benchmark_v2_support_seed", "0", "--benchmark_v2_shots_per_map", str(shots)]
    if config.get("benchmark_v2_asset_manifest"):
        command += ["--benchmark_v2_asset_manifest", config["benchmark_v2_asset_manifest"]]
    return command


def status(args: argparse.Namespace) -> dict:
    points = []
    for map_name in MAPS:
        for shots in SHOTS:
            base = model_dir(map_name, shots)
            config = base / "resolved_config.yaml"
            saved = base / "experiment_plan.json"
            model = base / "model.safetensors"
            state = base / "trainer_state.json"
            entry = {"map": map_name, "shots": shots, "snapshot": config.is_file() and saved.is_file(),
                     "model": model.is_file(), "completed_400": False}
            if state.is_file():
                try:
                    trainer = seen37.trainer_state(state)
                    entry["completed_400"] = model.is_file() and trainer.get("global_step") == 400 and trainer.get("max_steps") == 400
                except (ValueError, OSError, KeyError, json.JSONDecodeError):
                    pass
            points.append(entry)
    return {"source_files": {rel(path): path.is_file() for path in
                             (SOURCE, SOURCE.parent / "config.json", SOURCE.parent / "trainer_state.json")},
            "points": points}


def plan(args: argparse.Namespace) -> tuple[list[list[str]], list[tuple[Path, str]], dict]:
    validate_matrix()
    if args.experiment != FAMILY or args.seed != 42 or args.support_seed != 0:
        raise ValueError("Experiment/seed/support seed differs from fixed protocol")
    if args.report_to != "none":
        raise ValueError("Protocol fixes --report-to none")
    if args.zero_shot and args.shots is not None:
        raise ValueError("Zero-shot has no shots; omit --shots")
    if args.shots is None:
        args.shots = 100
    if args.shots not in SHOTS or args.task not in ("all", "loc"):
        raise ValueError("Invalid shots or localization task")
    if args.master_port < 0 or args.master_port > 65535:
        raise ValueError("--master-port must be 0 or 1..65535")
    if not args.cuda_device or "," in args.cuda_device or any(c.isspace() for c in args.cuda_device):
        raise ValueError("--cuda-device must select exactly one device")
    if args.zero_shot and (args.map is not None or args.command == "train"):
        raise ValueError("Zero-shot has no map, shots or training")
    if args.command == "aggregate" and args.map is not None:
        raise ValueError("Aggregate uses all four maps; omit --map")
    if args.command in ("train", "infer", "metrics") and not args.zero_shot and args.map not in MAPS:
        raise ValueError("Few-shot train/infer/metrics require --map")
    if args.command in ("validate", "source-check", "status") and args.execute:
        raise ValueError(f"{args.command} is read-only; omit --execute")
    commands: list[list[str]] = []
    writes: list[tuple[Path, str]] = []
    details = {"family": FAMILY, "zero_shot": args.zero_shot, "map": args.map,
               "shots": None if args.zero_shot else args.shots, "batch_size": args.test_loc_batch_size}
    if args.command == "validate":
        for map_name in MAPS:
            for shots in SHOTS:
                recipe(args, map_name, shots, zero_shot=False)
        recipe(args, None, None, zero_shot=True)
        details["validated_points"] = 16
        return commands, writes, details
    if args.command == "status":
        details.update(status(args))
        return commands, writes, details
    if args.command == "source-check":
        details["source"] = source_checkpoint(strict=True)[1]
        return commands, writes, details
    if args.command == "train":
        _, evidence = source_checkpoint(strict=args.execute)
        config = recipe(args, args.map, args.shots, zero_shot=False)
        base = model_dir(args.map, args.shots)
        if args.execute and base.exists() and any(base.iterdir()):
            raise ValueError(f"Refusing to overwrite adaptation directory: {base}")
        saved = plan_identity(config, evidence, args, args.map, args.shots)
        writes = [(base / "resolved_config.yaml", yaml_text(config)),
                  (base / "experiment_plan.json", json_text(saved))]
        commands = [train_command(config, base, args.shots, args)]
        details["source"] = evidence
        return commands, writes, details
    map_name = None if args.zero_shot else args.map
    shots = None if args.zero_shot else args.shots
    if args.zero_shot and args.command == "infer" and not (zero_dir() / "zero_shot_plan.json").is_file():
        _, evidence = source_checkpoint(strict=args.execute)
        config = recipe(args, None, None, zero_shot=True)
        saved = plan_identity(config, evidence, args, None, None)
        zero_base = zero_dir()
        writes += [(zero_base / "zero_shot_config.yaml", yaml_text(config)),
                   (zero_base / "zero_shot_plan.json", json_text(saved))]
    elif args.command != "aggregate":
        snapshot_base = zero_dir() if args.zero_shot else model_dir(args.map, args.shots)
        snapshot_name = "zero_shot_plan.json" if args.zero_shot else "experiment_plan.json"
        if not args.execute and not (snapshot_base / snapshot_name).is_file():
            _, evidence = source_checkpoint(strict=False)
            config = recipe(args, map_name, shots, zero_shot=args.zero_shot)
            saved = plan_identity(config, evidence, args, map_name, shots)
        else:
            config, saved, _ = read_snapshot(args, map_name, shots, strict=args.execute)
    if args.command == "aggregate":
        if args.zero_shot:
            if args.execute:
                config, saved, _ = read_snapshot(args, None, None, strict=True)
            else:
                config = recipe(args, None, None, zero_shot=True)
                saved = plan_identity(config, source_checkpoint(strict=False)[1], args, None, None)
            checkpoint = SOURCE
            output = eval_dir(None, args.shots, True, args.test_loc_batch_size)
            if args.execute:
                legacy.validate_inference(output, config, checkpoint, MAPS, SPLIT, "loc", None, None,
                                          saved["source"], args.test_loc_batch_size)
                legacy.validate_loc_summary(output / "benchmark_csgo_v2_loc.json", checkpoint,
                                            MAPS, SPLIT, None, None)
            details["summary"] = rel(output / "benchmark_csgo_v2_loc.json")
            return commands, writes, details
        if args.execute:
            for name in MAPS:
                local_config, local_saved, base = read_snapshot(args, name, args.shots, strict=True)
                checkpoint = final_model(base, strict=True)
                output = eval_dir(name, args.shots, False, args.test_loc_batch_size)
                legacy.validate_inference(output, local_config, checkpoint, (name,), SPLIT, "loc",
                                          args.shots, 0, local_saved["source"], args.test_loc_batch_size)
                legacy.validate_loc_summary(output / "benchmark_csgo_v2_loc.json", checkpoint,
                                            (name,), SPLIT, args.shots, 0)
        commands = [legacy.aggregate_command(FAMILY, args.shots, "loc", False,
                                             batch_size=args.test_loc_batch_size)]
        return commands, writes, details
    checkpoint = SOURCE if args.zero_shot else final_model(model_dir(args.map, args.shots), strict=args.execute)
    maps = MAPS if args.zero_shot else (args.map,)
    output = eval_dir(map_name, args.shots, args.zero_shot, args.test_loc_batch_size)
    details["source"] = saved["source"]
    if args.command == "metrics":
        # eval_csgo_loc.py writes localization metrics with inference; validation is CPU only.
        if args.execute:
            legacy.validate_inference(output, config, checkpoint, maps, SPLIT, "loc", shots,
                                      None if args.zero_shot else 0, saved["source"], args.test_loc_batch_size)
            legacy.validate_loc_summary(output / "benchmark_csgo_v2_loc.json", checkpoint,
                                        maps, SPLIT, shots, None if args.zero_shot else 0)
        details["summary"] = rel(output / "benchmark_csgo_v2_loc.json")
        return commands, writes, details
    if args.execute and output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite inference directory: {output}")
    ecfg = legacy.eval_config(config, checkpoint, SPLIT, maps, "loc", args.test_loc_batch_size)
    ecfg_path = output / "evaluation_config.yaml"
    ecfg_body = yaml_text(ecfg)
    eval_plan = {"schema_version": 1, "checkpoint": rel(checkpoint),
                 "checkpoint_sha256": digest(checkpoint) if args.execute else None,
                 "config_sha256": hashlib.sha256(ecfg_body.encode()).hexdigest(),
                 "split": SPLIT, "maps": list(maps), "seed": 42,
                 "support_seed": None if args.zero_shot else 0, "shots": shots,
                 "asset_manifest": config.get("benchmark_v2_asset_manifest"),
                 "source": saved["source"],
                 "inference_execution": execution_contract("loc", args.test_loc_batch_size)}
    writes += [(ecfg_path, ecfg_body), (output / "evaluation_plan.json", json_text(eval_plan))]
    commands = [infer_command(config, checkpoint, output, maps, args.test_loc_batch_size, shots)]
    return commands, writes, details


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("validate", "source-check", "status", "train", "infer", "metrics", "aggregate"))
    p.add_argument("--experiment", choices=(FAMILY,), default=FAMILY)
    p.add_argument("--zero-shot", action="store_true")
    p.add_argument("--map", choices=MAPS)
    p.add_argument("--shots", type=int)
    p.add_argument("--task", choices=("all", "loc"), default="all")
    p.add_argument("--test-loc-batch-size", type=positive_batch_size, default=16)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--master-port", type=int, default=0)
    p.add_argument("--asset-mode", choices=("minimal", "full"))
    p.add_argument("--data-dir")
    p.add_argument("--model-path")
    p.add_argument("--pi05-path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--support-seed", type=int, default=0)
    p.add_argument("--report-to", default="none")
    p.add_argument("--execute", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
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
            if args.command == "train":
                final_model(model_dir(args.map, args.shots), strict=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError,
            json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
