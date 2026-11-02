"""Rebuild an Exp32 training model and dataset for read-only gradient probes.

The experiment plan supplies the exact training CLI; train_csgo.train supplies
the model/LoRA/freeze and dataset construction. No Trainer is instantiated.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class GradientProbeContext:
    model: Any
    dataset: Any
    collator: Any
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _saved_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _gradient_checkpointing_snapshot(model: Any) -> dict[str, Any]:
    base = model.get_model() if hasattr(model, "get_model") else model
    result = {}
    for name in ("vision_tower", "language_model", "llm_connector", "dit", "action_dit"):
        component = getattr(base, name, None)
        if component is None:
            continue
        flags = []
        for module in component.modules():
            for attribute in ("gradient_checkpointing", "is_gradient_checkpointing"):
                value = getattr(module, attribute, None)
                if isinstance(value, bool):
                    flags.append(value)
        result[name] = {
            "enabled": any(flags) if flags else None,
            "flag_count": len(flags),
            "enabled_flag_count": sum(flags),
        }
    return result


def _disable_gradient_checkpointing_for_probe(model: Any) -> dict[str, Any]:
    """Disable constructor-default GC without changing the training path."""
    base = model.get_model() if hasattr(model, "get_model") else model
    components = [("model", model), ("base", base)]
    components.extend(
        (name, getattr(base, name))
        for name in ("vision_tower", "language_model", "llm_connector", "dit", "action_dit")
        if getattr(base, name, None) is not None
    )
    called = []
    seen = set()
    for name, component in components:
        if id(component) in seen:
            continue
        seen.add(id(component))
        for method_name in ("gradient_checkpointing_disable", "disable_gradient_checkpointing"):
            method = getattr(component, method_name, None)
            if callable(method):
                method()
                called.append(f"{name}.{method_name}")
                break
    cleared_flags = 0
    for module in model.modules():
        if isinstance(getattr(module, "gradient_checkpointing", None), bool):
            module.gradient_checkpointing = False
            cleared_flags += 1
    return {"disable_methods_called": called, "gradient_checkpointing_flags_cleared": cleared_flags}


def _prepare_deterministic_attention(model: Any) -> dict[str, Any]:
    """Route vision and SDPA attention away from fused nondeterministic kernels."""
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    vision_flash_changes = []
    for module_name, module in model.named_modules():
        config = getattr(module, "config", None)
        if "vision" in module_name.lower() and hasattr(module, "use_flash_attn"):
            previous = module.use_flash_attn
            module.use_flash_attn = False
            vision_flash_changes.append({"module": module_name,
                                         "from": bool(previous), "to": False})
        if "vision" in module_name.lower() and config is not None and hasattr(config, "use_flash_attn"):
            previous = config.use_flash_attn
            config.use_flash_attn = False
            vision_flash_changes.append({"module": (module_name or "<root>") + ".config",
                                         "from": bool(previous), "to": False})
    return {"vision_flash_flags": vision_flash_changes,
            "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
            "cudnn_sdp_enabled": (
                torch.backends.cuda.cudnn_sdp_enabled()
                if hasattr(torch.backends.cuda, "cudnn_sdp_enabled") else None
            )}


def _prepare_fp32_reference(model: Any) -> dict[str, Any]:
    """Change only the probe model's arithmetic, after strict BF16 loading."""
    model.float()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    deterministic_attention = _prepare_deterministic_attention(model)

    modules = list(model.named_modules())
    attention_configs = []
    seen_configs = set()
    for module_name, module in modules:
        config = getattr(module, "config", None)
        if config is not None and id(config) not in seen_configs and hasattr(config, "_attn_implementation"):
            seen_configs.add(id(config))
            attention_configs.append((module_name or "<root>", config, str(config._attn_implementation)))
    attention_changes = []
    for module_name, config, previous in attention_configs:
        config._attn_implementation = "eager"
        attention_changes.append({"module": module_name, "from": previous,
                                  "to": str(config._attn_implementation)})

    return {"attention_configs": attention_changes,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            **deterministic_attention}


def _replace_option(argv: list[str], option: str, value: str) -> None:
    if option not in argv:
        argv.extend((option, value))
        return
    position = argv.index(option)
    if position + 1 >= len(argv) or argv[position + 1].startswith("--"):
        raise ValueError(f"Training command has no value for {option}")
    argv[position + 1] = value


def _training_argv(plan: dict[str, Any], config_path: Path, *, bf16: bool,
                   gradient_checkpointing: bool) -> list[str]:
    command = plan.get("command")
    if not isinstance(command, list) or "train_csgo.py" not in command:
        raise ValueError("Saved experiment_plan.json lacks the training command")
    argv = command[command.index("train_csgo.py") + 1:].copy()
    if "--deepspeed" in argv:
        position = argv.index("--deepspeed")
        del argv[position:position + 2]
    _replace_option(argv, "--csgo_config", str(config_path))
    _replace_option(argv, "--report_to", "none")
    _replace_option(argv, "--bf16", str(bf16))
    _replace_option(argv, "--gradient_checkpointing", str(gradient_checkpointing))
    # All saved paths were relative to the repository at training time.
    for option in ("--model_name_or_path", "--csgo_image_folder"):
        if option in argv:
            position = argv.index(option) + 1
            value = Path(argv[position])
            if not value.is_absolute() and (ROOT / value).exists():
                argv[position] = str(ROOT / value)
    return argv


def _local_processor_snapshot(argv: list[str]) -> str | None:
    if "--mllm_hf_path" not in argv:
        return None
    position = argv.index("--mllm_hf_path") + 1
    value = argv[position]
    if Path(value).is_dir():
        return str(Path(value).resolve())
    try:
        from huggingface_hub import snapshot_download
        snapshot = snapshot_download(repo_id=value, local_files_only=True)
    except (OSError, ValueError):
        return None
    argv[position] = snapshot
    return snapshot


def _validate_checkpoint(checkpoint_dir: Path) -> tuple[dict[str, Any], dict[str, Any], int]:
    if not checkpoint_dir.name.startswith("checkpoint-"):
        raise ValueError(f"Expected checkpoint directory: {checkpoint_dir}")
    state_path = checkpoint_dir / "trainer_state.json"
    config_path = checkpoint_dir.parent / "resolved_config.yaml"
    plan_path = checkpoint_dir.parent / "experiment_plan.json"
    weights_path = checkpoint_dir / "model.safetensors"
    for path in (state_path, config_path, plan_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    step = int(state["global_step"])
    if checkpoint_dir.name != f"checkpoint-{step}":
        raise ValueError("Checkpoint path and trainer state disagree on global_step")
    if not isinstance(config, dict) or not isinstance(plan, dict):
        raise ValueError("Saved experiment configuration or plan is invalid")
    saved_identity = config.get("exp32_3maps_experiment")
    if (saved_identity is not None and saved_identity != checkpoint_dir.parent.name) or plan.get("experiment") != checkpoint_dir.parent.name:
        raise ValueError("Saved experiment identity differs from checkpoint directory")
    if config.get("benchmark_v2_split") != "seen_train":
        raise ValueError("Gradient probe requires the saved seen_train split")
    if config.get("train_maps") != ["de_ancient", "de_dust2", "de_nuke"]:
        raise ValueError("Gradient probe requires the saved three training maps")
    return config, plan, step


def load_gradient_probe_context(
    checkpoint_dir: str | Path, *, device: str | torch.device = "cpu",
    bf16: bool = True, gradient_checkpointing: bool = True,
    precision_reference: bool = False,
    deterministic_probe: bool = False,
    data_dir: str | Path | None = None,
) -> GradientProbeContext:
    """Load a checkpoint with exact training construction and no optimizer.

    Construction and strict weight loading run on CPU. Only after that may the
    model be moved to ``device``. The returned model remains in training mode.
    """
    if not bf16:
        raise ValueError(
            "FP32 probe is not supported by the training model: train_csgo.py casts "
            "all parameters to bfloat16 and model forward has bfloat16-specific paths"
        )
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    config, plan, step = _validate_checkpoint(checkpoint_dir)
    runtime = plan.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("Saved experiment runtime is invalid")
    argv = _training_argv(
        plan, checkpoint_dir.parent / "resolved_config.yaml",
        bf16=bf16, gradient_checkpointing=gradient_checkpointing,
    )
    processor_snapshot = _local_processor_snapshot(argv)
    overrides: dict[str, Any] = {}
    if data_dir is not None:
        local_data_dir = Path(data_dir).expanduser().resolve()
        if not local_data_dir.is_dir():
            raise FileNotFoundError(local_data_dir)
        overrides["data_dir"] = str(local_data_dir)
        _replace_option(argv, "--csgo_image_folder", str(local_data_dir))
    else:
        saved_data_dir = Path(config["data_dir"])
        local_data_dir = saved_data_dir if saved_data_dir.is_absolute() else ROOT / saved_data_dir
        if not local_data_dir.is_dir():
            raise FileNotFoundError(local_data_dir)
        overrides["data_dir"] = str(local_data_dir)

    # Import only on actual model construction; validation and metadata tests
    # can run without the heavyweight training dependencies.
    import train_csgo

    model, dataset, collator = train_csgo.train(
        attn_implementation="flash_attention_2" if str(device).startswith("cuda") else "sdpa",
        probe_request={
            "argv": argv,
            "checkpoint_weights": checkpoint_dir / "model.safetensors",
            "checkpoint_step": step,
            "config_overrides": overrides,
        },
    )
    checkpointing_disable = (
        _disable_gradient_checkpointing_for_probe(model)
        if not gradient_checkpointing else None
    )
    indices_by_map = {map_name: [] for map_name in config["train_maps"]}
    source_rows_by_map = {map_name: [] for map_name in config["train_maps"]}
    for index, entry in enumerate(dataset.data_entries):
        map_name = entry["map"]
        indices_by_map[map_name].append(index)
        source_rows_by_map[map_name].append({
            "dataset_index": index,
            "map": map_name,
            "file_frame": str(entry["file_frame"]),
            "row_id": str(entry.get("id", entry["file_frame"])),
        })
    if any(not indices for indices in indices_by_map.values()):
        raise ValueError("Training dataset is missing one or more seen_train maps")
    reference_adjustments = _prepare_fp32_reference(model) if precision_reference else None
    deterministic_attention = (
        _prepare_deterministic_attention(model)
        if deterministic_probe and not precision_reference else
        reference_adjustments if deterministic_probe else None
    )
    if str(device) != "cpu":
        model.to(device)
    model.train()
    checkpointing_modules = _gradient_checkpointing_snapshot(model)
    if not gradient_checkpointing and any(
        component["enabled"] for component in checkpointing_modules.values()
    ):
        raise RuntimeError("Gradient checkpointing remained enabled in the requested off mode")
    parameter_dtypes = Counter()
    trainable_parameters = 0
    total_parameters = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total_parameters += count
        parameter_dtypes[str(parameter.dtype)] += count
        if parameter.requires_grad:
            trainable_parameters += count
    expected_parameter_dtypes = (
        {"torch.float32"} if precision_reference
        else {"torch.bfloat16", "torch.float32"}
    )
    if (not parameter_dtypes or not set(parameter_dtypes).issubset(expected_parameter_dtypes)
            or (not precision_reference and "torch.bfloat16" not in parameter_dtypes)):
        raise RuntimeError(
            f"Probe expected {'FP32' if precision_reference else 'BF16 base / FP32 adapters'} "
            f"parameters, found {dict(parameter_dtypes)}"
        )
    micro_batch = int(runtime["micro_batch"])
    accumulation = int(runtime["gradient_accumulation_steps"])
    world_size = int(runtime["world_size"])
    saved_config_path = checkpoint_dir.parent / "resolved_config.yaml"
    manifest_path = _saved_path(config["benchmark_v2_manifest"])
    asset_manifest_value = config.get("benchmark_v2_asset_manifest")
    asset_manifest_path = _saved_path(asset_manifest_value) if asset_manifest_value else None
    metadata = {
        "checkpoint": str(checkpoint_dir),
        "checkpoint_step": step,
        "local_step": step,
        "experiment": plan["experiment"],
        "train_microbatch": micro_batch,
        "train_gradient_accumulation_steps": accumulation,
        "train_world_size": world_size,
        "train_effective_batch": micro_batch * accumulation * world_size,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_modules": checkpointing_modules,
        "gradient_checkpointing_disable": checkpointing_disable,
        "gradient_checkpointing_use_reentrant": False if gradient_checkpointing else None,
        "model_training": model.training,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "dtype": "float32" if precision_reference else "bfloat16",
        "parameter_dtype_summary": "float32" if precision_reference else (
            "bf16_fp32_mixed" if "torch.float32" in parameter_dtypes else "bfloat16"
        ),
        "precision_reference": precision_reference,
        "precision_reference_adjustments": reference_adjustments,
        "deterministic_probe": deterministic_probe,
        "deterministic_attention_adjustments": deterministic_attention,
        "precision_reference_note": (
            "Probe-only FP32 parameters and eager attention; the probe script also "
            "casts floating batch inputs to FP32. Noise/time and some forward "
            "constants still originate as BF16, so this is a numerical reference "
            "rather than a bitwise replay of training inputs."
            if precision_reference else None
        ),
        "parameter_dtype_numel": dict(parameter_dtypes),
        "resolved_config": config,
        "resolved_config_sha256": _sha256(saved_config_path),
        "training_runtime": runtime,
        "finetune_init_source": str(checkpoint_dir / "model.safetensors"),
        "processor_snapshot": processor_snapshot,
        "dataset_indices_by_map": indices_by_map,
        "dataset_source_rows_by_map": source_rows_by_map,
        "dataset_source": {
            "benchmark_v2_manifest": str(manifest_path),
            "benchmark_v2_manifest_sha256": _sha256(manifest_path),
            "benchmark_v2_split": config["benchmark_v2_split"],
            "benchmark_v2_asset_manifest": str(asset_manifest_path) if asset_manifest_path else None,
            "benchmark_v2_asset_manifest_sha256": _sha256(asset_manifest_path) if asset_manifest_path else None,
            "data_dir": str(local_data_dir),
        },
        "loss_weights": {
            key: getattr(model.config, key, None)
            for key in ("alpha_loc_loss", "alpha_loc_aux_loss", "alpha_gen_aux_loss", "alpha_loc_perception_loss")
        },
    }
    return GradientProbeContext(model, dataset, collator, metadata)
