import copy
import importlib
import logging
import os
import sys
from typing import Any, Dict

import torch
import yaml


def _resolve_path(base_dir: str, path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(base_dir, path_value))


def _load_external_loc_config(config_path: str, csgosquare_root: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = copy.deepcopy(config)
    model_cfg = config.get("model", {})
    if "vision_config" in model_cfg and model_cfg["vision_config"]:
        model_cfg["vision_config"] = _resolve_path(csgosquare_root, model_cfg["vision_config"])
    model_cfg["load_params"] = False
    config["model"] = model_cfg
    return config


def _extract_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state_dict[key[len("module."):]] = value
        else:
            cleaned_state_dict[key] = value
    return cleaned_state_dict


def build_frozen_external_loc_model(
    *,
    csgosquare_root: str,
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    csgosquare_root = os.path.abspath(csgosquare_root)
    config_path = _resolve_path(csgosquare_root, config_path)
    checkpoint_path = _resolve_path(csgosquare_root, checkpoint_path)

    if not os.path.isdir(csgosquare_root):
        raise FileNotFoundError(f"csgosquare root not found: {csgosquare_root}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"external loc config not found: {config_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"external loc checkpoint not found: {checkpoint_path}")

    if csgosquare_root not in sys.path:
        sys.path.insert(0, csgosquare_root)

    loc_module = importlib.import_module("models.localization_model")
    LocModel = getattr(loc_module, "LocModel")
    DoubleTowerLocModel = getattr(loc_module, "DoubleTowerLocModel")

    cfg = _load_external_loc_config(config_path, csgosquare_root)
    architecture = cfg.get("model", {}).get("architecture", "siamese")
    if architecture == "double_tower":
        external_model = DoubleTowerLocModel(cfg)
    else:
        external_model = LocModel(cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint if isinstance(checkpoint, dict) else {"model": checkpoint})
    msg = external_model.load_state_dict(state_dict, strict=False)
    logging.info(
        "Loaded external loc model from %s (missing=%d, unexpected=%d)",
        checkpoint_path,
        len(msg.missing_keys),
        len(msg.unexpected_keys),
    )

    external_model.to(device=device, dtype=torch.float32)
    external_model.eval()
    for param in external_model.parameters():
        param.requires_grad = False
    return external_model

