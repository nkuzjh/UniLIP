import argparse
import datetime
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps
from safetensors.torch import load_file as safe_load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor

from csgo_datasets.unified_task_dataset import (
    _build_csgo_entry,
    _build_loc_prompt_cache,
    _build_map_tensor_cache,
    img_process,
    img_resize_transform,
)
from unilip.model import Unified_UniLIP_InternVLForCausalLM
from unilip.utils import disable_torch_init


MAP_PATH_DICT = {
    "de_dust2": "de_dust2_radar_psd.png",
    "de_inferno": "de_inferno_radar_psd.png",
    "de_mirage": "de_mirage_radar_psd.png",
    "de_nuke": "de_nuke_blended_radar_psd.png",
    "de_ancient": "de_ancient_radar_psd.png",
    "de_anubis": "de_anubis_radar_psd.png",
    "de_golden": "de_golden_radar_tga.png",
    "de_overpass": "de_overpass_radar_psd.png",
    "de_palacio": "de_palacio_radar_tga.png",
    "de_train": "de_train_blended_radar_psd.png",
    "de_vertigo": "de_vertigo_blended_radar_psd.png",
    "cs_agency": "cs_agency_radar_tga.png",
    "cs_italy": "cs_italy_radar_psd.png",
    "cs_office": "cs_office_radar_psd.png",
}

VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
FRAME_DIFF_THRESHOLD = 2
TITLE_HEIGHT = 64


@dataclass(frozen=True)
class FrameKey:
    file_frame: str
    file_num: int
    frame_id: int


class InferenceArgs:
    def __init__(self, config_dict: Dict[str, object]):
        self.__dict__.update(config_dict)
        self.unilip_path = config_dict.get("unilip_path", "")
        self.unilip_factor = config_dict.get("unilip_factor", 5.85)
        self.fix_dit = config_dict.get("fix_dit", False)
        self.fix_connect = config_dict.get("fix_connect", False)
        self.fix_vit = config_dict.get("fix_vit", True)
        self.fix_llm = config_dict.get("fix_llm", True)
        self.mllm_path = config_dict.get("mllm_path", "")
        self.mllm_hf_path = config_dict.get("mllm_hf_path", "OpenGVLab/InternVL3-1B-hf")
        self.vae_path = config_dict.get("vae_path", "")
        self.dit_path = config_dict.get("dit_path", "")
        self.lazy_preprocess = True
        self.n_query = 256
        self.n_und_query = 0
        self.connect_layer = 6
        self.action_horizon = config_dict.get("action_horizon", 1)
        self.action_dim = config_dict.get("action_dim", 5)
        self.is_action_dit_dense_timestep = config_dict.get("is_action_dit_dense_timestep", False)
        self.action_dit_layer = config_dict.get("action_dit_layer", 3)
        self.mm_use_im_patch_token = config_dict.get("mm_use_im_patch_token", False)
        self.mm_use_im_start_end = config_dict.get("mm_use_im_start_end", False)
        self.tune_mm_mlp_adapter = False
        self.pretrain_mm_mlp_adapter = None
        self.version = "internvl"
        self.data_type = "mix"
        self.bf16 = True
        self.tf32 = True
        self.image_aspect_ratio = "square"
        self.model_max_length = 1024
        self.is_action_dit_projector = config_dict.get("is_action_dit_projector", False)
        self.is_loc_learnable_query = config_dict.get("is_loc_learnable_query", False)
        self.use_vit_cls_regression_head = config_dict.get("use_vit_cls_regression_head", False)
        self.use_vit_regression_head = config_dict.get("use_vit_regression_head", False)
        self.use_codex_vit_regression_head = config_dict.get("use_codex_vit_regression_head", False)
        self.use_pi05_action_dit = config_dict.get("use_pi05_action_dit", False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def json_safe(obj):
    if isinstance(obj, dict):
        return {key: json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [json_safe(value) for value in obj]
    if isinstance(obj, tuple):
        return [json_safe(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def parse_frame_key(file_frame: str) -> Optional[FrameKey]:
    import re

    stem = Path(file_frame).stem
    match = re.search(r"file_num(\d+)_frame_(\d+)", stem)
    if match is None:
        return None
    return FrameKey(file_frame=stem, file_num=int(match.group(1)), frame_id=int(match.group(2)))


def build_contiguous_tracks(
    records: Sequence[dict],
) -> Tuple[List[List[dict]], Dict[str, object]]:
    keyed_records = []
    unparsed = []
    for record in records:
        key = parse_frame_key(record["file_frame"])
        if key is None:
            unparsed.append(record["file_frame"])
            continue
        keyed_records.append((key, record))

    keyed_records.sort(key=lambda item: (item[1].get("map", ""), item[0].file_num, item[0].frame_id))
    raw_tracks: List[List[dict]] = []
    current: List[Tuple[FrameKey, dict]] = []

    for key, record in keyed_records:
        if not current:
            current = [(key, record)]
            continue
        last_key, last_record = current[-1]
        same_map = record.get("map") == last_record.get("map")
        same_file = key.file_num == last_key.file_num
        frame_delta = key.frame_id - last_key.frame_id
        contiguous = frame_delta <= FRAME_DIFF_THRESHOLD
        if same_map and same_file and contiguous:
            current.append((key, record))
        else:
            raw_tracks.append([item[1] for item in current])
            current = [(key, record)]
    if current:
        raw_tracks.append([item[1] for item in current])

    details = {
        "parsed_frame_count": len(keyed_records),
        "unparsed_frame_count": len(unparsed),
        "unparsed_examples": unparsed[:20],
        "raw_track_count": len(raw_tracks),
        "track_count": len(raw_tracks),
        "frame_diff_threshold": FRAME_DIFF_THRESHOLD,
        "track_summaries": [
            {
                "track_index": idx,
                "map": track[0].get("map"),
                "length": len(track),
                "first_file_frame": track[0]["file_frame"],
                "last_file_frame": track[-1]["file_frame"],
            }
            for idx, track in enumerate(raw_tracks)
        ],
    }
    return raw_tracks, details


def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model) -> None:
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


def smart_matching_state_dict_keys(state_dict, model) -> dict:
    del model
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("language_model.lm_head."):
            new_key = key[len("language_model.") :]
        elif key.startswith("language_model.model."):
            new_key = "model.language_model." + key[len("language_model.model.") :]
        elif key.startswith("vision_tower."):
            new_key = "model." + key
        elif key.startswith("multi_modal_projector."):
            new_key = "model." + key
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def unnormalize_pose(pred_tensor: torch.Tensor, z_range: dict) -> dict:
    x_norm, y_norm, z_norm, v_norm, h_norm = [float(v) for v in pred_tensor.cpu().numpy()]
    min_z = float(z_range["min_z"])
    max_z = float(z_range["max_z"])
    return {
        "x": x_norm * 1024.0,
        "y": y_norm * 1024.0,
        "z": z_norm * (max_z - min_z + 1e-6) + min_z,
        "angle_v": v_norm * 360.0,
        "angle_h": h_norm * 360.0,
    }


class ContinuousCSGOLocInferenceDataset(Dataset):
    def __init__(
        self,
        config: Dict[str, object],
        tokenizer,
        image_processor,
        image_aspect_ratio: str,
        split_name: str,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.data_dir = config["data_dir"]
        self.map_names = config.get("test_maps", config.get("val_maps", []))
        self.image_processor = image_processor
        self.image_aspect_ratio = image_aspect_ratio
        self.img_size = int(config.get("img_size", 448))
        self.is_resize_224 = self.img_size == 224
        self.data_entries = []
        self.map_z_range = {}
        self.loaded_split_paths = []

        for map_name in self.map_names:
            split_path = Path(self.data_dir) / map_name / "splits_20000_5000" / split_name
            if not split_path.is_file():
                print(f"Skip {map_name}: split not found: {split_path}")
                continue
            with open(split_path, "r", encoding="utf-8") as f:
                positions_data = json.load(f)
            self.loaded_split_paths.append(str(split_path))

            train_split_path = Path(self.data_dir) / map_name / "splits_20000_5000" / "train_split.json"
            z_ref_data = positions_data
            z_ref_source = split_name
            if train_split_path.is_file():
                with open(train_split_path, "r", encoding="utf-8") as f:
                    z_ref_data = json.load(f)
                z_ref_source = "train_split.json"

            z_values = [item["z"] for item in z_ref_data]
            min_z, max_z = min(z_values), max(z_values)
            self.map_z_range[map_name] = {"min_z": min_z, "max_z": max_z}
            print(f"{map_name}: loaded {len(positions_data)} frames from {split_path}")
            print(f"{map_name}: z range from {z_ref_source}: [{min_z:.4f}, {max_z:.4f}]")

            for pos_data in positions_data:
                self.data_entries.append(_build_csgo_entry(map_name, pos_data, min_z, max_z))

        if not self.data_entries:
            raise ValueError(f"No localization samples loaded from split={split_name}.")

        self.map_images = {}
        for map_name, filename in MAP_PATH_DICT.items():
            path = Path(self.data_dir) / map_name / filename
            if path.is_file():
                self.map_images[map_name] = Image.open(path).convert("RGB").resize((448, 448))

        self.map_tensor_cache_448, self.map_tensor_cache_224 = _build_map_tensor_cache(
            self.map_images,
            self.image_processor,
            self.image_aspect_ratio,
            self.is_resize_224,
        )
        self.loc_prompt_cache = _build_loc_prompt_cache(
            self.map_z_range.keys(),
            self.tokenizer,
            self.img_size,
            use_short_instruction=bool(config.get("use_short_instruction", False)),
        )

    def __len__(self) -> int:
        return len(self.data_entries)

    def __getitem__(self, idx: int) -> dict:
        data = self.data_entries[idx]
        map_name = data["map"]
        map_tensor_448 = self.map_tensor_cache_448[map_name]
        tensor_map = self.map_tensor_cache_224[map_name] if self.is_resize_224 else map_tensor_448

        fps_path = resolve_fps_path(self.data_dir, map_name, data["file_frame"], None)
        fps_img = Image.open(fps_path).convert("RGB")
        fps_tensor_448 = img_process([fps_img], self.image_processor, self.image_aspect_ratio)
        tensor_fps = img_resize_transform(fps_tensor_448) if self.is_resize_224 else fps_tensor_448
        loc_cache = self.loc_prompt_cache[map_name]

        return {
            "map_name": map_name,
            "ids": data["file_frame"],
            "und_image": tensor_fps,
            "aux_image": tensor_map,
            "input_ids": loc_cache["input_ids"],
            "labels": loc_cache["labels"],
            "actions": data["actions"],
            "pose_dict": data["pose_dict"],
            "z_range": {"min_z": data["z_min"], "max_z": data["z_max"]},
        }


class DataCollatorForLoc:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[dict]) -> dict:
        from unilip.constants import IGNORE_INDEX

        safe_len = self.tokenizer.model_max_length - 257
        input_ids = [item["input_ids"][:safe_len] for item in instances]
        labels = [item["labels"][:safe_len] for item in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        if input_ids.shape[1] > self.tokenizer.model_max_length:
            input_ids = input_ids[:, : self.tokenizer.model_max_length]
            labels = labels[:, : self.tokenizer.model_max_length]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "ids": [item["ids"] for item in instances],
            "und_image": torch.cat([item["und_image"] for item in instances], dim=0),
            "aux_image": torch.cat([item["aux_image"] for item in instances], dim=0),
            "actions": torch.stack([item["actions"] for item in instances], dim=0),
            "map_name": [item["map_name"] for item in instances],
            "pose_dict": [item["pose_dict"] for item in instances],
            "z_range": [item["z_range"] for item in instances],
        }


def configure_model(model, csgo_config: Dict[str, object], inference_args: InferenceArgs) -> None:
    model.config.img_size = getattr(inference_args, "img_size", csgo_config.get("img_size", False))
    model.config.is_action_dit_dense_timestep = getattr(
        inference_args,
        "is_action_dit_dense_timestep",
        csgo_config.get("is_action_dit_dense_timestep", False),
    )
    model.config.use_vit_cls_regression_head = csgo_config.get("use_vit_cls_regression_head", False)
    model.config.use_vit_regression_head = csgo_config.get("use_vit_regression_head", False)
    model.config.use_codex_vit_regression_head = csgo_config.get("use_codex_vit_regression_head", False)
    model.config.use_pi05_action_dit = csgo_config.get("use_pi05_action_dit", False)
    model.config.pi05_pytorch_weight_path = csgo_config.get("pi05_pytorch_weight_path", False)
    model.config.is_loc_aux_loss = csgo_config.get("is_loc_aux_loss", False)
    model.config.alpha_loc_aux_loss = csgo_config.get("alpha_loc_aux_loss", 1.0)
    model.config.alpha_loc_loss = csgo_config.get("alpha_loc_loss", 1.0)
    model.config.is_aciton_dit_vae_small_init = csgo_config.get("is_aciton_dit_vae_small_init", False)
    model.config.loc_use_circular_loss = csgo_config.get("loc_use_circular_loss", True)
    model.config.loc_xy_loss_weight = csgo_config.get("loc_xy_loss_weight", 1.0)
    model.config.loc_z_loss_weight = csgo_config.get("loc_z_loss_weight", 1.0)
    model.config.loc_angle_loss_weight = csgo_config.get("loc_angle_loss_weight", 2.0)
    model.config.is_exp5_eval_without_aciton_dit_premodules = getattr(
        inference_args,
        "is_exp5_eval_without_aciton_dit_premodules",
        csgo_config.get("is_exp5_eval_without_aciton_dit_premodules", False),
    )
    model.config.is_action_dit_projector = getattr(
        inference_args, "is_action_dit_projector", csgo_config.get("is_action_dit_projector", False)
    )
    model.config.is_loc_learnable_query = getattr(
        inference_args, "is_loc_learnable_query", csgo_config.get("is_loc_learnable_query", False)
    )


def run_continuous_loc_inference(args: argparse.Namespace, output_dir: Path) -> Path:
    with open(args.csgo_config, "r", encoding="utf-8") as f:
        csgo_config = yaml.safe_load(f)
    if args.data_dir is not None:
        csgo_config["data_dir"] = args.data_dir
    if args.map_name != "auto":
        csgo_config["test_maps"] = [args.map_name]
        csgo_config["val_maps"] = [args.map_name]
    if args.batch_size is not None:
        csgo_config["batch_size"] = args.batch_size

    disable_torch_init()
    set_seed(args.seed)
    inference_args = InferenceArgs(csgo_config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = Unified_UniLIP_InternVLForCausalLM.from_pretrained(
        csgo_config.get("model_name_or_path", "UniLIP-1B"),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoProcessor.from_pretrained(inference_args.mllm_hf_path).tokenizer
    tokenizer.model_max_length = inference_args.model_max_length
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            dict(pad_token="<pad>", additional_special_tokens=["[IMG]", "[/IMG]", "<image>"]),
            tokenizer,
            model,
        )
    elif "<image>" not in tokenizer.get_added_vocab():
        smart_tokenizer_and_embedding_resize(
            dict(additional_special_tokens=["[IMG]", "[/IMG]", "<image>"]),
            tokenizer,
            model,
        )

    from unilip import conversation as conversation_lib

    if inference_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[inference_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama3"]

    configure_model(model, csgo_config, inference_args)
    original_is_action_dit_projector = bool(getattr(model.config, "is_action_dit_projector", False))
    needs_action_norm_compat = not original_is_action_dit_projector and not any(
        bool(getattr(model.config, key, False))
        for key in (
            "use_vit_cls_regression_head",
            "use_vit_regression_head",
            "use_codex_vit_regression_head",
        )
    )
    if needs_action_norm_compat:
        # Current model code always uses action_dit_norm in the default Action-DiT path,
        # while some older configs did not set is_action_dit_projector. Temporarily
        # enable the branch so the norm module is constructed, then restore the flag.
        model.config.is_action_dit_projector = True
    model.get_model().initialize_vision_modules(model_args=inference_args)
    model.get_model().initialize_localization_modules(model_args=inference_args)
    if needs_action_norm_compat:
        model.config.is_action_dit_projector = original_is_action_dit_projector

    if csgo_config.get("is_lora", False):
        training_args = SimpleNamespace(
            is_lora=csgo_config.get("is_lora", True),
            lora_r=csgo_config.get("lora_r", 16),
            lora_alpha=csgo_config.get("lora_alpha", 16),
            lora_dropout=csgo_config.get("lora_dropout", 0.05),
        )
        model.inject_lora_to_sub_module(inference_args, training_args)

    ckpt_path = csgo_config["ckpt_path"]
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")
    if ckpt_path.endswith(".safetensors"):
        state_dict = safe_load_file(ckpt_path, device="cpu")
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")
    load_msg = model.load_state_dict(smart_matching_state_dict_keys(state_dict, model), strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Load message: {load_msg}")

    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    for _, param in model.named_parameters():
        param.requires_grad = False
    model.config.use_cache = True

    image_processor = AutoProcessor.from_pretrained(inference_args.mllm_hf_path).image_processor
    dataset = ContinuousCSGOLocInferenceDataset(
        csgo_config,
        tokenizer,
        image_processor,
        inference_args.image_aspect_ratio,
        args.split_name,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(csgo_config.get("batch_size", 1)),
        shuffle=False,
        collate_fn=DataCollatorForLoc(tokenizer),
    )

    results = []
    for batch in tqdm(dataloader, desc="Continuous localization inference"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        und_image = batch["und_image"].to(device)
        aux_image = batch["aux_image"].to(device)
        with torch.no_grad():
            pred_norm_loc_tensor = model.generate_action2(
                input_ids,
                attention_mask,
                labels,
                und_image,
                aux_image,
                num_steps=args.num_steps,
            ).squeeze(1).float().cpu()

        for sample_idx in range(pred_norm_loc_tensor.size(0)):
            pred_norm = pred_norm_loc_tensor[sample_idx]
            gt_norm = batch["actions"][sample_idx].squeeze(0).float().cpu()
            results.append(
                {
                    "file_frame": batch["ids"][sample_idx],
                    "map": batch["map_name"][sample_idx],
                    "pred_norm": pred_norm.tolist(),
                    "gt_norm": gt_norm.tolist(),
                    "pred": unnormalize_pose(pred_norm, batch["z_range"][sample_idx]),
                    "gt": batch["pose_dict"][sample_idx],
                    "source_split": args.split_name,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "conti_loc_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(results), f, indent=2)
    print(f"Saved continuous localization results: {results_path}")
    return results_path


def load_loc_results(path: Path) -> List[dict]:
    if not path.is_file() and path.name == "loc_results.json":
        fallback_path = path.with_name("conti_loc_results.json")
        if fallback_path.is_file():
            print(f"Localization result not found: {path}")
            print(f"Using sibling continuous result instead: {fallback_path}")
            path = fallback_path
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in loc_results json: {path}")
    normalized = []
    for item in records:
        if "file_frame" not in item or "pred" not in item or "gt" not in item:
            continue
        next_item = dict(item)
        next_item["file_frame"] = Path(str(next_item["file_frame"])).stem
        normalized.append(next_item)
    if not normalized:
        raise ValueError(f"No usable records found in loc_results: {path}")
    return normalized


def resolve_fps_path(data_dir: str, map_name: str, file_frame: str, gt_fps_dir: Optional[str]) -> Path:
    stem = Path(file_frame).stem
    search_dirs = []
    if gt_fps_dir:
        search_dirs.append(Path(gt_fps_dir))
    search_dirs.append(Path(data_dir) / map_name / "imgs")

    original_path = Path(file_frame)
    if original_path.is_file():
        return original_path

    for directory in search_dirs:
        for ext in VALID_IMAGE_EXTS:
            path = directory / f"{stem}{ext}"
            if path.is_file():
                return path
    raise FileNotFoundError(f"FPS image not found for {map_name}/{stem} under {search_dirs}")


def resolve_pred_fps_path(pred_search_dirs: Sequence[Path], map_name: str, file_frame: str) -> Optional[Path]:
    stem = Path(file_frame).stem
    for base_dir in pred_search_dirs:
        for directory in (base_dir, base_dir / map_name):
            for ext in VALID_IMAGE_EXTS:
                path = directory / f"{stem}{ext}"
                if path.is_file():
                    return path
    return None


def resolve_map_path(data_dir: str, map_name: str) -> Path:
    map_filename = MAP_PATH_DICT.get(map_name)
    if map_filename is None:
        raise ValueError(f"Unknown map name: {map_name}")
    map_path = Path(data_dir) / map_name / map_filename
    if not map_path.is_file():
        raise FileNotFoundError(f"Map image not found: {map_path}")
    return map_path


def infer_single_map(records: Sequence[dict], explicit_map_name: str) -> str:
    if explicit_map_name != "auto":
        return explicit_map_name
    map_names = sorted({item.get("map") for item in records if item.get("map")})
    if len(map_names) != 1:
        raise ValueError(f"Unable to infer a single map from records: {map_names}")
    return map_names[0]


def contain_resize(image: Image.Image, size: Tuple[int, int], fill=(18, 18, 18)) -> Image.Image:
    image = image.convert("RGB")
    canvas = Image.new("RGB", size, fill)
    image.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_title(panel: Image.Image, title: str, subtitle: str = "", extra: str = "") -> Image.Image:
    canvas = Image.new("RGB", (panel.width, panel.height + TITLE_HEIGHT), (20, 20, 20))
    canvas.paste(panel, (0, TITLE_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 7), title, fill=(245, 245, 245), font=font)
    if subtitle:
        draw.text((10, 24), subtitle, fill=(190, 190, 190), font=font)
    if extra:
        draw.text((10, 42), extra, fill=(190, 190, 190), font=font)
    return canvas


def draw_panel_label(image: Image.Image, lines: Sequence[str], fill: Tuple[int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_h = 14
    text_w = max((draw.textlength(line, font=font) for line in lines), default=0)
    text_h = line_h * len(lines) + 8
    draw.rectangle((0, 0, int(text_w) + 14, text_h), fill=(0, 0, 0))
    for idx, line in enumerate(lines):
        draw.text((7, 5 + idx * line_h), line, fill=fill, font=font)


def crop_pred_to_gt_size(pred_image: Image.Image, gt_image: Image.Image) -> Image.Image:
    return ImageOps.fit(
        pred_image.convert("RGB"),
        gt_image.size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def frame_info_text(record: dict) -> str:
    key = parse_frame_key(record["file_frame"])
    if key is None:
        return f"Frame {record['file_frame']}"
    return f"File {key.file_num} | Frame {key.frame_id}"


def pose_text(prefix: str, pose: dict) -> str:
    return (
        f"{prefix}: x={float(pose['x']):.1f}, y={float(pose['y']):.1f}, "
        f"z={float(pose['z']):.1f}, p={float(pose['angle_v']):.1f}, yaw={float(pose['angle_h']):.1f}"
    )


def pose_to_pixel(pose: dict, width: int, height: int) -> Tuple[float, float]:
    return float(pose["x"]) * width / 1024.0, float(pose["y"]) * height / 1024.0


def render_trajectory_panel(
    base_map: Image.Image,
    poses: Sequence[dict],
    *,
    color: Tuple[int, int, int],
    panel_size: int,
    trail_width: int,
    point_radius: int,
    draw_yaw: bool,
    title: str,
    subtitle: str,
    extra: str = "",
) -> Image.Image:
    map_rgba = base_map.convert("RGBA")
    overlay = Image.new("RGBA", map_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    points = [pose_to_pixel(pose, map_rgba.width, map_rgba.height) for pose in poses]

    if len(points) >= 2:
        draw.line(points, fill=(*color, 210), width=trail_width, joint="curve")
    for point in points[:-1]:
        x, y = point
        r = max(1, point_radius // 2)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(*color, 130))

    if points:
        x, y = points[-1]
        r = point_radius
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(*color, 245), outline=(0, 0, 0, 255), width=2)
        if draw_yaw:
            yaw_deg = float(poses[-1].get("angle_h", 0.0))
            yaw_rad = math.radians(yaw_deg)
            arrow_len = max(16, int(panel_size * 0.06))
            end = (x + math.cos(yaw_rad) * arrow_len, y + math.sin(yaw_rad) * arrow_len)
            draw.line((x, y, end[0], end[1]), fill=(*color, 255), width=max(2, trail_width))

    combined = Image.alpha_composite(map_rgba, overlay).convert("RGB")
    panel = contain_resize(combined, (panel_size, panel_size))
    return draw_title(panel, title, subtitle, extra=extra)


def render_fps_panel(
    gt_path: Path,
    pred_path: Optional[Path],
    *,
    panel_size: int,
    title: str,
    subtitle: str,
    extra: str,
) -> Image.Image:
    gt_image = Image.open(gt_path).convert("RGB")
    if pred_path is None:
        labeled_gt = gt_image.copy()
        draw_panel_label(labeled_gt, ["Ground Truth"], fill=(255, 190, 40))
        panel = contain_resize(labeled_gt, (panel_size, panel_size))
        return draw_title(panel, title, subtitle, extra=extra)

    pred_image = Image.open(pred_path).convert("RGB")
    pred_image = crop_pred_to_gt_size(pred_image, gt_image)
    labeled_pred = pred_image.copy()
    labeled_gt = gt_image.copy()
    draw_panel_label(labeled_pred, ["Prediction", "crop -> GT"], fill=(40, 255, 90))
    draw_panel_label(labeled_gt, ["Ground Truth"], fill=(255, 190, 40))

    separator_w = max(4, gt_image.width // 80)
    combined = Image.new("RGB", (gt_image.width * 2 + separator_w, gt_image.height), (0, 0, 0))
    combined.paste(labeled_pred, (0, 0))
    combined.paste(labeled_gt, (gt_image.width + separator_w, 0))
    panel = contain_resize(combined, (panel_size, panel_size))
    return draw_title(panel, title, subtitle, extra=extra)


def write_three_col_video(
    records: Sequence[dict],
    *,
    data_dir: str,
    gt_fps_dir: Optional[str],
    pred_search_dirs: Sequence[Path],
    map_name: str,
    output_video: Path,
    fps: float,
    panel_size: int,
    trail_width: int,
    point_radius: int,
    draw_yaw: bool,
    max_frames: Optional[int],
    video_codec: str,
    dump_frames_dir: Optional[Path],
) -> None:
    try:
        import cv2  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError("opencv-python is required to write mp4 video. Install requirements.txt first.") from exc

    output_video.parent.mkdir(parents=True, exist_ok=True)
    base_map = Image.open(resolve_map_path(data_dir, map_name)).convert("RGBA")
    selected_records = list(records[:max_frames]) if max_frames else list(records)
    if not selected_records:
        raise ValueError("No records selected for video rendering.")

    frame_w = panel_size * 3
    frame_h = panel_size + TITLE_HEIGHT
    codec_candidates = [video_codec]
    for codec in ("mp4v", "avc1", "H264", "XVID", "MJPG"):
        if codec not in codec_candidates:
            codec_candidates.append(codec)

    writer = None
    selected_codec = None
    for codec in codec_candidates:
        candidate = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (frame_w, frame_h),
        )
        if candidate.isOpened():
            writer = candidate
            selected_codec = codec
            break
        candidate.release()
    if writer is None:
        raise RuntimeError(f"Failed to open video writer for {output_video} with codecs={codec_candidates}")
    print(f"Video writer opened: codec={selected_codec}, fps={fps}, size={frame_w}x{frame_h}")

    if dump_frames_dir is not None:
        dump_frames_dir.mkdir(parents=True, exist_ok=True)

    current_segment: List[dict] = []
    previous_key: Optional[FrameKey] = None
    for idx, record in enumerate(tqdm(selected_records, desc="Rendering three-column video")):
        current_key = parse_frame_key(record["file_frame"])
        if previous_key is None:
            current_segment = [record]
        elif current_key is not None and previous_key is not None and current_key.file_num == previous_key.file_num:
            current_segment.append(record)
        else:
            current_segment = [record]
        previous_key = current_key
        fps_path = resolve_fps_path(data_dir, map_name, record["file_frame"], gt_fps_dir)
        pred_fps_path = resolve_pred_fps_path(pred_search_dirs, map_name, record["file_frame"])
        progress_text = f"Clip {idx + 1}/{len(selected_records)} | Trajectory {len(current_segment)} pts"
        left = render_fps_panel(
            fps_path,
            pred_fps_path,
            panel_size=panel_size,
            title=f"Current Frame: {frame_info_text(record)}",
            subtitle="Prediction vs Ground Truth" if pred_fps_path else "Ground Truth FPS",
            extra=progress_text,
        )
        middle = render_trajectory_panel(
            base_map,
            [item["pred"] for item in current_segment],
            color=(255, 80, 60),
            panel_size=panel_size,
            trail_width=trail_width,
            point_radius=point_radius,
            draw_yaw=draw_yaw,
            title="Predicted Trajectory",
            subtitle=pose_text("Pred", record["pred"]),
            extra=progress_text,
        )
        right = render_trajectory_panel(
            base_map,
            [item["gt"] for item in current_segment],
            color=(60, 190, 255),
            panel_size=panel_size,
            trail_width=trail_width,
            point_radius=point_radius,
            draw_yaw=draw_yaw,
            title="Ground Truth Trajectory",
            subtitle=pose_text("GT", record["gt"]),
            extra=progress_text,
        )
        frame = Image.new("RGB", (frame_w, frame_h), (0, 0, 0))
        frame.paste(left, (0, 0))
        frame.paste(middle, (panel_size, 0))
        frame.paste(right, (panel_size * 2, 0))
        frame_bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
        if dump_frames_dir is not None:
            cv2.imwrite(str(dump_frames_dir / f"{output_video.stem}_frame_{idx:06d}.jpg"), frame_bgr)

    writer.release()
    print(f"Saved video: {output_video}")

    cap = cv2.VideoCapture(str(output_video))
    readable = cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if readable else 0
    read_fps = cap.get(cv2.CAP_PROP_FPS) if readable else 0.0
    ok, _ = cap.read() if readable else (False, None)
    cap.release()
    print(f"Video validation: readable={readable}, frames={frame_count}, fps={read_fps:.3f}, first_frame={ok}")
    if not readable or not ok or frame_count <= 0:
        raise RuntimeError(
            "Video file was written but OpenCV cannot read frames back. "
            "Try --video_codec MJPG --output_video <path>.avi, or pass --dump_frames_dir to inspect rendered frames."
        )


def choose_tracks_or_rerun(args: argparse.Namespace, output_dir: Path) -> Tuple[List[List[dict]], Path]:
    results_path = Path(args.loc_results) if args.loc_results else None
    if args.force_infer or results_path is None:
        if not args.csgo_config:
            raise ValueError("--csgo_config is required when --loc_results is absent or --force_infer is set.")
        results_path = run_continuous_loc_inference(args, output_dir)

    records = load_loc_results(results_path)
    if args.map_name != "auto":
        records = [item for item in records if item.get("map") == args.map_name]
    tracks, details = build_contiguous_tracks(records)

    if not tracks and args.rerun_if_not_contiguous and args.csgo_config and not args.force_infer:
        print("Existing localization results do not contain a valid contiguous track; rerunning on continuous split.")
        results_path = run_continuous_loc_inference(args, output_dir)
        records = load_loc_results(results_path)
        if args.map_name != "auto":
            records = [item for item in records if item.get("map") == args.map_name]
        tracks, details = build_contiguous_tracks(records)

    details_path = output_dir / "conti_track_details.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(details), f, indent=2)
    print(f"Contiguous track details: {details_path}")

    if not tracks:
        raise ValueError(
            "No contiguous track found. "
            f"frame_diff_threshold={FRAME_DIFF_THRESHOLD}. "
            f"Details: {details}"
        )
    if args.track_index >= 0:
        if args.track_index >= len(tracks):
            raise ValueError(f"--track_index {args.track_index} out of range; found {len(tracks)} tracks.")
        tracks = [tracks[args.track_index]]

    tracks = sorted(
        tracks,
        key=lambda track: (
            track[0].get("map", ""),
            parse_frame_key(track[0]["file_frame"]).file_num,
            parse_frame_key(track[0]["file_frame"]).frame_id,
        ),
    )
    print(f"Selected track count: {len(tracks)}")
    for idx, track in enumerate(tracks):
        print(
            f"  track {idx}: length={len(track)}, "
            f"first={track[0]['file_frame']}, last={track[-1]['file_frame']}"
        )
    return tracks, results_path


def frame_key_or_raise(record: dict) -> FrameKey:
    key = parse_frame_key(record["file_frame"])
    if key is None:
        raise ValueError(f"Unable to parse frame key: {record['file_frame']}")
    return key


def output_videos_for_tracks(
    tracks: Sequence[Sequence[dict]],
    *,
    args: argparse.Namespace,
    output_dir: Path,
    data_dir: str,
    map_name: str,
    pred_search_dirs: Sequence[Path],
) -> List[Path]:
    max_frames_per_clip = args.max_duration * args.fps
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []

    for track_idx, track in enumerate(tracks):
        if len(track) < args.min_frames:
            continue

        selected_track = list(track)
        chunks = [
            selected_track[start_idx : start_idx + max_frames_per_clip]
            for start_idx in range(0, len(selected_track), max_frames_per_clip)
        ]

        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk) < args.min_frames // 2:
                continue

            start_key = frame_key_or_raise(chunk[0])
            end_key = frame_key_or_raise(chunk[-1])
            suffix = Path(args.output_video).suffix if args.output_video else ".mp4"
            out_name = f"three_col_track_{start_key.file_num}_f{start_key.frame_id}_to_f{end_key.frame_id}{suffix}"
            if len(chunks) > 1:
                out_name = (
                    f"three_col_track_{start_key.file_num}_clip{chunk_idx:02d}_"
                    f"f{start_key.frame_id}_to_f{end_key.frame_id}{suffix}"
                )
            out_path = output_dir / out_name
            print(f"Writing video: {out_name} ({len(chunk)} frames, track_index={track_idx})")
            write_three_col_video(
                chunk,
                data_dir=data_dir,
                gt_fps_dir=args.gt_fps_dir,
                pred_search_dirs=pred_search_dirs,
                map_name=map_name,
                output_video=out_path,
                fps=args.fps,
                panel_size=args.panel_size,
                trail_width=args.trail_width,
                point_radius=args.point_radius,
                draw_yaw=args.draw_yaw,
                max_frames=None,
                video_codec=args.video_codec,
                dump_frames_dir=Path(args.dump_frames_dir) if args.dump_frames_dir else None,
            )
            output_paths.append(out_path)

    if not output_paths:
        raise ValueError(
            f"No videos generated. Check min_frames={args.min_frames}, "
            f"max_duration={args.max_duration}, track_count={len(tracks)}."
        )
    return output_paths


def default_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    if args.loc_results:
        return Path(args.loc_results).resolve().parent
    config_stem = Path(args.csgo_config).stem if args.csgo_config else "csgo_conti_loc"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs_loc") / config_stem / f"conti_video_{timestamp}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a three-column CSGO continuous localization video.")
    parser.add_argument("--loc_results", type=str, default=None, help="Existing loc_results.json or conti_loc_results.json.")
    parser.add_argument("--csgo_config", type=str, default=None, help="Localization yaml config used when rerunning inference.")
    parser.add_argument("--force_infer", action="store_true", help="Always rerun localization on the continuous split.")
    parser.add_argument(
        "--rerun_if_not_contiguous",
        action="store_true",
        help="Rerun localization if --loc_results has no valid contiguous track.",
    )
    parser.add_argument("--split_name", type=str, default="continuous_unseen_clips.json")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--gt_fps_dir", type=str, default=None)
    parser.add_argument(
        "--pred_fps_dir",
        type=str,
        default=None,
        help="Optional existing prediction FPS dir. If omitted, video still renders GT FPS and localization trajectories.",
    )
    parser.add_argument("--map_name", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_video", type=str, default=None)
    parser.add_argument(
        "--track_index",
        type=int,
        default=-1,
        help="Track index to render. Default -1 renders every valid continuous track.",
    )
    parser.add_argument("--min_frames", type=int, default=10, help="Minimum frames for a track; chunks shorter than half are skipped.")
    parser.add_argument("--max_duration", type=int, default=10, help="Maximum seconds per output clip, matching frames_to_video.py.")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS, matching frames_to_video.py.")
    parser.add_argument("--video_codec", type=str, default="mp4v")
    parser.add_argument("--dump_frames_dir", type=str, default=None)
    parser.add_argument("--panel_size", type=int, default=512)
    parser.add_argument("--trail_width", type=int, default=4)
    parser.add_argument("--point_radius", type=int, default=7)
    parser.add_argument("--draw_yaw", action="store_true")
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_dir = default_output_dir(args)
    tracks, results_path = choose_tracks_or_rerun(args, output_dir)
    map_name = infer_single_map([record for track in tracks for record in track], args.map_name)

    data_dir = args.data_dir
    if data_dir is None and args.csgo_config:
        with open(args.csgo_config, "r", encoding="utf-8") as f:
            data_dir = yaml.safe_load(f).get("data_dir", "data/preprocessed_data")
    data_dir = data_dir or "data/preprocessed_data"

    if args.output_video:
        video_output_dir = Path(args.output_video).parent
    elif args.output_dir:
        video_output_dir = output_dir
    else:
        video_output_dir = output_dir / f"three_col_loc_videos_{map_name}"
    pred_search_dirs = []
    if args.pred_fps_dir:
        pred_search_dirs.append(Path(args.pred_fps_dir))
    pred_search_dirs.extend([
        output_dir / "gen_imgs" / map_name,
        output_dir / "gen_imgs",
        output_dir / "pred_imgs" / map_name,
        output_dir / "pred_imgs",
    ])
    print(f"Using localization results: {results_path}")
    print(f"Selected tracks: {len(tracks)}")
    print(f"Map: {map_name}")
    output_paths = output_videos_for_tracks(
        tracks,
        args=args,
        output_dir=video_output_dir,
        data_dir=data_dir,
        map_name=map_name,
        pred_search_dirs=pred_search_dirs,
    )
    print(f"Generated {len(output_paths)} videos in: {video_output_dir}")


if __name__ == "__main__":
    main()
