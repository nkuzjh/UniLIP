import argparse
import copy
import datetime
import json
import os
import random
import textwrap
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from safetensors.torch import load_file as safe_load_file
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from csgo_datasets.unified_task_dataset import (
    DataCollatorForUniLIPMultiTaskDataset,
    UniLIPMultiTaskBalancedDataset,
    UniLIPMultiTaskDataset,
)
from eval_csgo_loc import (
    InferenceArgs,
    smart_matching_state_dict_keys,
    smart_tokenizer_and_embedding_resize,
    unnormalize_pose,
)
from unilip import conversation as conversation_lib
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor01_to_uint8(image_01: torch.Tensor) -> np.ndarray:
    image_01 = image_01.detach().float().cpu().clamp(0.0, 1.0)
    return (image_01.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def processor_tensor_to_uint8(image_tensor: torch.Tensor, image_processor) -> np.ndarray:
    if image_tensor is None:
        return None

    image_tensor = image_tensor.detach().float().cpu()
    mean = getattr(image_processor, "image_mean", None)
    std = getattr(image_processor, "image_std", None)
    if mean is None or std is None:
        return tensor01_to_uint8(image_tensor)

    mean_tensor = torch.tensor(mean, dtype=image_tensor.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=image_tensor.dtype).view(-1, 1, 1)
    image_01 = (image_tensor * std_tensor + mean_tensor).clamp(0.0, 1.0)
    return (image_01.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def build_output_dir(args: argparse.Namespace, csgo_config_path: str) -> Path:
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs_debug") / "aux_loc_vis" / Path(csgo_config_path).stem / f"vis_{cur_time}"
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    return output_dir


def configure_model_from_csgo_config(model, inference_args: InferenceArgs, csgo_config: Dict[str, object]) -> None:
    for key, value in csgo_config.items():
        setattr(model.config, key, value)

    model.config.img_size = getattr(inference_args, "img_size", csgo_config.get("img_size", 448))
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
    model.config.is_action_dit_projector = getattr(
        inference_args,
        "is_action_dit_projector",
        csgo_config.get("is_action_dit_projector", False),
    )
    model.config.is_loc_learnable_query = getattr(
        inference_args,
        "is_loc_learnable_query",
        csgo_config.get("is_loc_learnable_query", False),
    )


def load_model_and_tokenizer(csgo_config: Dict[str, object], ckpt_path: str, device: str):
    disable_torch_init()
    inference_args = InferenceArgs(csgo_config)

    model = Unified_UniLIP_InternVLForCausalLM.from_pretrained(
        csgo_config.get("model_name_or_path", "UniLIP-1B"),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoProcessor.from_pretrained(inference_args.mllm_hf_path).tokenizer
    tokenizer.model_max_length = inference_args.model_max_length
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(
                pad_token="<pad>",
                additional_special_tokens=["[IMG]", "[/IMG]", "<image>"],
            ),
            tokenizer=tokenizer,
            model=model,
        )
    elif "<image>" not in tokenizer.get_added_vocab():
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(additional_special_tokens=["[IMG]", "[/IMG]", "<image>"]),
            tokenizer=tokenizer,
            model=model,
        )

    if inference_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[inference_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama3"]

    configure_model_from_csgo_config(model, inference_args, csgo_config)
    model.get_model().initialize_vision_modules(model_args=inference_args)
    model.get_model().initialize_localization_modules(model_args=inference_args)

    if csgo_config.get("is_lora", False):
        training_args = SimpleNamespace(
            is_lora=csgo_config.get("is_lora", True),
            lora_r=csgo_config.get("lora_r", 16),
            lora_alpha=csgo_config.get("lora_alpha", 16),
            lora_dropout=csgo_config.get("lora_dropout", 0.05),
        )
        model.inject_lora_to_sub_module(inference_args, training_args)

    if ckpt_path.endswith(".safetensors"):
        state_dict = safe_load_file(ckpt_path, device="cpu")
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = smart_matching_state_dict_keys(state_dict, model)
    model.load_state_dict(state_dict, strict=False)

    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    for _, param in model.named_parameters():
        param.requires_grad = False
    model.config.use_cache = True

    image_processor = AutoProcessor.from_pretrained(inference_args.mllm_hf_path).image_processor
    inference_args.image_processor = image_processor
    return model, tokenizer, inference_args


def build_generation_dataloader(
    csgo_config: Dict[str, object],
    tokenizer,
    data_args: InferenceArgs,
    batch_size: int,
) -> DataLoader:
    dataset_config = copy.deepcopy(csgo_config)
    if not dataset_config.get("is_multi_task_balanced", False):
        dataset_config["task_mix_ratio"] = 0.0

    dataset_cls = UniLIPMultiTaskBalancedDataset if dataset_config.get("is_multi_task_balanced", False) else UniLIPMultiTaskDataset
    dataset = dataset_cls(dataset_config, tokenizer, data_args)
    collator = DataCollatorForUniLIPMultiTaskDataset(tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)


def build_map_z_ranges(csgo_config: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    map_z_ranges = {}
    for map_name in csgo_config.get("train_maps", []):
        split_path = Path(csgo_config["data_dir"]) / map_name / "splits_20000_5000" / "train_split.json"
        if not split_path.exists():
            continue
        with open(split_path, "r", encoding="utf-8") as f:
            positions = json.load(f)
        z_values = [item["z"] for item in positions]
        map_z_ranges[map_name] = {
            "min_z": float(min(z_values)),
            "max_z": float(max(z_values)),
        }
    return map_z_ranges


def subset_batch(batch: Dict[str, object], indices: torch.Tensor) -> Dict[str, object]:
    subset = {}
    batch_size = len(batch["task_id"])
    index_list = indices.tolist()
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[0] == batch_size:
            subset[key] = value.index_select(0, indices)
        elif isinstance(value, list) and len(value) == batch_size:
            subset[key] = [value[i] for i in index_list]
        else:
            subset[key] = value
    return subset


def wrap_angle_norm_diff(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(pred - gt)
    diff[..., 4] = torch.minimum(diff[..., 4], 1.0 - diff[..., 4])
    return diff


def compute_full_pose_metrics(pred_norm: torch.Tensor, gt_norm: torch.Tensor, pred_phys: Dict[str, float], gt_phys: Dict[str, float]) -> Dict[str, float]:
    pred_norm = pred_norm.float()
    gt_norm = gt_norm.float()
    norm_abs_diff = wrap_angle_norm_diff(pred_norm.unsqueeze(0), gt_norm.unsqueeze(0)).squeeze(0)
    xy_dist = float(((pred_phys["x"] - gt_phys["x"]) ** 2 + (pred_phys["y"] - gt_phys["y"]) ** 2) ** 0.5)
    z_dist = float(abs(pred_phys["z"] - gt_phys["z"]))
    pitch_dist = float(abs(pred_phys["angle_v"] - gt_phys["angle_v"]))
    yaw_raw = abs(pred_phys["angle_h"] - gt_phys["angle_h"])
    yaw_dist = float(min(yaw_raw, 360.0 - yaw_raw))
    return {
        "Norm_L2_5D": float(torch.norm(norm_abs_diff, p=2).item()),
        "Norm_L2_XY": float(torch.norm(norm_abs_diff[:2], p=2).item()),
        "XY_Dist": xy_dist,
        "Z_Dist": z_dist,
        "Pitch_Dist": pitch_dist,
        "Yaw_Dist": yaw_dist,
    }


def build_overlay_map(radar_img_uint8: np.ndarray, gt_pose: Dict[str, float], clean_pose: Dict[str, float], aux_pose: Dict[str, float]) -> Image.Image:
    image = Image.fromarray(radar_img_uint8).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    scale_x = image.width / 1024.0
    scale_y = image.height / 1024.0

    def to_xy(pose):
        return pose["x"] * scale_x, pose["y"] * scale_y

    gt_xy = to_xy(gt_pose)
    clean_xy = to_xy(clean_pose)
    aux_xy = to_xy(aux_pose)

    draw.line([gt_xy, clean_xy], fill=(0, 255, 0, 180), width=3)
    draw.line([gt_xy, aux_xy], fill=(255, 0, 0, 180), width=3)
    draw.ellipse((gt_xy[0] - 6, gt_xy[1] - 6, gt_xy[0] + 6, gt_xy[1] + 6), fill=(255, 255, 255, 220), outline="black")
    draw.ellipse((clean_xy[0] - 7, clean_xy[1] - 7, clean_xy[0] + 7, clean_xy[1] + 7), outline=(0, 255, 0, 255), width=3)
    draw.ellipse((aux_xy[0] - 7, aux_xy[1] - 7, aux_xy[0] + 7, aux_xy[1] + 7), outline=(255, 0, 0, 255), width=3)
    return Image.alpha_composite(image, overlay).convert("RGB")


def build_result_text(
    file_frame: str,
    map_name: str,
    prompt: str,
    sigma_gen: float,
    timestep_gen: float,
    loc_time: float,
    full_clean: Dict[str, float],
    full_aux: Dict[str, float],
    clean_step: Dict[str, float],
    aux_step: Dict[str, float],
) -> List[str]:
    meta_text = "\n".join(
        [
            f"ID: {file_frame}",
            f"Map: {map_name}",
            f"Gen timestep: {timestep_gen:.1f}",
            f"Gen sigma: {sigma_gen:.4f}",
            f"Loc time: {loc_time:.4f}",
            "",
            "Prompt:",
            "\n".join(textwrap.wrap(prompt, width=36)),
        ]
    )
    metric_text = "\n".join(
        [
            "Full Pose Inference",
            f"Clean XY/Z/P/Y: {full_clean['XY_Dist']:.2f} / {full_clean['Z_Dist']:.2f} / {full_clean['Pitch_Dist']:.2f} / {full_clean['Yaw_Dist']:.2f}",
            f"Aux   XY/Z/P/Y: {full_aux['XY_Dist']:.2f} / {full_aux['Z_Dist']:.2f} / {full_aux['Pitch_Dist']:.2f} / {full_aux['Yaw_Dist']:.2f}",
            f"Clean Norm_L2_5D: {full_clean['Norm_L2_5D']:.4f}",
            f"Aux   Norm_L2_5D: {full_aux['Norm_L2_5D']:.4f}",
            "",
            "Single-Step Velocity",
            f"Clean raw_mse/raw_l1: {clean_step['raw_mse']:.6f} / {clean_step['raw_l1']:.6f}",
            f"Aux   raw_mse/raw_l1: {aux_step['raw_mse']:.6f} / {aux_step['raw_l1']:.6f}",
            f"Clean weighted_mse/l1: {clean_step['weighted_mse']:.6f} / {clean_step['weighted_l1']:.6f}",
            f"Aux   weighted_mse/l1: {aux_step['weighted_mse']:.6f} / {aux_step['weighted_l1']:.6f}",
        ]
    )
    return [meta_text, metric_text]


def save_sample_visualization(
    output_dir: Path,
    *,
    file_frame: str,
    map_name: str,
    prompt: str,
    radar_img_uint8: np.ndarray,
    gt_img_uint8: np.ndarray,
    pred_img_uint8: np.ndarray,
    overlay_map: Image.Image,
    sigma_gen: float,
    timestep_gen: float,
    loc_time: float,
    full_clean: Dict[str, float],
    full_aux: Dict[str, float],
    clean_step: Dict[str, float],
    aux_step: Dict[str, float],
) -> Dict[str, str]:
    diff_map = np.abs(pred_img_uint8.astype(np.int16) - gt_img_uint8.astype(np.int16)).mean(axis=-1)
    text_left, text_right = build_result_text(
        file_frame=file_frame,
        map_name=map_name,
        prompt=prompt,
        sigma_gen=sigma_gen,
        timestep_gen=timestep_gen,
        loc_time=loc_time,
        full_clean=full_clean,
        full_aux=full_aux,
        clean_step=clean_step,
        aux_step=aux_step,
    )

    fig, axes = plt.subplots(1, 7, figsize=(36, 6))
    axes[0].text(0.0, 1.0, text_left, ha="left", va="top", fontsize=10, wrap=True)
    axes[0].axis("off")
    axes[0].set_title("Meta", fontsize=12, fontweight="bold")

    axes[1].imshow(radar_img_uint8)
    axes[1].axis("off")
    axes[1].set_title("Radar", fontsize=12, fontweight="bold")

    axes[2].imshow(gt_img_uint8)
    axes[2].axis("off")
    axes[2].set_title("GT FPS", fontsize=12, fontweight="bold")

    axes[3].imshow(pred_img_uint8)
    axes[3].axis("off")
    axes[3].set_title("Aux-Loc x0_hat", fontsize=12, fontweight="bold")

    axes[4].imshow(diff_map, cmap="inferno")
    axes[4].axis("off")
    axes[4].set_title("|Pred - GT|", fontsize=12, fontweight="bold")

    axes[5].imshow(overlay_map)
    axes[5].axis("off")
    axes[5].set_title("Radar Overlay", fontsize=12, fontweight="bold")
    axes[5].text(
        0.98,
        0.02,
        "White dot: GT\nGreen circle: Clean pred\nRed circle: Aux pred",
        transform=axes[5].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="white",
        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=3.0),
    )

    axes[6].text(0.0, 1.0, text_right, ha="left", va="top", fontsize=10, wrap=True)
    axes[6].axis("off")
    axes[6].set_title("Loc Metrics", fontsize=12, fontweight="bold")

    plt.tight_layout()
    image_path = output_dir / "images" / f"{file_frame}.jpg"
    fig.savefig(image_path, dpi=120)
    plt.close(fig)
    return {"image_path": str(image_path)}


def summarize_records(records: List[Dict[str, object]]) -> Dict[str, float]:
    metric_pool = defaultdict(list)
    for item in records:
        for prefix in ["full_infer_clean_metrics", "full_infer_aux_metrics", "single_step_clean", "single_step_aux"]:
            for key, value in item[prefix].items():
                if isinstance(value, (int, float)):
                    metric_pool[f"{prefix}.{key}"].append(float(value))
    return {key: float(np.mean(values)) for key, values in metric_pool.items() if len(values) > 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize aux_loc debug paths for CSGO unified UniLIP.")
    parser.add_argument("--csgo_config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--loc_num_steps", type=int, default=10)
    args = parser.parse_args()

    with open(args.csgo_config, "r", encoding="utf-8") as f:
        csgo_config = yaml.safe_load(f)

    ckpt_path = args.ckpt_path or csgo_config.get("ckpt_path")
    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint path invalid: {ckpt_path}")

    if csgo_config.get("use_external_loc_model", False):
        raise ValueError("debug_aux_loc_vis.py currently only supports the internal loc head path.")
    if csgo_config.get("use_vit_regression_head", False) or csgo_config.get("use_vit_cls_regression_head", False):
        raise ValueError("debug_aux_loc_vis.py currently only supports the action-DiT loc branch.")
    if csgo_config.get("use_codex_vit_regression_head", False):
        raise ValueError("debug_aux_loc_vis.py currently only supports the action-DiT loc branch.")

    set_seed(args.seed)
    output_dir = build_output_dir(args, args.csgo_config)

    model, tokenizer, inference_args = load_model_and_tokenizer(csgo_config, ckpt_path, args.device)
    dataloader = build_generation_dataloader(csgo_config, tokenizer, inference_args, args.batch_size)
    map_z_ranges = build_map_z_ranges(csgo_config)

    records = []
    sample_count = 0

    for batch in dataloader:
        task_ids = batch["task_id"]
        gen_indices = (task_ids == 1).nonzero(as_tuple=True)[0]
        if gen_indices.numel() == 0:
            continue
        gen_batch = subset_batch(batch, gen_indices)

        current_bs = len(gen_batch["task_id"])
        remaining = args.num_samples - sample_count
        if current_bs > remaining:
            keep_indices = torch.arange(remaining, dtype=torch.long)
            gen_batch = subset_batch(gen_batch, keep_indices)
            current_bs = remaining

        gen_input_ids = gen_batch["input_ids"].to(args.device)
        gen_attention_mask = gen_batch["attention_mask"].to(args.device)
        gen_labels = gen_batch["labels"].to(args.device)
        gen_image = gen_batch["gen_image"].to(args.device)
        map_image = gen_batch["und_image"].to(args.device)
        aux_image = gen_batch["aux_image"].to(args.device)
        aux_loc_input_ids = gen_batch["aux_loc_input_ids"].to(args.device)
        aux_loc_labels = gen_batch["aux_loc_labels"].to(args.device)
        aux_loc_attention_mask = gen_batch["aux_loc_attention_mask"].to(args.device)
        actions = gen_batch["actions"].to(args.device)
        task_id_gen = gen_batch["task_id"].to(args.device)
        task_id_loc = torch.zeros_like(task_id_gen)

        with torch.no_grad():
            aux_debug = model.debug_build_aux_loc_pred_pixels(
                input_ids=gen_input_ids,
                attention_mask=gen_attention_mask,
                labels=gen_labels,
                gen_image=gen_image,
                und_image=map_image,
                aux_image=aux_image,
                task_id=task_id_gen,
            )
            pred_pixels_input = aux_debug["pred_pixels_input"]
            pred_pixels_norm = aux_debug["pred_pixels_norm"]
            timesteps_gen = aux_debug["timesteps_gen"]
            sigmas_gen = aux_debug["sigmas_gen"]

            clean_pred_norm = model.generate_action2(
                input_ids=aux_loc_input_ids,
                attention_mask=aux_loc_attention_mask,
                labels=aux_loc_labels,
                und_image=gen_image,
                aux_image=map_image,
                num_steps=args.loc_num_steps,
            ).squeeze(1).float().cpu()
            aux_pred_norm = model.generate_action2(
                input_ids=aux_loc_input_ids,
                attention_mask=aux_loc_attention_mask,
                labels=aux_loc_labels,
                und_image=pred_pixels_input,
                aux_image=map_image,
                num_steps=args.loc_num_steps,
            ).squeeze(1).float().cpu()

            step_debug = model.debug_compare_clean_vs_aux_single_step_loc(
                clean_fps_input=gen_image,
                aux_fps_input=pred_pixels_input,
                map_image_input=map_image,
                aux_loc_input_ids=aux_loc_input_ids,
                aux_loc_labels=aux_loc_labels,
                aux_loc_attention_mask=aux_loc_attention_mask,
                actions=actions,
                gen_sigma_for_weight=sigmas_gen,
                gen_image=gen_image,
                task_id=task_id_loc,
            )

        clean_step = step_debug["clean_metrics"]
        aux_step = step_debug["aux_metrics"]
        shared_loc_time = step_debug["shared_loc_time"].float().cpu()

        for sample_idx in range(current_bs):
            file_frame = str(gen_batch["ids"][sample_idx]).replace("_gen", "")
            map_name = gen_batch["map_name"][sample_idx]
            gt_pose_phys = gen_batch["pose_dict"][sample_idx]
            gt_pose_norm = gen_batch["actions"][sample_idx].float().cpu()
            z_range = map_z_ranges.get(map_name)
            if z_range is None:
                z_range = {"min_z": float(gt_pose_phys["z"]), "max_z": float(gt_pose_phys["z"])}

            clean_pred_phys = unnormalize_pose(clean_pred_norm[sample_idx], z_range)
            aux_pred_phys = unnormalize_pose(aux_pred_norm[sample_idx], z_range)
            full_clean_metrics = compute_full_pose_metrics(clean_pred_norm[sample_idx], gt_pose_norm, clean_pred_phys, gt_pose_phys)
            full_aux_metrics = compute_full_pose_metrics(aux_pred_norm[sample_idx], gt_pose_norm, aux_pred_phys, gt_pose_phys)

            radar_img_uint8 = processor_tensor_to_uint8(
                gen_batch["und_image"][sample_idx],
                inference_args.image_processor,
            )
            gt_img_uint8 = processor_tensor_to_uint8(
                gen_batch["gen_image"][sample_idx],
                inference_args.image_processor,
            )
            pred_img_uint8 = tensor01_to_uint8(pred_pixels_norm[sample_idx])
            overlay_map = build_overlay_map(radar_img_uint8, gt_pose_phys, clean_pred_phys, aux_pred_phys)

            clean_step_sample = {
                "raw_mse": float(clean_step["raw_mse"][sample_idx].cpu().item()),
                "raw_l1": float(clean_step["raw_l1"][sample_idx].cpu().item()),
                "weighted_mse": float(clean_step["weighted_mse"][sample_idx].cpu().item()),
                "weighted_l1": float(clean_step["weighted_l1"][sample_idx].cpu().item()),
            }
            aux_step_sample = {
                "raw_mse": float(aux_step["raw_mse"][sample_idx].cpu().item()),
                "raw_l1": float(aux_step["raw_l1"][sample_idx].cpu().item()),
                "weighted_mse": float(aux_step["weighted_mse"][sample_idx].cpu().item()),
                "weighted_l1": float(aux_step["weighted_l1"][sample_idx].cpu().item()),
            }

            vis_paths = save_sample_visualization(
                output_dir,
                file_frame=file_frame,
                map_name=map_name,
                prompt=gen_batch["raw_prompt"][sample_idx],
                radar_img_uint8=radar_img_uint8,
                gt_img_uint8=gt_img_uint8,
                pred_img_uint8=pred_img_uint8,
                overlay_map=overlay_map,
                sigma_gen=float(sigmas_gen[sample_idx].reshape(-1)[0].float().cpu().item()),
                timestep_gen=float(timesteps_gen[sample_idx].float().cpu().item()),
                loc_time=float(shared_loc_time[sample_idx].item()),
                full_clean=full_clean_metrics,
                full_aux=full_aux_metrics,
                clean_step=clean_step_sample,
                aux_step=aux_step_sample,
            )

            records.append(
                {
                    "file_frame": file_frame,
                    "map_name": map_name,
                    "prompt": gen_batch["raw_prompt"][sample_idx],
                    "gen_timestep": float(timesteps_gen[sample_idx].float().cpu().item()),
                    "gen_sigma": float(sigmas_gen[sample_idx].reshape(-1)[0].float().cpu().item()),
                    "loc_time": float(shared_loc_time[sample_idx].item()),
                    "gt_pose_norm": gt_pose_norm.tolist(),
                    "gt_pose_phys": gt_pose_phys,
                    "clean_pred_norm": clean_pred_norm[sample_idx].tolist(),
                    "clean_pred_phys": clean_pred_phys,
                    "aux_pred_norm": aux_pred_norm[sample_idx].tolist(),
                    "aux_pred_phys": aux_pred_phys,
                    "full_infer_clean_metrics": full_clean_metrics,
                    "full_infer_aux_metrics": full_aux_metrics,
                    "single_step_clean": clean_step_sample,
                    "single_step_aux": aux_step_sample,
                    "vis_image_path": vis_paths["image_path"],
                }
            )

        sample_count += current_bs
        if sample_count >= args.num_samples:
            break

    summary = {
        "csgo_config": args.csgo_config,
        "ckpt_path": ckpt_path,
        "num_samples": len(records),
        "seed": args.seed,
        "summary_metrics": summarize_records(records),
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved debug visualization results to: {output_dir}")


if __name__ == "__main__":
    main()


'''

CUDA_VISIBLE_DEVICES=0 python debug_aux_loc_vis.py --csgo_config csgo_configs/exp20_dust2.yaml --ckpt_path outputs/csgo_1b/exp20_dust2/checkpoint-8000/model.safetensors --num_samples 10 --batch_size 1


CUDA_VISIBLE_DEVICES=0 python debug_aux_loc_vis.py --csgo_config csgo_configs/exp20_dust2.yaml --ckpt_path outputs/csgo_1b/exp17_2_dust2/checkpoint-8000/model.safetensors --num_samples 10 --batch_size 1

'''
