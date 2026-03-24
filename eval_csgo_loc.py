import argparse
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import random
import datetime
from PIL import Image, ImageDraw
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor
from safetensors.torch import load_file as safe_load_file
from types import SimpleNamespace
from collections import defaultdict

# 引入 UniLIP 核心模块
from unilip.utils import disable_torch_init
from unilip.model import Unified_UniLIP_InternVLForCausalLM

# ==========================================
# 0. 基础工具 & Pi0.5 复用函数
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"随机种子已设置为: {seed}")

def vertical_concat(images):
    """复用 Pi0.5: 将多张 PIL Image 纵向拼接"""
    if not images: return None
    width = images[0].width
    total_height = sum(img.height for img in images)
    new_image = Image.new(images[0].mode, (width, total_height))
    y_offset = 0
    for img in images:
        if img.width != width:
            img = img.resize((width, img.height), Image.LANCZOS)
        new_image.paste(img, (0, y_offset))
        y_offset += img.height
    return new_image

def concat_images_horizontal_resize(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """复用 Pi0.5: 横向拼接，强制 img2 高度适配 img1"""
    if img1 is None or img2 is None: return img1 if img1 else img2
    w1, h1 = img1.size
    w2, h2 = img2.size
    aspect_ratio = h1 / h2
    new_w2 = int(w2 * aspect_ratio)
    new_h2 = h1
    img2_resized = img2.resize((new_w2, new_h2), Image.Resampling.LANCZOS)
    total_width = w1 + new_w2
    dst = Image.new(img1.mode, (total_width, h1))
    dst.paste(img1, (0, 0))
    dst.paste(img2_resized, (w1, 0))
    return dst


# ==========================================
# 1. 核心逻辑: Metric 计算 (升级版)
# ==========================================
def calculate_metrics(results, ckpt_path):
    """
    计算定位任务的核心指标 (包含 L2 Norm, SmoothL1, MSE)
    results: List[Dict], 包含 'gt', 'pred' (已经是反归一化后的物理坐标)
    """
    gt_norm = torch.tensor([item['gt_norm'] for item in results], dtype=torch.float32)
    pred_norm = torch.tensor([item['pred_norm'] for item in results], dtype=torch.float32)

    # --- 1. 计算绝对差值 (Abs Diff) ---
    # abs_diff = |pred - gt|
    norm_abs_diff = torch.abs(pred_norm - gt_norm)

    # --- 2. 处理 Yaw (Index 4) 的周期性 ---
    # 在 0~1 空间中，周期是 1.0
    # 修正后的误差 = min(raw_diff, 1.0 - raw_diff)
    yaw_diff = norm_abs_diff[:, 4]
    yaw_diff_wrapped = torch.min(yaw_diff, 1.0 - yaw_diff)
    norm_abs_diff[:, 4] = yaw_diff_wrapped # 更新 Diff Tensor

    norm_l2_xy = torch.norm(norm_abs_diff[:, :2], p=2, dim=1).mean().item()
    # 2. 5D L2 Norm (Normalized Space)
    # sqrt(dx^2 + dy^2 + dz^2 + dpitch^2 + dyaw^2)
    norm_l2_5d = torch.norm(norm_abs_diff, p=2, dim=1).mean().item()

    # ==========================================
    # Metric B: Losses (MSE & SmoothL1)
    # ==========================================
    zeros_target = torch.zeros_like(norm_abs_diff)
    # 1. MSE Loss (5D)
    # mean(error^2)
    norm_mse_loss_5d = F.mse_loss(norm_abs_diff, zeros_target, reduction='mean').item()
    # 2. SmoothL1 Loss (5D)
    # 这里的 beta=1.0 是默认值，可根据训练时的设定调整
    norm_smooth_l1_loss_5d = F.smooth_l1_loss(norm_abs_diff, zeros_target, reduction='mean', beta=1.0).item()

    # 1. 提取数据并转换为 Tensor 以利用 PyTorch 的 Loss 函数
    gt_list = []
    pred_list = []
    for item in results:
        gt = item['gt']
        pred = item['pred']
        # 构造向量 [x, y, z, pitch, yaw]
        gt_vec = [gt['x'], gt['y'], gt['z'], gt['angle_v'], gt['angle_h']]
        pred_vec = [pred['x'], pred['y'], pred['z'], pred['angle_v'], pred['angle_h']]
        gt_list.append(gt_vec)
        pred_list.append(pred_vec)

    gt_tensor = torch.tensor(gt_list, dtype=torch.float32)     # [N, 5]
    pred_tensor = torch.tensor(pred_list, dtype=torch.float32) # [N, 5]

    # --- 预处理: 处理 Yaw (Index 4) 的 360 度周期性 ---
    # 我们计算差值 diff，而不是直接用 pred，这样 Loss 计算才符合物理直觉
    # diff = pred - gt
    diff_tensor = pred_tensor - gt_tensor
    abs_diff = torch.abs(diff_tensor)

    # 对 Yaw (第4列) 做 wrap 处理: diff = (diff + 180) % 360 - 180
    # 或者更简单的 min(|d|, 360-|d|) 逻辑，但为了保留符号给 Loss 用，通常取最小夹角
    # 这里为了 L2/MSE/SmoothL1 计算距离（即误差大小），我们取绝对误差
    # 修正 Yaw 的绝对误差: min(err, 360 - err)
    yaw_err = abs_diff[:, 4]
    yaw_err = torch.min(yaw_err, 360.0 - yaw_err)
    abs_diff[:, 4] = yaw_err

    # --- Metric 1: 单项误差 ---
    xy_dist = torch.sqrt(abs_diff[:, 0]**2 + abs_diff[:, 1]**2).mean().item()
    z_dist = abs_diff[:, 2].mean().item()
    pitch_dist = abs_diff[:, 3].mean().item()
    yaw_dist = abs_diff[:, 4].mean().item()

    # --- Metric 2: XY L2 Norm (同 Mean XY Error) ---
    l2_xy = torch.norm(abs_diff[:, :2], p=2, dim=1).mean().item()
    # --- Metric 3: 5D L2 Norm ---
    # 定义：sqrt(dx^2 + dy^2 + dz^2 + dp^2 + dyaw^2)
    l2_5d = torch.norm(abs_diff, p=2, dim=1).mean().item()
    # --- Metric 4: 5D Loss (MSE & SmoothL1) ---
    # 这里的 Loss 是基于物理坐标的，所以数值会很大，但能反映物理偏离程度
    # 使用 abs_diff 作为输入，也就是计算 Loss(abs_diff, 0)
    mse_loss_5d = F.mse_loss(abs_diff, torch.zeros_like(abs_diff))
    smooth_l1_loss_5d = F.smooth_l1_loss(abs_diff, torch.zeros_like(abs_diff), beta=1.0)

    metrics = {
        "Norm_L2_XY": norm_l2_xy,
        "Norm_L2_5D": norm_l2_5d,
        "Norm_MSE_5D": norm_mse_loss_5d,
        "Norm_SmoothL1_5D": norm_smooth_l1_loss_5d,

        "XY_Dist": xy_dist,
        "Z_Dist": z_dist,
        "Pitch_Dist": pitch_dist,
        "Yaw_Dist": yaw_dist,

        # 顺便保留单项的平均绝对误差 (L1)
        "Norm_L1_X": abs_diff[:, 0].mean().item(),
        "Norm_L1_Y": abs_diff[:, 1].mean().item(),
        "Norm_L1_Z": abs_diff[:, 2].mean().item(),
        "Norm_L1_Pitch": abs_diff[:, 3].mean().item(),
        "Norm_L1_Yaw": abs_diff[:, 4].mean().item(),

        "L2_XY": l2_xy,       # 实际上等于 XY_Dist
        "L2_5D": l2_5d,
        "MSE_Loss_5D": mse_loss_5d.item(),
        "SmoothL1_Loss_5D": smooth_l1_loss_5d.item(),

        "ckpt_path": ckpt_path,
    }
    return metrics

# ==========================================
# 2. 可视化逻辑 (Pi0.5 风格)
# ==========================================
def visualize_map_results(vis_data_grouped, output_dir, map_path_dict, data_dir_base, vis_num_per_map=5):
    """
    按地图分组可视化：左侧 Radar 叠加 GT/Pred 点，右侧 FPS 列表
    """
    print(f"📊 Generating Pi0.5 Style Visualizations for {len(vis_data_grouped)} maps...")

    # 颜色列表 (用于区分不同样本)
    color_list = [
        (255, 0, 0, 180),   # Red
        (0, 255, 0, 180),   # Green
        (0, 0, 255, 180),   # Blue
        (255, 255, 0, 180), # Yellow
        (255, 0, 255, 180)  # Magenta
    ]

    for map_name, items in vis_data_grouped.items():
        # 限制每张图显示的样本数
        items_to_show = items[:vis_num_per_map]

        # 1. 加载底图 (Radar)
        map_filename = map_path_dict.get(map_name, 'de_dust2_radar_psd.png')
        map_path = os.path.join(data_dir_base, map_name, map_filename)
        if not os.path.exists(map_path):
            print(f"⚠️ Map image not found: {map_path}")
            continue

        base_map = Image.open(map_path).convert('RGBA')

        # 创建绘制层
        overlay = Image.new("RGBA", base_map.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        fps_img_list = []

        for i, item in enumerate(items_to_show):
            color = color_list[i % len(color_list)]

            # --- Draw GT (Solid Circle) ---
            # GT x,y 已经是 0-1024 范围，直接对应图片像素 (假设地图是 1024x1024)
            # 如果地图尺寸不是 1024，需要缩放
            scale_x = base_map.width / 1024.0
            scale_y = base_map.height / 1024.0

            gt_x, gt_y = item['gt']['x'] * scale_x, item['gt']['y'] * scale_y
            pred_x, pred_y = item['pred']['x'] * scale_x, item['pred']['y'] * scale_y

            # GT: Solid Circle
            r_gt = 6
            draw.ellipse((gt_x - r_gt, gt_y - r_gt, gt_x + r_gt, gt_y + r_gt), fill=color, outline='black', width=1)

            # --- Draw Pred (Hollow/X or different style) ---
            r_pred = 8
            # 绘制预测点 (带黑框的半透明点)
            draw.ellipse((pred_x - r_pred, pred_y - r_pred, pred_x + r_pred, pred_y + r_pred), fill=None, outline=color, width=3)
            # 可选：绘制连线
            draw.line([(gt_x, gt_y), (pred_x, pred_y)], fill='white', width=2)

            # --- Collect FPS Image ---
            # 在 FPS 图上画个框标记颜色对应关系
            fps_img = item['fps'].convert('RGB').copy()
            fps_draw = ImageDraw.Draw(fps_img)
            fps_draw.rectangle((0, 0, 40, 40), fill=color[:3]) # 左上角色块
            fps_img_list.append(fps_img)

        # 2. 合成 Radar + Overlay
        combined_map = Image.alpha_composite(base_map, overlay).convert('RGB')

        # 3. 合成 FPS Strip
        fps_strip = vertical_concat(fps_img_list)

        # 4. 最终横向拼接
        final_vis = concat_images_horizontal_resize(combined_map, fps_strip)

        # 保存
        save_path = os.path.join(output_dir, f"vis_map_{map_name}.jpg")
        final_vis.save(save_path)
        print(f"   -> Saved: {save_path}")


####### 从unified_task_dataset.py引入和训练一致的预处理函数
####### 简单的预处理辅助类无法准确拼接UniLIP的 <image> token
import copy
from csgo_datasets.unified_task_dataset import get_loc_prompt, preprocess_multimodal, preprocess, img_process, img_resize_transform
# ==========================================
# 3. 辅助类与 Dataset (精简版)
# ==========================================
# # ... (Prompt 模板代码同前) ...
# def get_loc_prompt(map_name):
#     # return f"Task: The following visual data of CS2 map '{map_name}' has been fused... Predict the 5D pose...\n<image>\n<image>"
#     LOC_PROMPT_TEMPLATE = (
#         f"Task: The following visual data of CS2 map '{map_name}' has been fused and inserted into this sequence:\n"
#         "1. First-Person View (FPV) Features.\n"
#         "2. Overhead Radar Map (RADAR) Features.\n"
#         "Analyze the spatial relationship between the FPV and the RADAR Map to determine the precise camera pose. "
#         "Predict the 5D pose (x, y, z, pitch, yaw) in the required format.\n"
#         "<image>\n<image>"
#     )
#     return LOC_PROMPT_TEMPLATE

# def add_template_for_loc(prompt_text):
#     return f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

# 模拟 ModelArguments
class InferenceArgs:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)
        ##### model_args
        # 补充 initialize_vision_modules 需要的默认参数 (如果 yaml 里没有)
        self.unilip_path = config_dict.get("unilip_path", "")
        self.unilip_factor = config_dict.get("unilip_factor", 5.85)
        self.fix_dit = False
        self.fix_connect = False
        self.fix_vit = True
        self.fix_llm = True
        self.mllm_path = config_dict.get("mllm_path", "")
        self.mllm_hf_path = config_dict.get("mllm_hf_path", "OpenGVLab/InternVL3-1B-hf")
        self.vae_path = config_dict.get("vae_path", "")
        self.dit_path = config_dict.get("dit_path", "")
        self.n_query = 256
        self.connect_layer = 6
        # 补充 initialize_localization_modules 需要的参数
        self.action_horizon = 1
        self.action_dim = 5
        self.is_action_dit_dense_timestep = config_dict.get("is_action_dit_dense_timestep", False)
        self.action_dit_layer = config_dict.get("action_dit_layer", 3)
        # 其他杂项
        self.mm_use_im_patch_token = config_dict.get("mm_use_im_patch_token", False)
        self.mm_use_im_start_end = config_dict.get("mm_use_im_start_end", False)
        self.tune_mm_mlp_adapter = False
        self.pretrain_mm_mlp_adapter = None
        self.version = "internvl"

        ##### data_args
        self.image_aspect_ratio = "square"
        ##### training_args
        self.model_max_length = 1024
        self.is_action_dit_projector = config_dict.get("is_action_dit_projector", False)
        self.is_loc_learnable_query = config_dict.get("is_loc_learnable_query", False)

        # lora
        is_lora = config_dict.get("is_lora", False)
        lora_r = config_dict.get("lora_r", False)
        lora_alpha = 16
        lora_dropout = 0.05
        lora_weight_path  = ""
        lora_bias = "none"

from csgo_datasets.unified_task_dataset import _build_loc_prompt_cache, _build_map_tensor_cache, _build_csgo_entry

# Dataset
class CSGOLocInferenceDataset(Dataset):
    def __init__(self, config, tokenizer, map_path_dict, image_processor, image_aspect_ratio):
        self.config = config
        self.tokenizer = tokenizer
        self.data_dir = config['data_dir']
        self.map_names = config.get('test_maps', config.get('val_maps', []))
        self.map_path_dict = map_path_dict
        self.image_processor = image_processor
        self.image_aspect_ratio = image_aspect_ratio
        self.data_entries = []
        self.map_z_range = {}
        self.img_size = config.get("img_size", 448)
        self.is_resize_224 = self.img_size == 224

        for map_name in self.map_names:
            split_path = f"{self.data_dir}/{map_name}/splits_20000_5000/test_split.json"
            if not os.path.exists(split_path):
                continue
            print(f"Loading CS2 Data Split {split_path}...")
            with open(split_path, "r", encoding="utf-8") as f:
                positions_data = json.load(f)

            max_z, min_z = -float("inf"), float("inf")
            for pos_data in positions_data:
                if pos_data["z"] > max_z:
                    max_z = pos_data["z"]
                if pos_data["z"] < min_z:
                    min_z = pos_data["z"]
            self.map_z_range[map_name] = {'max_z': max_z, 'min_z': min_z}

            for pos_data in positions_data:
                self.data_entries.append(_build_csgo_entry(map_name, pos_data, min_z, max_z))

        if config['debug'] and config.get('debug_num_val_data', False):
            sampled_num = config.get('debug_num_val_data', len(self.data_entries))
            self.data_entries = self.data_entries[:sampled_num]
            print([data['file_frame'] for data in self.data_entries])
        elif config['debug'] and config.get('debug_num_val_data', False) == False:
            indices = [335, 535, 707, 288, 21, 240, 20, 30, 809, 423, 857, 459, 557, 882, 893, 406, 24, 477, 407, 427, 453, 923, 925, 399, 752, 867, 547, 563, 424, 217, 789, 681]
            self.data_entries = [self.data_entries[i] for i in indices]
            print([data['file_frame'] for data in self.data_entries])
        elif config['debug']==False and config.get('debug_num_val_data', False):
            sampled_num = config.get('debug_num_val_data', len(self.data_entries))
            self.data_entries = random.sample(self.data_entries, sampled_num)
            print([data['file_frame'] for data in self.data_entries])

        # 预加载所有地图图片到内存
        self.map_images = {}
        for map_name, filename in map_path_dict.items():
            path = f"{config['data_dir']}/{map_name}/{filename}"
            if os.path.exists(path):
                img = Image.open(path).convert('RGB').resize((448, 448))
                self.map_images[map_name] = img
            else:
                print(f"Map image not found: {path}")

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
        )

        # 仅取前N个做测试，避免跑太久 (可选)
        # self.data_entries = self.data_entries[:50]
        print(f"✅ Loaded {len(self.data_entries)} test samples.")

    def __len__(self): return len(self.data_entries)

    def __getitem__(self, i):
        data = self.data_entries[i]
        map_name = data['map']
        # map_img_path = f"{self.data_dir}/{map_name}/{self.map_path_dict.get(map_name, 'de_dust2_radar_psd.png')}"
        # map_img = Image.open(map_img_path).convert('RGB')
        map_img = self.map_images.get(map_name).copy()
        map_tensor_448 = self.map_tensor_cache_448[map_name]
        tensor_map = self.map_tensor_cache_224[map_name] if self.is_resize_224 else map_tensor_448

        ext = ".jpg" if self.config['data_dir'] == 'data/preprocessed_data' else ".png"
        fps_path = f"{self.data_dir}/{map_name}/imgs/{data['file_frame']}{ext}"
        fps_img = Image.open(fps_path).convert('RGB')
        fps_tensor_448 = img_process([fps_img], self.image_processor, self.image_aspect_ratio)
        tensor_fps = img_resize_transform(fps_tensor_448) if self.is_resize_224 else fps_tensor_448

        pose_dict = data['pose_dict']
        gt_norm_tensor = data['actions']
        loc_cache = self.loc_prompt_cache[map_name]



        return {
            "task_id": 0,
            "map_name": map_name,
            "ids": data['file_frame'],
            "und_image": tensor_fps,   # Input: FPS
            "aux_image": tensor_map,   # Aux: Map
            "input_ids": loc_cache["input_ids"],
            "labels": loc_cache["labels"],
            "actions": gt_norm_tensor,

            "raw_prompt": loc_cache["raw_prompt"],
            "map_img": map_img,
            "fps_img": fps_img,
            "pose_dict": pose_dict,
            "z_range": {'min_z': data['z_min'], 'max_z': data['z_max']}


        }


# def collate_fn(batch): return batch

from typing import Dict, Sequence, Union, List
from dataclasses import dataclass
import transformers
from unilip.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_IDX

@dataclass
class DataCollatorForLoc(object):
    """
    Collate examples for UniLIP Multi-Task (Localization + Generation).
    Adapts inputs based on task_id.
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Union[Dict, List[Dict]]]) -> Dict[str, torch.Tensor]:
        # 1. 提取基础数据
        task_id_list = []
        input_ids_list = []
        labels_list = []
        ids_list = []
        batch_und_images = []
        batch_aux_images = []
        batch_actions = []
        batch_raw_prompt_list = []
        batch_map_img_list = []
        batch_fps_img_list = []
        batch_z_range_list = []
        batch_map_name_list = []
        batch_pose_dict_list = []

        # 2. 逐样本处理
        for instance in instances:
            task_id = instance.get("task_id", 0) # 默认为0(定位)以防万一
            _input_id = instance["input_ids"]
            _label = instance["labels"]
            _id = instance.get("ids", "unknown")

            # === Token 处理逻辑 ===
            # 为了防止超过模型最大长度，先做截断 (预留 257 个位置给生成 Token)
            # UniLIP 原逻辑：input_id[: max_len - 257]
            safe_len = self.tokenizer.model_max_length - 257
            _input_id = _input_id[:safe_len]
            _label = _label[:safe_len]

            task_id_list.append(task_id)
            input_ids_list.append(_input_id)
            labels_list.append(_label)
            ids_list.append(_id)

            # === 收集 Tensor 数据 ===
            if "und_image" in instance and instance["und_image"] is not None:
                batch_und_images.append(instance["und_image"])
            # [NEW] 收集 Aux Image (辅助图/Map)
            if "aux_image" in instance and instance["aux_image"] is not None:
                batch_aux_images.append(instance["aux_image"])
            # [NEW] 收集 Actions
            if "actions" in instance and instance["actions"] is not None:
                batch_actions.append(instance["actions"]) # [1, 5]
            if "raw_prompt" in instance:
                batch_raw_prompt_list.append(instance["raw_prompt"])
            if "map_name" in instance:
                batch_map_name_list.append(instance["map_name"])
            if "pose_dict" in instance:
                batch_pose_dict_list.append(instance["pose_dict"])
            if "map_img" in instance:
                batch_map_img_list.append(instance["map_img"])
            if "fps_img" in instance:
                batch_fps_img_list.append(instance["fps_img"])
            if "z_range" in instance:
                batch_z_range_list.append(instance["z_range"])

        # 3. Padding (Pad Input Ids & Labels)
        # batch_first=True -> [BS, Seq]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=IGNORE_INDEX
        )

        # 再次检查最大长度 (Padding 后可能会变长)
        if input_ids.shape[1] > self.tokenizer.model_max_length:
            print(f"Input length {input_ids.shape[1]} > {self.tokenizer.model_max_length}, truncating.")
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            labels = labels[:, :self.tokenizer.model_max_length]

        # 4. 构建 Batch 字典
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            ids=ids_list,
        )

        # 5. 堆叠图像 (Stack Images)
        # 辅助函数：安全堆叠
        def stack_images(img_list):
            if len(img_list) == 0:
                return None
            # 检查形状一致性
            if all(x is not None and x.shape == img_list[0].shape for x in img_list):
                return torch.cat(img_list, dim=0) # [BS, C, H, W] (因为 Dataset 返回的是 [1, C, H, W])
            else:
                print("Image shapes inconsistent in batch, returning list instead of tensor.")
                return img_list

        batch["und_image"] = stack_images(batch_und_images)
        batch["aux_image"] = stack_images(batch_aux_images)

        # 6. 堆叠其他 Tensor (Actions, Mask)
        if len(batch_actions) > 0:
            batch["actions"] = torch.stack(batch_actions, dim=0) # [BS, 1, 5]
        else:
            batch["actions"] = None

        if len(task_id_list) > 0:
            batch["task_id"] = torch.tensor(task_id_list, dtype=torch.long)
        if len(batch_raw_prompt_list) > 0:
            batch["raw_prompt"] = batch_raw_prompt_list
        if len(batch_map_name_list) > 0:
            batch["map_name"] = batch_map_name_list
        if len(batch_pose_dict_list) > 0:
            batch["pose_dict"] = batch_pose_dict_list
        if len(batch_map_img_list) > 0:
            batch["map_img"] = batch_map_img_list
        if len(batch_fps_img_list) > 0:
            batch["fps_img"] = batch_fps_img_list
        if len(batch_z_range_list) > 0:
            batch["z_range"] = batch_z_range_list

        return batch



def unnormalize_pose(pred_tensor, z_range):
    x_norm, y_norm, z_norm, v_norm, h_norm = pred_tensor.cpu().numpy()
    return {
        'x': x_norm * 1024.0, 'y': y_norm * 1024.0,
        'z': z_norm * (z_range['max_z'] - z_range['min_z'] + 1e-6) + z_range['min_z'],
        'angle_v': v_norm * 360.0, 'angle_h': h_norm * 360.0
    }

def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg

map_path_dict = {
    'de_dust2': 'de_dust2_radar_psd.png',
    'de_inferno': 'de_inferno_radar_psd.png',
    'de_mirage': 'de_mirage_radar_psd.png',
    'de_nuke': 'de_nuke_blended_radar_psd.png',
    'de_ancient': 'de_ancient_radar_psd.png',
    'de_anubis': 'de_anubis_radar_psd.png',
    'de_golden': 'de_golden_radar_tga.png',
    'de_overpass': 'de_overpass_radar_psd.png',
    'de_palacio': 'de_palacio_radar_tga.png',
    'de_train': 'de_train_blended_radar_psd.png',
    'de_vertigo': 'de_vertigo_blended_radar_psd.png',
    'cs_agency': 'cs_agency_radar_tga.png',
    'cs_italy': 'cs_italy_radar_psd.png',
    'cs_office': 'cs_office_radar_psd.png',
}

# ==========================================
# 4. 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csgo_config", type=str, required=True)
    args = parser.parse_args()
    with open(args.csgo_config, 'r') as f:
        csgo_config = yaml.safe_load(f)

    disable_torch_init()
    inference_args = InferenceArgs(csgo_config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed()

    cur_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs_loc/{args.csgo_config.split('/')[-1][:-5]}/test_{cur_time_str}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Init Model
    model = Unified_UniLIP_InternVLForCausalLM.from_pretrained(
        csgo_config.get('model_name_or_path', 'UniLIP-1B'),
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )

    # 2. Init Tokenizer
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
    elif not "<image>" in tokenizer.get_added_vocab():
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(additional_special_tokens=["[IMG]", "[/IMG]", "<image>"]),
            tokenizer=tokenizer,
            model=model,
        )
    from unilip import conversation as conversation_lib
    if inference_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[inference_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama3"]
    print(f"Using conversation format: {conversation_lib.default_conversation.version}")

    # 3. Init Action Modules
    model.config.img_size = getattr(inference_args, "img_size", csgo_config.get("img_size", False))
    model.config.is_action_dit_dense_timestep = getattr(inference_args, "is_action_dit_dense_timestep", False)

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

    model.config.is_exp5_eval_without_aciton_dit_premodules = getattr(inference_args, "is_exp5_eval_without_aciton_dit_premodules", False)

    model.config.is_action_dit_projector = getattr(inference_args, "is_action_dit_projector", False)
    model.config.is_loc_learnable_query = getattr(inference_args, "is_loc_learnable_query", False)

    model.get_model().initialize_vision_modules(model_args=inference_args)
    model.get_model().initialize_localization_modules(model_args=inference_args)

    # =====================================================================
    # [NEW] 3.5 Inject LoRA architecture before loading weights
    # =====================================================================
    if csgo_config.get('is_lora', False):
        print("🔧 Injecting LoRA architecture to match the checkpoint structure...")
        # 构造一个 mock 的 training_args 来触发你在模型里写的 inject_lora_to_sub_module。
        # 确保 csgo_config 中包含 'is_lora: True'，以及相应的 r, alpha, dropout
        training_args = SimpleNamespace(
            is_lora=csgo_config.get('is_lora', False), # 默认置为True，因为这套逻辑就是为了跑LoRA
            lora_r=csgo_config.get('lora_r', 16),
            lora_alpha=csgo_config.get('lora_alpha', 16),
            lora_dropout=csgo_config.get('lora_dropout', 0.05)
        )

        model.inject_lora_to_sub_module(inference_args, training_args)


    # 4. Init 其他配置
    image_processor = AutoProcessor.from_pretrained(inference_args.mllm_hf_path).image_processor
    image_aspect_ratio = inference_args.image_aspect_ratio

    # 5. Init vision tokenizer
    # 训练时，默认配置使得initialize_vision_tokenizer()中所有配置都不生效，因此省略eval这里的初始化
    # model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # 6. Load Weights
    ckpt_path = csgo_config['ckpt_path']
    print(f"📥 Loading Checkpoint: {ckpt_path}")
    if ckpt_path.endswith(".safetensors"):
        state_dict = safe_load_file(ckpt_path, device="cpu")
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    # 7. Set Model Eval
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = False
    model.config.use_cache = True

    # 8. Init Localization Inference Dataset
    test_dataset = CSGOLocInferenceDataset(csgo_config, tokenizer, map_path_dict, image_processor, image_aspect_ratio)
    data_collator = DataCollatorForLoc(tokenizer=tokenizer)
    dataloader = DataLoader(test_dataset, batch_size=csgo_config["batch_size"], shuffle=False, collate_fn=data_collator)

    # 9. Inference Loop
    results_json = []
    vis_data_grouped = defaultdict(list) # 按地图分组存储可视化数据
    vis_data_flag = True

    print("🚀 Starting Localization Inference...")
    for batch in tqdm(dataloader):
        map_name = batch["map_name"]
        ids = batch["ids"]

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        und_image = batch["und_image"].to(device)
        aux_image = batch["aux_image"].to(device)

        actions = batch["actions"]
        pose_dict = batch["pose_dict"]

        raw_prompt = batch["raw_prompt"]
        map_img = batch["map_img"]
        fps_img = batch["fps_img"]
        z_range = batch["z_range"]

        with torch.no_grad():
            pred_norm_loc_tensor = model.generate_action2(
                input_ids,
                attention_mask,
                labels,
                und_image,
                aux_image,
                num_steps=10
            ).squeeze(1).float().cpu()

        current_bs = pred_norm_loc_tensor.size(0)
        for sample_idx in range(current_bs):
            # 1. 归一化数据
            pred_norm = pred_norm_loc_tensor[sample_idx]
            gt_norm = actions[sample_idx].squeeze(0).float().cpu()

            # 2. 物理数据 (用于可视化)
            pred_phys = unnormalize_pose(pred_norm, z_range[sample_idx])
            gt_phys = pose_dict[sample_idx]

            # 3. 存储
            results_json.append({
                "file_frame": ids[sample_idx],
                "map": map_name[sample_idx],
                "pred_norm": pred_norm.tolist(), # Save as list
                "gt_norm": gt_norm.tolist(),   # Save as list
                "pred": pred_phys,
                "gt": gt_phys,
            })

            current_map = map_name[sample_idx]
            if len(vis_data_grouped[current_map]) < 20:
                vis_data_grouped[map_name[sample_idx]].append({
                    "fps": fps_img[sample_idx], "gt": gt_phys, "pred": pred_phys
                })
            if len(vis_data_grouped[current_map]) == 20 and vis_data_flag:
                # Visualization (Pi0.5 Style)
                visualize_map_results(
                    vis_data_grouped,
                    output_dir,
                    map_path_dict,
                    csgo_config['data_dir'],
                    vis_num_per_map=10 # 每张地图随机选5个样本可视化
                )
                vis_data_flag = False # 只可视化一次，避免重复生成

    # 3. Calculate Metrics (L2 5D, SmoothL1, etc.)
    print("📈 Calculating Metrics...")
    metrics = calculate_metrics(results_json, ckpt_path)
    print(json.dumps(metrics, indent=4))

    # Save Metrics & Results
    with open(os.path.join(output_dir, "loc_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(output_dir, "loc_results.json"), "w") as f:
        json.dump(results_json, f, indent=4)


    print(f"✅ Finished. Results at: {output_dir}")

if __name__ == "__main__":
    main()

# python eval_csgo_loc.py --csgo_config csgo_configs/test/exp3_1_loc.yaml

