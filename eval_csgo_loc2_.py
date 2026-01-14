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
def calculate_metrics(results):
    """
    计算定位任务的核心指标 (包含 L2 Norm, SmoothL1, MSE)
    results: List[Dict], 包含 'gt', 'pred' (已经是反归一化后的物理坐标)
    """
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

    # 对 Yaw (第4列) 做 wrap 处理: diff = (diff + 180) % 360 - 180
    # 或者更简单的 min(|d|, 360-|d|) 逻辑，但为了保留符号给 Loss 用，通常取最小夹角
    # 这里为了 L2/MSE/SmoothL1 计算距离（即误差大小），我们取绝对误差
    abs_diff = torch.abs(diff_tensor)

    # 修正 Yaw 的绝对误差: min(err, 360 - err)
    yaw_err = abs_diff[:, 4]
    yaw_err = torch.min(yaw_err, 360.0 - yaw_err)
    abs_diff[:, 4] = yaw_err

    # --- Metric 1: 单项误差 ---
    mean_xy_error = torch.sqrt(abs_diff[:, 0]**2 + abs_diff[:, 1]**2).mean().item()
    mean_z_error = abs_diff[:, 2].mean().item()
    mean_pitch_error = abs_diff[:, 3].mean().item()
    mean_yaw_error = abs_diff[:, 4].mean().item()

    # --- Metric 2: XY L2 Norm (同 Mean XY Error) ---
    l2_norm_xy = torch.norm(abs_diff[:, :2], p=2, dim=1).mean().item()

    # --- Metric 3: 5D L2 Norm ---
    # 定义：sqrt(dx^2 + dy^2 + dz^2 + dp^2 + dyaw^2)
    l2_norm_5d = torch.norm(abs_diff, p=2, dim=1).mean().item()

    # --- Metric 4: 5D Loss (MSE & SmoothL1) ---
    # 这里的 Loss 是基于物理坐标的，所以数值会很大，但能反映物理偏离程度
    # 使用 abs_diff 作为输入，也就是计算 Loss(abs_diff, 0)
    mse_loss_5d = F.mse_loss(abs_diff, torch.zeros_like(abs_diff))
    smooth_l1_loss_5d = F.smooth_l1_loss(abs_diff, torch.zeros_like(abs_diff), beta=1.0)

    metrics = {
        "Mean_XY_Error": mean_xy_error,
        "L2_Norm_XY": l2_norm_xy,       # 实际上等于 Mean_XY_Error
        "Mean_Z_Error": mean_z_error,
        "Mean_Pitch_Error": mean_pitch_error,
        "Mean_Yaw_Error": mean_yaw_error,
        "L2_Norm_5D": l2_norm_5d,
        "MSE_Loss_5D": mse_loss_5d.item(),
        "SmoothL1_Loss_5D": smooth_l1_loss_5d.item()
    }
    return metrics

# ==========================================
# 2. 可视化逻辑 (Pi0.5 风格)
# ==========================================
def visualize_map_results(vis_data_grouped, output_dir, map_path_dict, data_dir_base, vis_num_per_map=5):
    print(f"📊 Generating Pi0.5 Style Visualizations for {len(vis_data_grouped)} maps...")
    color_list = [
        (255, 0, 0, 180), (0, 255, 0, 180), (0, 0, 255, 180),
        (255, 255, 0, 180), (255, 0, 255, 180)
    ]

    for map_name, items in vis_data_grouped.items():
        items_to_show = items[:vis_num_per_map]

        map_filename = map_path_dict.get(map_name, 'de_dust2_radar_psd.png')
        map_path = os.path.join(data_dir_base, map_name, map_filename)
        if not os.path.exists(map_path):
            print(f"⚠️ Map image not found: {map_path}")
            continue

        base_map = Image.open(map_path).convert('RGBA')
        overlay = Image.new("RGBA", base_map.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        fps_img_list = []

        for i, item in enumerate(items_to_show):
            color = color_list[i % len(color_list)]
            scale_x = base_map.width / 1024.0
            scale_y = base_map.height / 1024.0

            gt_x, gt_y = item['gt']['x'] * scale_x, item['gt']['y'] * scale_y
            pred_x, pred_y = item['pred']['x'] * scale_x, item['pred']['y'] * scale_y

            # GT: Solid Circle
            r_gt = 6
            draw.ellipse((gt_x - r_gt, gt_y - r_gt, gt_x + r_gt, gt_y + r_gt), fill=color, outline='black', width=1)
            # Pred: Hollow Circle with Line
            r_pred = 8
            draw.ellipse((pred_x - r_pred, pred_y - r_pred, pred_x + r_pred, pred_y + r_pred), fill=None, outline=color, width=3)
            draw.line([(gt_x, gt_y), (pred_x, pred_y)], fill='white', width=2)

            # FPS Image with Color Tag
            fps_img = item['fps'].convert('RGB').copy()
            fps_draw = ImageDraw.Draw(fps_img)
            fps_draw.rectangle((0, 0, 40, 40), fill=color[:3])
            fps_img_list.append(fps_img)

        combined_map = Image.alpha_composite(base_map, overlay).convert('RGB')
        fps_strip = vertical_concat(fps_img_list)
        final_vis = concat_images_horizontal_resize(combined_map, fps_strip)

        save_path = os.path.join(output_dir, f"vis_map_{map_name}.jpg")
        final_vis.save(save_path)
        print(f"   -> Saved: {save_path}")

# ==========================================
# 3. 辅助类与 Dataset
# ==========================================
def get_loc_prompt(map_name):
    return f"Task: The following visual data of CS2 map '{map_name}' has been fused... Predict the 5D pose...\n<image>\n<image>"

def add_template_for_loc(prompt_text):
    return f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

# 模拟 ModelArguments
class InferenceArgs:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)
        self.unilip_path = config_dict.get("unilip_path", "")
        self.unilip_factor = config_dict.get("unilip_factor", 5.85)
        self.fix_dit = False; self.fix_connect = False
        self.mllm_hf_path = config_dict.get("mllm_hf_path", "OpenGVLab/InternVL3-1B-hf")
        self.n_query = 256; self.connect_layer = 6
        self.action_horizon = 1; self.action_dim = 5
        self.is_action_dit_dense_timestep = config_dict.get("is_action_dit_dense_timestep", False)
        self.action_dit_layer = config_dict.get("action_dit_layer", 3)
        self.mm_use_im_patch_token = False; self.mm_use_im_start_end = False
        self.tune_mm_mlp_adapter = False; self.pretrain_mm_mlp_adapter = None

# Dataset
class CSGOLocInferenceDataset(Dataset):
    def __init__(self, config, map_path_dict):
        self.config = config
        self.data_dir = config['data_dir']
        self.map_names = config['val_maps']
        self.map_path_dict = map_path_dict
        self.data_entries = []
        self.map_z_range = {}

        for map_name in self.map_names:
            split_path = f"{self.data_dir}/{map_name}/splits_20000_5000/test_split.json"
            if not os.path.exists(split_path): continue
            with open(split_path, "r", encoding="utf-8") as f: positions_data = json.load(f)
            zs = [d['z'] for d in positions_data]
            self.map_z_range[map_name] = {'max_z': max(zs), 'min_z': min(zs)}
            for pos_data in positions_data:
                self.data_entries.append({
                    'map': map_name, 'file_frame': pos_data['file_frame'],
                    'x': pos_data['x'], 'y': pos_data['y'], 'z': pos_data['z'],
                    'angle_v': pos_data['angle_v'], 'angle_h': pos_data['angle_h']
                })

        if config.get('debug', False):
            self.data_entries = random.sample(self.data_entries, min(20, len(self.data_entries)))

    def __len__(self): return len(self.data_entries)

    def __getitem__(self, i):
        data = self.data_entries[i]
        map_name = data['map']
        radar_path = f"{self.data_dir}/{map_name}/{self.map_path_dict.get(map_name, 'de_dust2_radar_psd.png')}"
        radar_img = Image.open(radar_path).convert('RGB')
        ext = ".jpg" if "preprocessed" in self.data_dir else ".png"
        fps_path = f"{self.data_dir}/{map_name}/imgs/{data['file_frame']}{ext}"
        fps_img = Image.open(fps_path).convert('RGB')

        pose_dict = {
            'x': data['x'], 'y': data['y'], 'z': data['z'],
            'angle_v': (data['angle_v'] / (2 * np.pi)) * 360.0,
            'angle_h': (data['angle_h'] / (2 * np.pi)) * 360.0
        }
        return {
            "map_name": map_name, "radar_img": radar_img, "fps_img": fps_img,
            "final_prompt": add_template_for_loc(get_loc_prompt(map_name)),
            "file_frame": data['file_frame'], "gt_pose": pose_dict,
            "z_range": self.map_z_range[map_name]
        }

def collate_fn(batch): return batch

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
    'de_dust2': 'de_dust2_radar_psd.png', 'de_inferno': 'de_inferno_radar_psd.png',
    'de_mirage': 'de_mirage_radar_psd.png', 'de_nuke': 'de_nuke_blended_radar_psd.png',
    'de_ancient': 'de_ancient_radar_psd.png', 'de_overpass': 'de_overpass_radar_psd.png',
    'de_vertigo': 'de_vertigo_blended_radar_psd.png', 'cs_italy': 'cs_italy_radar_psd.png',
    'cs_office': 'cs_office_radar_psd.png',
}

# ==========================================
# 4. 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csgo_config", type=str, required=True)
    args = parser.parse_args()

    with open(args.csgo_config, 'r') as f: csgo_config = yaml.safe_load(f)
    set_seed()

    cur_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs_loc/{args.csgo_config.split('/')[-1][:-5]}/test_{cur_time_str}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Init Model
    disable_torch_init()
    model_args = InferenceArgs(csgo_config)
    model = Unified_UniLIP_InternVLForCausalLM.from_pretrained(
        csgo_config.get('model_name_or_path', 'UniLIP-1B'),
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )

    model.config.is_action_dit_dense_timestep = model_args.is_action_dit_dense_timestep
    model.get_model().initialize_vision_modules(model_args=model_args)
    model.get_model().initialize_localization_modules(model_args=model_args)

    tokenizer = AutoProcessor.from_pretrained(model_args.mllm_hf_path).tokenizer
    tokenizer.model_max_length = 1024
    smart_tokenizer_and_embedding_resize(
        dict(pad_token="<pad>", additional_special_tokens=["[IMG]", "[/IMG]", "<image>"]),
        tokenizer, model
    )

    ckpt_path = csgo_config['ckpt_path']
    print(f"📥 Loading Checkpoint: {ckpt_path}")
    if ckpt_path.endswith(".safetensors"):
        state_dict = safe_load_file(ckpt_path, device="cpu")
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.to(device='cuda', dtype=torch.bfloat16)
    model.eval()
    model.config.use_cache = True

    image_processor = AutoProcessor.from_pretrained(model_args.mllm_hf_path).image_processor

    # 2. Inference Loop
    test_dataset = CSGOLocInferenceDataset(csgo_config, map_path_dict)
    dataloader = DataLoader(test_dataset, batch_size=csgo_config["batch_size"], shuffle=False, collate_fn=collate_fn)

    results_json = []
    vis_data_grouped = defaultdict(list)

    print("🚀 Starting Localization Inference...")
    for batch in tqdm(dataloader):
        prompts = [s['final_prompt'] for s in batch]
        fps_imgs = [s['fps_img'] for s in batch]
        radar_imgs = [s['radar_img'] for s in batch]

        fps_inputs = image_processor(images=fps_imgs, return_tensors="pt")['pixel_values'].to(model.device, dtype=model.dtype)
        radar_inputs = image_processor(images=radar_imgs, return_tensors="pt")['pixel_values'].to(model.device, dtype=model.dtype)

        with torch.no_grad():
            pred_actions = model.generate_action(
                text=prompts, tokenizer=tokenizer,
                und_images=fps_inputs, aux_images=radar_inputs,
                num_steps=10
            ).squeeze(1).float().cpu()

        for idx, sample in enumerate(batch):
            pred_raw = unnormalize_pose(pred_actions[idx], sample['z_range'])
            gt_pose = sample['gt_pose']

            res_item = {
                "file_frame": sample['file_frame'], "map": sample['map_name'],
                "gt": gt_pose, "pred": pred_raw
            }
            results_json.append(res_item)
            vis_data_grouped[sample['map_name']].append({
                "fps": sample['fps_img'], "gt": gt_pose, "pred": pred_raw
            })

    # 3. Calculate Metrics (L2 5D, SmoothL1, etc.)
    print("📈 Calculating Metrics...")
    metrics = calculate_metrics(results_json)
    print(json.dumps(metrics, indent=4))

    with open(os.path.join(output_dir, "loc_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(output_dir, "loc_results.json"), "w") as f:
        json.dump(results_json, f, indent=4)

    # 4. Visualization
    visualize_map_results(
        vis_data_grouped, output_dir, map_path_dict,
        csgo_config['data_dir'], vis_num_per_map=5
    )

    print(f"✅ Finished. Results at: {output_dir}")

if __name__ == "__main__":
    main()