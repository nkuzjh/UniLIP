import argparse
import os
import json
import torch
import numpy as np
import yaml
import random
import datetime
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor
import matplotlib.pyplot as plt

# 引入 UniLIP 核心模块
from unilip.utils import disable_torch_init
from unilip.model.builder import load_pretrained_model_general
from unilip.pipeline_edit import CustomEditPipeline
from unilip.mm_utils import get_model_name_from_path



def set_seed(seed=42):
    # 1. Python 内置 random
    random.seed(seed)
    # 2. 操作系统环境 (这对某些哈希操作是必须的，如 set/dict 的顺序)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 3. NumPy
    np.random.seed(seed)
    # 4. PyTorch CPU
    torch.manual_seed(seed)
    # 5. PyTorch GPU (如果可用)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 如果有多张显卡，为所有显卡设置
    # 6. 设置 CuDNN 后端以确保确定性 (会降低性能)
    # 如果你非常看重结果的逐位一致性，必须开启 deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"随机种子已设置为: {seed}")


# ==========================================
# 1. 复用辅助函数 (Prompt 构建 & Padding)
# ==========================================
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def build_sft_instruction_custom(pose_5d, map_name, z_max, z_min):
    # 与训练时完全一致的 Prompt 模板
    definition_text = (
        f"Task: Generate a First-Person View (FPV) image of CS2 map '{map_name}' based on the Radar Map and Camera Pose.\n"
        "Coordinate System Definition:\n"
        "- Map Size: 1024x1024 pixels.\n"
        "- Yaw: 0 degrees is East, increases Clockwise.\n"
        "- Pitch: 0 degrees is looking straight Down (at feet), 180 degrees is looking straight Up (at sky).\n"
        f"- Z-Height: Absolute vertical coordinate. Valid values are bounded by the map's global topology, ranging from the lowest point at {z_min:.2f} to the highest point at {z_max:.2f}."
    )
    pose_str = (
        f"Position(x={pose_5d['x']:.1f}, y={pose_5d['y']:.1f}, z={pose_5d['z']:.3f}), "
        f"Rotation(pitch={pose_5d['angle_v']:.1f}, yaw={pose_5d['angle_h']:.1f})"
    )
    full_instruction = f"{definition_text}\n\nCurrent Camera Pose: {pose_str}\n<image>"
    return full_instruction

def add_template_for_inference(prompt_text):
    # 将 SFT 指令包装成对话格式
    instruction = ('<|im_start|>user\n{input}<|im_end|>\n'
                   '<|im_start|>assistant\n<img>')

    # Positive Prompt: 你的 SFT 指令
    pos_prompt = instruction.format(input=prompt_text)

    # Negative/CFG Prompt: 保持训练时的通用指令
    # 注意：这里 <image> 也要包含
    cfg_prompt = instruction.format(input="Generate the view.\n<image>")

    return [pos_prompt, cfg_prompt]

# ==========================================
# 2. 轻量级推理数据集 (InferenceDataset)
# ==========================================
class CSGOInferenceDataset(Dataset):
    def __init__(self, config, map_path_dict):
        self.config = config
        self.data_dir = config['data_dir']
        self.map_names = config['val_maps']
        self.map_path_dict = map_path_dict

        self.data_entries = []
        self.map_z_range = {}
        print("🔄 Loading Test Data...")
        for map_name in self.map_names:
            # 读取测试集 split
            # 注意：这里强制读取 test_split.json
            split_path = f"{self.data_dir}/{map_name}/splits_20000_5000/test_split.json"

            with open(split_path, "r", encoding="utf-8") as f:
                positions_data = json.load(f)

            # 计算 Z 范围 (必须基于全集或 Train 集的统计，这里简化为当前 Split 的统计，建议最好硬编码或读取 train_split 统计)
            # 为了严谨，这里应该读取 train_split 来获取 z_min/z_max，防止 test 数据溢出
            # 这里简化处理：直接遍历 test set (生产环境建议读取 metadata)
            zs = [d['z'] for d in positions_data]
            self.map_z_range[map_name] = {'max_z': max(zs), 'min_z': min(zs)}

            for pos_data in positions_data:
                entry = {
                    'map': map_name,
                    'file_frame': pos_data['file_frame'],
                    'x': pos_data['x'],
                    'y': pos_data['y'],
                    'z': pos_data['z'],
                    'angle_v': pos_data['angle_v'],
                    'angle_h': pos_data['angle_h'],
                }
                self.data_entries.append(entry)

        if config['debug'] and config.get('debug_num_train_data', False):
            sampled_num = config.get('debug_num_train_data', len(self.data_entries))
            self.data_entries = self.data_entries[:sampled_num]
        elif config['debug'] and config.get('debug_num_train_data', False) == False:
            indices = [335, 535, 707, 288, 21, 240, 20, 30, 809, 423, 857, 459, 557, 882, 893, 406, 24, 477, 407, 427, 453, 923, 925, 399, 752, 867, 547, 563, 424, 217, 789, 681]
            self.data_entries = [self.data_entries[i] for i in indices]
        elif config['debug']==False and config.get('debug_num_train_data', False):
            sampled_num = config.get('debug_num_train_data', len(self.data_entries))
            self.data_entries = random.sample(self.data_entries, sampled_num)

        # 仅取前N个做测试，避免跑太久 (可选)
        # self.data_entries = self.data_entries[:50]
        print(f"✅ Loaded {len(self.data_entries)} test samples.")

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, i):
        data = self.data_entries[i]
        map_name = data['map']

        # 1. 加载 Radar (Input Condition)
        map_filename = self.map_path_dict.get(map_name, 'de_dust2_radar_psd.png')
        radar_path = f"{self.data_dir}/{map_name}/{map_filename}"
        radar_img = Image.open(radar_path).convert('RGB')

        # 2. 加载 GT FPS (Ground Truth for Vis)
        # 注意后缀，如果是 preprocessed_data 可能是 .jpg
        ext = ".jpg" if "preprocessed" in self.data_dir else ".png"
        fps_path = f"{self.data_dir}/{map_name}/imgs/{data['file_frame']}{ext}"
        gt_img = Image.open(fps_path).convert('RGB')

        # 3. 准备 Prompt 参数
        z_min = self.map_z_range[map_name]['min_z']
        z_max = self.map_z_range[map_name]['max_z']

        # 归一化 Z (0-1) 用于 Pose 数值展示
        z_norm = (data['z'] - z_min) / (z_max - z_min + 1e-6)

        # 弧度转角度
        pitch_deg = (data['angle_v'] / (2 * np.pi)) * 180.0
        yaw_deg = (data['angle_h'] / (2 * np.pi)) * 360.0

        pose_dict = {
            'x': data['x'], 'y': data['y'], 'z': data['z'],
             'angle_v': pitch_deg, 'angle_h': yaw_deg
        }

        # 4. 构建 Prompt
        # 注意：这里 z_max, z_min 传入真实物理值用于定义
        raw_prompt = build_sft_instruction_custom(pose_dict, map_name, z_max, z_min)

        return {
            "radar_img": radar_img,
            "gt_img": gt_img,
            "raw_prompt": raw_prompt,
            "file_frame": data['file_frame'],
            "pose_info": pose_dict
        }

def collate_fn(batch):
    return batch # 简单的 list 返回，不由 DataLoader 自动 stack tensor

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
# 3. 主推理逻辑
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csgo_config", type=str, required=True)
    args = parser.parse_args()

    with open(args.csgo_config, 'r') as f:
        csgo_config = yaml.safe_load(f)
    print("csgo_config: ", csgo_config)

    # 设置随机种子
    set_seed()

    cur_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs_eval/{args.csgo_config.split('/')[-1][:-5]}/test_{cur_time_str}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载模型
    disable_torch_init()
    model_name = get_model_name_from_path(csgo_config["ckpt_path"])
    print(f"🚀 Loading model from {csgo_config['ckpt_path']}...")
    tokenizer, model, context_len = load_pretrained_model_general(
        'UniLIP_InternVLForCausalLM', csgo_config["ckpt_path"], None, model_name
    )

    image_processor = AutoProcessor.from_pretrained(model.config.mllm_hf_path).image_processor

    # 初始化 Pipeline
    pipe = CustomEditPipeline(multimodal_encoder=model, tokenizer=tokenizer, image_processor=image_processor)

    test_dataset = CSGOInferenceDataset(
        csgo_config,
        map_path_dict
    )

    dataloader = DataLoader(test_dataset, batch_size=csgo_config["batch_size"], shuffle=False, collate_fn=collate_fn)

    # 3. 推理循环
    generator = torch.Generator(device=model.device).manual_seed(42)
    print("🚀 Starting Inference...")

    vis_data = [] # 存储第一批次用于可视化

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        # 批次内的每个样本逐个处理 (因为 Pipe 接口通常接受 List[Prompt] 但对应单张图片输入)
        # 为了兼容 CustomEditPipeline 的逻辑 (multimodal_prompts list 结构)

        for sample in batch:
            radar_img = sample['radar_img']
            raw_prompt = sample['raw_prompt']
            file_frame = sample['file_frame']

            # 构造 UniLIP 格式的 multimodal prompts
            # [Positive_Prompt, Negative_Prompt, Image]
            multimodal_prompts = add_template_for_inference(raw_prompt)
            multimodal_prompts.append(radar_img) # 必须 append PIL Image 对象

            # 执行生成
            with torch.no_grad():
                gen_img = pipe(
                    multimodal_prompts,
                    guidance_scale=csgo_config["guidance_scale"],
                    generator=generator
                )

            # 保存单张图片
            save_name = f"{file_frame}.png"
            gen_img.save(os.path.join(output_dir, save_name))

            # 收集数据用于可视化
            if len(vis_data) < 4:
                vis_data.append({
                    "radar": radar_img,
                    "gt": sample['gt_img'],
                    "gen": gen_img,
                    "pose": sample['pose_info'],
                    "prompt": raw_prompt
                })

    # 4. 可视化对比图 (Radar | GT | Gen)
    if len(vis_data) > 0:
        print("📊 Generating Visualization for the first batch...")
        fig, axes = plt.subplots(len(vis_data), 3, figsize=(15, 5 * len(vis_data)))
        if len(vis_data) == 1: axes = [axes]

        for i, item in enumerate(vis_data):
            # Radar
            axes[i][0].imshow(item['radar'])
            axes[i][0].set_title("Input: Radar Map")
            axes[i][0].axis('off')

            # GT FPS
            axes[i][1].imshow(item['gt'])
            axes[i][1].set_title("Ground Truth (FPS)")
            axes[i][1].axis('off')

            # Generated FPS
            axes[i][2].imshow(item['gen'])

            # 提取 Pose 字符串用于展示
            p = item['pose']
            title_str = f"Generated\nPos: ({p['x']:.1f}, {p['y']:.1f}, {p['z']:.2f})\nAng: ({p['angle_v']:.1f}, {p['angle_h']:.1f})"
            axes[i][2].set_title(title_str, color='blue', fontsize=10)
            axes[i][2].axis('off')

        plt.tight_layout()
        vis_save_path = os.path.join(output_dir, "vis_batch_0.png")
        plt.savefig(vis_save_path, dpi=150)
        print(f"✨ Visualization saved to {vis_save_path}")

    print(f"✅ Inference finished. Results saved to {output_dir}")

if __name__ == "__main__":
    main()



# python eval_csgo.py --csgo_config csgo_configs/test/exp0.yaml