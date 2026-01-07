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
from safetensors.torch import load_file as safe_load_file
import textwrap

# 引入 UniLIP 核心模块
from unilip.utils import disable_torch_init
from unilip.pipeline_edit import CustomEditPipeline
from unilip.mm_utils import get_model_name_from_path
from unilip.model.builder import load_pretrained_model_general


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
            if self.config.get("is_conti_gen", False):
                split_path = f"{self.data_dir}/{map_name}/splits_20000_5000/continuous_unseen_clips.json"
            else:
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
            print([data['file_frame'] for data in self.data_entries])
        elif config['debug'] and config.get('debug_num_train_data', False) == False:
            indices = [335, 535, 707, 288, 21, 240, 20, 30, 809, 423, 857, 459, 557, 882, 893, 406, 24, 477, 407, 427, 453, 923, 925, 399, 752, 867, 547, 563, 424, 217, 789, 681]
            self.data_entries = [self.data_entries[i] for i in indices]
            print([data['file_frame'] for data in self.data_entries])
        elif config['debug']==False and config.get('debug_num_train_data', False):
            sampled_num = config.get('debug_num_train_data', len(self.data_entries))
            self.data_entries = random.sample(self.data_entries, sampled_num)
            print([data['file_frame'] for data in self.data_entries])

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
            "map_name": map_name,
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
# 0. 升级版：权重加载辅助函数 (支持 .bin 和 .safetensors)
# ==========================================
def load_custom_checkpoint(model, ckpt_path):
    """
    智能加载权重：
    1. 区分 .bin 和 .safetensors
    2. 支持加载 pytorch_model.bin / model.safetensors
    3. 支持加载分离保存的 mm_projector
    """
    print(f"🚀 Processing checkpoint path: {ckpt_path}")
    ckpt_path = os.path.abspath(ckpt_path)

    state_dict = None

    # --- 情况 A: 这是一个文件 (例如 model.safetensors) ---
    if os.path.isfile(ckpt_path):
        print(f"   -> Loading directly from file {ckpt_path} ...")
        if ckpt_path.endswith(".safetensors"):
            state_dict = safe_load_file(ckpt_path, device="cpu")
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        msg = model.load_state_dict(state_dict, strict=False)
        print(f"   -> Direct load msg: {msg}")
        return # 这是一个完整权重文件，加载完直接返回

    # --- 情况 B: 这是一个文件夹 (例如 checkpoint-1000) ---
    # 1. 尝试加载基础权重 (Base Weights)
    potential_base_files = ["model.safetensors", "pytorch_model.bin"]
    for f in potential_base_files:
        full_path = os.path.join(ckpt_path, f)
        if os.path.exists(full_path):
            print(f"   -> Found base weights: {full_path}")
            if f.endswith(".safetensors"):
                state_dict = safe_load_file(full_path, device="cpu")
            else:
                state_dict = torch.load(full_path, map_location="cpu")

            msg = model.load_state_dict(state_dict, strict=False)
            print(f"   -> Base load msg: {msg}")
            break

    # 2. 尝试加载 Projector (如果训练代码将其单独保存了)
    # 逻辑：检查当前目录或上级目录的 mm_projector 文件夹
    folder_name = os.path.basename(ckpt_path)
    parent_dir = os.path.dirname(ckpt_path)

    projector_types = ["mm_projector", "gen_projector"]

    for proj_type in projector_types:
        candidates = []
        # 候选 1: 在当前目录内
        candidates.append(os.path.join(ckpt_path, f"{proj_type}.bin"))
        candidates.append(os.path.join(ckpt_path, f"{proj_type}.safetensors"))

        # 候选 2: 在上级平行目录 (针对 checkpoint-xxx 结构)
        if folder_name.startswith("checkpoint-"):
            candidates.append(os.path.join(parent_dir, proj_type, f"{folder_name}.bin"))
            candidates.append(os.path.join(parent_dir, proj_type, f"{folder_name}.safetensors"))

        loaded_proj = False
        for p_path in candidates:
            if os.path.exists(p_path):
                print(f"   -> Found {proj_type} at: {p_path}")
                if p_path.endswith(".safetensors"):
                    proj_weights = safe_load_file(p_path, device="cpu")
                else:
                    proj_weights = torch.load(p_path, map_location="cpu")

                # 加载
                model.load_state_dict(proj_weights, strict=False)
                loaded_proj = True
                print(f"      ✅ Loaded {proj_type} successfully.")
                break

        if not loaded_proj and state_dict is None:
             print(f"⚠️ Warning: Did not find separate {proj_type} file and no base model loaded.")

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
    os.makedirs(output_dir+"/gen_imgs", exist_ok=True)

    # 1. 加载模型
    disable_torch_init()
    # tokenizer, model, context_len = load_pretrained_model_general(
    #     'UniLIP_InternVLForCausalLM', csgo_config['ckpt_path']
    # )
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model_general(
        'UniLIP_InternVLForCausalLM', 'UniLIP-1B'
    )

    ckpt_path = csgo_config['ckpt_path']
    print(f"🚀 Loading model from {csgo_config['ckpt_path']}...")
    load_custom_checkpoint(model, ckpt_path)

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
    is_vis = True
    vis_data_num = 30

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

            os.makedirs(output_dir+f"/gen_imgs/{sample['map_name']}", exist_ok=True)
            save_name = f"gen_imgs/{sample['map_name']}/{file_frame}.jpg"
            gen_img.save(os.path.join(output_dir, save_name))

            # 收集数据用于可视化
            if len(vis_data) < vis_data_num:
                # 保存单张图片

                vis_data.append({
                    "map_name": sample['map_name'],
                    "radar": radar_img,
                    "gt": sample['gt_img'],
                    "gen": gen_img,
                    "pose": sample['pose_info'],
                    "prompt": raw_prompt
                })

            if is_vis and len(vis_data) == vis_data_num:
                print(f"📊 Generating Visualization for {len(vis_data)} samples...")
                batch_size = 5
                # 按步长 batch_size 循环
                for batch_start in range(0, len(vis_data), batch_size):
                    batch_end = min(batch_start + batch_size, len(vis_data))
                    batch_items = vis_data[batch_start:batch_end]

                    # 创建画布：5行4列
                    # figsize 需要设置宽一点以容纳文本，高一点以容纳5行
                    fig, axes = plt.subplots(batch_size, 4, figsize=(24, 6 * batch_size))

                    # 如果只有一行（batch_size=1的情况），axes是一维数组，强制转为二维
                    if batch_size == 1:
                        axes = [axes]
                    # 如果最后不满5个，axes仍然是5行，需要处理空行
                    elif len(batch_items) < batch_size:
                        pass

                    for i in range(batch_size):
                        # 获取当前行对应的 axes
                        ax_row = axes[i]

                        # 如果当前批次的数据已经用完（处理最后余数的情况），隐藏剩下的空图
                        if i >= len(batch_items):
                            for ax in ax_row:
                                ax.axis('off')
                            continue

                        item = batch_items[i]

                        # --- 第1列：Prompt Text ---
                        # 使用 textwrap 自动换行，防止文本溢出
                        wrapped_prompt = "\n".join(textwrap.wrap(item['prompt'], width=40))

                        ax_row[0].text(0, 1, wrapped_prompt, ha='left', va='top', fontsize=18, wrap=True)
                        ax_row[0].axis('off') # 关闭坐标轴
                        if i == 0: ax_row[0].set_title("Input Prompt", fontsize=14, fontweight='bold')

                        # --- 第2列：Radar ---
                        ax_row[1].imshow(item['radar'])
                        ax_row[1].axis('off')
                        if i == 0: ax_row[1].set_title("Input Radar", fontsize=14, fontweight='bold')

                        # --- 第3列：GT ---
                        ax_row[2].imshow(item['gt'])
                        ax_row[2].axis('off')
                        if i == 0: ax_row[2].set_title("Ground Truth", fontsize=14, fontweight='bold')

                        # --- 第4列：Generated ---
                        ax_row[3].imshow(item['gen'])
                        ax_row[3].axis('off')

                        # 把具体的 Pose 数值放在生成图的标题上，作为补充信息
                        p = item['pose']
                        pose_str = f"{item['map_name']}, Pred View\nPos:({p['x']:.0f},{p['y']:.0f},{p['z']:.0f}) Ang:({p['angle_v']:.0f},{p['angle_h']:.0f})"
                        ax_row[3].set_title(pose_str if i > 0 else f"Generated\n{pose_str}", fontsize=10, color='blue')

                    plt.tight_layout()

                    # 保存文件名：vis_0_5.jpg, vis_5_10.jpg ...
                    vis_save_path = os.path.join(output_dir, f"vis_batch_{batch_start}_{batch_end}.jpg")
                    plt.savefig(vis_save_path, dpi=100) # dpi 100 足够清晰且体积适中
                    plt.close(fig) # 极其重要：循环画图必须 close，否则内存泄漏

                    print(f"   -> Saved visualization batch: {vis_save_path}")

                is_vis = False

    print(f"✅ Inference finished. Results saved to {output_dir}")

if __name__ == "__main__":
    main()



# python eval_csgo.py --csgo_config csgo_configs/test/exp0.yaml