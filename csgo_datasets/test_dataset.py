import sys
import os

# 获取当前脚本文件的目录 (.../UniLIP/csgo_datasets)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录 (.../UniLIP)
parent_dir = os.path.dirname(current_dir)
# 将上一级目录加入到系统路径
sys.path.append(parent_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor

from csgo_datasets.unified_task_dataset import UniLIPMultiTaskDataset, map_path_dict

# 假设您的数据集代码保存在 dataset_multitask.py 中，请根据实际情况修改 import
# from dataset_multitask import UniLIPMultiTaskDataset, map_path_dict

# 为了让测试代码独立运行，我需要 mock 一下您的 Config 和 DataArgs
class MockConfig(dict):
    def __getattr__(self, name):
        return self.get(name)

class MockDataArgs:
    def __init__(self):
        # 使用 CLIP 的标准处理器参数
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        self.image_aspect_ratio = "pad"
        self.is_multimodal = True

# ==========================================
# 可视化辅助函数
# ==========================================
def denormalize_image(tensor):
    """反归一化 CLIP 的图片 Tensor，转回 RGB numpy"""
    # CLIP mean/std
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    # Reverse Normalize
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)

    # To Numpy (H, W, C)
    img_np = tensor.permute(1, 2, 0).numpy()
    return (img_np * 255).astype(np.uint8)

import textwrap

def visualize_sample(sample, idx, tokenizer, save_dir="_vis_results"):
    """可视化单个样本"""
    task_id = sample['task_id']
    task_name = "LOCALIZATION" if task_id == 0 else "GENERATION"
    wrapped_prompt = "\n".join(textwrap.wrap(sample['raw_prompt'] + f"\n\nPose: {sample['pose_dict']}", width=100))

    print(f"\n[{idx}] Task: {task_name}")
    print(f"    Loss Mask: {sample['loss_mask']}")
    print(f"    Map ID: {sample['map_id']}")
    print(f"    Actions (Pose): {sample['actions']}")
    print(f"    Pose: {sample['pose_dict']}")
    print(f"    input_ids: {sample['input_ids']}")
    print(f"    Raw Prompt: {sample['raw_prompt']}") # 只打印前100字符
    print(f"    decode(input_ids): {tokenizer.decode(sample['input_ids'])}")



    # 准备画布
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    plt.suptitle(f"Sample {idx}: {task_name}", fontsize=16)

    # --- 1. Und Image (理解流输入) ---
    # shape [1, 3, H, W] -> [3, H, W]
    und_img = denormalize_image(sample['und_image'][0])
    axes[0].imshow(und_img)
    axes[0].set_title("Und Image (Input)")
    axes[0].axis('off')

    # --- 2. Aux Image (辅助输入 - 仅定位任务有) ---
    aux_tensor = sample['aux_image'][0]
    if torch.all(aux_tensor == 0):
        # 全黑图
        axes[1].imshow(np.zeros_like(und_img))
        axes[1].set_title("Aux Image (Empty)")
    else:
        aux_img = denormalize_image(aux_tensor)
        axes[1].imshow(aux_img)
        axes[1].set_title("Aux Image (Map)")
    axes[1].axis('off')

    # --- 3. Gen Image (生成目标 - 仅生成任务有) ---
    gen_tensor = sample['gen_image'][0]
    if torch.all(gen_tensor == 0):
        axes[2].imshow(np.zeros_like(und_img))
        axes[2].set_title("Gen Target (Empty)")
    else:
        gen_img = denormalize_image(gen_tensor)
        axes[2].imshow(gen_img)
        axes[2].set_title("Gen Target (GT FPS)")
    axes[2].axis('off')

    fig.text(0.5, 0.05, f"Prompt:\n{wrapped_prompt}",
             ha='center', va='bottom', fontsize=14,
             bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="black", alpha=0.8),
             fontfamily='monospace')
    plt.subplots_adjust(bottom=0.25)

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"_test_dataset_sample_{idx}_{task_name}.jpg")
    plt.savefig(save_path)
    print(f"    Saved visualization to {save_path}")
    plt.close()

# ==========================================
# 主测试流程
# ==========================================
if __name__ == "__main__":
    # 1. 模拟配置
    # 请修改为您实际的数据路径
    DATA_DIR = "data/preprocessed_data"

    config = MockConfig({
        "data_dir": DATA_DIR,
        "train_maps": ['de_dust2','de_nuke', 'de_ancient'], # 测试单张地图
        "debug": True,              # 开启 debug 模式快速加载
        "debug_num_train_data": 100,
        "task_mix_ratio": 0.5,      # 50/50 混合
        "is_fps_dropout": True,     # 测试数据增强
        "erasing_p": 0.6
    })

    data_args = MockDataArgs()

    # 2. 模拟 Tokenizer (使用 Llama 或 CLIP tokenizer 均可，这里用简单的 AutoTokenizer)
    print("⏳ Loading Tokenizer...")
    try:
        # 尝试加载一个真实的 tokenizer，如果没有网络可以用本地路径
        tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B-hf")
    except:
        print("⚠️ Warning: Failed to load Vicuna tokenizer, using bert-base-uncased as fallback.")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 3. 初始化数据集
    # 注意：这里需要引用你定义的 Dataset 类
    # 假设你的代码在当前脚本的上面，或者 import 进来了
    try:
        dataset = UniLIPMultiTaskDataset(config, tokenizer, data_args)
    except Exception as e:
        print(f"❌ Dataset Init Failed: {e}")
        print("💡 Hint: 请检查 config['data_dir'] 路径是否正确！")
        exit()

    # 4. 抽取样本进行测试
    print(f"\n🚀 Start Testing... Dataset Length: {len(dataset)}")

    # 随机取 5 个样本
    indices = np.random.choice(len(dataset), 10, replace=False)

    for i, idx in enumerate(indices):
        try:
            sample = dataset[idx]
            visualize_sample(sample, idx, tokenizer)
        except Exception as e:
            print(f"❌ Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()

    print("\n✅ Test Finished. Check '_vis_results' folder.")