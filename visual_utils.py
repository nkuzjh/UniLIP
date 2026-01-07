import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def visualize_dataset_samples(dataset, processor, num_samples=20, save_path="_debug_dataset_samples.jpg", is_multi_task=False):
    """
    直接对已加载的 Dataset 进行抽样可视化，验证 Radar(Input) -> FPS(Target) 及 Prompt 对齐情况。

    Args:
        dataset: 已经初始化好的 CSGOWorldModelDataset 实例
        processor: 对应的 ImageProcessor (用于获取 mean/std 进行逆归一化)
        num_samples: 抽样数量
        save_path: 图片保存路径
    """
    print(f"👀 Visualizing first {num_samples} samples from dataset...")

    # 获取反归一化所需的均值和方差
    # 如果 processor 是 AutoProcessor，通常在 processor.image_processor 中
    # 如果传入的是 image_processor 直接使用即可
    img_processor = getattr(processor, "image_processor", processor)
    mean = np.array(img_processor.image_mean)
    std = np.array(img_processor.image_std)

    if is_multi_task:
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    else:
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    plt.subplots_adjust(hspace=0.4, wspace=0.1)

    # 兼容 num_samples=1 的情况
    if num_samples == 1: axes = np.array([axes])

    for i in range(num_samples):
        # 1. 获取数据 (__getitem__)
        sample = dataset[i]

        map_name = sample['map_name']
        min_z = dataset.map_z_range[map_name]['min_z']
        max_z = dataset.map_z_range[map_name]['max_z']
        fps_img_name = sample['ids']
        # 2. 提取图像 Tensor [1, C, H, W] -> Squeeze -> [C, H, W]
        # 注意: dataset 返回的是 'und_image' (Radar) 和 'gen_image' (FPS)
        radar_tensor = sample['und_image'].squeeze(0)
        aux_tensor = sample['aux_image'].squeeze(0) if sample.get('aux_image') is not None else None
        fps_tensor = sample['gen_image'].squeeze(0)
        loc_coords = sample['loc_coords'] if sample.get('loc_coords') is not None else sample['actions'][0]
        x,y,z,v,h = loc_coords[0],loc_coords[1],loc_coords[2],loc_coords[3],loc_coords[4]
        x=x*1024
        y=y*1024
        z=z*(max_z-min_z)+min_z
        v=v* 360
        h=h* 360

        # 3. 逆归一化 (Tensor -> Numpy Image)
        def denorm(tensor):
            img = tensor.permute(1, 2, 0).cpu().numpy() # [H, W, C]
            img = img * std + mean # 反归一化
            return np.clip(img, 0, 1)

        img_radar = denorm(radar_tensor)
        img_aux = denorm(aux_tensor) if aux_tensor is not None else None
        img_fps = denorm(fps_tensor)

        # 4. 解码文本 Prompt (验证坐标是否正确)
        # sample['input_ids'] 是 Tensor，需要转回 list 才能 decode
        input_ids = sample['input_ids']
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        # 使用 dataset 中的 tokenizer 进行解码
        decoded_text = dataset.tokenizer.decode(input_ids, skip_special_tokens=False)

        # 提取关键信息用于标题 (截取 Prompt 中的 Pose 部分)
        try:
            # 简单提取 Pose(...) 部分，避免标题太长
            pose_str = decoded_text.split("Current Camera Pose:")[-1].split("<img>")[0].strip()
            # 如果太长，换行显示
            if len(pose_str) > 50:
                # pose_str = pose_str[:50] + "\n" + pose_str[50:]
                # pose_str = '\n'.join([pose_str[i:i+50] for i in range(0, len(pose_str), 50)])
                import textwrap
                pose_str = textwrap.fill(pose_str, width=50)
        except:
            pose_str = "Prompt parsing failed"

        # 5. 绘图
        # 左侧: Radar Map
        axes[i, 0].imshow(img_radar)
        axes[i, 0].set_title(f"Sample {i} | Input: Radar, \npose {pose_str}", fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')

        # 右侧: FPS View
        axes[i, 1].imshow(img_fps)
        axes[i, 1].set_title(f"Target: FPS {fps_img_name}, \ngt_pose{x,y,z,v,h}", fontsize=9, color='darkblue')
        axes[i, 1].axis('off')

        if img_aux is not None:
            axes[i, 2].imshow(img_aux)
            axes[i, 2].set_title(f"Auxilliary for Multi-Task Training", fontsize=9, color='green')
            axes[i, 2].axis('off')

    # 保存
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    print(f"✨ Visualization saved to: {os.path.abspath(save_path)}")
    plt.close()


### MultiTaskDataset:
# sample.keys()
# dict_keys(['task_id', 'und_image', 'aux_image', 'gen_image', 'input_ids', 'labels', 'raw_prompt', 'actions', 'loss_mask', 'map_id', 'map_name', 'pose_dict'])



# print(train_dataset[0].keys())
# dict_keys(['input_ids', 'labels', 'und_image', 'gen_image', 'ids', 'loc_coords'])

# print(train_dataset[0]['input_ids'].shape)
# torch.Size([423])

# print(train_dataset[0]['labels'].shape)
# torch.Size([423])

# print(train_dataset[0]['und_image'].shape)
# torch.Size([1, 3, 448, 448])

# print(train_dataset[0]['gen_image'].shape)
# torch.Size([1, 3, 448, 448])

# print(train_dataset[0]['ids'])
# file_num159_frame_191

# loc_array = np.array([data['x'] / 1024, data['y'] / 1024, z_norm, pitch_deg, yaw_deg])
# loc_coords = torch.tensor(loc_array, dtype=torch.float32)

