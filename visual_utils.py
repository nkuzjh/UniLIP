import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import textwrap



def visualize_dataset_samples_v1(dataset, processor, num_samples=20, save_path="_debug_dataset_samples.jpg", is_multi_task=True):
    """
    直接对已加载的 Dataset 进行抽样可视化，验证 Radar(Input) -> FPS(Target) 及 Prompt 对齐情况。

    Args:
        dataset: 已经初始化好的 CSGOWorldModelDataset/UniLIPMultiTaskDataset 实例
        processor: 对应的 ImageProcessor (用于获取 mean/std 进行逆归一化)
        num_samples: 抽样数量
        save_path: 图片保存路径
        is_multi_task: 是否为多任务(强制开启4列显示)
    """
    print(f"👀 Visualizing first {num_samples} samples from dataset...")

    # 获取反归一化所需的均值和方差
    img_processor = getattr(processor, "image_processor", processor)
    mean = np.array(img_processor.image_mean)
    std = np.array(img_processor.image_std)

    # 改为 4 列对应: und_image, aux_image, gen_image, full_prompt
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 4 * num_samples))
    plt.subplots_adjust(hspace=0.4, wspace=0.1)

    # 兼容 num_samples=1 的情况
    if num_samples == 1: axes = np.array([axes])

    for i in range(num_samples):
        # 1. 获取数据 (__getitem__)
        sample = dataset[i]

        task_id = sample.get('task_id', 1)  # 0=Loc, 1=Gen
        map_name = sample['map_name']
        min_z = dataset.map_z_range[map_name]['min_z']
        max_z = dataset.map_z_range[map_name]['max_z']
        fps_img_name = sample['ids']

        # 2. 提取图像 Tensor (安全提取，防止 squeeze(0) 误伤 C 维度)
        def safe_get_img(tensor):
            if tensor is None: return None
            if tensor.dim() == 4: return tensor[0] # [1, C, H, W] -> [C, H, W]
            return tensor # [C, H, W]

        und_tensor = safe_get_img(sample['und_image'])
        aux_tensor = safe_get_img(sample.get('aux_image'))
        gen_tensor = safe_get_img(sample['gen_image'])

        # 提取真值坐标
        loc_coords = sample.get('loc_coords', sample['actions'])
        if loc_coords.dim() == 2:
            loc_coords = loc_coords[0] # [1, 5] -> [5]

        x, y, z, v, h = loc_coords[0].item(), loc_coords[1].item(), loc_coords[2].item(), loc_coords[3].item(), loc_coords[4].item()
        x = x * 1024
        y = y * 1024
        z = z * (max_z - min_z) + min_z
        v = v * 360
        h = h * 360

        # 3. 逆归一化 (Tensor -> Numpy Image)
        def denorm(tensor):
            if tensor is None: return None
            # 过滤全 0 张量（占位符图），直接返回纯黑，防止加了 mean 变成灰色
            if torch.max(tensor) == 0 and torch.min(tensor) == 0:
                return np.zeros((tensor.shape[1], tensor.shape[2], 3))

            img = tensor.permute(1, 2, 0).cpu().numpy() # [H, W, C]
            img = img * std + mean # 反归一化
            return np.clip(img, 0, 1)

        img_und = denorm(und_tensor)
        img_aux = denorm(aux_tensor)
        img_gen = denorm(gen_tensor)

        # 4. 解码文本 Prompt (过滤 IGNORE_INDEX 防止报错)
        input_ids = sample['input_ids']
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        # 过滤 -100 和特殊占位符
        from unilip.constants import IGNORE_INDEX
        clean_input_ids = [idx for idx in input_ids if idx != IGNORE_INDEX and idx > 0]
        decoded_text = dataset.tokenizer.decode(clean_input_ids, skip_special_tokens=True).strip()

        # 完整提取，不进行长度截断，只需做折行以适应单列宽度
        full_prompt = textwrap.fill(decoded_text, width=50)

        # 5. 绘图 (根据 Task 动态调整 Title)
        task_name = "LOCALIZATION" if task_id == 0 else "GENERATION"

        # [列 0]: 独立展示完整 Prompt
        axes[i, 0].axis('off')
        axes[i, 0].set_title("Full Text Prompt", fontsize=10, fontweight='bold', color='purple')
        # 将文本贴在列的左上角 (0.0, 1.0)
        axes[i, 0].text(0.0, 1.0, full_prompt,
                        transform=axes[i, 0].transAxes,
                        fontsize=9, va='top', ha='left',
                        bbox=dict(facecolor='whitesmoke', alpha=0.9, edgecolor='lightgray', boxstyle='round,pad=0.5'))

        # [列 1]: und_image
        if img_und is not None:
            axes[i, 1].imshow(img_und)
            title_und = "Input: FPS" if task_id == 0 else "Input: Radar"
            axes[i, 1].set_title(f"[{task_name}]\n{title_und} | {fps_img_name}", fontsize=10, fontweight='bold')
        axes[i, 1].axis('off')

        # [列 2]: aux_image
        if img_aux is not None:
            axes[i, 2].imshow(img_aux)
            title_aux = "Aux Input: Radar" if task_id == 0 else "Aux Input: Empty"
            axes[i, 2].set_title(title_aux, fontsize=10, color='green')
        axes[i, 2].axis('off')

        # [列 3]: gen_image
        if img_gen is not None:
            axes[i, 3].imshow(img_gen)
            title_gen = "Target: Empty" if task_id == 0 else "Target: FPS"
            axes[i, 3].set_title(f"{title_gen}\nGT: x:{x:.1f}, y:{y:.1f}, z:{z:.1f}\np:{v:.1f}, y:{h:.1f}",
                                 fontsize=10, color='darkblue')
        axes[i, 3].axis('off')

    # 保存
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"✨ Visualization saved to: {os.path.abspath(save_path)}")
    plt.close(fig)



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

        if img_aux is not None:
            axes[i, 1].imshow(img_aux)
            axes[i, 1].set_title(f"Auxilliary for \nMulti-Task Training", fontsize=9, color='green')
            axes[i, 1].axis('off')

        # 右侧: FPS View
        axes[i, 2].imshow(img_fps)
        axes[i, 2].set_title(f"Target: FPS {fps_img_name}, \ngt_pose{x,y,z,v,h}", fontsize=9, color='darkblue')
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




def visualize_dataset_samples_paired(dataset, processor, num_samples=5, save_path="_debug_dataset_paired.jpg"):
    """
    可视化成对采样 (Pair-wise Sampling) 的数据集。
    验证同一个 Index 返回的 Loc 和 Gen 任务数据是否共享同一场景，但输入/输出互逆。

    Args:
        dataset: 返回 [sample_loc, sample_gen] 的 Dataset
        processor: ImageProcessor (用于反归一化)
    """
    print(f"👀 Visualizing first {num_samples} pairs from dataset...")

    # 1. 获取反归一化参数
    img_processor = getattr(processor, "image_processor", processor)
    mean = np.array(img_processor.image_mean)
    std = np.array(img_processor.image_std)

    def denorm(tensor):
        """Tensor [C, H, W] -> Numpy [H, W, C] (0-1)"""
        if tensor is None: return np.zeros((448, 448, 3))
        # 处理可能存在的 Batch 维度 [1, C, H, W]
        if tensor.dim() == 4: tensor = tensor.squeeze(0)

        img = tensor.permute(1, 2, 0).cpu().numpy()
        img = img * std + mean
        return np.clip(img, 0, 1)

    # 2. 设置画布: 每行 3 列 (Map, FPS, Info Panel)
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6 * num_samples))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    if num_samples == 1: axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # --- A. 获取成对数据 ---
        # dataset[i] 现在返回的是一个 list: [sample_loc, sample_gen]
        pair_data = dataset[i]
        sample_loc = pair_data[0]
        sample_gen = pair_data[1]

        # 校验 Task ID
        assert sample_loc['task_id'] == 0, f"Index 0 should be Loc, got {sample_loc['task_id']}"
        assert sample_gen['task_id'] == 1, f"Index 1 should be Gen, got {sample_gen['task_id']}"

        # --- B. 提取图像 ---
        # 逻辑验证：
        # Loc Task: Input(Und)=FPS, Aux=Map
        # Gen Task: Input(Und)=Map, Target(Gen)=FPS

        # 我们从 Gen 任务取 Map (Und)，从 Loc 任务取 FPS (Und)
        # 这样能同时验证两个任务的 Input 也是正确的
        tensor_map = sample_gen['und_image']
        tensor_fps = sample_loc['und_image']

        # 也可以验证 Loc 的 Aux 是否等于 Map
        tensor_map_aux = sample_loc['aux_image']

        img_map = denorm(tensor_map)
        img_fps = denorm(tensor_fps)
        img_map_aux = denorm(tensor_map_aux)

        # --- C. 提取 Meta 信息 ---
        map_name = sample_loc['map_name']
        pose = sample_loc['pose_dict']

        # 解码 Prompt
        input_ids_loc = sample_loc['input_ids']
        if isinstance(input_ids_loc, torch.Tensor): input_ids_loc = input_ids_loc.tolist()
        text_loc = dataset.tokenizer.decode(input_ids_loc, skip_special_tokens=False)

        input_ids_gen = sample_gen['input_ids']
        if isinstance(input_ids_gen, torch.Tensor): input_ids_gen = input_ids_gen.tolist()
        text_gen = dataset.tokenizer.decode(input_ids_gen, skip_special_tokens=False)

        # --- D. 绘图 ---

        # Column 1: Radar Map (Gen Input / Loc Aux)
        # 为了展示 Aux 是否正确，我们在 Map 上叠加一个小图或者标题说明
        ax_map = axes[i, 0]
        ax_map.imshow(img_map)
        diff = np.mean(np.abs(img_map - img_map_aux)) # 验证一致性
        ax_map.set_title(f"Map '{map_name}'\nGen Input / Loc Aux\n(Consistency Diff: {diff:.1e})", fontsize=11, fontweight='bold')
        ax_map.axis('off')

        # Column 2: FPS View (Loc Input / Gen Target)
        ax_fps = axes[i, 1]
        ax_fps.imshow(img_fps)
        ax_fps.set_title(f"FPS View\nLoc Input / Gen Target\nID: {sample_loc['ids']}", fontsize=11, fontweight='bold')
        ax_fps.axis('off')

        # Column 3: Info Panel (Text & Pose)
        ax_text = axes[i, 2]
        ax_text.axis('off')

        # 构造信息文本
        pose_str = f"x={pose['x']:.1f}, y={pose['y']:.1f}, z={pose['z']:.2f}\npitch={pose['angle_v']:.1f}, yaw={pose['angle_h']:.1f}"

        # 截取 Prompt 关键部分
        prompt_loc_short = text_loc.split("Task:")[-1].split("Predict")[0].strip()[:100] + "..."
        prompt_gen_short = text_gen.split("Task:")[-1].split("Coordinate")[0].strip()[:100] + "..."

        info_text = (
            f"Sample Pair Index: {i}\n"
            f"--------------------------------\n"
            f"Map: {map_name}\n"
            f"GT Pose:\n{pose_str}\n"
            f"--------------------------------\n"
            f"[Task Loc] Raw Prompt Snippet:\n...{textwrap.fill(prompt_loc_short, 35)}\n"
            f"Loss Mask: {sample_loc['loss_mask'].tolist()}\n"
            f"--------------------------------\n"
            f"[Task Gen] Raw Prompt Snippet:\n...{textwrap.fill(prompt_gen_short, 35)}\n"
            f"Loss Mask: {sample_gen['loss_mask'].tolist()}\n"
        )

        ax_text.text(0, 0.95, info_text, fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", fc="#f9f9f9", ec="gray", alpha=0.5))

    # 保存
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    print(f"✨ Paired visualization saved to: {os.path.abspath(save_path)}")
    plt.close()