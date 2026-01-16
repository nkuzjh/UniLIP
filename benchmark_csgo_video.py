import os
import argparse
import re
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# 引入指标库
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
# 新增: FVD
from torchmetrics.video import FrechetVideoDistance

from transformers import CLIPProcessor, CLIPModel

# 新增: DreamSim (尝试导入，避免未安装报错)
try:
    from dreamsim import dreamsim
    DREAMSIM_AVAILABLE = True
except ImportError:
    DREAMSIM_AVAILABLE = False
    print("⚠️ DreamSim library not found. Install via 'pip install dreamsim' to use --dreamsim")

# ---------------------------------------------------------
# Utils: 数据加载
# ---------------------------------------------------------

class PairedImageDataset(Dataset):
    """
    用于 PSNR, SSIM, LPIPS, DreamSim, CLIP-I 等需要一一对应计算的指标
    """
    def __init__(self, gt_dir, pred_dir, size=(224, 224)):
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.gt_images = sorted(os.listdir(gt_dir))
        self.pred_images = sorted(os.listdir(pred_dir))
        self.filenames = [f for f in self.gt_images if f in self.pred_images]

        if len(self.filenames) == 0:
            raise ValueError(f"No common filenames found between {gt_dir} and {pred_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        gt_path = os.path.join(self.gt_dir, filename)
        pred_path = os.path.join(self.pred_dir, filename)

        gt_img = Image.open(gt_path).convert('RGB')
        pred_img = Image.open(pred_path).convert('RGB')

        return self.transform(gt_img), self.transform(pred_img)

class SingleImageDataset(Dataset):
    """用于 Aesthetic Score 或 FID"""
    def __init__(self, img_dir, filter_list=None, size=(224, 224)):
        self.img_dir = img_dir
        all_files = sorted(os.listdir(img_dir))

        # 如果提供了 filter_list (比如只计算配对的图片)，则过滤
        if filter_list:
            self.filenames = [f for f in all_files if f in filter_list]
        else:
            self.filenames = all_files

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.uint8)
        ])

        self.raw_transform = transforms.Compose([
             transforms.Resize(size),
             transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        path = os.path.join(self.img_dir, filename)
        img = Image.open(path).convert('RGB')
        return self.transform(img), self.raw_transform(img)

# --- 新增: 视频片段数据集 (用于 FVD) ---
class VideoClipDataset(Dataset):
    """
    将散装的帧图片自动组装成视频片段 (Clip)。
    FVD (I3D) 通常要求输入为 16 帧的片段。
    """
    def __init__(self, img_dir, clip_length=16, size=(224, 224)):
        self.img_dir = img_dir
        self.clip_length = clip_length
        self.size = size

        # 1. 扫描并解析所有文件
        files = []
        for f in os.listdir(img_dir):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                match = re.search(r'file_num(\d+)_frame_(\d+)', f)
                if match:
                    files.append({
                        'path': os.path.join(img_dir, f),
                        'file_num': int(match.group(1)),
                        'frame_id': int(match.group(2))
                    })

        # 2. 排序
        files.sort(key=lambda x: (x['file_num'], x['frame_id']))

        # 3. 分组为 Clips
        self.clips = []
        current_clip = []
        for i, item in enumerate(files):
            if not current_clip:
                current_clip.append(item)
                continue

            last_item = current_clip[-1]
            # 连续性检查: 同文件 & 帧号连续 (允许跳帧阈值2)
            if item['file_num'] == last_item['file_num'] and \
               (item['frame_id'] - last_item['frame_id'] <= 2):
                current_clip.append(item)
            else:
                # 连续性中断，重置
                current_clip = [item]

            # 如果凑够了 clip_length 帧，保存为一个样本
            if len(current_clip) == self.clip_length:
                self.clips.append(current_clip)
                # 滑动窗口: 如果想非重叠切分，这里清空 current_clip
                # 如果想重叠切分(数据更多)，这里 current_clip.pop(0)
                # FVD 计算通常使用非重叠片段即可
                current_clip = []

        print(f"🎬 Found {len(self.clips)} valid video clips (len={clip_length}) in {img_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.PILToTensor(), # FVD 需要 uint8 Tensor [0, 255]
        ])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_info = self.clips[idx]
        frames = []
        for info in clip_info:
            img = Image.open(info['path']).convert('RGB')
            # transform 返回 (C, H, W) uint8
            frames.append(self.transform(img))

        # Stack -> (T, C, H, W)
        video_tensor = torch.stack(frames)

        # FVD 要求输入格式: (B, C, T, H, W) 或 (B, T, C, H, W)
        # torchmetrics FVD 文档建议 (B, C, T, H, W)，这里返回 (C, T, H, W) 给 loader 堆叠
        return video_tensor.permute(1, 0, 2, 3)


# ---------------------------------------------------------
# Metric 1: FID
# ---------------------------------------------------------
def compute_fid(gt_dir, pred_dir, batch_size, device):
    print(f"🔄 Calculating FID on {device}...")
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # 获取文件名交集列表
    gt_files = set(os.listdir(gt_dir))
    pred_files = set(os.listdir(pred_dir))
    common_files = sorted(list(gt_files.intersection(pred_files)))

    dataset_gt = SingleImageDataset(gt_dir, filter_list=common_files, size=(299, 299))
    dataset_pred = SingleImageDataset(pred_dir, filter_list=common_files, size=(299, 299))

    loader_gt = DataLoader(dataset_gt, batch_size=batch_size, num_workers=4)
    loader_pred = DataLoader(dataset_pred, batch_size=batch_size, num_workers=4)

    for batch_uint8, _ in tqdm(loader_gt, desc="FID (GT)"):
        fid.update(batch_uint8.to(device), real=True)

    for batch_uint8, _ in tqdm(loader_pred, desc="FID (Pred)"):
        fid.update(batch_uint8.to(device), real=False)

    fid_score = fid.compute()
    print(f"✅ FID Score: {fid_score.item():.4f}")
    return fid_score.item()

# ---------------------------------------------------------
# Metric 2: LPIPS
# ---------------------------------------------------------
def compute_lpips(gt_dir, pred_dir, batch_size, device):
    print(f"🔄 Calculating LPIPS on {device}...")
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
    dataset = PairedImageDataset(gt_dir, pred_dir)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    total_lpips = 0.0
    count = 0
    with torch.no_grad():
        for gt_batch, pred_batch in tqdm(loader, desc="LPIPS"):
            gt_batch = (gt_batch.to(device) * 2.0 - 1.0)
            pred_batch = (pred_batch.to(device) * 2.0 - 1.0)
            score = lpips(pred_batch, gt_batch)
            total_lpips += score.item() * gt_batch.size(0)
            count += gt_batch.size(0)

    avg = total_lpips / count
    print(f"✅ LPIPS Score: {avg:.4f}")
    return avg

# ---------------------------------------------------------
# Metric 3 & 4: PSNR & SSIM
# ---------------------------------------------------------
def compute_psnr_ssim(gt_dir, pred_dir, batch_size, device):
    print(f"🔄 Calculating PSNR & SSIM on {device}...")
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    dataset = PairedImageDataset(gt_dir, pred_dir)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    with torch.no_grad():
        for gt_batch, pred_batch in tqdm(loader, desc="PSNR/SSIM"):
            gt_batch, pred_batch = gt_batch.to(device), pred_batch.to(device)
            total_psnr += psnr_metric(pred_batch, gt_batch).item() * gt_batch.size(0)
            total_ssim += ssim_metric(pred_batch, gt_batch).item() * gt_batch.size(0)
            count += gt_batch.size(0)

    print(f"✅ PSNR: {total_psnr/count:.4f} | SSIM: {total_ssim/count:.4f}")
    return total_psnr/count, total_ssim/count

# ---------------------------------------------------------
# Metric 5: CLIP Score
# ---------------------------------------------------------
def compute_clip_score(gt_dir, pred_dir, batch_size, device):
    print(f"🔄 Calculating CLIP Score on {device}...")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    dataset = PairedImageDataset(gt_dir, pred_dir)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    total_score = 0.0
    count = 0
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    with torch.no_grad():
        for gt_batch, pred_batch in tqdm(loader, desc="CLIP Score"):
            gt_norm = (gt_batch.to(device) - mean) / std
            pred_norm = (pred_batch.to(device) - mean) / std

            gt_emb = model.get_image_features(pixel_values=gt_norm)
            pred_emb = model.get_image_features(pixel_values=pred_norm)

            gt_emb = gt_emb / gt_emb.norm(dim=1, keepdim=True)
            pred_emb = pred_emb / pred_emb.norm(dim=1, keepdim=True)

            total_score += (gt_emb * pred_emb).sum(dim=1).sum().item()
            count += gt_batch.size(0)

    avg = total_score / count
    print(f"✅ CLIP Score: {avg:.4f}")
    return avg

# ---------------------------------------------------------
# Metric 6: Aesthetic Score
# ---------------------------------------------------------
class AestheticPredictor(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1024), torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64), torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16), torch.nn.Linear(16, 1)
        )
    def forward(self, x): return self.layers(x)

def compute_aesthetic_score(pred_dir, batch_size, device):
    print(f"🔄 Calculating Aesthetic Score on {device}...")
    model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(model_name).to(device)

    weight_url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
    weight_path = "aesthetic_model.pth"
    if not os.path.exists(weight_path):
        torch.hub.download_url_to_file(weight_url, weight_path)

    predictor = AestheticPredictor(768)
    predictor.load_state_dict(torch.load(weight_path, map_location=device))
    predictor.to(device).eval()

    dataset = SingleImageDataset(pred_dir, size=(224, 224))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    total = 0.0
    count = 0
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    with torch.no_grad():
        for _, img_batch in tqdm(loader, desc="Aesthetic"):
            img_norm = (img_batch.to(device) - mean) / std
            features = clip_model.get_image_features(pixel_values=img_norm)
            features = features / features.norm(dim=1, keepdim=True)
            total += predictor(features.float()).sum().item()
            count += img_batch.size(0)

    avg = total / count
    print(f"✅ Aesthetic Score: {avg:.4f}")
    return avg

# ---------------------------------------------------------
# New Metric 7: DreamSim (Perceptual)
# ---------------------------------------------------------
def compute_dreamsim(gt_dir, pred_dir, batch_size, device):
    if not DREAMSIM_AVAILABLE:
        print("❌ DreamSim not installed. Skipping.")
        return 0.0

    print(f"🔄 Calculating DreamSim on {device}...")
    # DreamSim model loading (pretrained=True loads OpenCLIP-ViT-B-32 variant by default)
    model, preprocess = dreamsim(pretrained=True, device=device)

    dataset = PairedImageDataset(gt_dir, pred_dir)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    total_dist = 0.0
    count = 0

    with torch.no_grad():
        for gt_batch, pred_batch in tqdm(loader, desc="DreamSim"):
            # DreamSim handles normalization internally if using their preprocess,
            # but here we have standard tensors [0,1].
            # DreamSim forward expects tensors.
            gt_batch = gt_batch.to(device)
            pred_batch = pred_batch.to(device)

            # DreamSim returns distance for each pair
            dist = model(pred_batch, gt_batch)
            total_dist += dist.sum().item()
            count += gt_batch.size(0)

    avg = total_dist / count
    print(f"✅ DreamSim Score: {avg:.4f} (Lower is better)")
    return avg

# ---------------------------------------------------------
# New Metric 8: FVD (Fréchet Video Distance)
# ---------------------------------------------------------
def compute_fvd(gt_dir, pred_dir, batch_size, device):
    print(f"🔄 Calculating FVD on {device}...")
    # I3D 是 FVD 的标准特征提取器
    fvd = FrechetVideoDistance(feature_extractor="i3d400", reset_real_features=False, reset_fake_features=False).to(device)

    # 视频数据处理: 16帧为一个Clip
    # FVD 需要视频输入，我们需要把图片组合成视频片段
    clip_len = 16
    dataset_gt = VideoClipDataset(gt_dir, clip_length=clip_len, size=(224, 224))
    dataset_pred = VideoClipDataset(pred_dir, clip_length=clip_len, size=(224, 224))

    if len(dataset_gt) == 0 or len(dataset_pred) == 0:
        print(f"❌ Not enough contiguous frames for FVD (Need {clip_len} frames per clip). Skipping.")
        return 0.0

    # FVD 比较耗显存，建议 batch_size 调小
    vid_batch_size = max(1, batch_size // 4)

    loader_gt = DataLoader(dataset_gt, batch_size=vid_batch_size, num_workers=4)
    loader_pred = DataLoader(dataset_pred, batch_size=vid_batch_size, num_workers=4)

    print(f"  - Found {len(dataset_gt)} Real Clips, {len(dataset_pred)} Fake Clips.")

    # Update GT (Real)
    for batch_vid in tqdm(loader_gt, desc="FVD (Real Clips)"):
        # batch_vid shape: (B, C, T, H, W), uint8 [0, 255]
        fvd.update(batch_vid.to(device), real=True)

    # Update Pred (Fake)
    for batch_vid in tqdm(loader_pred, desc="FVD (Fake Clips)"):
        fvd.update(batch_vid.to(device), real=False)

    fvd_score = fvd.compute()
    print(f"✅ FVD Score: {fvd_score.item():.4f} (Lower is better)")
    return fvd_score.item()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Image/Video Generation Metrics")
    parser.add_argument("--gt", type=str, required=True, help="Path to GT images")
    parser.add_argument("--pred", type=str, required=True, help="Path to Generated images")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--all", action="store_true", help="Run ALL metrics")
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--psnr_ssim", action="store_true")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--aesthetic", action="store_true")
    # New flags
    parser.add_argument("--dreamsim", action="store_true", help="Calculate DreamSim (Perceptual)")
    parser.add_argument("--fvd", action="store_true", help="Calculate FVD (Video)")

    args = parser.parse_args()
    results = {}

    if args.all or args.psnr_ssim:
        p, s = compute_psnr_ssim(args.gt, args.pred, args.batch_size, args.device)
        results['PSNR'], results['SSIM'] = p, s

    if args.all or args.lpips:
        results['LPIPS'] = compute_lpips(args.gt, args.pred, args.batch_size, args.device)

    if args.all or args.dreamsim:
        results['DreamSim'] = compute_dreamsim(args.gt, args.pred, args.batch_size, args.device)

    if args.all or args.fid:
        results['FID'] = compute_fid(args.gt, args.pred, args.batch_size, args.device)

    if args.all or args.fvd:
        results['FVD'] = compute_fvd(args.gt, args.pred, args.batch_size, args.device)

    if args.all or args.clip:
        results['CLIP'] = compute_clip_score(args.gt, args.pred, args.batch_size, args.device)

    if args.all or args.aesthetic:
        results['Aesthetic'] = compute_aesthetic_score(args.pred, args.batch_size, args.device)

    print("\n" + "="*30)
    print("📊 Final Results")
    print("="*30)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()