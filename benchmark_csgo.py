import os
import argparse
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
from transformers import CLIPProcessor, CLIPModel

# ---------------------------------------------------------
# Utils: 数据加载
# ---------------------------------------------------------

class PairedImageDataset(Dataset):
    """
    用于 PSNR, SSIM, LPIPS, CLIP-I 等需要一一对应计算的指标
    假设 GT 和 Pred 文件夹下文件名相同
    """
    def __init__(self, gt_dir, pred_dir, size=(224, 224)):
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        # 获取文件名交集，确保配对
        self.gt_images = sorted(os.listdir(gt_dir))
        self.pred_images = sorted(os.listdir(pred_dir))
        self.filenames = [f for f in self.gt_images if f in self.pred_images]

        if len(self.filenames) == 0:
            raise ValueError(f"No common filenames found between {gt_dir} and {pred_dir}")

        # 预处理: 转 Tensor 并归一化到 [0, 1]
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
    """用于 Aesthetic Score 或 FID (如果不需要配对加载)"""
    def __init__(self, img_dir, pred_dir, size=(224, 224)):
        # self.img_paths = sorted(glob(os.path.join(img_dir, "*.*")))
        self.img_dir = img_dir
        self.pred_dir = pred_dir
        # 获取文件名交集，确保pred和gt配对
        # 在收集的cs2数据集中train_images/val_images/test_images都是imgs的子集, 且放在一个目录下由splits.json划分
        self.images = sorted(os.listdir(img_dir))
        self.pred_images = sorted(os.listdir(pred_dir))
        self.filenames = [f for f in self.images if f in self.pred_images]

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(), # [0, 1]
            transforms.ConvertImageDtype(torch.uint8) # FID 需要 uint8 [0, 255]
        ])

        # 美学评分通常需要特定的 transform (CLIP transform)，在函数内部处理
        self.raw_transform = transforms.Compose([
             transforms.Resize(size),
             transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # path = self.img_paths[idx]
        filename = self.filenames[idx]
        path = os.path.join(self.img_dir, filename)
        img = Image.open(path).convert('RGB')
        return self.transform(img), self.raw_transform(img) # 返回 uint8 和 float 两种格式

# ---------------------------------------------------------
# Metric 1: FID (Frechet Inception Distance)
# ---------------------------------------------------------
def compute_fid(gt_dir, pred_dir, batch_size, device):
    print(f"🔄 Calculating FID on {device}...")
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # 定义简单的 Dataset，FID 不需要配对，只需要两个分布
    dataset_gt = SingleImageDataset(gt_dir, pred_dir, size=(299, 299)) # Inception 需要 299
    dataset_pred = SingleImageDataset(pred_dir, pred_dir, size=(299, 299))

    loader_gt = DataLoader(dataset_gt, batch_size=batch_size, num_workers=4)
    loader_pred = DataLoader(dataset_pred, batch_size=batch_size, num_workers=4)

    # Update GT
    for batch_uint8, _ in tqdm(loader_gt, desc="FID (GT Distribution)"):
        fid.update(batch_uint8.to(device), real=True)

    # Update Pred
    for batch_uint8, _ in tqdm(loader_pred, desc="FID (Pred Distribution)"):
        fid.update(batch_uint8.to(device), real=False)

    fid_score = fid.compute()
    print(f"✅ FID Score: {fid_score.item():.4f}")
    return fid_score.item()

# ---------------------------------------------------------
# Metric 2: LPIPS (Learned Perceptual Image Patch Similarity)
# ---------------------------------------------------------
def compute_lpips(gt_dir, pred_dir, batch_size, device):
    print(f"🔄 Calculating LPIPS on {device}...")
    # net_type 可选 'alex', 'vgg', 'squeeze'
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)

    dataset = PairedImageDataset(gt_dir, pred_dir)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    total_lpips = 0.0
    count = 0

    with torch.no_grad():
        for gt_batch, pred_batch in tqdm(loader, desc="LPIPS"):
            gt_batch = gt_batch.to(device) # [0, 1] float
            pred_batch = pred_batch.to(device)
            # Normalize to [-1, 1] for LPIPS
            gt_batch = gt_batch * 2.0 - 1.0
            pred_batch = pred_batch * 2.0 - 1.0

            score = lpips(pred_batch, gt_batch)
            total_lpips += score.item() * gt_batch.size(0) # torchmetrics 会自动平均，这里手动累加更可控
            count += gt_batch.size(0)

    avg_lpips = total_lpips / count
    print(f"✅ LPIPS Score: {avg_lpips:.4f}")
    return avg_lpips

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
            gt_batch = gt_batch.to(device)
            pred_batch = pred_batch.to(device)

            # Update metrics
            # 这里的 update 逻辑取决于 torchmetrics 版本，部分版本支持 update 后 compute，
            # 部分直接 forward 返回 batch 结果。这里手动累加 batch 均值。

            batch_psnr = psnr_metric(pred_batch, gt_batch)
            batch_ssim = ssim_metric(pred_batch, gt_batch)

            total_psnr += batch_psnr.item() * gt_batch.size(0)
            total_ssim += batch_ssim.item() * gt_batch.size(0)
            count += gt_batch.size(0)

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count

    print(f"✅ PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim

# ---------------------------------------------------------
# Metric 5: CLIP Score (Image-to-Image Semantic Consistency)
# ---------------------------------------------------------
def compute_clip_score(gt_dir, pred_dir, batch_size, device):
    """
    计算 GT 和 Pred 之间的 CLIP Embedding Cosine Similarity。
    衡量生成图是否保持了原图的语义信息。
    """
    print(f"🔄 Calculating CLIP Score (I2I) on {device}...")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    dataset = PairedImageDataset(gt_dir, pred_dir)
    # CLIP processor 通常需要原始 PIL image 或特定 transform，这里简化使用 tensor resize
    # 为了准确性，最好还原成 PIL 再过 processor，但这里为了速度使用 simple resize
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    total_clip_score = 0.0
    count = 0

    with torch.no_grad():
        for gt_batch, pred_batch in tqdm(loader, desc="CLIP Score"):
            # Resize for CLIP (224x224) - transform already did this, but CLIP needs normalization
            # 手动做 CLIP normalization
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

            gt_norm = (gt_batch.to(device) - mean) / std
            pred_norm = (pred_batch.to(device) - mean) / std

            # Get Embeddings
            gt_features = model.get_image_features(pixel_values=gt_norm)
            pred_features = model.get_image_features(pixel_values=pred_norm)

            # Normalize features
            gt_features = gt_features / gt_features.norm(dim=1, keepdim=True)
            pred_features = pred_features / pred_features.norm(dim=1, keepdim=True)

            # Cosine similarity
            similarity = (gt_features * pred_features).sum(dim=1)
            total_clip_score += similarity.sum().item()
            count += gt_batch.size(0)

    avg_clip = total_clip_score / count
    print(f"✅ CLIP Score: {avg_clip:.4f}")
    return avg_clip

# ---------------------------------------------------------
# Metric 6: Aesthetic Score
# ---------------------------------------------------------
class AestheticPredictor(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

def compute_aesthetic_score(pred_dir, batch_size, device):
    print(f"🔄 Calculating Aesthetic Score on {device}...")

    # 1. Load CLIP for features
    model_name = "openai/clip-vit-large-patch14" # Aesthetic predictors usually use ViT-L/14
    clip_model = CLIPModel.from_pretrained(model_name).to(device)

    # 2. Load Aesthetic MLP Head
    # 这里的权重来自 LAION-Aesthetics V2
    # 为了方便，这里尝试自动下载，如果失败请手动下载
    weight_url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
    weight_path = "aesthetic_model.pth"

    if not os.path.exists(weight_path):
        print("Downloading Aesthetic Predictor weights...")
        torch.hub.download_url_to_file(weight_url, weight_path)

    predictor = AestheticPredictor(768) # ViT-L/14 embedding dim is 768
    state_dict = torch.load(weight_path, map_location=device)
    predictor.load_state_dict(state_dict)
    predictor.to(device)
    predictor.eval()

    dataset = SingleImageDataset(pred_dir, pred_dir, size=(224, 224))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    total_score = 0.0
    count = 0

    with torch.no_grad():
        for _, img_batch in tqdm(loader, desc="Aesthetic Score"):
            # Normalize for CLIP
            img_batch = img_batch.to(device)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
            img_norm = (img_batch - mean) / std

            # Get CLIP features
            image_features = clip_model.get_image_features(pixel_values=img_norm)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            # Predict Score
            score = predictor(image_features.float())
            total_score += score.sum().item()
            count += img_batch.size(0)

    avg_score = total_score / count
    print(f"✅ Aesthetic Score: {avg_score:.4f}")
    return avg_score

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Image Generation Metrics")
    parser.add_argument("--gt", type=str, required=True, help="Path to Ground Truth images")
    parser.add_argument("--pred", type=str, required=True, help="Path to Predicted/Generated images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for calculation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Flags to enable specific metrics
    parser.add_argument("--fid", action="store_true", help="Calculate FID")
    parser.add_argument("--lpips", action="store_true", help="Calculate LPIPS")
    parser.add_argument("--psnr_ssim", action="store_true", help="Calculate PSNR and SSIM")
    parser.add_argument("--clip", action="store_true", help="Calculate CLIP Score (I2I)")
    parser.add_argument("--aesthetic", action="store_true", help="Calculate Aesthetic Score")
    parser.add_argument("--all", action="store_true", help="Calculate ALL metrics")

    args = parser.parse_args()

    results = {}

    if args.all or args.fid:
        results['FID'] = compute_fid(args.gt, args.pred, args.batch_size, args.device)

    if args.all or args.lpips:
        results['LPIPS'] = compute_lpips(args.gt, args.pred, args.batch_size, args.device)

    if args.all or args.psnr_ssim:
        psnr, ssim = compute_psnr_ssim(args.gt, args.pred, args.batch_size, args.device)
        results['PSNR'] = psnr
        results['SSIM'] = ssim

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



# python benchmark_csgo.py --gt ./data/gt --pred ./data/pred --all
# python benchmark_csgo.py --gt ./data/gt --pred ./data/pred --fid --lpips
# python benchmark_csgo.py --gt ./data/gt --pred ./data/pred --all --batch_size 8


# python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp2_1/test_20260103_220021/gen_imgs/de_dust2 --all --batch_size 8
# ==============================
# 📊 Final Results
# ==============================
# FID: 33.4978
# LPIPS: 0.2629
# PSNR: 19.8852
# SSIM: 0.5160
# CLIP: 0.8988
# Aesthetic: 4.4276
# ==============================

# CUDA_VISIBLE_DEVICES=1  python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp2_1/test_20260103_220021/gen_imgs/de_nuke --all --batch_size 2
# ==============================
# 📊 Final Results
# ==============================
# FID: 14.9368
# LPIPS: 0.3799
# PSNR: 16.8579
# SSIM: 0.4600
# CLIP: 0.9020
# Aesthetic: 4.9152
# ==============================

# CUDA_VISIBLE_DEVICES=0  python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp2_1/test_20260103_220021/gen_imgs/de_ancient --all --batch_size 2
# ==============================
# 📊 Final Results
# ==============================
# FID: 17.6177
# LPIPS: 0.2699
# PSNR: 21.3446
# SSIM: 0.4513
# CLIP: 0.9346
# Aesthetic: 4.9462




# python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp1/test_20251231_005658/gen_imgs --all --batch_size 2
# ==============================
# 📊 Final Results
# ==============================
# FID: 14.7104
# LPIPS: 0.2806
# PSNR: 17.8349
# SSIM: 0.4711
# CLIP: 0.9099
# Aesthetic: 4.7625
# ==============================




# python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp2/test_20260103_144824/gen_imgs/de_dust2 --all --batch_size
# ==============================
# 📊 Final Results
# ==============================
# FID: 14.8917
# LPIPS: 0.2797
# PSNR: 17.8463
# SSIM: 0.4699
# CLIP: 0.9114
# Aesthetic: 4.7218
# ==============================


# python benchmark_csgo.py --gt data/preprocessed_data/de_nuke/imgs --pred outputs_eval/exp2/test_20260103_144824/gen_imgs/de_nuke --all --batch_size 2
# ==============================
# 📊 Final Results
# ==============================
# FID: 13.7437
# LPIPS: 0.3116
# PSNR: 18.9030
# SSIM: 0.5467
# CLIP: 0.9117
# Aesthetic: 4.8210
# ==============================


# python benchmark_csgo.py --gt data/preprocessed_data/de_ancient/imgs --pred outputs_eval/exp2/test_20260103_144824/gen_imgs/de_ancient --all --batch_size 2
# ==============================
# 📊 Final Results
# ==============================
# FID: 15.1631
# LPIPS: 0.2801
# PSNR: 21.0726
# SSIM: 0.4663
# CLIP: 0.9327
# Aesthetic: 4.9281
# ==============================


# python benchmark_csgo.py --gt data/preprocessed_data/de_dust2/imgs --pred outputs_eval/exp3_1/test_20260109_013431/gen_imgs/de_dust2 --all --batch_size 2
# ==============================
# 📊 Final Results
# ==============================
# FID: 38.8108
# LPIPS: 0.5660
# PSNR: 11.9137
# SSIM: 0.3808
# CLIP: 0.8030
# Aesthetic: 4.7126
# ==============================