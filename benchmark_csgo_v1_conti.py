import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

_FVD_REPO_ROOT = Path(__file__).resolve().parent / "third_party" / "PyTorch-Frechet-Video-Distance"
if str(_FVD_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_FVD_REPO_ROOT))
from fvd_metric import compute_fvd as compute_fvd_metric

from benchmark_csgo_v1 import (
    collect_coverage,
    compute_aesthetic_score,
    compute_boundary_metrics,
    compute_clip_score,
    compute_external_locator_metrics,
    compute_fid,
    compute_inception_score,
    compute_lpips,
    compute_pixel_metrics,
    compute_psnr_ssim,
    infer_map_name,
    resolve_eval_output_root,
)


CONTI_OUTPUT_ORDER = [
    "Coverage_GT",
    "Coverage_Pred",
    "Common_Count",
    "Track_Count",
    "Seq_Frame_Count",
    "PSNR",
    "SSIM",
    "Boundary_F1",
    "LPIPS",
    "Pixel_MAE_255",
    "Pixel_Exact_Acc",
    "Pixel_Within_1_Acc",
    "Locator_XY_Dist",
    "Locator_Z_Dist",
    "Locator_Pitch_Dist",
    "Locator_Yaw_Dist",
    "Locator_Norm_L2_5D",
    "FID",
    "IS",
    "CLIP",
    "Aesthetic",
    "Seq-PSNR",
    "Seq-SSIM",
    "Seq-LPIPS",
    "Seq-MAE",
    "Seq-Exact-Acc",
    "Seq-Within_1-Acc",
    "Temporal_Warping_Error",
    "Temporal_Difference_Error",
    "Flicker_Score",
    "Optical_Flow_EPE",
    "FVD",
]


@dataclass(frozen=True)
class FrameRecord:
    filename: str
    file_num: int
    frame_id: int


def parse_conti_filename(filename: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"file_num(\d+)_frame_(\d+)", filename)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def build_tracks(
    filenames: Sequence[str],
    frame_diff_threshold: int,
    min_track_len: int,
) -> Tuple[List[List[FrameRecord]], Dict[str, object]]:
    parsed_records: List[FrameRecord] = []
    unparsed_files = []
    for filename in filenames:
        parsed = parse_conti_filename(filename)
        if parsed is None:
            unparsed_files.append(filename)
            continue
        parsed_records.append(FrameRecord(filename=filename, file_num=parsed[0], frame_id=parsed[1]))

    parsed_records.sort(key=lambda item: (item.file_num, item.frame_id))
    raw_tracks: List[List[FrameRecord]] = []
    current_track: List[FrameRecord] = []

    for record in parsed_records:
        if not current_track:
            current_track = [record]
            continue
        last = current_track[-1]
        is_same_file = record.file_num == last.file_num
        is_contiguous = record.frame_id - last.frame_id <= frame_diff_threshold
        if is_same_file and is_contiguous:
            current_track.append(record)
        else:
            raw_tracks.append(current_track)
            current_track = [record]
    if current_track:
        raw_tracks.append(current_track)

    tracks = [track for track in raw_tracks if len(track) >= min_track_len]
    dropped_short_tracks = [track for track in raw_tracks if len(track) < min_track_len]
    track_summaries = [
        {
            "track_index": idx,
            "file_num": track[0].file_num,
            "length": len(track),
            "first_frame_id": track[0].frame_id,
            "last_frame_id": track[-1].frame_id,
            "first_filename": track[0].filename,
            "last_filename": track[-1].filename,
        }
        for idx, track in enumerate(tracks)
    ]
    details = {
        "parsed_frame_count": len(parsed_records),
        "unparsed_common_files": unparsed_files,
        "unparsed_common_files_count": len(unparsed_files),
        "raw_track_count": len(raw_tracks),
        "dropped_short_track_count": len(dropped_short_tracks),
        "track_summaries": track_summaries,
    }
    return tracks, details


def load_rgb_uint8(path: str, size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size), Image.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def load_track_pair(
    gt_dir: str,
    pred_dir: str,
    track: Sequence[FrameRecord],
    size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    gt_frames = []
    pred_frames = []
    for record in track:
        gt_frames.append(load_rgb_uint8(os.path.join(gt_dir, record.filename), size))
        pred_frames.append(load_rgb_uint8(os.path.join(pred_dir, record.filename), size))
    return np.stack(gt_frames, axis=0), np.stack(pred_frames, axis=0)


def tensor_from_uint8(frames: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float().div(255.0).to(device)


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if len(values) == 0:
        return None
    return float(np.mean(values))


def require_cv2():
    try:
        import cv2  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "benchmark_csgo_v1_conti.py requires opencv-python for temporal optical-flow metrics."
        ) from exc
    return cv2


def compute_sequence_metrics(
    gt_dir: str,
    pred_dir: str,
    tracks: Sequence[Sequence[FrameRecord]],
    size: int,
    device: str,
) -> Dict[str, Optional[float]]:
    if len(tracks) == 0:
        return {
            "Seq-PSNR": None,
            "Seq-SSIM": None,
            "Seq-LPIPS": None,
            "Seq-MAE": None,
            "Seq-Exact-Acc": None,
            "Seq-Within_1-Acc": None,
        }

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    track_psnr = []
    track_ssim = []
    track_lpips = []
    track_mae = []
    track_exact = []
    track_within_1 = []

    with torch.no_grad():
        for track in tqdm(tracks, desc="Sequence paired metrics"):
            gt_uint8, pred_uint8 = load_track_pair(gt_dir, pred_dir, track, size)
            gt_tensor = tensor_from_uint8(gt_uint8, device)
            pred_tensor = tensor_from_uint8(pred_uint8, device)

            mse = (pred_tensor - gt_tensor).square().flatten(start_dim=1).mean(dim=1)
            psnr = torch.where(
                mse <= 1e-12,
                torch.full_like(mse, 100.0),
                10.0 * torch.log10(1.0 / mse.clamp_min(1e-12)),
            )
            track_psnr.append(psnr.mean().item())
            track_ssim.append(ssim_metric(pred_tensor, gt_tensor).item())
            track_lpips.append(lpips_metric(pred_tensor * 2.0 - 1.0, gt_tensor * 2.0 - 1.0).item())

            diff = np.abs(pred_uint8.astype(np.int16) - gt_uint8.astype(np.int16))
            track_mae.append(float(diff.mean()))
            exact = np.all(diff == 0, axis=-1)
            within_1 = np.all(diff <= 1, axis=-1)
            track_exact.append(float(exact.mean()))
            track_within_1.append(float(within_1.mean()))

    return {
        "Seq-PSNR": mean_or_none(track_psnr),
        "Seq-SSIM": mean_or_none(track_ssim),
        "Seq-LPIPS": mean_or_none(track_lpips),
        "Seq-MAE": mean_or_none(track_mae),
        "Seq-Exact-Acc": mean_or_none(track_exact),
        "Seq-Within_1-Acc": mean_or_none(track_within_1),
    }


def estimate_farneback_flow(prev_rgb: np.ndarray, next_rgb: np.ndarray) -> np.ndarray:
    cv2 = require_cv2()
    prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def warp_with_flow(prev_rgb: np.ndarray, flow: np.ndarray) -> np.ndarray:
    cv2 = require_cv2()
    height, width = prev_rgb.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (grid_x - flow[..., 0]).astype(np.float32)
    map_y = (grid_y - flow[..., 1]).astype(np.float32)
    return cv2.remap(
        prev_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def compute_temporal_metrics(
    gt_dir: str,
    pred_dir: str,
    tracks: Sequence[Sequence[FrameRecord]],
    size: int,
) -> Dict[str, Optional[float]]:
    if len(tracks) == 0:
        return {
            "Temporal_Warping_Error": None,
            "Temporal_Difference_Error": None,
            "Flicker_Score": None,
            "Optical_Flow_EPE": None,
        }

    track_warping = []
    track_difference = []
    track_flicker = []
    track_flow_epe = []

    for track in tqdm(tracks, desc="Temporal metrics"):
        gt_uint8, pred_uint8 = load_track_pair(gt_dir, pred_dir, track, size)
        warping_errors = []
        difference_errors = []
        flicker_errors = []
        flow_epe_errors = []

        for idx in range(len(track) - 1):
            gt_prev = gt_uint8[idx]
            gt_next = gt_uint8[idx + 1]
            pred_prev = pred_uint8[idx]
            pred_next = pred_uint8[idx + 1]

            gt_delta = gt_next.astype(np.float32) - gt_prev.astype(np.float32)
            pred_delta = pred_next.astype(np.float32) - pred_prev.astype(np.float32)
            difference_errors.append(float(np.abs(pred_delta - gt_delta).mean()))
            flicker_errors.append(float(np.abs(np.abs(pred_delta) - np.abs(gt_delta)).mean()))

            flow_gt = estimate_farneback_flow(gt_prev, gt_next)
            flow_pred = estimate_farneback_flow(pred_prev, pred_next)
            flow_epe = np.linalg.norm(flow_pred - flow_gt, axis=-1)
            flow_epe_errors.append(float(flow_epe.mean()))

            pred_prev_warped = warp_with_flow(pred_prev, flow_gt)
            warping_errors.append(
                float(np.abs(pred_prev_warped.astype(np.float32) - pred_next.astype(np.float32)).mean())
            )

        track_warping.append(float(np.mean(warping_errors)))
        track_difference.append(float(np.mean(difference_errors)))
        track_flicker.append(float(np.mean(flicker_errors)))
        track_flow_epe.append(float(np.mean(flow_epe_errors)))

    return {
        "Temporal_Warping_Error": mean_or_none(track_warping),
        "Temporal_Difference_Error": mean_or_none(track_difference),
        "Flicker_Score": mean_or_none(track_flicker),
        "Optical_Flow_EPE": mean_or_none(track_flow_epe),
    }


def build_fvd_clips(
    tracks: Sequence[Sequence[FrameRecord]],
    clip_length: int,
    clip_stride: int,
) -> List[List[str]]:
    clips = []
    for track in tracks:
        if len(track) < clip_length:
            continue
        for start_idx in range(0, len(track) - clip_length + 1, clip_stride):
            clips.append([record.filename for record in track[start_idx:start_idx + clip_length]])
    return clips


class VideoClipDataset(Dataset):
    def __init__(self, image_dir: str, clips: Sequence[Sequence[str]], size: int):
        self.image_dir = image_dir
        self.clips = [list(clip) for clip in clips]
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            transforms.PILToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> torch.Tensor:
        frames = []
        for filename in self.clips[idx]:
            image = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
            frames.append(self.transform(image))
        return torch.stack(frames, dim=1)


def compute_fvd_for_tracks(
    gt_dir: str,
    pred_dir: str,
    tracks: Sequence[Sequence[FrameRecord]],
    clip_length: int,
    clip_stride: int,
    size: int,
    batch_size: int,
    device: str,
) -> Tuple[Optional[float], int]:
    clips = build_fvd_clips(tracks, clip_length=clip_length, clip_stride=clip_stride)
    if len(clips) == 0:
        return None, 0

    fvd_batch_size = max(1, batch_size // 4)
    gt_dataset = VideoClipDataset(gt_dir, clips, size=size)
    pred_dataset = VideoClipDataset(pred_dir, clips, size=size)
    gt_loader = DataLoader(gt_dataset, batch_size=fvd_batch_size, num_workers=4)
    pred_loader = DataLoader(pred_dataset, batch_size=fvd_batch_size, num_workers=4)

    gt_clips = []
    pred_clips = []
    with torch.no_grad():
        for batch in tqdm(gt_loader, desc="FVD (GT clips)"):
            gt_clips.append(batch.cpu())
        for batch in tqdm(pred_loader, desc="FVD (Pred clips)"):
            pred_clips.append(batch.cpu())

    gt_clips = torch.cat(gt_clips, dim=0)
    pred_clips = torch.cat(pred_clips, dim=0)
    fvd_score = compute_fvd_metric(
        gt_clips,
        pred_clips,
        max_items=len(clips),
        device=device,
        batch_size=fvd_batch_size,
    )
    if isinstance(fvd_score, torch.Tensor):
        fvd_score = fvd_score.item()
    return float(fvd_score), len(clips)


def ordered_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    return {key: metrics.get(key) for key in CONTI_OUTPUT_ORDER}


def print_results(metrics: Dict[str, object]) -> None:
    print("\n" + "=" * 34)
    print("CSGO Simulator Benchmark V1 Conti")
    print("=" * 34)
    for key in CONTI_OUTPUT_ORDER:
        value = metrics.get(key)
        if value is None:
            print(f"{key}: None")
        elif isinstance(value, int):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {float(value):.6f}")
    print("=" * 34)


def write_results_json(
    json_path: Path,
    *,
    experiment_name: Optional[str],
    timestamp: Optional[str],
    map_name: str,
    gt_dir: str,
    pred_dir: str,
    metrics: Dict[str, object],
    coverage: Dict[str, object],
    args: argparse.Namespace,
    locator_details: Dict[str, object],
    boundary_details: Dict[str, object],
    track_details: Dict[str, object],
    fvd_clip_count: int,
) -> None:
    payload = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "map_name": map_name,
        "gt_dir": str(Path(gt_dir).resolve()),
        "pred_dir": str(Path(pred_dir).resolve()),
        "metrics_order": CONTI_OUTPUT_ORDER,
        "metrics_ordered": ordered_metrics(metrics),
        "unmatched_pred_files": coverage["unmatched_pred_files"],
        "unmatched_pred_files_count": len(coverage["unmatched_pred_files"]),
        "missing_pred_files": coverage["missing_pred_files"],
        "missing_pred_files_count": len(coverage["missing_pred_files"]),
        "gt_count": coverage["GT_Count"],
        "pred_count": coverage["Pred_Count"],
        "common_count": coverage["Common_Count"],
        "config": {
            "paired_size": args.paired_size,
            "batch_size": args.batch_size,
            "device": args.device,
            "data_dir": args.data_dir,
            "pose_json": args.pose_json,
            "external_loc_repo_root": args.external_loc_repo_root,
            "external_loc_config_path": args.external_loc_config_path,
            "external_loc_checkpoint_path": args.external_loc_checkpoint_path,
            "edge_quantile": args.edge_quantile,
            "edge_tolerance": args.edge_tolerance,
            "frame_diff_threshold": args.frame_diff_threshold,
            "min_track_len": args.min_track_len,
            "clip_length": args.clip_length,
            "clip_stride": args.clip_stride,
            "fvd_size": args.fvd_size,
            "locator_pose_json_paths": locator_details.get("locator_pose_json_paths", []),
            "locator_z_min": locator_details.get("locator_z_min"),
            "locator_z_max": locator_details.get("locator_z_max"),
        },
        "details": {
            "boundary": boundary_details,
            "tracks": track_details,
            "fvd_clip_count": fvd_clip_count,
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_benchmark_v1_conti(args: argparse.Namespace) -> Dict[str, object]:
    coverage = collect_coverage(args.gt, args.pred)
    common_files = coverage["common_files"]
    if len(common_files) == 0:
        raise ValueError("No common filenames found between GT and Pred directories.")

    map_name = infer_map_name(args.gt, args.pred, args.map_name)
    output_root, experiment_name, timestamp, _ = resolve_eval_output_root(args.pred, map_name)
    json_path = output_root / f"benchmark_csgo_v1_conti_{map_name}.json"

    print(f"Experiment: {experiment_name}")
    print(f"Timestamp: {timestamp}")
    print(f"Map: {map_name}")
    print(f"Output JSON: {json_path}")

    tracks, track_details = build_tracks(
        common_files,
        frame_diff_threshold=args.frame_diff_threshold,
        min_track_len=args.min_track_len,
    )
    seq_frame_count = sum(len(track) for track in tracks)

    metrics: Dict[str, object] = {
        "Coverage_GT": float(coverage["Coverage_GT"]),
        "Coverage_Pred": float(coverage["Coverage_Pred"]),
        "Common_Count": int(coverage["Common_Count"]),
        "Track_Count": int(len(tracks)),
        "Seq_Frame_Count": int(seq_frame_count),
    }

    psnr, ssim = compute_psnr_ssim(args.gt, args.pred, common_files, args.paired_size, args.batch_size, args.device)
    metrics["PSNR"] = psnr
    metrics["SSIM"] = ssim
    boundary_results = compute_boundary_metrics(
        args.gt,
        args.pred,
        common_files,
        args.paired_size,
        args.batch_size,
        args.device,
        args.edge_quantile,
        args.edge_tolerance,
    )
    metrics["Boundary_F1"] = boundary_results["Boundary_F1"]
    boundary_details = {key: value for key, value in boundary_results.items() if key != "Boundary_F1"}
    metrics["LPIPS"] = compute_lpips(args.gt, args.pred, common_files, args.paired_size, args.batch_size, args.device)
    metrics.update(compute_pixel_metrics(args.gt, args.pred, common_files, args.paired_size, args.batch_size))

    locator_results = compute_external_locator_metrics(
        pred_dir=args.pred,
        filenames=common_files,
        data_dir=args.data_dir,
        map_name=map_name,
        pose_json=args.pose_json,
        batch_size=args.batch_size,
        device=args.device,
        external_loc_repo_root=args.external_loc_repo_root,
        external_loc_config_path=args.external_loc_config_path,
        external_loc_checkpoint_path=args.external_loc_checkpoint_path,
    )
    locator_details = {
        "locator_pose_json_paths": locator_results.pop("locator_pose_json_paths"),
        "locator_z_min": locator_results.pop("locator_z_min"),
        "locator_z_max": locator_results.pop("locator_z_max"),
    }
    metrics.update(locator_results)

    metrics["FID"] = compute_fid(args.gt, args.pred, common_files, args.batch_size, args.device)
    metrics["IS"] = compute_inception_score(args.pred, common_files, args.batch_size, args.device)
    metrics["CLIP"] = compute_clip_score(args.gt, args.pred, common_files, args.batch_size, args.device)
    metrics["Aesthetic"] = compute_aesthetic_score(args.pred, common_files, args.batch_size, args.device)

    metrics.update(compute_sequence_metrics(args.gt, args.pred, tracks, args.paired_size, args.device))
    metrics.update(compute_temporal_metrics(args.gt, args.pred, tracks, args.paired_size))
    fvd_score, fvd_clip_count = compute_fvd_for_tracks(
        args.gt,
        args.pred,
        tracks,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        size=args.fvd_size,
        batch_size=args.batch_size,
        device=args.device,
    )
    metrics["FVD"] = fvd_score

    print_results(metrics)
    write_results_json(
        json_path,
        experiment_name=experiment_name,
        timestamp=timestamp,
        map_name=map_name,
        gt_dir=args.gt,
        pred_dir=args.pred,
        metrics=metrics,
        coverage=coverage,
        args=args,
        locator_details=locator_details,
        boundary_details=boundary_details,
        track_details=track_details,
        fvd_clip_count=fvd_clip_count,
    )
    print(f"Saved JSON: {json_path}")
    return {
        "json_path": str(json_path),
        "metrics": metrics,
        "map_name": map_name,
        "gt_dir": str(Path(args.gt).resolve()),
        "pred_dir": str(Path(args.pred).resolve()),
        "common_count": int(coverage["Common_Count"]),
        "track_count": int(len(tracks)),
        "seq_frame_count": int(seq_frame_count),
        "fvd_clip_count": int(fvd_clip_count),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSGO simulator continuous image benchmark v1")
    parser.add_argument("--gt", "--gt_dir", dest="gt", type=str, required=True, help="Path to ground-truth image dir.")
    parser.add_argument("--pred", "--pred_dir", dest="pred", type=str, required=True, help="Path to generated image dir.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--paired_size", type=int, default=448)
    parser.add_argument("--edge_quantile", type=float, default=0.85)
    parser.add_argument("--edge_tolerance", type=int, default=2)
    parser.add_argument("--frame_diff_threshold", type=int, default=2)
    parser.add_argument("--min_track_len", type=int, default=4)
    parser.add_argument("--clip_length", type=int, default=16)
    parser.add_argument("--clip_stride", type=int, default=16)
    parser.add_argument("--fvd_size", type=int, default=224)
    parser.add_argument("--data_dir", type=str, default="data/preprocessed_data")
    parser.add_argument("--map_name", type=str, default="auto")
    parser.add_argument("--pose_json", type=str, default="auto")
    parser.add_argument("--external_loc_repo_root", type=str, default="csgosquare")
    parser.add_argument("--external_loc_config_path", type=str, default="configs_reg_newdata/exp5_2.yaml")
    parser.add_argument(
        "--external_loc_checkpoint_path",
        type=str,
        default="checkpoints_reg_newdata/exp5_2/20251227_091745/current_model.pth",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_benchmark_v1_conti(args)


if __name__ == "__main__":
    main()
