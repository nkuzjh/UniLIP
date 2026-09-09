import argparse
import hashlib
import inspect
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from transformers import CLIPModel

from unilip.model.external_loc_model_loader import build_frozen_external_loc_model


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TAU = 2.0 * np.pi

SIMULATOR_OUTPUT_ORDER = [
    "Coverage_GT",
    "Coverage_Pred",
    "Common_Count",
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
]

MAP_PATH_DICT = {
    "de_dust2": "de_dust2_radar_psd.png",
    "de_inferno": "de_inferno_radar_psd.png",
    "de_mirage": "de_mirage_radar_psd.png",
    "de_nuke": "de_nuke_blended_radar_psd.png",
    "de_ancient": "de_ancient_radar_psd.png",
    "de_anubis": "de_anubis_radar_psd.png",
    "de_golden": "de_golden_radar_tga.png",
    "de_overpass": "de_overpass_radar_psd.png",
    "de_palacio": "de_palacio_radar_tga.png",
    "de_train": "de_train_blended_radar_psd.png",
    "de_vertigo": "de_vertigo_blended_radar_psd.png",
    "cs_agency": "cs_agency_radar_tga.png",
    "cs_italy": "cs_italy_radar_psd.png",
    "cs_office": "cs_office_radar_psd.png",
}


def is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in VALID_IMAGE_EXTS


def list_image_files(image_dir: str) -> List[str]:
    return sorted(f for f in os.listdir(image_dir) if is_image_file(f))


def _selection_value(selection: object, key: str, default: object = None) -> object:
    if isinstance(selection, Mapping):
        return selection.get(key, default)
    return getattr(selection, key, default)


def _map_scoped_selection_value(value: object, map_name: str) -> object:
    if isinstance(value, Mapping) and map_name in value:
        return value[map_name]
    return value


def _row_value(row: object, key: str, default: object = None) -> object:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _row_stem(row: object) -> str:
    raw_value = _row_value(row, "file_frame")
    if raw_value is None:
        raw_value = _row_value(row, "filename")
    if raw_value is None:
        raw_value = _row_value(row, "stem")
    if raw_value is None:
        raise ValueError(f"Benchmark v2 row has no file_frame/filename/stem: {row!r}")
    return Path(str(raw_value)).stem


def benchmark_v2_selection_rows(selection: object, map_name: str) -> List[object]:
    rows = _selection_value(selection, "rows")
    rows = _map_scoped_selection_value(rows, map_name)
    if rows is None or isinstance(rows, (str, bytes)):
        raise ValueError(f"Benchmark v2 selection has no row list for map={map_name}")
    try:
        result = list(rows)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"Benchmark v2 selection rows are not iterable for map={map_name}") from exc
    if len({_row_stem(row) for row in result}) != len(result):
        raise ValueError(f"Benchmark v2 selection contains duplicate rows for map={map_name}")
    return result


def benchmark_v2_z_range(selection: object, map_name: str) -> Dict[str, float]:
    z_ranges = _selection_value(selection, "z_ranges")
    z_range = _map_scoped_selection_value(z_ranges, map_name)
    if not isinstance(z_range, Mapping):
        raise ValueError(f"Benchmark v2 selection has no global z range for map={map_name}")
    try:
        z_min = float(z_range["z_min"])
        z_max = float(z_range["z_max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Benchmark v2 z range for map={map_name}: {z_range!r}") from exc
    if not z_max > z_min:
        raise ValueError(f"Benchmark v2 z_max must be greater than z_min for map={map_name}")
    return {"z_min": z_min, "z_max": z_max}


def benchmark_v2_clips(selection: object, map_name: str) -> List[object]:
    clips_by_map = _selection_value(selection, "clips_by_map", {})
    clips = _map_scoped_selection_value(clips_by_map, map_name)
    if isinstance(clips, Mapping) and "clips" in clips:
        clips = clips["clips"]
    if clips is None:
        return []
    if isinstance(clips, (str, bytes)):
        raise ValueError(f"Benchmark v2 clips are not a list for map={map_name}")
    return list(clips)  # type: ignore[arg-type]


def benchmark_v2_image_extension(selection: object) -> str:
    extension = _selection_value(selection, "image_extension", ".jpg")
    extension = str(extension)
    return extension if extension.startswith(".") else f".{extension}"


def benchmark_v2_radar_path(selection: object, map_name: str) -> str:
    radar_paths = _selection_value(selection, "radar_paths", {})
    radar_path = _map_scoped_selection_value(radar_paths, map_name)
    if radar_path is None:
        raise ValueError(f"Benchmark v2 selection has no radar path for map={map_name}")
    path = Path(os.fspath(radar_path)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark v2 radar does not exist for map={map_name}: {path}")
    return str(path)


def benchmark_v2_selection_details(
    selection: object,
    map_name: str,
    rows: Sequence[object],
    expected_stems: Sequence[str],
) -> Dict[str, int]:
    clips = benchmark_v2_clips(selection, map_name)
    clip_frame_count = 0
    for clip in clips:
        if isinstance(clip, Mapping):
            frames = clip.get("frames", clip.get("rows", []))
        else:
            frames = clip
        clip_frame_count += len(list(frames)) if frames is not None else 0
    return {
        "rows": len(rows),
        "expected_stems": len(expected_stems),
        "clips": len(clips),
        "clip_frames": clip_frame_count,
    }


def benchmark_v2_enabled(args: argparse.Namespace) -> bool:
    manifest = getattr(args, "benchmark_v2_manifest", None)
    split = getattr(args, "benchmark_v2_split", None)
    asset_manifest = getattr(args, "benchmark_v2_asset_manifest", None)
    if bool(manifest) != bool(split):
        raise ValueError("--benchmark_v2_manifest and --benchmark_v2_split must be provided together")
    if asset_manifest and not manifest:
        raise ValueError("--benchmark_v2_asset_manifest requires --benchmark_v2_manifest")
    return bool(manifest)


def _asset_provenance_value(source: object, keys: Sequence[str]) -> object:
    """Read an asset provenance field from selection-like objects.

    The core selection currently exposes ``asset_*`` fields, while older
    evaluator artifacts may use the ``benchmark_v2_*`` spelling.  Keeping the
    lookup here makes metric code tolerant of either representation without
    changing the legacy source backend.
    """

    if source is None:
        return None
    nested_values = [
        _selection_value(source, "asset_provenance"),
        _selection_value(source, "benchmark_v2_asset"),
    ]
    for key in keys:
        value = _selection_value(source, key)
        if value not in (None, ""):
            return value
    for nested in nested_values:
        if isinstance(nested, Mapping):
            for key in keys:
                value = nested.get(key)
                if value not in (None, ""):
                    return value
    return None


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, ValueError):
        return None


def _asset_report_selected_hash(path: Path) -> str | None:
    """Best-effort fallback for reports produced before Selection hashes."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    selected = payload.get("selected_images")
    if isinstance(selected, Mapping):
        value = selected.get("sha256")
        return str(value) if value not in (None, "") else None
    return None


def benchmark_v2_asset_provenance(
    selection: object = None,
    args: object = None,
) -> Dict[str, object]:
    """Return the canonical asset provenance emitted by v2 metric artifacts.

    Missing fields mean the historical source backend.  This is intentional:
    exp31--36 artifacts predate the asset backend and must remain readable.
    """

    backend = _asset_provenance_value(
        selection,
        ("asset_backend", "benchmark_v2_asset_backend", "backend"),
    )
    if backend in (None, ""):
        backend = _asset_provenance_value(
            args,
            ("asset_backend", "benchmark_v2_asset_backend", "backend"),
        )
    manifest_value = _asset_provenance_value(
        selection,
        (
            "asset_manifest_path",
            "benchmark_v2_asset_manifest",
            "asset_manifest",
            "manifest_path",
            "path",
        ),
    )
    if manifest_value in (None, ""):
        manifest_value = _asset_provenance_value(
            args,
            (
                "asset_manifest_path",
                "benchmark_v2_asset_manifest",
                "asset_manifest",
                "manifest_path",
                "path",
            ),
        )
    manifest_path = None
    if manifest_value not in (None, ""):
        manifest_path = str(Path(os.fspath(manifest_value)).expanduser().resolve())
    if backend in (None, ""):
        backend = "minimal" if manifest_path else "source"
    backend = str(backend)
    if backend not in {"source", "minimal"}:
        raise ValueError(
            "Benchmark v2 asset backend must be 'source' or 'minimal', "
            f"got {backend!r}"
        )

    manifest_hash = _asset_provenance_value(
        selection,
        (
            "asset_manifest_sha256",
            "benchmark_v2_asset_manifest_sha256",
            "asset_manifest_hash",
            "manifest_sha256",
            "sha256",
        ),
    )
    if manifest_hash in (None, ""):
        manifest_hash = _asset_provenance_value(
            args,
            (
                "asset_manifest_sha256",
                "benchmark_v2_asset_manifest_sha256",
                "asset_manifest_hash",
                "manifest_sha256",
                "sha256",
            ),
        )
    if manifest_hash in (None, "") and manifest_path:
        manifest_hash = _sha256_file(Path(manifest_path))

    selected_hash = _asset_provenance_value(
        selection,
        (
            "selected_images_sha256",
            "benchmark_v2_selected_images_sha256",
            "selected_image_sha256",
            "selected_images_hash",
        ),
    )
    if selected_hash in (None, ""):
        selected_hash = _asset_provenance_value(
            args,
            (
                "selected_images_sha256",
                "benchmark_v2_selected_images_sha256",
                "selected_image_sha256",
            ),
        )
    if selected_hash in (None, "") and manifest_path:
        selected_hash = _asset_report_selected_hash(Path(manifest_path))

    if backend == "source" and any(
        value not in (None, "")
        for value in (manifest_path, manifest_hash, selected_hash)
    ):
        raise ValueError(
            "Benchmark v2 source provenance cannot carry a minimal asset "
            "manifest or checksum identity"
        )
    if backend == "source":
        manifest_path = None
        manifest_hash = None
        selected_hash = None
    canonical = {
        "asset_backend": backend,
        "asset_manifest_path": manifest_path,
        "asset_manifest_sha256": str(manifest_hash) if manifest_hash else None,
        "selected_images_sha256": str(selected_hash) if selected_hash else None,
    }
    # The runner's persisted inference contract uses the benchmark-prefixed
    # names.  Keep the core Selection spellings as well so this evaluator can
    # consume either contract during a rolling upgrade.
    provenance = {
        **canonical,
        "benchmark_v2_asset_manifest": manifest_path,
        "benchmark_v2_asset_backend": backend,
        "benchmark_v2_asset_manifest_sha256": canonical["asset_manifest_sha256"],
        "benchmark_v2_selected_images_sha256": canonical["selected_images_sha256"],
    }
    if backend == "minimal":
        provenance["benchmark_v2_asset"] = {
            "manifest": manifest_path,
            "backend": backend,
            "sha256": canonical["asset_manifest_sha256"],
            "selected_images_sha256": canonical["selected_images_sha256"],
        }
    return provenance


def load_benchmark_v2_selection(args: argparse.Namespace, map_name: str) -> object:
    """Load a map/split selection without importing the v2 loader in legacy mode."""

    if not benchmark_v2_enabled(args):
        return None
    try:
        from csgo_datasets.benchmark_v2 import load_benchmark_v2_selection_from_args
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Benchmark v2 evaluation requires csgo_datasets.benchmark_v2 "
            "with load_benchmark_v2_selection_from_args(...)."
        ) from exc

    loader_kwargs = {
        "map_names": [map_name],
        "support_seed": getattr(args, "benchmark_v2_support_seed", 0),
        "shots_per_map": getattr(args, "benchmark_v2_shots_per_map", 100),
        "data_dir": getattr(args, "data_dir", None),
    }
    asset_manifest = getattr(args, "benchmark_v2_asset_manifest", None)
    if asset_manifest not in (None, ""):
        # ``asset_manifest_path`` is the current core spelling.  The
        # signature check keeps this wrapper usable with a source-only core
        # during a rolling deployment, while failing clearly if minimal mode
        # is requested before the core supports it.
        try:
            parameters = inspect.signature(load_benchmark_v2_selection_from_args).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "asset_manifest_path" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            loader_kwargs["asset_manifest_path"] = asset_manifest
        elif "benchmark_v2_asset_manifest" in parameters:
            loader_kwargs["benchmark_v2_asset_manifest"] = asset_manifest
        else:
            raise ValueError(
                "Benchmark v2 core loader does not support --benchmark_v2_asset_manifest"
            )
    selection = load_benchmark_v2_selection_from_args(
        getattr(args, "benchmark_v2_manifest"),
        getattr(args, "benchmark_v2_split"),
        **loader_kwargs,
    )
    if asset_manifest not in (None, ""):
        backend = _selection_value(selection, "asset_backend", "source")
        if backend != "minimal":
            raise ValueError(
                "Benchmark v2 core loader did not activate the minimal asset backend"
            )
    return selection


def validate_benchmark_v2_coverage(coverage: Mapping[str, object], allow_incomplete: bool) -> None:
    missing = list(coverage.get("missing_pred_files", []))
    extra = list(coverage.get("unmatched_pred_files", []))
    if (missing or extra) and not allow_incomplete:
        raise ValueError(
            "Benchmark v2 evaluation requires exactly one prediction set for the "
            "selected rows; "
            f"missing={len(missing)}, missing_examples={missing[:20]}, "
            f"extra={len(extra)}, extra_examples={extra[:20]}. "
            "Pass --allow_incomplete_benchmark_v2 only for debugging partial outputs."
        )


def load_benchmark_v2_inference_provenance(
    output_root: Path,
    args: argparse.Namespace,
    map_name: str,
    selection: object = None,
) -> Optional[dict]:
    path = output_root / "inference_manifest.json"
    if not path.is_file():
        if getattr(args, "allow_missing_inference_manifest", False):
            return None
        raise FileNotFoundError(
            f"Benchmark v2 inference provenance is required: {path}. "
            "Pass --allow_missing_inference_manifest only for external/debug predictions."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid inference manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Inference manifest must be a JSON object: {path}")
    expected_manifest = Path(args.benchmark_v2_manifest).resolve()
    recorded_manifest = payload.get("benchmark_v2_manifest")
    if not recorded_manifest or Path(recorded_manifest).resolve() != expected_manifest:
        raise ValueError(
            "Inference/evaluation Benchmark v2 manifest mismatch: "
            f"recorded={recorded_manifest!r}, expected={str(expected_manifest)!r}"
        )
    if payload.get("benchmark_v2_split") != args.benchmark_v2_split:
        raise ValueError(
            "Inference/evaluation Benchmark v2 split mismatch: "
            f"recorded={payload.get('benchmark_v2_split')!r}, "
            f"expected={args.benchmark_v2_split!r}"
        )
    maps = payload.get("maps")
    if not isinstance(maps, list) or map_name not in maps:
        raise ValueError(f"Inference manifest does not include map {map_name!r}: {maps!r}")
    if not payload.get("ckpt_path"):
        raise ValueError(f"Inference manifest has no checkpoint provenance: {path}")
    if selection is not None:
        expected_assets = benchmark_v2_asset_provenance(selection, args)
        recorded_assets = benchmark_v2_asset_provenance(payload)
        if expected_assets != recorded_assets:
            raise ValueError(
                "Inference/evaluation Benchmark v2 asset provenance mismatch: "
                f"recorded={recorded_assets!r}, expected={expected_assets!r}"
            )
    return {"path": str(path.resolve()), "payload": payload}


def collect_benchmark_v2_coverage(
    gt_dir: str,
    pred_dir: str,
    rows: Sequence[object],
    image_extension: str = ".jpg",
) -> Dict[str, object]:
    """Compute coverage against manifest stems, never against the whole GT directory."""

    normalized_extension = str(image_extension)
    if not normalized_extension.startswith("."):
        normalized_extension = f".{normalized_extension}"
    expected_stems = [_row_stem(row) for row in rows]
    if len(set(expected_stems)) != len(expected_stems):
        raise ValueError("Benchmark v2 expected rows contain duplicate file stems")

    gt_files = list_image_files(gt_dir)
    pred_files = list_image_files(pred_dir)
    gt_by_stem: Dict[str, List[str]] = {}
    pred_by_stem: Dict[str, List[str]] = {}
    for filename in gt_files:
        gt_by_stem.setdefault(Path(filename).stem, []).append(filename)
    for filename in pred_files:
        pred_by_stem.setdefault(Path(filename).stem, []).append(filename)

    expected_gt_files: List[str] = []
    for stem in expected_stems:
        matches = sorted(gt_by_stem.get(stem, []))
        if len(matches) != 1:
            raise ValueError(
                f"Benchmark v2 expected stem must occur exactly once in GT: "
                f"stem={stem}, matches={matches}"
            )
        manifest_filename = f"{stem}{normalized_extension}"
        expected_gt_files.append(manifest_filename if manifest_filename in matches else matches[0])

    common_files: List[str] = []
    missing_pred_files: List[str] = []
    selected_pred_files = set()
    for stem, gt_filename in zip(expected_stems, expected_gt_files):
        matches = sorted(pred_by_stem.get(stem, []))
        if gt_filename in matches:
            common_files.append(gt_filename)
            selected_pred_files.add(gt_filename)
        else:
            missing_pred_files.append(gt_filename)

    unmatched_pred_files = sorted(set(pred_files) - selected_pred_files)
    expected_count = len(expected_gt_files)
    pred_count = len(pred_files)
    common_count = len(common_files)
    return {
        "gt_files": expected_gt_files,
        "pred_files": pred_files,
        "common_files": common_files,
        "missing_pred_files": missing_pred_files,
        "unmatched_pred_files": unmatched_pred_files,
        "expected_file_stems": expected_stems,
        "gt_filename_by_stem": dict(zip(expected_stems, expected_gt_files)),
        "Coverage_GT": common_count / expected_count if expected_count > 0 else 0.0,
        "Coverage_Pred": common_count / pred_count if pred_count > 0 else 0.0,
        "Common_Count": common_count,
        "GT_Count": expected_count,
        "Pred_Count": pred_count,
    }


def collect_coverage(gt_dir: str, pred_dir: str) -> Dict[str, object]:
    gt_files = set(list_image_files(gt_dir))
    pred_files = set(list_image_files(pred_dir))
    common_files = sorted(gt_files & pred_files)
    missing_pred_files = sorted(gt_files - pred_files)
    unmatched_pred_files = sorted(pred_files - gt_files)
    gt_count = len(gt_files)
    pred_count = len(pred_files)
    common_count = len(common_files)
    return {
        "gt_files": sorted(gt_files),
        "pred_files": sorted(pred_files),
        "common_files": common_files,
        "missing_pred_files": missing_pred_files,
        "unmatched_pred_files": unmatched_pred_files,
        "Coverage_GT": common_count / gt_count if gt_count > 0 else 0.0,
        "Coverage_Pred": common_count / pred_count if pred_count > 0 else 0.0,
        "Common_Count": common_count,
        "GT_Count": gt_count,
        "Pred_Count": pred_count,
    }


class PairedImageDataset(Dataset):
    def __init__(self, gt_dir: str, pred_dir: str, filenames: Sequence[str], size: int, uint8: bool = False):
        if len(filenames) == 0:
            raise ValueError("No common filenames found between GT and Pred directories.")
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.filenames = list(filenames)
        if uint8:
            self.transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
                transforms.PILToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        gt_img = Image.open(os.path.join(self.gt_dir, filename)).convert("RGB")
        pred_img = Image.open(os.path.join(self.pred_dir, filename)).convert("RGB")
        return filename, self.transform(gt_img), self.transform(pred_img)


class SingleImageDataset(Dataset):
    def __init__(self, image_dir: str, filenames: Sequence[str], size: int, uint8: bool = True):
        if len(filenames) == 0:
            raise ValueError(f"No image files selected in {image_dir}.")
        self.image_dir = image_dir
        self.filenames = list(filenames)
        if uint8:
            self.transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
                transforms.PILToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        img = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
        return filename, self.transform(img)


class LocatorImageDataset(Dataset):
    def __init__(
        self,
        pred_dir: str,
        filenames: Sequence[str],
        pose_index: Dict[str, dict],
        map_tensor: torch.Tensor,
        z_min: float,
        z_max: float,
    ):
        self.pred_dir = pred_dir
        self.filenames = list(filenames)
        self.pose_index = pose_index
        self.map_tensor = map_tensor
        self.z_min = z_min
        self.z_max = z_max
        self.fps_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        stem = Path(filename).stem
        pose = self.pose_index[stem]
        pred_img = Image.open(os.path.join(self.pred_dir, filename)).convert("RGB")
        fps_tensor = self.fps_transform(pred_img)

        z_norm = (pose["z"] - self.z_min) / (self.z_max - self.z_min + 1e-6)
        gt_norm = torch.tensor(
            [
                pose["x"] / 1024.0,
                pose["y"] / 1024.0,
                z_norm,
                pose["angle_v"] / TAU,
                pose["angle_h"] / TAU,
            ],
            dtype=torch.float32,
        )
        return filename, fps_tensor, self.map_tensor, gt_norm


def compute_psnr_ssim(gt_dir: str, pred_dir: str, filenames: Sequence[str], size: int, batch_size: int, device: str):
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    dataset = PairedImageDataset(gt_dir, pred_dir, filenames, size=size, uint8=False)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    with torch.no_grad():
        for _, gt_batch, pred_batch in tqdm(loader, desc="PSNR/SSIM"):
            gt_batch = gt_batch.to(device)
            pred_batch = pred_batch.to(device)
            total_psnr += psnr_metric(pred_batch, gt_batch).item() * gt_batch.size(0)
            total_ssim += ssim_metric(pred_batch, gt_batch).item() * gt_batch.size(0)
            count += gt_batch.size(0)
    return total_psnr / count, total_ssim / count


def compute_lpips(gt_dir: str, pred_dir: str, filenames: Sequence[str], size: int, batch_size: int, device: str):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    dataset = PairedImageDataset(gt_dir, pred_dir, filenames, size=size, uint8=False)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    total_lpips = 0.0
    count = 0
    with torch.no_grad():
        for _, gt_batch, pred_batch in tqdm(loader, desc="LPIPS"):
            gt_batch = gt_batch.to(device) * 2.0 - 1.0
            pred_batch = pred_batch.to(device) * 2.0 - 1.0
            score = lpips(pred_batch, gt_batch)
            total_lpips += score.item() * gt_batch.size(0)
            count += gt_batch.size(0)
    return total_lpips / count


def compute_pixel_metrics(gt_dir: str, pred_dir: str, filenames: Sequence[str], size: int, batch_size: int):
    dataset = PairedImageDataset(gt_dir, pred_dir, filenames, size=size, uint8=True)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    total_abs = 0.0
    total_values = 0
    exact_pixels = 0
    within_1_pixels = 0
    total_pixels = 0
    with torch.no_grad():
        for _, gt_batch, pred_batch in tqdm(loader, desc="Pixel metrics"):
            gt_i16 = gt_batch.to(torch.int16)
            pred_i16 = pred_batch.to(torch.int16)
            abs_diff = torch.abs(pred_i16 - gt_i16)
            total_abs += abs_diff.sum().item()
            total_values += abs_diff.numel()
            exact_pixels += (abs_diff == 0).all(dim=1).sum().item()
            within_1_pixels += (abs_diff <= 1).all(dim=1).sum().item()
            total_pixels += abs_diff.shape[0] * abs_diff.shape[2] * abs_diff.shape[3]
    return {
        "Pixel_MAE_255": total_abs / total_values,
        "Pixel_Exact_Acc": exact_pixels / total_pixels if total_pixels > 0 else 0.0,
        "Pixel_Within_1_Acc": within_1_pixels / total_pixels if total_pixels > 0 else 0.0,
    }


def _sobel_edge_map(image_batch: torch.Tensor, edge_quantile: float) -> torch.Tensor:
    rgb_to_gray = image_batch[:, 0:1] * 0.299 + image_batch[:, 1:2] * 0.587 + image_batch[:, 2:3] * 0.114
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image_batch.device,
        dtype=image_batch.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image_batch.device,
        dtype=image_batch.dtype,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(rgb_to_gray, sobel_x, padding=1)
    grad_y = F.conv2d(rgb_to_gray, sobel_y, padding=1)
    edge_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
    flat_edge_mag = edge_mag.flatten(start_dim=1)
    thresholds = torch.quantile(flat_edge_mag, edge_quantile, dim=1).view(-1, 1, 1, 1)
    return edge_mag > thresholds


def compute_boundary_metrics(
    gt_dir: str,
    pred_dir: str,
    filenames: Sequence[str],
    size: int,
    batch_size: int,
    device: str,
    edge_quantile: float,
    edge_tolerance: int,
):
    dataset = PairedImageDataset(gt_dir, pred_dir, filenames, size=size, uint8=False)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    kernel_size = edge_tolerance * 2 + 1
    total_pred_edges = 0.0
    total_gt_edges = 0.0
    total_precision_hits = 0.0
    total_recall_hits = 0.0
    total_pixels = 0
    with torch.no_grad():
        for _, gt_batch, pred_batch in tqdm(loader, desc="Boundary F1"):
            gt_batch = gt_batch.to(device)
            pred_batch = pred_batch.to(device)
            gt_edges = _sobel_edge_map(gt_batch, edge_quantile)
            pred_edges = _sobel_edge_map(pred_batch, edge_quantile)

            gt_edges_float = gt_edges.float()
            pred_edges_float = pred_edges.float()
            gt_edges_dilated = F.max_pool2d(gt_edges_float, kernel_size=kernel_size, stride=1, padding=edge_tolerance) > 0
            pred_edges_dilated = F.max_pool2d(pred_edges_float, kernel_size=kernel_size, stride=1, padding=edge_tolerance) > 0

            total_precision_hits += (pred_edges & gt_edges_dilated).sum().item()
            total_recall_hits += (gt_edges & pred_edges_dilated).sum().item()
            total_pred_edges += pred_edges.sum().item()
            total_gt_edges += gt_edges.sum().item()
            total_pixels += gt_edges.numel()

    precision = total_precision_hits / total_pred_edges if total_pred_edges > 0 else 0.0
    recall = total_recall_hits / total_gt_edges if total_gt_edges > 0 else 0.0
    boundary_f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "Boundary_F1": boundary_f1,
        "Boundary_Precision": precision,
        "Boundary_Recall": recall,
        "Boundary_GT_Edge_Ratio": total_gt_edges / total_pixels if total_pixels > 0 else 0.0,
        "Boundary_Pred_Edge_Ratio": total_pred_edges / total_pixels if total_pixels > 0 else 0.0,
    }


def compute_fid(gt_dir: str, pred_dir: str, filenames: Sequence[str], batch_size: int, device: str):
    fid = FrechetInceptionDistance(feature=2048).to(device)
    dataset_gt = SingleImageDataset(gt_dir, filenames, size=299, uint8=True)
    dataset_pred = SingleImageDataset(pred_dir, filenames, size=299, uint8=True)
    loader_gt = DataLoader(dataset_gt, batch_size=batch_size, num_workers=4)
    loader_pred = DataLoader(dataset_pred, batch_size=batch_size, num_workers=4)
    for _, batch_uint8 in tqdm(loader_gt, desc="FID (GT)"):
        fid.update(batch_uint8.to(device), real=True)
    for _, batch_uint8 in tqdm(loader_pred, desc="FID (Pred)"):
        fid.update(batch_uint8.to(device), real=False)
    return fid.compute().item()


def compute_inception_score(pred_dir: str, filenames: Sequence[str], batch_size: int, device: str):
    inception = InceptionScore().to(device)
    dataset = SingleImageDataset(pred_dir, filenames, size=299, uint8=True)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    with torch.no_grad():
        for _, batch_uint8 in tqdm(loader, desc="IS"):
            inception.update(batch_uint8.to(device))
    score = inception.compute()
    if isinstance(score, tuple):
        return score[0].item()
    return score.item()


def compute_clip_score(gt_dir: str, pred_dir: str, filenames: Sequence[str], batch_size: int, device: str):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    dataset = PairedImageDataset(gt_dir, pred_dir, filenames, size=224, uint8=False)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    total = 0.0
    count = 0
    with torch.no_grad():
        for _, gt_batch, pred_batch in tqdm(loader, desc="CLIP"):
            gt_norm = (gt_batch.to(device) - mean) / std
            pred_norm = (pred_batch.to(device) - mean) / std
            gt_features = model.get_image_features(pixel_values=gt_norm)
            pred_features = model.get_image_features(pixel_values=pred_norm)
            gt_features = gt_features / gt_features.norm(dim=1, keepdim=True)
            pred_features = pred_features / pred_features.norm(dim=1, keepdim=True)
            total += (gt_features * pred_features).sum(dim=1).sum().item()
            count += gt_batch.size(0)
    return total / count


class AestheticPredictor(torch.nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def compute_aesthetic_score(pred_dir: str, filenames: Sequence[str], batch_size: int, device: str):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_model.eval()
    weight_url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
    weight_path = "aesthetic_model.pth"
    if not os.path.exists(weight_path):
        torch.hub.download_url_to_file(weight_url, weight_path)

    predictor = AestheticPredictor(768)
    predictor.load_state_dict(torch.load(weight_path, map_location=device))
    predictor.to(device).eval()

    dataset = SingleImageDataset(pred_dir, filenames, size=224, uint8=False)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    total = 0.0
    count = 0
    with torch.no_grad():
        for _, img_batch in tqdm(loader, desc="Aesthetic"):
            img_norm = (img_batch.to(device) - mean) / std
            features = clip_model.get_image_features(pixel_values=img_norm)
            features = features / features.norm(dim=1, keepdim=True)
            total += predictor(features.float()).sum().item()
            count += img_batch.size(0)
    return total / count


def infer_map_name(gt_dir: str, pred_dir: str, explicit_map_name: Optional[str]) -> str:
    if explicit_map_name and explicit_map_name != "auto":
        return explicit_map_name
    pred_path = Path(pred_dir).resolve()
    if pred_path.name in MAP_PATH_DICT:
        return pred_path.name
    gt_path = Path(gt_dir).resolve()
    if gt_path.name == "imgs" and gt_path.parent.name in MAP_PATH_DICT:
        return gt_path.parent.name
    if gt_path.name in MAP_PATH_DICT:
        return gt_path.name
    raise ValueError("Unable to infer map_name. Please pass --map_name explicitly.")


def resolve_eval_output_root(pred_dir: str, map_name: str):
    pred_path = Path(pred_dir).resolve()
    parts = pred_path.parts
    if "gen_imgs" in parts:
        idx = parts.index("gen_imgs")
        output_root = Path(*parts[:idx])
        experiment_name = parts[idx - 2] if idx >= 2 else None
        timestamp = parts[idx - 1] if idx >= 1 else None
    else:
        output_root = pred_path
        experiment_name = pred_path.parent.name
        timestamp = pred_path.name
    json_path = output_root / f"benchmark_csgo_v1_{map_name}.json"
    return output_root, experiment_name, timestamp, json_path


def load_pose_index(
    data_dir: str,
    map_name: str,
    filenames: Sequence[str],
    pose_json: Optional[str],
    benchmark_v2_rows: Optional[Sequence[object]] = None,
    benchmark_v2_z_range: Optional[Mapping[str, object]] = None,
    benchmark_v2_manifest: Optional[str] = None,
):
    if benchmark_v2_rows is not None:
        pose_entries = list(benchmark_v2_rows)
        pose_index: Dict[str, dict] = {}
        for entry in pose_entries:
            entry_map = _row_value(entry, "map", map_name)
            if entry_map != map_name:
                continue
            stem = _row_stem(entry)
            if stem in pose_index:
                raise ValueError(f"Duplicate Benchmark v2 pose row for stem={stem}")
            pose_index[stem] = dict(entry) if isinstance(entry, Mapping) else vars(entry)

        if benchmark_v2_z_range is None:
            raise ValueError("Benchmark v2 locator evaluation requires manifest z_ranges")
        z_min = float(benchmark_v2_z_range["z_min"])
        z_max = float(benchmark_v2_z_range["z_max"])
        if not z_max > z_min:
            raise ValueError("Benchmark v2 locator z_max must be greater than z_min")
        loaded_paths = [str(Path(benchmark_v2_manifest).resolve())] if benchmark_v2_manifest else []
    else:
        candidate_paths = []
        if pose_json and pose_json != "auto":
            candidate_paths.append(Path(pose_json))
        else:
            split_dir = Path(data_dir) / map_name / "splits_20000_5000"
            candidate_paths.extend([
                split_dir / "test_split.json",
                split_dir / "continuous_unseen_clips.json",
            ])

        pose_entries = []
        loaded_paths = []
        for path in candidate_paths:
            if path.is_file():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pose_entries.extend(data)
                loaded_paths.append(str(path))
        if not pose_entries:
            raise FileNotFoundError(f"No pose json found for map={map_name}: {candidate_paths}")

        pose_index = {entry["file_frame"]: entry for entry in pose_entries if entry.get("map", map_name) == map_name}

    stems = [Path(name).stem for name in filenames]
    missing_pose = sorted(stem for stem in stems if stem not in pose_index)
    if missing_pose:
        raise ValueError(
            f"Pose metadata missing for {len(missing_pose)} common files. "
            f"Examples: {missing_pose[:20]}"
        )

    if benchmark_v2_rows is None:
        matched_entries = [pose_index[stem] for stem in stems]
        z_values = [entry["z"] for entry in matched_entries]
        z_min = min(z_values)
        z_max = max(z_values)

    return pose_index, z_min, z_max, loaded_paths


def build_map_tensor(
    data_dir: str,
    map_name: str,
    radar_path: Optional[str] = None,
) -> torch.Tensor:
    if radar_path is None:
        map_filename = MAP_PATH_DICT.get(map_name)
        if map_filename is None:
            raise ValueError(f"Unknown map_name for locator map image: {map_name}")
        map_path = Path(data_dir) / map_name / map_filename
    else:
        map_path = Path(radar_path)
    if not map_path.is_file():
        raise FileNotFoundError(f"Map image not found: {map_path}")
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(Image.open(map_path).convert("RGB"))


def compute_external_locator_metrics(
    pred_dir: str,
    filenames: Sequence[str],
    data_dir: str,
    map_name: str,
    pose_json: Optional[str],
    batch_size: int,
    device: str,
    external_loc_repo_root: str,
    external_loc_config_path: str,
    external_loc_checkpoint_path: str,
    benchmark_v2_rows: Optional[Sequence[object]] = None,
    benchmark_v2_z_range: Optional[Mapping[str, object]] = None,
    benchmark_v2_manifest: Optional[str] = None,
    benchmark_v2_radar: Optional[str] = None,
):
    pose_index, z_min, z_max, loaded_pose_paths = load_pose_index(
        data_dir,
        map_name,
        filenames,
        pose_json,
        benchmark_v2_rows=benchmark_v2_rows,
        benchmark_v2_z_range=benchmark_v2_z_range,
        benchmark_v2_manifest=benchmark_v2_manifest,
    )
    map_tensor = build_map_tensor(data_dir, map_name, radar_path=benchmark_v2_radar)
    dataset = LocatorImageDataset(pred_dir, filenames, pose_index, map_tensor, z_min, z_max)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    locator = build_frozen_external_loc_model(
        csgosquare_root=external_loc_repo_root,
        config_path=external_loc_config_path,
        checkpoint_path=external_loc_checkpoint_path,
        device=torch.device(device),
    )
    locator.eval()

    pred_norms = []
    gt_norms = []
    with torch.no_grad():
        for _, fps_batch, map_batch, gt_norm in tqdm(loader, desc="External locator"):
            fps_batch = fps_batch.to(device)
            map_batch = map_batch.to(device)
            pred_norm = locator(fps_batch, map_batch).detach().cpu().float()
            pred_norms.append(pred_norm)
            gt_norms.append(gt_norm.float())

    pred_norm = torch.cat(pred_norms, dim=0)
    gt_norm = torch.cat(gt_norms, dim=0)

    norm_diff = torch.abs(pred_norm - gt_norm)
    yaw_delta = torch.remainder(pred_norm[:, 4] - gt_norm[:, 4] + 0.5, 1.0) - 0.5
    norm_diff[:, 4] = torch.abs(yaw_delta)

    xy_dist = torch.norm((pred_norm[:, :2] - gt_norm[:, :2]) * 1024.0, p=2, dim=1).mean().item()
    z_dist = (torch.abs(pred_norm[:, 2] - gt_norm[:, 2]) * (z_max - z_min + 1e-6)).mean().item()
    pitch_dist = (torch.abs(pred_norm[:, 3] - gt_norm[:, 3]) * 360.0).mean().item()
    yaw_dist = (torch.abs(yaw_delta) * 360.0).mean().item()
    norm_l2_5d = torch.norm(norm_diff, p=2, dim=1).mean().item()

    return {
        "Locator_XY_Dist": xy_dist,
        "Locator_Z_Dist": z_dist,
        "Locator_Pitch_Dist": pitch_dist,
        "Locator_Yaw_Dist": yaw_dist,
        "Locator_Norm_L2_5D": norm_l2_5d,
        "locator_pose_json_paths": loaded_pose_paths,
        "locator_z_min": z_min,
        "locator_z_max": z_max,
    }


def ordered_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {key: metrics[key] for key in SIMULATOR_OUTPUT_ORDER}


def print_results(metrics: Dict[str, float]) -> None:
    print("\n" + "=" * 30)
    print("CSGO Simulator Benchmark V1")
    print("=" * 30)
    for key in SIMULATOR_OUTPUT_ORDER:
        value = metrics[key]
        if isinstance(value, int):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f}")
    print("=" * 30)


def write_results_json(
    json_path: Path,
    *,
    experiment_name: Optional[str],
    timestamp: Optional[str],
    map_name: str,
    gt_dir: str,
    pred_dir: str,
    metrics: Dict[str, float],
    coverage: Dict[str, object],
    args: argparse.Namespace,
    locator_details: Dict[str, object],
    boundary_details: Dict[str, object],
    benchmark_v2_selection: Optional[Dict[str, int]] = None,
    inference_provenance: Optional[dict] = None,
    asset_provenance: Optional[Mapping[str, object]] = None,
) -> None:
    payload = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "map_name": map_name,
        "gt_dir": str(Path(gt_dir).resolve()),
        "pred_dir": str(Path(pred_dir).resolve()),
        "metrics_order": SIMULATOR_OUTPUT_ORDER,
        "metrics_ordered": ordered_metrics(metrics),
        "unmatched_pred_files": coverage["unmatched_pred_files"],
        "unmatched_pred_files_count": len(coverage["unmatched_pred_files"]),
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
            "locator_pose_json_paths": locator_details.get("locator_pose_json_paths", []),
            "locator_z_min": locator_details.get("locator_z_min"),
            "locator_z_max": locator_details.get("locator_z_max"),
            "edge_quantile": args.edge_quantile,
            "edge_tolerance": args.edge_tolerance,
        },
        "details": {
            "boundary": boundary_details,
        },
    }
    if benchmark_v2_selection is not None:
        manifest = getattr(args, "benchmark_v2_manifest", None)
        normalized_assets = dict(asset_provenance or benchmark_v2_asset_provenance(args=args))
        payload.update({
            "benchmark_v2_manifest": str(Path(manifest).resolve()) if manifest else None,
            "benchmark_v2_split": getattr(args, "benchmark_v2_split", None),
            "benchmark_v2_selection_counts": benchmark_v2_selection,
            "missing_pred_files": coverage["missing_pred_files"],
            "inference_provenance": inference_provenance,
            **normalized_assets,
        })
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_benchmark_v1(args: argparse.Namespace) -> Dict[str, object]:
    v2_mode = benchmark_v2_enabled(args)
    if v2_mode:
        map_name = infer_map_name(args.gt, args.pred, args.map_name)
        selection = load_benchmark_v2_selection(args, map_name)
        if selection is None:
            raise ValueError("Benchmark v2 loader returned no selection")
        selection_rows = benchmark_v2_selection_rows(selection, map_name)
        selection_z_range = benchmark_v2_z_range(selection, map_name)
        selection_radar = benchmark_v2_radar_path(selection, map_name)
        coverage = collect_benchmark_v2_coverage(
            args.gt,
            args.pred,
            selection_rows,
            benchmark_v2_image_extension(selection),
        )
        selection_details = benchmark_v2_selection_details(
            selection,
            map_name,
            selection_rows,
            coverage["expected_file_stems"],
        )
    else:
        coverage = collect_coverage(args.gt, args.pred)
        map_name = infer_map_name(args.gt, args.pred, args.map_name)
        selection = None
        selection_rows = None
        selection_z_range = None
        selection_radar = None
        selection_details = None
    if selection is not None:
        validate_benchmark_v2_coverage(
            coverage,
            allow_incomplete=bool(getattr(args, "allow_incomplete_benchmark_v2", False)),
        )
    common_files = coverage["common_files"]
    if len(common_files) == 0:
        raise ValueError("No common filenames found between GT and Pred directories.")

    output_root, experiment_name, timestamp, json_path = resolve_eval_output_root(args.pred, map_name)
    if selection is not None:
        json_path = output_root / f"benchmark_csgo_v2_{map_name}.json"
        inference_provenance = load_benchmark_v2_inference_provenance(
            output_root, args, map_name, selection=selection
        )
        asset_provenance = benchmark_v2_asset_provenance(selection, args)
    else:
        inference_provenance = None
        asset_provenance = None
    print(f"Experiment: {experiment_name}")
    print(f"Timestamp: {timestamp}")
    print(f"Map: {map_name}")
    print(f"Output JSON: {json_path}")

    metrics = {
        "Coverage_GT": float(coverage["Coverage_GT"]),
        "Coverage_Pred": float(coverage["Coverage_Pred"]),
        "Common_Count": int(coverage["Common_Count"]),
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
    boundary_details = {
        key: value
        for key, value in boundary_results.items()
        if key != "Boundary_F1"
    }
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
        benchmark_v2_rows=selection_rows,
        benchmark_v2_z_range=selection_z_range,
        benchmark_v2_manifest=getattr(args, "benchmark_v2_manifest", None),
        benchmark_v2_radar=selection_radar,
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
        benchmark_v2_selection=selection_details,
        inference_provenance=inference_provenance,
        asset_provenance=asset_provenance,
    )
    print(f"Saved JSON: {json_path}")
    result = {
        "json_path": str(json_path),
        "metrics": metrics,
        "map_name": map_name,
        "gt_dir": str(Path(args.gt).resolve()),
        "pred_dir": str(Path(args.pred).resolve()),
        "common_count": int(coverage["Common_Count"]),
    }
    if selection_details is not None:
        result["benchmark_v2_selection_counts"] = selection_details
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSGO simulator image benchmark v1")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground-truth image directory.")
    parser.add_argument("--pred", type=str, required=True, help="Path to generated image directory.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--paired_size", type=int, default=448)
    parser.add_argument("--edge_quantile", type=float, default=0.85)
    parser.add_argument("--edge_tolerance", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default="data/preprocessed_data")
    parser.add_argument("--map_name", type=str, default="auto")
    parser.add_argument("--pose_json", type=str, default="auto")
    parser.add_argument(
        "--benchmark_v2_manifest",
        type=str,
        default=None,
        help="Optional Benchmark v2 manifest path; enables manifest-driven evaluation.",
    )
    parser.add_argument(
        "--benchmark_v2_split",
        type=str,
        default=None,
        help="Benchmark v2 split selector, used together with --benchmark_v2_manifest.",
    )
    parser.add_argument(
        "--benchmark_v2_asset_manifest",
        type=str,
        default=None,
        help=(
            "Optional verified minimal Benchmark v2 asset manifest. When set, "
            "the selected images/radars are loaded from that bundle."
        ),
    )
    parser.add_argument(
        "--allow_incomplete_benchmark_v2",
        action="store_true",
        help="Debug only: compute v2 metrics on available predictions when coverage is incomplete.",
    )
    parser.add_argument(
        "--allow_missing_inference_manifest",
        action="store_true",
        help="Debug/external predictions only: allow v2 metrics without inference_manifest.json.",
    )
    parser.add_argument("--external_loc_repo_root", type=str, default="csgosquare")
    parser.add_argument("--external_loc_config_path", type=str, default="configs_reg_newdata/exp5_2.yaml")
    parser.add_argument(
        "--external_loc_checkpoint_path",
        type=str,
        default="checkpoints_reg_newdata/exp5_2/20251227_091745/current_model.pth",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_benchmark_v1(args)


if __name__ == "__main__":
    main()
