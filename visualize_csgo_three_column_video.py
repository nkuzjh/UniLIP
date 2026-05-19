from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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


@dataclass(frozen=True)
class FrameRecord:
    filename: str
    file_num: int
    frame_id: int


def is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in VALID_IMAGE_EXTS


def list_image_files(image_dir: str) -> List[str]:
    return sorted(f for f in os.listdir(image_dir) if is_image_file(f))


def collect_coverage(gt_dir: str, pred_dir: str) -> Dict[str, object]:
    gt_files = set(list_image_files(gt_dir))
    pred_files = set(list_image_files(pred_dir))
    common_files = sorted(gt_files & pred_files)
    return {
        "gt_files": sorted(gt_files),
        "pred_files": sorted(pred_files),
        "common_files": common_files,
        "missing_pred_files": sorted(gt_files - pred_files),
        "unmatched_pred_files": sorted(pred_files - gt_files),
        "GT_Count": len(gt_files),
        "Pred_Count": len(pred_files),
        "Common_Count": len(common_files),
    }


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


def parse_conti_filename(filename: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"file_num(\d+)_frame_(\d+)", Path(filename).name)
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
    return tracks, {
        "parsed_frame_count": len(parsed_records),
        "unparsed_common_files": unparsed_files,
        "unparsed_common_files_count": len(unparsed_files),
        "raw_track_count": len(raw_tracks),
        "dropped_short_track_count": len(raw_tracks) - len(tracks),
        "frame_diff_threshold": frame_diff_threshold,
        "track_summaries": track_summaries,
    }


def infer_output_root(pred_dir: str) -> Path:
    pred_path = Path(pred_dir).resolve()
    parts = pred_path.parts
    if "gen_imgs" in parts:
        gen_idx = parts.index("gen_imgs")
        return Path(*parts[:gen_idx])
    return pred_path


def build_video_name(map_name: str, chunk: Sequence[FrameRecord]) -> str:
    start = chunk[0]
    end = chunk[-1]
    return f"three_column_video_{map_name}_file{start.file_num}_f{start.frame_id}_to_f{end.frame_id}.mp4"


def resolve_video_output_path(
    output_arg: Optional[str],
    pred_dir: str,
    map_name: str,
    chunk: Sequence[FrameRecord],
    multiple_outputs: bool,
) -> Path:
    video_name = build_video_name(map_name, chunk)
    if not output_arg:
        return infer_output_root(pred_dir) / video_name

    output_path = Path(output_arg)
    if output_path.suffix.lower() != ".mp4":
        return output_path / video_name
    if multiple_outputs:
        return output_path.with_name(f"{output_path.stem}_{Path(video_name).stem}{output_path.suffix}")
    return output_path


def resolve_map_path(data_dir: str, map_name: str, map_image: Optional[str]) -> Path:
    if map_image:
        map_path = Path(map_image)
        if not map_path.is_file():
            raise FileNotFoundError(f"Map image not found: {map_path}")
        return map_path

    map_filename = MAP_PATH_DICT.get(map_name)
    if map_filename is None:
        raise ValueError(f"Unknown map_name={map_name}. Please pass --map_image explicitly.")

    map_path = Path(data_dir) / map_name / map_filename
    if not map_path.is_file():
        raise FileNotFoundError(f"Map image not found: {map_path}")
    return map_path


def load_pose_index(
    data_dir: str,
    map_name: str,
    filenames: Sequence[str],
    pose_json: Optional[str],
) -> Tuple[Dict[str, dict], List[str]]:
    candidate_paths = []
    if pose_json and pose_json != "auto":
        candidate_paths.append(Path(pose_json))
    else:
        split_dir = Path(data_dir) / map_name / "splits_20000_5000"
        candidate_paths.extend([
            split_dir / "test_split.json",
            split_dir / "continuous_unseen_clips.json",
        ])

    pose_index: Dict[str, dict] = {}
    loaded_paths = []
    for path in candidate_paths:
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        loaded_paths.append(str(path))
        for entry in entries:
            if entry.get("map", map_name) != map_name:
                continue
            file_frame = entry.get("file_frame")
            if file_frame:
                pose_index[file_frame] = entry

    if not pose_index:
        raise FileNotFoundError(f"No pose metadata found. Tried: {candidate_paths}")

    stems = [Path(filename).stem for filename in filenames]
    missing_pose = sorted(stem for stem in stems if stem not in pose_index)
    if missing_pose:
        raise ValueError(
            f"Pose metadata missing for {len(missing_pose)} selected frames. "
            f"Examples: {missing_pose[:20]}"
        )

    return pose_index, loaded_paths


def build_video_chunks(
    tracks: Sequence[Sequence[FrameRecord]],
    selected_track_indices: Sequence[int],
    max_frames_per_clip: int,
    min_frames: int,
) -> List[Tuple[int, int, List[FrameRecord]]]:
    if max_frames_per_clip <= 0:
        raise ValueError("--max_duration * --fps must be greater than 0.")

    chunks: List[Tuple[int, int, List[FrameRecord]]] = []
    min_chunk_frames = max(1, min_frames // 2)
    for track_index in selected_track_indices:
        track = list(tracks[track_index])
        if len(track) < min_frames:
            continue

        track_chunks = [
            track[start_idx:start_idx + max_frames_per_clip]
            for start_idx in range(0, len(track), max_frames_per_clip)
        ]
        for chunk_index, chunk in enumerate(track_chunks):
            if len(chunk) < min_chunk_frames:
                continue
            chunks.append((track_index, chunk_index, chunk))

    return chunks


def get_resample_filter():
    image_module, _ = require_pil()
    if hasattr(image_module, "Resampling"):
        return image_module.Resampling.BICUBIC
    return image_module.BICUBIC


def require_cv2():
    try:
        import cv2  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "visualize_csgo_three_column_video.py requires opencv-python to write mp4 videos. "
            "Install it with `pip install opencv-python` or use the repository requirements."
        ) from exc
    return cv2


def require_numpy():
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "visualize_csgo_three_column_video.py requires numpy to convert PIL frames for video writing. "
            "Install it with `pip install numpy` or use the repository requirements."
        ) from exc
    return np


def require_pil():
    if Image is None or ImageDraw is None:
        raise ImportError(
            "visualize_csgo_three_column_video.py requires Pillow to render map/FPS panels. "
            "Install it with `pip install Pillow` or use the repository requirements."
        )
    return Image, ImageDraw


def fit_to_square(image: Image.Image, size: int, stretch: bool) -> Image.Image:
    image_module, _ = require_pil()
    image = image.convert("RGB")
    if stretch:
        return image.resize((size, size), get_resample_filter())

    scale = min(size / image.width, size / image.height)
    new_width = max(1, int(round(image.width * scale)))
    new_height = max(1, int(round(image.height * scale)))
    resized = image.resize((new_width, new_height), get_resample_filter())
    canvas = image_module.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - new_width) // 2, (size - new_height) // 2))
    return canvas


def load_image_panel(path: Path, panel_size: int, stretch: bool) -> Image.Image:
    image_module, _ = require_pil()
    image = image_module.open(path).convert("RGB")
    return fit_to_square(image, panel_size, stretch=stretch)


def center_crop_or_resize_to_size(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    target_width, target_height = target_size
    image = image.convert("RGB")
    if image.width < target_width or image.height < target_height:
        scale = max(target_width / image.width, target_height / image.height)
        resized_width = max(target_width, int(round(image.width * scale)))
        resized_height = max(target_height, int(round(image.height * scale)))
        image = image.resize((resized_width, resized_height), get_resample_filter())

    left = max(0, (image.width - target_width) // 2)
    top = max(0, (image.height - target_height) // 2)
    return image.crop((left, top, left + target_width, top + target_height))


def load_matched_fps_panels(
    pred_path: Path,
    gt_path: Path,
    panel_size: int,
    stretch: bool,
    crop_pred_to_gt: bool,
) -> Tuple[Image.Image, Image.Image]:
    image_module, _ = require_pil()
    gt_image = image_module.open(gt_path).convert("RGB")
    pred_image = image_module.open(pred_path).convert("RGB")
    if crop_pred_to_gt:
        pred_image = center_crop_or_resize_to_size(pred_image, gt_image.size)
    pred_panel = fit_to_square(pred_image, panel_size, stretch=stretch)
    gt_panel = fit_to_square(gt_image, panel_size, stretch=stretch)
    return pred_panel, gt_panel


def draw_panel_text(image: Image.Image, lines: Sequence[str], fill: Tuple[int, int, int]) -> Image.Image:
    cv2 = require_cv2()
    np = require_numpy()
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    bgr_color = (int(fill[2]), int(fill[1]), int(fill[0]))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    x_pos = 10
    y_pos = 28
    line_gap = 24
    for line_idx, line in enumerate(lines):
        y_text = y_pos + line_idx * line_gap
        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(
            bgr,
            (x_pos - 4, y_text - text_height - 5),
            (x_pos + text_width + 4, y_text + baseline + 4),
            (0, 0, 0),
            -1,
        )
        cv2.putText(bgr, line, (x_pos, y_text), font, font_scale, bgr_color, thickness, cv2.LINE_AA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pose_to_map_xy(pose: dict, map_width: int, map_height: int) -> Tuple[float, float]:
    return float(pose["x"]) * map_width / 1024.0, float(pose["y"]) * map_height / 1024.0


def draw_gt_trajectory(
    base_map: Image.Image,
    points: Sequence[Tuple[float, float]],
    current_idx: int,
) -> Image.Image:
    image_module, image_draw_module = require_pil()
    map_rgba = base_map.convert("RGBA")
    overlay = image_module.new("RGBA", map_rgba.size, (0, 0, 0, 0))
    draw = image_draw_module.Draw(overlay)

    visible_points = list(points[: current_idx + 1])
    if not visible_points:
        return map_rgba.convert("RGB")

    line_width = max(2, round(min(map_rgba.size) * 0.005))
    history_radius = max(2, round(min(map_rgba.size) * 0.004))
    current_radius = max(5, round(min(map_rgba.size) * 0.009))

    if len(visible_points) >= 2:
        draw.line(visible_points, fill=(0, 255, 80, 220), width=line_width, joint="curve")

    for point in visible_points[:-1]:
        x_coord, y_coord = point
        draw.ellipse(
            (
                x_coord - history_radius,
                y_coord - history_radius,
                x_coord + history_radius,
                y_coord + history_radius,
            ),
            fill=(0, 255, 80, 170),
            outline=(0, 0, 0, 180),
            width=1,
        )

    x_coord, y_coord = visible_points[-1]
    draw.ellipse(
        (
            x_coord - current_radius,
            y_coord - current_radius,
            x_coord + current_radius,
            y_coord + current_radius,
        ),
        fill=(255, 48, 48, 235),
        outline=(255, 255, 255, 240),
        width=max(2, line_width // 2),
    )

    return image_module.alpha_composite(map_rgba, overlay).convert("RGB")


def make_frame(
    *,
    base_map: Image.Image,
    gt_points: Sequence[Tuple[float, float]],
    current_idx: int,
    pred_path: Path,
    gt_path: Path,
    record: FrameRecord,
    panel_size: int,
    stretch_fps: bool,
    crop_pred_to_gt: bool,
    draw_text: bool,
) -> Image.Image:
    image_module, _ = require_pil()
    map_with_track = draw_gt_trajectory(base_map, gt_points, current_idx)
    map_panel = fit_to_square(map_with_track, panel_size, stretch=True)
    pred_panel, gt_panel = load_matched_fps_panels(
        pred_path,
        gt_path,
        panel_size=panel_size,
        stretch=stretch_fps,
        crop_pred_to_gt=crop_pred_to_gt,
    )
    if draw_text:
        map_panel = draw_panel_text(
            map_panel,
            [
                "GT Trajectory",
                f"File {record.file_num} | Frame {record.frame_id}",
                f"Step {current_idx + 1}/{len(gt_points)}",
            ],
            fill=(0, 255, 80),
        )
        pred_panel = draw_panel_text(
            pred_panel,
            ["Prediction (Ours)", f"File {record.file_num} | Frame {record.frame_id}"],
            fill=(0, 255, 0),
        )
        gt_panel = draw_panel_text(gt_panel, ["Ground Truth"], fill=(0, 165, 255))
    combined = image_module.new("RGB", (panel_size * 3, panel_size), (0, 0, 0))
    combined.paste(map_panel, (0, 0))
    combined.paste(pred_panel, (panel_size, 0))
    combined.paste(gt_panel, (panel_size * 2, 0))
    return combined


def write_summary_json(
    output_path: Path,
    *,
    map_name: str,
    map_path: Path,
    gt_dir: str,
    pred_dir: str,
    pose_json_paths: Sequence[str],
    track_index: int,
    chunk_index: int,
    track: Sequence[FrameRecord],
    written_frames: int,
    skipped_frames: Sequence[str],
    args: argparse.Namespace,
) -> None:
    payload = {
        "map_name": map_name,
        "map_path": str(map_path.resolve()),
        "gt_dir": str(Path(gt_dir).resolve()),
        "pred_dir": str(Path(pred_dir).resolve()),
        "pose_json_paths": list(pose_json_paths),
        "output_video": str(output_path.resolve()),
        "track_index": track_index,
        "chunk_index": chunk_index,
        "track_file_num": track[0].file_num if track else None,
        "track_length": len(track),
        "written_frames": written_frames,
        "skipped_frames": list(skipped_frames),
        "first_filename": track[0].filename if track else None,
        "last_filename": track[-1].filename if track else None,
        "config": {
            "fps": args.fps,
            "max_duration": args.max_duration,
            "min_frames": args.min_frames,
            "panel_size": args.panel_size,
            "frame_diff_threshold": args.frame_diff_threshold,
            "min_track_len": args.min_track_len,
            "crop_pred_to_gt": args.crop_pred_to_gt,
            "draw_text": args.draw_text,
            "stretch_fps": args.stretch_fps,
            "skip_missing": args.skip_missing,
        },
    }
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_track_video(
    *,
    track: Sequence[FrameRecord],
    track_index: int,
    chunk_index: int,
    pose_index: Dict[str, dict],
    pose_json_paths: Sequence[str],
    gt_dir: str,
    pred_dir: str,
    base_map: Image.Image,
    map_name: str,
    map_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2 = require_cv2()
    np = require_numpy()

    map_width, map_height = base_map.size
    gt_points = [
        pose_to_map_xy(pose_index[Path(record.filename).stem], map_width, map_height)
        for record in track
    ]

    frame_size = (args.panel_size * 3, args.panel_size)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open cv2.VideoWriter for {output_path}")

    written_frames = 0
    skipped_frames = []
    try:
        desc = f"Track {track_index} chunk {chunk_index}"
        for frame_idx, record in enumerate(tqdm(track, desc=desc, leave=False)):
            pred_path = Path(pred_dir) / record.filename
            gt_path = Path(gt_dir) / record.filename
            if not pred_path.is_file() or not gt_path.is_file():
                if args.skip_missing:
                    skipped_frames.append(record.filename)
                    continue
                missing = pred_path if not pred_path.is_file() else gt_path
                raise FileNotFoundError(f"Frame image not found: {missing}")

            frame_rgb = make_frame(
                base_map=base_map,
                gt_points=gt_points,
                current_idx=frame_idx,
                pred_path=pred_path,
                gt_path=gt_path,
                record=record,
                panel_size=args.panel_size,
                stretch_fps=args.stretch_fps,
                crop_pred_to_gt=args.crop_pred_to_gt,
                draw_text=args.draw_text,
            )
            frame = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2BGR)
            writer.write(frame)
            written_frames += 1
    finally:
        writer.release()

    if written_frames == 0:
        raise RuntimeError(f"No frames were written for track {track_index}.")

    write_summary_json(
        output_path,
        map_name=map_name,
        map_path=map_path,
        gt_dir=gt_dir,
        pred_dir=pred_dir,
        pose_json_paths=pose_json_paths,
        track_index=track_index,
        chunk_index=chunk_index,
        track=track,
        written_frames=written_frames,
        skipped_frames=skipped_frames,
        args=args,
    )
    print(f"Saved video: {output_path} ({written_frames} frames)")


def run(args: argparse.Namespace) -> Dict[str, object]:
    image_module, _ = require_pil()
    coverage = collect_coverage(args.gt, args.pred)
    common_files = coverage["common_files"]
    if len(common_files) == 0:
        raise ValueError("No common filenames found between GT and Pred directories.")

    map_name = infer_map_name(args.gt, args.pred, args.map_name)
    map_path = resolve_map_path(args.data_dir, map_name, args.map_image)
    base_map = image_module.open(map_path).convert("RGB")

    tracks, track_details = build_tracks(
        common_files,
        frame_diff_threshold=args.frame_diff_threshold,
        min_track_len=args.min_track_len,
    )
    if not tracks:
        raise ValueError(
            "No valid continuous tracks found. "
            f"Parsed frames={track_details['parsed_frame_count']}, "
            f"unparsed common files={track_details['unparsed_common_files_count']}."
        )

    if args.all_tracks or args.track_index is None or args.track_index < 0:
        selected_track_indices = list(range(len(tracks)))
    else:
        selected_track_indices = [args.track_index]

    for track_index in selected_track_indices:
        if track_index >= len(tracks):
            raise IndexError(f"--track_index {track_index} out of range. Available: 0..{len(tracks) - 1}")

    max_frames_per_clip = args.max_duration * args.fps
    video_chunks = build_video_chunks(
        tracks,
        selected_track_indices,
        max_frames_per_clip=max_frames_per_clip,
        min_frames=args.min_frames,
    )
    if not video_chunks:
        raise ValueError(
            "No video chunks selected. Lower --min_frames or increase available paired frames."
        )

    selected_filenames = [
        record.filename
        for _, _, chunk in video_chunks
        for record in chunk
    ]
    pose_index, pose_json_paths = load_pose_index(args.data_dir, map_name, selected_filenames, args.pose_json)

    print(f"Map: {map_name}")
    print(f"Map image: {map_path}")
    print(f"Common frames: {coverage['Common_Count']}")
    print(f"Tracks: {len(tracks)}")
    print(
        "All tracks: "
        + ", ".join(
            f"#{track_idx}: {len(track)} frames ({track[0].filename} -> {track[-1].filename})"
            for track_idx, track in enumerate(tracks)
        )
    )
    print(f"Selected tracks for video output: {selected_track_indices}")
    print(
        f"Video chunks: {len(video_chunks)} "
        f"(max_duration={args.max_duration}s, fps={args.fps}, "
        f"max_frames_per_clip={max_frames_per_clip}, min_frames={args.min_frames})"
    )
    print(f"Pose JSON: {pose_json_paths}")

    output_paths = []
    multiple_outputs = len(video_chunks) > 1
    for track_index, chunk_index, chunk in video_chunks:
        output_path = resolve_video_output_path(
            args.output,
            args.pred,
            map_name,
            chunk,
            multiple_outputs=multiple_outputs,
        )
        write_track_video(
            track=chunk,
            track_index=track_index,
            chunk_index=chunk_index,
            pose_index=pose_index,
            pose_json_paths=pose_json_paths,
            gt_dir=args.gt,
            pred_dir=args.pred,
            base_map=base_map,
            map_name=map_name,
            map_path=map_path,
            output_path=output_path,
            args=args,
        )
        output_paths.append(str(output_path))

    return {
        "map_name": map_name,
        "track_count": len(tracks),
        "video_count": len(video_chunks),
        "selected_track_indices": selected_track_indices,
        "output_paths": output_paths,
        "track_details": track_details,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a three-column CSGO video: "
            "GT trajectory on radar map | predicted FPS frames | GT FPS frames."
        )
    )
    parser.add_argument("--gt", "--gt_dir", dest="gt", type=str, required=True, help="Ground-truth FPS image dir.")
    parser.add_argument("--pred", "--pred_dir", dest="pred", type=str, required=True, help="Predicted FPS image dir.")
    parser.add_argument("--data_dir", type=str, default="data/preprocessed_data", help="CSGO preprocessed data root.")
    parser.add_argument("--map_name", type=str, default="auto", help="Map name, or auto.")
    parser.add_argument("--map_image", type=str, default=None, help="Optional explicit radar map image path.")
    parser.add_argument("--pose_json", type=str, default="auto", help="Pose metadata json, or auto.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory, or mp4 name pattern base. Multi-video output appends source file/frame ranges.",
    )
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS.")
    parser.add_argument("--max_duration", type=int, default=10, help="Maximum duration in seconds for each output video.")
    parser.add_argument("--min_frames", type=int, default=10, help="Minimum frame threshold, matching frames_to_video.py.")
    parser.add_argument("--panel_size", type=int, default=448, help="Square size for each of the three panels.")
    parser.add_argument(
        "--track_index",
        type=int,
        default=None,
        help="Render only this continuous track index. Omit or pass a negative value to render all tracks.",
    )
    parser.add_argument("--all_tracks", action="store_true", help="Render every valid track. This is now the default.")
    parser.add_argument(
        "--frame_diff_threshold",
        type=int,
        default=2,
        help=(
            "Max frame id gap inside a track. Defaults to 2, matching frames_to_video.py."
        ),
    )
    parser.add_argument(
        "--min_track_len",
        type=int,
        default=1,
        help="Internal pre-filter before chunking. Leave at 1 to match frames_to_video.py.",
    )
    parser.add_argument("--codec", type=str, default="mp4v", help="OpenCV fourcc codec, e.g. mp4v or avc1.")
    parser.add_argument(
        "--crop_pred_to_gt",
        action="store_true",
        default=True,
        help="Center-crop prediction images to the original GT image size before panel resizing.",
    )
    parser.add_argument(
        "--no_crop_pred_to_gt",
        dest="crop_pred_to_gt",
        action="store_false",
        help="Disable prediction center-crop to GT size.",
    )
    parser.add_argument("--draw_text", action="store_true", default=True, help="Draw panel labels and current frame info.")
    parser.add_argument("--no_draw_text", dest="draw_text", action="store_false", help="Disable text overlays.")
    parser.add_argument("--stretch_fps", action="store_true", help="Stretch FPS images instead of letterboxing.")
    parser.add_argument("--skip_missing", action="store_true", help="Skip missing images instead of raising.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
