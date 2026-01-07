
import os
import cv2
import re
import argparse
import numpy as np
from tqdm import tqdm



def parse_filename(filename):
    """
    解析文件名，返回 (file_num, frame_id)
    支持格式: "file_num12_frame_1005.jpg" 或包含此前缀的路径
    """
    match = re.search(r'file_num(\d+)_frame_(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def frames_to_video(args):
    """
    将文件夹中的序列帧合成为视频。
    逻辑：
    1. 扫描所有图片并按 (file_num, frame_id) 排序。
    2. 自动检测不连续点（文件ID变化或帧号跳变），将其切分为不同的 "Track"。
    3. 每个 "Track" 内部，再根据 max_duration (10s) 切分为多个 "Clip"。
    """

    img_dir = args.img_dir
    output_dir = args.output_dir
    fps = args.fps
    max_frames_per_clip = args.max_duration * fps

    os.makedirs(output_dir, exist_ok=True)

    # 1. 扫描并排序图片
    print(f"🔍 Scanning images in {img_dir}...")
    valid_files = []
    for f in os.listdir(img_dir):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            rid, fid = parse_filename(f)
            if rid is not None:
                valid_files.append({
                    'path': os.path.join(img_dir, f),
                    'file_num': rid,
                    'frame_id': fid,
                    'name': f
                })

    # 核心排序：先按文件号，再按帧号
    valid_files.sort(key=lambda x: (x['file_num'], x['frame_id']))

    if not valid_files:
        print("❌ No valid images found matching 'file_num*_frame_*' pattern.")
        return

    print(f"✅ Found {len(valid_files)} frames. Grouping into tracks...")

    # 2. 分组逻辑 (Group into Tracks)
    # Track 是指一段物理上连续的轨迹（中间没有断帧）
    tracks = []
    current_track = []

    # 定义断帧阈值 (与你生成数据时的逻辑保持一致)
    FRAME_DIFF_THRESHOLD = 2

    for i, item in enumerate(valid_files):
        if not current_track:
            current_track.append(item)
            continue

        last_item = current_track[-1]

        # 判断连续性条件：
        # 1. 同一个原始录制文件 (file_num 相同)
        # 2. 帧号连续 (差值 <= 阈值)
        is_same_file = (item['file_num'] == last_item['file_num'])
        is_continuous = (item['frame_id'] - last_item['frame_id'] <= FRAME_DIFF_THRESHOLD)

        if is_same_file and is_continuous:
            current_track.append(item)
        else:
            # 结束当前 Track，开启新 Track
            if current_track:
                tracks.append(current_track)
            current_track = [item]

    if current_track:
        tracks.append(current_track)

    print(f"📋 Identified {len(tracks)} continuous tracks.")

    # 3. 视频合成逻辑 (Synthesis)
    video_count = 0

    for track_idx, track in enumerate(tracks):
        # 过滤过短的轨迹
        if len(track) < args.min_frames:
            continue

        # 获取图片尺寸
        first_img = cv2.imread(track[0]['path'])
        if first_img is None:
            continue
        height, width, layers = first_img.shape
        size = (width, height)

        # 按 max_frames_per_clip 切分这个 Track
        # range(0, len(track), 100) -> [0, 100, 200...]
        chunks = [track[i:i + max_frames_per_clip] for i in range(0, len(track), max_frames_per_clip)]

        for chunk_idx, chunk in enumerate(chunks):
            # 只有当切片长度足够时才保存 (可选)
            if len(chunk) < args.min_frames // 2:
                continue

            # 构造输出文件名
            # 格式: track_{原始文件号}_{起始帧}_{结束帧}.mp4
            start_info = chunk[0]
            end_info = chunk[-1]
            out_name = f"track_{start_info['file_num']}_f{start_info['frame_id']}_to_f{end_info['frame_id']}.mp4"
            out_path = os.path.join(output_dir, out_name)

            print(f"  🎬 Writing Video {video_count+1}: {out_name} ({len(chunk)} frames)...")

            # 初始化 VideoWriter
            # mp4v 兼容性较好，或者用 'XVID' 生成 .avi
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, size)

            for frame_info in chunk:
                img = cv2.imread(frame_info['path'])
                if img is not None:
                    # 可选：在视频左上角写入帧信息，方便 Debug
                    if args.draw_text:
                        text = f"File {frame_info['file_num']} | Frame {frame_info['frame_id']}"
                        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

                    out.write(img)

            out.release()
            video_count += 1

    print(f"\n🎉 All done! Generated {video_count} videos in '{output_dir}'.")



def build_file_index(directory):
    """
    扫描目录，构建 (file_num, frame_id) -> file_path 的索引字典
    """
    index = {}
    if not os.path.exists(directory):
        return index

    print(f"🔍 Scanning directory: {directory}...")
    for f in os.listdir(directory):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            rid, fid = parse_filename(f)
            if rid is not None:
                index[(rid, fid)] = os.path.join(directory, f)
    return index

def compare_frames_to_video(args):
    img_dir = args.img_dir  # 生成的图片目录
    gt_dir = args.gt_dir    # Ground Truth 图片目录
    output_dir = args.output_dir
    fps = args.fps
    max_frames_per_clip = args.max_duration * fps

    os.makedirs(output_dir, exist_ok=True)

    # 1. 建立 GT 索引 (用于快速查找匹配帧)
    gt_index = build_file_index(gt_dir)
    if not gt_index:
        print(f"⚠️ Warning: No valid images found in GT dir: {gt_dir}. Right side will be black.")

    # 2. 扫描生成图片并匹配 GT
    print(f"🔍 Scanning generated images in {img_dir}...")
    valid_pairs = []

    for f in os.listdir(img_dir):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            rid, fid = parse_filename(f)
            if rid is not None:
                # 尝试找到对应的 GT 路径
                gt_path = gt_index.get((rid, fid), None)

                valid_pairs.append({
                    'gen_path': os.path.join(img_dir, f),
                    'gt_path': gt_path, # 如果没找到则是 None
                    'file_num': rid,
                    'frame_id': fid
                })

    # 排序：先按文件号，再按帧号
    valid_pairs.sort(key=lambda x: (x['file_num'], x['frame_id']))

    if not valid_pairs:
        print("❌ No valid generated images found.")
        return

    print(f"✅ Found {len(valid_pairs)} frames. Grouping into tracks...")

    # 3. 分组逻辑 (Group into Tracks)
    tracks = []
    current_track = []
    FRAME_DIFF_THRESHOLD = 2

    for item in valid_pairs:
        if not current_track:
            current_track.append(item)
            continue

        last_item = current_track[-1]

        is_same_file = (item['file_num'] == last_item['file_num'])
        is_continuous = (item['frame_id'] - last_item['frame_id'] <= FRAME_DIFF_THRESHOLD)

        if is_same_file and is_continuous:
            current_track.append(item)
        else:
            if current_track:
                tracks.append(current_track)
            current_track = [item]

    if current_track:
        tracks.append(current_track)

    print(f"📋 Identified {len(tracks)} continuous tracks.")

    # 4. 视频合成逻辑
    video_count = 0

    for track in tracks:
        if len(track) < args.min_frames:
            continue

        # 获取第一帧尺寸以初始化 VideoWriter
        first_img = cv2.imread(track[0]['gen_path'])
        if first_img is None: continue

        h, w, _ = first_img.shape
        # 输出视频宽度翻倍 (左 Gen + 右 GT)
        # 中间加个 10像素的黑色分割线美观一点
        padding = 10
        size = (w * 2 + padding, h)

        # 切分长片段
        chunks = [track[i:i + max_frames_per_clip] for i in range(0, len(track), max_frames_per_clip)]

        for chunk in chunks:
            if len(chunk) < args.min_frames // 2: continue

            start_info = chunk[0]
            end_info = chunk[-1]
            out_name = f"compare_file{start_info['file_num']}_f{start_info['frame_id']}_to_f{end_info['frame_id']}.mp4"
            out_path = os.path.join(output_dir, out_name)

            print(f"  🎬 Writing Video: {out_name} ({len(chunk)} frames)...")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, size)

            for frame_info in chunk:
                # 读取生成图
                img_gen = cv2.imread(frame_info['gen_path'])
                if img_gen is None: continue

                # 读取 GT 图
                if frame_info['gt_path'] and os.path.exists(frame_info['gt_path']):
                    img_gt = cv2.imread(frame_info['gt_path'])
                    # 容错：如果 GT 尺寸和 Gen 不一致，强制 resize GT
                    if img_gt.shape != img_gen.shape:
                        img_gt = cv2.resize(img_gt, (w, h))
                else:
                    # 如果找不到 GT，用纯黑图填充，并写上 "Missing GT"
                    img_gt = np.zeros_like(img_gen)
                    cv2.putText(img_gt, "Missing GT", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                # 绘制标签 (Prediction vs Ground Truth)
                if args.draw_text:
                    # 左上角信息
                    info_text = f"File {frame_info['file_num']} | Frame {frame_info['frame_id']}"

                    # 左图 (Pred) 标签
                    cv2.putText(img_gen, "Prediction (Ours)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img_gen, info_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    # 右图 (GT) 标签
                    cv2.putText(img_gt, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                # 左右拼接
                # 创建分割线
                separator = np.zeros((h, padding, 3), dtype=np.uint8)
                combined = cv2.hconcat([img_gen, separator, img_gt])

                out.write(combined)

            out.release()
            video_count += 1

    print(f"\n🎉 All done! Generated {video_count} comparison videos in '{output_dir}'.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="生成的图片文件夹 (Prediction)")
    parser.add_argument("--gt_dir", type=str, help="真实图片文件夹 (Ground Truth)")
    parser.add_argument("--output_dir", type=str, default="comparison_videos", help="输出路径")
    parser.add_argument("--fps", type=int, default=10, help="视频帧率")
    parser.add_argument("--max_duration", type=int, default=10, help="最大时长(秒)")
    parser.add_argument("--min_frames", type=int, default=10, help="最小帧数阈值")
    parser.add_argument("--draw_text", action="store_true", default=True, help="是否绘制文字标签")

    args = parser.parse_args()
    if args.gt_dir is not None:
        compare_frames_to_video(args)
    else:
        frames_to_video(args)




# python frames_to_video.py --img_dir my_generated_frames --output_dir my_videos --max_duration 10 --draw_text

# python frames_to_video.py --img_dir outputs_eval/exp2_1/test_20260103_220021/gen_imgs/de_dust2 --output_dir outputs_eval/exp2_1/test_20260103_220021/gen_videos/de_dust2 --max_duration 10 --draw_text

# python frames_to_video.py --img_dir outputs_eval/exp2_1/test_20260103_220021/gen_imgs/de_nuke --output_dir outputs_eval/exp2_1/test_20260103_220021/gen_videos/de_nuke --max_duration 10 --draw_text

# python frames_to_video.py --img_dir outputs_eval/exp2_1/test_20260103_220021/gen_imgs/de_ancient --output_dir outputs_eval/exp2_1/test_20260103_220021/gen_videos/de_ancient --max_duration 10 --draw_text





# python frames_to_video.py --img_dir my_generated_frames --gt_dir path/to/all_original_frames --output_dir video_comparison_results --max_duration 10

# python frames_to_video.py --img_dir outputs_eval/exp2_1/test_20260103_220021/gen_imgs/de_dust2 --gt_dir data/preprocessed_data/de_dust2/imgs --output_dir outputs_eval/exp2_1/test_20260103_220021/gen_compared_videos/de_dust2 --max_duration 10

# python frames_to_video.py --img_dir outputs_eval/exp2_1/test_20260103_220021/gen_imgs/de_nuke --gt_dir data/preprocessed_data/de_nuke/imgs --output_dir outputs_eval/exp2_1/test_20260103_220021/gen_compared_videos/de_nuke --max_duration 10

# python frames_to_video.py --img_dir outputs_eval/exp2_1/test_20260103_220021/gen_imgs/de_ancient --gt_dir data/preprocessed_data/de_ancient/imgs --output_dir outputs_eval/exp2_1/test_20260103_220021/gen_compared_videos/de_ancient --max_duration 10

