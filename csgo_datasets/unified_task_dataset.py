import os
import json
import random
import logging
import numpy as np
from PIL import Image
import copy
from typing import Dict, Sequence, Union, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms.functional import convert_image_dtype
from torchvision.transforms import InterpolationMode
import transformers

from unilip import conversation as conversation_lib
from unilip.mm_utils import tokenizer_image_token
from unilip.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_IDX

from csgo_datasets.utils import CoarseDropout, GridDropout
from csgo_datasets.random_erasing import RandomErasing



IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'



# 任务 A: 定位 (Map + FPS -> Pose)
# ==========================================
# A.1 提示词模板 (Prompt Templates)
# ==========================================
def get_loc_prompt(map_name):
    LOC_PROMPT_TEMPLATE = (
        f"Task: The following visual data of CS2 map '{map_name}' has been fused and inserted into this sequence:\n"
        "1. First-Person View (FPV) Features.\n"
        "2. Overhead Radar Map (RADAR) Features.\n"
        "Analyze the spatial relationship between the FPV and the RADAR Map to determine the precise camera pose. "
        "Predict the 5D pose (x, y, z, pitch, yaw) in the required format.\n"
        "<image>\n<image>"
    )
    return LOC_PROMPT_TEMPLATE



# 任务 B: 生成 (Map + Pose -> FPS)
# ==========================================
# B.1 辅助函数: 图像 Padding (复用 UniLIP 逻辑)
# ==========================================
def expand2square(pil_img, background_color):
    """将图片填充为正方形，背景色通常为 image_mean"""

    # GPT优化，增加直接在tensor上操作的分支，避免pil和tensor之间的重复转换
    if isinstance(pil_img, torch.Tensor):
        width = pil_img.shape[-1]
        height = pil_img.shape[-2]
        if width == height:
            return pil_img
        if width > height:
            pad_top = (width - height) // 2
            pad_bottom = width - height - pad_top
            pad = (0, 0, pad_top, pad_bottom)
        else:
            pad_left = (height - width) // 2
            pad_right = height - width - pad_left
            pad = (pad_left, pad_right, 0, 0)
        fill = torch.as_tensor(background_color, dtype=pil_img.dtype, device=pil_img.device).view(-1, 1, 1)
        padded = F.pad(pil_img, pad, value=0.0)
        if width > height:
            padded[:, :pad_top, :] = fill
            padded[:, padded.shape[-2] - pad_bottom:, :] = fill
        else:
            padded[:, :, :pad_left] = fill
            padded[:, :, padded.shape[-1] - pad_right:] = fill
        return padded

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

def img_process(images, processor, image_aspect_ratio):
    """调用 HuggingFace processor 处理图像列表"""
    if image_aspect_ratio == "pad":
        # 计算填充背景色 (基于 processor 的均值)
        # background_color = tuple(int(x * 255) for x in processor.image_mean) # GPT优化
        background_color = tuple(float(x) for x in processor.image_mean)
        images = [expand2square(img, background_color) for img in images]
        # 转换为 Tensor
        images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
    else:
        images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
    return images

from torchvision import transforms
img_resize_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
])

# ==========================================
# B.2 Prompt 构建函数
# ==========================================
def build_sft_instruction_custom(pose_5d, map_name, z_max, z_min):
    definition_text = (
        f"Task: Generate a First-Person View (FPV) image of CS2 map '{map_name}' based on the Radar Map and Camera Pose.\n"
        "Coordinate System Definition:\n"
        "- Map Size: 1024x1024 pixels.\n"
        "- Yaw: 0 degrees is East, increases Clockwise.\n"
        "- Pitch: 0 degrees is looking straight Down (at feet), 180 degrees is looking straight Up (at sky).\n"
        f"- Z-Height: Absolute vertical coordinate. Valid values are bounded by the map's global topology, ranging from the lowest point at {z_min:.2f} to the highest point at {z_max:.2f}."
    )
    # 注意：这里使用了传入字典的 key，确保和你下面的 __getitem__ 构造一致
    pose_str = (
        f"Position(x={pose_5d['x']:.1f}, y={pose_5d['y']:.1f}, z={pose_5d['z']:.3f}), "
        f"Rotation(pitch={pose_5d['angle_v']:.1f}, yaw={pose_5d['angle_h']:.1f})"
    )
    full_instruction = f"{definition_text}\n\nCurrent Camera Pose: {pose_str}\n<image>"
    return full_instruction

def preprocess_multimodal(sources: Sequence[str], img_size: int = 448) -> Dict:
    # NOTE: default to 256 tokens for 448x448
    if img_size==224:
        und_placeholder = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * 64}{IMG_END_TOKEN}' # resize224
    else:
        und_placeholder = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * 256}{IMG_END_TOKEN}'
    gen_placeholder = ""
    # "[IMG]" + "<image>" * data_args.n_query + "[/IMG]"
    inst_type = None
    for source in sources:  # [instance]
        for sentence in source:
            if sentence["from"] == "human" and "<image>" in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, und_placeholder).strip()
                inst_type = "und"
            elif sentence["from"] == "gpt" and "<image>" in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, gen_placeholder).strip()
                inst_type = "gen"
    return sources, inst_type

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def preprocess_internvl(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。") -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    # tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n'}}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' %}{{ '<IMG_CONTEXT>\n' }}{% elif content['type'] == 'video' %}{{ '<video>\n' }}{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{{'<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}"
    # tokenizer.chat_template = chat_template
    if getattr(tokenizer, "chat_template", None) != chat_template:
        tokenizer.chat_template = chat_template

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        # input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)

            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id



        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    input_ids = input_ids[:, :-2]
    targets = targets[:, :-2]
    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    # tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    # tokenizer.chat_template = chat_template
    if getattr(tokenizer, "chat_template", None) != chat_template:
        tokenizer.chat_template = chat_template

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)

            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id



        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )




def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    # tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    # if has_image:
    if has_image and "<image>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)

            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id



        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )



def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        # assert DEFAULT_IMAGE_TOKEN in source[0]['value'] or DEFAULT_IMAGE_TOKEN in source[1]['value']
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.version == "llama3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "internvl":
        return preprocess_internvl(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    logging.info(f"conversations: {conversations}")
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)



# ==========================================
# C 数据集信息
# ==========================================
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

map_to_id_dict = {
    'de_dust2': 0,
    'de_inferno': 1,
    'de_mirage': 2,
    'de_nuke': 3,
    'de_ancient':4,
    'de_anubis': 5,
    'de_golden': 6,
    'de_overpass': 7,
    'de_palacio': 8,
    'de_train': 9,
    'de_vertigo': 10,
    'cs_agency': 11,
    'cs_italy': 12,
    'cs_office': 13
}

id_to_map_dict = {k: i for i, k in enumerate(map_to_id_dict.keys())}

# GPT优化
def _build_loc_prompt_cache(map_names, tokenizer, img_size):
    cache = {}
    for map_name in map_names:
        user_text_loc = get_loc_prompt(map_name)
        sources_loc = {
            "conversations": [
                {"from": "human", "value": user_text_loc},
                {"from": "gpt", "value": ""}
            ]
        }
        sources_loc, _ = preprocess_multimodal([copy.deepcopy(sources_loc["conversations"])], img_size)
        pre_dict_loc = preprocess(sources_loc, tokenizer, has_image=True)
        cache[map_name] = {
            "raw_prompt": user_text_loc,
            "input_ids": pre_dict_loc["input_ids"][0].clone(),
            "labels": pre_dict_loc["labels"][0].clone(),
        }
    return cache

# GPT优化
def _build_map_tensor_cache(map_images, image_processor, image_aspect_ratio, is_resize_224):
    map_tensor_cache_448 = {}
    map_tensor_cache_224 = {}
    for map_name, map_img in map_images.items():
        map_tensor_448 = img_process([map_img], image_processor, image_aspect_ratio).contiguous()
        map_tensor_cache_448[map_name] = map_tensor_448
        if is_resize_224:
            map_tensor_cache_224[map_name] = img_resize_transform(map_tensor_448).contiguous()
    return map_tensor_cache_448, map_tensor_cache_224

# GPT优化
def _load_fps_image(config, map_name, file_frame):
    ext = ".jpg" if config['data_dir'] == 'data/preprocessed_data' else ".png"
    fps_path = f"{config['data_dir']}/{map_name}/imgs/{file_frame}{ext}"
    with Image.open(fps_path) as img:
        return img.convert('RGB')

# GPT优化
INV_1024 = 1.0 / 1024.0
INV_TAU = 1.0 / (2.0 * np.pi)

def _build_csgo_entry(map_name, pos_data, z_min, z_max):
    z_norm = (pos_data['z'] - z_min) / (z_max - z_min + 1e-6)
    pitch_deg = pos_data['angle_v'] * INV_TAU * 360.0
    yaw_deg = pos_data['angle_h'] * INV_TAU * 360.0
    return {
        'map': map_name,
        'file_frame': pos_data['file_frame'],
        'map_id': map_to_id_dict.get(map_name, 0),
        'actions': torch.tensor([
            pos_data['x'] * INV_1024,
            pos_data['y'] * INV_1024,
            z_norm,
            pos_data['angle_v'] * INV_TAU,
            pos_data['angle_h'] * INV_TAU,
        ], dtype=torch.bfloat16).unsqueeze(0),
        'pose_dict': {
            'x': pos_data['x'],
            'y': pos_data['y'],
            'z': pos_data['z'],
            'angle_v': pitch_deg,
            'angle_h': yaw_deg,
        },
        'z_min': z_min,
        'z_max': z_max,
    }



# ==========================================
# D 多任务数据集类
# ==========================================
class UniLIPMultiTaskDataset(Dataset):
    def __init__(self, config, tokenizer, data_args):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.data_args = data_args

        # 任务混合比例: 0.6 表示 60% 概率定位, 40% 概率生成
        self.mix_ratio = config.get('task_mix_ratio', 0.5)
        # GPT优化
        self.img_size = config.get("img_size", 448)
        self.is_resize_224 = self.img_size == 224
        self.cfg_drop_prob = config.get("cfg_drop_prob", 0.1)

        self.data_entries = []
        self.map_z_range = {}

        logging.info("🔄 Loading Multi-Task CS2 Dataset...")

        # --- 1. 加载数据索引 (逻辑复用 CsgoTrainDataset_IT) ---
        for map_name in config["train_maps"]:
            position_data_path = f"{config['data_dir']}/{map_name}/splits_20000_5000/train_split.json"
            if config['debug']:
                position_data_path = f"{config['data_dir']}/{map_name}/splits_20000_5000/test_split.json"

            logging.info(f"Loading CS2 Data Split {position_data_path}...")
            with open(position_data_path, "r", encoding="utf-8") as f:
                positions_data = json.load(f)

            # 计算 Z 轴范围
            max_z, min_z = -float('inf'), float('inf')
            for data in positions_data:
                if data['z'] > max_z: max_z = data['z']
                if data['z'] < min_z: min_z = data['z']
            self.map_z_range[map_name] = {'max_z': max_z, 'min_z': min_z}

            # 载入数据
            for pos_data in positions_data:
                # entry = {
                #     'map': map_name,
                #     'file_frame': pos_data['file_frame'],
                #     'x': pos_data['x'],
                #     'y': pos_data['y'],
                #     'z': pos_data['z'],
                #     'angle_v': pos_data['angle_v'], # Radian
                #     'angle_h': pos_data['angle_h'], # Radian
                # }
                # self.data_entries.append(entry) # GPT优化
                self.data_entries.append(_build_csgo_entry(map_name, pos_data, min_z, max_z))

        # Debug 采样
        if config.get('debug', False):
            sampled_num = config.get('debug_num_train_data', 100)
            # self.data_entries = self.data_entries[:sampled_num]
            self.data_entries = random.sample(self.data_entries, sampled_num)
            logging.info([(data['map'], data['file_frame']) for data in self.data_entries])

        # unilip trainner 需要的 list_data_dict 格式
        self.list_data_dict = {
            "type": ["CS2_Multi_Task"] * len(self.data_entries),
            "id": [str(entry['map'] + "_" + entry['file_frame']) for entry in self.data_entries]
        }

        # 预加载所有地图图片到内存
        self.map_images = {}
        for map_name, filename in map_path_dict.items():
            path = f"{config['data_dir']}/{map_name}/{filename}"
            if os.path.exists(path):
                # img = Image.open(path).convert('RGB')
                # # 预先做 Resize 以省内存 (如果 processor 需要 448)
                # # img = img.resize((448, 448))
                # self.map_images[map_name] = img # GPT优化
                self.map_images[map_name] = convert_image_dtype(
                    read_image(path, mode=ImageReadMode.RGB),
                    torch.float32,
                )
            else:
                logging.warning(f"Map image not found: {path}")

        # GPT优化
        self.map_tensor_cache_448, self.map_tensor_cache_224 = _build_map_tensor_cache(
            self.map_images,
            self.data_args.image_processor,
            self.data_args.image_aspect_ratio,
            self.is_resize_224,
        )
        # GPT优化
        self.loc_prompt_cache = _build_loc_prompt_cache(
            self.map_z_range.keys(),
            self.tokenizer,
            self.img_size,
        )
        # GPT优化
        self._fps_dropout_transform = transforms.Compose([
            transforms.ToTensor(),
            CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
            GridDropout(grid_size=4, p=0.3),
            RandomErasing(probability=config.get('erasing_p', 0.3), mean=[0.0, 0.0, 0.0])
        ])

        logging.info(f"✅ Total entries: {len(self.data_entries)}")


    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        # 1. 获取基础数据
        data = self.data_entries[idx]
        map_name = data['map']

        # 2. 决定当前样本的任务类型
        # task_id = 0 -> Localization (定位)
        # task_id = 1 -> Generation (生成)
        task_id = 0 if random.random() < self.mix_ratio else 1

        # 3. 加载图像资源
        # Map Image
        # map_filename = map_path_dict.get(map_name, 'de_dust2_radar_psd.png')
        # map_path = f"{self.config['data_dir']}/{map_name}/{map_filename}"
        # map_img_pil = Image.open(map_path).convert('RGB')
        # map_img_pil = self.map_images.get(map_name).copy()  # GPT优化
        map_tensor_448 = self.map_tensor_cache_448[map_name]
        map_tensor = self.map_tensor_cache_224[map_name] if self.is_resize_224 else map_tensor_448

        # FPS Image
        # ext = ".jpg" if self.config['data_dir'] == 'data/preprocessed_data' else ".png"
        # fps_path = f"{self.config['data_dir']}/{map_name}/imgs/{data['file_frame']}{ext}"
        # fps_img_pil = Image.open(fps_path).convert('RGB') # GPT优化
        fps_img_pil = _load_fps_image(self.config, map_name, data['file_frame'])

        # 4. 计算归一化坐标 (Common for both tasks)
        # z_info = self.map_z_range[map_name]
        # x_norm = data['x'] / 1024.0
        # y_norm = data['y'] / 1024.0
        # z_norm = (data['z'] - z_info['min_z']) / (z_info['max_z'] - z_info['min_z'] + 1e-6)
        # v_norm = data['angle_v'] / (2 * np.pi) # Pitch 0-1
        # pitch_deg = (data['angle_v'] / (2 * np.pi)) * 360.0
        # h_norm = data['angle_h'] / (2 * np.pi) # Yaw 0-1
        # yaw_deg = (data['angle_h'] / (2 * np.pi)) * 360.0

        # # Ground Truth Tensor for Localization Head
        # loc_coords_norm = torch.tensor([x_norm, y_norm, z_norm, v_norm, h_norm], dtype=torch.bfloat16)
        # pose_dict = {
        #             'x': data['x'],
        #             'y': data['y'],
        #             'z': data['z'],
        #             'angle_v': pitch_deg,
        #             'angle_h': yaw_deg,
        #         }  # GPT优化
        pose_dict = data['pose_dict']
        loc_coords_norm = data['actions']

        # 5. 任务分流处理
        if task_id == 0:
            # =========================================
            # Task: LOCALIZATION (Map + FPS -> Pose)
            # =========================================
            # Text Prompt
            # user_text = get_loc_prompt(map_name)
            # sources = {
            #     "conversations": [
            #         {"from": "human", "value": user_text},
            #         {"from": "gpt", "value": ""} # Assistant 回复位置Token
            #     ]
            # }
            # sources, _ = preprocess_multimodal(copy.deepcopy([sources["conversations"]]), self.config.get("img_size", 448))
            # preprocess_dict = preprocess(sources, self.tokenizer, has_image=True)
            # input_ids = preprocess_dict["input_ids"][0]
            # labels = preprocess_dict["labels"][0] # GPT优化
            loc_cache = self.loc_prompt_cache[map_name]
            user_text = loc_cache["raw_prompt"]
            input_ids = loc_cache["input_ids"]
            labels = loc_cache["labels"]

            # FPS Image & MAP Image
            if self.config.get('is_fps_dropout', False):
                fps_img_pil = self.loc_fps_transform(self.config, fps_img_pil)
            # all_images = [fps_img_pil, map_img_pil]
            # process_images = img_process(
            #     all_images,
            #     self.data_args.image_processor,
            #     self.data_args.image_aspect_ratio
            # ) # shape: [2, C, H, W]
            # if self.config.get("img_size", 448)==224:
            #     und_image = img_resize_transform(process_images[:-1]) # [1, C, H, W]
            #     aux_image = img_resize_transform(process_images[-1:]) # [1, C, H, W]
            # else:
            #     und_image = process_images[:-1] # [1, C, H, W]
            #     aux_image = process_images[-1:] # [1, C, H, W] # GPT优化
            fps_tensor_448 = img_process([fps_img_pil], self.data_args.image_processor, self.data_args.image_aspect_ratio)
            und_image = img_resize_transform(fps_tensor_448) if self.is_resize_224 else fps_tensor_448
            aux_image = map_tensor.clone()

            # Target Image for Generator (DiT)
            # 定位任务不需要生成，所以给一个全黑的或者随机的占位符，Loss Mask 会把它忽略
            # gen_image = torch.zeros_like(process_images[:-1]) # GPT优化
            gen_image = torch.zeros_like(fps_tensor_448)

            # Head Mask: 开启 Pose Head，关闭 Gen Head
            # [Loc_Loss_Weight, Gen_Loss_Weight]
            loss_mask = torch.tensor([1.0, 0.0], dtype=torch.float32)

            # aux_loc_input_ids = copy.deepcopy(input_ids)
            # aux_loc_labels = copy.deepcopy(labels) # GPT优化
            aux_loc_input_ids = input_ids
            aux_loc_labels = labels

        else:
            # =========================================
            # Task: GENERATION (Map + Pose -> FPS)
            # =========================================
            # 模拟 Classifier-Free Guidance (CFG) 训练
            # 10% 概率给空 Prompt ("Generate...")，90% 概率给完整 Prompt
            # if random.random() > 0.1: # GPT优化
            if random.random() > self.cfg_drop_prob:
                # Positive Prompt
                # user_text = build_sft_instruction_custom(pose_dict, map_name, z_info['max_z'], z_info['min_z'])
                user_text = build_sft_instruction_custom(pose_dict, map_name, data['z_max'], data['z_min'])
            else:
                # CFG Negative/Generic Prompt
                # 注意：必须包含 <image> 且位置要和 Positive Prompt 里的位置一致(这里都在最后)
                user_text = "Generate the view.\n<image>"

            # 构建 UniLIP 标准对话格式
            sources = {
                "conversations": [
                    {"from": "human", "value": user_text},
                    {"from": "gpt", "value": "<image>"} # Assistant 回复生成的图片 Token
                ]
            }
            # 处理 <image> token, 替换为 <img><IMG_CONTEXT>*256<img>
            sources, _ = preprocess_multimodal(copy.deepcopy([sources["conversations"]]), self.config.get("img_size", 448))
            # Tokenize 文本得到 input_ids 和 labels(已替换IGNORE_INDEX); has_image=True 防止多模态输入时, tokenizer 报错
            preprocess_dict = preprocess(sources, self.tokenizer, has_image=True)
            input_ids = preprocess_dict["input_ids"][0]
            labels = preprocess_dict["labels"][0]

            # 将 Radar 和 FPS 打包一起处理，确保经过相同的 Preprocess
            # 顺序：[Radar(Und), FPS(Gen)]
            # all_images = [map_img_pil, fps_img_pil]
            # process_images = img_process(
            #     all_images,
            #     self.data_args.image_processor,
            #     self.data_args.image_aspect_ratio
            # ) # shape: [2, C, H, W]
            # if self.config.get("img_size", 448)==224:
            #     und_image = img_resize_transform(process_images[:-1]) # [1, C, H, W]
            # else:
            #     und_image = process_images[:-1] # [1, C, H, W]
            # gen_image = process_images[-1:] # [1, C, H, W] # GPT优化
            fps_tensor_448 = img_process([fps_img_pil], self.data_args.image_processor, self.data_args.image_aspect_ratio)
            und_image = map_tensor.clone()
            gen_image = fps_tensor_448

            aux_image = torch.zeros_like(und_image)

            loss_mask = torch.tensor([0.0, 1.0], dtype=torch.float32)

            # if self.config.get('is_loc_aux_loss', False):
            # aux_loc_user_text = get_loc_prompt(map_name)
            # aux_loc_sources = {
            #     "conversations": [
            #         {"from": "human", "value": aux_loc_user_text},
            #         {"from": "gpt", "value": ""} # Assistant 回复位置Token
            #     ]
            # }
            # aux_loc_sources, _ = preprocess_multimodal(copy.deepcopy([aux_loc_sources["conversations"]]), self.config.get("img_size", 448))
            # aux_loc_preprocess_dict = preprocess(aux_loc_sources, self.tokenizer, has_image=True)
            # aux_loc_input_ids = aux_loc_preprocess_dict["input_ids"][0]
            # aux_loc_labels = aux_loc_preprocess_dict["labels"][0] # GPT优化
            aux_loc_cache = self.loc_prompt_cache[map_name]
            aux_loc_input_ids = aux_loc_cache["input_ids"]
            aux_loc_labels = aux_loc_cache["labels"]


        # 6. 返回字典
        # 兼容 UniLIP 和 OpenPI 的字段命名
        return {
            "task_id": task_id,             # 0=Loc, 1=Gen
            "ids": data['file_frame'],
            "und_image": und_image,         # 理解流输入 - 定位=fps 生成=map
            "aux_image": aux_image,         # 辅助输入   - 定位=map 生成=Empty
            "gen_image": gen_image,         # 生成流输出 - 定位=Empty 生成=fps
            "input_ids": input_ids,
            "labels": labels,
            "raw_prompt": user_text,
            # "actions": loc_coords_norm.unsqueeze(0), # [1, 5] 定位真值 (生成任务时此值存在但loss_mask为0)# GPT优化
            "actions": loc_coords_norm,
            "loss_mask": loss_mask,         # [1.0, 0.0] or [0.0, 1.0]
            # "map_id": map_to_id_dict.get(map_name, 0),# GPT优化
            "map_id": data['map_id'],
            "map_name": map_name,
            "pose_dict": pose_dict,
            "aux_loc_input_ids": aux_loc_input_ids,
            "aux_loc_labels": aux_loc_labels,
        }

    # def loc_fps_transform(self, config, fps_img_pil):
    #     # fps_transform = transforms.Compose([
    #     #     transforms.ToTensor(),
    #     #     CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
    #     #     GridDropout(grid_size=4, p=0.3),
    #     #     RandomErasing(probability=config.get('erasing_p', 0.3), mean=[0.0, 0.0, 0.0])
    #     # ])
    #     # fps_img = fps_transform(fps_img_pil)
    #     fps_img = self._fps_dropout_transform(fps_img_pil)
    #     fps_img = np.array(fps_img.permute(1,2,0))
    #     fps_img = Image.fromarray((fps_img*255).astype(np.uint8))

    #     return fps_img
    # GPT优化
    def loc_fps_transform(self, config, fps_img_tensor):
        return self._fps_dropout_transform(fps_img_tensor)






@dataclass
class DataCollatorForUniLIPMultiTaskDataset(object):
    """
    Collate examples for UniLIP Multi-Task (Localization + Generation).
    Adapts inputs based on task_id.
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Union[Dict, List[Dict]]]) -> Dict[str, torch.Tensor]:
        # 1. 提取基础数据
        # [NEW] Flatten logic
        # 如果 Dataset 返回的是成对的列表，先展平
        flat_instances = []
        for item in instances:
            if isinstance(item, list):
                flat_instances.extend(item) # [Loc, Gen] -> Loc, Gen
            else:
                flat_instances.append(item)

        # 将展平后的列表赋值回 instances，后续逻辑保持完全不变
        instances = flat_instances
        # ... (以下原有的处理逻辑保持不变) ...

        # 1. 提取基础数据
        # 注意：这里需要处理 task_id 来决定是否添加生成专用的占位符 token
        task_id_list = []
        input_ids_list = []
        labels_list = []
        ids_list = []
        aux_loc_input_ids_list = []
        aux_loc_labels_list = []

        # 新增字段的容器
        batch_gen_images = []
        batch_und_images = []
        batch_aux_images = []
        batch_actions = []
        batch_loss_masks = []
        batch_map_ids = []

        batch_raw_prompt_list = []
        batch_map_name_list = []
        batch_pose_dict_list = []

        # 2. 逐样本处理
        for instance in instances:
            task_id = instance.get("task_id", 1) # 默认为1(生成)以防万一
            _input_id = instance["input_ids"]
            _label = instance["labels"]
            _id = instance.get("ids", "unknown")
            _aux_loc_input_ids = instance["aux_loc_input_ids"]
            _aux_loc_labels = instance["aux_loc_labels"]

            # === Token 处理逻辑 ===
            # 为了防止超过模型最大长度，先做截断 (预留 257 个位置给生成 Token)
            # UniLIP 原逻辑：input_id[: max_len - 257]
            safe_len = self.tokenizer.model_max_length - 257
            _input_id = _input_id[:safe_len]
            _label = _label[:safe_len]
            _aux_loc_input_ids = _aux_loc_input_ids[:safe_len]
            _aux_loc_labels = _aux_loc_labels[:safe_len]

            # [关键分支]
            if task_id == 1:
                # ==> 生成任务 (Generation)
                # 需要在末尾追加 257 个 Latent Query Tokens 供 DiT 使用
                # 构造 Image Token 序列
                img_tokens = torch.full((257,), IMAGE_TOKEN_IDX, dtype=_input_id.dtype, device=_input_id.device)
                # UniLIP 特殊操作：第一个 Token 设为 151665 (可能是特定的 Start Token 或 Separator)
                # 请确保这个 ID 在您的 Tokenizer 中是合法的，或者沿用原代码的硬编码
                img_tokens[0] = 151665

                # 拼接到 input_ids
                _input_id = torch.cat([_input_id, img_tokens])

                # 构造 Label (对于生成部分，Label 也是 IMAGE_TOKEN_IDX，但在 Model 内部会被 Mask 掉不计算 CE Loss)
                img_labels = torch.full((257,), IMAGE_TOKEN_IDX, dtype=_label.dtype, device=_label.device)
                img_labels[0] = 151665
                _label = torch.cat([_label, img_labels])

            else:
                # ==> 定位任务 (Localization)
                # 不需要追加 Latent Queries，因为我们是用 Action Head 基于前面的文本特征预测
                # 保持 input_ids 原样 (已经包含 <image> placeholders for input images)
                pass

            task_id_list.append(task_id)
            input_ids_list.append(_input_id)
            labels_list.append(_label)
            ids_list.append(_id)
            aux_loc_input_ids_list.append(_aux_loc_input_ids)
            aux_loc_labels_list.append(_aux_loc_labels)

            # === 收集 Tensor 数据 ===
            if "gen_image" in instance and instance["gen_image"] is not None:
                batch_gen_images.append(instance["gen_image"])

            if "und_image" in instance and instance["und_image"] is not None:
                batch_und_images.append(instance["und_image"])

            # [NEW] 收集 Aux Image (辅助图/Map)
            if "aux_image" in instance and instance["aux_image"] is not None:
                batch_aux_images.append(instance["aux_image"])

            # [NEW] 收集 Actions
            if "actions" in instance and instance["actions"] is not None:
                batch_actions.append(instance["actions"]) # [1, 5]

            # [NEW] 收集 Loss Mask
            if "loss_mask" in instance and instance["loss_mask"] is not None:
                batch_loss_masks.append(instance["loss_mask"]) # [2]

            if "map_id" in instance:
                batch_map_ids.append(instance["map_id"])

            if "raw_prompt" in instance:
                batch_raw_prompt_list.append(instance["raw_prompt"])
            if "map_name" in instance:
                batch_map_name_list.append(instance["map_name"])
            if "pose_dict" in instance:
                batch_pose_dict_list.append(instance["pose_dict"])


        # 3. Padding (Pad Input Ids & Labels)
        # batch_first=True -> [BS, Seq]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=IGNORE_INDEX
        )
        aux_loc_input_ids = torch.nn.utils.rnn.pad_sequence(
            aux_loc_input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        aux_loc_labels = torch.nn.utils.rnn.pad_sequence(
            aux_loc_labels_list, batch_first=True, padding_value=IGNORE_INDEX
        )


        # 再次检查最大长度 (Padding 后可能会变长)
        if input_ids.shape[1] > self.tokenizer.model_max_length:
            logging.warning(f"Input length {input_ids.shape[1]} > {self.tokenizer.model_max_length}, truncating.")
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            labels = labels[:, :self.tokenizer.model_max_length]
        if aux_loc_input_ids.shape[1] > self.tokenizer.model_max_length:
            logging.warning(f"Input length {aux_loc_input_ids.shape[1]} > {self.tokenizer.model_max_length}, truncating.")
            input_ids = aux_loc_input_ids[:, :self.tokenizer.model_max_length]
            aux_loc_labels = aux_loc_labels[:, :self.tokenizer.model_max_length]

        # 4. 构建 Batch 字典
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            ids=ids_list,
            aux_loc_input_ids=aux_loc_input_ids,
            aux_loc_labels=aux_loc_labels,
            aux_loc_attention_mask=aux_loc_input_ids.ne(self.tokenizer.pad_token_id),
        )

        # 5. 堆叠图像 (Stack Images)
        # 辅助函数：安全堆叠
        def stack_images(img_list):
            if len(img_list) == 0:
                return None
            # 检查形状一致性
            if all(x is not None and x.shape == img_list[0].shape for x in img_list):
                return torch.cat(img_list, dim=0) # [BS, C, H, W] (因为 Dataset 返回的是 [1, C, H, W])
            else:
                logging.warning("Image shapes inconsistent in batch, returning list instead of tensor.")
                return img_list

        batch["gen_image"] = stack_images(batch_gen_images)
        batch["und_image"] = stack_images(batch_und_images)
        batch["aux_image"] = stack_images(batch_aux_images)

        # 6. 堆叠其他 Tensor (Actions, Mask)
        if len(batch_actions) > 0:
            batch["actions"] = torch.stack(batch_actions, dim=0) # [BS, 1, 5]
        else:
            batch["actions"] = None

        if len(batch_loss_masks) > 0:
            batch["loss_mask"] = torch.stack(batch_loss_masks, dim=0) # [BS, 2]
        else:
            batch["loss_mask"] = None

        if len(batch_map_ids) > 0:
            batch["map_id"] = torch.tensor(batch_map_ids, dtype=torch.long)

        if len(task_id_list) > 0:
            batch["task_id"] = torch.tensor(task_id_list, dtype=torch.long)

        if len(batch_raw_prompt_list) > 0:
            batch["raw_prompt"] = batch_raw_prompt_list
        if len(batch_map_name_list) > 0:
            batch["map_name"] = batch_map_name_list
        if len(batch_pose_dict_list) > 0:
            batch["pose_dict"] = batch_pose_dict_list

        return batch





class UniLIPMultiTaskBalancedDataset(Dataset):
    def __init__(self, config, tokenizer, data_args):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.data_args = data_args

        # 任务混合比例: 0.6 表示 60% 概率定位, 40% 概率生成
        self.mix_ratio = config.get('task_mix_ratio', 0.5)
        self.img_size = config.get("img_size", 448)
        self.is_resize_224 = self.img_size == 224
        self.cfg_drop_prob = config.get("cfg_drop_prob", 0.1)

        self.data_entries = []
        self.map_z_range = {}

        logging.info("🔄 Loading Multi-Task CS2 Dataset...")

        # --- 1. 加载数据索引 (逻辑复用 CsgoTrainDataset_IT) ---
        for map_name in config["train_maps"]:
            position_data_path = f"{config['data_dir']}/{map_name}/splits_20000_5000/train_split.json"
            if config['debug']:
                position_data_path = f"{config['data_dir']}/{map_name}/splits_20000_5000/test_split.json"

            logging.info(f"Loading CS2 Data Split {position_data_path}...")
            with open(position_data_path, "r", encoding="utf-8") as f:
                positions_data = json.load(f)

            # 计算 Z 轴范围
            max_z, min_z = -float('inf'), float('inf')
            for data in positions_data:
                if data['z'] > max_z: max_z = data['z']
                if data['z'] < min_z: min_z = data['z']
            self.map_z_range[map_name] = {'max_z': max_z, 'min_z': min_z}

            # 载入数据
            for pos_data in positions_data:
                self.data_entries.append(_build_csgo_entry(map_name, pos_data, min_z, max_z))

        # Debug 采样
        if config.get('debug', False):
            sampled_num = config.get('debug_num_train_data', 100)
            # self.data_entries = self.data_entries[:sampled_num]
            self.data_entries = random.sample(self.data_entries, sampled_num)
            logging.info([(data['map'], data['file_frame']) for data in self.data_entries])

        self.list_data_dict = {
            "type": ["CS2_Multi_Task"] * len(self.data_entries),
            "id": [str(entry['map'] + "_" + entry['file_frame']) for entry in self.data_entries]
        }

        # 预加载所有地图图片到内存
        self.map_images = {}
        for map_name, filename in map_path_dict.items():
            path = f"{config['data_dir']}/{map_name}/{filename}"
            if os.path.exists(path):
                self.map_images[map_name] = convert_image_dtype(
                    read_image(path, mode=ImageReadMode.RGB),
                    torch.float32,
                )
            else:
                logging.warning(f"Map image not found: {path}")

        self.map_tensor_cache_448, self.map_tensor_cache_224 = _build_map_tensor_cache(
            self.map_images,
            self.data_args.image_processor,
            self.data_args.image_aspect_ratio,
            self.is_resize_224,
        )

        self.loc_prompt_cache = _build_loc_prompt_cache(
            self.map_z_range.keys(),
            self.tokenizer,
            self.img_size,
        )

        self._fps_dropout_transform = transforms.Compose([
            transforms.ToTensor(),
            CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
            GridDropout(grid_size=4, p=0.3),
            RandomErasing(probability=config.get('erasing_p', 0.3), mean=[0.0, 0.0, 0.0])
        ])

        logging.info(f"✅ Total entries: {len(self.data_entries)}")

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        # 1. 获取基础数据
        data = self.data_entries[idx]
        map_name = data['map']

        # 2. 加载图像资源
        map_tensor_448 = self.map_tensor_cache_448[map_name]
        map_tensor = self.map_tensor_cache_224[map_name] if self.is_resize_224 else map_tensor_448
        fps_img_pil = _load_fps_image(self.config, map_name, data['file_frame'])
        if self.config.get('is_fps_dropout', False):
            fps_img_pil = self.loc_fps_transform(self.config, fps_img_pil)
        tensor_fps_448 = img_process([fps_img_pil], self.data_args.image_processor, self.data_args.image_aspect_ratio)
        tensor_fps = img_resize_transform(tensor_fps_448) if self.is_resize_224 else tensor_fps_448
        tensor_map = map_tensor.clone()
        tensor_empty = torch.zeros_like(tensor_map)
        tensor_empty_448 = torch.zeros_like(tensor_fps_448)

        # 3. 计算归一化坐标 (Common for both tasks)
        pose_dict = data['pose_dict']
        loc_coords_norm = data['actions']

        # 4. 任务分流处理

        # =========================================
        # 4.1 Task: LOCALIZATION (Map + FPS -> Pose)
        # =========================================
        task_id_loc = 0
        # Tokenize Loc
        loc_cache = self.loc_prompt_cache[map_name]
        # Text Prompt
        user_text_loc = loc_cache["raw_prompt"]

        # 构造 Loc 样本字典
        sample_loc = {
            "task_id": task_id_loc,
            "ids": f"{data['file_frame']}_loc",
            "und_image": tensor_fps,   # Input: FPS
            "aux_image": tensor_map,   # Aux: Map
            "gen_image": tensor_empty_448, # Target: None (Empty)
            "input_ids": loc_cache["input_ids"],
            "labels": loc_cache["labels"],
            "raw_prompt": user_text_loc,
            "actions": loc_coords_norm,
            "loss_mask": torch.tensor([1.0, 0.0], dtype=torch.float32), # [Loc_Loss_Weight, Gen_Loss_Weight]
            "map_id": data['map_id'],
            "map_name": map_name,
            "pose_dict": pose_dict,
            # Loc 任务的 Aux Loc Input 就是它自己
            "aux_loc_input_ids": loc_cache["input_ids"],
            "aux_loc_labels": loc_cache["labels"],
        }

        # =========================================
        # 4.2 Task: GENERATION (Map + Pose -> FPS)
        # =========================================
        task_id_gen = 1
        # 模拟 Classifier-Free Guidance (CFG) 训练
        # 10% 概率给空 Prompt ("Generate...")，90% 概率给完整 Prompt
        if random.random() > self.cfg_drop_prob:
            user_text_gen = build_sft_instruction_custom(pose_dict, map_name, data['z_max'], data['z_min'])
        else:
            user_text_gen = "Generate the view.\n<image>"
        sources_gen = {
            "conversations": [
                {"from": "human", "value": user_text_gen},
                {"from": "gpt", "value": "<image>"} # Assistant 回复生成的图片 Token, collator中使用256*learnable_latent_query替换
            ]
        }
        # 处理 <image> token, 替换为 <img><IMG_CONTEXT>*256<img>
        sources_gen, _ = preprocess_multimodal(copy.deepcopy([sources_gen["conversations"]]), self.config.get("img_size", 448))
        # Tokenize 文本得到 input_ids 和 labels(已替换IGNORE_INDEX); has_image=True 防止多模态输入时, tokenizer 报错
        pre_dict_gen = preprocess(sources_gen, self.tokenizer, has_image=True)

        # 这里的 Aux Loc Input 需要单独构建 (Prompt 为 Loc Prompt)
        # 为了给 Gen 任务提供计算 Loc Aux Loss 的上下文
        # 我们可以直接复用上面 Loc 任务算好的 input_ids，或者重新算一遍
        # 为了效率，直接复用 pre_dict_loc 即可 (因为是同一个样本)

        # 构造 Gen 样本字典
        sample_gen = {
            "task_id": task_id_gen,
            "ids": f"{data['file_frame']}_gen",
            "und_image": tensor_map,   # Input: Map
            "aux_image": tensor_empty, # Aux: None
            "gen_image": tensor_fps_448,   # Target: FPS
            "input_ids": pre_dict_gen["input_ids"][0],
            "labels": pre_dict_gen["labels"][0],
            "raw_prompt": user_text_gen,
            "actions": loc_coords_norm,
            "loss_mask": torch.tensor([0.0, 1.0], dtype=torch.float32), # [Loc_Loss_Weight, Gen_Loss_Weight]
            "map_id": data['map_id'],
            "map_name": map_name,
            "pose_dict": pose_dict,
            # Gen 任务携带 Loc Prompt 用于一致性 Loss
            "aux_loc_input_ids":  loc_cache["input_ids"],
            "aux_loc_labels":  loc_cache["labels"],
        }

        # 5. 返回列表 [Loc, Gen]
        return [sample_loc, sample_gen]

    def loc_fps_transform(self, config, fps_img_tensor):
        return self._fps_dropout_transform(fps_img_tensor)








import torch
import matplotlib.pyplot as plt
import numpy as np
import textwrap

def denormalize_image(tensor):
    """反归一化 CLIP 图片 Tensor -> Numpy Uint8"""
    if tensor is None: return None
    # 确保在 CPU
    tensor = tensor.cpu().detach()

    # CLIP Mean/Std
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    # 反归一化
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)

    # [C, H, W] -> [H, W, C]
    return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

def visualize_batch(batch, tokenizer, max_samples=5, save_path="_vis_results/_trainner_batch.jpg"):
    """
    可视化 Batch 中的前 N 个样本。
    包含：文本信息面板 + Und Image + Aux Image + Gen Image
    """
    batch_size = len(batch['input_ids'])
    limit = min(batch_size, max_samples)

    # 创建大图：每个样本占一行 (高度6)，宽24
    fig, axes = plt.subplots(limit, 4, figsize=(30, 8 * limit))
    if limit == 1: axes = axes.reshape(1, -1) # 处理单样本情况

    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    for i in range(limit):
        # --- 1. 获取基础信息 ---
        # 尝试从 tensor 或 list 中获取，兼容两种格式
        def get_item(key, idx):
            if key not in batch or batch[key] is None: return "N/A"
            item = batch[key]
            if isinstance(item, torch.Tensor):
                return item[idx].cpu().detach()
            elif isinstance(item, list):
                return item[idx]
            return "Unknown Type"

        task_id = get_item("task_id", i)
        task_name = "GENERATION" if task_id == 1 else "LOCALIZATION"

        sample_id = get_item("ids", i)
        loss_mask = get_item("loss_mask", i)
        actions = get_item("actions", i)
        raw_prompt = get_item("raw_prompt", i)
        pose_dict = get_item("pose_dict", i)

        # --- 2. 绘制文本面板 (第一列) ---
        ax_text = axes[i, 0]
        ax_text.axis('off')

        # 构建信息字符串
        info_str = (
            f"Index: {i} | ID: {sample_id}\n"
            f"----------------------------------------\n"
            f"Task: {task_name} (ID: {task_id})\n"
            f"Loss Mask: {loss_mask}\n"
            f"Actions (GT): {actions}\n"
            f"Pose Dict: {pose_dict}\n"
            f"----------------------------------------\n"
            f"Raw Prompt:\n{textwrap.fill(str(raw_prompt), width=80)}\n"
        )

        # 显示 Input IDs 解码片段
        input_ids = batch['input_ids'][i]
        decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
        # 只显示最后 200 个字符避免太长
        info_str += f"----------------------------------------\n"
        info_str += f"Tokenized (Fist 500 chars ... Last 500 chars):\n{textwrap.fill(decoded[:500] + ' ... ... ' + decoded[-500:], width=80)}"

        # 显示 Labels 解码片段
        labels = batch['labels'][i]
        decoded_labels = tokenizer.decode(labels[labels!=-100], skip_special_tokens=False)
        # 只显示最后 200 个字符避免太长
        info_str += f"----------------------------------------\n"
        info_str += f"Tokenized Labels (Fist 100 chars ... Last 100 chars):\n{textwrap.fill(decoded_labels[:100] + ' ... ... ' + decoded_labels[-100:], width=80)}"

        ax_text.text(0, 1, info_str, fontsize=10, verticalalignment='top', fontfamily='monospace')

        # --- 3. 绘制图片 (后三列) ---
        img_keys = [
            ("und_image", "Und Image (Input)"),
            ("aux_image", "Aux Image (Map/Wrist)"),
            ("gen_image", "Gen Image (Target)")
        ]

        for col_idx, (key, title) in enumerate(img_keys):
            ax_img = axes[i, col_idx + 1]

            if key in batch and batch[key] is not None:
                # 检查 batch[key] 的形状，如果是 [BS, C, H, W]
                img_tensor = batch[key][i]

                # 检查是否全黑/全零 (表示未使用的占位符)
                if torch.all(img_tensor == 0):
                    ax_img.text(0.5, 0.5, "Empty / Masked", ha='center', va='center')
                    ax_img.set_facecolor("#f0f0f0") # 灰色背景
                else:
                    img_np = denormalize_image(img_tensor)
                    ax_img.imshow(img_np)
            else:
                ax_img.text(0.5, 0.5, "None", ha='center', va='center')

            ax_img.set_title(title, fontsize=12, fontweight='bold')
            ax_img.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Batch visualization saved to {save_path}")
    plt.close()

# # --- 使用示例 ---
# if __name__ == "__main__":
#     # 假设你已经运行了 collator 得到了 batch
#     # from transformers import AutoTokenizer
#     # tokenizer = AutoTokenizer.from_pretrained("...")
#     # visualize_batch(batch, tokenizer)
#     # visualize_batch(batch, model.text_tokenizer)

#     pass





#### Debug at huggingface.transformers
# /home/jiahao/miniconda3/envs/UniLIP/lib/python3.11/site-packages/transformers/trainer.py
# Line 2579
# epoch_dataloader = train_dataloader
# batch = next(iter(epoch_dataloader))
    # batch.keys()
    # dict_keys(['input_ids', 'labels', 'attention_mask', 'ids', 'gen_image', 'und_image', 'aux_image', 'actions', 'loss_mask', 'map_id', 'task_id'])

# visualize_batch(batch, model.text_tokenizer)

# Line 2618
# batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
    # len(batch_samples)
    # 1
    # batch_samples[0].keys()
    # dict_keys(['input_ids', 'labels', 'attention_mask', 'ids', 'gen_image', 'und_image', 'aux_image', 'actions', 'loss_mask', 'map_id', 'task_id', 'raw_prompt', 'map_name', 'pose_dict'])

# Line 4110
    # outputs = model(**inputs)
    # inputs.keys()
    # dict_keys(['input_ids', 'labels', 'attention_mask', 'ids', 'gen_image', 'und_image', 'aux_image', 'actions', 'loss_mask', 'map_id', 'task_id', 'raw_prompt', 'map_name', 'pose_dict', 'num_items_in_batch'])

