import os
import io
import copy
import sys
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import time
import torch, gc
import glob
import transformers
import tokenizers
import random
from unilip.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_IDX
from torch.utils.data import Dataset
from unilip.train.nonmix_trainer import NonMixTrainer
from unilip import conversation as conversation_lib
from unilip.model import *
from unilip.mm_utils import tokenizer_image_token
from PIL import Image, ImageFile
from datasets import load_dataset, concatenate_datasets
from pathlib import Path
from datasets.utils.logging import set_verbosity_info
from transformers import logging as tf_logging
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor
from unilip.conversation import Conversation, SeparatorStyle
from tqdm import tqdm
from copy import deepcopy


import numpy as np
import yaml
from visual_utils import visualize_dataset_samples
from csgo_datasets.unified_task_dataset import UniLIPMultiTaskDataset, DataCollatorForUniLIPMultiTaskDataset
import datetime
import wandb


import os
# 设置为 false，禁止 tokenizers 并行，防止死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"


ImageFile.LOAD_TRUNCATED_IMAGES = True
transform_und_images = T.Compose([T.Resize(448, interpolation=InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(448)])

set_verbosity_info()
tf_logging.set_verbosity_info()

local_rank = None
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


def rank0_print(*args):
    if local_rank == 0:
        logging.info(*args)


from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


def set_seed(seed=42):
    # 1. Python 内置 random
    random.seed(seed)
    # 2. 操作系统环境 (这对某些哈希操作是必须的，如 set/dict 的顺序)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 3. NumPy
    np.random.seed(seed)
    # 4. PyTorch CPU
    torch.manual_seed(seed)
    # 5. PyTorch GPU (如果可用)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 如果有多张显卡，为所有显卡设置
    # 6. 设置 CuDNN 后端以确保确定性 (会降低性能)
    # 如果你非常看重结果的逐位一致性，必须开启 deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.info(f"随机种子已设置为: {seed}")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    gen_vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_gen_mlp_adapter: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    gen_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    n_query: Optional[int] = field(default=729)  # clip 576, siglip 729
    n_und_query: Optional[int] = field(default=729)  # clip 576, siglip 729
    gen_pooling: Optional[str] = field(default="all")  # options are: pool2d_3, pool2d_9, seq_3, seq_9, seq_27
    unilip_path: Optional[str] = field(default="")
    unilip_factor: Optional[float] = field(default=5.85)
    weighting_scheme: Optional[str] = field(default='logit_normal')
    fix_dit: bool = field(default=False)
    fix_connect: bool = field(default=False)
    fix_vit: bool = field(default=True)
    fix_llm: bool = field(default=True)
    connect_layer: int=field(default=6)
    mllm_path: Optional[str] = field(default="")
    mllm_hf_path: Optional[str] = field(default="")
    vae_path: Optional[str] = field(default="")
    dit_path: Optional[str] = field(default="")

    action_dit_layer: Optional[int] = field(default=3)
    is_action_dit_dense_timestep: Optional[bool] = field(default=False)



@dataclass
class DataArguments:
    csgo_config: str = ""
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    csgo_image_folder: Optional[str] = field(default=None)
    # edit_image_folder: Optional[str] = field(default=None)
    # gen_repeat: Optional[int] = field(default=1)
    # edit_repeat: Optional[int] = field(default=1)
    shortcaption_image_folder: Optional[str] = field(default=None)
    data_type: Optional[str] = field(default="mix")
    image_aspect_ratio: str = "square"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."},
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."},
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    bf16: bool = True
    pretrain_path : str = "none"


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_vision_tower_state_maybe_zero_3(named_params, keys_to_match=[""]):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, vision_tower: str):
    """Collects the state dict and dump to disk."""

    # if getattr(trainer.args, "tune_vision_model", False):

    if trainer.deepspeed:
        torch.cuda.synchronize()


    # Only save Adapter
    keys_to_match = ["mm_projector"]
    if getattr(trainer.args, "use_im_start_end", False):
        keys_to_match.extend(["embed_tokens", "embed_in"])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split("/")[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith("checkpoint-"):
            mm_projector_folder = os.path.join(parent_folder, "mm_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(
                weight_to_save,
                os.path.join(mm_projector_folder, f"{current_folder}.bin"),
            )
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))

    keys_to_match = ["gen_projector"]
    if getattr(trainer.args, "use_im_start_end", False):
        keys_to_match.extend(["embed_tokens", "embed_in"])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split("/")[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith("checkpoint-"):
            mm_projector_folder = os.path.join(parent_folder, "gen_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(
                weight_to_save,
                os.path.join(mm_projector_folder, f"{current_folder}.bin"),
            )
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f"gen_projector.bin"))

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):


    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg

    model.text_tokenizer = tokenizer


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



def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    # NOTE: default to 256 tokens for 448x448
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

def preprocess_internvl(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。") -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n'}}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' %}{{ '<IMG_CONTEXT>\n' }}{% elif content['type'] == 'video' %}{{ '<video>\n' }}{% elif content['type'] == 'text' %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{{'<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}"
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

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
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
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
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



class LazySupervisedMixDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(LazySupervisedMixDataset, self).__init__()

        self.data_args = data_args
        list_data_dict = []

        # image generation
        if self.data_args.gen_image_folder is not None:
            train_dataset = self.load_gen(self.data_args.gen_image_folder)
            for _ in range(self.data_args.gen_repeat):
                list_data_dict.append(train_dataset)
            logging.info(f"finish loading gen image {len(train_dataset)}")

        # image editing
        if self.data_args.edit_image_folder is not None:
            train_dataset = self.load_edit(self.data_args.edit_image_folder)
            for _ in range(self.data_args.edit_repeat):
                list_data_dict.append(train_dataset)
            logging.info(f"finish loading edit image {len(train_dataset)}")

        if len(list_data_dict) > 1:
            list_data_dict = concatenate_datasets(list_data_dict)
        else:
            list_data_dict = list_data_dict[0]

        logging.info(f"Totoal number of training instance: {len(list_data_dict)}")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

        type_list = self.list_data_dict["type"]
        self.type_to_indices = {}
        for idx, t in enumerate(type_list):
            self.type_to_indices.setdefault(t, []).append(idx)

    def __len__(self):
        return len(self.list_data_dict)

    def load_gen(self, gen_path):
        data_files = glob.glob(os.path.join(gen_path, "*.tar"))
        train_dataset = load_dataset("webdataset", data_files=data_files, split="train", num_proc=128, cache_dir='../../cache_web')
        train_dataset = train_dataset.rename_column("jpg", "output_image")
        train_dataset = train_dataset.rename_column("txt", "input_prompt")
        train_dataset = train_dataset.add_column('type', len(train_dataset) * ['T2I'])
        train_dataset = train_dataset.add_column('image_path', len(train_dataset) * [None])
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
            ["output_image", "input_prompt", "type", "image_path"])])
        return train_dataset

    def load_edit(self, edit_path):
        data_files = glob.glob(os.path.join(edit_path, "*.tar"))
        train_dataset = load_dataset("webdataset", data_files=data_files, split="train", num_proc=128, cache_dir='../../cache_web')
        train_dataset = train_dataset.rename_column("input.jpg", "input_image")
        train_dataset = train_dataset.rename_column("output.jpg", "output_image")
        train_dataset = train_dataset.rename_column("txt", "input_prompt")
        train_dataset = train_dataset.add_column('type', len(train_dataset) * ['TI2I'])
        train_dataset = train_dataset.add_column('image_path', len(train_dataset) * [None])
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
            ["input_image", "output_image", "input_prompt", "type", "image_path"])])
        return train_dataset

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        while True:
            try:
                sources = self.list_data_dict[i]
            except Exception as e:
                logging.info(f"data error {i}")
                i = random.randint(0, len(self.list_data_dict) - 1)
                continue
            use_und = False
            if 'input_image' in sources and sources['input_image'] is not None:
                use_und = True
                if random.random() > 0.1:
                    sources["conversations"] = [
                        {"from": "human", "value": f"Edit the image: {sources['input_prompt']}\n<image>"},
                        {"from": "gpt", "value": "<image>"},
                    ]
                else:
                    # for cfg
                    sources["conversations"] = [
                            {"from": "human", "value": f"Edit the image.\n<image>"},
                        {"from": "gpt", "value": "<image>"},
                    ]
            else:
                use_und = False
                if random.random() > 0.1:
                    if sources['input_prompt'] is None or len(sources['input_prompt']) > 1000:
                        i = random.choice(self.type_to_indices[sources['type']])
                        continue
                    sources["conversations"] = [
                        {"from": "human", "value": f"Generate an image: {sources['input_prompt']}"},
                        {"from": "gpt", "value": "<image>"},
                    ]
                else:
                    # for cfg
                    sources["conversations"] = [
                        {"from": "human", "value": f"Generate an image."},
                        {"from": "gpt", "value": "<image>"},
                    ]

            if "output_image" in sources:

                def img_process(images, processor, image_aspect_ratio):
                    if image_aspect_ratio == "pad":

                        def expand2square(pil_img, background_color):
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

                        images = [expand2square(img, tuple(int(x * 255) for x in processor.image_mean)) for img in images]
                        images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
                    else:
                        images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
                    return images


                output_image = sources['output_image']
                # concat input and output, default last image is output image
                if use_und:
                    input_images = deepcopy(sources['input_image'])
                    all_images = [input_images]
                    all_images.append(output_image)
                else:
                    all_images = [output_image]

                all_pil_images = []

                for img in all_images:
                    try:
                        img = img.convert("RGB")
                        if img.size[0] == 1 or img.size[1] == 1:
                            logging.info(f"wrong size: {img.size}", )
                            all_pil_images = None
                            break
                        all_pil_images.append(img)
                    except Exception as e:
                        logging.info(f"Error opening image {img}: {e}")
                        all_pil_images = None
                        break  # Skip to the next image if there's an error



                # If no valid images were found, randomly pick another item
                if all_pil_images is None:
                    logging.info(sources)
                    logging.info(f"warning false image!!!!!!")
                    i = random.choice(self.type_to_indices[sources['type']])
                    continue


                sources, inst_type = preprocess_multimodal(copy.deepcopy([sources["conversations"]]), self.data_args)
            else:
                sources = copy.deepcopy([sources["conversations"]])
            data_dict = preprocess(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

            # image exist in the data
            if "output_image" in self.list_data_dict[i]:
                process_images = img_process(
                    all_pil_images,
                    self.data_args.image_processor,
                    self.data_args.image_aspect_ratio,
                )
                if use_und:
                    data_dict["gen_image"] = process_images[-1:]
                    data_dict["und_image"] = process_images[:-1]
                else:
                    data_dict["gen_image"] = process_images

            data_dict["ids"] = self.list_data_dict[i]["id"] if "id" in self.list_data_dict[i] else "unk"
            return data_dict





# ==========================================
# 1. 辅助函数: 图像 Padding (复用 UniLIP 逻辑)
# ==========================================
def expand2square(pil_img, background_color):
    """将图片填充为正方形，背景色通常为 image_mean"""
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
        background_color = tuple(int(x * 255) for x in processor.image_mean)
        images = [expand2square(img, background_color) for img in images]
        # 转换为 Tensor
        images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
    else:
        images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
    return images

# ==========================================
# 2. Prompt 构建函数 (你提供的代码)
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

# ==========================================
# 3. 核心 Dataset 类
# ==========================================
class CSGOWorldModelDataset(Dataset):
    """
    适配 UniLIP 接口的 CS2 生成任务数据集。
    行为模仿 LazySupervisedMixDataset 的 'Edit' (TI2I) 模式。
    """
    def __init__(self, config, tokenizer, data_args):
        super(CSGOWorldModelDataset, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.data_args = data_args

        # --- A. 加载数据索引 (复用 CsgoTrainDataset_IT 逻辑) ---
        self.data_entries = []
        self.map_z_range = {}

        # 你的地图文件映射
        self.map_path_dict = map_path_dict

        logging.info("🔄 Loading CS2 Dataset Index...")
        for map_name in config["train_maps"]:
            # 1. 确定路径
            if config['data_dir'] == 'data/preprocessed_data':
                if config['debug']:
                    position_data_path = f"{config['data_dir']}/{map_name}/splits_20000_5000/test_split.json"
                else:
                    position_data_path = f"{config['data_dir']}/{map_name}/splits_20000_5000/train_split.json"
            else:
                position_data_path = f"{config['data_dir']}/{map_name}/positions.json"

            # 2. 读取 JSON
            if not os.path.exists(position_data_path):
                logging.info(f"⚠️ Warning: Path not found {position_data_path}, skipping.")
                continue

            logging.info(f"Loading CS2 Data Split {position_data_path}...")
            with open(position_data_path, "r", encoding="utf-8") as f:
                positions_data = json.load(f)

            # 3. 计算 Z 轴范围 (用于归一化)
            max_z, min_z = -float('inf'), float('inf')
            for data in positions_data:
                if data['z'] > max_z: max_z = data['z']
                if data['z'] < min_z: min_z = data['z']
            self.map_z_range[map_name] = {'max_z': max_z, 'min_z': min_z}

            # 4. 存入 entries
            for pos_data in positions_data:
                # 构造 entry
                entry = {
                    'map': map_name,
                    'file_frame': pos_data['file_frame'],
                    'x': pos_data['x'],
                    'y': pos_data['y'],
                    'z': pos_data['z'],
                    'angle_v': pos_data['angle_v'], # Pitch (Radians usually)
                    'angle_h': pos_data['angle_h'], # Yaw (Radians usually)
                }
                self.data_entries.append(entry)

        # 5. 数据过滤 (CSBS/Dust2 only logic)
        if config['data_dir'] == 'data/processed_data':
            logging.info(f"📊 Final total entries : {len(self.data_entries)}")
            self.data_entries = [data for data in self.data_entries if (data['map']=='de_dust2' and data['x']!=562 and data['y']!=736) or (data['map']!='de_dust2')]
            logging.info(f"📊 after filter damaged entries: {len(self.data_entries)}")
            self.data_entries = self.data_entries[:-2000]

        if config['debug'] and config.get('debug_num_train_data', False):
            sampled_num = config.get('debug_num_train_data', len(self.data_entries))
            self.data_entries = self.data_entries[:sampled_num]
            logging.info([data['file_frame'] for data in self.data_entries])
        elif config['debug'] and config.get('debug_num_train_data', False) == False:
            indices = [335, 535, 707, 288, 21, 240, 20, 30, 809, 423, 857, 459, 557, 882, 893, 406, 24, 477, 407, 427, 453, 923, 925, 399, 752, 867, 547, 563, 424, 217, 789, 681]
            self.data_entries = [self.data_entries[i] for i in indices]
            logging.info([data['file_frame'] for data in self.data_entries])
        elif config['debug']==False and config.get('debug_num_train_data', False):
            # self.data_entries = self.data_entries[:config.get('debug_num_train_data', len(self.data_entries))]
            sampled_num = config.get('debug_num_train_data', len(self.data_entries))
            self.data_entries = random.sample(self.data_entries, sampled_num)
            logging.info([data['file_frame'] for data in self.data_entries])

        logging.info(f"✅ CS2 Dataset Loaded. Total train entries: {len(self.data_entries)}")
        self.list_data_dict = {
            "type": ["CS2_Gen"] * len(self.data_entries),
            "id": [str(entry['file_frame']) for entry in self.data_entries]
        }

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, i):
        # 容错重试机制
        while True:
            try:
                data = self.data_entries[i]
                map_name = data['map']

                # --- Step 1: 加载图像 ---
                # A. Input Condition (Radar Map) -> 对应 UniLIP 的 und_image
                map_filename = self.map_path_dict.get(map_name, 'de_dust2_radar_psd.png')
                map_path = f"{self.config['data_dir']}/{map_name}/{map_filename}"
                input_image = Image.open(map_path).convert('RGB') # Radar

                # B. Output Target (FPS Image) -> 对应 UniLIP 的 gen_image
                ext = ".jpg" if self.config['data_dir'] == 'data/preprocessed_data' else ".png"
                fps_path = f"{self.config['data_dir']}/{map_name}/imgs/{data['file_frame']}{ext}"
                output_image = Image.open(fps_path).convert('RGB') # FPS

                # --- Step 2: 准备 Prompt 数据 ---
                # Z 归一化
                z_min = self.map_z_range[map_name]['min_z']
                z_max = self.map_z_range[map_name]['max_z']
                z_norm = (data['z'] - z_min) / (z_max - z_min + 1e-6)

                # 角度转换: 原始数据通常是归一化到0~1之间的弧度 (0-2pi)，需转为度数
                # 你的 Prompt 定义: Yaw 0=East, Pitch 0=Down
                angle_h = data['angle_h'] / (2 * np.pi)
                yaw_deg = (data['angle_h'] / (2 * np.pi)) * 360.0

                angle_v = data['angle_v'] / (2 * np.pi)
                pitch_deg = (data['angle_v'] / (2 * np.pi)) * 360.0

                loc_array = np.array([data['x'] / 1024, data['y'] / 1024, z_norm, angle_v, angle_h])
                loc_coords = torch.tensor(loc_array, dtype=torch.float32)
                pose_dict = {
                    'x': data['x'],
                    'y': data['y'],
                    'z': data['z'],
                    'angle_v': pitch_deg,
                    'angle_h': yaw_deg,
                }

                # --- Step 3: 构建对话 (Conversations) ---
                # 模拟 Classifier-Free Guidance (CFG) 训练
                # 10% 概率给空 Prompt ("Generate...")，90% 概率给完整 Prompt

                if random.random() > 0.1:
                    # Positive Prompt
                    user_text = build_sft_instruction_custom(pose_dict, map_name, z_max, z_min)
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

                # --- Step 4: Tokenization ---
                # 处理 <image> token 替换
                sources, _ = preprocess_multimodal(deepcopy([sources["conversations"]]), self.data_args)

                # Tokenize 文本得到 input_ids 和 labels
                # has_image=True 很重要，防止 tokenizer 报错
                data_dict = preprocess(sources, self.tokenizer, has_image=True)

                # 解包 batch (preprocess 默认返回 batch size 1)
                data_dict = dict(
                    input_ids=data_dict["input_ids"][0],
                    labels=data_dict["labels"][0]
                )

                # --- Step 5: 图像 Tensor 处理 ---
                # 将 Radar 和 FPS 打包一起处理，确保经过相同的 Preprocess
                # 顺序：[Radar(Und), FPS(Gen)]
                all_images = [input_image, output_image]

                process_images = img_process(
                    all_images,
                    self.data_args.image_processor,
                    self.data_args.image_aspect_ratio
                ) # shape: [2, C, H, W]

                # 分配给 UniLIP 训练脚本识别的 Key
                # und_image: 输入条件 (Radar)
                # gen_image: 监督目标 (FPS)
                data_dict["und_image"] = process_images[:-1] # [1, C, H, W]
                data_dict["gen_image"] = process_images[-1:] # [1, C, H, W]

                # 这里的 id 是为了 logging 方便
                data_dict["ids"] = str(data['file_frame'])
                data_dict["loc_coords"] = loc_coords
                data_dict["map_name"] = map_name

                return data_dict

            except Exception as e:
                logging.info(f"Error loading index {i}: {e}")
                # 随机换一个数据重试，防止训练中断
                i = random.randint(0, len(self.data_entries) - 1)





@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "ids"))
        multi_input_ids = []
        multi_labels = []
        i_s_pos = []
        for input_id, label in zip(input_ids, labels):
            input_id = input_id[: self.tokenizer.model_max_length - 257]
            label = label[: self.tokenizer.model_max_length - 257]
            i_s_pos.append(input_id.shape[0]+1)
            img_id = torch.full((257,), IMAGE_TOKEN_IDX, dtype=input_id.dtype, device=input_id.device)
            img_id[0] = 151665
            input_id = torch.cat([input_id, img_id])
            img_label = torch.full((257,), IMAGE_TOKEN_IDX, dtype=label.dtype, device=label.device)
            img_label[0] = 151665
            label = torch.cat([label, img_label])
            multi_input_ids.append(input_id)
            multi_labels.append(label)

        input_ids = multi_input_ids
        labels = multi_labels

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        if input_ids.shape[1] > self.tokenizer.model_max_length:
            logging.info(f"Warning input with length {input_ids.shape[1]} is longer than max length {self.tokenizer.model_max_length}")
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        batch_gen_images = []
        batch_und_images = []

        for instance in instances:
            if "gen_image" in instance:
                batch_gen_images.append(instance["gen_image"])


        if len(batch_gen_images) > 0:
            if all(x is not None and y.shape == batch_gen_images[0][0].shape for x in batch_gen_images for y in x):
                batch["gen_image"] = torch.cat([images for images in batch_gen_images], dim=0)
            else:
                batch["gen_image"] = batch_gen_images
        else:
            batch["gen_image"] = None


        for instance in instances:
            if "und_image" in instance:
                batch_und_images.append(instance["und_image"])  ## 1*1024*1176


        # logging.info(f"batch_und_images {batch_und_images}")
        if len(batch_und_images) > 0:
            batch["und_image"] = torch.cat([images for images in batch_und_images], dim=0)
        else:
            batch["und_image"] = None
        batch["ids"] = ids

        batch["i_s_pos"] = i_s_pos

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:

    if data_args.data_type == "mix":
        train_dataset = LazySupervisedMixDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    else:
        raise ValueError("Unknown data type. Please check the Dataloader type.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def unlock_vit(training_args, model_args, vision_tower):
    for n, p in vision_tower.named_parameters():
        p.requires_grad = True


def train(attn_implementation=None):

    global local_rank

    set_seed()

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.lr_scheduler_kwargs = {"min_lr": 1e-5}

    with open(data_args.csgo_config, 'r') as f:
        csgo_config = yaml.safe_load(f)

    cur_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{training_args.output_dir.replace('outputs', 'logs')}/train_{cur_time_str}"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
            format=f'%(asctime)s - %(levelname)s - %(message)s - (%(process)d:%(filename)s:%(lineno)s)',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'train.log')),
                logging.StreamHandler()
            ],
            force=True
    )
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger = logging.getLogger(__name__)

    if "wandb" in training_args.report_to and training_args.local_rank in [-1, 0]:
        # 如果 args 里有 run_name 就用，没有就自动生成
        run_name = training_args.run_name if training_args.run_name else f"{pathlib.Path(training_args.output_dir).name}"

        key_file = Path(".wandb_api_key.txt")
        api_key = None
        if key_file.exists():
            api_key = key_file.read_text(encoding="utf-8").strip()
            logging.info(f"Loaded WandB API key from {key_file}")

        if api_key:
            wandb.login(key=api_key)
        wandb.init(
            entity="zhh",
            project="UniLIP_csgo_fpsgen",
            name=run_name,
            config={**vars(model_args), **vars(data_args), **vars(training_args), **csgo_config},
        )


    logging.info(f"model_args: {model_args}")
    logging.info(f"data_args: {data_args}")
    logging.info(f"training_args: {training_args}")
    logging.info(f"csgo_config: {csgo_config}")

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    if csgo_config.get("is_multi_task"):
        model = Unified_UniLIP_InternVLForCausalLM.from_pretrained(
            model_args.model_name_or_path, # UniLIP-1B with new unified_unilip config
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args,
        )
    else:
        model = UniLIP_InternVLForCausalLM.from_pretrained(
            model_args.model_name_or_path, #UniLIP-1B
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args,
        )
    model.config.use_cache = False

    if model_args.freeze_backbone: # True
        for (n, p) in model.get_model().named_parameters():
            p.requires_grad = False
        for (n, p) in model.lm_head.named_parameters():
            p.requires_grad = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = AutoProcessor.from_pretrained(model_args.mllm_hf_path).tokenizer
    tokenizer.model_max_length = training_args.model_max_length
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(
                pad_token="<pad>",
                additional_special_tokens=["[IMG]", "[/IMG]", "<image>"],
            ),
            tokenizer=tokenizer,
            model=model,
        )
    elif not "<image>" in tokenizer.get_added_vocab():
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(additional_special_tokens=["[IMG]", "[/IMG]", "<image>"]),
            tokenizer=tokenizer,
            model=model,
        )
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama3"]
    logging.info(f"Using conversation format: {conversation_lib.default_conversation.version}")

    model.config.is_action_dit_dense_timestep = model_args.is_action_dit_dense_timestep = csgo_config.get("is_action_dit_dense_timestep", False)

    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    # fix connect False
    # fix dit False
    # unilip load from checkpoint!!! unilip = vision_tower + multi_modal_projector
    # DiT load from checkpoint!!!
    # Connector load from checkpoint!!! = llm_connector + projector
    # latent_queries load from checkpoint!!!

    # Unified_UniLIP_InternVLForCausalLM.from_pretrained()启用了low_mem和accelerator且UniLIP权重中不包含新增的定位模块action_dit，导致action_dit没有正确加载Qwen2Model的权重，这里避开Unified_UniLIP_InternVLForCausalLM的初始化和from_pretrained方法。重新加载action_dit的权重，防止梯度爆炸
    model.get_model().initialize_localization_modules(model_args=model_args)


    if not model_args.fix_vit:
        for p in model.get_model().vision_tower.parameters():
            p.requires_grad = True
        for p in model.get_model().multi_modal_projector.parameters():
            p.requires_grad = True
    if not model_args.fix_llm:
        for p in model.get_model().language_model.parameters():
            p.requires_grad = True

    data_args.image_processor = AutoProcessor.from_pretrained(model_args.mllm_hf_path).image_processor

    data_args.is_multimodal = True
    data_args.n_query = model_args.n_query
    data_args.n_und_query = model_args.n_und_query

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter

    # Calculate total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.get_model().parameters())
    trainable_params = sum(p.numel() for p in model.get_model().parameters() if p.requires_grad)

    train_param_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            train_param_names.append(name)
            logging.info(f"     trainable params: {name}")
    # logging.info("trainable params", train_param_names)
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Trainable parameters: {trainable_params}")
    logging.info(f"trainable percent: {100*trainable_params / total_params:2f} %")


    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id

    if training_args.pretrain_path != 'none':
        pretrain_path = training_args.pretrain_path
        msg = model.load_state_dict(torch.load(pretrain_path), strict=False)
        logging.info(f"load pretrain: {pretrain_path}")
        logging.info(msg)

    # def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:

    #     if data_args.data_type == "mix":
    #         train_dataset = LazySupervisedMixDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    #     else:
    #         raise ValueError("Unknown data type. Please check the Dataloader type.")

    #     data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    if csgo_config.get("is_multi_task", False):
        train_dataset = UniLIPMultiTaskDataset(csgo_config, tokenizer, data_args)
        eval_dataset = None
        data_collator = DataCollatorForUniLIPMultiTaskDataset(tokenizer=tokenizer)
    else:
        train_dataset = CSGOWorldModelDataset(csgo_config, tokenizer, data_args)
        eval_dataset = None
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    if local_rank in [-1, 0]:
        visualize_dataset_samples(
            train_dataset,
            data_args.image_processor,
            num_samples=20,
            save_path="_debug_dataset_samples.jpg",
            is_multi_task=csgo_config.get("is_multi_task", False)
        )

    trainer = NonMixTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        # **data_module,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    from tabulate import tabulate


    if trainer.is_world_process_zero():
        stat = []
        for i, (n, p) in enumerate(trainer.model.named_parameters()):
            stat.append([i, n, p.shape, p.requires_grad])
        logging.info(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        vision_tower=model_args.vision_tower,
    )

    if training_args.local_rank in [-1, 0]:
        wandb.finish()


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
    # train(attn_implementation="sdpa")





# CUDA_VISIBLE_DEVICES=1 python train_csgo.py --csgo_config csgo_configs/exp0.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs_csgo_1b --num_train_epochs 5 --per_device_train_batch_size 32 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 16 --lazy_preprocess True --n_query 256 --n_und_query 0 --fix_dit False --fix_connect False --fix_llm True


# --report_to none --run_name unilip_intern_vl_1b

# --pretrain_path UniLIP-1B/model.safetensors



#  --lr_scheduler_kwargs {\"min_lr\":1e-5}

# --unilip_path ../tokenizer_ckpt/1b_unilip.pth \

# --mllm_path OpenGVLab/InternVL3-1B \
#
# --vae_path mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers \
# --dit_path Efficient-Large-Model/Sana_600M_512px_diffusers \

# --gen_image_folder ${GEN_IMG_FOLDER} \
# --edit_image_folder ${EDIT_IMG_FOLDER} \
# --gen_repeat 1 \
# --edit_repeat 3 \




# CUDA_VISIBLE_DEVICES=1 python train_csgo.py --csgo_config csgo_configs/exp1.yaml --deepspeed deepspeed_scripts/zero0.json --model_name_or_path UniLIP-1B --unilip_factor 10.6 --mllm_hf_path OpenGVLab/InternVL3-1B-hf --version internvl --data_type "mix" --csgo_image_folder data/preprocessed_data --mm_use_im_start_end False --mm_use_im_patch_token False --bf16 True --output_dir outputs/csgo_1b/exp1 --num_train_epochs 100 --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_strategy "no" --save_strategy "steps" --save_steps 5000 --save_total_limit 1 --learning_rate 1e-4 --weight_decay 0. --warmup_ratio 0.003 --lr_scheduler_type "cosine_with_min_lr" --model_max_length 1024 --logging_steps 1 --tf32 True --gradient_checkpointing True --dataloader_num_workers 16 --lazy_preprocess True --n_query 256 --n_und_query 0 --fix_dit False --fix_connect False --fix_llm True
