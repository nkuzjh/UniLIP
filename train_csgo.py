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
from unilip.model.external_loc_model_loader import build_frozen_external_loc_model
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
from transformers import TrainerCallback, PrinterCallback, ProgressCallback

from peft import LoraConfig, get_peft_model, TaskType, PeftModel

import numpy as np
import yaml
from visual_utils import visualize_dataset_samples_v1, visualize_dataset_samples, visualize_dataset_samples_paired
from csgo_datasets.unified_task_dataset import UniLIPMultiTaskDataset, DataCollatorForUniLIPMultiTaskDataset, UniLIPMultiTaskBalancedDataset
import datetime
import wandb


import os
# 设置为 false，禁止 tokenizers 并行，防止死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"


ImageFile.LOAD_TRUNCATED_IMAGES = True
transform_und_images = T.Compose([T.Resize(448, interpolation=InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(448)])
# transform_und_images = T.Compose([T.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(224)]) #resize224
img_resize_transform = T.Compose([
    T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
])#resize224

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



import torch
import transformers.modeling_utils



from transformers import TrainerCallback
from torch.profiler import profile, record_function, ProfilerActivity, schedule

class ProfilerCallback(TrainerCallback):
    def __init__(self, profile_dir):
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=10, warmup=2, active=50, repeat=1), # 等2步，热身2步，记录3步
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
            record_shapes=True,
            profile_memory=True, # 顺便看看显存变化
            with_stack=True
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self.profiler.start()

    def on_step_end(self, args, state, control, **kwargs):
        self.profiler.step()

    def on_train_end(self, args, state, control, **kwargs):
        self.profiler.stop()



# =================================================================
# [Monkey Patch] 强制所有 Gradient Checkpointing 使用 use_reentrant=False
# 修复 "Trying to backward through the graph a second time" 错误，
# 同时避免 LoRA + 冻结骨干下 reentrant checkpoint 因输入不带 grad 直接切断梯度图。
# =================================================================
_ORIG_CHECKPOINT = torch.utils.checkpoint.checkpoint
def _force_non_reentrant_checkpoint(func, *args, **kwargs):
    # 强制覆盖参数
    kwargs['use_reentrant'] = False
    return _ORIG_CHECKPOINT(func, *args, **kwargs)

# 同时替换 torch 原生入口和 transformers 兼容入口，避免仍有模块绕过 modeling_utils.checkpoint
torch.utils.checkpoint.checkpoint = _force_non_reentrant_checkpoint
transformers.modeling_utils.checkpoint = _force_non_reentrant_checkpoint
print("🔧 Monkey Patch Applied: Forced use_reentrant=False for all checkpoints.")



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
    train_mm_projector_only: bool = field(default=False)
    train_shared_llm_tail_only: bool = field(default=False)
    shared_llm_tail_num_layers: int = field(default=2)
    shared_llm_tail_lora_enabled: bool = field(default=False)
    shared_llm_tail_lora_mode: str = field(default="lora_only")
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

    # lora_enable: bool = False
    is_lora: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"

    mm_projector_lr: Optional[float] = None
    shared_llm_tail_lr: Optional[float] = None
    shared_llm_tail_lora_r: int = 16
    shared_llm_tail_lora_alpha: int = 32
    shared_llm_tail_lora_dropout: float = 0.05
    shared_llm_tail_lora_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    bf16: bool = True
    pretrain_path : str = "none"

    action_dit_projector_lr: float = 1e-3
    action_dit_lr: float = 1e-4
    is_action_dit_projector: bool = False
    loc_learnable_query_lr: float = 5e-4
    is_loc_learnable_query: bool = False
    logging_steps=1,
    logging_strategy="steps",

    profile_dir="./profiler_logs", # PyTorch Profiler精确查看训练时间耗费
    enable_step_timing: bool = True
    step_timing_sync_cuda: bool = True




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
    multimodal_keywords = ["mm_projector", "multi_modal_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)




def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, vision_tower: str, is_lora: bool):
    """Collects the state dict and dump to disk."""

    # if getattr(trainer.args, "tune_vision_model", False):

    if trainer.deepspeed:
        torch.cuda.synchronize()

    def save_weight_dict(weights, folder_name, filename):
        if trainer.args.local_rank in [0, -1]:
            if current_folder.startswith("checkpoint-"):
                save_folder = os.path.join(parent_folder, folder_name)
                os.makedirs(save_folder, exist_ok=True)
                torch.save(weights, os.path.join(save_folder, f"{current_folder}.bin"))
            else:
                torch.save(weights, os.path.join(output_dir, f"{filename}.bin"))

    # Only save Adapter
    ## mm_projector
    keys_to_match = ["mm_projector", "multi_modal_projector"]
    if getattr(trainer.args, "use_im_start_end", False):
        keys_to_match.extend(["embed_tokens", "embed_in"])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split("/")[-1]
    parent_folder = os.path.dirname(output_dir)
    save_weight_dict(weight_to_save, "mm_projector", "mm_projector")

    ## gen_projector
    keys_to_match = ["gen_projector"]
    if getattr(trainer.args, "use_im_start_end", False):
        keys_to_match.extend(["embed_tokens", "embed_in"])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split("/")[-1]
    parent_folder = os.path.dirname(output_dir)
    save_weight_dict(weight_to_save, "gen_projector", "gen_projector")

    ## localization head
    action_keys = [
        "action_dit_projector",
        "action_dit_norm",
        "action_in_proj",
        "action_out_proj",
        "time_mlp_in",
        "time_mlp_out",
        "loc_learnable_query",
        "regression_loc_head",
        "cross_view_fusion",
        "vit_loc_fusion",
    ]
    # 使用 fuzzy match 提取这些模块的权重
    action_weights = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), action_keys)
    if len(action_weights) > 0:
        save_weight_dict(action_weights, "action_heads", "action_heads")

    # LoRA Adapter 保存逻辑
    if is_lora:
        # LoRA Adapter 保存需要特殊处理，因为它们散落在各个子模块中
        # 我们需要遍历所有子模块，找到 PeftModel 并调用其 save_pretrained
        if trainer.args.local_rank in [0, -1]:
            # 获取原始模型 (解包 DDP/DeepSpeed)
            model_to_save = trainer.model
            # 如果被 DeepSpeed 包装，尝试获取 module
            if hasattr(model_to_save, "module"):
                model_to_save = model_to_save.module

            # 这里的 get_model() 是 Unified_UniLIP_InternVLModel
            base_model = model_to_save.get_model()

            # 定义可能包含 LoRA 的子模块名称
            lora_module_names = ["vision_tower", "language_model", "llm_connector", "dit", "action_dit"]

            for module_name in lora_module_names:
                if hasattr(base_model, module_name):
                    sub_module = getattr(base_model, module_name)
                    # 检查是否为 PeftModel
                    if isinstance(sub_module, PeftModel):
                        # 构造保存路径，例如: output_dir/checkpoint-500/lora_adapters/vision_tower
                        adapter_save_dir = os.path.join(output_dir, "lora_adapters", module_name)
                        os.makedirs(adapter_save_dir, exist_ok=True)
                        logging.info(f"💾 Saving LoRA adapter for {module_name} to {adapter_save_dir}")
                        sub_module.save_pretrained(adapter_save_dir)

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def load_checkpoint_state_dict(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load_file
        state_dict = safe_load_file(ckpt_path, device="cpu")
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    normalized_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod."):]
        normalized_state_dict[new_key] = value
    return normalized_state_dict


def extract_checkpoint_tag(ckpt_path: str) -> str:
    ckpt_path = Path(ckpt_path)
    ckpt_dir = ckpt_path.parent.name if ckpt_path.parent.name else ckpt_path.stem
    exp_dir = ckpt_path.parent.parent.name if ckpt_path.parent.parent.name else "unknown_exp"
    return f"{exp_dir}_{ckpt_dir}"


def append_init_tags_to_output_dir(output_dir: str, base_init_ckpt_path=None, gen_init_ckpt_path=None, loc_init_ckpt_path=None) -> str:
    suffixes = []
    if base_init_ckpt_path:
        suffixes.append(f"initbase_{extract_checkpoint_tag(base_init_ckpt_path)}")
    if gen_init_ckpt_path:
        suffixes.append(f"initgen_{extract_checkpoint_tag(gen_init_ckpt_path)}")
    if loc_init_ckpt_path:
        suffixes.append(f"initloc_{extract_checkpoint_tag(loc_init_ckpt_path)}")
    if not suffixes:
        return output_dir
    return f"{output_dir}_{'_'.join(suffixes)}"


def key_matches_prefix(key: str, prefix: str) -> bool:
    return key == prefix or key.startswith(prefix + ".")


def build_generation_init_prefixes(model_config) -> List[str]:
    return [
        "model.latent_queries",
        "model.llm_connector",
        "model.projector",
        "model.dit",
        "model.vae_decoder",
    ]


def build_localization_init_prefixes(model_config) -> List[str]:
    if getattr(model_config, "use_external_loc_model", False):
        return []

    prefixes = []

    if getattr(model_config, "use_codex_vit_regression_head", False):
        prefixes.extend([
            "model.vit_loc_fusion",
            "model.regression_loc_head",
            "model.action_dit_norm",
        ])
        if getattr(model_config, "is_action_dit_projector", False):
            prefixes.append("model.action_dit_projector")
        return prefixes

    if getattr(model_config, "use_vit_cls_regression_head", False):
        prefixes.extend([
            "model.regression_loc_head",
            "model.action_dit_norm",
        ])
        if getattr(model_config, "is_action_dit_projector", False):
            prefixes.append("model.action_dit_projector")
        return prefixes

    if getattr(model_config, "use_vit_regression_head", False):
        prefixes.extend([
            "model.cross_view_fusion",
            "model.regression_loc_head",
            "model.action_dit_norm",
        ])
        if getattr(model_config, "is_action_dit_projector", False):
            prefixes.append("model.action_dit_projector")
        return prefixes

    if getattr(model_config, "use_pi05_action_dit", False):
        prefixes.append("model.action_dit_connector")

    prefixes.extend([
        "model.action_dit_norm",
        "model.action_dit",
        "model.action_in_proj",
        "model.action_out_proj",
        "model.time_mlp_in",
        "model.time_mlp_out",
    ])

    if getattr(model_config, "is_action_dit_projector", False):
        prefixes.append("model.action_dit_projector")
    if getattr(model_config, "is_loc_learnable_query", False):
        prefixes.append("model.loc_learnable_query")

    return prefixes


def load_partial_checkpoint(model, ckpt_path: str, prefixes: List[str], tag: str):
    if not prefixes:
        raise RuntimeError(f"No prefixes configured for partial checkpoint load: {tag}")

    src_state_dict = load_checkpoint_state_dict(ckpt_path)
    src_state_dict = smart_matching_state_dict_keys(src_state_dict, model)
    target_state_dict = model.state_dict()

    filtered_state_dict = {}
    matched_keys = []
    missing_target_keys = []
    shape_mismatch_keys = []

    for key, value in src_state_dict.items():
        if not any(key_matches_prefix(key, prefix) for prefix in prefixes):
            continue
        matched_keys.append(key)
        if key not in target_state_dict:
            missing_target_keys.append(key)
            continue
        if tuple(target_state_dict[key].shape) != tuple(value.shape):
            shape_mismatch_keys.append((key, tuple(value.shape), tuple(target_state_dict[key].shape)))
            continue
        filtered_state_dict[key] = value

    if len(filtered_state_dict) == 0:
        raise RuntimeError(
            f"Partial checkpoint load for {tag} produced 0 loadable keys from {ckpt_path}. "
            f"Matched={len(matched_keys)}, missing_target={len(missing_target_keys)}, shape_mismatch={len(shape_mismatch_keys)}"
        )

    msg = model.load_state_dict(filtered_state_dict, strict=False)
    logging.info(
        "Loaded partial checkpoint for %s from %s: prefixes=%s, matched=%d, loaded=%d, missing_target=%d, shape_mismatch=%d",
        tag,
        ckpt_path,
        prefixes,
        len(matched_keys),
        len(filtered_state_dict),
        len(missing_target_keys),
        len(shape_mismatch_keys),
    )
    if missing_target_keys:
        logging.info("Partial checkpoint missing target keys for %s (showing up to 10): %s", tag, missing_target_keys[:10])
    if shape_mismatch_keys:
        logging.info("Partial checkpoint shape mismatches for %s (showing up to 10): %s", tag, shape_mismatch_keys[:10])
    logging.info(msg)


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



def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments, img_size: int = 448) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
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
                if self.config.img_size==224:
                    data_dict["und_image"] = process_images[:-1] # [1, C, H, W]
                else:
                    data_dict["und_image"] = img_resize_transform(process_images[:-1]) # [1, C, H, W]
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


class UniLIPLogCallback(TrainerCallback):
    def __init__(self, trainer=None):
        self.trainer = trainer
        self.latest_timing_metrics = {}

    def bind_trainer(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer is None or not getattr(args, "enable_step_timing", False):
            return control

        timing_metrics = self.trainer.consume_step_timing_metrics()
        if timing_metrics:
            self.latest_timing_metrics = {
                "step_total_time": round(timing_metrics["step_total_time"], 6),
                "batch_load_time": round(timing_metrics["batch_load_time"], 6),
                "prepare_inputs_time": round(timing_metrics["prepare_inputs_time"], 6),
                "forward_time": round(timing_metrics["forward_time"], 6),
                "backward_time": round(timing_metrics["backward_time"], 6),
                "other_iteration_time": round(timing_metrics["other_iteration_time"], 6),
            }
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        每当达到 logging_steps 时触发。
        logs 字典通常包含: {'loss': ..., 'learning_rate': ..., 'epoch': ...}
        """
        if state.is_local_process_zero and logs is not None:
            if self.latest_timing_metrics:
                logs.update(self.latest_timing_metrics)
            # 格式化打印
            log_msg = f"[ Step {state.global_step} ] "

            # 打印 Loss
            if "loss" in logs:
                log_msg += f"Loss: {logs['loss']:.6f} | "

            # 打印 LR
            if "learning_rate" in logs:
                log_msg += f"LR: {logs['learning_rate']:.4e} | "

            # 打印 Epoch
            if "epoch" in logs:
                log_msg += f"Epoch: {logs['epoch']:.2f}"

            # 如果你有自定义的 loss (需要配合修改 Trainer.compute_loss 才能看到)
            if "other_info" in logs:
                log_msg += f" | Other Info: {logs['other_info']}"

            if "step_total_time" in logs:
                log_msg += (
                    f" | Timing(total={logs['step_total_time']:.4f}s"
                    f", batch={logs['batch_load_time']:.4f}s"
                    f", prepare={logs['prepare_inputs_time']:.4f}s"
                    f", forward={logs['forward_time']:.4f}s"
                    f", backward={logs['backward_time']:.4f}s"
                    f", other={logs['other_iteration_time']:.4f}s)"
                )

            # 强制打印到控制台
            logging.info(log_msg)
            self.latest_timing_metrics = {}
            # 或者直接 print，防止 logging 被过滤
            # print(log_msg, flush=True)


def _piecewise_linear_value(step, steps, values):
    if len(steps) != len(values) or len(steps) == 0:
        raise ValueError("alpha_loc_aux schedule steps/values must have the same non-zero length")

    if step <= steps[0]:
        return float(values[0])

    for idx in range(1, len(steps)):
        if step <= steps[idx]:
            s0, s1 = steps[idx - 1], steps[idx]
            v0, v1 = values[idx - 1], values[idx]
            if s1 == s0:
                return float(v1)
            ratio = (step - s0) / float(s1 - s0)
            return float(v0 + ratio * (v1 - v0))

    return float(values[-1])


def _periodic_binary_gate(step, cycle_steps, on_steps, start_step=0):
    if cycle_steps <= 0:
        raise ValueError("cycle_steps must be > 0")
    if on_steps < 0 or on_steps > cycle_steps:
        raise ValueError("on_steps must be in [0, cycle_steps]")
    if start_step < 0:
        raise ValueError("start_step must be >= 0")

    if step < start_step:
        return 1.0

    phase = (step - start_step) % cycle_steps
    return 1.0 if phase < on_steps else 0.0


class AlphaLocAuxControlCallback(TrainerCallback):
    def __init__(
        self,
        steps=None,
        values=None,
        use_step_gate=False,
        gate_cycle_steps=0,
        gate_on_steps=0,
        gate_start_step=0,
        base_alpha=1.0,
    ):
        self.steps = [int(x) for x in steps] if steps is not None else None
        self.values = [float(x) for x in values] if values is not None else None
        self.use_step_gate = bool(use_step_gate)
        self.gate_cycle_steps = int(gate_cycle_steps)
        self.gate_on_steps = int(gate_on_steps)
        self.gate_start_step = int(gate_start_step)
        self.base_alpha = float(base_alpha)

        if self.steps is not None:
            if len(self.steps) != len(self.values) or len(self.steps) == 0:
                raise ValueError("alpha_loc_aux schedule steps/values must have the same non-zero length")
            if any(self.steps[idx] < self.steps[idx - 1] for idx in range(1, len(self.steps))):
                raise ValueError("alpha_loc_aux schedule steps must be non-decreasing")

        if self.use_step_gate:
            if self.gate_cycle_steps <= 0:
                raise ValueError("loc_aux_gate_cycle_steps must be > 0 when step gate is enabled.")
            if self.gate_on_steps < 0 or self.gate_on_steps > self.gate_cycle_steps:
                raise ValueError("loc_aux_gate_on_steps must be in [0, loc_aux_gate_cycle_steps].")
            if self.gate_start_step < 0:
                raise ValueError("loc_aux_gate_start_step must be >= 0.")

    def _compute_alpha(self, step):
        if self.steps is not None:
            alpha = _piecewise_linear_value(step, self.steps, self.values)
        else:
            alpha = self.base_alpha

        if self.use_step_gate:
            alpha *= _periodic_binary_gate(
                step=step,
                cycle_steps=self.gate_cycle_steps,
                on_steps=self.gate_on_steps,
                start_step=self.gate_start_step,
            )
        return float(alpha)

    def _set_alpha(self, model, step):
        alpha = self._compute_alpha(step)

        candidates = [model]
        if hasattr(model, "module"):
            candidates.append(model.module)
        if hasattr(model, "model"):
            candidates.append(model.model)
        if hasattr(model, "module") and hasattr(model.module, "model"):
            candidates.append(model.module.model)

        for candidate in candidates:
            if hasattr(candidate, "config"):
                candidate.config.alpha_loc_aux_loss = alpha

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._set_alpha(model, 0)
        return control

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._set_alpha(model, state.global_step)
        return control


class AlternatingAuxLossControlCallback(TrainerCallback):
    def __init__(
        self,
        loc_steps=None,
        loc_values=None,
        gen_steps=None,
        gen_values=None,
        base_loc_alpha=1.0,
        base_gen_alpha=1.0,
        use_alternating=True,
        cycle_steps=2,
        loc_active_phase=0,
        gen_active_phase=1,
        start_step=0,
    ):
        self.loc_steps = [int(x) for x in loc_steps] if loc_steps is not None else None
        self.loc_values = [float(x) for x in loc_values] if loc_values is not None else None
        self.gen_steps = [int(x) for x in gen_steps] if gen_steps is not None else None
        self.gen_values = [float(x) for x in gen_values] if gen_values is not None else None
        self.base_loc_alpha = float(base_loc_alpha)
        self.base_gen_alpha = float(base_gen_alpha)
        self.use_alternating = bool(use_alternating)
        self.cycle_steps = int(cycle_steps)
        self.loc_active_phase = int(loc_active_phase)
        self.gen_active_phase = int(gen_active_phase)
        self.start_step = int(start_step)

        for name, steps, values in [
            ("alpha_loc_aux", self.loc_steps, self.loc_values),
            ("alpha_gen_aux", self.gen_steps, self.gen_values),
        ]:
            if steps is None:
                continue
            if values is None or len(steps) != len(values) or len(steps) == 0:
                raise ValueError(f"{name} schedule steps/values must have the same non-zero length")
            if any(steps[idx] < steps[idx - 1] for idx in range(1, len(steps))):
                raise ValueError(f"{name} schedule steps must be non-decreasing")

        if self.use_alternating:
            if self.cycle_steps <= 0:
                raise ValueError("aux_loss_alternate_cycle_steps must be > 0.")
            if self.start_step < 0:
                raise ValueError("aux_loss_alternate_start_step must be >= 0.")
            if not (0 <= self.loc_active_phase < self.cycle_steps):
                raise ValueError("aux_loc_active_phase must be in [0, aux_loss_alternate_cycle_steps).")
            if not (0 <= self.gen_active_phase < self.cycle_steps):
                raise ValueError("aux_gen_active_phase must be in [0, aux_loss_alternate_cycle_steps).")

    def _scheduled_value(self, step, steps, values, base_alpha):
        if steps is None:
            return float(base_alpha)
        return _piecewise_linear_value(step, steps, values)

    def _compute_alphas(self, step):
        loc_alpha = self._scheduled_value(step, self.loc_steps, self.loc_values, self.base_loc_alpha)
        gen_alpha = self._scheduled_value(step, self.gen_steps, self.gen_values, self.base_gen_alpha)

        if self.use_alternating and step >= self.start_step:
            phase = (step - self.start_step) % self.cycle_steps
            loc_alpha = loc_alpha if phase == self.loc_active_phase else 0.0
            gen_alpha = gen_alpha if phase == self.gen_active_phase else 0.0
        return float(loc_alpha), float(gen_alpha)

    def _set_alphas(self, model, step):
        loc_alpha, gen_alpha = self._compute_alphas(step)

        candidates = [model]
        if hasattr(model, "module"):
            candidates.append(model.module)
        if hasattr(model, "model"):
            candidates.append(model.model)
        if hasattr(model, "module") and hasattr(model.module, "model"):
            candidates.append(model.module.model)

        for candidate in candidates:
            if hasattr(candidate, "config"):
                candidate.config.alpha_loc_aux_loss = loc_alpha
                candidate.config.alpha_gen_aux_loss = gen_alpha

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._set_alphas(model, 0)
        return control

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._set_alphas(model, state.global_step)
        return control


class AlphaLocScheduleCallback(TrainerCallback):
    def __init__(self, steps, values):
        self.steps = [int(x) for x in steps]
        self.values = [float(x) for x in values]

        if len(self.steps) != len(self.values) or len(self.steps) == 0:
            raise ValueError("alpha_loc schedule steps/values must have the same non-zero length")
        if any(self.steps[idx] < self.steps[idx - 1] for idx in range(1, len(self.steps))):
            raise ValueError("alpha_loc schedule steps must be non-decreasing")

    def _set_alpha(self, model, step):
        alpha = _piecewise_linear_value(step, self.steps, self.values)

        candidates = [model]
        if hasattr(model, "module"):
            candidates.append(model.module)
        if hasattr(model, "model"):
            candidates.append(model.model)
        if hasattr(model, "module") and hasattr(model.module, "model"):
            candidates.append(model.module.model)

        for candidate in candidates:
            if hasattr(candidate, "config"):
                candidate.config.alpha_loc_loss = alpha

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._set_alpha(model, 0)
        return control

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._set_alpha(model, state.global_step)
        return control

def smart_matching_state_dict_keys(state_dict, model):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("language_model.lm_head."):
            new_k = k[len("language_model."):]
        elif k.startswith("language_model.model."):
            new_k =  "model.language_model." + k[len("language_model.model."):]
        elif k.startswith("vision_tower."):
            new_k = "model." + k
        elif k.startswith("multi_modal_projector."):
            new_k = "model." + k
        else:
            new_k = k
        new_state_dict[new_k] = v

    logging.info(f"replace language_model.lm_head. to language_model.")
    logging.info(f"replace language_model.model. to model.language_model.")
    logging.info(f"replace vision_tower. to model.vision_tower.")
    logging.info(f"replace multi_modal_projector. to model.multi_modal_projector.")
    return new_state_dict

def train(attn_implementation=None):

    global local_rank

    set_seed()

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.deepspeed="deepspeed_scripts/zero0.json"
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


    if training_args.dataloader_num_workers == 0:
        auto_workers = max(4, min(16, (os.cpu_count() or 8) // 2))
        training_args.dataloader_num_workers = auto_workers
        logging.info(f"Auto setting dataloader_num_workers to {training_args.dataloader_num_workers} based on CPU count.")
    if hasattr(training_args, "dataloader_persistent_workers"):
        training_args.dataloader_persistent_workers = training_args.dataloader_num_workers > 0
    if hasattr(training_args, "dataloader_prefetch_factor") and training_args.dataloader_num_workers > 0:
        training_args.dataloader_prefetch_factor = csgo_config.get("dataloader_prefetch_factor", 4)
    if hasattr(training_args, "dataloader_pin_memory"):
        training_args.dataloader_pin_memory = csgo_config.get("dataloader_pin_memory", True)

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
                    llm_int8_skip_modules=["mm_projector", "multi_modal_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    if csgo_config.get("is_multi_task"):
        logging.info(f"Start Unified_UniLIP_InternVLForCausalLM.from_pretrained: {model_args.model_name_or_path}")
        model = Unified_UniLIP_InternVLForCausalLM.from_pretrained(
            model_args.model_name_or_path, # UniLIP-1B with new unified_unilip config
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args,
        )
        logging.info(f"Finish Unified_UniLIP_InternVLForCausalLM.from_pretrained: {model_args.model_name_or_path}")
    else:
        logging.info(f"Start UniLIP_InternVLForCausalLM.from_pretrained: {model_args.model_name_or_path}")
        model = UniLIP_InternVLForCausalLM.from_pretrained(
            model_args.model_name_or_path, #UniLIP-1B
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args,
        )
        logging.info(f"Finish UniLIP_InternVLForCausalLM.from_pretrained: {model_args.model_name_or_path}")
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

    model.config.img_size = model_args.img_size = csgo_config.get("img_size", False)
    model.config.is_action_dit_dense_timestep = model_args.is_action_dit_dense_timestep = csgo_config.get("is_action_dit_dense_timestep", False)

    model.config.use_vit_regression_head = csgo_config.get("use_vit_regression_head", False)
    model.config.use_vit_cls_regression_head = csgo_config.get("use_vit_cls_regression_head", False)
    model.config.use_codex_vit_regression_head = csgo_config.get("use_codex_vit_regression_head", False)
    model.config.use_pi05_action_dit = csgo_config.get("use_pi05_action_dit", False)
    model.config.pi05_pytorch_weight_path = csgo_config.get("pi05_pytorch_weight_path", False)
    model.config.is_loc_aux_loss = csgo_config.get("is_loc_aux_loss", False)
    model.config.alpha_loc_aux_loss = csgo_config.get("alpha_loc_aux_loss", 1.0)
    model.config.is_loc_aux_step_gate = csgo_config.get("is_loc_aux_step_gate", False)
    model.config.loc_aux_gate_cycle_steps = int(csgo_config.get("loc_aux_gate_cycle_steps", 0))
    model.config.loc_aux_gate_on_steps = int(csgo_config.get("loc_aux_gate_on_steps", 0))
    model.config.loc_aux_gate_start_step = int(csgo_config.get("loc_aux_gate_start_step", 0))
    model.config.is_aux_loc_em_loss = csgo_config.get("is_aux_loc_em_loss", False)
    model.config.aux_loc_em_num_samples = int(csgo_config.get("aux_loc_em_num_samples", 1))
    model.config.aux_loc_em_weight_mode = csgo_config.get("aux_loc_em_weight_mode", "softmax_loss")
    model.config.aux_loc_em_candidate_tau = float(csgo_config.get("aux_loc_em_candidate_tau", 1.0))
    model.config.aux_loc_em_backward_mode = csgo_config.get("aux_loc_em_backward_mode", "weighted_all")
    model.config.aux_loc_em_share_loc_noise = csgo_config.get("aux_loc_em_share_loc_noise", True)
    model.config.is_aux_loc_uncertainty_loss = csgo_config.get("is_aux_loc_uncertainty_loss", False)
    model.config.aux_loc_unc_num_samples = int(csgo_config.get("aux_loc_unc_num_samples", 2))
    model.config.aux_loc_unc_metric = csgo_config.get("aux_loc_unc_metric", "residual_l1_normed")
    model.config.aux_loc_unc_tau = float(csgo_config.get("aux_loc_unc_tau", 1.0))
    model.config.aux_loc_unc_min_weight = float(csgo_config.get("aux_loc_unc_min_weight", 0.05))
    model.config.aux_loc_unc_share_loc_noise = csgo_config.get("aux_loc_unc_share_loc_noise", True)
    model.config.aux_loc_unc_eps = float(csgo_config.get("aux_loc_unc_eps", 1e-6))
    model.config.is_aux_loc_combined_em_unc_loss = csgo_config.get("is_aux_loc_combined_em_unc_loss", False)
    model.config.aux_loc_combined_num_samples = int(csgo_config.get("aux_loc_combined_num_samples", 2))
    model.config.aux_loc_combined_candidate_tau = float(csgo_config.get("aux_loc_combined_candidate_tau", 1.0))
    model.config.aux_loc_combined_unc_metric = csgo_config.get("aux_loc_combined_unc_metric", "residual_l1_normed")
    model.config.aux_loc_combined_unc_tau = float(csgo_config.get("aux_loc_combined_unc_tau", 1.0))
    model.config.aux_loc_combined_unc_min_weight = float(csgo_config.get("aux_loc_combined_unc_min_weight", 0.05))
    model.config.aux_loc_combined_share_loc_noise = csgo_config.get("aux_loc_combined_share_loc_noise", True)
    model.config.aux_loc_combined_unc_eps = float(csgo_config.get("aux_loc_combined_unc_eps", 1e-6))
    model.config.is_gen_aux_loss = csgo_config.get("is_gen_aux_loss", False)
    model.config.alpha_gen_aux_loss = csgo_config.get("alpha_gen_aux_loss", 0.0)
    model.config.alpha_gen_aux_schedule_steps = csgo_config.get("alpha_gen_aux_schedule_steps", None)
    model.config.alpha_gen_aux_schedule_values = csgo_config.get("alpha_gen_aux_schedule_values", None)
    model.config.is_aux_loss_alternating = csgo_config.get("is_aux_loss_alternating", False)
    model.config.aux_loss_alternate_cycle_steps = int(csgo_config.get("aux_loss_alternate_cycle_steps", 2))
    model.config.aux_loc_active_phase = int(csgo_config.get("aux_loc_active_phase", 0))
    model.config.aux_gen_active_phase = int(csgo_config.get("aux_gen_active_phase", 1))
    model.config.aux_loss_alternate_start_step = int(csgo_config.get("aux_loss_alternate_start_step", 0))
    model.config.aux_gen_pose_condition_type = csgo_config.get("aux_gen_pose_condition_type", "pose_token_mlp")
    model.config.aux_gen_pose_condition_mode = csgo_config.get("aux_gen_pose_condition_mode", "delta_from_gt")
    model.config.aux_gen_freeze_gen_head = csgo_config.get("aux_gen_freeze_gen_head", True)
    model.config.aux_gen_update_scope = csgo_config.get("aux_gen_update_scope", "loc_head_only")
    model.config.aux_gen_use_loc_samples = csgo_config.get("aux_gen_use_loc_samples", True)
    model.config.aux_gen_pose_projector_trainable = csgo_config.get("aux_gen_pose_projector_trainable", False)
    model.config.aux_gen_pose_injection_scale = float(csgo_config.get("aux_gen_pose_injection_scale", 1.0))
    model.config.is_repa_loss = csgo_config.get("is_repa_loss", False)
    model.config.alpha_repa_loss = csgo_config.get("alpha_repa_loss", 0.0)
    model.config.repa_teacher_type = csgo_config.get("repa_teacher_type", "dinov2")
    model.config.repa_teacher_name_or_path = csgo_config.get("repa_teacher_name_or_path", "facebook/dinov2-base")
    model.config.repa_teacher_input_size = int(csgo_config.get("repa_teacher_input_size", 224))
    model.config.repa_teacher_hidden_size = int(csgo_config.get("repa_teacher_hidden_size", 768))
    model.config.repa_dit_layer_idx = int(csgo_config.get("repa_dit_layer_idx", 6))
    model.config.repa_align_type = csgo_config.get("repa_align_type", "patch_wise")
    model.config.repa_expected_num_patches = int(csgo_config.get("repa_expected_num_patches", 256))
    model.config.repa_projector_type = csgo_config.get("repa_projector_type", "mlp3_silu")
    model.config.repa_mlp_num_layers = int(csgo_config.get("repa_mlp_num_layers", 3))
    model.config.repa_mlp_activation = csgo_config.get("repa_mlp_activation", "silu")
    model.config.repa_mlp_hidden_ratio = float(csgo_config.get("repa_mlp_hidden_ratio", 1.0))
    model.config.repa_use_spatial_norm = csgo_config.get("repa_use_spatial_norm", False)
    model.config.repa_conv_kernel_size = int(csgo_config.get("repa_conv_kernel_size", 3))
    model.config.repa_spatial_norm_gamma = float(csgo_config.get("repa_spatial_norm_gamma", 1.0))
    model.config.repa_detach_condition = csgo_config.get("repa_detach_condition", True)
    model.config.is_loc_repa_loss = csgo_config.get("is_loc_repa_loss", False)
    model.config.alpha_loc_repa_loss = csgo_config.get("alpha_loc_repa_loss", 0.0)
    model.config.loc_repa_teacher_ckpt_path = csgo_config.get("loc_repa_teacher_ckpt_path", None)
    model.config.loc_repa_feature_type = csgo_config.get("loc_repa_feature_type", "action_prefix_tokens")
    model.config.loc_repa_loss_type = csgo_config.get("loc_repa_loss_type", "cosine")
    model.config.loc_repa_use_und_tokens_only = csgo_config.get("loc_repa_use_und_tokens_only", True)
    model.config.loc_repa_timestep_weight = csgo_config.get("loc_repa_timestep_weight", "linear_1m_sigma")
    model.config.is_noisy_loc_loss = csgo_config.get("is_noisy_loc_loss", False)
    model.config.noisy_loc_ratio = float(csgo_config.get("noisy_loc_ratio", 0.0))
    model.config.noisy_loc_image_source = csgo_config.get("noisy_loc_image_source", "latent_space")
    model.config.noisy_loc_sigma_sampling = csgo_config.get("noisy_loc_sigma_sampling", "gen_matched")
    model.config.noisy_loc_weight_type = csgo_config.get("noisy_loc_weight_type", "linear_1m_sigma")
    model.config.alpha_loc_loss = csgo_config.get("alpha_loc_loss", 1.0)
    model.config.alpha_loc_schedule_steps = csgo_config.get("alpha_loc_schedule_steps", None)
    model.config.alpha_loc_schedule_values = csgo_config.get("alpha_loc_schedule_values", None)
    model.config.alpha_loc_aux_schedule_steps = csgo_config.get("alpha_loc_aux_schedule_steps", None)
    model.config.alpha_loc_aux_schedule_values = csgo_config.get("alpha_loc_aux_schedule_values", None)
    model.config.is_aciton_dit_vae_small_init = csgo_config.get("is_aciton_dit_vae_small_init", 5e-4)
    model.config.loc_use_circular_loss = csgo_config.get("loc_use_circular_loss", True)
    model.config.loc_xy_loss_weight = csgo_config.get("loc_xy_loss_weight", 1.0)
    model.config.loc_z_loss_weight = csgo_config.get("loc_z_loss_weight", 1.0)
    model.config.loc_angle_loss_weight = csgo_config.get("loc_angle_loss_weight", 2.0)
    model.config.use_external_loc_model = csgo_config.get("use_external_loc_model", False)
    model.config.external_loc_input_size = csgo_config.get("external_loc_input_size", 224)
    model.config.external_loc_use_circular_loss = csgo_config.get("external_loc_use_circular_loss", True)
    model.config.base_init_ckpt_path = csgo_config.get("base_init_ckpt_path", None)
    model.config.gen_init_ckpt_path = csgo_config.get("gen_init_ckpt_path", None)
    model.config.loc_init_ckpt_path = csgo_config.get("loc_init_ckpt_path", None)
    model_args.train_mm_projector_only = csgo_config.get(
        "train_mm_projector_only",
        getattr(model_args, "train_mm_projector_only", False),
    )
    model.config.train_mm_projector_only = model_args.train_mm_projector_only
    model_args.train_shared_llm_tail_only = csgo_config.get(
        "train_shared_llm_tail_only",
        getattr(model_args, "train_shared_llm_tail_only", False),
    )
    model.config.train_shared_llm_tail_only = model_args.train_shared_llm_tail_only
    model_args.shared_llm_tail_num_layers = int(csgo_config.get(
        "shared_llm_tail_num_layers",
        getattr(model_args, "shared_llm_tail_num_layers", 2),
    ))
    model.config.shared_llm_tail_num_layers = model_args.shared_llm_tail_num_layers
    model_args.shared_llm_tail_lora_enabled = csgo_config.get(
        "shared_llm_tail_lora_enabled",
        getattr(model_args, "shared_llm_tail_lora_enabled", False),
    )
    model.config.shared_llm_tail_lora_enabled = model_args.shared_llm_tail_lora_enabled
    model_args.shared_llm_tail_lora_mode = csgo_config.get(
        "shared_llm_tail_lora_mode",
        getattr(model_args, "shared_llm_tail_lora_mode", "lora_only"),
    )
    model.config.shared_llm_tail_lora_mode = model_args.shared_llm_tail_lora_mode

    model.config.is_action_dit_projector =  training_args.is_action_dit_projector = csgo_config.get("is_action_dit_projector", False)
    model.config.action_dit_projector_lr =  training_args.action_dit_projector_lr = csgo_config.get("action_dit_projector_lr", 1e-3)
    model.config.action_dit_lr = training_args.action_dit_lr = csgo_config.get("action_dit_lr", training_args.learning_rate)
    model.config.mm_projector_lr = training_args.mm_projector_lr = csgo_config.get(
        "mm_projector_lr",
        training_args.mm_projector_lr,
    )
    model.config.shared_llm_tail_lr = training_args.shared_llm_tail_lr = csgo_config.get(
        "shared_llm_tail_lr",
        training_args.shared_llm_tail_lr,
    )
    model.config.shared_llm_tail_lora_r = training_args.shared_llm_tail_lora_r = int(csgo_config.get(
        "shared_llm_tail_lora_r",
        training_args.shared_llm_tail_lora_r,
    ))
    model.config.shared_llm_tail_lora_alpha = training_args.shared_llm_tail_lora_alpha = int(csgo_config.get(
        "shared_llm_tail_lora_alpha",
        training_args.shared_llm_tail_lora_alpha,
    ))
    model.config.shared_llm_tail_lora_dropout = training_args.shared_llm_tail_lora_dropout = float(csgo_config.get(
        "shared_llm_tail_lora_dropout",
        training_args.shared_llm_tail_lora_dropout,
    ))
    model.config.shared_llm_tail_lora_lr = training_args.shared_llm_tail_lora_lr = csgo_config.get(
        "shared_llm_tail_lora_lr",
        training_args.shared_llm_tail_lora_lr,
    )

    if model_args.shared_llm_tail_lora_enabled:
        if not model_args.train_shared_llm_tail_only:
            raise ValueError("shared_llm_tail_lora_enabled=True requires train_shared_llm_tail_only=True.")
        if model_args.shared_llm_tail_num_layers <= 6:
            raise ValueError("shared_llm_tail_lora_enabled=True requires shared_llm_tail_num_layers > 6.")
        if model_args.shared_llm_tail_lora_mode != "lora_only":
            raise ValueError("This version only supports shared_llm_tail_lora_mode='lora_only'.")
        if training_args.shared_llm_tail_lora_lr is None:
            raise ValueError("shared_llm_tail_lora_enabled=True requires shared_llm_tail_lora_lr.")
        if not model_args.fix_llm:
            raise ValueError("shared_llm_tail_lora_enabled=True is incompatible with fix_llm=False (whole-LLM finetuning/LoRA).")
    model.config.is_loc_learnable_query =  training_args.is_loc_learnable_query = csgo_config.get("is_loc_learnable_query", False)
    model.config.loc_learnable_query_lr =  training_args.loc_learnable_query_lr = csgo_config.get("loc_learnable_query_lr", 5e-4)
    model.config.is_lora = training_args.is_lora = csgo_config.get("is_lora", False)

    model_args.gradient_checkpointing = training_args.gradient_checkpointing

    if model.config.is_loc_aux_step_gate:
        if model.config.loc_aux_gate_cycle_steps <= 0:
            raise ValueError("loc_aux_gate_cycle_steps must be > 0 when is_loc_aux_step_gate=True.")
        if model.config.loc_aux_gate_on_steps < 0 or model.config.loc_aux_gate_on_steps > model.config.loc_aux_gate_cycle_steps:
            raise ValueError("loc_aux_gate_on_steps must be in [0, loc_aux_gate_cycle_steps].")
        if model.config.loc_aux_gate_start_step < 0:
            raise ValueError("loc_aux_gate_start_step must be >= 0.")

    if model.config.is_noisy_loc_loss:
        if not (0.0 <= model.config.noisy_loc_ratio <= 1.0):
            raise ValueError("noisy_loc_ratio must be in [0, 1] when is_noisy_loc_loss=True.")
        if model.config.noisy_loc_image_source != "latent_space":
            raise ValueError("Current implementation only supports noisy_loc_image_source='latent_space'.")
        if model.config.noisy_loc_sigma_sampling != "gen_matched":
            raise ValueError("Current implementation only supports noisy_loc_sigma_sampling='gen_matched'.")
        if model.config.noisy_loc_weight_type != "linear_1m_sigma":
            raise ValueError("Current implementation only supports noisy_loc_weight_type='linear_1m_sigma'.")

    aux_loc_multisample_modes = [
        bool(model.config.is_aux_loc_em_loss),
        bool(model.config.is_aux_loc_uncertainty_loss),
        bool(model.config.is_aux_loc_combined_em_unc_loss),
    ]
    if sum(aux_loc_multisample_modes) > 1:
        raise ValueError(
            "Only one aux-loc multi-sample mode can be enabled: "
            "is_aux_loc_em_loss, is_aux_loc_uncertainty_loss, is_aux_loc_combined_em_unc_loss."
        )

    if model.config.is_aux_loc_em_loss:
        if not model.config.is_loc_aux_loss:
            raise ValueError("is_aux_loc_em_loss=True requires is_loc_aux_loss=True.")
        if model.config.aux_loc_em_num_samples < 1:
            raise ValueError("aux_loc_em_num_samples must be >= 1 when is_aux_loc_em_loss=True.")
        if model.config.aux_loc_em_weight_mode != "softmax_loss":
            raise ValueError("Current implementation only supports aux_loc_em_weight_mode='softmax_loss'.")
        if model.config.aux_loc_em_candidate_tau <= 0.0:
            raise ValueError("aux_loc_em_candidate_tau must be > 0 when is_aux_loc_em_loss=True.")
        if model.config.aux_loc_em_backward_mode != "weighted_all":
            raise ValueError("Current implementation only supports aux_loc_em_backward_mode='weighted_all'.")
        if model.config.use_external_loc_model:
            raise ValueError("is_aux_loc_em_loss=True currently supports only the internal action-DiT loc branch.")
        if (
            model.config.use_vit_regression_head
            or model.config.use_vit_cls_regression_head
            or model.config.use_codex_vit_regression_head
        ):
            raise ValueError("is_aux_loc_em_loss=True currently supports only the action-DiT loc branch.")

    if model.config.is_aux_loc_uncertainty_loss:
        if not model.config.is_loc_aux_loss:
            raise ValueError("is_aux_loc_uncertainty_loss=True requires is_loc_aux_loss=True.")
        if model.config.aux_loc_unc_num_samples != 2:
            raise ValueError("Current implementation requires aux_loc_unc_num_samples=2.")
        if model.config.aux_loc_unc_metric != "residual_l1_normed":
            raise ValueError("Current implementation only supports aux_loc_unc_metric='residual_l1_normed'.")
        if model.config.aux_loc_unc_tau <= 0.0:
            raise ValueError("aux_loc_unc_tau must be > 0 when is_aux_loc_uncertainty_loss=True.")
        if not (0.0 <= model.config.aux_loc_unc_min_weight <= 1.0):
            raise ValueError("aux_loc_unc_min_weight must be in [0, 1].")
        if not model.config.aux_loc_unc_share_loc_noise:
            raise ValueError("Current implementation requires aux_loc_unc_share_loc_noise=True.")
        if model.config.aux_loc_unc_eps <= 0.0:
            raise ValueError("aux_loc_unc_eps must be > 0 when is_aux_loc_uncertainty_loss=True.")
        if model.config.use_external_loc_model:
            raise ValueError("is_aux_loc_uncertainty_loss=True currently supports only the internal action-DiT loc branch.")
        if (
            model.config.use_vit_regression_head
            or model.config.use_vit_cls_regression_head
            or model.config.use_codex_vit_regression_head
        ):
            raise ValueError("is_aux_loc_uncertainty_loss=True currently supports only the action-DiT loc branch.")

    if model.config.is_aux_loc_combined_em_unc_loss:
        if not model.config.is_loc_aux_loss:
            raise ValueError("is_aux_loc_combined_em_unc_loss=True requires is_loc_aux_loss=True.")
        if model.config.aux_loc_combined_num_samples != 2:
            raise ValueError("Current implementation requires aux_loc_combined_num_samples=2.")
        if model.config.aux_loc_combined_candidate_tau <= 0.0:
            raise ValueError("aux_loc_combined_candidate_tau must be > 0 when is_aux_loc_combined_em_unc_loss=True.")
        if model.config.aux_loc_combined_unc_metric != "residual_l1_normed":
            raise ValueError("Current implementation only supports aux_loc_combined_unc_metric='residual_l1_normed'.")
        if model.config.aux_loc_combined_unc_tau <= 0.0:
            raise ValueError("aux_loc_combined_unc_tau must be > 0 when is_aux_loc_combined_em_unc_loss=True.")
        if not (0.0 <= model.config.aux_loc_combined_unc_min_weight <= 1.0):
            raise ValueError("aux_loc_combined_unc_min_weight must be in [0, 1].")
        if not model.config.aux_loc_combined_share_loc_noise:
            raise ValueError("Current implementation requires aux_loc_combined_share_loc_noise=True.")
        if model.config.aux_loc_combined_unc_eps <= 0.0:
            raise ValueError("aux_loc_combined_unc_eps must be > 0 when is_aux_loc_combined_em_unc_loss=True.")
        if model.config.use_external_loc_model:
            raise ValueError("is_aux_loc_combined_em_unc_loss=True currently supports only the internal action-DiT loc branch.")
        if (
            model.config.use_vit_regression_head
            or model.config.use_vit_cls_regression_head
            or model.config.use_codex_vit_regression_head
        ):
            raise ValueError("is_aux_loc_combined_em_unc_loss=True currently supports only the action-DiT loc branch.")

    if model.config.is_gen_aux_loss:
        if not model.config.is_loc_aux_loss:
            raise ValueError("is_gen_aux_loss=True expects is_loc_aux_loss=True for the paired aux-loss experiment.")
        if model.config.aux_gen_pose_condition_type != "pose_token_mlp":
            raise ValueError("Current implementation only supports aux_gen_pose_condition_type='pose_token_mlp'.")
        if model.config.aux_gen_pose_condition_mode not in {"delta_from_gt", "absolute"}:
            raise ValueError("aux_gen_pose_condition_mode must be one of {'delta_from_gt', 'absolute'}.")
        if model.config.aux_gen_update_scope != "loc_head_only":
            raise ValueError("Current implementation only supports aux_gen_update_scope='loc_head_only'.")
        if not model.config.aux_gen_use_loc_samples:
            raise ValueError("Current implementation requires aux_gen_use_loc_samples=True.")
        if model.config.use_external_loc_model:
            raise ValueError("is_gen_aux_loss=True currently supports only the internal action-DiT loc branch.")
        if (
            model.config.use_vit_regression_head
            or model.config.use_vit_cls_regression_head
            or model.config.use_codex_vit_regression_head
        ):
            raise ValueError("is_gen_aux_loss=True currently supports only the action-DiT loc branch.")
        if model.config.aux_gen_pose_injection_scale <= 0.0:
            raise ValueError("aux_gen_pose_injection_scale must be > 0.")
        if model.config.is_aux_loss_alternating:
            if model.config.aux_loss_alternate_cycle_steps <= 0:
                raise ValueError("aux_loss_alternate_cycle_steps must be > 0.")
            if not (0 <= model.config.aux_loc_active_phase < model.config.aux_loss_alternate_cycle_steps):
                raise ValueError("aux_loc_active_phase must be in [0, aux_loss_alternate_cycle_steps).")
            if not (0 <= model.config.aux_gen_active_phase < model.config.aux_loss_alternate_cycle_steps):
                raise ValueError("aux_gen_active_phase must be in [0, aux_loss_alternate_cycle_steps).")
            if model.config.aux_loss_alternate_start_step < 0:
                raise ValueError("aux_loss_alternate_start_step must be >= 0.")

    if model.config.is_repa_loss:
        if model.config.repa_teacher_type not in {"dinov2", "unilip_vision"}:
            raise ValueError("Current implementation only supports repa_teacher_type in {'dinov2', 'unilip_vision'}.")
        if model.config.repa_align_type != "patch_wise":
            raise ValueError("Current implementation only supports repa_align_type='patch_wise'.")
        if model.config.repa_projector_type not in {"mlp3_silu", "conv_spatialnorm"}:
            raise ValueError("Current implementation only supports repa_projector_type in {'mlp3_silu', 'conv_spatialnorm'}.")
        if model.config.repa_projector_type == "mlp3_silu":
            if model.config.repa_mlp_activation != "silu":
                raise ValueError("Current implementation only supports repa_mlp_activation='silu'.")
        if model.config.repa_projector_type == "conv_spatialnorm":
            if model.config.repa_conv_kernel_size != 3:
                raise ValueError("Current implementation only supports repa_conv_kernel_size=3 for conv_spatialnorm.")

    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    # fix connect False
    # fix dit False
    # unilip load from checkpoint!!! unilip = vision_tower + multi_modal_projector
    # DiT load from checkpoint!!!
    # Connector load from checkpoint!!! = llm_connector + projector
    # latent_queries load from checkpoint!!!

    # Unified_UniLIP_InternVLForCausalLM.from_pretrained()启用了low_mem和accelerator且UniLIP权重中不包含新增的定位模块action_dit，导致action_dit没有正确加载Qwen2Model的权重，这里避开Unified_UniLIP_InternVLForCausalLM的__init__和from_pretrained方法。重新加载action_dit的权重，防止梯度爆炸
    # model.get_model().initialize_localization_modules(model_args=model_args)
    skip_internal_loc_modules = csgo_config.get(
        "skip_internal_loc_modules",
        bool(model.config.use_external_loc_model),
    )
    if not skip_internal_loc_modules:
        model.get_model().initialize_localization_modules(model_args=model_args)
    else:
        logging.info("Skip UniLIP internal localization module initialization.")

    if getattr(model, "initialize_repa_modules_from_config", None) is not None:
        model.initialize_repa_modules_from_config()
    if getattr(model, "initialize_aux_gen_modules_from_config", None) is not None:
        model.initialize_aux_gen_modules_from_config()

    model.to(torch.bfloat16)

    if model.config.use_external_loc_model:
        csgosquare_root = csgo_config.get("external_loc_repo_root", "csgosquare")
        external_loc_cfg = csgo_config.get(
            "external_loc_config_path",
            "configs_reg_newdata/exp5_2.yaml",
        )
        external_loc_ckpt = csgo_config.get(
            "external_loc_checkpoint_path",
            "checkpoints_reg_newdata/exp5_2/20251227_091745/current_model.pth",
        )
        external_loc_model = build_frozen_external_loc_model(
            csgosquare_root=csgosquare_root,
            config_path=external_loc_cfg,
            checkpoint_path=external_loc_ckpt,
            device=torch.device(training_args.device),
        )
        if not hasattr(model, "set_external_localization_model"):
            raise RuntimeError("Current model does not support external localization model attachment.")
        model.set_external_localization_model(external_loc_model)
        logging.info(
            "External localization model attached (frozen eval): root=%s, cfg=%s, ckpt=%s",
            csgosquare_root,
            external_loc_cfg,
            external_loc_ckpt,
        )

    data_args.image_processor = AutoProcessor.from_pretrained(model_args.mllm_hf_path).image_processor

    data_args.is_multimodal = True
    data_args.n_query = model_args.n_query
    data_args.n_und_query = model_args.n_und_query

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter

    # #### 实际这里不需要设置gc，因为training_args中的gc开启传入trainer初始化时会自动开启所有module的gc
    if training_args.gradient_checkpointing:
    # if 1: ### 单独关闭action_dit的gc，其他module开启以避免pi05 gemma300m的两次backward bug
        # Component-specific gradient checkpointing
        base_model = model.get_model()
        # 1. Vision Tower - aux_loc_loss需要梯度经过vlm，所以无论是否需要更新这部分权重，都要开启gc
        if hasattr(base_model, 'vision_tower'):# and not model_args.fix_vit:
            if hasattr(base_model.vision_tower, 'gradient_checkpointing_enable'):
                base_model.vision_tower.gradient_checkpointing_enable()
            logging.info("✅ Enabled gradient checkpointing for vision tower")
        # 2. Language Model
        if hasattr(base_model, 'language_model'):# and not model_args.fix_llm:
            if hasattr(base_model.language_model, 'gradient_checkpointing_enable'):
                base_model.language_model.gradient_checkpointing_enable()
            logging.info("✅ Enabled gradient checkpointing for language model")
        # 3. LLM Connector models - llm slices
        if hasattr(base_model, 'llm_connector'):
            base_model.llm_connector.gradient_checkpointing_enable()
            logging.info("✅ Enabled gradient checkpointing for LLM Connector")
        # 4. DiT models - SANA DiT
        if hasattr(base_model, 'dit'):
            base_model.dit.enable_gradient_checkpointing()
            logging.info("✅ Enabled gradient checkpointing for Gen DiT")
        # 5. Action DiT - llm slices or Pi0.5 Action DiT
        if hasattr(base_model, 'action_dit') and hasattr(base_model.action_dit, 'gradient_checkpointing_enable'):
            base_model.action_dit.gradient_checkpointing_enable()
            logging.info("✅ Enabled gradient checkpointing for Loc DiT")

    if getattr(training_args, 'is_lora', False):
        model.inject_lora_to_sub_module(model_args, training_args)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"🚀 After LoRA: Total parameters: {total_params}")
        logging.info(f"🚀 After LoRA: Trainable parameters: {trainable_params}")
        logging.info(f"🚀 After LoRA: Trainable percent: {100*trainable_params / total_params:.2f} %")

    else:
        # Calculate total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.get_model().parameters())
        trainable_params = sum(p.numel() for p in model.get_model().parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params}")
        logging.info(f"Trainable parameters: {trainable_params}")
        logging.info(f"trainable percent: {100*trainable_params / total_params:2f} %")


    train_param_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            train_param_names.append(name)
            # logging.info(f"     trainable params: {name}")
    logging.info(f"trainable layers: {len(train_param_names)}")


    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id

    base_init_ckpt_path = csgo_config.get("base_init_ckpt_path", None)
    gen_init_ckpt_path = csgo_config.get("gen_init_ckpt_path", None)
    loc_init_ckpt_path = csgo_config.get("loc_init_ckpt_path", None)
    resume_ckpt_path = csgo_config.get("resume_ckpt_path", None)
    has_selective_init = any([base_init_ckpt_path, gen_init_ckpt_path, loc_init_ckpt_path])

    if training_args.pretrain_path != 'none' and has_selective_init:
        raise ValueError("pretrain_path is mutually exclusive with base/gen/loc init_ckpt_path.")
    if resume_ckpt_path is not None and has_selective_init:
        raise ValueError("resume_ckpt_path is mutually exclusive with base/gen/loc init_ckpt_path.")

    if training_args.pretrain_path != 'none':
        pretrain_path = training_args.pretrain_path
        state_dict = load_checkpoint_state_dict(pretrain_path)
        state_dict = smart_matching_state_dict_keys(state_dict, model)
        msg = model.load_state_dict(state_dict, strict=False)
        logging.info(f"load pretrain: {pretrain_path}")
        logging.info(msg)

    if has_selective_init:
        if base_init_ckpt_path is not None:
            base_state_dict = load_checkpoint_state_dict(base_init_ckpt_path)
            base_state_dict = smart_matching_state_dict_keys(base_state_dict, model)
            msg = model.load_state_dict(base_state_dict, strict=False)
            logging.info(f"Loaded base init checkpoint: {base_init_ckpt_path}")
            logging.info(msg)

        if gen_init_ckpt_path is not None:
            gen_prefixes = build_generation_init_prefixes(model.config)
            load_partial_checkpoint(model, gen_init_ckpt_path, gen_prefixes, "generation ckpt init")

        if loc_init_ckpt_path is not None:
            if getattr(model.config, "use_external_loc_model", False):
                raise ValueError("loc_init_ckpt_path is not supported when use_external_loc_model=True.")
            if skip_internal_loc_modules:
                raise ValueError("loc_init_ckpt_path requires skip_internal_loc_modules=False so internal localization modules exist.")
            loc_prefixes = build_localization_init_prefixes(model.config)
            load_partial_checkpoint(model, loc_init_ckpt_path, loc_prefixes, "localization ckpt init")

        training_args.output_dir = append_init_tags_to_output_dir(
            training_args.output_dir,
            base_init_ckpt_path=base_init_ckpt_path,
            gen_init_ckpt_path=gen_init_ckpt_path,
            loc_init_ckpt_path=loc_init_ckpt_path,
        )
        logging.info("Updated output_dir for selective init experiment: %s", training_args.output_dir)

    if getattr(model.config, "is_loc_repa_loss", False):
        loc_repa_teacher_ckpt_path = getattr(model.config, "loc_repa_teacher_ckpt_path", None)
        if not loc_repa_teacher_ckpt_path:
            raise ValueError("is_loc_repa_loss=True requires loc_repa_teacher_ckpt_path.")
        if not hasattr(model, "initialize_loc_repa_teacher_from_state_dict"):
            raise RuntimeError("Current model does not support loc_repa teacher attachment.")

        teacher_state_dict = load_checkpoint_state_dict(loc_repa_teacher_ckpt_path)
        teacher_state_dict = smart_matching_state_dict_keys(teacher_state_dict, model)
        model.initialize_loc_repa_teacher_from_state_dict(teacher_state_dict)
        logging.info(
            "Loc-REPA teacher attached (frozen eval): ckpt=%s, feature_type=%s, loss_type=%s, train_mm_projector_only=%s, train_shared_llm_tail_only=%s",
            loc_repa_teacher_ckpt_path,
            getattr(model.config, "loc_repa_feature_type", "action_prefix_tokens"),
            getattr(model.config, "loc_repa_loss_type", "cosine"),
            getattr(model_args, "train_mm_projector_only", False),
            getattr(model_args, "train_shared_llm_tail_only", False),
        )

    if getattr(model.config, "is_repa_loss", False):
        if not hasattr(model, "initialize_repa_teacher"):
            raise RuntimeError("Current model does not support REPA teacher attachment.")
        model.initialize_repa_teacher()


    # def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:

    #     if data_args.data_type == "mix":
    #         train_dataset = LazySupervisedMixDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    #     else:
    #         raise ValueError("Unknown data type. Please check the Dataloader type.")

    #     data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    if csgo_config.get("is_multi_task", False):
        if csgo_config.get("is_multi_task_balanced", False):
            train_dataset = UniLIPMultiTaskBalancedDataset(csgo_config, tokenizer, data_args)
        else:
            train_dataset = UniLIPMultiTaskDataset(csgo_config, tokenizer, data_args)
        eval_dataset = None
        data_collator = DataCollatorForUniLIPMultiTaskDataset(tokenizer=tokenizer)
    else:
        train_dataset = CSGOWorldModelDataset(csgo_config, tokenizer, data_args)
        eval_dataset = None
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    if local_rank in [-1, 0]:
        if csgo_config.get("is_multi_task_balanced", False):
            visualize_dataset_samples_paired(
                train_dataset,
                data_args.image_processor,
                num_samples=20,
                save_path="_debug_dataset_samples.jpg",
            )
        else:
            visualize_dataset_samples_v1(
                train_dataset,
                data_args.image_processor,
                num_samples=20,
                save_path="_debug_dataset_samples.jpg",
                is_multi_task=csgo_config.get("is_multi_task", False)
            )



    # 自定义resume训练加载ckpt，支持resume的单卡多卡切换
    if resume_ckpt_path is not None:
        print(f"📥 Loading Checkpoint: {resume_ckpt_path}")
        state_dict = load_checkpoint_state_dict(resume_ckpt_path)
        state_dict = smart_matching_state_dict_keys(state_dict, model)
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

        training_args.output_dir = f"{training_args.output_dir}_resume_from_{resume_ckpt_path.split('/')[-2]}"



    unilip_log_callback = UniLIPLogCallback()
    callbacks = [unilip_log_callback]

    alpha_loc_schedule_steps = csgo_config.get("alpha_loc_schedule_steps", None)
    alpha_loc_schedule_values = csgo_config.get("alpha_loc_schedule_values", None)
    if alpha_loc_schedule_steps is not None or alpha_loc_schedule_values is not None:
        if alpha_loc_schedule_steps is None or alpha_loc_schedule_values is None:
            raise ValueError("alpha_loc schedule requires both alpha_loc_schedule_steps and alpha_loc_schedule_values")
        callbacks.append(
            AlphaLocScheduleCallback(
                steps=alpha_loc_schedule_steps,
                values=alpha_loc_schedule_values,
            )
        )

    alpha_loc_aux_schedule_steps = csgo_config.get("alpha_loc_aux_schedule_steps", None)
    alpha_loc_aux_schedule_values = csgo_config.get("alpha_loc_aux_schedule_values", None)
    alpha_gen_aux_schedule_steps = csgo_config.get("alpha_gen_aux_schedule_steps", None)
    alpha_gen_aux_schedule_values = csgo_config.get("alpha_gen_aux_schedule_values", None)
    is_loc_aux_step_gate = bool(csgo_config.get("is_loc_aux_step_gate", False))
    is_aux_loss_alternating = bool(csgo_config.get("is_aux_loss_alternating", False))
    if alpha_loc_aux_schedule_steps is not None or alpha_loc_aux_schedule_values is not None:
        if alpha_loc_aux_schedule_steps is None or alpha_loc_aux_schedule_values is None:
            raise ValueError("alpha_loc_aux schedule requires both alpha_loc_aux_schedule_steps and alpha_loc_aux_schedule_values")
    if alpha_gen_aux_schedule_steps is not None or alpha_gen_aux_schedule_values is not None:
        if alpha_gen_aux_schedule_steps is None or alpha_gen_aux_schedule_values is None:
            raise ValueError("alpha_gen_aux schedule requires both alpha_gen_aux_schedule_steps and alpha_gen_aux_schedule_values")
    if is_aux_loss_alternating:
        callbacks.append(
            AlternatingAuxLossControlCallback(
                loc_steps=alpha_loc_aux_schedule_steps,
                loc_values=alpha_loc_aux_schedule_values,
                gen_steps=alpha_gen_aux_schedule_steps,
                gen_values=alpha_gen_aux_schedule_values,
                base_loc_alpha=float(csgo_config.get("alpha_loc_aux_loss", 1.0)),
                base_gen_alpha=float(csgo_config.get("alpha_gen_aux_loss", 0.0)),
                use_alternating=True,
                cycle_steps=int(csgo_config.get("aux_loss_alternate_cycle_steps", 2)),
                loc_active_phase=int(csgo_config.get("aux_loc_active_phase", 0)),
                gen_active_phase=int(csgo_config.get("aux_gen_active_phase", 1)),
                start_step=int(csgo_config.get("aux_loss_alternate_start_step", 0)),
            )
        )
    elif alpha_loc_aux_schedule_steps is not None or alpha_loc_aux_schedule_values is not None or is_loc_aux_step_gate:
        callbacks.append(
            AlphaLocAuxControlCallback(
                steps=alpha_loc_aux_schedule_steps,
                values=alpha_loc_aux_schedule_values,
                use_step_gate=is_loc_aux_step_gate,
                gate_cycle_steps=int(csgo_config.get("loc_aux_gate_cycle_steps", 0)),
                gate_on_steps=int(csgo_config.get("loc_aux_gate_on_steps", 0)),
                gate_start_step=int(csgo_config.get("loc_aux_gate_start_step", 0)),
                base_alpha=float(csgo_config.get("alpha_loc_aux_loss", 1.0)),
            )
        )

    trainer = NonMixTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        # **data_module,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    unilip_log_callback.bind_trainer(trainer)
    logging.info("NonMixTrainer.Callbacks: %s", trainer.callback_handler.callback_list)
    # trainer.remove_callback(PrinterCallback)
    # trainer.remove_callback(ProgressCallback)



    from tabulate import tabulate
    if trainer.is_world_process_zero():
        # stat = []
        # for i, (n, p) in enumerate(trainer.model.named_parameters()):
        #     stat.append([i, n, p.shape, p.requires_grad])
        # logging.info(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))

        trainer.create_optimizer() # 步数随便填个大概就行，提供给 optimizer 初始化用，后续 trainer.train() 会根据实际步数重新创建 scheduler 和 optimizer 的 state，不受这里的 num_training_steps 影响
        # 1. 建立 [参数内存地址 -> 学习率] 的映射字典
        param_id_to_lr = {}

        # 注意：确保此时 trainer.optimizer 已经被初始化！
        if trainer.optimizer is not None:
            for group in trainer.optimizer.param_groups:
                group_lr = group.get('initial_lr', group.get('lr', 'Unknown'))
                for p in group['params']:
                    param_id_to_lr[id(p)] = group_lr
        else:
            logging.warning("Trainer optimizer is not initialized yet. LR will show as 'N/A'.")

        # 2. 收集统计信息
        stat = []
        for i, (n, p) in enumerate(trainer.model.named_parameters()):
            # 通过 id(p) 查找对应的学习率，如果找不到（比如被冻结没放进优化器），设为 N/A 或 0.0
            lr = param_id_to_lr.get(id(p), "N/A" if not p.requires_grad else "Missing")
            stat.append([i, n, p.shape, p.requires_grad, lr])

        # 3. 打印表格
        logging.info("\n" + tabulate(stat, headers=["idx", "name", "shape", "trainable", "lr"]))



    # 把回调加入 Trainer
    # trainer.add_callback(ProfilerCallback(profile_dir="./profiler_logs"))


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
        is_lora=getattr(training_args, 'is_lora', False),
    )


    if training_args.local_rank in [-1, 0]:
        wandb.finish()



if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
    # train(attn_implementation="sdpa")





